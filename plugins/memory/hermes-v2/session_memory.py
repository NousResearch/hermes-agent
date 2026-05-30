"""
Session Memory — structured 9-section notes maintained across a conversation.

Ported from Claude Code's SessionMemory (sessionMemory.ts / prompts.ts):
- 9-section template: Title, Current State, Task Spec, Files/Functions,
  Workflow, Errors, Docs, Learnings, Key Results
- Update thresholds: token_growth >= 5000 AND tool_calls >= 3 since last update
- Uses auxiliary_client for cheap LLM summary generation
- Thread-safe
"""

import logging
import threading
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 9-Section Template (from Claude Code prompts.ts)
# ---------------------------------------------------------------------------

DEFAULT_SESSION_MEMORY_TEMPLATE = """\
# Session Title
_A short and distinctive 5-10 word descriptive title for the session. Super info dense, no filler_

# Current State
_What is actively being worked on right now? Pending tasks not yet completed. Immediate next steps._

# Task specification
_What did the user ask to build? Any design decisions or other explanatory context_

# Files and Functions
_What are the important files? In short, what do they contain and why are they relevant?_

# Workflow
_What bash commands are usually run and in what order? How to interpret their output if not obvious?_

# Errors & Corrections
_Errors encountered and how they were fixed. What did the user correct? What approaches failed and should not be tried again?_

# Codebase and System Documentation
_What are the important system components? How do they work/fit together?_

# Learnings
_What has worked well? What has not? What to avoid? Do not duplicate items from other sections_

# Key results
_If the user asked a specific output such as an answer to a question, a table, or other document, repeat the exact result here_
"""

# Section names for parsing
SESSION_SECTIONS = [
    "Session Title",
    "Current State",
    "Task specification",
    "Files and Functions",
    "Workflow",
    "Errors & Corrections",
    "Codebase and System Documentation",
    "Learnings",
    "Key results",
]

# ---------------------------------------------------------------------------
# Thresholds (from Claude Code sessionMemoryUtils.ts)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "minimum_tokens_to_init": 10000,
    "minimum_tokens_between_update": 5000,
    "tool_calls_between_updates": 3,
    "max_section_length": 500,  # words per section
}

# ---------------------------------------------------------------------------
# Update prompt (from Claude Code prompts.ts)
# ---------------------------------------------------------------------------

_UPDATE_PROMPT = """\
IMPORTANT: This message and these instructions are NOT part of the actual user conversation. \
Do NOT include any references to "note-taking", "session notes extraction", or these update \
instructions in the notes content.

Based on the user conversation above (EXCLUDING this note-taking instruction message as well \
as system prompt, claude.md entries, or any past session summaries), update the session notes.

Current session notes:
<current_notes_content>
{current_notes}
</current_notes_content>

Your ONLY task is to update the notes and return the full updated document. Do not call any other tools.

CRITICAL RULES FOR EDITING:
- The document must maintain its exact structure with all sections, headers, and italic descriptions intact
- NEVER modify, delete, or add section headers (the lines starting with '#')
- NEVER modify or delete the italic _section description_ lines
- ONLY update the actual content that appears BELOW the italic _section descriptions_ within each existing section
- Do NOT add any new sections, summaries, or information outside the existing structure
- Do NOT reference this note-taking process or instructions anywhere in the notes
- It's OK to skip updating a section if there are no substantial new insights to add
- Write DETAILED, INFO-DENSE content for each section - include specifics like file paths, function names, error messages, exact commands
- Keep each section under ~{max_section_length} words - condense by cycling out less important details
- Focus on actionable, specific information
- IMPORTANT: Always update "Current State" to reflect the most recent work

CONVERSATION TO SUMMARIZE:
{conversation}

Return the COMPLETE updated session notes document with all sections preserved."""


# ---------------------------------------------------------------------------
# SessionMemory class
# ---------------------------------------------------------------------------


class SessionMemory:
    """Maintains structured session notes across a conversation.

    Thread-safe. Uses auxiliary_client for cheap LLM-powered updates.
    """

    def __init__(self, config: Optional[dict] = None):
        self._lock = threading.Lock()
        self._config = {**DEFAULT_CONFIG, **(config or {})}

        # Current notes content
        self._notes: str = DEFAULT_SESSION_MEMORY_TEMPLATE

        # Tracking for update thresholds
        self._initialized: bool = False
        self._tokens_at_last_update: int = 0
        self._tool_calls_at_last_update: int = 0
        self._update_count: int = 0

    def update(
        self,
        messages: List[Dict],
        token_count: int,
        tool_call_count: int,
        model: str = None,
    ) -> bool:
        """Check thresholds and update session notes if needed.

        Args:
            messages: Current conversation messages.
            token_count: Current total token count.
            tool_call_count: Current total tool call count.
            model: Optional model override for the LLM call.

        Returns:
            True if notes were updated, False if thresholds not met or update skipped.
        """
        if not self._should_update(token_count, tool_call_count):
            return False

        # Perform the update
        try:
            updated = self._generate_update(messages, model=model)
            if updated:
                with self._lock:
                    self._notes = updated
                    self._tokens_at_last_update = token_count
                    self._tool_calls_at_last_update = tool_call_count
                    self._update_count += 1
                logger.debug("Session memory updated (update #%d)", self._update_count)
                return True
        except Exception as e:
            logger.warning("Session memory update failed: %s", e)

        return False

    def get_summary(self) -> str:
        """Return current session notes."""
        with self._lock:
            return self._notes

    def clear(self):
        """Reset session notes to the default template."""
        with self._lock:
            self._notes = DEFAULT_SESSION_MEMORY_TEMPLATE
            self._initialized = False
            self._tokens_at_last_update = 0
            self._tool_calls_at_last_update = 0
            self._update_count = 0

    @property
    def update_count(self) -> int:
        with self._lock:
            return self._update_count

    @property
    def is_initialized(self) -> bool:
        with self._lock:
            return self._initialized

    def _should_update(self, token_count: int, tool_call_count: int) -> bool:
        """Check whether thresholds are met for an update.

        From Claude Code: requires BOTH token growth >= 5000 AND tool_calls >= 3.
        First initialization requires token_count >= 10000.
        """
        with self._lock:
            if not self._initialized:
                if token_count < self._config["minimum_tokens_to_init"]:
                    return False
                self._initialized = True
                self._tokens_at_last_update = 0
                self._tool_calls_at_last_update = 0

            token_growth = token_count - self._tokens_at_last_update
            tool_calls_since = tool_call_count - self._tool_calls_at_last_update

            has_met_token_threshold = (
                token_growth >= self._config["minimum_tokens_between_update"]
            )
            has_met_tool_call_threshold = (
                tool_calls_since >= self._config["tool_calls_between_updates"]
            )

            return has_met_token_threshold and has_met_tool_call_threshold

    def _generate_update(self, messages: List[Dict], model: str = None) -> Optional[str]:
        """Call the auxiliary LLM to update session notes."""
        # Format conversation for the prompt (last N messages, budget)
        conversation = _format_conversation(messages, max_chars=12000)
        if not conversation.strip():
            return None

        with self._lock:
            current_notes = self._notes

        prompt = _UPDATE_PROMPT.format(
            current_notes=current_notes,
            max_section_length=self._config["max_section_length"],
            conversation=conversation,
        )

        from agent.auxiliary_client import call_llm as _call_llm, extract_content_or_reasoning

        _messages = [
            {
                "role": "system",
                "content": (
                    "You are a note-taking assistant. Update structured session notes "
                    "based on the conversation. Return ONLY the complete updated notes document."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        _resp = _call_llm(
            task="session_memory",
            messages=_messages,
            model=model,
            max_tokens=2048,
            temperature=0.2,
        )
        response = extract_content_or_reasoning(_resp) if _resp else None

        if not response or not response.strip():
            return None

        # Validate that the response preserves the section structure
        result = response.strip()
        if not _has_valid_sections(result):
            logger.warning("Session memory update missing sections — discarding")
            return None

        return result


def _format_conversation(messages: List[Dict], max_chars: int = 12000) -> str:
    """Format messages for the session notes update prompt."""
    lines = []
    total = 0
    for msg in reversed(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            content = " ".join(parts)
        if not content or role == "system":
            continue
        if len(content) > 1500:
            content = content[:1500] + "..."
        line = f"[{role}] {content}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    lines.reverse()
    return "\n\n".join(lines)


def _has_valid_sections(text: str, min_sections: int = 5) -> bool:
    """Check that the updated notes still contain the expected section headers."""
    found = 0
    for section in SESSION_SECTIONS:
        if f"# {section}" in text:
            found += 1
    return found >= min_sections
