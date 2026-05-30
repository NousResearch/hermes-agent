"""
Automatic Memory Extraction — post-response hook that extracts durable memories.

Cannibalized from Claude Code (services/extractMemories/) and adapted for Hermes:
- Uses auxiliary_client (cheap model) instead of a forked full agent
- Pre-injects manifest of existing memories to prevent duplicates
- Processes only messages since last extraction (cursor tracking)
- Lightweight: structured JSON output, no tool calls needed
- Cursor tracking: only process NEW messages since last extraction
- Mutual exclusion: skip if main agent wrote memories this turn
- Trailing run stash: coalesce overlapping extraction requests

Triggered after every N assistant responses (configurable via extract_interval).
"""

import json
import logging
import threading
import time
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction prompt — adapted from Claude Code + HiveMind quality scoring
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are a memory extraction system. Review the recent conversation and extract \
any durable facts worth remembering long-term.

EXISTING MEMORIES (do NOT duplicate these):
{manifest}

RECENT CONVERSATION:
{recent_messages}

EXTRACT memories that are:
- User preferences, corrections, or personal details (type: "preference" or "correction")
- Environment facts: OS, tools, project structure, installed software (type: "project")
- Conventions, workflow patterns, or recurring instructions (type: "preference")
- Pointers to external systems, URLs, credentials locations (type: "reference")
- Corrections to previous agent behavior (type: "correction")

DO NOT extract:
- Task progress, session outcomes, or temporary state
- Facts easily re-derived from files or commands
- Anything already captured in existing memories above
- Raw data, code snippets, or verbose technical details
- Conversational artifacts, pleasantries, or reasoning steps
- Things that are only relevant to the current task and won't matter next session

IMPORTANCE SCORING (1-10):
- 1-3: Trivial, task-specific, easily re-derived. DO NOT SAVE.
- 4-5: Mildly useful but low durability. Only save if very concise.
- 6-7: Clearly durable fact that will matter in future sessions.
- 8-10: Critical preference, correction, or architectural decision.

For each memory worth saving (importance >= 5), output a JSON object on its own line:
{{"target": "memory"|"user", "type": "preference"|"correction"|"project"|"reference"|"general", "importance": 5-10, "content": "concise fact"}}

If nothing is worth saving, output exactly: NONE

Output ONLY the JSON lines or NONE, no other text.
Maximum 5 entries per extraction. Quality over quantity."""


# ---------------------------------------------------------------------------
# Extractor state — module-level closure state (ported from Claude Code)
# ---------------------------------------------------------------------------

class _ExtractorState:
    """Mutable state for the extraction subsystem.

    Ported from Claude Code's initExtractMemories() closure:
    - last_extracted_message_index: cursor so each run only considers new messages
    - _agent_wrote_memory: set by the main agent when it writes memories this turn
    - _in_progress: True while extraction is running
    - _pending_context: stashed context for trailing run when overlapping
    - _lock: mutual exclusion for state access
    - _engine: optional MemoryEngine reference for persisting cursor to SQLite
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.last_extracted_message_index: int = -1
        self._agent_wrote_memory: bool = False
        self._in_progress: bool = False
        self._pending_context: Optional[Dict] = None
        self._turns_since_last_extraction: int = 0
        self._engine = None  # Set when extraction runs; used for cursor persistence

    def persist_cursor(self, index: int):
        """Save cursor to SQLite so it survives process restarts."""
        with self._lock:
            self.last_extracted_message_index = index
        if self._engine:
            try:
                self._engine._set_meta("extractor_cursor", str(index))
            except Exception:
                pass  # Best effort

    def load_cursor(self, engine):
        """Load persisted cursor from SQLite on startup."""
        self._engine = engine
        try:
            val = engine._get_meta("extractor_cursor")
            if val is not None:
                self.last_extracted_message_index = int(val)
        except (ValueError, TypeError):
            pass

    def mark_agent_wrote_memory(self):
        """Called by the main agent when it writes memories this turn."""
        with self._lock:
            self._agent_wrote_memory = True

    def clear_agent_wrote_memory(self):
        """Reset the flag at the start of a new turn."""
        with self._lock:
            self._agent_wrote_memory = False

    @property
    def agent_wrote_memory(self) -> bool:
        with self._lock:
            return self._agent_wrote_memory

    @property
    def in_progress(self) -> bool:
        with self._lock:
            return self._in_progress


# Module-level singleton
_state = _ExtractorState()


def get_extractor_state() -> _ExtractorState:
    """Access the module-level extractor state (for testing or external use)."""
    return _state


def reset_extractor_state():
    """Reset extractor state (for testing)."""
    global _state
    _state = _ExtractorState()


def extract_memories_background(
    recent_messages: List[Dict],
    memory_store,
    auxiliary_client=None,
    model: str = None,
    session_id: str = None,
):
    """Run memory extraction in a background thread.

    Args:
        recent_messages: Last N messages from the conversation.
        memory_store: MemoryStore instance (with engine for SQLite mode).
        auxiliary_client: AuxiliaryClient for cheap LLM calls. If None, skipped.
        model: Model name override for the extraction call.
        session_id: Current session ID for attribution.
    """
    if auxiliary_client is None:
        logger.debug("No auxiliary_client — skipping memory extraction")
        return

    engine = getattr(memory_store, 'engine', None) or getattr(memory_store, '_engine', None)
    if engine is None:
        logger.debug("No MemoryEngine — skipping memory extraction (flat mode)")
        return

    state = _state

    # Load persisted cursor from SQLite on first use (survives restart)
    if state._engine is None:
        state.load_cursor(engine)

    # Mutual exclusion: if the main agent wrote memories this turn, skip
    # extraction and advance the cursor past these messages
    if state.agent_wrote_memory:
        state.persist_cursor(len(recent_messages) - 1)
        logger.debug(
            "[extractMemories] skipping — agent already wrote to memory this turn"
        )
        return

    # If extraction is already in progress, stash context for trailing run
    if state.in_progress:
        with state._lock:
            logger.debug(
                "[extractMemories] extraction in progress — stashing for trailing run"
            )
            state._pending_context = {
                "recent_messages": recent_messages,
                "memory_store": memory_store,
                "auxiliary_client": auxiliary_client,
                "model": model,
                "session_id": session_id,
                "engine": engine,
            }
        return

    def _extract():
        _run_extraction(
            recent_messages=recent_messages,
            engine=engine,
            auxiliary_client=auxiliary_client,
            model=model,
            session_id=session_id,
            is_trailing_run=False,
        )

    t = threading.Thread(target=_extract, daemon=True, name="memory-extractor")
    t.start()


def _run_extraction(
    recent_messages: List[Dict],
    engine,
    auxiliary_client,
    model: str = None,
    session_id: str = None,
    is_trailing_run: bool = False,
):
    """Perform extraction with cursor tracking, mutual exclusion, and trailing run support."""
    state = _state

    with state._lock:
        state._in_progress = True

    try:
        # Cursor tracking: only process messages since last extraction
        cursor = state.last_extracted_message_index
        if cursor >= 0 and cursor < len(recent_messages) - 1:
            new_messages = recent_messages[cursor + 1:]
        elif cursor >= len(recent_messages) - 1:
            # No new messages since last extraction
            logger.debug("[extractMemories] no new messages since cursor %d", cursor)
            return
        else:
            # cursor == -1 or invalid — process all
            new_messages = recent_messages

        if not new_messages:
            return

        # Perform the actual extraction
        _do_extraction(
            recent_messages=new_messages,
            engine=engine,
            auxiliary_client=auxiliary_client,
            model=model,
            session_id=session_id,
        )

        # Advance cursor on success (persisted to SQLite for restart survival)
        state.persist_cursor(len(recent_messages) - 1)

    except Exception as e:
        logger.warning("Memory extraction failed: %s", e)
    finally:
        # Check for trailing run
        trailing = None
        with state._lock:
            state._in_progress = False
            trailing = state._pending_context
            state._pending_context = None

        if trailing:
            logger.debug(
                "[extractMemories] running trailing extraction for stashed context"
            )
            _run_extraction(
                recent_messages=trailing["recent_messages"],
                engine=trailing["engine"],
                auxiliary_client=trailing["auxiliary_client"],
                model=trailing.get("model"),
                session_id=trailing.get("session_id"),
                is_trailing_run=True,
            )


def _do_extraction(
    recent_messages: List[Dict],
    engine,
    auxiliary_client,
    model: str = None,
    session_id: str = None,
):
    """Perform the actual extraction (runs in background thread)."""
    start = time.monotonic()

    # Build manifest of existing memories for dedup
    manifest = engine.get_manifest()

    # Format recent messages for the prompt
    formatted = _format_messages(recent_messages, max_chars=8000)
    if not formatted.strip():
        return

    prompt = _EXTRACTION_PROMPT.format(
        manifest=manifest,
        recent_messages=formatted,
    )

    # Call auxiliary LLM
    try:
        response = auxiliary_client.call_llm(
            prompt=prompt,
            system_message="You extract structured memories from conversations. Output JSON lines only.",
            model=model,
            max_tokens=1024,
            temperature=0.1,
        )
    except Exception as e:
        logger.debug("Extraction LLM call failed: %s", e)
        return

    if not response or not response.strip():
        return

    response = response.strip()
    if response.upper() == "NONE":
        logger.debug("Memory extraction: nothing to save")
        return

    # Parse JSON lines with importance filtering and per-extraction cap
    saved = 0
    skipped_low_importance = 0
    max_per_extraction = 5  # Hard cap — quality over quantity

    for line in response.split("\n"):
        if saved >= max_per_extraction:
            logger.debug("Extraction cap reached (%d), stopping", max_per_extraction)
            break

        line = line.strip()
        if not line or line.upper() == "NONE":
            continue

        # Strip markdown code fences if model wraps output
        if line.startswith("```"):
            continue

        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        target = entry.get("target", "memory")
        mem_type = entry.get("type", "general")
        content = entry.get("content", "").strip()
        importance = entry.get("importance", 5)

        if not content or target not in ("memory", "user"):
            continue

        # Importance gate — reject anything below threshold
        # Corrections and preferences get a +1 bonus (higher value per the research)
        effective_importance = importance
        if mem_type in ("correction", "preference"):
            effective_importance = min(10, importance + 1)

        if effective_importance < 5:
            skipped_low_importance += 1
            logger.debug(
                "Skipped low-importance extraction (score=%d): %s",
                importance, content[:60],
            )
            continue

        result = engine.add(
            content=content,
            target=target,
            type=mem_type,
            source="extraction",
            session_id=session_id,
        )
        if result.get("success"):
            saved += 1
            logger.debug(
                "Auto-extracted [%s/%s] importance=%d: %s",
                target, mem_type, importance, content[:60],
            )

    elapsed = time.monotonic() - start
    if saved or skipped_low_importance:
        logger.info(
            "Memory extraction: saved %d, skipped %d low-importance in %.1fs",
            saved, skipped_low_importance, elapsed,
        )


def _format_messages(messages: List[Dict], max_chars: int = 8000) -> str:
    """Format messages for the extraction prompt, truncating to budget."""
    lines = []
    total = 0
    # Process in reverse (most recent first) then reverse back
    for msg in reversed(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Handle multipart content (text blocks)
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            content = " ".join(parts)

        if not content or role == "system":
            continue

        # Truncate individual messages
        if len(content) > 1000:
            content = content[:1000] + "..."

        line = f"[{role}] {content}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)

    lines.reverse()
    return "\n\n".join(lines)
