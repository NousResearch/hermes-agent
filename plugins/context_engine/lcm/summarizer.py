"""Summarization logic for LCM context compaction.

Extracted from ContextCompressor for reuse and testability.
Uses structured templates (Goal, Progress, Decisions, Files, Next Steps)
inspired by Pi-mono and OpenCode.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.auxiliary_client import call_llm

logger = logging.getLogger(__name__)

# Token budget limits for summaries
MIN_SUMMARY_TOKENS = 512
MAX_SUMMARY_TOKENS = 4000
SUMMARY_RATIO = 0.20  # Proportion of compressed content for summary

# Placeholder for pruned tool outputs
PRUNED_TOOL_PLACEHOLDER = "[Old tool output cleared to save context space]"

# Prefix for summary messages in conversation
SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION] Earlier turns in this conversation were compacted "
    "to save context space. The summary below describes work that was "
    "already completed, and the current session state may still reflect "
    "that work (for example, files may already be changed). Use the summary "
    "and the current state to continue from where things left off, and "
    "avoid repeating work:"
)


@dataclass
class SummarizerConfig:
    """Configuration for the Summarizer."""
    model: str = ""
    min_tokens: int = MIN_SUMMARY_TOKENS
    max_tokens: int = MAX_SUMMARY_TOKENS
    ratio: float = SUMMARY_RATIO
    timeout: float = 45.0
    temperature: float = 0.3


class Summarizer:
    """Generate structured summaries of conversation turns for context compaction.

    Supports iterative updates: when a previous summary exists, the new
    summary incorporates both the old content and new turns, preserving
    important context across multiple compactions.
    """

    def __init__(
        self,
        config: SummarizerConfig,
        token_estimator: Optional["TokenEstimator"] = None,
    ):
        self.config = config
        self.token_estimator = token_estimator
        self._previous_summary: Optional[str] = None

    def compute_summary_budget(self, turns: List[Dict[str, Any]]) -> int:
        """Scale summary token budget with the amount of content being compressed."""
        if self.token_estimator:
            content_tokens = self.token_estimator.estimate(turns)
        else:
            # Fallback: rough char-based estimate
            total_chars = sum(len(str(m.get("content", "") or "")) for m in turns)
            content_tokens = total_chars // 4

        budget = int(content_tokens * self.config.ratio)
        return max(self.config.min_tokens, min(budget, self.config.max_tokens))

    def serialize_for_summary(self, turns: List[Dict[str, Any]], max_chars: int = 3000) -> str:
        """Serialize conversation turns into labeled text for the summarizer.

        Includes tool call arguments and result content (up to max_chars
        per message) so the summarizer can preserve specific details like
        file paths, commands, and outputs.
        """
        parts = []
        for msg in turns:
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""

            # Tool results: keep more content
            if role == "tool":
                tool_id = msg.get("tool_call_id", "")
                if len(content) > max_chars:
                    content = self._truncate_middle(content, max_chars)
                parts.append(f"[TOOL RESULT {tool_id}]: {content}")
                continue

            # Assistant messages: include tool call names AND arguments
            if role == "assistant":
                if len(content) > max_chars:
                    content = self._truncate_middle(content, max_chars)
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tc_parts = []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            name = fn.get("name", "?")
                            args = fn.get("arguments", "")
                            # Truncate long arguments but keep enough for context
                            if len(args) > 500:
                                args = args[:400] + "..."
                            tc_parts.append(f"  {name}({args})")
                        else:
                            # Handle object-style tool calls
                            fn = getattr(tc, "function", None)
                            name = getattr(fn, "name", "?") if fn else "?"
                            tc_parts.append(f"  {name}(...)")
                    content += "\n[Tool calls:\n" + "\n".join(tc_parts) + "\n]"
                parts.append(f"[ASSISTANT]: {content}")
                continue

            # User and other roles
            if len(content) > max_chars:
                content = self._truncate_middle(content, max_chars)
            parts.append(f"[{role.upper()}]: {content}")

        return "\n\n".join(parts)

    def summarize(
        self,
        turns: List[Dict[str, Any]],
        previous_summary: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a structured summary of conversation turns.

        Uses a structured template (Goal, Progress, Decisions, Files, Next Steps).
        When a previous summary exists, generates an iterative update instead
        of summarizing from scratch.

        Returns None if all attempts fail - the caller should drop
        the middle turns without a summary rather than inject a useless
        placeholder.
        """
        if not turns:
            return None

        summary_budget = self.compute_summary_budget(turns)
        content_to_summarize = self.serialize_for_summary(turns)

        # Use provided previous_summary or stored one
        prev = previous_summary or self._previous_summary

        if prev:
            prompt = self._build_iterative_prompt(prev, content_to_summarize, summary_budget)
        else:
            prompt = self._build_initial_prompt(content_to_summarize, summary_budget)

        # ── Layer 1: call_llm with full provider fallback chain ────────
        # call_llm already tries OpenRouter → Nous → Custom → Codex → Anthropic
        # etc. plus payment exhaustion fallback. If ANY provider is available,
        # it will be used.
        summary = self._try_llm_summarize(prompt, summary_budget)
        if summary is not None:
            return self._with_summary_prefix(summary)

        # ── Layer 2: Deterministic extractive fallback ─────────────────
        # No LLM available at all. Build a structured summary by extracting
        # key information from the turns directly. Not as good as LLM but
        # preserves critical context (file paths, commands, decisions).
        logger.info("LCM: Using deterministic extractive fallback for summary")
        summary = self._deterministic_summary(turns, prev)
        if summary is not None:
            self._previous_summary = summary
            return self._with_summary_prefix(summary)

        return None

    def _try_llm_summarize(self, prompt: str, summary_budget: int) -> Optional[str]:
        """Try to summarize using call_llm with the full provider chain."""
        try:
            call_kwargs = {
                "task": "compression",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
                "max_tokens": summary_budget * 2,
                "timeout": self.config.timeout,
            }
            if self.config.model:
                call_kwargs["model"] = self.config.model

            response = call_llm(**call_kwargs)
            content = response.choices[0].message.content

            # Handle cases where content is not a string (e.g., dict from llama.cpp)
            if not isinstance(content, str):
                content = str(content) if content else ""

            summary = content.strip()
            self._previous_summary = summary
            return summary

        except RuntimeError:
            logger.warning(
                "Context compression: no provider available for LLM summary. "
                "Falling back to deterministic extractive summary."
            )
            return None
        except Exception as e:
            logger.warning("Failed to generate LLM context summary: %s. "
                         "Falling back to deterministic extractive summary.", e)
            return None

    def _deterministic_summary(
        self,
        turns: List[Dict[str, Any]],
        previous_summary: Optional[str] = None,
    ) -> Optional[str]:
        """Build a structured summary without any LLM call.

        Uses sumy's LexRank algorithm (graph-based sentence centrality)
        for extractive summarization. Falls back to simple extraction
        if sumy is unavailable.

        LexRank ranks sentences by their centrality in a similarity graph —
        sentences most representative of the whole content get selected.
        This is significantly better than naive concatenation because it:
        - Scores and ranks sentences by importance
        - Selects a budget-fitting subset
        - Preserves the most informative content
        """
        if not turns:
            return None

        # Serialize turns to text for sumy
        all_text = self.serialize_for_summary(turns, max_chars=1500)

        # Try sumy extractive summarization first
        summary = self._sumy_extract(all_text, sentence_count=6)
        if summary is None:
            # sumy not available — fall back to simple extraction
            summary = self._simple_extract(turns)

        if summary is None:
            return None

        parts: List[str] = []
        if previous_summary:
            parts.append(previous_summary)
            parts.append("--- Updates since last compaction ---")

        parts.append(summary)
        result = "\n".join(parts)
        return result if result.strip() else None

    def _sumy_extract(self, text: str, sentence_count: int = 6) -> Optional[str]:
        """Extract key sentences using sumy's LexRank algorithm.

        Returns None if sumy is not available or produces no output.
        """
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.lex_rank import LexRankSummarizer
            from sumy.nlp.stemmers import Stemmer
            from sumy.utils import get_stop_words

            if not text or len(text.strip()) < 50:
                return None

            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            stemmer = Stemmer("english")
            summarizer = LexRankSummarizer(stemmer)
            summarizer.stop_words = get_stop_words("english")

            sentences = list(summarizer(parser.document, sentence_count))
            if not sentences:
                return None

            return " ".join(str(s).strip() for s in sentences)

        except ImportError:
            logger.debug("sumy not available for extractive summarization, using simple fallback")
            return None
        except Exception as exc:
            logger.debug("sumy extractive summary failed: %s", exc)
            return None

    def _simple_extract(self, turns: List[Dict[str, Any]]) -> Optional[str]:
        """Simple keyword-based extraction fallback when sumy is unavailable.

        Extracts tool calls, user intents, and key results by role-based
        partitioning. Less intelligent than sumy but zero-dependency.
        """
        parts: List[str] = []

        user_msgs = []
        tool_calls_seen = []
        tool_results = []
        assistant_msgs = []

        for msg in turns:
            role = msg.get("role", "")
            content = str(msg.get("content", "") or "")[:500]

            if role == "user":
                user_msgs.append(content)
            elif role == "assistant":
                assistant_msgs.append(content)
                for tc in msg.get("tool_calls", []):
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        name = fn.get("name", "?")
                        args = str(fn.get("arguments", ""))[:200]
                        tool_calls_seen.append(f"{name}({args})")
            elif role == "tool":
                tool_results.append(content[:300])

        if user_msgs:
            parts.append("## User Messages")
            for i, m in enumerate(user_msgs[-5:], 1):
                parts.append(f"  {i}. {m[:200]}")

        if tool_calls_seen:
            parts.append("## Tool Calls Made")
            for tc in tool_calls_seen[-10:]:
                parts.append(f"  - {tc}")

        if tool_results:
            parts.append("## Key Tool Results")
            for r in tool_results[-5:]:
                parts.append(f"  - {r[:200]}")

        if assistant_msgs:
            parts.append("## Assistant Responses")
            for m in assistant_msgs[-3:]:
                if m.strip():
                    parts.append(f"  - {m[:200]}")

        result = "\n".join(parts)
        return result if result.strip() else None

    def reset(self):
        """Clear stored previous summary for a new session."""
        self._previous_summary = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _truncate_middle(self, text: str, max_len: int) -> str:
        """Truncate middle of text, keeping head and tail."""
        if len(text) <= max_len:
            return text
        head_len = int(max_len * 0.67)  # ~2000 chars for 3000 max
        tail_len = max_len - head_len
        return text[:head_len] + "\n...[truncated]...\n" + text[-tail_len:]

    def _build_initial_prompt(self, content: str, budget: int) -> str:
        """Build prompt for first-time summarization."""
        return f"""Create a structured handoff summary for a later assistant that will continue this conversation after earlier turns are compacted.

TURNS TO SUMMARIZE:
{content}

Use this exact structure:

## Goal
[What the user is trying to accomplish]

## Constraints & Preferences
[User preferences, coding style, constraints, important decisions]

## Progress
### Done
[Completed work - include specific file paths, commands run, results obtained]
### In Progress
[Work currently underway]
### Blocked
[Any blockers or issues encountered]

## Key Decisions
[Important technical decisions and why they were made]

## Relevant Files
[Files read, modified, or created - with brief note on each]

## Next Steps
[What needs to happen next to continue the work]

## Critical Context
[Any specific values, error messages, configuration details, or data that would be lost without explicit preservation]

Target ~{budget} tokens. Be specific - include file paths, command outputs, error messages, and concrete values rather than vague descriptions. The goal is to prevent the next assistant from repeating work or losing important details.

Write only the summary body. Do not include any preamble or prefix."""

    def _build_iterative_prompt(
        self, previous_summary: str, new_content: str, budget: int
    ) -> str:
        """Build prompt for iterative summary update."""
        return f"""You are updating a context compaction summary. A previous compaction produced the summary below. New conversation turns have occurred since then and need to be incorporated.

PREVIOUS SUMMARY:
{previous_summary}

NEW TURNS TO INCORPORATE:
{new_content}

Update the summary using this exact structure. PRESERVE all existing information that is still relevant. ADD new progress. Move items from "In Progress" to "Done" when completed. Remove information only if it is clearly obsolete.

## Goal
[What the user is trying to accomplish - preserve from previous summary, update if goal evolved]

## Constraints & Preferences
[User preferences, coding style, constraints, important decisions - accumulate across compactions]

## Progress
### Done
[Completed work - include specific file paths, commands run, results obtained]
### In Progress
[Work currently underway]
### Blocked
[Any blockers or issues encountered]

## Key Decisions
[Important technical decisions and why they were made]

## Relevant Files
[Files read, modified, or created - with brief note on each. Accumulate across compactions.]

## Next Steps
[What needs to happen next to continue the work]

## Critical Context
[Any specific values, error messages, configuration details, or data that would be lost without explicit preservation]

Target ~{budget} tokens. Be specific - include file paths, command outputs, error messages, and concrete values rather than vague descriptions.

Write only the summary body. Do not include any preamble or prefix."""

    @staticmethod
    def _with_summary_prefix(summary: str) -> str:
        """Normalize summary text to the current compaction handoff format."""
        text = (summary or "").strip()
        # Remove any existing prefix
        for prefix in (SUMMARY_PREFIX, "[CONTEXT SUMMARY]:"):
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip()
                break
        return f"{SUMMARY_PREFIX}\n{text}" if text else SUMMARY_PREFIX
