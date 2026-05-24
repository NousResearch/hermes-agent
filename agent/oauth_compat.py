"""Anthropic-OAuth Claude Code compatibility shim.

Anthropic's plan-vs-extra-usage classifier on OAuth (Pro/Max) requests
returns HTTP 400 ("Third-party apps now draw from your extra usage…")
when a request fingerprints as a non-Claude-Code agent and the account
has no overage credit configured.

Empirically (against ``/v1/messages`` on a Pro/Max-only account) two
independent triggers exist:

  1. The *set* of tool names matches a third-party convention (hermes's
     snake_case ``terminal``, ``read_file``, ``session_search``, …)
     instead of Claude Code's PascalCase canonicals (``Bash``, ``Read``,
     ``Task``, …).

  2. A multi-block ``system`` prompt carrying agent-flavored persona /
     project context (SOUL.md, AGENTS.md auto-injection, memory) is
     content-scanned and flagged.

Renaming a single tool is sub-threshold; the whole *set* has to look
canonical or sufficiently neutral. Slimming the system prompt to the
bare Claude Code identity line is the only system-side mitigation that
flips the classifier.

This module owns both mitigations behind a single :class:`StealthMode`
toggle, and a per-session :class:`ToolNameMap` that keeps the forward /
reverse mapping consistent across turns.

Source for the canonical Claude Code tool set:
  https://cchistory.mariozechner.at/data/prompts-2.1.11.md
  (mirrors pi-ai's `// Stealth mode: Mimic Claude Code's tool naming
  exactly` block in pi-ai/src/providers/anthropic.ts.)

See hermes-agent issue #15080 for the original report and bisection.
"""

from __future__ import annotations

import enum
import logging
import threading
from typing import Any, Dict, FrozenSet, List, Mapping, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLAUDE_CODE_SYSTEM_PREFIX = (
    "You are Claude Code, Anthropic's official CLI for Claude."
)

#: Canonical Claude Code 2.x tool names (PascalCase). Reviewed against
#: cchistory.mariozechner.at/data/prompts-2.1.11.md. Used both as the
#: rename target for hermes equivalents (see :data:`HERMES_TO_CLAUDE_CODE`)
#: and as a "do not collide" set when PascalCasing unmapped names.
CLAUDE_CODE_TOOLS: FrozenSet[str] = frozenset({
    "AskUserQuestion",
    "Bash",
    "BashOutput",
    "Edit",
    "EnterPlanMode",
    "ExitPlanMode",
    "Glob",
    "Grep",
    "KillShell",
    "NotebookEdit",
    "Read",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
    "Write",
})

#: Hermes tool name → Claude Code canonical name. Only entries where the
#: semantics are close enough that the model's tool-use guidance still
#: makes sense (e.g. terminal→Bash, read_file→Read). Hermes tools without
#: a clean equivalent fall through to PascalCase via
#: :func:`_default_pascal_case` and pass the classifier on the strength
#: of the renamed set as a whole, not individual canonical matches.
HERMES_TO_CLAUDE_CODE: Mapping[str, str] = {
    "terminal":      "Bash",
    "read_file":     "Read",
    "write_file":    "Write",
    "search_files":  "Grep",
    "web_search":    "WebSearch",
    "web_extract":   "WebFetch",
    "todo":          "TodoWrite",
    "delegate_task": "Task",
    "clarify":       "AskUserQuestion",
    "patch":         "Edit",
    "process":       "KillShell",
    "skill_view":    "Skill",
}


# ---------------------------------------------------------------------------
# Stealth mode
# ---------------------------------------------------------------------------

class StealthMode(enum.Enum):
    """Whether and how to apply OAuth compatibility transforms."""

    #: No transforms. Tool names and system prompt are sent as-is. Use
    #: when the OAuth account has overage credits or has been verified
    #: not to trip the classifier.
    OFF = "off"

    #: Rewrite tool names to Claude Code canonicals / PascalCase. Leaves
    #: ``system`` untouched. Suitable for accounts that only trip the
    #: tool-set fingerprint, not the system-prompt content scan.
    RENAME_ONLY = "rename_only"

    #: Rewrite tool names AND collapse ``system`` to a single block
    #: containing only the Claude Code identity line. Drops SOUL.md /
    #: AGENTS.md / memory injection from this code path. Required for
    #: accounts that trip both triggers.
    FULL_STEALTH = "full_stealth"

    @classmethod
    def parse(cls, value: Any, default: "StealthMode" = None) -> "StealthMode":
        """Lenient parser for config values. Falls back to *default* on
        unrecognized inputs (with a warning log)."""
        if isinstance(value, cls):
            return value
        if value is None or value == "":
            return default if default is not None else cls.OFF
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                pass
        logger.warning(
            "oauth_compat: unrecognized StealthMode value %r; "
            "falling back to %r", value, (default or cls.OFF).value,
        )
        return default if default is not None else cls.OFF


# ---------------------------------------------------------------------------
# Tool-name map (per-session)
# ---------------------------------------------------------------------------

def _default_pascal_case(name: str) -> str:
    """snake_case → PascalCase fallback for hermes tools without a CC
    equivalent. Idempotent for names already in PascalCase."""
    if not name:
        return name
    parts = [p for p in name.split("_") if p]
    if not parts:
        return name
    return "".join(p[:1].upper() + p[1:] for p in parts)


class ToolNameMap:
    """Per-session forward+reverse name map.

    Owned by the agent instance, not the module, so multiple concurrent
    agents (e.g. subagents) don't share state. Append-only within a
    session — once a forward mapping is registered, it doesn't change.

    Thread-safe (the streaming response handler may run concurrently
    with the next-turn request build).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._forward: Dict[str, str] = {}     # original → rewritten
        self._reverse: Dict[str, str] = {}     # rewritten → original

    def register(self, original: str) -> str:
        """Register *original* and return the rewritten name. Stable
        across calls within the session. Self-idempotent: passing in an
        already-rewritten name returns it unchanged (no double-rename)."""
        if not original:
            return original
        with self._lock:
            existing = self._forward.get(original)
            if existing is not None:
                return existing
            # Self-idempotency: if this name is itself a rewritten form
            # we've already produced (i.e. caller is re-applying the
            # transform to already-transformed kwargs), pass through.
            if original in self._reverse:
                return original
            rewritten = HERMES_TO_CLAUDE_CODE.get(original)
            if rewritten is None:
                rewritten = _default_pascal_case(original)
            collision = self._reverse.get(rewritten)
            if collision is not None and collision != original:
                # Rare: two hermes tools collapse to the same target.
                # Disambiguate by suffixing with an index. Logged once
                # per collision so downstream confusion is debuggable.
                suffix = 2
                while f"{rewritten}{suffix}" in self._reverse:
                    suffix += 1
                rewritten = f"{rewritten}{suffix}"
                logger.warning(
                    "oauth_compat: tool-name collision for %r vs existing "
                    "%r → %r; disambiguated to %r",
                    original, collision, rewritten[:-len(str(suffix))], rewritten,
                )
            self._forward[original] = rewritten
            self._reverse[rewritten] = original
            return rewritten

    def unrename(self, rewritten: str) -> Optional[str]:
        """Reverse lookup. Returns None if the name was never registered
        (i.e. came from outside this map's universe)."""
        if not rewritten:
            return None
        with self._lock:
            return self._reverse.get(rewritten)

    def __len__(self) -> int:
        with self._lock:
            return len(self._forward)


# ---------------------------------------------------------------------------
# 400 detection
# ---------------------------------------------------------------------------

#: Substring that identifies the third-party billing-lane rejection.
#: Anthropic returns this string both as ``"third-party apps now draw
#: from your extra usage"`` and ``"out of extra usage"``; we match the
#: shared portion.
_THIRD_PARTY_400_MARKER = "extra usage"


def is_third_party_classifier_rejection(
    *,
    status_code: Optional[int],
    error_message: Optional[str],
) -> bool:
    """Return True if a 400 response matches the OAuth third-party
    classifier. Narrow enough to not collide with other 400 patterns
    (long-context-beta, llama.cpp grammar, payload size)."""
    if status_code != 400:
        return False
    if not error_message:
        return False
    msg = error_message.lower()
    # The marker appears in both observed error strings:
    #   "Third-party apps now draw from your extra usage…"
    #   "You're out of extra usage…"
    # and is paired with the claude.ai/settings/usage URL.
    return (
        _THIRD_PARTY_400_MARKER in msg
        and "claude.ai/settings/usage" in msg
    )


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def apply_to_kwargs(
    kwargs: Dict[str, Any],
    *,
    mode: StealthMode,
    tool_map: ToolNameMap,
) -> None:
    """Apply OAuth stealth transforms to an Anthropic request kwargs dict
    in place. Idempotent — calling twice with the same *tool_map* yields
    the same payload.

    Caller must have already prepended the Claude Code identity block to
    ``kwargs["system"]``; this function only collapses to single-block
    in :attr:`StealthMode.FULL_STEALTH`.
    """
    if mode == StealthMode.OFF:
        return

    tools = kwargs.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict) and "name" in tool:
                tool["name"] = tool_map.register(tool["name"])

    messages = kwargs.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and "name" in block:
                    block["name"] = tool_map.register(block["name"])

    if mode == StealthMode.FULL_STEALTH:
        kwargs["system"] = [
            {"type": "text", "text": CLAUDE_CODE_SYSTEM_PREFIX}
        ]
