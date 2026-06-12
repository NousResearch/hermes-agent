"""Declarative, cache-safe overrides for named system-prompt fragments.

Every Hermes-authored fragment in the stable tier of the system prompt
carries a stable string *key* (see :data:`FRAGMENT_KEYS`).  Users reshape any
of them from ``config.yaml`` under ``agent.prompt_overrides`` without editing
source:

.. code-block:: yaml

    agent:
      prompt_overrides:
        task_completion:      {mode: replace, text: "..."}
        tool_use_enforcement: {mode: append,  text: "..."}
        google_operational:   {mode: remove}
        # shorthand: a bare string means replace
        steer_channel: "Use the steer channel sparingly."

Design contract — **overrides are pure data, resolved once at prompt-build
time**.  There is no callable hook and no conditional logic, by design: the
assembled prompt must stay a deterministic function of (agent, config) so it
is byte-stable across turns and the upstream prefix cache stays warm.  With no
overrides configured the output is byte-identical to the un-overridden prompt.

A fragment override only takes effect when that fragment is actually emitted
this session (e.g. ``computer_use`` only ships when the computer-use tool is
loaded).  ``append``/``prepend`` to a fragment that isn't present this session
silently no-ops — there is nothing to append to.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

VALID_MODES = ("replace", "append", "prepend", "remove")

# Canonical registry of every override-addressable fragment key.  Keep in sync
# with the ``emit(...)`` call sites in ``agent/system_prompt.py``.  Surfaced to
# users for discovery (``hermes`` docs / tooling) — an override map keyed by
# string is only usable if the keys are enumerable.
FRAGMENT_KEYS: Dict[str, str] = {
    "identity": "Agent identity slot (SOUL.md content, or the default identity).",
    "hermes_help": "Pointer to the hermes-agent skill/docs for questions about Hermes itself.",
    "task_completion": "Universal 'finish the job' / no-fabrication guidance (all models).",
    "tool_guidance": "Composed per-tool behavioral guidance (memory, session_search, skills, kanban).",
    "steer_channel": "Note explaining the mid-turn steer channel in tool results.",
    "computer_use": "macOS background computer-use workflow guidance.",
    "nous_subscription": "Nous subscription / portal guidance block.",
    "tool_use_enforcement": "Tells the model to actually call tools instead of describing intent.",
    "google_operational": "Gemini/Gemma operational directives (absolute paths, parallel calls, etc.).",
    "execution_discipline": "GPT/Codex/Grok execution discipline (tool persistence, verification).",
    "skills": "Skills system-prompt block (available skills and how to load them).",
    "model_identity": "Explicit model-identity line (Alibaba Coding Plan API name workaround).",
    "environment_hints": "Execution-environment hints (WSL, Termux, embedder-supplied hint).",
    "environment_probe": "Local Python/pip/uv/PEP-668 toolchain probe line.",
    "active_profile": "Active Hermes profile note and cross-profile write guidance.",
    "platform_hints": "Platform-specific operational hints (from PLATFORM_HINTS or a plugin).",
}


def normalize_overrides(raw: Any) -> Dict[str, Dict[str, str]]:
    """Validate and normalize ``agent.prompt_overrides`` config into a clean map.

    Accepts the loosely-typed YAML value and returns ``{key: {"mode": ...,
    "text": ...}}`` containing only well-formed entries for known fragment
    keys.  Malformed entries are dropped with a warning rather than raising —
    a bad override should never block prompt assembly.
    """
    if not raw:
        return {}
    if not isinstance(raw, dict):
        logger.warning(
            "agent.prompt_overrides must be a mapping of fragment-key -> override; "
            "got %s. Ignoring.", type(raw).__name__,
        )
        return {}

    result: Dict[str, Dict[str, str]] = {}
    for key, spec in raw.items():
        if key not in FRAGMENT_KEYS:
            logger.warning(
                "agent.prompt_overrides: unknown fragment key %r (ignored). "
                "Valid keys: %s", key, ", ".join(sorted(FRAGMENT_KEYS)),
            )
            continue

        # Shorthand: a bare string is a full replacement.
        if isinstance(spec, str):
            result[key] = {"mode": "replace", "text": spec}
            continue

        if not isinstance(spec, dict):
            logger.warning(
                "agent.prompt_overrides[%r] must be a string or mapping; got %s. "
                "Ignored.", key, type(spec).__name__,
            )
            continue

        mode = str(spec.get("mode", "replace")).lower().strip()
        if mode not in VALID_MODES:
            logger.warning(
                "agent.prompt_overrides[%r]: invalid mode %r (valid: %s). Ignored.",
                key, mode, ", ".join(VALID_MODES),
            )
            continue

        if mode == "remove":
            result[key] = {"mode": "remove", "text": ""}
            continue

        text = spec.get("text", "")
        if text is None:
            text = ""
        if not isinstance(text, str):
            logger.warning(
                "agent.prompt_overrides[%r].text must be a string; got %s. Ignored.",
                key, type(text).__name__,
            )
            continue
        if mode in ("append", "prepend") and not text.strip():
            logger.warning(
                "agent.prompt_overrides[%r]: %s mode with empty text is a no-op. "
                "Ignored.", key, mode,
            )
            continue
        result[key] = {"mode": mode, "text": text}

    return result


def apply_fragment_override(
    overrides: Optional[Dict[str, Dict[str, str]]],
    key: str,
    text: Optional[str],
) -> Optional[str]:
    """Apply any configured override for ``key`` to ``text``.

    Returns the (possibly transformed) fragment text, or ``None`` when the
    fragment should be dropped (``remove`` mode, or an empty result).  ``text``
    is the default fragment content Hermes would emit with no override.
    """
    if not overrides:
        return text
    spec = overrides.get(key)
    if not spec:
        return text

    mode = spec.get("mode", "replace")
    if mode == "remove":
        return None

    new = spec.get("text", "")
    if mode == "replace":
        return new
    if mode == "append":
        if not text or not text.strip():
            return new
        return f"{text}\n\n{new}"
    if mode == "prepend":
        if not text or not text.strip():
            return new
        return f"{new}\n\n{text}"
    return text


__all__ = [
    "FRAGMENT_KEYS",
    "VALID_MODES",
    "normalize_overrides",
    "apply_fragment_override",
]
