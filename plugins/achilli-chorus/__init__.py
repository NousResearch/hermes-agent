"""
Achilli Chorus -- Multi-agent output harmonization.

Collects subagent results via subagent_stop hook. When a batch of siblings
completes, uses ctx.llm to synthesize unified output. Provides chorus_status
and harmonize_results tools.

Note: works with child summary strings (from subagent_stop payload), not raw
output. Parent agent should read full subagent results for detailed work.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Batch buffer
# ---------------------------------------------------------------------------

_BATCH: List[Dict[str, Any]] = []
_BATCH_START: Optional[float] = None
_LAST_HARMONIZED: Optional[float] = None
_HARMONIZATION_PENDING: bool = False


def _get_batch_window() -> float:
    import os
    try:
        return float(os.environ.get("ACHILLI_CHORUS_BATCH_WINDOW", "60"))
    except (ValueError, TypeError):
        return 60.0


def _get_max_tokens() -> int:
    import os
    try:
        return int(os.environ.get("ACHILLI_CHORUS_MAX_TOKENS", "1000"))
    except (ValueError, TypeError):
        return 1000


def _llm_disabled() -> bool:
    import os
    return os.environ.get("ACHILLI_CHORUS_DISABLE_LLM", "").lower() in {
        "1", "true", "yes"
    }


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def _on_subagent_stop(
    child_role: Optional[str] = None,
    child_summary: Optional[str] = None,
    child_status: Optional[str] = None,
    duration_ms: int = 0,
    parent_session_id: Optional[str] = None,
    **_: Any,
) -> None:
    """Buffer each subagent result for batch harmonization."""
    global _BATCH, _BATCH_START, _HARMONIZATION_PENDING

    entry = {
        "timestamp": time.time(),
        "child_role": child_role or "unnamed",
        "child_summary": (child_summary or "")[:500],
        "child_status": child_status or "unknown",
        "duration_ms": duration_ms,
        "parent_session_id": parent_session_id,
    }

    if not _BATCH:
        _BATCH_START = entry["timestamp"]

    _BATCH.append(entry)

    # Check if enough time has passed since first child to declare batch complete
    window = _get_batch_window()
    elapsed = entry["timestamp"] - (_BATCH_START or entry["timestamp"])
    if elapsed >= window:
        _HARMONIZATION_PENDING = True
        logger.debug(
            "chorus: batch ready (%d children, %.1fs elapsed)", len(_BATCH), elapsed
        )
    else:
        logger.debug(
            "chorus: buffered child '%s' (%d in batch, %.1fs/%.1fs window)",
            child_role, len(_BATCH), elapsed, window,
        )


def _on_session_end(**_: Any) -> None:
    """Trigger harmonization if a complete batch is pending."""
    global _HARMONIZATION_PENDING
    if _HARMONIZATION_PENDING:
        logger.debug("chorus: session end triggered harmonization check")
        # Actual harmonization is done via harmonize_results tool to avoid
        # blocking the session end hook. We just set the flag.
        _HARMONIZATION_PENDING = False


# ---------------------------------------------------------------------------
# Harmonization
# ---------------------------------------------------------------------------


def _do_harmonize(ctx: Any) -> str:
    """Run LLM synthesis on the current batch. Returns markdown string."""
    if not _BATCH:
        return "No subagent results to harmonize."

    if _llm_disabled():
        return _format_text_report()

    max_tokens = _get_max_tokens()
    batch_json = json.dumps(_BATCH, indent=2, default=str)

    prompt = (
        "You are a synthesis engine. Below are results from parallel subagent "
        "tasks that were working on the same overall goal.\n\n"
        "## Subagent Results\n\n"
        f"{batch_json}\n\n"
        "## Instructions\n\n"
        "Analyze these results and produce:\n"
        "1. **Conflicts**: Where do the results contradict each other?\n"
        "2. **Overlaps**: Where did multiple children do similar work?\n"
        "3. **Gaps**: What should have been covered but was not?\n"
        "4. **Synthesis**: A unified recommendation that reconciles conflicts\n"
        "5. **Review flags**: Items that need human verification\n\n"
        "Keep it concise. Use bullet points."
    )

    try:
        result = ctx.llm(prompt, max_tokens=max_tokens)
        if isinstance(result, dict):
            return result.get("content", result.get("text", str(result)))
        return str(result)
    except Exception as exc:
        logger.warning("chorus: LLM harmonization failed: %s", exc)
        return f"Harmonization LLM call failed: {exc}\n\nRaw results:\n{_format_text_report()}"


def _format_text_report() -> str:
    """Format batch as plain text when LLM is disabled or fails."""
    lines = ["# Chorus Harmonization Report\n"]
    for i, entry in enumerate(_BATCH):
        lines.append(f"## Child {i + 1}: {entry['child_role']} ({entry['child_status']})")
        lines.append(f"Duration: {entry['duration_ms']}ms")
        lines.append("")
        lines.append(entry["child_summary"] or "(empty)")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def _chorus_status_tool(*_, **__) -> dict:
    """Return current harmonization buffer status."""
    window = _get_batch_window()
    elapsed = 0.0
    if _BATCH and _BATCH_START:
        elapsed = time.time() - _BATCH_START
    return {
        "buffered_children": len(_BATCH),
        "batch_window_s": window,
        "elapsed_in_window_s": round(elapsed, 1),
        "batch_ready": elapsed >= window and len(_BATCH) >= 2,
        "harmonization_pending": _HARMONIZATION_PENDING,
        "last_harmonized": _LAST_HARMONIZED,
        "roles": [e.get("child_role", "unnamed") for e in _BATCH],
        "statuses": {e.get("child_role", "?"): e.get("child_status") for e in _BATCH},
    }


def _harmonize_results_tool(ctx: Any = None, *_, **__) -> str:
    """Run or return harmonization of the current batch. Requires ctx."""
    global _LAST_HARMONIZED

    if ctx is None:
        return "Error: ctx not available for harmonization. Use via agent tool dispatch."

    if len(_BATCH) < 2:
        return "Harmonization requires at least 2 subagent results. Current buffer: {}".format(
            len(_BATCH)
        )

    result = _do_harmonize(ctx)
    _LAST_HARMONIZED = time.time()
    return result


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    ctx.register_hook("subagent_stop", _on_subagent_stop)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_tool(
        name="chorus_status",
        description="Return the current chorus harmonization buffer status (achilli-chorus)",
        parameters={"type": "object", "properties": {}},
        handler=_chorus_status_tool,
    )
    # harmonize_results needs ctx access; register it but document that
    # it works best when called by the agent after children complete
    ctx.register_tool(
        name="harmonize_results",
        description=(
            "Harmonize buffered subagent results using the host LLM. "
            "Requires at least 2 children in the buffer. "
            "Call after a batch of subagents completes."
        ),
        parameters={"type": "object", "properties": {}},
        handler=lambda **kw: _harmonize_results_tool(ctx=ctx, **kw),
    )
