"""loop-diagnostics — action dependency graph recorder plugin.

Wires ``hermes_cli.observability.loop_diagnostics_recorder`` to the
observer hooks so a Kanban worker's actions are captured as a dependency
graph (nodes + directed causal/data/loop/retry edges) that the diagnosis
engine consumes on failure.

Zero-cost when disabled
-----------------------
The recorder contract requires that disabled diagnostics add negligible
behavioral impact.  This plugin reads ``kanban.loop_diagnostics.enabled``
at registration time and registers NO hooks when disabled — so the
no-listener fast path in ``model_tools._emit_post_tool_call_hook`` (the
``has_hook`` gate) is never even entered.  There is no per-tool overhead
beyond the plugin manager's existing "no listener" dict lookup.

Identity
--------
The recorder resolves task/run identity from the environment the Kanban
dispatcher pins on worker spawn (``HERMES_KANBAN_TASK`` /
``HERMES_KANBAN_RUN_ID`` / ``HERMES_KANBAN_BOARD``).  In a normal agent
session (no kanban worker), those vars are absent, so the recorder stays
disabled even when the plugin is enabled — no stray trace files.

Fail-open
---------
Every hook callback is wrapped; a recorder error is logged at debug and
never disturbs the worker.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_recorder: Any = None


def _get_recorder() -> Any:
    """Return the process-wide recorder, or None when disabled.

    The recorder is created lazily on first hook event so a worker that
    never runs a tool (e.g. fails at spawn) writes no trace file.
    """
    global _recorder
    if _recorder is None:
        from hermes_cli.observability.loop_diagnostics_recorder import (
            LoopDiagnosticsRecorder,
        )

        rec = LoopDiagnosticsRecorder()
        if not rec.enabled:
            logger.debug(
                "loop-diagnostics: disabled (missing kanban worker identity)"
            )
            _recorder = False
            return None
        rec.start()
        _recorder = rec
    return _recorder if _recorder is not False else None


def _on_pre_tool_call(**kwargs: Any) -> None:
    try:
        rec = _get_recorder()
        if rec is not None:
            rec.on_pre_tool_call(**kwargs)
    except Exception as exc:
        logger.debug("loop-diagnostics: pre_tool_call failed (%s)", exc)


def _on_post_tool_call(**kwargs: Any) -> None:
    try:
        rec = _get_recorder()
        if rec is not None:
            rec.on_post_tool_call(**kwargs)
    except Exception as exc:
        logger.debug("loop-diagnostics: post_tool_call failed (%s)", exc)


def _on_subagent_start(**kwargs: Any) -> None:
    try:
        rec = _get_recorder()
        if rec is not None:
            rec.on_subagent_start(**kwargs)
    except Exception as exc:
        logger.debug("loop-diagnostics: subagent_start failed (%s)", exc)


def _on_subagent_stop(**kwargs: Any) -> None:
    try:
        rec = _get_recorder()
        if rec is not None:
            rec.on_subagent_stop(**kwargs)
    except Exception as exc:
        logger.debug("loop-diagnostics: subagent_stop failed (%s)", exc)


def _finalize(outcome: str = "completed", error: Optional[str] = None) -> None:
    global _recorder
    rec = _recorder
    if rec is None or rec is False:
        return
    try:
        rec.finish(outcome=outcome, error=error)
    except Exception as exc:
        logger.debug("loop-diagnostics: finalize failed (%s)", exc)
    finally:
        _recorder = False


def _on_session_end(**kwargs: Any) -> None:
    # Session ended without a terminal failure — record as completed.
    _finalize(outcome="completed")


def _on_session_finalize(**kwargs: Any) -> None:
    outcome = kwargs.get("outcome") or "completed"
    error = kwargs.get("error") or kwargs.get("error_message")
    _finalize(outcome=outcome, error=error)


def _load_enabled() -> bool:
    try:
        from hermes_cli.observability.loop_diagnostics_recorder import (
            load_recorder_config,
        )

        return bool(load_recorder_config().get("enabled"))
    except Exception as exc:
        logger.debug("loop-diagnostics: config unavailable (%s)", exc)
        return False


def register(ctx) -> None:
    """Register recorder hooks only when diagnostics are enabled."""
    if not _load_enabled():
        logger.debug("loop-diagnostics: disabled by config; zero hooks")
        return
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("subagent_start", _on_subagent_start)
    ctx.register_hook("subagent_stop", _on_subagent_stop)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_hook("on_session_finalize", _on_session_finalize)
    logger.debug("loop-diagnostics: recorder hooks registered")
