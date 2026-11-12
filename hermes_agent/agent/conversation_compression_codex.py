"""Codex app-server compaction: the route ``compress_context`` takes for Codex-owned threads.

Split from ``agent/conversation_compression.py`` (facade size cap), which imports it at module level;
this module must never import the facade at module level (import cycle), so it late-imports the
facade's shared helpers inside functions. It logs under the facade's logger name so log consumers keep
one source.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Optional

from agent.conversation_compression_telemetry import _emit_bypassed_attempt_telemetry

logger = logging.getLogger("agent.conversation_compression")


def _codex_compaction_cooldown_remaining(agent: Any) -> float:
    """Seconds left on this session's compaction-failure cooldown (0 = clear)."""
    compressor = getattr(agent, "context_compressor", None)
    getter = getattr(compressor, "get_active_compression_failure_cooldown", None)
    if not callable(getter):
        return 0.0
    try:
        state = getter(refresh=True)
    except Exception:
        logger.debug("codex compaction cooldown lookup failed", exc_info=True)
        return 0.0
    try:
        return max(0.0, float(state.get("remaining_seconds") or 0.0)) if state else 0.0
    except (TypeError, ValueError):
        return 0.0


def _record_codex_compaction_failure(agent: Any, error: str) -> None:
    """Arm the shared compression-failure cooldown after a failed codex compaction.
    The codex path returns the transcript unchanged, so without a cooldown the still-over-threshold session
    would retry every turn."""
    from agent.context_compressor import _SUMMARY_FAILURE_COOLDOWN_SECONDS
    from agent.conversation_compression import _swallow
    compressor = getattr(agent, "context_compressor", None)
    recorder = getattr(compressor, "_record_compression_failure_cooldown", None)
    if not callable(recorder):
        return
    with _swallow('codex compaction cooldown persist failed', exc_info=True):
        recorder(_SUMMARY_FAILURE_COOLDOWN_SECONDS, error)


def _compress_context_via_codex_app_server(
    agent: Any, messages: list, system_message: Optional[str], *, approx_tokens: Optional[int] = None,
    task_id: str = "default", force: bool = False, started_at: float, attempt_seed: dict[str, Any],
) -> tuple[list, str]:
    """Route compaction to Codex app-server for Codex-owned threads.
    Rewriting the local transcript would not shrink the Codex thread, so Codex compacts its own thread and
    Hermes' transcript is left unchanged. Every exit, raising ones included, logs one attempt record."""
    from agent.conversation_compression import (
        COMPACTION_STATUS, _CompressionActivityHeartbeat, _emit_compaction_done, _existing_system_prompt,
        _reset_read_dedup_caches, _swallow,
    )

    def _record(commit_status: str, failure_class: Optional[str], method: str = "none") -> None:
        _emit_bypassed_attempt_telemetry(
            agent, started_at, commit_status=commit_status, failure_class=failure_class, approx_tokens=approx_tokens,
            route="codex_app_server", method=method, attempt_seed=attempt_seed,
        )

    _sid = getattr(agent, "session_id", None) or "none"
    _tokens = f"{approx_tokens:,}" if approx_tokens else "unknown"
    auto_mode = str(getattr(agent, "codex_app_server_auto_compaction", "native") or "native").lower()
    if auto_mode not in {"native", "hermes", "off"}:
        auto_mode = "native"
    skip_reason = skip_status = skip_class = None
    if not force and auto_mode != "hermes":
        skip_reason, skip_status, skip_class = f"mode={auto_mode} force=false", "skipped", f"codex_auto_{auto_mode}"
    elif not force:
        # Automatic entrypoints honor the compressor-owned cooldown: a recent compaction
        # failed, and retrying every turn is what thrashes.
        _cooldown_remaining = _codex_compaction_cooldown_remaining(agent)
        if _cooldown_remaining > 0:
            skip_reason = f"failure cooldown active for {_cooldown_remaining:.0f}s"
            skip_status, skip_class = "blocked", "blocked:cooldown"
    codex_session = getattr(agent, "_codex_session", None)
    if skip_reason is None and codex_session is None:
        skip_reason, skip_status, skip_class = "no active codex thread", "skipped", "codex_no_thread"
    if skip_reason is not None:
        logger.info(
            "codex app-server compaction skipped: %s (session=%s messages=%d tokens=~%s)", skip_reason, _sid,
            len(messages), _tokens,
        )
        _record(skip_status, skip_class)
        return messages, _existing_system_prompt(agent, system_message)
    logger.info("codex app-server compaction started: session=%s messages=%d tokens=~%s", _sid, len(messages), _tokens)
    with contextlib.suppress(Exception):
        agent._emit_status(COMPACTION_STATUS)
    _activity_heartbeat = _CompressionActivityHeartbeat(agent, emit_client_status=True).start()
    try:
        result = codex_session.compact_thread()
    except BaseException as exc:
        _activity_heartbeat.stop("context compression failed")
        # Same class shape as the Hermes route's raising exits; the caller still gets the original error.
        _record("aborted", f"exception:{type(exc).__name__}")
        raise
    failed = bool(getattr(result, "interrupted", False) or getattr(result, "error", None))
    _activity_heartbeat.stop("context compression failed" if failed else "context compression completed")
    if getattr(result, "should_retire", False):
        with contextlib.suppress(Exception):
            codex_session.close()
        agent._codex_session = None
    if failed:
        with contextlib.suppress(Exception):
            agent._emit_warning(f"⚠ Codex app-server compaction failed: {result.error}")
        # The transcript is returned unchanged, so the session is still over
        # threshold. Without a brake the next turn retries immediately.
        _record_codex_compaction_failure(agent, str(getattr(result, "error", None) or "compaction interrupted"))
        _record("failed", "codex_compaction_failed")
        return messages, _existing_system_prompt(agent, system_message)
    with _swallow('codex compaction bookkeeping failed', exc_info=True):
        from agent.codex_runtime import _record_codex_app_server_compaction, _record_codex_app_server_usage
        _record_codex_app_server_compaction(agent, result, approx_tokens=approx_tokens, force=True)
        # An empty usage report must consume the pending verdict, not leave deferral
        # armed until a later turn; minimal test engines may lack update_from_response.
        if hasattr(agent.context_compressor, "update_from_response"):
            _record_codex_app_server_usage(agent, result, messages=messages)
    _reset_read_dedup_caches(task_id, session_id=agent.session_id or "")
    logger.info(
        "codex app-server compaction done: session=%s thread=%s turn=%s", _sid,
        getattr(result, "thread_id", None) or "", getattr(result, "turn_id", None) or "",
    )
    existing_prompt = _existing_system_prompt(agent, system_message)
    _record("committed", None, method="provider")
    # Terminal edge only on success — failure/interrupt paths above return
    # without it, matching the main compress_context() gating.
    _emit_compaction_done(agent)
    return messages, existing_prompt
