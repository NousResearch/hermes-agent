"""Call-site API for process-level shared-metrics facts.

Setup completion, slash commands, compression, model switches, fallbacks and extension installs
happen outside any single tool or model call, so their producers call these functions directly.
Every function takes RAW values (normalization lives in shared_metrics_fields), is a no-op unless
shared metrics are enabled, and never raises into the caller.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from . import shared_metrics_contract as contract
from . import shared_metrics_fields as fields_

logger = logging.getLogger(__name__)


def _emit(mark: str, build: Callable[..., dict[str, str] | None], **raw: Any) -> None:
    try:
        from .relay_shared_metrics import enabled, record_process_mark

        if not enabled():
            return
        data = build(**raw)
        if data is not None:
            record_process_mark(mark, data)
    except Exception:
        logger.debug("Shared-metrics %s not recorded", mark, exc_info=True)


def emit_saved(marks: list[tuple[str, dict[str, str]]]) -> int:
    """``_emit`` for facts recovered from a file: how many rows are settled (saved, rejected, or
    collection off), so the caller deletes the file only then. Blocks on the store: no hot paths."""
    try:
        from .relay_shared_metrics import record_process_marks_saved

        return record_process_marks_saved(marks) if marks else 0
    except Exception:
        logger.debug("Shared-metrics recovered rows not recorded", exc_info=True)
        return 0


def record_setup_completed(*, surface: str, provider: str | None) -> None:
    _emit(contract.SETUP_COMPLETED_MARK, fields_.setup_completed_fields, surface=surface, provider=provider)


def record_slash_command(*, command: str, surface: str) -> None:
    _emit(contract.SLASH_COMMAND_MARK, fields_.slash_command_fields, command=command, surface=surface)


def record_compression(
    *, trigger: str, outcome: str, tokens_before: int | None, context_length: int | None,
    failure_class: str | None = None,
) -> None:
    _emit(
        contract.COMPRESSION_MARK, fields_.compression_fields, trigger=trigger, outcome=outcome,
        tokens_before=tokens_before, context_length=context_length, failure_class=failure_class,
    )


# Per-thread, not on the compressor or the agent: abort paths restore the compressor snapshot (seed
# included) before they emit, and a stalled attempt's worker can unwind after its stall-fallback retry
# began on another thread. An attempt begins and emits on one thread (the pool worker or the caller).
_compression_attempt = threading.local()

# Attempts that ended before anything could fail: another path held the lock, the transcript had
# nothing summarizable (no LLM call was made), the user stopped it, or a newer attempt replaced it.
# Counting these as ``failed`` made a session that is simply all protected tail read as broken.
_SKIPPED_COMPRESSION_CLASSES = contract.COMPRESSION_SKIP_CLASSES


def begin_compression_attempt(trigger: str, tokens_before: Any) -> None:
    _compression_attempt.pending = (trigger, tokens_before)


def finish_compression_attempt(
    commit_status: str, failure_class: str | None, context_length: Any, agent: Any = None,
) -> None:
    """Count this thread's pending compression attempt once; a committed one broke the prompt cache."""
    pending, _compression_attempt.pending = getattr(_compression_attempt, "pending", None), None
    if not pending:
        return
    # Classify the stored value, so the outcome and the recorded reason always agree.
    reason = fields_.compression_failure_class("failed", failure_class)
    outcome = (
        "success" if commit_status == "committed"
        else "skipped" if reason in _SKIPPED_COMPRESSION_CLASSES else "failed"
    )
    record_compression(
        trigger=pending[0], outcome=outcome, tokens_before=pending[1], context_length=context_length,
        failure_class=reason,
    )
    if outcome == "success" and agent is not None:
        from .shared_metrics_efficiency import record_cache_break

        record_cache_break(agent, "compression")


def record_gateway_slash_command(event: Any) -> None:
    """Count a user-typed gateway slash command once: a queued busy-path event re-enters
    ``_handle_message`` when drained, so the event carries the mark."""
    command = event.get_command()
    if not command or getattr(event, "_slash_command_counted", False):
        return
    event._slash_command_counted = True
    record_slash_command(command=command, surface="gateway")


def record_model_switch(
    *, from_provider: str | None, to_provider: str | None, surface: str, from_model: str | None = None,
    session_id: str | None = None,
) -> None:
    """``from_model`` also counts the switch as friction against the model the user left;
    ``session_id`` (the switched session) also counts how many turns that model served first."""
    _emit(
        contract.MODEL_SWITCH_MARK, fields_.model_switch_fields,
        from_provider=from_provider, to_provider=to_provider, surface=surface,
    )
    if from_model:
        from .shared_metrics_model import record_model_friction

        record_model_friction("switch_away", provider=from_provider, model=from_model)
    if session_id:
        try:
            from .relay_shared_metrics import record_model_switch_after

            record_model_switch_after(str(session_id))
        except Exception:
            logger.debug("Shared-metrics model_switch_after not recorded", exc_info=True)


def record_fallback(*, from_provider: str | None, to_provider: str | None, reason: Any) -> None:
    _emit(
        contract.FALLBACK_MARK, fields_.fallback_fields,
        from_provider=from_provider, to_provider=to_provider, reason=reason,
    )


def record_extension_install(*, kind: str, source: str, name: str | None, outcome: str) -> None:
    _emit(
        contract.EXTENSION_INSTALL_MARK, fields_.extension_install_fields,
        kind=kind, source=source, name=name, outcome=outcome,
    )
