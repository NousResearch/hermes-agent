"""Truth Ledger plugin package.

Composes lifecycle hooks and optional slash command registration:
  /truth-ledger status
  /truth-ledger review
  /truth-ledger process [--limit N] [--apply]
  /truth-ledger rebuild [--apply]
  /truth-ledger retract <fact-id> [--apply]
  /truth-ledger export [--apply]
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from hermes_constants import get_hermes_home

from .commands import handle_truth_ledger_command
from .redaction import sanitize_payload
from .schemas import validate_document
from .spool import TruthSpool

_LOG = logging.getLogger(__name__)
_ACTIVE_PROFILE = "default"
_MAX_SEEN_ENVELOPES = 1024
_SEEN_ENVELOPES: OrderedDict[tuple[str, str, str], None] = OrderedDict()
_SEEN_ENVELOPES_LOCK = Lock()


def _remember_envelope_key(dedupe_key: tuple[str, str, str]) -> bool:
    with _SEEN_ENVELOPES_LOCK:
        if dedupe_key in _SEEN_ENVELOPES:
            return False
        _SEEN_ENVELOPES[dedupe_key] = None
        while len(_SEEN_ENVELOPES) > _MAX_SEEN_ENVELOPES:
            _SEEN_ENVELOPES.popitem(last=False)
        return True


def _forget_envelope_key(dedupe_key: tuple[str, str, str]) -> None:
    with _SEEN_ENVELOPES_LOCK:
        _SEEN_ENVELOPES.pop(dedupe_key, None)


def register(ctx) -> None:
    global _ACTIVE_PROFILE
    _ACTIVE_PROFILE = str(getattr(ctx, "profile_name", "default") or "default")

    if hasattr(ctx, "register_hook"):
        ctx.register_hook("on_session_start", on_session_start)
        ctx.register_hook("post_llm_call", on_post_llm_call)

    if hasattr(ctx, "register_command"):
        def _handle_command(raw_args: str):
            return handle_truth_ledger_command(raw_args, runtime_ctx=ctx)

        ctx.register_command(
            "truth-ledger",
            handler=_handle_command,
            description="Truth Ledger operator commands: status/review/process/rebuild/retract/export.",
            args_hint="status|review|process|rebuild|retract <fact-id>|export",
            admin_only=True,
        )


def _truth_root() -> Path:
    return get_hermes_home() / "truth-ledger"


def _is_eligible_turn(kwargs: dict[str, Any]) -> bool:
    if not bool(kwargs.get("completed", False)):
        return False
    if bool(kwargs.get("failed", False)) or bool(kwargs.get("interrupted", False)):
        return False

    turn_exit_reason = str(kwargs.get("turn_exit_reason") or "")
    if not turn_exit_reason.startswith("text_response("):
        return False

    if kwargs.get("kanban_task_id"):
        return False
    if bool(kwargs.get("is_subagent", False)):
        return False
    if int(kwargs.get("delegate_depth") or 0) > 0:
        return False
    return True


def _now_rfc3339_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _bounded_text(value: Any, *, max_chars: int = 65_536) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _build_source_envelope(kwargs: dict[str, Any], profile: str) -> dict[str, Any]:
    envelope = {
        "schema_name": "truth-ledger.source-envelope.v1",
        "schema_version": 1,
        "captured_at": _now_rfc3339_utc(),
        "profile": profile,
        "session_id": kwargs.get("session_id"),
        "turn_id": kwargs.get("turn_id"),
        "origin": {
            "platform": str(kwargs.get("platform") or "cli"),
            "conversation_id": kwargs.get("conversation_id"),
            "chat_id": kwargs.get("chat_id"),
            "thread_id": kwargs.get("thread_id"),
            "chat_type": kwargs.get("chat_type"),
            "speaker_id": kwargs.get("speaker_id"),
        },
        "input": {"user_message": _bounded_text(kwargs.get("user_message"))},
        "output": {"assistant_response": _bounded_text(kwargs.get("assistant_response"))},
    }

    sanitized = sanitize_payload(envelope)
    input_payload = sanitized.get("input") if isinstance(sanitized, dict) else None
    output_payload = sanitized.get("output") if isinstance(sanitized, dict) else None
    if isinstance(input_payload, dict):
        input_payload["user_message"] = _bounded_text(input_payload.get("user_message"))
    if isinstance(output_payload, dict):
        output_payload["assistant_response"] = _bounded_text(output_payload.get("assistant_response"))

    validate_document("source-envelope.v1", sanitized)
    return sanitized


def _enqueue_source_envelope(envelope: dict[str, Any]) -> dict[str, Any]:
    spool = TruthSpool(_truth_root())
    return spool.enqueue(envelope)


def on_post_llm_call(**kwargs: Any) -> None:
    if not _is_eligible_turn(kwargs):
        return

    profile = str(kwargs.get("profile_name") or _ACTIVE_PROFILE or "default")
    session_id = str(kwargs.get("session_id") or "")
    turn_id = str(kwargs.get("turn_id") or "")
    if not session_id or not turn_id:
        return

    dedupe_key = (profile, session_id, turn_id)
    if not _remember_envelope_key(dedupe_key):
        return

    try:
        result = _enqueue_source_envelope(_build_source_envelope(kwargs, profile=profile))
        if not bool(result.get("ok", False)):
            _forget_envelope_key(dedupe_key)
    except Exception:
        _forget_envelope_key(dedupe_key)
        _LOG.debug("truth-ledger post_llm_call enqueue failed (fail-open)", exc_info=True)


def on_session_start(**_kwargs: Any) -> None:
    try:
        spool = TruthSpool(_truth_root())
        spool.recover_orphan_payloads()
        spool.recover_stale_processing(stale_seconds=900)
    except Exception:
        _LOG.debug("truth-ledger on_session_start recovery failed (fail-open)", exc_info=True)
