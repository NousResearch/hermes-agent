"""Test factory for session records: :func:`session_record` builds a full :class:`SessionRecord` with every field set, by default the record a backend returns for a fresh ``cli`` session, and takes keyword overrides for any field.

Test plumbing; nothing here is product code.
"""

from __future__ import annotations

from typing import Any

from persistence import SessionId
from persistence.conversation.sessions import SessionLineageKind, SessionRecord

#: The record of a never-touched session created as ``NewSession("sess", "cli")``, apart from its creation instant.
SESSION_RECORD_DEFAULTS: dict[str, Any] = {
    "session_id": SessionId("sess"),
    "source": "cli",
    "created_source": "cli",
    "user_id": None,
    "session_key": None,
    "chat_id": None,
    "chat_type": None,
    "thread_id": None,
    "display_name": None,
    "origin": None,
    "expiry_finalized": False,
    "model": None,
    "model_config": {},
    "system_prompt": None,
    "system_prompt_hash": None,
    "parent_session_id": None,
    "lineage_kind": SessionLineageKind.NONE,
    "reset_from": None,
    "branched_from": None,
    "started_at": 0.0,
    "ended_at": None,
    "end_reason": None,
    "message_count": 0,
    "tool_call_count": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_tokens": 0,
    "cache_write_tokens": 0,
    "reasoning_tokens": 0,
    "api_call_count": 0,
    "estimated_cost_usd": None,
    "actual_cost_usd": None,
    "billing_provider": None,
    "billing_base_url": None,
    "billing_mode": None,
    "cost_status": None,
    "cost_source": None,
    "pricing_version": None,
    "cwd": None,
    "git_branch": None,
    "git_repo_root": None,
    "git_metadata_generation": 0,
    "title": None,
    "title_source": None,
    "last_activity_at": None,
    "last_activity_description": None,
    "last_activity_provenance": None,
    "handoff_state": None,
    "handoff_platform": None,
    "handoff_error": None,
    "compression_failure_cooldown_until": None,
    "compression_failure_error": None,
    "compression_fallback_streak": 0,
    "compression_ineffective_count": 0,
    "compression_recovery_deadline": None,
    "compression_overload_streak": 0,
    "profile_name": None,
    "transport_profile": None,
    "rewind_count": 0,
    "archived": False,
    "auto_archived": False,
    "pinned": False,
    "hidden": False,
    "last_read_at": None,
    "tool_pin": None,
    "transcript_revision": 0,
    "runtime_revision": 0,
    "runtime_generation": 0,
    "lineage_origin_id": None,
    "model_config_corrupt": False,
    "model_config_json": None,
    "origin_json": None,
    "tool_names": None,
}


def session_record(**overrides: Any) -> SessionRecord:
    """A full record: :data:`SESSION_RECORD_DEFAULTS` with ``overrides`` applied; an unknown field name is a ``TypeError``."""
    return SessionRecord(**(SESSION_RECORD_DEFAULTS | overrides))
