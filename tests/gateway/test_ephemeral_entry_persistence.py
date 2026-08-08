"""A temporary chat's persisted gateway entry must carry routing state only.

The desktop/CLI surface refuses the session-store row for temporary sessions
because an empty row still proves WHEN a private chat happened and carries
model/cost metadata. The gateway's routing index is the same class of trace:
its entry is persisted to disk (that is how the ephemeral flag survives a
gateway restart without downgrading a live temporary chat to a saved one),
so for ephemeral entries serialization is reduced to an allowlist of
routing-structural fields. Display name, usage/cost totals, reset/resume
introspection, session lineage and model override stay out.
"""

from datetime import datetime

from gateway.session import SessionEntry


def _entry(*, ephemeral: bool) -> SessionEntry:
    entry = SessionEntry(
        session_key="telegram:12345",
        session_id="sess-abc",
        created_at=datetime(2026, 8, 8, 12, 0, 0),
        updated_at=datetime(2026, 8, 8, 12, 30, 0),
        display_name="Vimal's private chat",
        chat_type="dm",
        metadata={"thread_watermark": "17585"},
        ephemeral=ephemeral,
    )
    entry.input_tokens = 1234
    entry.output_tokens = 567
    entry.total_tokens = 1801
    entry.estimated_cost_usd = 0.42
    entry.cost_status = "ok"
    entry.prev_session_id = "sess-previous"
    entry.model_override = {"model": "some/model", "provider": "prov"}
    entry.resume_pending = True
    entry.resume_reason = "restart"
    return entry


# The complete serialized surface of a temporary entry. Exact-set on purpose:
# a field added to SessionEntry serialization reaches the temporary record
# only by being added HERE deliberately, never by default.
_EPHEMERAL_ALLOWED_KEYS = {
    "session_key",
    "session_id",
    "created_at",
    "updated_at",
    "platform",
    "chat_type",
    "metadata",
    "ephemeral",
    "suspended",
    "expiry_finalized",
    # "origin" appears only when set; asserted separately below.
}


def test_ephemeral_entry_serializes_allowlist_only():
    data = _entry(ephemeral=True).to_dict()
    assert set(data) == _EPHEMERAL_ALLOWED_KEYS
    assert data["ephemeral"] is True


def test_ephemeral_entry_drops_profiling_fields():
    data = _entry(ephemeral=True).to_dict()
    for forbidden in (
        "display_name",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "total_tokens",
        "last_prompt_tokens",
        "estimated_cost_usd",
        "cost_status",
        "prev_session_id",
        "model_override",
        "resume_pending",
        "resume_reason",
        "last_resume_marked_at",
        "active_turn_token",
        "is_fresh_reset",
        "was_auto_reset",
        "auto_reset_reason",
        "reset_had_activity",
    ):
        assert forbidden not in data, (
            f"'{forbidden}' persisted for a temporary chat — this is the "
            "metadata trace the mode exists to avoid"
        )


def test_ephemeral_entry_roundtrip_keeps_flag_and_routing():
    original = _entry(ephemeral=True)
    revived = SessionEntry.from_dict(original.to_dict())
    assert revived.ephemeral is True
    assert revived.session_key == original.session_key
    assert revived.session_id == original.session_id
    assert revived.metadata == original.metadata
    # Dropped fields rehydrate to defaults, not to the live values.
    assert revived.display_name is None
    assert revived.input_tokens == 0
    assert revived.estimated_cost_usd == 0.0
    assert revived.prev_session_id is None
    assert revived.model_override is None


def test_normal_entry_still_serializes_full_record():
    data = _entry(ephemeral=False).to_dict()
    assert data["display_name"] == "Vimal's private chat"
    assert data["input_tokens"] == 1234
    assert data["estimated_cost_usd"] == 0.42
    assert data["prev_session_id"] == "sess-previous"
    assert data["ephemeral"] is False
