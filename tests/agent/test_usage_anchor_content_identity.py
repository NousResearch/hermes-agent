"""An unchanged API message keeps its usage anchor across display-only rewrites."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent.memory_manager import build_memory_context_block
from agent.session_persistence import _db_flush_row
from agent.turn_context import _preflight_request_tokens, build_api_messages, substitute_api_content
from agent.usage_anchor import capture_usage_anchor, message_fingerprint, restore_usage_anchor, set_usage_anchor
from hermes_state import SessionDB


def _agent(**attrs):
    return SimpleNamespace(
        model="fixture", provider="custom", base_url="https://example.invalid/v1",
        api_mode="chat_completions", tools=[], ephemeral_system_prompt="",
        _copy_reasoning_content_for_api=lambda *_: None,
        _should_sanitize_tool_calls=lambda: False, **attrs,
    )


@pytest.mark.parametrize("rewrite", ["persist_override", "sanitized_display"])
def test_persisted_display_rewrite_preserves_priced_wire_message(tmp_path, rewrite):
    sent = "visible question\n\n" + build_memory_context_block("recalled context")
    messages = [{"role": "user", "content": sent}]
    original = deepcopy(messages)
    sid = "wire-identity"
    db_path = tmp_path / "state.db"
    with SessionDB(db_path) as db:
        db.create_session(sid, source="cli")
        agent = _agent(session_id=sid, _session_db=db,
                       _persist_user_message_override="visible question" if rewrite == "persist_override" else None)
        db.append_messages_batch(sid, [_db_flush_row(agent, messages[0], True)])
        set_usage_anchor(agent, capture_usage_anchor(30_000, 25, messages))
        assert messages == original

    with SessionDB(db_path) as db:
        restored = db.get_messages_as_conversation(sid)
        assert restored[0]["content"] != sent
        resumed = _agent(session_id=sid, _session_db=db, _usage_anchor=None)
        wire, _ = build_api_messages(
            resumed, restored, current_turn_user_idx=None, ext_prefetch_cache="",
            plugin_user_context="", moa_config=None, active_system_prompt="",
        )
        assert wire[0]["content"] == sent
        restore_usage_anchor(resumed, restored)
        assert _preflight_request_tokens(resumed, restored, "") == 30_025

        # A real change to replayed content still invalidates the old provider reading.
        restored[0]["api_content"] += " changed"
        resumed._usage_anchor = None
        restore_usage_anchor(resumed, restored)
        assert resumed._usage_anchor is None


@pytest.mark.parametrize("role,sidecar", [
    ("user", "wire"), ("assistant", "wire"), ("user", ""),
    ("user", {"text": "ignored"}), ("tool", "ignored"), ("system", "ignored"),
])
def test_fingerprint_tracks_replayed_content_without_mutating_message(role, sidecar):
    message = {"role": role, "content": "display", "api_content": sidecar}
    original = deepcopy(message)
    wire = deepcopy(message)
    substitute_api_content(wire)
    assert message_fingerprint(message) == message_fingerprint(wire)
    edited = dict(message, content="new display")
    wire_edited = deepcopy(edited)
    substitute_api_content(wire_edited)
    assert (message_fingerprint(message) == message_fingerprint(edited)) == (wire == wire_edited)
    if role == "user" and sidecar == "wire":
        assert message_fingerprint(message) != message_fingerprint(dict(message, api_content="new wire"))
    assert message == original
