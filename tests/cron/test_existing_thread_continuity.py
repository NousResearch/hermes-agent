"""Existing origin threads retain cron delivery context across session cold starts."""

from types import SimpleNamespace

import pytest

from cron.scheduler_delivery import _seed_live_delivery_sessions
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore


def target(store, **overrides):
    values = {
        "job": {"id": "research", "name": "research", "attach_to_session": True},
        "origin": {
            "platform": "slack",
            "chat_id": "C00000000",
            "thread_id": "100.000000",
            "scope_id": "T00000000",
        },
        "runtime_adapter": SimpleNamespace(_session_store=store),
        "platform_name": "slack",
        "chat_id": "C00000000",
        "thread_id": "100.000000",
        "opened_thread_id": None,
        "mirror_text": "research evidence",
        "is_dm_target": False,
        "origin_target": True,
        "origin_user_id": None,
        "mirror_this_target": True,
        "in_channel_surface": False,
        "inchannel_continuable": False,
    }
    return SimpleNamespace(**(values | overrides))


def reply_source(team="T00000000", user="U00000000"):
    return SessionSource(
        platform=Platform.SLACK,
        chat_id="C00000000",
        chat_type="group",
        thread_id="100.000000",
        scope_id=team,
        user_id=user,
    )


def test_existing_origin_thread_survives_restart_and_is_shared_with_members(tmp_path):
    config = GatewayConfig()
    sessions = tmp_path / "sessions"
    store = SessionStore(sessions, config)
    try:
        _seed_live_delivery_sessions(target(store), "100.000001")
        assert len(store._entries) == 1
        seeded_id = next(iter(store._entries.values())).session_id
        transcript = store.load_transcript(seeded_id)
        assert len(transcript) == 1
        assert transcript[0]["role"] == "user"
    finally:
        store.close_all_db_handles()
    restarted = SessionStore(sessions, config)
    try:
        for user in ("U00000000", "U00000001"):
            entry = restarted.get_or_create_session(reply_source(user=user))
            assert entry.session_id == seeded_id
            assert restarted.load_transcript(entry.session_id) == transcript
        other_team = restarted.get_or_create_session(reply_source(team="T00000001"))
        assert other_team.session_id != seeded_id
        assert restarted.load_transcript(other_team.session_id) == []
    finally:
        restarted.close_all_db_handles()


def test_existing_origin_thread_preserves_history_and_appends_once(tmp_path):
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    try:
        entry = store.get_or_create_session(reply_source())
        messages = [
            {"role": "user", "content": "Check the evidence"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_evidence",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "content": "evidence", "tool_call_id": "call_evidence"},
            {"role": "assistant", "content": "Evidence checked"},
        ]
        for message in messages:
            store.append_to_transcript(entry.session_id, message)
        history = store.load_transcript(entry.session_id)
        _seed_live_delivery_sessions(target(store), "100.000001")
        assert (
            store.get_or_create_session(reply_source()).session_id == entry.session_id
        )
        transcript = store.load_transcript(entry.session_id)
        assert transcript[:-1] == history
        assert len(transcript) == len(history) + 1
        assert transcript[-1]["role"] == "user"
    finally:
        store.close_all_db_handles()


@pytest.mark.parametrize(
    "overrides",
    [
        {"mirror_this_target": False},
        {"origin_target": False, "mirror_this_target": False},
        {"origin_target": False},
        {"thread_id": None, "mirror_this_target": False},
    ],
)
def test_existing_thread_seed_does_not_expand_target_authorization(tmp_path, overrides):
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    try:
        _seed_live_delivery_sessions(target(store, **overrides), "100.000001")
        assert store._entries == {}
    finally:
        store.close_all_db_handles()
