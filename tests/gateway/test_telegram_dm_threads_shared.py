"""``platforms.telegram.extra.dm_threads``: ``shared`` keys a Telegram DM as one session whatever
``message_thread_id`` its messages carry; ``per_thread`` (default) keeps one session per thread."""

import logging

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, SessionStore
from tests.gateway.test_base_topic_sessions import DummyTelegramAdapter


def _store(tmp_path, monkeypatch, extra):
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t", extra=extra)})
    return SessionStore(sessions_dir=tmp_path, config=config)


def _dm(thread_id):
    return SessionSource(platform=Platform.TELEGRAM, chat_id="111", chat_type="dm", user_id="111",
                         thread_id=thread_id)


def _session_ids(store):
    return {store.get_or_create_session(_dm(tid)).session_id for tid in ("501", "777", None)}


@pytest.mark.parametrize("extra, expected", [
    ({}, 3),
    ({"dm_threads": "per_thread"}, 3),
    ({"dm_threads": "shared"}, 1),
    # Operator-declared topics stay their own workspaces under shared.
    ({"dm_threads": "shared", "dm_topics": [{"chat_id": 111, "topics": [{"name": "W", "thread_id": 501}]}]}, 2),
])
def test_dm_thread_ids_share_one_session_only_when_opted_in(tmp_path, monkeypatch, extra, expected):
    assert len(_session_ids(_store(tmp_path, monkeypatch, extra))) == expected


def test_topic_mode_lanes_stay_isolated_under_shared(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch, {"dm_threads": "shared"})
    store._db.enable_telegram_topic_mode(chat_id="111", user_id="111")
    assert len(_session_ids(store)) == 3


def test_unknown_mode_warns_and_keeps_per_thread(tmp_path, monkeypatch, caplog):
    store = _store(tmp_path, monkeypatch, {"dm_threads": "shard"})
    with caplog.at_level(logging.WARNING, logger="gateway.session_dm_threads"):
        assert len(_session_ids(store)) == 3
    assert "per_thread, shared" in caplog.text


def test_adapter_batch_key_matches_the_store_key(tmp_path, monkeypatch):
    """The adapter's batching/busy key and the store's routing key must agree, or a shared DM
    still runs one busy guard per thread."""
    store = _store(tmp_path, monkeypatch, {"dm_threads": "shared"})
    adapter = DummyTelegramAdapter()
    adapter.config.extra["dm_threads"] = "shared"
    adapter.set_session_store(store)
    keys = {adapter._event_session_key(MessageEvent(text="hi", source=_dm(tid))) for tid in ("501", "777")}
    assert keys == {store._generate_session_key(_dm("501"))} == {"agent:main:telegram:dm:111"}
