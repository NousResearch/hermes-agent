"""Session reuse must refresh ``origin.message_id`` for reply anchoring.

``SessionEntry.origin`` is captured once at session creation and was never
updated afterwards, so its ``message_id`` went stale after the first turn.
Background deliveries (cron results, delegate_task completions) read
``origin.message_id`` as their reply anchor. On platforms that route replies
by message_id — notably Feishu topic groups, where ``thread_id`` alone does
not pin the reply to the correct topic — a stale/missing anchor made the
delivery spawn a new top-level thread instead of replying in the ongoing
conversation.

The fix refreshes ``entry.origin.message_id`` from the freshest inbound
``source.message_id`` on every session-reuse path and persists it.

Covers:
  - a second turn on the same routing key refreshes origin.message_id in
    memory AND persists it to sessions.json (survives a simulated restart)
  - a turn with no inbound message_id leaves the previous anchor intact
    (no clobbering the last-known-good anchor with None)
"""
import json

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore


def _make_source(message_id=None) -> SessionSource:
    return SessionSource(
        platform=Platform.FEISHU,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="group",
        thread_id="t1",
        message_id=message_id,
    )


@pytest.fixture
def store_factory(tmp_path, monkeypatch):
    """Build SessionStores over a shared sessions dir, without SQLite."""

    def _raise():
        raise RuntimeError("SQLite disabled in test")

    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", _raise)

    def _make() -> SessionStore:
        store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
        assert store._db is None
        return store

    return _make


def _sessions_json(tmp_path) -> dict:
    raw = (tmp_path / "sessions.json").read_text(encoding="utf-8")
    return json.loads(raw)


def test_reuse_refreshes_and_persists_origin_message_id(store_factory, tmp_path):
    store = store_factory()

    # Turn 1: session is created; origin captures the first inbound id.
    first = store.get_or_create_session(_make_source(message_id="om_turn1"))
    session_key = first.session_key
    assert first.origin.message_id == "om_turn1"

    # Turn 2: same routing key reuses the session; a NEW inbound id arrives.
    second = store.get_or_create_session(_make_source(message_id="om_turn2"))
    assert second.session_id == first.session_id  # genuine reuse, not reset
    assert second.origin.message_id == "om_turn2"  # refreshed in memory

    # The refreshed anchor must be persisted, so a background delivery loaded
    # from a restarted gateway still replies into the right topic.
    persisted = _sessions_json(tmp_path)[session_key]
    assert persisted["origin"]["message_id"] == "om_turn2"

    # Simulated restart reads the same dir back.
    store2 = store_factory()
    reloaded = store2.get_or_create_session(_make_source(message_id="om_turn3"))
    assert reloaded.session_id == first.session_id
    assert reloaded.origin.message_id == "om_turn3"


def test_reuse_without_message_id_keeps_previous_anchor(store_factory, tmp_path):
    store = store_factory()

    first = store.get_or_create_session(_make_source(message_id="om_turn1"))
    session_key = first.session_key
    assert first.origin.message_id == "om_turn1"

    # A reuse turn with no inbound message_id must NOT clobber the last-known
    # good anchor with None.
    second = store.get_or_create_session(_make_source(message_id=None))
    assert second.session_id == first.session_id
    assert second.origin.message_id == "om_turn1"

    persisted = _sessions_json(tmp_path)[session_key]
    assert persisted["origin"]["message_id"] == "om_turn1"
