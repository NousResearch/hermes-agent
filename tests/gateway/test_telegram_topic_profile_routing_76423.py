"""Issue #76423 — Gateway routes source.profile into telegram topic state."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from hermes_state import SessionDB
from gateway.config import Platform
from gateway.session import SessionSource


CHAT = "208214988"


def _source(profile=None, thread_id="42"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id=CHAT,
        chat_id=CHAT,
        user_name="tester",
        chat_type="dm",
        thread_id=thread_id,
        profile=profile,
    )


def test_gateway_uses_source_profile_not_global(tmp_path: Path):
    from gateway.run import GatewayRunner

    assert GatewayRunner._telegram_topic_profile_name(_source("coder")) == "coder"
    assert GatewayRunner._telegram_topic_profile_name(_source(None)) == "default"

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="sess-coder", source="telegram", user_id=CHAT, profile_name="coder")
    db.enable_telegram_topic_mode(chat_id=CHAT, user_id=CHAT, profile_name="coder")

    runner = object.__new__(GatewayRunner)
    runner._session_db = db
    assert runner._telegram_topic_mode_enabled(_source("coder")) is True
    assert runner._telegram_topic_mode_enabled(_source("other")) is False
    assert runner._telegram_topic_mode_enabled(_source(None)) is False

    runner._record_telegram_topic_binding(
        _source("coder", "42"),
        SimpleNamespace(session_key="k", session_id="sess-coder"),
    )
    assert db.get_telegram_topic_binding(
        chat_id=CHAT, thread_id="42", profile_name="coder",
    ) is not None
    assert db.get_telegram_topic_binding(
        chat_id=CHAT, thread_id="42", profile_name="default",
    ) is None
    db.close()


def test_thread_metadata_carries_routed_profile():
    """Outbound send metadata must include hermes_profile for prune (#76423)."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    # Avoid full constructor; stub the target builder to a simple dict.
    runner._thread_metadata_for_target = lambda *a, **k: {"thread_id": "42"}
    meta = runner._thread_metadata_for_source(_source("coder", "42"))
    assert meta is not None
    assert meta["hermes_profile"] == "coder"


def test_cooldowns_namespaced_by_profile():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    a = _source("alpha")
    b = _source("beta")
    # First send for each profile should be allowed independently.
    assert runner._should_send_telegram_lobby_reminder(a) is True
    assert runner._should_send_telegram_lobby_reminder(b) is True
    # Immediate re-hit same profile is suppressed; the other profile is not.
    assert runner._should_send_telegram_lobby_reminder(a) is False
    assert runner._should_send_telegram_capability_hint(a) is True
    assert runner._should_send_telegram_capability_hint(b) is True
    assert runner._should_send_telegram_capability_hint(a) is False


def test_primary_adapter_prunes_routed_profile_not_stamp(tmp_path: Path):
    """profile_routes: transport adapter may be primary (default) while the
    turn is routed to another profile — prune must use send metadata."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="sess-default", source="telegram", user_id=CHAT)
    db.create_session(session_id="sess-coder", source="telegram", user_id=CHAT, profile_name="coder")
    db.bind_telegram_topic(
        chat_id=CHAT, thread_id="99", user_id=CHAT,
        session_key="kd", session_id="sess-default", profile_name="default",
    )
    db.bind_telegram_topic(
        chat_id=CHAT, thread_id="99", user_id=CHAT,
        session_key="kc", session_id="sess-coder", profile_name="coder",
    )

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter._session_store = SimpleNamespace(_db=db)
    # Transport is the primary/default adapter stamp...
    adapter._hermes_profile_name = "default"
    # ...but this send is for the routed coder profile.
    adapter._prune_stale_dm_topic_binding(
        CHAT, "99", metadata={"hermes_profile": "coder"},
    )

    assert db.get_telegram_topic_binding(
        chat_id=CHAT, thread_id="99", profile_name="coder",
    ) is None
    assert db.get_telegram_topic_binding(
        chat_id=CHAT, thread_id="99", profile_name="default",
    ) is not None
    db.close()
