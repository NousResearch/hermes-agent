"""Real-DB regression coverage for gateway session recovery reset policy."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import SessionEntry, SessionSource, SessionStore
from hermes_state import SessionDB


def _source(chat_id: str = "contact-1") -> SessionSource:
    return SessionSource(
        platform=Platform.WEIXIN,
        chat_id=chat_id,
        chat_type="dm",
        user_id=chat_id,
    )


def _store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: SessionResetPolicy,
) -> tuple[SessionStore, SessionDB]:
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    store = SessionStore(
        sessions_dir=hermes_home / "sessions",
        config=GatewayConfig(
            default_reset_policy=policy,
            write_sessions_json=False,
        ),
    )
    assert isinstance(store._db, SessionDB)
    return store, store._db


def _seed_gateway_session(
    store: SessionStore,
    db: SessionDB,
    source: SessionSource,
    session_id: str,
    *,
    last_active: datetime,
    end_reason: str | None = "agent_close",
    with_message: bool = True,
) -> str:
    session_key = store._generate_session_key(source)
    db.create_session(
        session_id,
        source.platform.value,
        user_id=source.user_id,
        session_key=session_key,
        chat_id=source.chat_id,
        chat_type=source.chat_type,
        thread_id=source.thread_id,
    )
    if with_message:
        db.append_message(session_id, "user", "old message")
        db._conn.execute(
            "UPDATE messages SET timestamp = ? WHERE session_id = ?",
            (last_active.timestamp(), session_id),
        )
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        ((last_active - timedelta(minutes=2)).timestamp(), session_id),
    )
    db._conn.commit()
    if end_reason:
        db.end_session(session_id, end_reason)
    return session_key


def test_expired_db_recovery_creates_reset_session_with_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, db = _store(
        tmp_path,
        monkeypatch,
        SessionResetPolicy(mode="idle", idle_minutes=60),
    )
    source = _source()
    old_id = "old-expired"
    _seed_gateway_session(
        store,
        db,
        source,
        old_id,
        last_active=datetime.now() - timedelta(days=5),
    )

    recovered = store.get_or_create_session(source)

    assert recovered.session_id != old_id
    assert recovered.was_auto_reset is True
    assert recovered.auto_reset_reason == "idle"
    assert recovered.prev_session_id == old_id
    assert recovered.reset_had_activity is True
    assert db.get_session(old_id)["end_reason"] == "idle"
    assert db.get_session(recovered.session_id)["end_reason"] is None


def test_fresh_db_recovery_preserves_session_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, db = _store(
        tmp_path,
        monkeypatch,
        SessionResetPolicy(mode="idle", idle_minutes=60),
    )
    source = _source()
    old_id = "old-fresh"
    _seed_gateway_session(
        store,
        db,
        source,
        old_id,
        last_active=datetime.now() - timedelta(minutes=10),
    )

    recovered = store.get_or_create_session(source)

    assert recovered.session_id == old_id
    assert recovered.was_auto_reset is False
    assert db.get_session(old_id)["end_reason"] is None


def test_mode_none_recovers_old_session_without_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, db = _store(
        tmp_path,
        monkeypatch,
        SessionResetPolicy(mode="none"),
    )
    source = _source()
    old_id = "old-mode-none"
    _seed_gateway_session(
        store,
        db,
        source,
        old_id,
        last_active=datetime.now() - timedelta(days=30),
    )

    recovered = store.get_or_create_session(source)

    assert recovered.session_id == old_id
    assert recovered.was_auto_reset is False
    assert db.get_session(old_id)["end_reason"] is None


def test_concurrent_explicit_close_is_not_reopened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, db = _store(
        tmp_path,
        monkeypatch,
        SessionResetPolicy(mode="idle", idle_minutes=60),
    )
    source = _source()
    old_id = "closed-during-recovery"
    _seed_gateway_session(
        store,
        db,
        source,
        old_id,
        last_active=datetime.now() - timedelta(minutes=10),
        end_reason=None,
    )
    conditional_reopen = db.reopen_recoverable_session

    def close_then_reopen(session_id: str) -> bool:
        db.end_session(session_id, "session_switch")
        return conditional_reopen(session_id)

    monkeypatch.setattr(
        db,
        "reopen_recoverable_session",
        close_then_reopen,
    )

    recovered = store.get_or_create_session(source)

    assert recovered.session_id != old_id
    assert recovered.was_auto_reset is False
    assert db.get_session(old_id)["end_reason"] == "session_switch"


@pytest.mark.parametrize("end_reason", [None, "agent_close", "ws_orphan_reap"])
def test_conditional_reopen_accepts_only_recoverable_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    end_reason: str | None,
) -> None:
    store, db = _store(
        tmp_path,
        monkeypatch,
        SessionResetPolicy(mode="none"),
    )
    source = _source()
    session_id = f"recoverable-{end_reason or 'live'}"
    _seed_gateway_session(
        store,
        db,
        source,
        session_id,
        last_active=datetime.now(),
        end_reason=end_reason,
    )

    assert db.reopen_recoverable_session(session_id) is True
    assert db.get_session(session_id)["end_reason"] is None


def test_conditional_reopen_preserves_explicit_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, db = _store(
        tmp_path,
        monkeypatch,
        SessionResetPolicy(mode="none"),
    )
    source = _source()
    session_id = "explicit-boundary"
    _seed_gateway_session(
        store,
        db,
        source,
        session_id,
        last_active=datetime.now(),
        end_reason="session_switch",
    )

    assert db.reopen_recoverable_session(session_id) is False
    assert db.get_session(session_id)["end_reason"] == "session_switch"


def test_startup_prune_does_not_repoint_expired_recoverable_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, db = _store(
        tmp_path,
        monkeypatch,
        SessionResetPolicy(mode="idle", idle_minutes=60),
    )
    source = _source()
    old_id = "startup-expired"
    session_key = _seed_gateway_session(
        store,
        db,
        source,
        old_id,
        last_active=datetime.now() - timedelta(days=5),
    )
    old_time = datetime.now() - timedelta(days=5)
    routing_entry = SessionEntry(
        session_key=session_key,
        session_id=old_id,
        created_at=old_time,
        updated_at=old_time,
        origin=source,
        platform=source.platform,
        chat_type=source.chat_type,
    )
    db.save_gateway_routing_entry(
        session_key,
        json.dumps(routing_entry.to_dict()),
        scope=str(store.sessions_dir.resolve()),
    )

    restarted = SessionStore(sessions_dir=store.sessions_dir, config=store.config)
    restarted._ensure_loaded()

    assert session_key not in restarted._entries
    assert restarted._db.get_session(old_id)["end_reason"] == "idle"


def test_recovery_queries_report_real_last_activity_for_both_lookup_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, db = _store(
        tmp_path,
        monkeypatch,
        SessionResetPolicy(mode="none"),
    )
    source = _source()
    last_active = datetime.now() - timedelta(hours=3)
    session_key = _seed_gateway_session(
        store,
        db,
        source,
        "last-active",
        last_active=last_active,
        end_reason=None,
    )

    exact = db.find_latest_gateway_session_for_peer(
        source=source.platform.value,
        user_id=source.user_id,
        session_key=session_key,
        chat_id=source.chat_id,
        chat_type=source.chat_type,
    )
    fallback = db.find_latest_gateway_session_for_peer(
        source=source.platform.value,
        user_id=source.user_id,
        session_key="missing-exact-key",
        chat_id=source.chat_id,
        chat_type=source.chat_type,
    )

    assert exact["last_active"] == pytest.approx(last_active.timestamp())
    assert fallback["last_active"] == pytest.approx(last_active.timestamp())
    assert exact["has_messages"] == 1
    assert fallback["has_messages"] == 1


def test_recovery_last_activity_falls_back_to_started_at(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, db = _store(
        tmp_path,
        monkeypatch,
        SessionResetPolicy(mode="none"),
    )
    source = _source("contact-no-message")
    session_key = _seed_gateway_session(
        store,
        db,
        source,
        "no-message-time",
        last_active=datetime.now() - timedelta(hours=3),
        end_reason=None,
        with_message=False,
    )
    # Legacy/corrupt counters can outlive pruned message rows. The recovery
    # query still admits that row and must provide a usable activity time.
    db._conn.execute(
        "UPDATE sessions SET message_count = 1 WHERE id = ?",
        ("no-message-time",),
    )
    db._conn.commit()

    recovered = db.find_latest_gateway_session_for_peer(
        source=source.platform.value,
        user_id=source.user_id,
        session_key=session_key,
        chat_id=source.chat_id,
        chat_type=source.chat_type,
    )

    assert recovered["last_active"] == pytest.approx(recovered["started_at"])
    assert recovered["has_messages"] == 0
