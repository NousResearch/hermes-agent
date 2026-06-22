"""HRM-T0a step 8 — legacy mode + missing-app_id fail-closed + inventory.

Covers:

- Pre-HRM (started_at < hrm_t0a_applied_at) sessions and Telegram-DM
  forum-thread sessions are detected as legacy and their state is NOT
  silently migrated by the routing pre-pass nor by the slash mutators.
- ``/topic`` state-mutating subcommands (switch / new / clear / bind-
  thread / unbind-thread / move-last / move-range) are refused under
  legacy mode with the standard ``gateway.topic.legacy_thread_readonly``
  banner. Read-only subcommands (``/topic``, ``/topic list``,
  ``/topic help``) continue to work.
- Normal legacy routing keeps using the legacy thread-derived
  session key (no ``:topic:`` segment); the pointer pre-pass remains a
  no-op for legacy principals because the pointer table has no row.
- Missing ``topic_default_app_id`` causes pointer-mutating subcommands
  to fail closed with a setup-required message rather than silently
  writing under a bogus app_id.
- :func:`legacy_inventory` returns counts/principal/chat/thread/session
  ids only — never any message content, prompt, response, or title.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

import gateway.active_topic as active_topic_module
from gateway.active_topic import (
    LEGACY_READONLY_MESSAGE,
    is_legacy_principal_route,
    legacy_inventory,
    set_registered_check,
    _reset_legacy_banner_for_tests,
)
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore, build_session_key
from gateway.slash_commands import GatewaySlashCommandsMixin
from hermes_state import SessionDB


# ── Fixtures / helpers ────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_module_state():
    active_topic_module._reset_locks_for_tests()
    set_registered_check(None)
    _reset_legacy_banner_for_tests()
    yield
    active_topic_module._reset_locks_for_tests()
    set_registered_check(None)
    _reset_legacy_banner_for_tests()


def _source(**overrides) -> SessionSource:
    defaults = dict(
        platform=Platform.TELEGRAM,
        chat_id="208214988",
        user_id="208214988",
        chat_type="dm",
        thread_id="42",
    )
    defaults.update(overrides)
    return SessionSource(**defaults)


def _ok_checker():
    async def _check(app_id, topic_id):
        return True
    return _check


@dataclass
class FakeAdapter:
    sent: list = None
    raise_on_send: Exception | None = None

    def __post_init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        if self.raise_on_send is not None:
            raise self.raise_on_send
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata})


class FakeRunner(GatewaySlashCommandsMixin):
    def __init__(self, *, config, session_db, adapter):
        self.config = config
        self._session_db = session_db
        self.adapters = {Platform.TELEGRAM: adapter}
        self._is_user_authorized = lambda src: True


def _config(*, slash_ux_enabled=True, app_id="hermes-agent") -> GatewayConfig:
    return GatewayConfig(
        topic_pointer_mode_enabled=True,
        topic_default_app_id=app_id,
        topic_slash_ux_enabled=slash_ux_enabled,
    )


def _make_event(text: str, *, source=None):
    from gateway.platforms.base import MessageEvent, MessageType
    src = source or _source()
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=src,
    )


def _seed_pre_migration_session(db: SessionDB, *, platform="telegram", user_id="208214988"):
    """Create a session whose started_at is BEFORE the migration marker.

    Then apply the migration so ``hrm_t0a_applied_at`` is set above
    ``started_at`` (matching the production migration high-water-mark).
    """
    pre_ts = 1_000.0
    sid = f"pre-{user_id}"
    db.create_session(session_id=sid, source=platform, user_id=str(user_id))
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        (pre_ts, sid),
    )
    db._conn.commit()
    db.apply_hrm_t0a_migration()
    return sid


# ── Detection ──────────────────────────────────────────────────────────


def test_is_legacy_principal_route_returns_false_before_migration(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        is_legacy, reason = is_legacy_principal_route(
            db, _source(), app_id="hermes-agent"
        )
        # No migration ran ⇒ no marker ⇒ we don't classify by marker.
        assert is_legacy is False
        assert reason == ""
    finally:
        db.close()


def test_is_legacy_principal_route_flags_pre_migration_session(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _seed_pre_migration_session(db)
        is_legacy, reason = is_legacy_principal_route(
            db, _source(), app_id="hermes-agent"
        )
        assert is_legacy is True
        assert reason == "pre_migration_session"
    finally:
        db.close()


def test_is_legacy_principal_route_flags_telegram_forum_thread(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.apply_telegram_topic_migration()
        db.enable_telegram_topic_mode(
            chat_id="208214988",
            user_id="208214988",
            has_topics_enabled=True,
            allows_users_to_create_topics=True,
        )
        is_legacy, reason = is_legacy_principal_route(
            db, _source(), app_id="hermes-agent"
        )
        assert is_legacy is True
        assert reason == "telegram_forum_thread"
    finally:
        db.close()


def test_existing_pointer_overrides_pre_migration_legacy(tmp_path):
    """Once a pointer is written for this principal, the user has opted into
    pointer mode and the pre-migration sentinel must not lock them out.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _seed_pre_migration_session(db)
        db.set_active_topic(
            platform="telegram",
            user_id="208214988",
            chat_id="208214988",
            app_id="hermes-agent",
            topic_id="lane-a",
            updated_by="slash:test",
        )
        is_legacy, reason = is_legacy_principal_route(
            db, _source(), app_id="hermes-agent"
        )
        assert is_legacy is False
        assert reason == ""
    finally:
        db.close()


# ── No-auto-migration: routing pre-pass + slash dispatch ───────────────


def test_legacy_session_route_uses_legacy_key_unchanged(tmp_path):
    """Routing pre-pass must NOT silently mint a topic-routed key for a
    legacy principal — the legacy thread-derived ``build_session_key``
    path stays in play.
    """
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        _seed_pre_migration_session(db)
    finally:
        db.close()

    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    store = SessionStore(cfg.sessions_dir, cfg)
    store._db = SessionDB(db_path=db_path)
    set_registered_check(_ok_checker())
    try:
        key = store._generate_session_key(_source())
        # No pointer row → pre-pass returns None → legacy key.
        assert "topic:" not in key
        assert key == build_session_key(_source())
    finally:
        store._db.close()


def test_legacy_slash_switch_refused_with_banner(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_pre_migration_session(db)

    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    adapter = FakeAdapter()
    runner = FakeRunner(config=cfg, session_db=db, adapter=adapter)
    set_registered_check(_ok_checker())

    ev = _make_event("/topic switch research")
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert resp == LEGACY_READONLY_MESSAGE
        # No banner sent because the mutator was refused before banner emit.
        assert adapter.sent == []
        # Pointer table untouched.
        assert (
            db.read_active_topic(
                platform="telegram",
                user_id="208214988",
                chat_id="208214988",
                app_id="hermes-agent",
            )
            is None
        )
    finally:
        db.close()


@pytest.mark.parametrize(
    "subcommand",
    [
        "switch foo",
        "new foo",
        "clear",
        "bind-thread",
        "unbind-thread",
        "move-last 1 --to bar --dry-run",
        "move-range 1..2 --to bar --dry-run",
    ],
)
def test_legacy_refuses_all_mutator_subcommands(tmp_path, subcommand):
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_pre_migration_session(db)
    set_registered_check(_ok_checker())

    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    adapter = FakeAdapter()
    runner = FakeRunner(config=cfg, session_db=db, adapter=adapter)

    ev = _make_event(f"/topic {subcommand}")
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert resp == LEGACY_READONLY_MESSAGE, (
            f"subcommand {subcommand!r} must be refused under legacy mode"
        )
        assert adapter.sent == []
    finally:
        db.close()


def test_legacy_readonly_subcommands_still_work(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_pre_migration_session(db)
    set_registered_check(_ok_checker())

    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter())

    try:
        # /topic (no args) — show
        resp = asyncio.run(runner._handle_topic_command(_make_event("/topic")))
        assert "no active topic" in resp.lower()
        # /topic list
        resp = asyncio.run(runner._handle_topic_command(_make_event("/topic list")))
        assert "no active topic" in resp.lower()
        # /topic help
        resp = asyncio.run(runner._handle_topic_command(_make_event("/topic help")))
        assert resp.startswith("/topic ")
        assert "active-topic-pointer" in resp
    finally:
        db.close()


def test_legacy_refusal_logs_only_once(tmp_path, caplog):
    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_pre_migration_session(db)
    set_registered_check(_ok_checker())

    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter())

    caplog.set_level("INFO", logger="gateway.active_topic")
    try:
        ev = _make_event("/topic switch foo")
        asyncio.run(runner._handle_topic_command(ev))
        asyncio.run(runner._handle_topic_command(ev))
        asyncio.run(runner._handle_topic_command(ev))
    finally:
        db.close()
    banner_records = [
        r for r in caplog.records if "legacy mode engaged" in r.getMessage()
    ]
    assert len(banner_records) == 1, (
        f"expected exactly one legacy banner log, got {len(banner_records)}"
    )


# ── Missing topic_default_app_id fail-closed ───────────────────────────


def test_mutator_with_no_default_app_id_returns_setup_required(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    set_registered_check(_ok_checker())
    cfg = _config(app_id=None)
    cfg.sessions_dir = tmp_path / "sessions"
    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter())
    try:
        resp = asyncio.run(
            runner._handle_topic_command(_make_event("/topic switch research"))
        )
        assert "setup required" in resp.lower()
        assert "topic.default_app_id" in resp
        # No write occurred.
        with db._lock:
            tables = {
                r[0]
                for r in db._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        # active_topic_pointer is only created on first explicit write —
        # the fail-closed mutator must NOT have triggered the migration.
        assert "active_topic_pointer" not in tables
    finally:
        db.close()


@pytest.mark.parametrize(
    "subcommand",
    [
        "switch foo",
        "new foo",
        "clear",
        "bind-thread",
        "unbind-thread",
        "move-last 1 --to bar --dry-run",
        "move-range 1..2 --to bar --dry-run",
    ],
)
def test_all_mutators_require_topic_default_app_id(tmp_path, subcommand):
    db = SessionDB(db_path=tmp_path / "state.db")
    set_registered_check(_ok_checker())
    cfg = _config(app_id=None)
    cfg.sessions_dir = tmp_path / "sessions"
    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter())
    try:
        resp = asyncio.run(
            runner._handle_topic_command(_make_event(f"/topic {subcommand}"))
        )
        assert "setup required" in resp.lower(), (
            f"subcommand {subcommand!r} must surface setup-required"
        )
    finally:
        db.close()


def test_readonly_subcommands_without_app_id_still_respond(tmp_path):
    """Show/list/help do not require app_id because no write happens."""
    db = SessionDB(db_path=tmp_path / "state.db")
    set_registered_check(_ok_checker())
    cfg = _config(app_id=None)
    cfg.sessions_dir = tmp_path / "sessions"
    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter())
    try:
        resp = asyncio.run(runner._handle_topic_command(_make_event("/topic help")))
        assert resp.startswith("/topic ")
        resp = asyncio.run(runner._handle_topic_command(_make_event("/topic")))
        assert isinstance(resp, str)
    finally:
        db.close()


# ── Metadata-only legacy inventory ─────────────────────────────────────


def test_legacy_inventory_excludes_message_content(tmp_path):
    """Inventory must NEVER include message text — only structural ids/counts."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        sid = _seed_pre_migration_session(db)
        # Append distinctive messages so the test can prove the inventory
        # does not surface them.
        secret = "TOPSECRETSENTINEL-987654321"
        db.append_message(sid, "user", secret)
        db.append_message(sid, "assistant", f"reply about {secret}")
        # Set a title to make sure titles also aren't leaked.
        try:
            db.set_session_title(sid, "do-not-leak-this-title")
        except Exception:
            pass
        # Forum-thread legacy state.
        db.apply_telegram_topic_migration()
        db.enable_telegram_topic_mode(
            chat_id="208214988",
            user_id="208214988",
            has_topics_enabled=True,
            allows_users_to_create_topics=True,
        )

        inv = legacy_inventory(db)
        # Counts present, principals/chats listed.
        assert inv["pre_migration_session_count"] >= 1
        assert inv["telegram_forum_thread_chat_count"] >= 1
        assert any(
            p["platform"] == "telegram" and p["user_id"] == "208214988"
            for p in inv["pre_migration_principals"]
        )
        assert any(c["chat_id"] == "208214988" for c in inv["telegram_forum_thread_chats"])

        # No message content leakage — recursive scan of the entire dict.
        import json
        blob = json.dumps(inv, default=str)
        assert secret not in blob
        assert "reply about" not in blob
        assert "do-not-leak-this-title" not in blob
    finally:
        db.close()


def test_legacy_inventory_safe_on_empty_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        inv = legacy_inventory(db)
        assert inv["pre_migration_session_count"] == 0
        assert inv["telegram_forum_thread_chat_count"] == 0
        assert inv["pre_migration_principals"] == []
        assert inv["telegram_forum_thread_chats"] == []
        assert inv["telegram_forum_thread_bindings"] == []
    finally:
        db.close()


def test_legacy_inventory_with_none_db_returns_zero_shape():
    inv = legacy_inventory(None)
    assert inv["pre_migration_session_count"] == 0
    assert inv["telegram_forum_thread_chat_count"] == 0
    assert inv["pre_migration_principals"] == []
    assert inv["telegram_forum_thread_chats"] == []
    assert inv["telegram_forum_thread_bindings"] == []
