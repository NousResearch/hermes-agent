"""HRM-T0a step 5 — ``/topic`` slash UX + precondition (running-loop wiring).

Covers:

- **Precondition** (route live async inbound through
  ``resolve_topic_session_key_async``): E2E running-loop routing — a wired
  async registry checker and active pointer make the resolved key contain
  ``topic:`` even inside an event loop (the sync resolver short-circuits
  here; this test proves the async path is wired).
- **Precondition** — concurrent slash-write vs inbound-resolve: the per-key
  asyncio lock serialises a switch task and a resolver task so no
  half-applied route leaks.
- **Step 5** — slash UX switch happy path emits banner, returns "".
- **Step 5** — banner-emit failure rolls the pointer back AND surfaces a
  visible error.
- **Step 5** — slash UX is default-deny (legacy handler runs unless
  ``topic_slash_ux_enabled`` is on).
- **Step 5** — list / clear / bind-thread / unbind-thread basic flows.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.active_topic as active_topic_module
from gateway.active_topic import (
    PlatformPrincipal,
    read_active_topic,
    set_active_topic,
    set_registered_check,
)
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from gateway.slash_commands import GatewaySlashCommandsMixin
from hermes_state import SessionDB


# ── Fixtures / helpers ────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_module_state():
    active_topic_module._reset_locks_for_tests()
    set_registered_check(None)
    yield
    active_topic_module._reset_locks_for_tests()
    set_registered_check(None)


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
    """Adapter stub the slash handler can ``await adapter.send(...)`` on."""
    sent: list = None
    raise_on_send: Exception | None = None

    def __post_init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        if self.raise_on_send is not None:
            raise self.raise_on_send
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata})


class FakeRunner(GatewaySlashCommandsMixin):
    """Just-enough surface to drive the slash mixin in tests.

    The mixin reads ``self.config``, ``self._session_db``, ``self.adapters``,
    and ``self._is_user_authorized``. Nothing else is touched in the
    pointer-subcommand path.
    """

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
    """Build a minimal MessageEvent-shaped object for the slash mixin."""
    from gateway.platforms.base import MessageEvent, MessageType

    src = source or _source()
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=src,
    )


# ── Precondition: E2E running-loop routing ────────────────────────────


def test_e2e_running_loop_async_resolver_returns_topic_key(tmp_path):
    """With a wired async registry checker and active pointer, the live
    async path produces a topic-routed key inside a running loop — the
    sync resolver would fail-closed here (its ``asyncio.run`` raises
    ``RuntimeError`` inside a loop) so this proves the async wiring.
    """
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.set_active_topic(
        platform="telegram",
        user_id="208214988",
        chat_id="208214988",
        app_id="hermes-agent",
        topic_id="research",
        updated_by="x",
    )
    db.close()

    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    store = SessionStore(cfg.sessions_dir, cfg)
    store._db = SessionDB(db_path=db_path)
    set_registered_check(_ok_checker())

    async def run():
        return await store._generate_session_key_async(_source())

    try:
        key = asyncio.run(run())
        assert "topic:" in key
        assert key.endswith(":research"), key
    finally:
        store._db.close()


def test_get_or_create_session_async_routes_under_running_loop(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.set_active_topic(
        platform="telegram",
        user_id="208214988",
        chat_id="208214988",
        app_id="hermes-agent",
        topic_id="lane",
        updated_by="x",
    )
    db.close()

    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    store = SessionStore(cfg.sessions_dir, cfg)
    store._db = SessionDB(db_path=db_path)
    set_registered_check(_ok_checker())

    async def run():
        entry = await store.get_or_create_session_async(_source())
        return entry.session_key

    try:
        key = asyncio.run(run())
        assert key.endswith(":lane"), key
    finally:
        store._db.close()


# ── Precondition: concurrent slash-write vs inbound-resolve ───────────


def test_concurrent_switch_and_resolve_serialise_under_per_key_lock(tmp_path):
    """Switch + resolve race on the same principal: the per-key asyncio
    lock makes the resolver observe a fully-applied write — never a
    partial/half-decision route. The resolver returns either the pre-
    switch topic or the post-switch topic, never a stale composition.
    """
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.set_active_topic(
        platform="telegram",
        user_id="208214988",
        chat_id="208214988",
        app_id="hermes-agent",
        topic_id="before",
        updated_by="x",
    )
    db.close()

    set_registered_check(_ok_checker())

    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    store = SessionStore(cfg.sessions_dir, cfg)
    store._db = SessionDB(db_path=db_path)
    principal = PlatformPrincipal.from_source(_source(), app_id="hermes-agent")

    async def race():
        switch_task = asyncio.create_task(
            set_active_topic(
                store._db,
                principal,
                topic_id="after",
                updated_by="slash:test",
            )
        )
        # Spin up many resolvers so they have to all queue on the same lock.
        resolve_tasks = [
            asyncio.create_task(store._generate_session_key_async(_source()))
            for _ in range(20)
        ]
        await switch_task
        keys = await asyncio.gather(*resolve_tasks)
        return keys

    try:
        keys = asyncio.run(race())
        # Every resolver saw an applied snapshot — only the two valid
        # topics ever show up; no "topic:" key without a known slug.
        suffixes = {k.rsplit(":", 1)[-1] for k in keys}
        assert suffixes.issubset({"before", "after"}), suffixes
        # And the lock is released after every resolver returns — no
        # lock leak hanging around past the critical section.
        assert not active_topic_module._LOCKS.get(principal.key, (None,))[
            0
        ].locked()
    finally:
        store._db.close()


# ── Slash UX: default-deny ────────────────────────────────────────────


def test_topic_slash_ux_default_deny_routes_to_legacy(tmp_path):
    """When ``topic_slash_ux_enabled`` is False (default), the legacy
    Telegram-DM forum-thread handler runs — the new pointer dispatcher
    is NOT engaged. We assert by sending /topic on a *non*-Telegram
    source: the legacy path returns the "not Telegram DM" error.
    """
    cfg = _config(slash_ux_enabled=False)
    cfg.sessions_dir = tmp_path / "sessions"
    db = SessionDB(db_path=tmp_path / "state.db")
    adapter = FakeAdapter()
    runner = FakeRunner(config=cfg, session_db=db, adapter=adapter)

    discord_src = SessionSource(
        platform=Platform.DISCORD, chat_id="c", user_id="u", chat_type="dm"
    )
    runner.adapters = {Platform.DISCORD: adapter}

    ev = _make_event("/topic switch research", source=discord_src)
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        # Legacy handler refuses non-Telegram-DM /topic.
        assert "Telegram" in resp or "telegram" in resp
    finally:
        db.close()


# ── Slash UX: switch happy path emits banner ──────────────────────────


def test_topic_switch_emits_banner_and_writes_pointer(tmp_path):
    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    db = SessionDB(db_path=tmp_path / "state.db")
    set_registered_check(_ok_checker())
    adapter = FakeAdapter()
    runner = FakeRunner(config=cfg, session_db=db, adapter=adapter)

    ev = _make_event("/topic switch research")
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert resp == ""  # banner already sent
        assert adapter.sent and adapter.sent[0]["text"] == "[topic → research]"
        row = db.read_active_topic(
            platform="telegram",
            user_id="208214988",
            chat_id="208214988",
            app_id="hermes-agent",
        )
        assert row and row["topic_id"] == "research"
    finally:
        db.close()


def test_topic_switch_rejects_invalid_slug(tmp_path):
    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    db = SessionDB(db_path=tmp_path / "state.db")
    set_registered_check(_ok_checker())
    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter())

    ev = _make_event("/topic switch bad!slug")
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert "not a valid topic slug" in resp
    finally:
        db.close()


def test_topic_switch_refuses_when_topic_not_registered(tmp_path):
    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    db = SessionDB(db_path=tmp_path / "state.db")

    async def reject_check(app_id, topic_id):
        return False
    set_registered_check(reject_check)
    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter())

    ev = _make_event("/topic switch unknown")
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert "not registered" in resp
        # No pointer was written.
        row = db.read_active_topic(
            platform="telegram",
            user_id="208214988",
            chat_id="208214988",
            app_id="hermes-agent",
        )
        assert row is None
    finally:
        db.close()


# ── Slash UX: banner-emit failure rolls the pointer back ──────────────


def test_topic_switch_banner_failure_rolls_back_pointer(tmp_path):
    """Owner confirmation contract: if the banner can't be emitted, the
    pointer write must NOT silently hold — it rolls back to the prior
    value (or clear when none) and a visible error is returned.
    """
    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    db = SessionDB(db_path=tmp_path / "state.db")
    set_registered_check(_ok_checker())
    # Adapter that ALWAYS raises on send → banner emission fails.
    adapter = FakeAdapter(raise_on_send=RuntimeError("transport down"))
    runner = FakeRunner(config=cfg, session_db=db, adapter=adapter)

    ev = _make_event("/topic switch research")
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert "banner emit error" in resp
        assert "rolled back" in resp
        # Pointer is back to absent (no prior).
        row = db.read_active_topic(
            platform="telegram",
            user_id="208214988",
            chat_id="208214988",
            app_id="hermes-agent",
        )
        assert row is None, f"expected rollback to absent, got: {row!r}"
    finally:
        db.close()


def test_topic_switch_banner_failure_restores_prior_pointer(tmp_path):
    """When a switch banner fails AND a prior pointer existed, the
    rollback restores the *prior* topic, not just clears.
    """
    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    db = SessionDB(db_path=tmp_path / "state.db")
    set_registered_check(_ok_checker())

    # Seed an existing pointer at "before".
    db.set_active_topic(
        platform="telegram",
        user_id="208214988",
        chat_id="208214988",
        app_id="hermes-agent",
        topic_id="before",
        updated_by="seed",
    )

    adapter = FakeAdapter(raise_on_send=RuntimeError("transport down"))
    runner = FakeRunner(config=cfg, session_db=db, adapter=adapter)

    ev = _make_event("/topic switch after")
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert "rolled back" in resp
        row = db.read_active_topic(
            platform="telegram",
            user_id="208214988",
            chat_id="208214988",
            app_id="hermes-agent",
        )
        assert row and row["topic_id"] == "before"
    finally:
        db.close()


# ── Slash UX: list / clear / bind-thread / unbind-thread ──────────────


def test_topic_list_shows_active_pointer(tmp_path):
    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    db = SessionDB(db_path=tmp_path / "state.db")
    db.set_active_topic(
        platform="telegram",
        user_id="208214988",
        chat_id="208214988",
        app_id="hermes-agent",
        topic_id="research",
        updated_by="x",
    )
    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter())

    ev = _make_event("/topic list")
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert "[topic] active: research" == resp
    finally:
        db.close()


def test_topic_clear_removes_pointer_and_emits_banner(tmp_path):
    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    db = SessionDB(db_path=tmp_path / "state.db")
    db.set_active_topic(
        platform="telegram",
        user_id="208214988",
        chat_id="208214988",
        app_id="hermes-agent",
        topic_id="research",
        updated_by="x",
    )
    adapter = FakeAdapter()
    runner = FakeRunner(config=cfg, session_db=db, adapter=adapter)

    ev = _make_event("/topic clear")
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert resp == ""
        assert adapter.sent and adapter.sent[0]["text"] == "[topic cleared]"
        row = db.read_active_topic(
            platform="telegram",
            user_id="208214988",
            chat_id="208214988",
            app_id="hermes-agent",
        )
        assert row is None
    finally:
        db.close()


def test_topic_clear_banner_failure_rolls_back_pointer(tmp_path):
    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    db = SessionDB(db_path=tmp_path / "state.db")
    db.set_active_topic(
        platform="telegram",
        user_id="208214988",
        chat_id="208214988",
        app_id="hermes-agent",
        topic_id="research",
        updated_by="x",
    )
    adapter = FakeAdapter(raise_on_send=RuntimeError("nope"))
    runner = FakeRunner(config=cfg, session_db=db, adapter=adapter)

    ev = _make_event("/topic clear")
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert "rolled back" in resp
        row = db.read_active_topic(
            platform="telegram",
            user_id="208214988",
            chat_id="208214988",
            app_id="hermes-agent",
        )
        assert row and row["topic_id"] == "research"
    finally:
        db.close()


def test_topic_bind_thread_records_thread_id(tmp_path):
    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    db = SessionDB(db_path=tmp_path / "state.db")
    db.set_active_topic(
        platform="telegram",
        user_id="208214988",
        chat_id="208214988",
        app_id="hermes-agent",
        topic_id="research",
        updated_by="x",
    )
    adapter = FakeAdapter()
    runner = FakeRunner(config=cfg, session_db=db, adapter=adapter)

    ev = _make_event("/topic bind-thread")  # source.thread_id = "42"
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert resp == ""
        assert "thread 42" in adapter.sent[0]["text"]
        row = db.read_active_topic(
            platform="telegram",
            user_id="208214988",
            chat_id="208214988",
            app_id="hermes-agent",
        )
        assert row and row["bound_thread_id"] == "42"
    finally:
        db.close()


def test_topic_unbind_thread_clears_binding(tmp_path):
    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    db = SessionDB(db_path=tmp_path / "state.db")
    db.set_active_topic(
        platform="telegram",
        user_id="208214988",
        chat_id="208214988",
        app_id="hermes-agent",
        topic_id="research",
        bound_thread_id="42",
        updated_by="x",
    )
    adapter = FakeAdapter()
    runner = FakeRunner(config=cfg, session_db=db, adapter=adapter)

    ev = _make_event("/topic unbind-thread")
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert resp == ""
        assert "thread unbound" in adapter.sent[0]["text"]
        row = db.read_active_topic(
            platform="telegram",
            user_id="208214988",
            chat_id="208214988",
            app_id="hermes-agent",
        )
        assert row and row["bound_thread_id"] is None
    finally:
        db.close()


def test_topic_no_args_shows_no_active(tmp_path):
    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    db = SessionDB(db_path=tmp_path / "state.db")
    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter())

    ev = _make_event("/topic")
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert "no active topic" in resp
    finally:
        db.close()


def test_topic_help_text(tmp_path):
    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    db = SessionDB(db_path=tmp_path / "state.db")
    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter())

    ev = _make_event("/topic help")
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert "/topic" in resp and "switch" in resp and "bind-thread" in resp
    finally:
        db.close()


def test_topic_unknown_subcommand(tmp_path):
    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    db = SessionDB(db_path=tmp_path / "state.db")
    runner = FakeRunner(config=cfg, session_db=db, adapter=FakeAdapter())

    ev = _make_event("/topic frobnicate")
    try:
        resp = asyncio.run(runner._handle_topic_command(ev))
        assert "unknown subcommand" in resp
    finally:
        db.close()
