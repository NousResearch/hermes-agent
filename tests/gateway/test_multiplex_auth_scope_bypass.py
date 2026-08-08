"""Regression: multiplex authorization must not run outside profile secret scope.

Residual P1 bypass after #80026: cold-path message handlers enter
``_profile_runtime_scope``, but busy-session handling, startup-resume owner
validation, and adapter authorization callbacks could authorize against the
process-global allowlist. A foreign profile's env value must not authorize
another profile's busy action or resume a revoked session owner.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import secret_scope as ss
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionEntry, SessionSource


@pytest.fixture(autouse=True)
def _reset_scope_state(monkeypatch):
    for key in (
        "DISCORD_ALLOWED_USERS",
        "DISCORD_ALLOW_ALL_USERS",
        "TELEGRAM_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


def _discord_runner(*, profile="profile-b", allow_from=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    extra = {}
    if allow_from is not None:
        extra["allow_from"] = allow_from
    adapter = SimpleNamespace(
        send=AsyncMock(),
        config=SimpleNamespace(extra=extra),
        enforces_own_access_policy=False,
    )
    runner.adapters = {Platform.DISCORD: adapter}
    runner._profile_adapters = {profile: {Platform.DISCORD: adapter}}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_stores = {}
    return runner, adapter


def _discord_dm(user_id: str, *, profile="profile-b") -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        user_id=user_id,
        chat_id="test-dm",
        user_name=user_id,
        chat_type="dm",
        profile=profile,
    )


def _bind_empty_profile_home(runner, monkeypatch, tmp_path, *, env_body: str = ""):
    """Point scoped auth at an isolated profile home with optional .env body."""
    profile_home = tmp_path / "profile-b"
    profile_home.mkdir(exist_ok=True)
    (profile_home / ".env").write_text(env_body, encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "_resolve_profile_home_for_source",
        lambda source: profile_home,
    )
    return profile_home


class TestIsUserAuthorizedScoped:
    def test_foreign_process_allowlist_rejected_under_empty_scope(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("DISCORD_ALLOWED_USERS", "111")
        runner, _adapter = _discord_runner(allow_from="222")
        _bind_empty_profile_home(runner, monkeypatch, tmp_path)
        ss.set_multiplex_active(True)

        assert runner._is_user_authorized_scoped(_discord_dm("111")) is False
        assert runner._is_user_authorized_scoped(_discord_dm("222")) is True

    def test_scoped_allowlist_authorizes_only_own_profile(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DISCORD_ALLOWED_USERS", "111")
        runner, _adapter = _discord_runner(allow_from=None)
        _bind_empty_profile_home(
            runner,
            monkeypatch,
            tmp_path,
            env_body="DISCORD_ALLOWED_USERS=222\n",
        )
        ss.set_multiplex_active(True)

        assert runner._is_user_authorized_scoped(_discord_dm("111")) is False
        assert runner._is_user_authorized_scoped(_discord_dm("222")) is True

    def test_single_profile_mode_uses_process_environ(self, monkeypatch):
        monkeypatch.setenv("DISCORD_ALLOWED_USERS", "111")
        runner, _adapter = _discord_runner(allow_from="222")
        runner.config = GatewayConfig(multiplex_profiles=False)
        ss.set_multiplex_active(False)
        assert runner._is_user_authorized_scoped(
            _discord_dm("111", profile=None)
        ) is True
        assert runner._is_user_authorized_scoped(
            _discord_dm("222", profile=None)
        ) is False


class TestAdapterAuthCheckScope:
    def test_adapter_callback_rejects_foreign_process_allowlist(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("DISCORD_ALLOWED_USERS", "111")
        runner, _adapter = _discord_runner(allow_from="222", profile="coder")
        _bind_empty_profile_home(runner, monkeypatch, tmp_path)
        ss.set_multiplex_active(True)

        check = runner._make_adapter_auth_check(Platform.DISCORD, profile_name="coder")
        assert check("111", "dm", "test-dm") is False
        assert check("222", "dm", "test-dm") is True

    def test_adapter_callback_stamps_profile_and_uses_scoped_auth(self):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        captured: dict = {}

        def fake_scoped(source):
            captured["profile"] = source.profile
            return source.user_id == "ok"

        runner._is_user_authorized_scoped = fake_scoped
        check = runner._make_adapter_auth_check(Platform.WECOM, profile_name="coder")
        assert check("ok", "dm", "c1") is True
        assert captured["profile"] == "coder"
        assert check("nope", "dm", "c1") is False


class TestBusySessionAuthScope:
    @pytest.mark.asyncio
    async def test_busy_path_rejects_foreign_process_allowlist(
        self, monkeypatch, tmp_path
    ):
        from gateway.platforms.base import MessageEvent, MessageType
        from gateway.run import GatewayRunner

        monkeypatch.setenv("DISCORD_ALLOWED_USERS", "111")
        runner, _adapter = _discord_runner(allow_from="222", profile="profile-b")
        _bind_empty_profile_home(runner, monkeypatch, tmp_path)
        ss.set_multiplex_active(True)

        runner._draining = False
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._busy_ack_ts = {}

        foreign = MessageEvent(
            text="inject",
            message_type=MessageType.TEXT,
            source=_discord_dm("111", profile="profile-b"),
            message_id="m1",
        )
        handled = await GatewayRunner._handle_active_session_busy_message(
            runner, foreign, "agent:main:discord:dm:test-dm"
        )
        assert handled is True  # silently dropped as unauthorized

    @pytest.mark.asyncio
    async def test_profile_busy_handler_stamps_profile(self, monkeypatch, tmp_path):
        from gateway.platforms.base import MessageEvent, MessageType
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        captured: dict = {}

        async def fake_busy(event, session_key):
            captured["profile"] = getattr(event.source, "profile", None)
            return True

        runner._handle_active_session_busy_message = fake_busy
        profile_home = tmp_path / "coder"
        profile_home.mkdir()
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: profile_home if name == "coder" else tmp_path / name,
        )

        handler = GatewayRunner._make_profile_busy_session_handler(runner, "coder")
        source = SessionSource(
            platform=Platform.DISCORD,
            user_id="u1",
            chat_id="c1",
            chat_type="dm",
            profile=None,
        )
        event = MessageEvent(
            text="hi",
            message_type=MessageType.TEXT,
            source=source,
            message_id="1",
        )
        assert await handler(event, "sk") is True
        assert captured["profile"] == "coder"
        assert source.profile == "coder"


class TestStartupResumeAuthScope:
    def test_resume_skips_owner_authorized_only_by_foreign_allowlist(
        self, monkeypatch, tmp_path
    ):
        """Revoked owner: foreign process allowlist must not resume the session."""
        monkeypatch.setenv("DISCORD_ALLOWED_USERS", "revoked-user")
        runner, adapter = _discord_runner(allow_from=None, profile="profile-b")
        _bind_empty_profile_home(runner, monkeypatch, tmp_path)
        ss.set_multiplex_active(True)

        runner._running_agents = {}
        runner._running_agents_ts = {}
        runner._background_tasks = set()
        runner._persist_active_agents = MagicMock()
        runner._AUTO_RESUME_REASONS = frozenset(
            {"restart_timeout", "shutdown_timeout", "restart_interrupted"}
        )
        runner._is_session_running = lambda _key: False
        runner._session_state = lambda key: SimpleNamespace(
            turn=SimpleNamespace(agent=None, started_ts=None)
        )
        runner._adapter_for_source = lambda source: adapter
        adapter.handle_message = AsyncMock()

        entry = SessionEntry(
            session_key="agent:main:discord:dm:test-dm",
            session_id="sid",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            origin=_discord_dm("revoked-user", profile="profile-b"),
            platform=Platform.DISCORD,
            chat_type="dm",
            resume_pending=True,
            resume_reason="restart_timeout",
            last_resume_marked_at=datetime.now(),
        )
        store = MagicMock()
        store._lock = MagicMock()
        store._lock.__enter__ = MagicMock(return_value=None)
        store._lock.__exit__ = MagicMock(return_value=False)
        store._ensure_loaded_locked = MagicMock()
        store._entries = {entry.session_key: entry}
        runner.session_store = store

        monkeypatch.setattr(
            "gateway.restart_loop_guard.check_and_record",
            lambda *_a, **_k: False,
        )
        runner._restart_loop_guard_config = lambda: (5, 600)

        from gateway.run import GatewayRunner

        scheduled = GatewayRunner._schedule_resume_pending_sessions(runner)
        assert scheduled == 0
        adapter.handle_message.assert_not_called()
        runner._persist_active_agents.assert_not_called()

    def test_resume_allows_owner_on_profile_scoped_allowlist(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("DISCORD_ALLOWED_USERS", "foreign-only")
        runner, _adapter = _discord_runner(allow_from=None, profile="profile-b")
        _bind_empty_profile_home(
            runner,
            monkeypatch,
            tmp_path,
            env_body="DISCORD_ALLOWED_USERS=owner-user\n",
        )
        ss.set_multiplex_active(True)

        source = _discord_dm("owner-user", profile="profile-b")
        assert runner._is_user_authorized_scoped(source) is True
        assert runner._is_user_authorized_scoped(
            _discord_dm("foreign-only", profile="profile-b")
        ) is False
