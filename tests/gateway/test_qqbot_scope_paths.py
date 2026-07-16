"""End-to-end profile-scope coverage for the QQ (qqbot) authorization,
startup-validation, and direct-send paths.

Complements ``test_qqbot_credential_isolation.py`` (adapter-level resolver):
the adapter intake fix alone is not enough because three other paths read the
same per-profile ``QQ_*`` values independently:

- gateway authorization (``AuthorizationMixin._is_user_authorized``) reads
  ``QQ_ALLOW_ALL_USERS`` when deciding whether to honor an allow-all opt-in;
- secondary-profile startup validation
  (``gateway.run._own_policy_open_startup_violation``) reads the platform
  opt-in while running inside ``_profile_runtime_scope``;
- the ``send_message`` tool's direct REST path (``_send_qqbot``) falls back to
  ``QQ_APP_ID`` / ``QQ_CLIENT_SECRET``.

Each must resolve through the active profile secret scope (scope wins over
``os.environ``; a profile that did NOT opt in must not inherit the primary
profile's environ opt-in) while unscoped single-profile deployments keep the
legacy ``os.environ`` behavior.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import secret_scope as ss
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource


@pytest.fixture(autouse=True)
def _reset_scope_state(monkeypatch):
    for key in (
        "QQ_ALLOW_ALL_USERS",
        "QQ_ALLOWED_USERS",
        "QQ_GROUP_ALLOWED_USERS",
        "QQ_APP_ID",
        "QQ_CLIENT_SECRET",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


def _make_qq_runner():
    """Minimal runner whose authz path reaches the QQ env checks."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)

    default_adapter = SimpleNamespace(
        send=AsyncMock(),
        enforces_own_access_policy=True,
        _dm_policy="allowlist",
        _group_policy="pairing",
    )
    secondary_adapter = SimpleNamespace(
        send=AsyncMock(),
        enforces_own_access_policy=True,
        _dm_policy="open",
        _group_policy="open",
    )
    runner.adapters = {Platform.QQBOT: default_adapter}
    runner._profile_adapters = {"coder": {Platform.QQBOT: secondary_adapter}}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    return runner


def _qq_dm_source(profile="coder"):
    return SessionSource(
        platform=Platform.QQBOT,
        user_id="user-1",
        chat_id="dm-chat",
        user_name="user-1",
        chat_type="dm",
        profile=profile,
    )


class TestAuthzAllowAllScope:
    def test_scoped_allow_all_honored(self):
        # The secondary profile opted in via its own .env (scope); os.environ
        # has no opt-in. Authorization must honor the scoped value.
        runner = _make_qq_runner()
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"QQ_ALLOW_ALL_USERS": "true"})
        try:
            assert runner._is_user_authorized(_qq_dm_source()) is True
        finally:
            ss.reset_secret_scope(tok)

    def test_scope_does_not_inherit_environ_opt_in(self, monkeypatch):
        # The PRIMARY profile opted in via os.environ; the secondary profile's
        # scope has no opt-in. The secondary must NOT inherit the primary's
        # allow-all (this is the cross-profile leak the fix closes).
        monkeypatch.setenv("QQ_ALLOW_ALL_USERS", "true")
        runner = _make_qq_runner()
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({})
        try:
            assert runner._is_user_authorized(_qq_dm_source()) is False
        finally:
            ss.reset_secret_scope(tok)

    def test_single_profile_environ_unchanged(self, monkeypatch):
        # Multiplex inactive, no scope: legacy os.environ behavior preserved.
        monkeypatch.setenv("QQ_ALLOW_ALL_USERS", "true")
        runner = _make_qq_runner()
        assert runner._is_user_authorized(_qq_dm_source(profile=None)) is True


class TestStartupValidatorScope:
    @staticmethod
    def _open_dm_config():
        return GatewayConfig(
            platforms={
                Platform.QQBOT: PlatformConfig(
                    enabled=True, extra={"dm_policy": "open"}
                )
            }
        )

    def test_scoped_opt_in_clears_violation(self):
        # Mirrors _start_one_profile_adapters: the validator runs inside the
        # profile scope, so the profile's own opt-in must clear the violation.
        from gateway.run import _own_policy_open_startup_violation

        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"QQ_ALLOW_ALL_USERS": "true"})
        try:
            assert _own_policy_open_startup_violation(self._open_dm_config()) is None
        finally:
            ss.reset_secret_scope(tok)

    def test_scope_does_not_inherit_environ_opt_in(self, monkeypatch):
        # Primary's environ opt-in must not silently bless a secondary
        # profile whose own scope never opted in.
        from gateway.run import _own_policy_open_startup_violation

        monkeypatch.setenv("QQ_ALLOW_ALL_USERS", "true")
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({})
        try:
            violation = _own_policy_open_startup_violation(self._open_dm_config())
        finally:
            ss.reset_secret_scope(tok)
        assert violation is not None
        assert "qqbot" in violation

    def test_unscoped_environ_unchanged(self, monkeypatch):
        # Single-profile startup (no scope installed) keeps reading environ.
        from gateway.run import _own_policy_open_startup_violation

        monkeypatch.setenv("QQ_ALLOW_ALL_USERS", "true")
        assert _own_policy_open_startup_violation(self._open_dm_config()) is None


class TestDirectSendScope:
    @staticmethod
    def _fake_httpx(captured):
        class _Resp:
            status_code = 500

            @staticmethod
            def json():
                return {}

        class _AsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, url, **kwargs):
                captured.append(kwargs.get("json") or {})
                return _Resp()

        module = types.ModuleType("httpx")
        module.AsyncClient = _AsyncClient
        return module

    @pytest.mark.asyncio
    async def test_scoped_credentials_win_over_environ(self, monkeypatch):
        from tools.send_message_tool import _send_qqbot

        captured = []
        monkeypatch.setitem(sys.modules, "httpx", self._fake_httpx(captured))
        monkeypatch.setenv("QQ_APP_ID", "global-app")
        monkeypatch.setenv("QQ_CLIENT_SECRET", "global-secret")
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope(
            {"QQ_APP_ID": "profileA-app", "QQ_CLIENT_SECRET": "profileA-secret"}
        )
        try:
            await _send_qqbot(
                PlatformConfig(enabled=True, extra={}), "chat-1", "hi"
            )
        finally:
            ss.reset_secret_scope(tok)
        assert captured, "token request never issued"
        assert captured[0]["appId"] == "profileA-app"
        assert captured[0]["clientSecret"] == "profileA-secret"

    @pytest.mark.asyncio
    async def test_unscoped_falls_back_to_environ(self, monkeypatch):
        from tools.send_message_tool import _send_qqbot

        captured = []
        monkeypatch.setitem(sys.modules, "httpx", self._fake_httpx(captured))
        monkeypatch.setenv("QQ_APP_ID", "env-app")
        monkeypatch.setenv("QQ_CLIENT_SECRET", "env-secret")
        await _send_qqbot(PlatformConfig(enabled=True, extra={}), "chat-1", "hi")
        assert captured, "token request never issued"
        assert captured[0]["appId"] == "env-app"
        assert captured[0]["clientSecret"] == "env-secret"
