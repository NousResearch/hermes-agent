"""Regression tests for the multiplex-profile admin-tier scoping bugs
sweeper flagged on PR #59857:

1. ``plugins/platforms/telegram/adapter.py``'s exec-approval button gate
   used to resolve the runner via ``_message_handler.__self__``, which is
   ``None`` for a secondary multiplexed adapter (its handler is a closure
   from ``_make_profile_message_handler``, not a bound method) -- silently
   failing OPEN (any authorized user treated as admin) even when that
   profile configured ``allow_admin_from``.

2. ``gateway/run.py``'s turn-dispatch non-admin marking read ``self.config``,
   which is the PRIMARY profile's config -- wrong for a secondary
   multiplexed profile's turn, which runs inside that profile's OWN
   ``_profile_runtime_scope``.

Both are fixed by resolving the policy fresh (profile-bound closure /
``load_gateway_config()``) instead of reading a cached attribute off
whatever object happens to be reachable.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner, _resolve_approval_session_is_non_admin
from gateway.session import SessionSource


def _tiered_config(admin_ids):
    return GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True, token="***",
                extra={"allow_admin_from": list(admin_ids)},
            )
        }
    )


def _source(user_id: str) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id=user_id,
    )


def _make_runner(config) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = config
    return runner


class TestMakeAdapterAdminPolicyCheck:
    """GatewayRunner._make_adapter_admin_policy_check -- the injected,
    profile-bound resolver that replaces __self__ introspection."""

    def test_profile_home_none_uses_self_config(self):
        runner = _make_runner(_tiered_config(admin_ids=["999"]))
        check = runner._make_adapter_admin_policy_check(profile_home=None)

        assert check(_source("999")) is True
        assert check(_source("12345")) is False

    def test_profile_home_set_uses_scoped_config_not_self_config(self, monkeypatch):
        """The exact bug: for a secondary profile, self.config must be
        IGNORED entirely -- only the scoped load matters. Deliberately make
        self.config say the opposite of the scoped config so a test that
        accidentally reads self.config would be caught."""
        # self.config would let user "12345" through (empty admin list == disabled).
        runner = _make_runner(_tiered_config(admin_ids=[]))

        scoped_config = _tiered_config(admin_ids=["999"])
        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: scoped_config)

        calls = []

        def _fake_scope(profile_home):
            calls.append(profile_home)
            from contextlib import contextmanager

            @contextmanager
            def _cm():
                yield
            return _cm()

        monkeypatch.setattr("gateway.run._profile_runtime_scope", _fake_scope)

        check = runner._make_adapter_admin_policy_check(profile_home="/fake/profile/home")

        # Scoped config's tier (admin=["999"]) applies, NOT self.config's
        # (which would have let "12345" through as "no tier configured").
        assert check(_source("999")) is True
        assert check(_source("12345")) is False
        assert calls == ["/fake/profile/home", "/fake/profile/home"]

    def test_resolution_error_fails_toward_existing_behavior(self, monkeypatch):
        """An unresolvable policy must not newly lock out an
        already-authorized approver (matches the pre-existing __self__-path
        fail-toward-permissive convention)."""
        runner = _make_runner(_tiered_config(admin_ids=["999"]))
        check = runner._make_adapter_admin_policy_check(profile_home="/fake")
        monkeypatch.setattr(
            "gateway.run._profile_runtime_scope",
            MagicMock(side_effect=RuntimeError("boom")),
        )
        assert check(_source("12345")) is True


class TestAdapterButtonGateUsesInjectedCheckNotHandlerIntrospection:
    """The exact scenario sweeper described: a secondary multiplexed
    adapter's _message_handler is a closure (no __self__). Before the fix,
    TelegramAdapter._is_callback_user_admin's getattr(..., "__self__", None)
    chain resolved to None -> runner_config None -> fail-open (return True)
    even with allow_admin_from configured for that profile.
    """

    def _make_adapter(self):
        import sys
        from unittest.mock import MagicMock as _MM

        if "telegram" not in sys.modules or not hasattr(sys.modules["telegram"], "__file__"):
            mod = _MM()
            mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
            mod.error.NetworkError = type("NetworkError", (OSError,), {})
            mod.error.TimedOut = type("TimedOut", (OSError,), {})
            mod.error.BadRequest = type("BadRequest", (Exception,), {})
            for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
                sys.modules.setdefault(name, mod)
            sys.modules.setdefault("telegram.error", mod.error)

        from plugins.platforms.telegram.adapter import TelegramAdapter
        config = PlatformConfig(enabled=True, token="test-token")
        adapter = TelegramAdapter(config)
        adapter._bot = _MM()
        adapter._app = _MM()
        return adapter

    def test_closure_handler_with_no_self_still_enforces_configured_tier(self):
        """Regression: _message_handler with no __self__ (the secondary
        multiplex shape) must not fail open once set_admin_policy_check has
        been used to inject a real resolver."""
        adapter = self._make_adapter()

        async def _closure_handler(event):
            return None
        adapter._message_handler = _closure_handler
        assert not hasattr(adapter._message_handler, "__self__")

        runner = _make_runner(_tiered_config(admin_ids=["999"]))
        adapter.set_admin_policy_check(runner._make_adapter_admin_policy_check())

        assert adapter._is_callback_user_admin("12345", chat_id="12345") is False
        assert adapter._is_callback_user_admin("999", chat_id="12345") is True

    def test_no_admin_policy_check_registered_falls_back_to_permissive(self):
        """Bare-adapter test paths (nothing injected at all) keep today's
        single-tier behavior: any authorized user may approve."""
        adapter = self._make_adapter()
        assert adapter._is_callback_user_admin("12345", chat_id="12345") is True


class TestResolveApprovalSessionIsNonAdmin:
    """gateway.run._resolve_approval_session_is_non_admin.

    Non-multiplex (the common case): base_config IS already correct, used
    directly with zero extra disk I/O on the turn-dispatch hot path.

    Multiplex active: loads config fresh via gateway.config.load_gateway_config()
    instead of trusting base_config, since for a secondary profile's turn
    (already inside that profile's _profile_runtime_scope) base_config is
    still the PRIMARY profile's parsed config.
    """

    def test_admin_user_is_not_flagged_non_admin(self):
        base_config = _tiered_config(admin_ids=["999"])
        assert _resolve_approval_session_is_non_admin(_source("999"), base_config) is False

    def test_non_admin_user_is_flagged(self):
        base_config = _tiered_config(admin_ids=["999"])
        assert _resolve_approval_session_is_non_admin(_source("12345"), base_config) is True

    def test_no_tier_configured_is_a_no_op(self):
        base_config = _tiered_config(admin_ids=[])
        assert _resolve_approval_session_is_non_admin(_source("12345"), base_config) is False

    def test_non_multiplex_never_calls_load_gateway_config(self, monkeypatch):
        """Zero extra disk I/O for the common (non-multiplex) case -- base_config
        is already correct, so the expensive fresh load must not fire at all."""
        calls = []
        monkeypatch.setattr(
            "gateway.config.load_gateway_config",
            lambda: calls.append(1) or _tiered_config(admin_ids=["999"]),
        )
        base_config = _tiered_config(admin_ids=["999"])
        _resolve_approval_session_is_non_admin(_source("12345"), base_config)
        assert calls == []

    def test_multiplex_active_ignores_base_config_uses_fresh_scoped_load(self, monkeypatch):
        """The exact bug: for a multiplexed secondary-profile turn,
        base_config (the primary profile's config) must be IGNORED entirely.
        Deliberately make base_config say the opposite of the freshly-loaded
        scoped config so a test that accidentally reads base_config would be
        caught."""
        # base_config says "12345" is fine (disabled tier) -- the scoped
        # config (what a multiplexed secondary profile's turn actually sees)
        # says only "999" is admin. If the fix regresses to reading
        # base_config, this user would wrongly NOT be flagged non-admin.
        base_config = _tiered_config(admin_ids=[])
        base_config.multiplex_profiles = True

        scoped_config = _tiered_config(admin_ids=["999"])
        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: scoped_config)

        assert _resolve_approval_session_is_non_admin(_source("12345"), base_config) is True
        assert _resolve_approval_session_is_non_admin(_source("999"), base_config) is False

    def test_missing_base_config_is_a_safe_no_op_not_a_raise(self):
        """Regression: the call site passes ``getattr(self, "config", None)``,
        not a bare ``self.config`` -- a runner test fixture with no .config
        attribute at all (test_reasoning_command.py's minimal GatewayRunner
        double) must get a safe no-op here, not an AttributeError escaping
        the call site (which has no enclosing try/except of its own; only
        this function's internals do)."""
        assert _resolve_approval_session_is_non_admin(_source("12345"), None) is False
