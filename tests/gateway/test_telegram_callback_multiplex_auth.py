"""Telegram inline-callback auth under multiplex profile routing (#86296).

Under ``gateway.multiplex_profiles`` the primary Telegram event handler is a
closure installed via ``_make_default_profile_platform_event_handler`` — it has
no bound ``__self__``, so the adapter cannot recover the runner through
``_message_handler.__self__`` and callback auth fell back to env-only,
**skipping the routed profile's pairing store**. A pairing-authorized operator
could therefore send text (authorized through the route) but was rejected when
tapping an inline **Approve Once** / clarify / dangerous-command button.

The fix installs a runner-owned resolver on the adapter
(``set_callback_auth_resolver``) that stamps the routed profile and enters its
runtime scope before the normal gateway authorization decision. These tests
cover both seams: the adapter preferring the resolver, and the runner's
``_authorize_platform_callback`` applying route resolution + fail-closed rules.
"""

import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig, Platform
from gateway.session import SessionSource


# -- Fake telegram modules (minimal stubs) --------------------------------

_fake_telegram_error = types.ModuleType("telegram.error")


class _TelegramError(Exception):
    pass


_fake_telegram_error.TelegramError = _TelegramError
_fake_telegram_error.BadRequest = type("BadRequest", (_TelegramError,), {})
_fake_telegram_error.NetworkError = type("NetworkError", (_TelegramError,), {})

_fake_telegram_constants = types.ModuleType("telegram.constants")
_fake_telegram_constants.ParseMode = SimpleNamespace(HTML="HTML")

_fake_telegram_request = types.ModuleType("telegram.request")
_fake_telegram_request.HTTPXRequest = type(
    "HTTPXRequest", (), {"__init__": lambda *a, **kw: None}
)

_fake_telegram_ext = types.ModuleType("telegram.ext")
_fake_telegram_ext.ApplicationBuilder = type(
    "ApplicationBuilder",
    (),
    {"token": lambda self, *a: self, "build": lambda self: None},
)

_fake_telegram = types.ModuleType("telegram")
_fake_telegram.error = _fake_telegram_error
_fake_telegram.constants = _fake_telegram_constants
_fake_telegram.ext = _fake_telegram_ext
_fake_telegram.request = _fake_telegram_request


@pytest.fixture(autouse=True)
def _inject_fake_telegram(monkeypatch):
    monkeypatch.setitem(sys.modules, "telegram", _fake_telegram)
    monkeypatch.setitem(sys.modules, "telegram.error", _fake_telegram_error)
    monkeypatch.setitem(sys.modules, "telegram.constants", _fake_telegram_constants)
    monkeypatch.setitem(sys.modules, "telegram.ext", _fake_telegram_ext)
    monkeypatch.setitem(sys.modules, "telegram.request", _fake_telegram_request)


def _make_adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = object.__new__(TelegramAdapter)
    adapter.config = config
    adapter._config = config
    adapter._platform = Platform.TELEGRAM
    adapter._connected = True
    adapter._message_handler = None
    adapter._callback_auth_resolver = None
    return adapter


# -- Adapter seam: prefer the installed resolver --------------------------


class TestAdapterPrefersResolver:
    def test_resolver_receives_routed_source_and_result_flows_through(
        self, monkeypatch
    ):
        """The resolver is called with the callback SessionSource and its
        allow/deny decision is returned verbatim — no env fallthrough."""
        # No env allowlist: the legacy fallback would fail closed (deny),
        # so a True result can only come from the resolver.
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)

        seen = {}

        def _resolver(source):
            seen["source"] = source
            return True

        adapter = _make_adapter()
        adapter._callback_auth_resolver = _resolver

        assert (
            adapter._is_callback_user_authorized(
                "999",
                chat_id="-1001234567890",
                chat_type="supergroup",
                thread_id=None,
                user_name="op",
            )
            is True
        )
        src = seen["source"]
        assert isinstance(src, SessionSource)
        assert src.platform == Platform.TELEGRAM
        assert src.user_id == "999"
        assert src.chat_id == "-1001234567890"
        assert src.chat_type == "group"  # supergroup w/o thread → group

    def test_resolver_deny_does_not_fall_through_to_env(self, monkeypatch):
        """A resolver denial is final even when an env allowlist would allow."""
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "999")
        adapter = _make_adapter()
        adapter._callback_auth_resolver = lambda source: False

        assert adapter._is_callback_user_authorized("999", chat_id="-100") is False

    def test_resolver_exception_falls_back_to_env(self, monkeypatch):
        """If the resolver raises, the adapter falls back to env-only auth."""
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "999")

        def _boom(source):
            raise RuntimeError("resolver blew up")

        adapter = _make_adapter()
        adapter._callback_auth_resolver = _boom

        assert adapter._is_callback_user_authorized("999", chat_id="-100") is True

    def test_no_resolver_still_fail_closed_without_allowlist(self, monkeypatch):
        """No resolver, no allowlist, no allow-all → deny (preserves #24457)."""
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        adapter = _make_adapter()
        assert adapter._is_callback_user_authorized("999") is False


# -- Runner seam: _authorize_platform_callback ----------------------------


@contextmanager
def _noop_scope(_home):
    yield


def _make_runner(monkeypatch, *, multiplex=True):
    """Minimal GatewayRunner with the resolver method bound to real code."""
    from gateway.run import GatewayRunner

    runner = MagicMock(spec=GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=multiplex, profile_routes=[])
    runner._authorize_platform_callback = (
        GatewayRunner._authorize_platform_callback.__get__(runner)
    )
    runner._resolve_profile_home_for_source = MagicMock(return_value="/home")
    # Neutralize the real env/HOME side effects of entering a profile scope.
    monkeypatch.setattr("gateway.run._profile_runtime_scope", _noop_scope)
    return runner


def _callback_source(profile=None):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001234567890",
        chat_type="group",
        user_id="999",
        profile=profile,
    )


class TestAuthorizePlatformCallback:
    def test_matched_route_stamps_profile_then_authorizes(self, monkeypatch):
        """A matching route stamps ``source.profile`` before the auth decision
        so the routed profile's pairing store is consulted (issue items 3-5)."""
        runner = _make_runner(monkeypatch)
        runner._profile_name_for_source = MagicMock(return_value="secondary")
        seen = {}

        def _auth(source):
            seen["profile"] = source.profile
            return True

        runner._is_user_authorized = _auth

        source = _callback_source()
        assert runner._authorize_platform_callback(source) is True
        assert source.profile == "secondary"
        assert seen["profile"] == "secondary"

    def test_denied_operator_is_rejected(self, monkeypatch):
        """An operator not approved on the routed profile is denied (item 6)."""
        runner = _make_runner(monkeypatch)
        runner._profile_name_for_source = MagicMock(return_value="secondary")
        runner._is_user_authorized = MagicMock(return_value=False)

        assert runner._authorize_platform_callback(_callback_source()) is False

    def test_route_to_unserved_profile_fails_closed(self, monkeypatch):
        """An explicit route to an unserved profile fails closed and never
        reaches the authorization decision (item 8)."""
        from gateway.profile_routing import ProfileRouteRejected

        runner = _make_runner(monkeypatch)
        runner._profile_name_for_source = MagicMock(
            side_effect=ProfileRouteRejected("secondary")
        )
        runner._is_user_authorized = MagicMock(return_value=True)

        source = _callback_source()
        assert runner._authorize_platform_callback(source) is False
        assert source.profile_route_rejected is True
        runner._is_user_authorized.assert_not_called()

    def test_unmatched_chat_uses_default_profile(self, monkeypatch):
        """No matching route leaves ``profile`` None → default pairing store
        (item 7)."""
        runner = _make_runner(monkeypatch)
        runner._profile_name_for_source = MagicMock(return_value=None)
        seen = {}

        def _auth(source):
            seen["profile"] = source.profile
            return True

        runner._is_user_authorized = _auth

        source = _callback_source()
        assert runner._authorize_platform_callback(source) is True
        assert source.profile is None
        assert seen["profile"] is None

    def test_non_multiplex_skips_routing(self, monkeypatch):
        """With multiplexing off, no route resolution runs; the source goes
        straight to the authorization decision."""
        runner = _make_runner(monkeypatch, multiplex=False)
        runner._profile_name_for_source = MagicMock()
        runner._is_user_authorized = MagicMock(return_value=True)

        assert runner._authorize_platform_callback(_callback_source()) is True
        runner._profile_name_for_source.assert_not_called()
