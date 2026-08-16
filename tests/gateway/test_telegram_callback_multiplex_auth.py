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

    def test_resolver_exception_fails_closed(self, monkeypatch):
        """A resolver error must DENY, not fall through to the process-wide env
        allowlist. The env fallback is not scoped to the routed profile, so
        honoring it on a resolver error would cross the profile-specific
        pairing boundary (#86296). Fail closed instead."""
        # Env allowlist WOULD allow "999"; the resolver raising must not let it.
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "999")

        def _boom(source):
            raise RuntimeError("resolver blew up")

        adapter = _make_adapter()
        adapter._callback_auth_resolver = _boom

        assert adapter._is_callback_user_authorized("999", chat_id="-100") is False

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


# -- Real callback-dispatch gating: ea:/sc:/cl: families -------------------


class _FakeUser:
    def __init__(self, uid, first_name="op"):
        self.id = uid
        self.first_name = first_name


class _FakeChat:
    def __init__(self, chat_id, chat_type="group"):
        self.id = chat_id
        self.type = chat_type


class _FakeMessage:
    def __init__(self, chat_id, chat_type="group", thread_id=None):
        self.chat_id = chat_id
        self.chat = _FakeChat(chat_id, chat_type)
        self.message_thread_id = thread_id


class _FakeQuery:
    def __init__(self, data, user_id="999", chat_id="-1001234567890"):
        self.data = data
        self.from_user = _FakeUser(user_id)
        self.message = _FakeMessage(chat_id)
        self.answers = []

    async def answer(self, text=None, **kwargs):
        self.answers.append(text)


class _FakeUpdate:
    def __init__(self, query):
        self.callback_query = query


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def _dispatch_adapter(resolver):
    adapter = _make_adapter()
    adapter._callback_auth_resolver = resolver
    # Registries the ea:/sc:/cl: blocks consult *after* the auth gate.
    adapter._approval_state = {1: "sess-ea"}
    adapter._slash_confirm_state = {"cid": "sess-sc"}
    adapter._clarify_state = {"lid": "sess-cl"}
    return adapter


class TestCallbackDispatchGating:
    """Drive the real ``_handle_callback_query`` for each gated button family
    (``ea:`` approval, ``sc:`` slash-confirm, ``cl:`` clarify) so the dispatch
    path itself — not a mocked seam — is proven to deny an unauthorized
    operator via the installed resolver. This is the exact tap path #86296
    reproduced (a pairing-authorized operator was wrongly rejected; here we
    assert the inverse boundary: an *un*authorized operator is refused and the
    pending action is never consumed)."""

    @pytest.mark.parametrize(
        "data, registry_attr, key, denial_marker",
        [
            ("ea:once:1", "_approval_state", 1, "approve"),
            ("sc:once:cid", "_slash_confirm_state", "cid", "answer this prompt"),
            ("cl:lid:0", "_clarify_state", "lid", "answer this prompt"),
        ],
    )
    def test_unauthorized_operator_denied_and_state_untouched(
        self, data, registry_attr, key, denial_marker
    ):
        adapter = _dispatch_adapter(lambda source: False)
        query = _FakeQuery(data)
        _run(adapter._handle_callback_query(_FakeUpdate(query), None))
        # Denied with an authorization message ...
        assert query.answers, "expected a denial answer"
        assert "not authorized" in query.answers[-1].lower()
        assert denial_marker in query.answers[-1].lower()
        # ... and the pending action was NOT consumed/resolved.
        assert key in getattr(adapter, registry_attr)

    @pytest.mark.parametrize(
        "data, registry_attr, key",
        [
            ("ea:once:1", "_approval_state", 1),
            ("sc:once:cid", "_slash_confirm_state", "cid"),
            ("cl:lid:0", "_clarify_state", "lid"),
        ],
    )
    def test_authorized_operator_passes_gate(self, data, registry_attr, key):
        """A resolver-authorized operator clears the gate: the block proceeds
        past the auth check to the registry lookup (here empty → 'already
        resolved'), proving the deny branch was not taken."""
        adapter = _dispatch_adapter(lambda source: True)
        # Empty registry so the post-gate step short-circuits without running
        # the heavy resolve/render code — we only assert the gate was cleared.
        setattr(adapter, registry_attr, {})
        query = _FakeQuery(data)
        _run(adapter._handle_callback_query(_FakeUpdate(query), None))
        assert query.answers
        assert "not authorized" not in query.answers[-1].lower()
        assert "already" in query.answers[-1].lower()


# -- Fully wired: runner registration + real per-profile PairingStore ------


@contextmanager
def _clean_auth_env(monkeypatch):
    """Strip every Telegram auth env so ``_is_user_authorized`` reaches the
    pairing-store check instead of short-circuiting on an allowlist."""
    for var in (
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "TELEGRAM_ALLOW_BOTS",
        "TELEGRAM_GROUP_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


def _wired_runner(monkeypatch, *, routed_profile, pairing_stores, default_store):
    """A GatewayRunner exposing the REAL callback-authorization chain:
    ``_authorize_platform_callback`` → ``_is_user_authorized`` →
    ``_pairing_store_for`` → per-profile ``PairingStore``.

    Only the seams the issue lets us stub are stubbed: route *name* resolution
    (covered live by ``TestAuthorizePlatformCallback``), the adapter-lookup
    helpers, the profile runtime scope, and profile-home resolution. Env
    allowlists and adapter-delegation are neutralized so the pairing store is
    the sole authorization authority — i.e. routing the callback to the wrong
    store, or dropping resolver registration, both surface as a denied gate.
    """
    from gateway.run import GatewayRunner

    runner = MagicMock(spec=GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=True, profile_routes=[])
    runner.pairing_stores = pairing_stores
    runner.pairing_store = default_store
    runner._profile_name_for_source = MagicMock(return_value=routed_profile)
    # Bind the real authorization chain to this runner instance.
    runner._authorize_platform_callback = (
        GatewayRunner._authorize_platform_callback.__get__(runner)
    )
    runner._is_user_authorized = GatewayRunner._is_user_authorized.__get__(runner)
    runner._pairing_store_for = GatewayRunner._pairing_store_for.__get__(runner)
    # Benign adapter-lookup helpers: never fail open through the delegation gate
    # (a bare MagicMock here would auto-truthy and admit everyone).
    runner._adapter_authorization_is_upstream = MagicMock(return_value=False)
    runner._adapter_enforces_own_access_policy = MagicMock(return_value=False)
    runner._adapter_profile_for_source = MagicMock(return_value=None)
    runner._adapter_for_source = MagicMock(return_value=None)
    runner._resolve_profile_home_for_source = MagicMock(return_value="/home")
    monkeypatch.setattr("gateway.run._profile_runtime_scope", _noop_scope)
    return runner


@pytest.fixture
def _pairing_env(monkeypatch, tmp_path):
    """Real per-profile + default PairingStores under a temp HERMES_HOME, with
    operator ``999`` paired ONLY in the ``secondary`` profile."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from gateway.pairing import PairingStore

    secondary = PairingStore(profile="secondary")
    secondary._approve_user("telegram", "999", "op")
    default_store = PairingStore(profile="default")
    # Sanity: the operator is isolated to the routed profile.
    assert secondary.is_approved("telegram", "999") is True
    assert default_store.is_approved("telegram", "999") is False
    return {"secondary": secondary}, default_store


class TestFullyWiredPairingStoreDispatch:
    """End-to-end callback gating with the resolver actually *registered* on the
    adapter (``set_callback_auth_resolver``) and a real per-profile
    ``PairingStore`` behind ``_is_user_authorized``. This is the regression the
    seam-level tests could not catch: it would fail if resolver registration
    were removed from the runner, or if the callback were authorized against the
    wrong (default) pairing store — the exact #86296 failure mode."""

    @pytest.mark.parametrize(
        "data, registry_attr, key",
        [
            ("ea:once:1", "_approval_state", 1),
            ("sc:once:cid", "_slash_confirm_state", "cid"),
            ("cl:lid:0", "_clarify_state", "lid"),
        ],
    )
    def test_paired_operator_on_routed_profile_clears_gate(
        self, monkeypatch, _pairing_env, data, registry_attr, key
    ):
        with _clean_auth_env(monkeypatch):
            pairing_stores, default_store = _pairing_env
            runner = _wired_runner(
                monkeypatch,
                routed_profile="secondary",
                pairing_stores=pairing_stores,
                default_store=default_store,
            )
            adapter = _make_adapter()
            # REAL registration: the runner installs its own resolver.
            adapter.set_callback_auth_resolver(runner._authorize_platform_callback)
            setattr(adapter, registry_attr, {})
            query = _FakeQuery(data)
            _run(adapter._handle_callback_query(_FakeUpdate(query), None))

        assert query.answers
        assert "not authorized" not in query.answers[-1].lower()
        assert "already" in query.answers[-1].lower()

    @pytest.mark.parametrize(
        "data, registry_attr, key, denial_marker",
        [
            ("ea:once:1", "_approval_state", 1, "approve"),
            ("sc:once:cid", "_slash_confirm_state", "cid", "answer this prompt"),
            ("cl:lid:0", "_clarify_state", "lid", "answer this prompt"),
        ],
    )
    def test_unpaired_default_route_denies_and_keeps_state(
        self, monkeypatch, _pairing_env, data, registry_attr, key, denial_marker
    ):
        """No matching route → the default store (where the operator is NOT
        paired) is consulted, the tap is denied, and the pending action is not
        consumed. Proves the allow branch above is not vacuous and that the
        authorization really flows through the per-source store selection."""
        with _clean_auth_env(monkeypatch):
            pairing_stores, default_store = _pairing_env
            runner = _wired_runner(
                monkeypatch,
                routed_profile=None,
                pairing_stores=pairing_stores,
                default_store=default_store,
            )
            adapter = _make_adapter()
            adapter.set_callback_auth_resolver(runner._authorize_platform_callback)
            adapter._approval_state = {1: "sess-ea"}
            adapter._slash_confirm_state = {"cid": "sess-sc"}
            adapter._clarify_state = {"lid": "sess-cl"}
            query = _FakeQuery(data)
            _run(adapter._handle_callback_query(_FakeUpdate(query), None))

        assert query.answers
        assert "not authorized" in query.answers[-1].lower()
        assert denial_marker in query.answers[-1].lower()
        assert key in getattr(adapter, registry_attr)


class TestProductionWiringRegistersResolver:
    """Guard the *production* registration, not just a hand-installed resolver.

    ``start`` and ``_platform_reconnect_watcher`` both wire the primary adapter
    through ``_wire_primary_adapter_handlers``. The dispatch tests above call
    ``set_callback_auth_resolver`` themselves, so they would stay green if that
    single production line were deleted — the exact regression #86296 guards
    against. This drives the real helper and asserts the runner installs its own
    ``_authorize_platform_callback`` as the adapter's callback-auth resolver.
    """

    def test_wire_primary_adapter_handlers_installs_callback_auth_resolver(self):
        from gateway.run import GatewayRunner

        runner = MagicMock()
        # Bind only the real method under test; every collaborator stays a mock.
        runner._wire_primary_adapter_handlers = (
            GatewayRunner._wire_primary_adapter_handlers.__get__(runner)
        )
        adapter = MagicMock()

        runner._wire_primary_adapter_handlers(adapter)

        # The production choke point must register the runner's own resolver
        # (both start and reconnect route through this same helper).
        adapter.set_callback_auth_resolver.assert_called_once_with(
            runner._authorize_platform_callback
        )
        # Sibling sanity: the primary event handler is still wired here too, so
        # the helper is the genuine primary-adapter setup path.
        adapter.set_platform_event_handler.assert_called_once()
