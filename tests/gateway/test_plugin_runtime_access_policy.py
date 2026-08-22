"""Central authorization contracts for plugin-provided platform policy."""

import builtins
from collections.abc import Mapping
import logging
import sys
from types import SimpleNamespace
import weakref

import pytest

from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.config import Platform
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.session import SessionSource


PLATFORM_NAME = "runtime_auth_test"
ALLOWED_USERS_ENV = "RUNTIME_AUTH_TEST_ALLOWED_USERS"
ALLOW_ALL_ENV = "RUNTIME_AUTH_TEST_ALLOW_ALL_USERS"


class _Adapter:
    pass


class _ExplodingString:
    def __str__(self):
        raise RuntimeError("identity conversion unavailable")


class _Runner(GatewayAuthorizationMixin):
    def __init__(self):
        self.adapters = {}
        self._profile_adapters = {}
        self.pairing_store = None
        self.pairing_stores = {}
        self.config = SimpleNamespace(platforms={})


def _source(user_id: str, *, profile: str | None = None) -> SessionSource:
    return SessionSource(
        platform=Platform(PLATFORM_NAME),
        user_id=user_id,
        chat_id="test-chat",
        user_name=user_id,
        chat_type="dm",
        profile=profile,
    )


@pytest.fixture(autouse=True)
def _clean_registry_and_env(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "gateway.run",
        SimpleNamespace(logger=logging.getLogger("gateway.run")),
    )
    scope = platform_registry.current_scope_key()
    for name in (
        ALLOWED_USERS_ENV,
        ALLOW_ALL_ENV,
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(name, raising=False)
    platform_registry.unregister(PLATFORM_NAME, scope=scope)
    yield
    platform_registry.unregister(PLATFORM_NAME, scope=scope)


def _register_runtime_policy(*, resolver=None, normalizer=None):
    platform_registry.register(
        PlatformEntry(
            name=PLATFORM_NAME,
            label="Runtime auth test",
            adapter_factory=lambda _cfg: None,
            check_fn=lambda: True,
            allowed_users_env=ALLOWED_USERS_ENV,
            allow_all_env=ALLOW_ALL_ENV,
            authorization_config_fn=resolver,
            authorization_user_normalizer=normalizer,
        ),
        scope=platform_registry.current_scope_key(),
    )


def test_runtime_authorization_hooks_are_appended_to_platform_entry_abi():
    fields = list(PlatformEntry.__dataclass_fields__)

    assert fields[-2:] == [
        "authorization_config_fn",
        "authorization_user_normalizer",
    ]


def test_runtime_resolver_receives_transport_owning_profile():
    profiles = []
    _register_runtime_policy(resolver=lambda profile: profiles.append(profile) or {})
    runner = _Runner()
    adapter = _Adapter()
    runner._profile_adapters = {
        "transport-profile": {Platform(PLATFORM_NAME): adapter},
    }
    source = _source("alice", profile="routed-profile")
    setattr(source, "_transport_adapter_ref", weakref.ref(adapter))

    assert runner._is_user_authorized(source) is False
    assert profiles == ["transport-profile"]


def test_runtime_resolver_allowed_users_feed_central_authorization():
    policy = {"allowed_users": ["alice"]}
    _register_runtime_policy(resolver=lambda _profile: policy)
    runner = _Runner()

    assert runner._is_user_authorized(_source("alice")) is True
    assert runner._is_user_authorized(_source("bob")) is False


def test_runtime_resolver_allow_all_feeds_central_authorization():
    policy = {"allow_all_users": True}
    _register_runtime_policy(resolver=lambda _profile: policy)
    runner = _Runner()

    assert runner._is_user_authorized(_source("alice")) is True


@pytest.mark.parametrize("resolved", [["alice"], "alice", 1, False, object()])
def test_runtime_resolver_non_mapping_output_fails_closed(monkeypatch, resolved):
    _register_runtime_policy(resolver=lambda _profile: resolved)
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")

    assert _Runner()._is_user_authorized(_source("alice")) is False


@pytest.mark.parametrize(
    "allowed_users",
    [
        "alice",
        {"alice": True},
        1,
        None,
        ["alice", 2],
        [object()],
        [_ExplodingString()],
    ],
)
def test_runtime_resolver_rejects_malformed_allowed_users(
    monkeypatch, allowed_users
):
    _register_runtime_policy(
        resolver=lambda _profile: {"allowed_users": allowed_users}
    )
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")

    assert _Runner()._is_user_authorized(_source("alice")) is False


@pytest.mark.parametrize("allow_all", ["true", "false", 1, 0, None, [], object()])
def test_runtime_resolver_rejects_non_boolean_allow_all(monkeypatch, allow_all):
    _register_runtime_policy(
        resolver=lambda _profile: {"allow_all_users": allow_all}
    )
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")

    assert _Runner()._is_user_authorized(_source("alice")) is False


def test_runtime_resolver_rejects_unknown_policy_keys(monkeypatch):
    _register_runtime_policy(
        resolver=lambda _profile: {"allowed_users": ["alice"], "allow_admins": True}
    )
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")

    assert _Runner()._is_user_authorized(_source("alice")) is False


class _ExplodingMapping(Mapping):
    def __iter__(self):
        raise RuntimeError("policy keys unavailable")

    def __len__(self):
        return 1

    def __getitem__(self, _key):
        raise RuntimeError("policy value unavailable")


def test_runtime_resolver_mapping_validation_exception_fails_closed(monkeypatch):
    _register_runtime_policy(resolver=lambda _profile: _ExplodingMapping())
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")

    assert _Runner()._is_user_authorized(_source("alice")) is False


def test_runtime_resolver_valid_absent_empty_and_false_values_do_not_deny_global_grant(
    monkeypatch,
):
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")

    for policy in (None, {}, {"allowed_users": []}, {"allow_all_users": False}):
        _register_runtime_policy(resolver=lambda _profile, policy=policy: policy)
        assert _Runner()._is_user_authorized(_source("alice")) is True


def test_malformed_runtime_resolver_denies_even_with_authoritative_plugin_env(
    monkeypatch,
):
    _register_runtime_policy(resolver=lambda _profile: ["invalid"])
    monkeypatch.setenv(ALLOWED_USERS_ENV, "alice")
    monkeypatch.setenv(ALLOW_ALL_ENV, "true")

    assert _Runner()._is_user_authorized(_source("alice")) is False


def test_explicit_scoped_env_values_override_runtime_policy(monkeypatch):
    from agent import secret_scope

    _register_runtime_policy(
        resolver=lambda _profile: {
            "allowed_users": ["resolver-user"],
            "allow_all_users": True,
        }
    )
    monkeypatch.setenv(ALLOWED_USERS_ENV, "other-profile-user")
    monkeypatch.setenv(ALLOW_ALL_ENV, "true")
    scoped_values = {ALLOWED_USERS_ENV: "scoped-user", ALLOW_ALL_ENV: "false"}
    token = secret_scope.set_secret_scope(scoped_values)
    secret_scope.set_multiplex_active(True)
    try:
        runner = _Runner()
        assert runner._is_user_authorized(_source("scoped-user")) is True
        assert runner._is_user_authorized(_source("resolver-user")) is False
        assert runner._is_user_authorized(_source("other-profile-user")) is False

        scoped_values[ALLOWED_USERS_ENV] = ""
        scoped_values[ALLOW_ALL_ENV] = ""
        assert runner._is_user_authorized(_source("scoped-user")) is False
        assert runner._is_user_authorized(_source("resolver-user")) is False
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)


@pytest.mark.parametrize(
    "failure",
    ["import", "scope", "multiplex", "membership", "value"],
)
def test_scoped_gate_lookup_failures_deny_permissive_runtime_policy(
    monkeypatch, failure
):
    from agent import secret_scope

    _register_runtime_policy(resolver=lambda _profile: {"allow_all_users": True})

    class FailingScope(dict):
        def __contains__(self, key):
            if failure == "membership":
                raise RuntimeError("membership unavailable")
            return super().__contains__(key)

        def get(self, key, default=None):
            if failure == "value":
                raise RuntimeError("value unavailable")
            return super().get(key, default)

    scope = FailingScope({ALLOW_ALL_ENV: "false"})
    monkeypatch.setattr(
        secret_scope,
        "current_secret_scope",
        lambda: (_ for _ in ()).throw(RuntimeError("scope unavailable"))
        if failure == "scope"
        else scope,
    )
    monkeypatch.setattr(
        secret_scope,
        "is_multiplex_active",
        lambda: (_ for _ in ()).throw(RuntimeError("multiplex unavailable"))
        if failure == "multiplex"
        else True,
    )
    if failure == "import":
        real_import = builtins.__import__

        def fail_import(name, *args, **kwargs):
            if name == "agent.secret_scope":
                raise ImportError("scope import unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_import)

    assert _Runner()._is_user_authorized(_source("alice")) is False


def test_plugin_normalizer_applies_to_sender_and_env_or_resolver_allowlists(
    monkeypatch,
):
    def normalize(value):
        return str(value).strip().lower().removeprefix("alias:") or None

    _register_runtime_policy(
        resolver=lambda _profile: {"allowed_users": ["alias:BOB"]},
        normalizer=normalize,
    )
    runner = _Runner()

    monkeypatch.setenv(ALLOWED_USERS_ENV, "alias:ALICE")
    assert runner._is_user_authorized(_source("ALICE")) is True

    monkeypatch.delenv(ALLOWED_USERS_ENV)
    assert runner._is_user_authorized(_source("BOB")) is True


def test_plugin_normalizer_none_denies_before_wildcard_or_pairing(monkeypatch):
    _register_runtime_policy(normalizer=lambda _value: None)
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "*")
    runner = _Runner()
    setattr(runner, "pairing_store", SimpleNamespace(is_approved=lambda *_args: True))

    assert runner._is_user_authorized(_source("malformed")) is False


def test_malformed_buzz_sender_is_denied_before_any_allowlist(monkeypatch):
    from plugins.platforms.buzz.settings import normalize_user_ref

    _register_runtime_policy(normalizer=normalize_user_ref)
    monkeypatch.setenv(ALLOWED_USERS_ENV, "*")

    assert _Runner()._is_user_authorized(_source("not-a-public-key")) is False


def test_plugin_global_allowlist_alias_is_normalized(monkeypatch):
    def normalize(value):
        return str(value).strip().lower().removeprefix("alias:") or None

    _register_runtime_policy(normalizer=normalize)
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "alias:ALICE")

    assert _Runner()._is_user_authorized(_source("alice")) is True


def test_plugin_pairing_uses_canonical_sender_identity():
    def normalize(value):
        return str(value).strip().lower().removeprefix("alias:") or None

    _register_runtime_policy(normalizer=normalize)
    calls = []
    runner = _Runner()
    setattr(
        runner,
        "pairing_store",
        SimpleNamespace(
            is_approved=lambda platform, user: calls.append((platform, user)) or True
        ),
    )

    assert runner._is_user_authorized(_source("alias:ALICE")) is True
    assert calls == [(PLATFORM_NAME, "alice")]


def test_plugin_wildcard_survives_allowlist_normalization(monkeypatch):
    def normalize(value):
        return str(value).strip().lower().removeprefix("alias:") or None

    _register_runtime_policy(normalizer=normalize)
    monkeypatch.setenv(ALLOWED_USERS_ENV, "*")

    assert _Runner()._is_user_authorized(_source("ALICE")) is True


def test_runtime_resolver_exception_fails_closed(monkeypatch):
    def fail(_profile):
        raise RuntimeError("policy unavailable")

    _register_runtime_policy(resolver=fail)
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")

    assert _Runner()._is_user_authorized(_source("alice")) is False


def test_configured_identity_normalizer_exception_fails_closed(monkeypatch):
    def fail(_value):
        raise RuntimeError("identity unavailable")

    _register_runtime_policy(normalizer=fail)
    monkeypatch.setenv(ALLOWED_USERS_ENV, "alice")

    assert _Runner()._is_user_authorized(_source("alice")) is False


def test_inbound_sender_normalizer_exception_fails_closed(monkeypatch):
    calls = 0

    def fail_for_sender(value):
        nonlocal calls
        calls += 1
        if calls == 1:
            return value
        raise RuntimeError("sender identity unavailable")

    _register_runtime_policy(normalizer=fail_for_sender)
    monkeypatch.setenv(ALLOWED_USERS_ENV, "alice")
    authorized = True
    try:
        authorized = _Runner()._is_user_authorized(_source("alice"))
    except RuntimeError:
        pass

    assert authorized is False


def test_stock_platform_allowlist_behavior_is_unchanged(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "12345")
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="12345",
        chat_id="test-chat",
        user_name="Alice",
        chat_type="dm",
    )

    assert _Runner()._is_user_authorized(source) is True


def test_plugin_without_authorization_hooks_keeps_env_allowlist_behavior(monkeypatch):
    _register_runtime_policy()
    monkeypatch.setenv(ALLOWED_USERS_ENV, "alice")

    assert _Runner()._is_user_authorized(_source("alice")) is True
    assert _Runner()._is_user_authorized(_source("bob")) is False
