"""Tests for plugin-provided runtime authorization configuration."""

from types import SimpleNamespace

import pytest

from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.config import Platform
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.session import SessionSource


PLATFORM_NAME = "runtime_auth_test"
AUTH_ENV_VARS = (
    "RUNTIME_AUTH_TEST_ALLOWED_USERS",
    "RUNTIME_AUTH_TEST_ALLOW_ALL_USERS",
    "GATEWAY_ALLOWED_USERS",
    "GATEWAY_ALLOW_ALL_USERS",
)


class _Runner(GatewayAuthorizationMixin):
    def __init__(self):
        self.adapters = {}
        self._profile_adapters = {}
        self.pairing_store = None
        self.pairing_stores = {}
        self.config = SimpleNamespace(platforms={})


def _source(user_id: str, *, profile=None) -> SessionSource:
    source = SessionSource(
        platform=Platform(PLATFORM_NAME),
        user_id=user_id,
        chat_id="test-chat",
        user_name=user_id,
        chat_type="dm",
    )
    source.profile = profile
    return source


@pytest.fixture(autouse=True)
def _clean_registry_and_env(monkeypatch):
    for name in AUTH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    platform_registry.unregister(PLATFORM_NAME)
    yield
    platform_registry.unregister(PLATFORM_NAME)


def _register_runtime_policy(policy, *, normalizer=None, resolver=None):
    platform_registry.register(
        PlatformEntry(
            name=PLATFORM_NAME,
            label="Runtime auth test",
            adapter_factory=lambda _cfg: None,
            check_fn=lambda: True,
            allowed_users_env="RUNTIME_AUTH_TEST_ALLOWED_USERS",
            allow_all_env="RUNTIME_AUTH_TEST_ALLOW_ALL_USERS",
            authorization_config_fn=resolver or (lambda _profile: policy),
            authorization_user_normalizer=normalizer,
        )
    )


def test_plugin_runtime_allowlist_is_applied_by_central_authorization():
    policy = {"allowed_users": ["alice"], "allow_all_users": False}
    _register_runtime_policy(policy)
    runner = _Runner()

    assert runner._is_user_authorized(_source("alice")) is True
    assert runner._is_user_authorized(_source("bob")) is False

    policy["allowed_users"] = ["bob"]

    assert runner._is_user_authorized(_source("alice")) is False
    assert runner._is_user_authorized(_source("bob")) is True


def test_plugin_runtime_allow_all_can_be_revoked():
    policy = {"allowed_users": [], "allow_all_users": False}
    _register_runtime_policy(policy)
    runner = _Runner()

    assert runner._is_user_authorized(_source("alice")) is False
    policy["allow_all_users"] = True
    assert runner._is_user_authorized(_source("alice")) is True
    policy.pop("allow_all_users")
    assert runner._is_user_authorized(_source("alice")) is False


def test_explicit_plugin_env_overrides_runtime_policy(monkeypatch):
    _register_runtime_policy(
        {"allowed_users": ["yaml-user"], "allow_all_users": True}
    )
    monkeypatch.setenv("RUNTIME_AUTH_TEST_ALLOWED_USERS", "env-user")
    monkeypatch.setenv("RUNTIME_AUTH_TEST_ALLOW_ALL_USERS", "false")
    runner = _Runner()

    assert runner._is_user_authorized(_source("env-user")) is True
    assert runner._is_user_authorized(_source("yaml-user")) is False


def test_explicit_empty_plugin_env_suppresses_runtime_allowlist(monkeypatch):
    _register_runtime_policy(
        {"allowed_users": ["yaml-user"], "allow_all_users": False}
    )
    monkeypatch.setenv("RUNTIME_AUTH_TEST_ALLOWED_USERS", "")
    runner = _Runner()

    assert runner._is_user_authorized(_source("yaml-user")) is False


def test_multiplex_profile_env_does_not_fall_through_to_process_env(monkeypatch):
    from agent import secret_scope

    _register_runtime_policy(
        {"allowed_users": ["yaml-user"], "allow_all_users": False}
    )
    monkeypatch.setenv("RUNTIME_AUTH_TEST_ALLOWED_USERS", "other-profile-user")
    token = secret_scope.set_secret_scope(
        {"RUNTIME_AUTH_TEST_ALLOWED_USERS": "active-profile-user"}
    )
    secret_scope.set_multiplex_active(True)
    try:
        runner = _Runner()
        assert runner._is_user_authorized(_source("active-profile-user")) is True
        assert runner._is_user_authorized(_source("other-profile-user")) is False
        assert runner._is_user_authorized(_source("yaml-user")) is False
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)


def test_multiplex_missing_profile_keys_do_not_use_process_env(monkeypatch):
    from agent import secret_scope

    _register_runtime_policy({})
    monkeypatch.setenv("RUNTIME_AUTH_TEST_ALLOWED_USERS", "other-profile-user")
    monkeypatch.setenv("RUNTIME_AUTH_TEST_ALLOW_ALL_USERS", "true")
    token = secret_scope.set_secret_scope({})
    secret_scope.set_multiplex_active(True)
    try:
        runner = _Runner()
        assert runner._is_user_authorized(_source("other-profile-user")) is False
        assert runner._is_user_authorized(_source("any-user")) is False
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)


def test_multiplex_unscoped_authorization_does_not_use_process_env(monkeypatch):
    from agent import secret_scope

    _register_runtime_policy({})
    monkeypatch.setenv("RUNTIME_AUTH_TEST_ALLOW_ALL_USERS", "true")
    secret_scope.set_multiplex_active(True)
    try:
        assert _Runner()._is_user_authorized(_source("any-user")) is False
    finally:
        secret_scope.set_multiplex_active(False)


def test_runtime_resolver_receives_source_profile():
    profiles = []

    def resolve(profile):
        profiles.append(profile)
        return {"allowed_users": ["alice"]}

    _register_runtime_policy({}, resolver=resolve)

    assert _Runner()._is_user_authorized(
        _source("alice", profile="secondary")
    ) is True
    assert profiles == ["secondary"]


def test_plugin_normalizer_applies_to_explicit_env_allowlist(monkeypatch):
    from plugins.platforms.buzz.adapter import hex_to_npub, _normalize_user_ref

    pubkey = "a" * 64
    npub = hex_to_npub(pubkey)
    assert npub is not None
    _register_runtime_policy({}, normalizer=_normalize_user_ref)
    monkeypatch.setenv("RUNTIME_AUTH_TEST_ALLOWED_USERS", npub)

    assert _Runner()._is_user_authorized(_source(pubkey)) is True

    monkeypatch.setenv("RUNTIME_AUTH_TEST_ALLOWED_USERS", pubkey.upper())
    assert _Runner()._is_user_authorized(_source(pubkey)) is True


def test_runtime_hooks_are_appended_to_platform_entry_abi():
    fields = list(PlatformEntry.__dataclass_fields__)

    assert fields.index("authorization_config_fn") > fields.index(
        "standalone_sender_fn"
    )
    assert fields.index("authorization_user_normalizer") > fields.index(
        "standalone_sender_fn"
    )
