"""Real registry/runner authorization contracts, synthetic policy only."""
from types import SimpleNamespace
import pytest
from gateway.config import Platform
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.session import SessionSource
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

buzz = load_plugin_adapter("buzz")


def runner():
    from gateway.run import GatewayRunner
    value = object.__new__(GatewayRunner)
    value.adapters = {}
    value._profile_adapters = {}
    value.pairing_store = None
    from gateway.config import GatewayConfig
    value.config = GatewayConfig()
    return value


def source(user="a" * 64, platform="runtime_live_auth", profile=None):
    return SessionSource(platform=Platform(platform), user_id=user, user_name=user,
                         chat_id="synthetic", chat_type="dm", profile=profile)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in ("BUZZ_ALLOWED_USERS", "BUZZ_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(name, raising=False)
    scope = platform_registry.current_scope_key()
    old = platform_registry.snapshot_registration("runtime_live_auth", scope=scope)
    yield
    current = platform_registry.snapshot_registration("runtime_live_auth", scope=scope)
    platform_registry.restore_registration("runtime_live_auth", current, old, scope=scope)


def test_registry_policy_is_consumed_before_pairing_and_grants():
    calls = []
    policy = {"allowed_users": ["canonical"], "allow_all_users": False}
    entry = PlatformEntry(name="runtime_live_auth", label="Synthetic", adapter_factory=lambda _: None, check_fn=lambda: True)
    # Dynamic attributes let the baseline reach a behavioral assertion (not an ABI/import error).
    entry.authorization_config_fn = lambda profile: calls.append(profile) or policy
    entry.authorization_user_normalizer = lambda user: "canonical" if user in {"alias", "canonical"} else None
    platform_registry.register(entry, scope=platform_registry.current_scope_key())
    r = runner()
    assert r._is_user_authorized(source("alias")) is True
    assert calls == [None]
    r.pairing_store = SimpleNamespace(is_approved=lambda *_: True)
    assert r._is_user_authorized(source("invalid")) is False
    policy["allowed_users"] = []
    r.pairing_store = None
    assert r._is_user_authorized(source("alias")) is False


def atomic_policy(home, policy):
    import json
    temporary = home / "next.yaml"
    temporary.write_text(json.dumps({"buzz": policy}))
    temporary.replace(home / "config.yaml")


@pytest.fixture
def registered_buzz(tmp_path, monkeypatch):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from gateway.config import PlatformConfig
    home = tmp_path / "owner"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Explicit .env keeps the test independent of aggregate secret-scope state.
    (home / ".env").write_text("")
    atomic_policy(home, {"allowed_users": ["a" * 64]})
    manager = PluginManager()
    buzz.register(PluginContext(PluginManifest(name="buzz"), manager))
    adapter = buzz.BuzzAdapter(PlatformConfig(enabled=True, extra={"allowed_users": ["a" * 64]}))
    r = runner()
    r.adapters = {adapter.platform: adapter}
    adapter.set_authorization_check(r._make_adapter_auth_check(adapter.platform))
    try:
        yield home, r, adapter
    finally:
        manager.unload()


def test_registered_buzz_live_revoke_empty_delete_without_reconstruction(registered_buzz):
    home, r, adapter = registered_buzz
    check = lambda user: adapter._is_sender_authorized(user, "dm", "synthetic")
    assert check("a" * 64) is True
    assert check("b" * 64) is False
    atomic_policy(home, {"allowed_users": ["b" * 64]})
    assert check("a" * 64) is False
    assert check("b" * 64) is True
    atomic_policy(home, {"allowed_users": []})
    assert check("b" * 64) is False
    atomic_policy(home, {"allow_all_users": True})
    assert check("b" * 64) is True
    assert check("not-a-key") is False
    (home / "config.yaml").unlink()
    assert check("a" * 64) is False
    assert check("b" * 64) is False


def test_transport_home_survives_routed_scope_and_same_name_home(registered_buzz, tmp_path, monkeypatch):
    import weakref
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    home, r, adapter = registered_buzz
    other = tmp_path / "different-root" / "profiles" / "same"
    other.mkdir(parents=True)
    atomic_policy(other, {"allow_all_users": True})
    (other / ".env").write_text("BUZZ_ALLOW_ALL_USERS=true\n")
    event = source(platform="buzz", profile="routed")
    event._transport_adapter_ref = weakref.ref(adapter)
    assert r._is_user_authorized(event) is True
    token = set_hermes_home_override(other)
    try:
        assert r._is_user_authorized(event) is True
        atomic_policy(home, {"allowed_users": []})
        assert r._is_user_authorized(event) is False
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("malformed", ["buzz: [", "buzz: {extra: []}"])
def test_exact_scope_last_good_then_delete_and_new_home(registered_buzz, tmp_path, malformed):
    from plugins.platforms.buzz import settings
    home, r, adapter = registered_buzz
    fn = getattr(platform_registry.get("buzz"), "authorization_config_fn", None)
    assert callable(fn)
    assert fn(home=home)["allowed_users"] == ["a" * 64]
    (home / "config.yaml").write_text(malformed)
    assert fn(home=home)["allowed_users"] == ["a" * 64]
    other = tmp_path / "same"
    other.mkdir()
    (other / "config.yaml").write_text(malformed)
    assert fn(home=other)["allowed_users"] == []
    (home / "config.yaml").unlink()
    assert fn(home=home)["allowed_users"] == []


def test_live_yaml_bridge_never_pins_authority(monkeypatch):
    import os
    monkeypatch.setattr(buzz, "_profile_scoped", lambda: False)
    buzz._apply_yaml_config({}, {"extra": {"allowed_users": ["a" * 64], "allow_all_users": True}})
    assert "BUZZ_ALLOWED_USERS" not in os.environ
    assert "BUZZ_ALLOW_ALL_USERS" not in os.environ


def test_global_wildcard_preserved_after_strict_sender_validation(registered_buzz, monkeypatch):
    home, r, adapter = registered_buzz
    atomic_policy(home, {"allowed_users": []})
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "*")
    assert r._is_user_authorized(source(platform="buzz")) is True
    assert r._is_user_authorized(source("invalid", platform="buzz")) is False


@pytest.mark.parametrize("bad", [["a"], "a", False, {"allowed_users": "a"}, {"allowed_users": [1]}, {"allow_all_users": 1}, {"unknown": True}])
def test_malformed_registry_policy_denies_even_pairing(bad):
    entry = PlatformEntry(name="runtime_live_auth", label="Synthetic", adapter_factory=lambda _: None, check_fn=lambda: True,
                          authorization_config_fn=lambda _: bad)
    platform_registry.register(entry, scope=platform_registry.current_scope_key())
    r = runner()
    r.pairing_store = SimpleNamespace(is_approved=lambda *_: True)
    assert r._is_user_authorized(source()) is False


def test_managed_alias_precedence_empty_env_and_expansion_retention(registered_buzz, tmp_path, monkeypatch):
    home, r, adapter = registered_buzz
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    atomic_policy(home, {"allowed_users": ["a" * 64]})
    (managed / "config.yaml").write_text('gateway:\n  platforms:\n    buzz:\n      extra:\n        allowed_users: ["' + "b" * 64 + '"]\n')
    check = lambda user: adapter._is_sender_authorized(user, "dm", "synthetic")
    assert check("a" * 64) is False
    assert check("b" * 64) is True
    (managed / ".env").write_text("BUZZ_ALLOWED_USERS=\n")
    assert check("b" * 64) is False
    (managed / ".env").unlink()
    (managed / "config.yaml").unlink()
    managed.rmdir()
    # Inherited managed disappearance exposes local policy; not universal fail-closed.
    assert check("a" * 64) is True
    atomic_policy(home, {"allowed_users": ["${LIVE_SYNTHETIC_USER}"]})
    (home / ".env").write_text("LIVE_SYNTHETIC_USER=" + "b" * 64 + "\n")
    assert check("b" * 64) is True
    (home / "config.yaml").write_text("buzz: [")
    (home / ".env").write_text("LIVE_SYNTHETIC_USER=" + "a" * 64 + "\n")
    # Expanded last-good retention during malformed YAML is inherited.
    assert check("b" * 64) is True
    assert check("a" * 64) is False
