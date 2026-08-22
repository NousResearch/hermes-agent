"""Live, profile-scoped policy contracts for the Buzz plugin."""

import builtins
from importlib import import_module, util
import logging
import os
import sys
import threading
from types import SimpleNamespace
import weakref

import pytest
import yaml

from plugins.platforms.buzz import settings


VALID_NPUB = "npub1nl2u0wnd8mezfknc74q7pl9ec58h9nrrakce4tnk434qgaxl4psqe5twr6"


def test_policy_defaults_fail_closed():
    spec = util.find_spec("plugins.platforms.buzz.settings")

    assert spec is not None
    settings = import_module("plugins.platforms.buzz.settings")
    assert settings.policy_from_config({}) == {
        "allowed_users": [],
        "allow_all_users": False,
        "require_mention": True,
        "thread_require_mention": True,
    }


def test_identity_normalization_accepts_only_exact_hex_or_valid_npub():
    assert settings.normalize_user_ref("A" * 64) == "a" * 64
    assert settings.normalize_user_ref(VALID_NPUB) == (
        "9fd5c7ba6d3ef224da78f541e0fcb9c50f72cc63edb19aae76ac6a0474dfa860"
    )
    for invalid in ("", "a" * 63, "g" * 64, VALID_NPUB[:-1] + "x", "nprofile1abc"):
        assert settings.normalize_user_ref(invalid) is None

    with pytest.raises(ValueError, match="invalid public key"):
        settings.policy_from_config({
            "gateway": {
                "platforms": {"buzz": {"extra": {"allowed_users": ["not-a-key"]}}}
            }
        })


def test_canonical_and_legacy_policy_precedence_matches_buzz_loading():
    config = {
        "gateway": {
            "platforms": {
                "buzz": {
                    "allow_all_users": True,
                    "extra": {
                        "allowed_users": ["a" * 64],
                        "allow_all_users": False,
                        "require_mention": True,
                    },
                }
            },
            "buzz": {
                "require_mention": True,
                "extra": {"require_mention": False},
            },
        },
        "platforms": {
            "buzz": {
                "thread_require_mention": True,
                "extra": {"thread_require_mention": False},
            }
        },
        "buzz": {
            "allow_all_users": False,
            "extra": {"allow_all_users": True},
        },
    }

    assert settings.policy_from_config(config) == {
        "allowed_users": ["a" * 64],
        "allow_all_users": True,
        "require_mention": False,
        "thread_require_mention": False,
    }

    with pytest.raises(ValueError, match="must be a boolean"):
        settings.policy_from_config({
            "gateway": {"platforms": {"buzz": {"allow_all_users": "maybe"}}}
        })


def _write_policy(home, **values):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"gateway": {"platforms": {"buzz": {"extra": values}}}}),
        encoding="utf-8",
    )


def test_explicit_profiles_resolve_correct_homes_and_stay_isolated(
    monkeypatch, tmp_path
):
    root = tmp_path / "hermes"
    open_home = root / "profiles" / "open"
    locked_home = root / "profiles" / "locked"
    _write_policy(open_home, allow_all_users=True, require_mention=False)
    _write_policy(locked_home, allow_all_users=False, require_mention=True)
    monkeypatch.setenv("HERMES_HOME", str(root))

    assert settings.effective_runtime_policy("open")["allow_all_users"] is True
    assert settings.effective_runtime_policy("locked")["allow_all_users"] is False
    assert settings.effective_runtime_policy("open")["require_mention"] is False
    assert settings.effective_runtime_policy("locked")["require_mention"] is True


def test_managed_config_overlay_is_authoritative(monkeypatch, tmp_path):
    root = tmp_path / "hermes"
    profile_home = root / "profiles" / "work"
    managed = tmp_path / "managed"
    _write_policy(profile_home, allow_all_users=True, require_mention=False)
    _write_policy(managed, allow_all_users=False, require_mention=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))

    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    assert settings.effective_runtime_policy("work") == {
        "allowed_users": [],
        "allow_all_users": False,
        "require_mention": True,
        "thread_require_mention": True,
    }


def test_malformed_user_yaml_retains_exact_profile_last_valid_or_fails_closed(
    monkeypatch, tmp_path
):
    root = tmp_path / "hermes"
    safe_home = root / "profiles" / "safe"
    cold_home = root / "profiles" / "cold"
    _write_policy(safe_home, allow_all_users=True, require_mention=False)
    cold_home.mkdir(parents=True)
    (cold_home / "config.yaml").write_text("gateway: [\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    loader = settings.RuntimePolicyLoader()

    assert loader.load("safe")["allow_all_users"] is True
    (safe_home / "config.yaml").write_text("gateway: [\n", encoding="utf-8")
    assert loader.load("safe")["allow_all_users"] is True
    assert loader.load("safe")["require_mention"] is False
    assert loader.load("cold") == settings.policy_from_config({})


def test_malformed_managed_yaml_never_falls_through_to_permissive_user_policy(
    monkeypatch, tmp_path
):
    root = tmp_path / "hermes"
    warm_home = root / "profiles" / "warm"
    cold_home = root / "profiles" / "cold-managed"
    managed = tmp_path / "managed"
    _write_policy(warm_home, allow_all_users=True, require_mention=False)
    _write_policy(cold_home, allow_all_users=True, require_mention=False)
    _write_policy(managed, allow_all_users=False, require_mention=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    loader = settings.RuntimePolicyLoader()
    assert loader.load("warm")["allow_all_users"] is False

    (managed / "config.yaml").write_text("gateway: [\n", encoding="utf-8")
    managed_scope.invalidate_managed_cache()
    assert loader.load("warm")["allow_all_users"] is False
    assert loader.load("warm")["require_mention"] is True
    assert loader.load("cold-managed") == settings.policy_from_config({})


def test_live_user_and_managed_changes_reload_and_key_deletion_revokes(
    monkeypatch, tmp_path
):
    root = tmp_path / "hermes"
    profile_home = root / "profiles" / "work"
    managed = tmp_path / "managed"
    _write_policy(profile_home, allow_all_users=True, require_mention=False)
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    loader = settings.RuntimePolicyLoader()

    assert loader.load("work")["allow_all_users"] is True
    _write_policy(profile_home, allow_all_users=False, require_mention=True)
    assert loader.load("work")["allow_all_users"] is False

    _write_policy(managed, allow_all_users=True, require_mention=False)
    assert loader.load("work")["allow_all_users"] is True
    assert loader.load("work")["require_mention"] is False

    (managed / "config.yaml").write_text("{}\n", encoding="utf-8")
    (profile_home / "config.yaml").write_text(
        "gateway:\n  platforms:\n    buzz:\n      extra: {}\n", encoding="utf-8"
    )
    assert loader.load("work") == settings.policy_from_config({})


def test_environment_overrides_have_presence_precedence_and_fail_closed():
    base = settings.policy_from_config({
        "buzz": {
            "extra": {
                "allowed_users": ["a" * 64],
                "allow_all_users": True,
                "require_mention": False,
                "thread_require_mention": False,
            }
        }
    })
    values = {
        "BUZZ_ALLOWED_USERS": "b" * 64,
        "BUZZ_ALLOW_ALL_USERS": "false",
        "BUZZ_REQUIRE_MENTION": "true",
        "BUZZ_THREAD_REQUIRE_MENTION": "true",
    }
    assert settings.with_environment_overrides(base, values.get) == {
        "allowed_users": ["b" * 64],
        "allow_all_users": False,
        "require_mention": True,
        "thread_require_mention": True,
    }

    values["BUZZ_ALLOWED_USERS"] = ""
    values["BUZZ_ALLOW_ALL_USERS"] = ""
    effective = settings.with_environment_overrides(base, values.get)
    assert effective["allowed_users"] == []
    assert effective["allow_all_users"] is False

    values["BUZZ_ALLOWED_USERS"] = "not-a-key"
    assert settings.effective_policy_with_environment(base, values.get) == (
        settings.policy_from_config({})
    )


def test_named_profile_environment_is_isolated_and_introspection_has_names_only(
    monkeypatch, tmp_path
):
    root = tmp_path / "hermes"
    open_home = root / "profiles" / "open"
    closed_home = root / "profiles" / "closed"
    _write_policy(open_home, allow_all_users=False)
    _write_policy(closed_home, allow_all_users=False)
    (open_home / ".env").write_text(
        "BUZZ_ALLOW_ALL_USERS=true\nBUZZ_REQUIRE_MENTION=false\n",
        encoding="utf-8",
    )
    (closed_home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    assert settings.effective_runtime_policy("open")["allow_all_users"] is True
    assert settings.effective_runtime_policy("closed")["allow_all_users"] is False
    fields = settings.environment_override_fields("open")
    assert fields == {"allow_all_users", "require_mention"}
    assert "true" not in repr(fields).lower()
    assert str(open_home) not in repr(fields)


def test_managed_environment_dominates_named_profile_policy(monkeypatch, tmp_path):
    root = tmp_path / "hermes"
    profile_home = root / "profiles" / "work"
    managed = tmp_path / "managed"
    _write_policy(profile_home, allow_all_users=False, require_mention=True)
    (profile_home / ".env").write_text(
        "BUZZ_ALLOW_ALL_USERS=true\nBUZZ_REQUIRE_MENTION=false\n",
        encoding="utf-8",
    )
    managed.mkdir()
    (managed / ".env").write_text(
        "BUZZ_ALLOW_ALL_USERS=false\nBUZZ_REQUIRE_MENTION=true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    assert settings.effective_runtime_policy("work")["allow_all_users"] is False
    assert settings.effective_runtime_policy("work")["require_mention"] is True
    assert settings.environment_override_fields("work") == {
        "allow_all_users",
        "require_mention",
    }


@pytest.mark.parametrize(
    "managed_bytes",
    [b"BUZZ_ALLOW_ALL_USERS=true\nmalformed\n", b"BUZZ_ALLOW_ALL_USERS=\xff\n"],
    ids=["malformed", "invalid-utf8"],
)
def test_invalid_managed_environment_fails_closed_without_profile_fallback(
    monkeypatch, tmp_path, managed_bytes
):
    root = tmp_path / "hermes"
    profile_home = root / "profiles" / "work"
    managed = tmp_path / "managed"
    _write_policy(profile_home, allow_all_users=True, require_mention=False)
    (profile_home / ".env").write_text(
        "BUZZ_ALLOW_ALL_USERS=true\nBUZZ_REQUIRE_MENTION=false\n",
        encoding="utf-8",
    )
    managed.mkdir()
    (managed / ".env").write_bytes(managed_bytes)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))

    assert settings.effective_runtime_policy("work") == settings.policy_from_config({})
    metadata = settings.environment_override_fields("work")
    assert metadata == set()
    assert metadata.error == "environment_unavailable"
    assert str(managed) not in repr(metadata)


def test_unreadable_managed_environment_fails_closed_without_profile_fallback(
    monkeypatch, tmp_path
):
    root = tmp_path / "hermes"
    profile_home = root / "profiles" / "work"
    managed = tmp_path / "managed"
    _write_policy(profile_home, allow_all_users=True)
    (profile_home / ".env").write_text("BUZZ_ALLOW_ALL_USERS=true\n", encoding="utf-8")
    managed.mkdir()
    managed_env = managed / ".env"
    managed_env.write_text("BUZZ_ALLOW_ALL_USERS=false\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    original_read_text = settings.Path.read_text

    def deny_managed_env(path, *args, **kwargs):
        if path == managed_env:
            raise PermissionError("denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(settings.Path, "read_text", deny_managed_env)

    assert settings.effective_runtime_policy("work") == settings.policy_from_config({})
    assert settings.environment_override_fields("work").error == (
        "environment_unavailable"
    )


@pytest.mark.parametrize(
    "failure",
    ["import", "get", "unscoped", "membership", "value"],
)
def test_current_secret_scope_acquisition_failures_revoke_permissive_policy(
    monkeypatch, tmp_path, failure
):
    from agent import secret_scope

    root = tmp_path / "hermes"
    _write_policy(root, allow_all_users=True, require_mention=False)
    monkeypatch.setenv("HERMES_HOME", str(root))
    token = None

    if failure == "import":
        original_import = builtins.__import__

        def fail_secret_scope_import(name, *args, **kwargs):
            if name == "agent.secret_scope":
                raise ImportError("unavailable")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_secret_scope_import)
    elif failure == "get":
        monkeypatch.setattr(
            secret_scope,
            "get_secret",
            lambda _name: (_ for _ in ()).throw(RuntimeError()),
        )
    elif failure == "unscoped":
        monkeypatch.setattr(
            secret_scope,
            "get_secret",
            lambda _name: (_ for _ in ()).throw(secret_scope.UnscopedSecretError()),
        )
    elif failure == "membership":

        class BrokenScope(dict):
            def get(self, key, default=None):
                raise RuntimeError("membership unavailable")

        token = secret_scope.set_secret_scope(BrokenScope())
    else:

        class BrokenValue:
            def __str__(self):
                raise RuntimeError("value unavailable")

        monkeypatch.setattr(secret_scope, "get_secret", lambda _name: BrokenValue())

    try:
        assert settings.effective_runtime_policy() == settings.policy_from_config({})
        metadata = settings.environment_override_fields()
        assert metadata == set()
        assert metadata.error == "environment_unavailable"
    finally:
        if token is not None:
            secret_scope.reset_secret_scope(token)


@pytest.mark.parametrize(
    "reference", ["${BUZZ_POLICY_FLAG}", "${env:BUZZ_POLICY_FLAG}"]
)
def test_managed_policy_refs_expand_once_through_managed_environment(
    monkeypatch, tmp_path, reference
):
    root = tmp_path / "hermes"
    profile_home = root / "profiles" / "work"
    managed = tmp_path / "managed"
    _write_policy(profile_home, allow_all_users=True)
    (profile_home / ".env").write_text("BUZZ_POLICY_FLAG=true\n", encoding="utf-8")
    _write_policy(managed, allow_all_users=reference)
    (managed / ".env").write_text("BUZZ_POLICY_FLAG=false\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("BUZZ_POLICY_FLAG", "true")

    assert settings.RuntimePolicyLoader().load("work")["allow_all_users"] is False


@pytest.mark.parametrize("managed_value", [None, ""])
def test_missing_or_empty_managed_policy_ref_fails_closed_without_process_leak(
    monkeypatch, tmp_path, managed_value
):
    root = tmp_path / "hermes"
    profile_home = root / "profiles" / "work"
    managed = tmp_path / "managed"
    _write_policy(profile_home, allow_all_users=False)
    (profile_home / ".env").write_text("", encoding="utf-8")
    _write_policy(managed, allow_all_users="${env:BUZZ_POLICY_FLAG}")
    if managed_value is not None:
        (managed / ".env").write_text(
            f"BUZZ_POLICY_FLAG={managed_value}\n", encoding="utf-8"
        )
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("BUZZ_POLICY_FLAG", "true")

    loader = settings.RuntimePolicyLoader()
    assert loader.load("work") == settings.policy_from_config({})
    (managed / "config.yaml").write_text("gateway: [\n", encoding="utf-8")
    assert loader.load("work") == settings.policy_from_config({})


@pytest.mark.parametrize(
    "reference", ["${BUZZ_POLICY_FLAG}", "${env:BUZZ_POLICY_FLAG}"]
)
def test_managed_policy_ref_wins_over_process_environment_in_multiplex(
    monkeypatch, tmp_path, reference
):
    from agent import secret_scope

    root = tmp_path / "hermes"
    managed = tmp_path / "managed"
    _write_policy(root, allow_all_users=True)
    _write_policy(managed, allow_all_users=reference)
    (managed / ".env").write_text("BUZZ_POLICY_FLAG=false\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("BUZZ_POLICY_FLAG", "true")
    secret_scope.set_multiplex_active(True)
    try:
        assert settings.RuntimePolicyLoader().load()["allow_all_users"] is False
    finally:
        secret_scope.set_multiplex_active(False)


def test_runtime_policy_cache_serializes_validation_and_last_valid_publication(
    monkeypatch, tmp_path
):
    root = tmp_path / "hermes"
    _write_policy(root, allow_all_users="${BUZZ_POLICY_FLAG}")
    monkeypatch.setenv("HERMES_HOME", str(root))
    loader = settings.RuntimePolicyLoader()
    stale_blocked = threading.Event()
    second_attempting = threading.Event()
    second_done = threading.Event()
    release_stale = threading.Event()

    def controlled_getter(_profile):
        if threading.current_thread().name == "stale-policy-read":

            def stale_value(_name):
                stale_blocked.set()
                assert release_stale.wait(2)
                return "true"

            return stale_value
        return lambda _name: None

    monkeypatch.setattr(settings, "_environment_getter", controlled_getter)

    stale_result = []
    restrictive_result = []
    stale = threading.Thread(
        target=lambda: stale_result.append(loader.load()), name="stale-policy-read"
    )

    def load_restrictive():
        second_attempting.set()
        restrictive_result.append(loader.load())
        second_done.set()

    restrictive = threading.Thread(target=load_restrictive)
    stale.start()
    assert stale_blocked.wait(2)
    restrictive.start()
    assert second_attempting.wait(2)
    assert not second_done.wait(0.1)
    release_stale.set()
    stale.join(2)
    restrictive.join(2)

    assert stale_result[0]["allow_all_users"] is True
    assert restrictive_result[0] == settings.policy_from_config({})
    (root / "config.yaml").write_text("gateway: [\n", encoding="utf-8")
    assert loader.load() == settings.policy_from_config({})


def test_user_policy_env_ref_restricts_and_tracks_profile_value_rotation(
    monkeypatch, tmp_path
):
    root = tmp_path / "hermes"
    profile_home = root / "profiles" / "work"
    _write_policy(profile_home, allow_all_users=True)
    (profile_home / ".env").write_text("BUZZ_POLICY_FLAG=false\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("BUZZ_POLICY_FLAG", "true")
    loader = settings.RuntimePolicyLoader()

    assert loader.load("work")["allow_all_users"] is True
    _write_policy(profile_home, allow_all_users="${BUZZ_POLICY_FLAG}")
    assert loader.load("work")["allow_all_users"] is False

    (profile_home / ".env").write_text("BUZZ_POLICY_FLAG=true\n", encoding="utf-8")
    assert loader.load("work")["allow_all_users"] is True


@pytest.mark.parametrize("scoped_env", [None, ""])
def test_missing_or_empty_user_policy_env_ref_revokes_last_permissive_policy(
    monkeypatch, tmp_path, scoped_env
):
    root = tmp_path / "hermes"
    profile_home = root / "profiles" / "work"
    _write_policy(profile_home, allow_all_users=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    loader = settings.RuntimePolicyLoader()
    assert loader.load("work")["allow_all_users"] is True

    _write_policy(profile_home, allow_all_users="${env:BUZZ_POLICY_FLAG}")
    env_text = "" if scoped_env is None else f"BUZZ_POLICY_FLAG={scoped_env}\n"
    (profile_home / ".env").write_text(env_text, encoding="utf-8")
    monkeypatch.setenv("BUZZ_POLICY_FLAG", "true")

    assert loader.load("work") == settings.policy_from_config({})
    (profile_home / "config.yaml").write_text("gateway: [\n", encoding="utf-8")
    assert loader.load("work") == settings.policy_from_config({})


def test_same_user_policy_text_uses_each_exact_profile_environment(
    monkeypatch, tmp_path
):
    root = tmp_path / "hermes"
    open_home = root / "profiles" / "open"
    closed_home = root / "profiles" / "closed"
    for home, value in ((open_home, "true"), (closed_home, "false")):
        _write_policy(home, allow_all_users="${env:BUZZ_POLICY_FLAG}")
        (home / ".env").write_text(f"BUZZ_POLICY_FLAG={value}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("BUZZ_POLICY_FLAG", "true")

    assert settings.effective_runtime_policy("open")["allow_all_users"] is True
    assert settings.effective_runtime_policy("closed")["allow_all_users"] is False


def test_unscoped_user_policy_env_ref_fails_closed_in_multiplex(monkeypatch, tmp_path):
    from agent import secret_scope

    root = tmp_path / "hermes"
    _write_policy(root, allow_all_users="${BUZZ_POLICY_FLAG}")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("BUZZ_POLICY_FLAG", "true")
    secret_scope.set_multiplex_active(True)
    try:
        assert settings.RuntimePolicyLoader().load()["allow_all_users"] is False
    finally:
        secret_scope.set_multiplex_active(False)


def test_single_profile_user_policy_env_ref_keeps_process_fallback(
    monkeypatch, tmp_path
):
    root = tmp_path / "hermes"
    _write_policy(root, allow_all_users="${BUZZ_POLICY_FLAG}")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("BUZZ_POLICY_FLAG", "true")

    assert settings.RuntimePolicyLoader().load()["allow_all_users"] is True


def test_single_profile_exact_dotenv_precedes_process_fallback(monkeypatch, tmp_path):
    root = tmp_path / "hermes"
    _write_policy(root, allow_all_users=True)
    (root / ".env").write_text("BUZZ_POLICY_FLAG=false\n", encoding="utf-8")
    _write_policy(root, allow_all_users="${BUZZ_POLICY_FLAG}")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("BUZZ_POLICY_FLAG", "true")

    assert settings.RuntimePolicyLoader().load()["allow_all_users"] is False


def test_multiplex_unscoped_policy_never_reads_process_environment(monkeypatch):
    from agent.secret_scope import set_multiplex_active

    monkeypatch.setenv("BUZZ_ALLOW_ALL_USERS", "true")
    set_multiplex_active(True)
    try:
        assert settings.effective_runtime_policy()["allow_all_users"] is False
        assert "allow_all_users" not in settings.environment_override_fields()
    finally:
        set_multiplex_active(False)


def test_registration_drives_live_central_authorization_in_transport_profile(
    monkeypatch, tmp_path
):
    from gateway.authz_mixin import GatewayAuthorizationMixin
    from gateway.config import Platform
    from gateway.platform_registry import PlatformEntry, platform_registry
    from gateway.session import SessionSource
    from plugins.platforms.buzz import adapter as buzz_adapter

    root = tmp_path / "hermes"
    owner_a = root / "profiles" / "owner-a"
    owner_b = root / "profiles" / "owner-b"
    _write_policy(owner_a, allowed_users=["a" * 64])
    _write_policy(owner_b, allowed_users=["b" * 64])
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setitem(
        sys.modules,
        "gateway.run",
        SimpleNamespace(logger=logging.getLogger("gateway.run")),
    )

    class Runner(GatewayAuthorizationMixin):
        def __init__(self):
            self.adapters = {}
            self._profile_adapters = {}
            self.pairing_store = None
            self.pairing_stores = {}
            self.config = SimpleNamespace(platforms={})

    captured = {}

    class Context:
        def register_platform(self, **kwargs):
            captured.update(kwargs)

    buzz_adapter.register(Context())
    assert captured["authorization_config_fn"] is settings.effective_authorization_policy
    assert captured["authorization_user_normalizer"] is settings.normalize_user_ref
    assert settings.effective_authorization_policy("owner-a") == {
        "allowed_users": ["a" * 64],
        "allow_all_users": False,
    }

    scope = platform_registry.current_scope_key()
    platform_registry.unregister("buzz", scope=scope)
    platform_registry.register(
        PlatformEntry(source="plugin", **captured),
        scope=scope,
    )
    try:
        runner = Runner()

        class Adapter:
            pass

        adapter_a = Adapter()
        adapter_b = Adapter()
        runner._profile_adapters = {
            "owner-a": {Platform("buzz"): adapter_a},
            "owner-b": {Platform("buzz"): adapter_b},
        }

        def source(user_id, adapter):
            inbound = SessionSource(
                platform=Platform("buzz"),
                user_id=user_id,
                chat_id="shared-channel",
                user_name=user_id,
                chat_type="group",
                profile="routed-runtime",
            )
            setattr(inbound, "_transport_adapter_ref", weakref.ref(adapter))
            return inbound

        sender_a = source("a" * 64, adapter_a)
        sender_b = source("a" * 64, adapter_b)
        assert runner._is_user_authorized(sender_a) is True
        assert runner._is_user_authorized(sender_b) is False

        replacement = owner_a / "config.yaml.next"
        replacement.write_text(
            yaml.safe_dump({
                "gateway": {
                    "platforms": {"buzz": {"extra": {"allowed_users": ["c" * 64]}}}
                }
            }),
            encoding="utf-8",
        )
        replacement.replace(owner_a / "config.yaml")

        assert runner._is_user_authorized(sender_a) is False
        assert runner._is_user_authorized(source("c" * 64, adapter_a)) is True
    finally:
        platform_registry.unregister("buzz", scope=scope)


def test_yaml_bridge_keeps_transport_but_never_snapshots_live_policy(monkeypatch):
    from plugins.platforms.buzz import adapter as buzz_adapter

    policy_env = (
        "BUZZ_ALLOWED_USERS",
        "BUZZ_ALLOW_ALL_USERS",
        "BUZZ_REQUIRE_MENTION",
        "BUZZ_THREAD_REQUIRE_MENTION",
    )
    transport_env = (
        "BUZZ_RELAY_URL",
        "BUZZ_CHANNELS",
        "BUZZ_HOME_CHANNEL",
        "BUZZ_POLL_INTERVAL",
        "BUZZ_CLI_PATH",
        "BUZZ_TRANSPORT",
    )
    for name in policy_env + transport_env:
        monkeypatch.delenv(name, raising=False)

    buzz_adapter._apply_yaml_config(
        {},
        {
            "extra": {
                "relay_url": "https://relay.invalid",
                "channels": ["general", "ops"],
                "home_channel": "general",
                "poll_interval": 7,
                "cli_path": "/opt/buzz",
                "transport": "poll",
                "allowed_users": ["a" * 64],
                "allow_all_users": True,
                "require_mention": False,
                "thread_require_mention": False,
            }
        },
    )

    assert {name for name in policy_env if name in os.environ} == set()
    assert os.environ["BUZZ_RELAY_URL"] == "https://relay.invalid"
    assert os.environ["BUZZ_CHANNELS"] == "general,ops"
    assert os.environ["BUZZ_HOME_CHANNEL"] == "general"
    assert os.environ["BUZZ_POLL_INTERVAL"] == "7"
    assert os.environ["BUZZ_CLI_PATH"] == "/opt/buzz"
    assert os.environ["BUZZ_TRANSPORT"] == "poll"
