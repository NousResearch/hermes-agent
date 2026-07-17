"""Tests for hermes_cli/fallback_config.py — fallback config resolution."""

import threading
import time

from hermes_cli import fallback_config
from hermes_cli.fallback_config import resolve_entry_api_key


class TestResolveEntryApiKey:
    def test_inline_api_key_wins(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        entry = {"provider": "custom", "api_key": "inline-key", "key_env": "FB_KEY"}
        assert resolve_entry_api_key(entry) == "inline-key"

    def test_key_env_resolves_from_environment(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        assert resolve_entry_api_key({"key_env": "FB_KEY"}) == "env-key"

    def test_api_key_env_alias(self, monkeypatch):
        monkeypatch.setenv("FB_ALIAS_KEY", "alias-key")
        assert resolve_entry_api_key({"api_key_env": "FB_ALIAS_KEY"}) == "alias-key"

    def test_unset_env_var_returns_none(self, monkeypatch):
        monkeypatch.delenv("FB_MISSING", raising=False)
        # None (not "") lets resolve_runtime_provider fall through to the
        # provider's standard credential resolution.
        assert resolve_entry_api_key({"key_env": "FB_MISSING"}) is None

    def test_empty_env_var_returns_none(self, monkeypatch):
        monkeypatch.setenv("FB_EMPTY", "   ")
        assert resolve_entry_api_key({"key_env": "FB_EMPTY"}) is None

    def test_no_key_fields_returns_none(self):
        assert resolve_entry_api_key({"provider": "openrouter", "model": "glm"}) is None

    def test_non_dict_returns_none(self):
        assert resolve_entry_api_key(None) is None
        assert resolve_entry_api_key("nope") is None  # type: ignore[arg-type]

    def test_whitespace_inline_key_falls_through_to_env(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        entry = {"api_key": "   ", "key_env": "FB_KEY"}
        assert resolve_entry_api_key(entry) == "env-key"

    def test_key_env_uses_active_profile_secret_scope(self, monkeypatch):
        from agent.secret_scope import reset_secret_scope, set_secret_scope

        monkeypatch.setenv("FB_KEY", "default-profile-key")
        token = set_secret_scope({"FB_KEY": "secondary-profile-key"})
        try:
            assert resolve_entry_api_key({"key_env": "FB_KEY"}) == (
                "secondary-profile-key"
            )
        finally:
            reset_secret_scope(token)

    def test_key_env_does_not_fall_through_outside_active_profile_scope(
        self, monkeypatch
    ):
        from agent.secret_scope import reset_secret_scope, set_secret_scope

        monkeypatch.setenv("FB_KEY", "default-profile-key")
        token = set_secret_scope({})
        try:
            assert resolve_entry_api_key({"key_env": "FB_KEY"}) is None
        finally:
            reset_secret_scope(token)


def test_strict_loader_serializes_yaml_parsing(tmp_path, monkeypatch):
    """Concurrent strict refreshes must not enter libyaml simultaneously."""
    from hermes_cli import config, managed_scope

    config_path = tmp_path / "config.yaml"
    config_path.write_text("fallback_providers: []\n", encoding="utf-8")
    monkeypatch.setattr(config, "get_config_path", lambda: config_path)
    monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: None)

    state = {"active": 0, "max_active": 0}
    state_lock = threading.Lock()

    def slow_safe_load(_stream):
        with state_lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        try:
            time.sleep(0.05)
            return {"fallback_providers": []}
        finally:
            with state_lock:
                state["active"] -= 1

    monkeypatch.setattr(fallback_config.yaml, "safe_load", slow_safe_load)

    start = threading.Barrier(3)
    results = []

    def load():
        start.wait()
        results.append(fallback_config.load_fallback_chain_strict())

    threads = [threading.Thread(target=load) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=2)

    assert not any(thread.is_alive() for thread in threads)
    assert results == [[], []]
    assert state["max_active"] == 1


def test_strict_loader_expands_fallback_secrets_from_active_profile(
    tmp_path, monkeypatch
):
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from hermes_cli import config, managed_scope

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "fallback_providers:\n"
        "  - provider: custom:secondary\n"
        "    model: secondary-model\n"
        "    api_key: ${FB_KEY}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "get_config_path", lambda: config_path)
    monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: None)
    monkeypatch.setenv("FB_KEY", "default-profile-key")

    token = set_secret_scope({"FB_KEY": "secondary-profile-key"})
    try:
        assert fallback_config.load_fallback_chain_strict()[0]["api_key"] == (
            "secondary-profile-key"
        )
    finally:
        reset_secret_scope(token)


def test_profile_env_expansion_preserves_explicit_empty_value():
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    token = set_secret_scope({"EMPTY_KEY": ""})
    try:
        assert fallback_config._expand_profile_env_vars("${EMPTY_KEY}") == ""
    finally:
        reset_secret_scope(token)
