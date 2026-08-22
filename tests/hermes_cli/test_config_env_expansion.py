"""Tests for ${ENV_VAR} substitution in config.yaml values."""

import pytest
from hermes_cli.config import _expand_env_vars, load_config


class TestExpandEnvVars:
    def test_simple_substitution(self):
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("MY_KEY", "secret123")
            assert _expand_env_vars("${MY_KEY}") == "secret123"




    def test_non_string_values_untouched(self):
        assert _expand_env_vars(42) == 42
        assert _expand_env_vars(3.14) == 3.14
        assert _expand_env_vars(True) is True
        assert _expand_env_vars(None) is None




class TestLoadConfigExpansion:
    def test_load_config_expands_env_vars(self, tmp_path, monkeypatch):
        config_yaml = (
            "model:\n"
            "  api_key: ${GOOGLE_API_KEY}\n"
            "platforms:\n"
            "  telegram:\n"
            "    token: ${TELEGRAM_BOT_TOKEN}\n"
            "plain: no-substitution\n"
        )
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        monkeypatch.setenv("GOOGLE_API_KEY", "gsk-test-key")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1234567:ABC-token")
        # Patch the imported function's own globals. Other tests may reload
        # hermes_cli.config, making string-target monkeypatches hit a different
        # module object than this collection-time imported load_config().
        monkeypatch.setitem(load_config.__globals__, "get_config_path", lambda: config_file)

        config = load_config()

        assert config["model"]["api_key"] == "gsk-test-key"
        assert config["platforms"]["telegram"]["token"] == "1234567:ABC-token"
        assert config["plain"] == "no-substitution"


class TestLoadConfigCacheEnvStaleness:
    """The load_config() cache must not pin expansions made against a stale
    environment (#58514): a load before load_hermes_dotenv() runs, or an env
    var rotated in-process, must not keep serving the old expansion."""

    def test_env_var_appearing_after_first_load_invalidates_cache(self, tmp_path, monkeypatch):
        config_yaml = "auxiliary:\n  vision:\n    api_key: ${LATE_DOTENV_KEY_58514}\n"
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        monkeypatch.delenv("LATE_DOTENV_KEY_58514", raising=False)
        monkeypatch.setitem(load_config.__globals__, "get_config_path", lambda: config_file)

        # First load happens before the var exists (pre-dotenv): literal kept.
        assert load_config()["auxiliary"]["vision"]["api_key"] == "${LATE_DOTENV_KEY_58514}"

        # .env load brings the var in — same file mtime/size, env changed.
        monkeypatch.setenv("LATE_DOTENV_KEY_58514", "nvapi-real")
        assert load_config()["auxiliary"]["vision"]["api_key"] == "nvapi-real"


    def test_unchanged_env_still_serves_cache(self, tmp_path, monkeypatch):
        config_yaml = "providers:\n  mistral:\n    api_key: ${STABLE_KEY_58514}\n"
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        monkeypatch.setenv("STABLE_KEY_58514", "key-stable")
        monkeypatch.setitem(load_config.__globals__, "get_config_path", lambda: config_file)

        load_config()
        # load_config_readonly() returns the cached object itself, so object
        # identity across calls proves the cache-hit path was taken (a rebuild
        # would produce a fresh dict).
        readonly = load_config.__globals__["load_config_readonly"]
        first = readonly()
        second = readonly()

        assert first is second
        assert first["providers"]["mistral"]["api_key"] == "key-stable"

    def test_explicit_environment_is_isolated_from_process_cache(
        self, tmp_path, monkeypatch
    ):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "values:\n"
            "  bare: ${PROFILE_REF_83647}\n"
            "  prefixed: ${env:PROFILE_REF_83647}\n"
        )
        monkeypatch.setenv("PROFILE_REF_83647", "process-b")
        monkeypatch.setitem(load_config.__globals__, "get_config_path", lambda: config_file)

        assert load_config(env={"PROFILE_REF_83647": "profile-a"})["values"] == {
            "bare": "profile-a",
            "prefixed": "profile-a",
        }
        assert load_config(env={})["values"] == {
            "bare": "${PROFILE_REF_83647}",
            "prefixed": "${env:PROFILE_REF_83647}",
        }
        assert load_config()["values"] == {
            "bare": "process-b",
            "prefixed": "process-b",
        }

    def test_explicit_environment_lkg_does_not_cross_profiles(
        self, tmp_path, monkeypatch
    ):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model:\n  default: ${PROFILE_LKG_83647}\n")
        monkeypatch.setenv("PROFILE_LKG_83647", "process-b")
        monkeypatch.setitem(load_config.__globals__, "get_config_path", lambda: config_file)

        assert load_config(env={"PROFILE_LKG_83647": "profile-a"})["model"]["default"] == (
            "profile-a"
        )
        config_file.write_text("model: [broken")

        assert load_config(env={"PROFILE_LKG_83647": "profile-a"})["model"]["default"] == (
            "profile-a"
        )
        assert "profile-a" not in str(
            load_config(env={"PROFILE_LKG_83647": "profile-b"})
        )

    def test_explicit_environment_keeps_managed_overlay_global(
        self, tmp_path, monkeypatch
    ):
        home = tmp_path / "home"
        managed = tmp_path / "managed"
        home.mkdir()
        managed.mkdir()
        config_file = home / "config.yaml"
        config_file.write_text(
            "model:\n"
            "  default: user-${SHARED_CONFIG_REF}\n"
            "auxiliary:\n"
            "  vision:\n"
            "    api_key: ${SHARED_CONFIG_REF}\n"
        )
        (managed / "config.yaml").write_text(
            "model:\n  default: managed-${SHARED_CONFIG_REF}\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
        monkeypatch.setenv("SHARED_CONFIG_REF", "process-b")
        monkeypatch.setitem(load_config.__globals__, "get_config_path", lambda: config_file)

        from hermes_cli import config as config_module, managed_scope

        config_module._LOAD_CONFIG_CACHE.clear()
        managed_scope.invalidate_managed_cache()
        loaded = load_config(env={"SHARED_CONFIG_REF": "profile-a"})

        assert loaded["model"]["default"] == "managed-process-b"
        assert loaded["auxiliary"]["vision"]["api_key"] == "profile-a"

    def test_explicit_environment_lkg_keeps_unresolved_managed_ref(
        self, tmp_path, monkeypatch
    ):
        home = tmp_path / "home"
        managed = tmp_path / "managed"
        home.mkdir()
        managed.mkdir()
        config_file = home / "config.yaml"
        config_file.write_text("model:\n  default: user-${MANAGED_LKG_REF_83647}\n")
        (managed / "config.yaml").write_text(
            "model:\n  default: managed-${MANAGED_LKG_REF_83647}\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
        monkeypatch.delenv("MANAGED_LKG_REF_83647", raising=False)
        monkeypatch.setitem(load_config.__globals__, "get_config_path", lambda: config_file)

        from hermes_cli import config as config_module, managed_scope

        config_module._LOAD_CONFIG_CACHE.clear()
        managed_scope.invalidate_managed_cache()
        profile_env = {"MANAGED_LKG_REF_83647": "profile-a"}

        assert load_config(env=profile_env)["model"]["default"] == (
            "managed-${MANAGED_LKG_REF_83647}"
        )
        config_file.write_text("model: [broken")

        assert load_config(env=profile_env)["model"]["default"] == (
            "managed-${MANAGED_LKG_REF_83647}"
        )

    def test_explicit_environment_lkg_refreshes_rotated_managed_ref(
        self, tmp_path, monkeypatch
    ):
        home = tmp_path / "home"
        managed = tmp_path / "managed"
        home.mkdir()
        managed.mkdir()
        config_file = home / "config.yaml"
        config_file.write_text(
            "model:\n  default: user-${MANAGED_LKG_ROTATE_REF_83647}\n"
        )
        (managed / "config.yaml").write_text(
            "model:\n  default: managed-${MANAGED_LKG_ROTATE_REF_83647}\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
        monkeypatch.setenv("MANAGED_LKG_ROTATE_REF_83647", "process-a")
        monkeypatch.setitem(load_config.__globals__, "get_config_path", lambda: config_file)

        from hermes_cli import config as config_module, managed_scope

        config_module._LOAD_CONFIG_CACHE.clear()
        managed_scope.invalidate_managed_cache()
        profile_env = {"MANAGED_LKG_ROTATE_REF_83647": "profile-value"}

        assert load_config(env=profile_env)["model"]["default"] == "managed-process-a"
        monkeypatch.setenv("MANAGED_LKG_ROTATE_REF_83647", "process-b")
        config_file.write_text("model: [broken")

        loaded = load_config(env=profile_env)
        assert loaded["model"]["default"] == "managed-process-b"
        assert "process-a" not in str(loaded)
        assert "profile-value" not in str(loaded)

    def test_explicit_environment_lkg_tracks_only_referenced_values(
        self, tmp_path, monkeypatch
    ):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model:\n  default: ${PROFILE_REF_83647}\n")
        monkeypatch.setitem(load_config.__globals__, "get_config_path", lambda: config_file)

        from hermes_cli import config as config_module

        env = {
            "PROFILE_REF_83647": "profile-a",
            "OPENVIKING_API_KEY": "unreferenced-openviking-sentinel",
        }
        load_config(env=env)

        user_snapshot, managed_snapshot, expanded = (
            config_module._LAST_EXPANDED_CONFIG_BY_EXPLICIT_ENV[str(config_file)]
        )
        assert user_snapshot == {"PROFILE_REF_83647": "profile-a"}
        assert managed_snapshot == {}
        assert "OPENVIKING_API_KEY" not in user_snapshot
        assert "unreferenced-openviking-sentinel" not in repr(
            (user_snapshot, managed_snapshot, expanded)
        )


class TestLoadCliConfigExpansion:
    """Verify that load_cli_config() also expands ${VAR} references."""

    def test_cli_config_ignores_empty_terminal_section(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("terminal:\n")

        monkeypatch.setattr("cli._hermes_home", tmp_path)

        from cli import load_cli_config
        config = load_cli_config()

        assert isinstance(config["terminal"], dict)
        assert config["terminal"]["env_type"] == "local"


    def test_cli_config_unresolved_kept_verbatim(self, tmp_path, monkeypatch):
        config_yaml = (
            "auxiliary:\n"
            "  vision:\n"
            "    api_key: ${UNSET_CLI_VAR_ABC}\n"
        )
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        monkeypatch.delenv("UNSET_CLI_VAR_ABC", raising=False)
        monkeypatch.setattr("cli._hermes_home", tmp_path)

        from cli import load_cli_config
        config = load_cli_config()

        assert config["auxiliary"]["vision"]["api_key"] == "${UNSET_CLI_VAR_ABC}"
