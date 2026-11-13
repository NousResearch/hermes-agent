"""Tests for ${ENV_VAR} substitution in config.yaml values."""

from hermes_cli.config import _expand_env_vars, load_config


class TestExpandEnvVars:


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


class TestExpansionUnderProfileScope:
    """``${VAR}`` refs must resolve against the active profile's secret scope,
    not the shared process environment (#84079): under multiplex every
    secondary profile otherwise "had" the default profile's token and fanned
    out.  Outside multiplex the scope is an overlay and environ still applies."""

    def test_scoped_ref_never_reads_another_profiles_environ(self, monkeypatch):
        from agent import secret_scope as ss

        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "default-token")
        was_active = ss.is_multiplex_active()
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"OTHER_KEY": "x"})  # profile-b: no matrix token
        try:
            assert _expand_env_vars("${MATRIX_ACCESS_TOKEN}") == "${MATRIX_ACCESS_TOKEN}"
            assert _expand_env_vars("${env:MATRIX_ACCESS_TOKEN}") == "${env:MATRIX_ACCESS_TOKEN}"
        finally:
            ss.reset_secret_scope(token)
        token = ss.set_secret_scope({"MATRIX_ACCESS_TOKEN": "c-token"})
        try:
            assert _expand_env_vars("${MATRIX_ACCESS_TOKEN}") == "c-token"
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(was_active)
        # Unscoped (default profile / single-profile CLI): legacy environ read.
        assert _expand_env_vars("${MATRIX_ACCESS_TOKEN}") == "default-token"

    def test_override_home_without_scope_takes_that_homes_scope_values(self, tmp_path, monkeypatch):
        """A record routed to a served profile's log is formatted under that profile's HERMES_HOME
        override with no secret scope bound. A ref its scope holds answers as the scope would: a key
        only the profile holds resolves, the administrator-managed .env beats the profile's value,
        and a global name such as PATH still comes from the process env. A name the profile does not
        hold falls back to the process env, as it did without the override."""
        from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
        from hermes_cli import managed_scope
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        home = tmp_path / "profile"
        home.mkdir()
        (home / ".env").write_text(
            "PROFILE_ONLY_KEY=from-profile\nMANAGED_KEY=from-profile\nPATH=/from/profile\n"
        )
        managed = tmp_path / "managed"
        managed.mkdir()
        (managed / ".env").write_text("MANAGED_KEY=from-managed\n")
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
        managed_scope.invalidate_managed_cache()
        monkeypatch.delenv("PROFILE_ONLY_KEY", raising=False)
        monkeypatch.setenv("PATH", "/from/environ")
        monkeypatch.setenv("LAUNCH_ONLY_KEY", "from-environ")
        refs = ["${env:PROFILE_ONLY_KEY}", "${env:MANAGED_KEY}", "${env:PATH}", "${env:NOWHERE_KEY}"]

        token = set_hermes_home_override(str(home))
        try:
            unscoped = [_expand_env_vars(ref) for ref in refs]
            launch_only = _expand_env_vars("${env:LAUNCH_ONLY_KEY}")
            scope = set_secret_scope(build_profile_secret_scope(home), profile_home=str(home))
            try:
                scoped = [_expand_env_vars(ref) for ref in refs]
            finally:
                reset_secret_scope(scope)
        finally:
            reset_hermes_home_override(token)
            managed_scope.invalidate_managed_cache()

        assert unscoped == scoped
        assert unscoped == ["from-profile", "from-managed", "/from/environ", "${env:NOWHERE_KEY}"]
        assert launch_only == "from-environ"
        # Without the override the launch environment answers again.
        assert _expand_env_vars("${env:PROFILE_ONLY_KEY}") == "${env:PROFILE_ONLY_KEY}"
