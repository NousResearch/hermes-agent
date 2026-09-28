"""Tests for ${ENV_VAR} substitution in config.yaml values."""

from hermes_cli.config import _expand_env_vars, load_config


class TestExpandEnvVars:


    def test_non_string_values_untouched(self):
        assert _expand_env_vars(42) == 42
        assert _expand_env_vars(3.14) == 3.14
        assert _expand_env_vars(True) is True
        assert _expand_env_vars(None) is None

    def test_default_operator_fills_unset_or_empty_and_yields_to_a_set_var(self, monkeypatch):
        """``${VAR:-x}`` (port of openclaw/openclaw#155164): the default wins when the var is unset
        OR empty (bash ``:-``), a set var wins over the default, ``${env:VAR:-x}`` shares the grammar,
        and a default carrying ``$``/``{`` is not parsed (nested shapes stay literal as before)."""
        from hermes_cli.config import _env_ref_snapshot, _preserve_env_ref_templates

        monkeypatch.delenv("OC155164_UNSET", raising=False)
        monkeypatch.setenv("OC155164_EMPTY", "")
        monkeypatch.setenv("OC155164_SET", "from-env")
        cfg = {
            "unset": "${OC155164_UNSET:-fallback}",
            "empty": "${OC155164_EMPTY:-fallback}",
            "set": "${env:OC155164_SET:-fallback}",
            "composite": "https://${OC155164_UNSET:-api.example.com}/v1",
            "empty_default": "[${OC155164_UNSET:-}]",
            "nested": "${OC155164_UNSET:-${OC155164_SET}}",
        }
        assert _expand_env_vars(cfg) == {
            "unset": "fallback", "empty": "fallback", "set": "from-env",
            "composite": "https://api.example.com/v1", "empty_default": "[]",
            "nested": "${OC155164_UNSET:-${OC155164_SET}}",
        }
        # The cache-staleness snapshot keys on the REAL var name, so a later export of the var
        # invalidates an expansion that resolved from the default (#58514 contract).
        assert "OC155164_UNSET" in _env_ref_snapshot(cfg)
        # Persisting the loaded config restores the authored template, never the inlined default.
        assert _preserve_env_ref_templates("fallback", "${OC155164_UNSET:-fallback}") == "${OC155164_UNSET:-fallback}"


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
