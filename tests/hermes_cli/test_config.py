"""Tests for hermes_cli configuration management."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from hermes_cli.config import (
    DEFAULT_CONFIG,
    check_config_version,
    get_hermes_home,
    ensure_hermes_home,
    get_compatible_custom_providers,
    _explicit_config_paths,
    _normalize_max_turns_config,
    is_provider_enabled,
    load_config,
    load_env,
    migrate_config,
    read_raw_config,
    remove_env_value,
    save_config,
    save_env_value,
    save_env_value_secure,
    sanitize_env_file,
    set_config_value,
    unset_config_value,
    write_platform_config_field,
    _sanitize_env_lines,
)


class TestGetHermesHome:
    def test_default_path(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_HOME", None)
            home = get_hermes_home()
            if sys.platform == "win32":
                # Windows default is %LOCALAPPDATA%\hermes — see
                # hermes_constants._get_platform_default_hermes_home.
                local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
                base = (
                    Path(local_appdata)
                    if local_appdata
                    else Path.home() / "AppData" / "Local"
                )
                assert home == base / "hermes"
            else:
                assert home == Path.home() / ".hermes"


class TestEnsureHermesHome:

    def test_creates_default_soul_md_if_missing(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            ensure_hermes_home()
            soul_path = tmp_path / "SOUL.md"
            assert soul_path.exists()
            assert soul_path.read_text(encoding="utf-8").strip() != ""


    def test_upgrades_legacy_template_soul_md(self, tmp_path):
        # Older installers seeded a comment-only scaffold that shadowed the
        # runtime default. A SOUL.md still matching that scaffold carries no
        # user persona and should be upgraded in place to DEFAULT_SOUL_MD.
        from hermes_cli.default_soul import DEFAULT_SOUL_MD, _LEGACY_TEMPLATE_SOULS

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            soul_path = tmp_path / "SOUL.md"
            soul_path.write_text(_LEGACY_TEMPLATE_SOULS[0] + "\n", encoding="utf-8")
            ensure_hermes_home()
            assert soul_path.read_text(encoding="utf-8") == DEFAULT_SOUL_MD

    # The pre-#95681 DEFAULT_SOUL_MD text, hardcoded (not read from the
    # module) so this fixture keeps testing the OLD text regardless of any
    # future change to _LEGACY_TEMPLATE_SOULS's length or ordering.
    _PRE_REWRITE_DEFAULT_SOUL = (
        "You are Hermes Agent, an intelligent AI assistant created by Nous "
        "Research. You are helpful, knowledgeable, and direct. You assist "
        "users with a wide range of tasks including answering questions, "
        "writing and editing code, analyzing information, creative work, "
        "and executing actions via your tools. You communicate clearly, "
        "admit uncertainty when appropriate, and prioritize being "
        "genuinely useful over being verbose unless otherwise directed "
        "below. Be targeted and efficient in your exploration and "
        "investigations."
    )

    def test_upgrades_pre_rewrite_default_soul_md(self, tmp_path):
        # Every install seeded between the old DEFAULT_SOUL_MD's introduction
        # and its #95681 rewrite got the old text auto-written on first run —
        # not user-authored, so it's just as safe to upgrade in place as the
        # comment-only scaffolds above. Regression test for that upgrade path.
        from hermes_cli.default_soul import DEFAULT_SOUL_MD

        assert self._PRE_REWRITE_DEFAULT_SOUL != DEFAULT_SOUL_MD  # sanity: fixture predates the rewrite

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            soul_path = tmp_path / "SOUL.md"
            soul_path.write_text(self._PRE_REWRITE_DEFAULT_SOUL, encoding="utf-8")
            ensure_hermes_home()
            assert soul_path.read_text(encoding="utf-8") == DEFAULT_SOUL_MD

    def test_does_not_upgrade_user_customized_soul_md(self, tmp_path):
        # A SOUL.md that merely starts with the old default but was edited by
        # the user carries real intent and must never be silently overwritten.
        from hermes_cli.default_soul import DEFAULT_SOUL_MD

        customized = self._PRE_REWRITE_DEFAULT_SOUL + " Also: always answer in rhyming couplets."

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            soul_path = tmp_path / "SOUL.md"
            soul_path.write_text(customized, encoding="utf-8")
            ensure_hermes_home()
            content = soul_path.read_text(encoding="utf-8")
            assert content == customized
            assert content != DEFAULT_SOUL_MD





class TestLoadConfigDefaults:
    def test_returns_defaults_when_no_file(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config = load_config()
            assert config["model"] == DEFAULT_CONFIG["model"]
            assert config["agent"]["max_turns"] == DEFAULT_CONFIG["agent"]["max_turns"]
            assert "max_turns" not in config
            assert "terminal" in config
            assert config["terminal"]["backend"] == "local"
            assert config["display"]["interim_assistant_messages"] is True

    def test_legacy_root_level_max_turns_migrates_to_agent_config(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config_path = tmp_path / "config.yaml"
            config_path.write_text("max_turns: 42\n")

            config = load_config()
            assert config["agent"]["max_turns"] == 42
            assert "max_turns" not in config


class TestLoadConfigParseFailure:
    """A YAML parse failure must NOT silently fall back to defaults.

    Before issue #23570 this was a single ``print(...)`` that scrolled past
    on the first invocation — users saw aux-fallback misbehavior with no clue
    their config.yaml was being ignored. The helper must:
      * log at WARNING (so ``hermes logs`` surfaces it)
      * also write to stderr (so it's visible at startup even before
        ``setup_logging()`` has wired up file handlers)
      * dedup on (path, mtime_ns, size) so concurrent loads don't spam
      * re-warn after the user edits the file (different mtime)
    """




    def test_corrupt_config_is_backed_up(self, tmp_path, capsys):
        """A broken config.yaml is snapshotted to a timestamped .bak so the
        user's recoverable overrides survive a later wizard/config-set rewrite.

        Ported from google-gemini/gemini-cli#21541 (policy-file TOML recovery),
        adapted: we back up but deliberately do NOT reset config.yaml.
        """
        from hermes_cli import config as cfg_mod
        cfg_mod._CONFIG_PARSE_WARNED.clear()

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            broken = "\tmodel: test/custom\nbroken indent:\n"
            (tmp_path / "config.yaml").write_text(broken)

            load_config()
            err = capsys.readouterr().err

            baks = list(tmp_path.glob("config.yaml.corrupt.*.bak"))
            assert len(baks) == 1, f"expected one backup, got {baks}"
            # Backup preserves the original broken content verbatim
            assert baks[0].read_text() == broken
            # Original config.yaml is left untouched (not reset to clean state)
            assert (tmp_path / "config.yaml").read_text() == broken
            # User is told where the backup landed
            assert str(baks[0]) in err



    def test_last_known_good_retained_within_process(self, tmp_path, capsys):
        """Port of openai/codex#31188's invariant: a parse failure must not
        silently replace the effective config (policy included) with
        defaults when the process already loaded a good config.

        Scenario: long-running gateway, user mid-edits config.yaml into
        broken YAML. Before this fix the next load_config() dropped every
        override — including ``approvals.deny`` security rules. Now the
        last successfully loaded config keeps being served until the file
        parses again.
        """
        import time
        from hermes_cli import config as cfg_mod
        cfg_mod._CONFIG_PARSE_WARNED.clear()

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            cfg = tmp_path / "config.yaml"
            cfg.write_text(
                "model:\n  default: test/custom-model\n"
                "approvals:\n  deny:\n    - 'curl*evil.com*'\n"
            )

            good = load_config()
            assert good["model"]["default"] == "test/custom-model"
            assert good["approvals"]["deny"] == ["curl*evil.com*"]
            capsys.readouterr()

            # Corrupt the file (mtime must change to bust the cache)
            time.sleep(0.05)
            cfg.write_text("approvals:\n  deny: [unclosed\n  :::bad {{{\n")

            after = load_config()
            # Last-known-good retained — NOT defaults
            assert after["model"]["default"] == "test/custom-model"
            assert after["approvals"]["deny"] == ["curl*evil.com*"]
            # Warning says we kept the previous config, not defaults
            err = capsys.readouterr().err
            assert "previously loaded config" in err





class TestEmptyConfigSections:
    """Empty section keys (``terminal:`` with no value) parse as YAML None
    and must not replace the default dict for that section (#58277)."""

    def test_null_section_keeps_defaults_in_load_config(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            (tmp_path / "config.yaml").write_text(
                "model:\n  default: test/custom\n"
                "terminal:\n"
                "display:\n"
            )
            config = load_config()
            assert config["model"]["default"] == "test/custom"
            assert isinstance(config["terminal"], dict)
            assert config["terminal"] == DEFAULT_CONFIG["terminal"]
            assert isinstance(config["display"], dict)

    def test_null_override_of_non_dict_default_still_applies(self, tmp_path):
        """None only shields dict defaults — explicit null for a scalar
        key remains an override (unchanged behavior)."""
        from hermes_cli.config import _deep_merge

        merged = _deep_merge({"scalar": 5, "section": {"a": 1}},
                             {"scalar": None, "section": None})
        assert merged["scalar"] is None
        assert merged["section"] == {"a": 1}


class TestSaveAndLoadRoundtrip:
    @staticmethod
    def _deny_config_reads(config_path):
        real_open = open

        def fake_open(file, mode="r", *args, **kwargs):
            if Path(file) == config_path and "r" in mode:
                raise PermissionError("denied")
            return real_open(file, mode, *args, **kwargs)

        return fake_open

    def test_roundtrip(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config = load_config()
            config["model"] = "test/custom-model"
            config["agent"]["max_turns"] = 42
            save_config(config)

            reloaded = load_config()
            assert reloaded["model"] == "test/custom-model"
            assert reloaded["agent"]["max_turns"] == 42

            saved = yaml.safe_load((tmp_path / "config.yaml").read_text())
            assert saved["agent"]["max_turns"] == 42
            assert "max_turns" not in saved

    def test_save_config_refuses_to_overwrite_unreadable_existing_config(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        original = "model: test/original\n"
        config_path.write_text(original, encoding="utf-8")

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            with patch("builtins.open", side_effect=self._deny_config_reads(config_path)):
                with pytest.raises(RuntimeError, match="Refusing to overwrite"):
                    save_config({"model": "test/replacement"})

        assert config_path.read_text(encoding="utf-8") == original








    def test_config_set_refuses_to_overwrite_unparseable_existing_config(self, tmp_path):
        """Unparseable YAML must not be replaced with a single-key document.

        Regression for the set/unset wipe class: a bare except around YAML
        load used to treat parse failure as {}, then atomic-write only the
        new key — destroying every prior override with no .corrupt backup.
        """
        config_path = tmp_path / "config.yaml"
        original = (
            "model:\n"
            "  default: claude-opus\n"
            "  provider: anthropic\n"
            "gateway:\n"
            "  platforms:\n"
            "    telegram:\n"
            "      enabled: true\n"
            "broken: [unterminated\n"
        )
        config_path.write_text(original, encoding="utf-8")

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            with pytest.raises(RuntimeError, match="not valid YAML"):
                set_config_value("model.default", "gpt-4o")

        assert config_path.read_text(encoding="utf-8") == original
        assert list(tmp_path.glob("config.yaml.corrupt.*.bak")), (
            "parse-failure path should snapshot a corrupt backup before refusing"
        )

    def test_config_unset_refuses_to_overwrite_unparseable_existing_config(self, tmp_path):
        """Unset must refuse the same way — env-sync paths used to write {}."""
        config_path = tmp_path / "config.yaml"
        original = "model:\n  default: keep-me\nbroken: [unterminated\n"
        config_path.write_text(original, encoding="utf-8")
        (tmp_path / ".env").write_text("TERMINAL_TIMEOUT=30\n", encoding="utf-8")

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            with pytest.raises(RuntimeError, match="not valid YAML"):
                unset_config_value("terminal.timeout")

        assert config_path.read_text(encoding="utf-8") == original
        assert (tmp_path / ".env").read_text(encoding="utf-8") == "TERMINAL_TIMEOUT=30\n"
        assert list(tmp_path.glob("config.yaml.corrupt.*.bak")), (
            "unset parse-failure path should snapshot a corrupt backup before refusing"
        )

    def test_config_set_refuses_non_mapping_root(self, tmp_path):
        """A list/scalar root parses without raising but would still wipe."""
        config_path = tmp_path / "config.yaml"
        original = "- just\n- a\n- list\n"
        config_path.write_text(original, encoding="utf-8")

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            with pytest.raises(RuntimeError, match="must be a mapping"):
                set_config_value("model.default", "gpt-4o")

        assert config_path.read_text(encoding="utf-8") == original
        assert list(tmp_path.glob("config.yaml.corrupt.*.bak")), (
            "non-mapping root should snapshot a corrupt backup before refusing"
        )

    def test_config_unset_refuses_non_mapping_root(self, tmp_path):
        """Unset shares the same non-mapping refuse path as set."""
        config_path = tmp_path / "config.yaml"
        original = "- just\n- a\n- list\n"
        config_path.write_text(original, encoding="utf-8")

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            with pytest.raises(RuntimeError, match="must be a mapping"):
                unset_config_value("model.default")

        assert config_path.read_text(encoding="utf-8") == original
        assert list(tmp_path.glob("config.yaml.corrupt.*.bak"))

    def test_config_set_allows_valid_empty_mapping(self, tmp_path):
        """A genuine empty {} config must still be writable (not a false refuse)."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("{}\n", encoding="utf-8")

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            set_config_value("model.default", "gpt-4o")

        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved == {"model": {"default": "gpt-4o"}}

    def test_atomic_config_write_refuses_unparseable_existing_config(self, tmp_path):
        """Shared chokepoint must refuse unparseable YAML, not only unreadable."""
        from hermes_cli.config import atomic_config_write

        config_path = tmp_path / "config.yaml"
        original = "broken: [unterminated\n"
        config_path.write_text(original, encoding="utf-8")

        with pytest.raises(RuntimeError, match="not valid YAML"):
            atomic_config_write(config_path, {"model": {"provider": "openai"}})

        assert config_path.read_text(encoding="utf-8") == original
        assert list(tmp_path.glob("config.yaml.corrupt.*.bak"))

class TestSaveEnvValueSecure:

    def test_secure_save_returns_metadata_only(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            result = save_env_value_secure("GITHUB_TOKEN", "ghp_test_secret")
            assert result == {
                "success": True,
                "stored_as": "GITHUB_TOKEN",
                "validated": False,
            }
            assert "secret" not in str(result).lower()



    def test_save_env_value_preserves_existing_file_mode_on_posix(self, tmp_path):
        """Regression for #31518: pre-existing .env mode (e.g. 0640 for a
        Docker bind-mount that the operator chose) survives subsequent
        writes. Previously _secure_file ran unconditionally after the
        mode-restore branch and re-tightened to 0600.
        """
        if os.name == "nt":
            return

        env_path = tmp_path / ".env"
        env_path.write_text("EXISTING=value\n")
        os.chmod(env_path, 0o640)

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            save_env_value("TENOR_API_KEY", "sk-test-secret")

        env_mode = env_path.stat().st_mode & 0o777
        assert env_mode == 0o640, f"expected 0o640, got {oct(env_mode)}"

    def test_save_env_value_quotes_values_containing_hash(self, tmp_path):
        """Regression test for #30355."""
        from dotenv import dotenv_values

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}, clear=False):
            os.environ.pop("ANTHROPIC_TOKEN", None)
            token = "sk-ant-oat01-abc#xyz#more"
            save_env_value("ANTHROPIC_TOKEN", token)

            content = (tmp_path / ".env").read_text(encoding="utf-8")
            assert f'ANTHROPIC_TOKEN="{token}"' in content

            parsed = dotenv_values(str(tmp_path / ".env"))
            assert parsed["ANTHROPIC_TOKEN"] == token
            assert load_env()["ANTHROPIC_TOKEN"] == token


    def test_save_env_value_already_quoted_input_is_not_double_wrapped_idempotently(
        self, tmp_path
    ):
        """Callers pass raw values; if a value literally contains quote
        characters, escaping+wrap is the dialect (#57249). Re-saving the
        same raw value is stable (no quote growth). load_env round-trips.
        """
        # User-typed value that already includes surrounding quotes as data.
        raw = '"/Users/me/Application Support/key"'
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}, clear=False):
            os.environ.pop("TERMINAL_SSH_KEY", None)
            save_env_value("TERMINAL_SSH_KEY", raw)
            first = (tmp_path / ".env").read_text(encoding="utf-8")
            save_env_value("TERMINAL_SSH_KEY", raw)
            second = (tmp_path / ".env").read_text(encoding="utf-8")
            assert first == second
            # One outer wrap layer only (escaped inner quotes, not nested wraps).
            line = [
                ln for ln in first.splitlines() if ln.startswith("TERMINAL_SSH_KEY=")
            ][0]
            assert line.startswith('TERMINAL_SSH_KEY="')
            assert line.endswith('"')
            assert line.count('TERMINAL_SSH_KEY="') == 1
            # Escaping dialect end-to-end: load sees the raw input, not stripped quotes.
            assert load_env()["TERMINAL_SSH_KEY"] == raw


class TestRemoveEnvValue:
    def test_removes_key_from_env_file(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("KEY_A=value_a\nKEY_B=value_b\nKEY_C=value_c\n")
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path), "KEY_B": "value_b"}):
            result = remove_env_value("KEY_B")
            assert result is True
            content = env_path.read_text()
            assert "KEY_B" not in content
            assert "KEY_A=value_a" in content
            assert "KEY_C=value_c" in content


    def test_clears_os_environ_even_when_not_in_file(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("OTHER=stuff\n")
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path), "ORPHAN_KEY": "orphan"}):
            remove_env_value("ORPHAN_KEY")
            assert "ORPHAN_KEY" not in os.environ

    def test_remove_env_value_preserves_existing_file_mode_on_posix(self, tmp_path):
        """Regression: pre-existing .env mode (e.g. 0640 for a Docker
        bind-mount the operator chose) survives a remove just as it does a
        save. Previously _secure_file ran unconditionally after the
        mode-restore branch and re-tightened to 0600 — the same bug fixed
        in save_env_value (#33699), in the sibling remove path.
        """
        if os.name == "nt":
            return

        env_path = tmp_path / ".env"
        env_path.write_text("KEEP=value\nDROP=gone\n")
        os.chmod(env_path, 0o640)

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path), "DROP": "gone"}):
            removed = remove_env_value("DROP")

        assert removed is True
        assert "DROP" not in env_path.read_text()
        env_mode = env_path.stat().st_mode & 0o777
        assert env_mode == 0o640, f"expected 0o640, got {oct(env_mode)}"


class TestSaveConfigAtomicity:
    """Verify save_config uses atomic writes (tempfile + os.replace)."""

    def test_no_partial_write_on_crash(self, tmp_path):
        """If save_config crashes mid-write, the previous file stays intact."""
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            # Write an initial config
            config = load_config()
            config["model"] = "original-model"
            save_config(config)

            config_path = tmp_path / "config.yaml"
            assert config_path.exists()

            # Simulate a crash during yaml.dump by making atomic_yaml_write's
            # yaml.dump raise after the temp file is created but before replace.
            with patch("utils.yaml.dump", side_effect=OSError("disk full")):
                try:
                    config["model"] = "should-not-persist"
                    save_config(config)
                except OSError:
                    pass

            # Original file must still be intact
            reloaded = load_config()
            assert reloaded["model"] == "original-model"

    def test_no_leftover_temp_files(self, tmp_path):
        """Failed writes must clean up their temp files."""
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config = load_config()
            save_config(config)

            with patch("utils.yaml.dump", side_effect=OSError("disk full")):
                try:
                    save_config(config)
                except OSError:
                    pass

            # No .tmp files should remain
            tmp_files = list(tmp_path.glob(".*config*.tmp"))
            assert tmp_files == []

    def test_atomic_write_creates_valid_yaml(self, tmp_path):
        """The written file must be valid YAML matching the input."""
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config = load_config()
            config["model"] = "test/atomic-model"
            config["agent"]["max_turns"] = 77
            save_config(config)

            # Read raw YAML to verify it's valid and correct
            config_path = tmp_path / "config.yaml"
            with open(config_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            assert raw["model"] == "test/atomic-model"
            assert raw["agent"]["max_turns"] == 77


class TestSanitizeEnvLines:
    """Tests for semantics-preserving .env line normalization."""





    def test_migrate_reports_normalized_line_formatting(self, capsys):
        latest_version = DEFAULT_CONFIG["_config_version"]
        with (
            patch("hermes_cli.config.sanitize_env_file", return_value=2),
            patch(
                "hermes_cli.config.check_config_version",
                return_value=(latest_version, latest_version),
            ),
            patch("hermes_cli.config.read_raw_config", return_value={}),
            patch("hermes_cli.config.get_missing_env_vars", return_value=[]),
            patch("hermes_cli.config.get_missing_config_fields", return_value=[]),
            patch("hermes_cli.config.get_missing_skill_config_vars", return_value=[]),
        ):
            migrate_config(interactive=False)

        assert capsys.readouterr().out == (
            "  ✓ Normalized .env line formatting (2 line(s) changed)\n"
        )





    def test_glm_suffix_collision_not_split(self):
        """GLM_API_KEY / GLM_BASE_URL must not be mangled by LM_API_KEY / LM_BASE_URL suffixes (#17138)."""
        lines = [
            "GLM_API_KEY=glm-secret\n",
            "GLM_BASE_URL=https://api.z.ai/api/paas/v4\n",
        ]
        result = _sanitize_env_lines(lines)
        assert result == lines, f"GLM_* lines were corrupted by suffix collision: {result}"






    def test_sanitize_env_file_does_not_rewrite_value_semantics(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "FAL_KEY=good\n"
            "OPENROUTER_API_KEY=valFIRECRAWL_API_KEY=val2\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            fixes = sanitize_env_file()
            assert fixes == 0

            content = env_file.read_text()
            assert content == (
                "FAL_KEY=good\n"
                "OPENROUTER_API_KEY=valFIRECRAWL_API_KEY=val2\n"
            )

    def test_sanitize_env_file_noop_on_clean_file(self, tmp_path):
        """No changes when file is already clean."""
        env_file = tmp_path / ".env"
        env_file.write_text("GOOD_KEY=good\nOTHER_KEY=other\n")
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            fixes = sanitize_env_file()
            assert fixes == 0


class TestOptionalEnvVarsRegistry:
    """Verify that key env vars are registered in OPTIONAL_ENV_VARS."""

    def test_tavily_api_key_registered(self):
        """TAVILY_API_KEY is listed in OPTIONAL_ENV_VARS."""
        from hermes_cli.config import OPTIONAL_ENV_VARS
        assert "TAVILY_API_KEY" in OPTIONAL_ENV_VARS


    def test_tavily_api_key_has_url(self):
        """TAVILY_API_KEY has a URL."""
        from hermes_cli.config import OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["TAVILY_API_KEY"]["url"] == "https://app.tavily.com/home"

    def test_tavily_in_env_vars_by_version(self):
        """TAVILY_API_KEY is listed in ENV_VARS_BY_VERSION."""
        from hermes_cli.config import ENV_VARS_BY_VERSION
        all_vars = []
        for vars_list in ENV_VARS_BY_VERSION.values():
            all_vars.extend(vars_list)
        assert "TAVILY_API_KEY" in all_vars

    def test_max_iterations_not_offered_as_env_var(self):
        """HERMES_MAX_ITERATIONS must NOT be in OPTIONAL_ENV_VARS (issue #17534).

        Offering it as an editable env var (dashboard, `hermes setup`) lets a
        user write it to .env, recreating the stale ghost that shadows
        config.yaml's agent.max_turns. The iteration budget is configured ONLY
        via config.yaml; HERMES_MAX_ITERATIONS remains a read-only backward-compat
        fallback in the gateway/CLI, never a promoted write target.
        """
        from hermes_cli.config import OPTIONAL_ENV_VARS
        assert "HERMES_MAX_ITERATIONS" not in OPTIONAL_ENV_VARS


class TestMemoryProviderEnvVarsRegistry:
    """Every memory provider that reads an API key from the environment must
    have that key catalogued in OPTIONAL_ENV_VARS so the dashboard Keys page
    and `hermes setup` surface it (previously only Honcho was listed, leaving
    Hindsight/Supermemory/Mem0/RetainDB/ByteRover/OpenViking invisible).

    This is a behavior contract, not a snapshot: it asserts each provider's
    primary credential key is present, tool-categorised, and password-masked —
    not a frozen count of entries.
    """

    # provider primary-credential env key -> the tool-call name it powers.
    MEMORY_PROVIDER_KEYS = {
        "HONCHO_API_KEY": "honcho_context",
        "HINDSIGHT_API_KEY": "hindsight_recall",
        "SUPERMEMORY_API_KEY": "supermemory_search",
        "MEM0_API_KEY": "mem0_search",
        "RETAINDB_API_KEY": "retaindb_search",
        "BRV_API_KEY": "brv_query",
        "OPENVIKING_API_KEY": "viking_search",
    }

    def test_memory_provider_keys_are_catalogued(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS
        missing = [k for k in self.MEMORY_PROVIDER_KEYS if k not in OPTIONAL_ENV_VARS]
        assert not missing, f"memory provider keys missing from OPTIONAL_ENV_VARS: {missing}"


    def test_memory_provider_keys_advertise_their_tool(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS
        for key, tool in self.MEMORY_PROVIDER_KEYS.items():
            assert tool in OPTIONAL_ENV_VARS[key].get("tools", []), key


class TestConfigMigrationSecretPrompts:
    def test_required_secret_env_prompt_uses_masked_prompt(self, tmp_path, monkeypatch):
        from hermes_cli import config as cfg_mod

        saved = {}

        monkeypatch.setattr(cfg_mod, "sanitize_env_file", lambda: 0)
        monkeypatch.setattr(cfg_mod, "check_config_version", lambda: (999, 999))
        monkeypatch.setattr(cfg_mod, "get_missing_config_fields", lambda: [])
        monkeypatch.setattr(cfg_mod, "get_missing_skill_config_vars", lambda: [])
        monkeypatch.setattr(
            cfg_mod,
            "get_missing_env_vars",
            lambda required_only=True: [
                {
                    "name": "TEST_API_KEY",
                    "description": "Test key",
                    "prompt": "Test API key",
                    "password": True,
                }
            ]
            if required_only
            else [],
        )
        def fake_masked_secret_prompt(prompt):
            saved["prompt"] = prompt
            return "secret"

        monkeypatch.setattr(cfg_mod, "masked_secret_prompt", fake_masked_secret_prompt)
        monkeypatch.setattr(
            cfg_mod,
            "save_env_value",
            lambda name, value: saved.update({name: value}),
        )

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            results = cfg_mod.migrate_config(interactive=True, quiet=True)

        assert saved["prompt"] == "  Test API key: "
        assert saved["TEST_API_KEY"] == "secret"
        assert results["env_added"] == ["TEST_API_KEY"]


class TestConfigVersionDetection:
    def test_check_config_version_uses_raw_on_disk_version(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model: {}\n", encoding="utf-8")

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            assert load_config()["_config_version"] == DEFAULT_CONFIG["_config_version"]
            assert check_config_version() == (0, DEFAULT_CONFIG["_config_version"])


class TestConfigSupportFloor:
    """Auto-migration support floor (v12).

    Configs below ``SUPPORT_FLOOR_VERSION`` are refused: the file stays
    byte-for-byte untouched, a clear actionable message is surfaced (stdout
    when not quiet + stderr always + results['warnings']), and the process
    continues without crashing — matching the fail-safe posture for
    unparseable configs. Configs at or above the floor migrate exactly as
    before the floor was introduced (parity fixtures below).
    """

    def _write_config(self, tmp_path, data):
        config_path = tmp_path / "config.yaml"
        text = yaml.safe_dump(data)
        config_path.write_text(text, encoding="utf-8")
        return config_path, text

    def test_v11_config_is_refused_and_untouched(self, tmp_path, capsys):
        config_path, original = self._write_config(
            tmp_path,
            {
                "_config_version": 11,
                "custom_providers": [
                    {"name": "Old", "base_url": "http://localhost:1234/v1"}
                ],
            },
        )
        (tmp_path / ".env").write_text("ANTHROPIC_TOKEN=old-token\n")
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            results = migrate_config(interactive=False, quiet=False)

            # File untouched — no migration, no version bump, no rewrite.
            assert config_path.read_text(encoding="utf-8") == original
            # .env untouched too (the retired <12 steps used to clear tokens).
            assert load_env().get("ANTHROPIC_TOKEN") == "old-token"

        captured = capsys.readouterr()
        expected_fragment = (
            "This config predates version 12 (~2 years old) and can no "
            "longer be auto-migrated."
        )
        assert expected_fragment in captured.out
        assert expected_fragment in captured.err
        assert "run `hermes setup` to regenerate" in captured.out
        assert "_config_version: 12" in captured.out
        assert any(expected_fragment in w for w in results["warnings"])
        # No 'Config version: X → Y' line — nothing was migrated.
        assert "Config version:" not in captured.out

    def test_v11_quiet_still_warns_on_stderr_only(self, tmp_path, capsys):
        config_path, original = self._write_config(
            tmp_path, {"_config_version": 11}
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            results = migrate_config(interactive=False, quiet=True)
        assert config_path.read_text(encoding="utf-8") == original
        captured = capsys.readouterr()
        assert "can no longer be auto-migrated" in captured.err
        assert captured.out == ""
        assert results["warnings"]

    def test_floor_message_uses_display_hermes_home(self):
        from hermes_cli.config_migrations import support_floor_message
        from hermes_constants import display_hermes_home

        msg = support_floor_message()
        assert f"{display_hermes_home()}/config.yaml" in msg

    def test_registry_has_no_targets_below_floor(self):
        from hermes_cli.config_migrations import (
            MIGRATIONS,
            SUPPORT_FLOOR_VERSION,
        )

        assert SUPPORT_FLOOR_VERSION == 12
        assert all(target >= SUPPORT_FLOOR_VERSION for target, _ in MIGRATIONS)
        # v12's own step is retained: a config AT v11 is refused, but a
        # config AT v12 must still receive every remaining migration.
        assert MIGRATIONS[0][0] == 12

    # ── Parity fixtures ──────────────────────────────────────────────
    # Expected outputs captured by running migrate_config from origin/main
    # (commit 28524adb0e, pre-floor) in a subprocess against these exact
    # fixtures. The floor must not change behavior for v12+ configs.

    _V12_FIXTURE = {
        "_config_version": 12,
        "model": {"default": "openai/gpt-5.4", "provider": "openrouter"},
        "display": {"tool_progress_overrides": {"telegram": "verbose"}},
        "stt": {"model": "base", "provider": "local"},
        "compression": {"summary_model": "gpt-x", "summary_provider": "auto"},
        "model_catalog": {"ttl_hours": 24},
        "memory": {"write_mode": "approve"},
        "delegation": {"max_async_children": 8},
        "agent": {"verify_on_stop": True},
    }
    _V12_EXPECTED = {
        "_config_version": 33,
        # agent.verify_on_stop stays materialised here: the fixture's on-disk
        # config explicitly set it (True), so _explicit_config_paths preserves
        # the key through the v32 flip even though False now equals the
        # schema default.
        "agent": {"verify_on_stop": False},
        "auxiliary": {"compression": {"model": "gpt-x"}},
        "compression": {},
        "delegation": {"max_concurrent_children": 8},
        "display": {
            "platforms": {"telegram": {"tool_progress": "verbose"}},
            "tool_progress_overrides": {"telegram": "verbose"},
        },
        "memory": {"write_approval": True},
        "model": {"default": "openai/gpt-5.4", "provider": "openrouter"},
        "model_catalog": {"ttl_hours": 1},
        "plugins": {"enabled": []},
        "stt": {"provider": "local"},
    }

    _V20_FIXTURE = {
        "_config_version": 20,
        "model": {"default": "anthropic/claude-fable-5", "provider": "nous"},
        "plugins": {"disabled": ["foo"]},
        "skills": {"write_mode": "on"},
        "model_catalog": {"ttl_hours": 24},
        "agent": {},
    }
    _V20_EXPECTED = {
        "_config_version": 33,
        # v31 writes verify_on_stop=False, but False now equals the schema
        # default (opt-in) so the write invariant strips it from disk.
        "agent": {},
        "model": {"default": "anthropic/claude-fable-5", "provider": "nous"},
        "model_catalog": {"ttl_hours": 1},
        "plugins": {"disabled": ["foo"], "enabled": []},
    }

    _ENV_FIXTURE = (
        "LLM_MODEL=old-model\nOPENAI_MODEL=old-openai\nOPENROUTER_API_KEY=test\n"
    )

    @pytest.mark.parametrize(
        "fixture,expected,expected_env",
        [
            (
                _V12_FIXTURE,
                _V12_EXPECTED,
                # v12→13 clears LLM_MODEL/OPENAI_MODEL for configs below 13.
                "LLM_MODEL=\nOPENAI_MODEL=\nOPENROUTER_API_KEY=test\n",
            ),
            (_V20_FIXTURE, _V20_EXPECTED, _ENV_FIXTURE),
        ],
        ids=["v12", "v20"],
    )
    def test_at_or_above_floor_migrates_identically_to_pre_floor(
        self, tmp_path, fixture, expected, expected_env
    ):
        config_path, _ = self._write_config(tmp_path, fixture)
        (tmp_path / ".env").write_text(self._ENV_FIXTURE, encoding="utf-8")
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            migrate_config(interactive=False, quiet=True)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        # Pin the golden version the fixtures were captured at, then compare
        # the rest against the same-latest expectation. If _config_version has
        # advanced past 33, only the version key may differ.
        assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]
        raw.pop("_config_version")
        exp = dict(expected)
        exp.pop("_config_version")
        if DEFAULT_CONFIG["_config_version"] == 33:
            assert raw == exp
        else:  # future migrations appended — golden subset must still hold
            for key, val in exp.items():
                assert raw.get(key) == val, f"parity drift on {key!r}"
        assert (tmp_path / ".env").read_text(encoding="utf-8") == expected_env


class TestCustomProviderCompatibility:
    """Custom provider compatibility across legacy and v12+ config schemas.

    The v11→12 step (_migrate_to_12) is retained in the registry per the
    support-floor policy, but migrate_config() refuses sub-v12 configs, so
    these tests drive run_migrations() directly to keep the step covered.
    """

    @staticmethod
    def _run_ladder(current_ver: int):
        from hermes_cli.config_migrations import run_migrations

        results = {"env_added": [], "config_added": [], "warnings": []}
        run_migrations(current_ver, results, quiet=True)
        return results


        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            self._run_ladder(11)
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        assert raw["providers"]["openai-direct"] == {
            "api": "https://api.openai.com/v1",
            "api_key": "test-key",
            "default_model": "gpt-5-mini",
            "name": "OpenAI Direct",
            "transport": "codex_responses",
        }
        # custom_providers removed by migration — runtime reads via compat layer
        assert "custom_providers" not in raw

    def test_v11_upgrade_preserves_custom_provider_model_metadata(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        model_map = {
            "kimi-k2.6": {"context_length": 262144},
            "moonshotai/Kimi-K2.6-ACED": {"context_length": 131072},
        }
        config_path.write_text(
            yaml.safe_dump(
                {
                    "_config_version": 11,
                    "custom_providers": [
                        {
                            "name": "Kimi Coding Plan",
                            "base_url": "https://api.kimi.example.com/coding",
                            "api_key_env": "KIMI_CODING_API_KEY",
                            "api_mode": "anthropic_messages",
                            "model": "kimi-k2.6",
                            "models": model_map,
                            "context_length": 262144,
                            "rate_limit_delay": 0.25,
                            "discover_models": False,
                            "extra_body": {
                                "chat_template_kwargs": {"enable_thinking": False}
                            },
                        },
                        {
                            "name": "List Models",
                            "base_url": "https://list.example.com/v1",
                            "models": ["alpha", "beta"],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            self._run_ladder(11)
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            compatible = get_compatible_custom_providers(raw)

        assert "custom_providers" not in raw
        provider = raw["providers"]["kimi-coding-plan"]
        assert provider["api"] == "https://api.kimi.example.com/coding"
        assert provider["key_env"] == "KIMI_CODING_API_KEY"
        assert provider["transport"] == "anthropic_messages"
        assert provider["default_model"] == "kimi-k2.6"
        assert provider["models"] == model_map
        assert provider["context_length"] == 262144
        assert provider["rate_limit_delay"] == 0.25
        assert provider["discover_models"] is False
        assert provider["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }
        assert raw["providers"]["list-models"]["models"] == {
            "alpha": {},
            "beta": {},
        }

        compatible_provider = next(
            entry for entry in compatible if entry["provider_key"] == "kimi-coding-plan"
        )
        assert compatible_provider["models"] == model_map
        assert compatible_provider["key_env"] == "KIMI_CODING_API_KEY"

    def test_providers_dict_resolves_at_runtime(self, tmp_path):
        """After migration deleted custom_providers, get_compatible_custom_providers
        still finds entries from the providers dict."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "_config_version": 17,
                    "providers": {
                        "openai-direct": {
                            "api": "https://api.openai.com/v1",
                            "api_key": "test-key",
                            "default_model": "gpt-5-mini",
                            "name": "OpenAI Direct",
                            "transport": "codex_responses",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            compatible = get_compatible_custom_providers()

        assert len(compatible) == 1
        assert compatible[0]["name"] == "OpenAI Direct"
        assert compatible[0]["base_url"] == "https://api.openai.com/v1"
        assert compatible[0]["provider_key"] == "openai-direct"
        assert compatible[0]["api_mode"] == "codex_responses"


    def test_compatible_custom_providers_prefers_base_url_then_url_then_api(self, tmp_path):
        """URL field precedence is base_url > url > api (PR #9332)."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "_config_version": 17,
                    "providers": {
                        "my-provider": {
                            "name": "My Provider",
                            "api": "https://api.example.com/v1",
                            "url": "https://url.example.com/v1",
                            "base_url": "https://base.example.com/v1",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            compatible = get_compatible_custom_providers()

        assert compatible == [
            {
                "name": "My Provider",
                "base_url": "https://base.example.com/v1",
                "provider_key": "my-provider",
            }
        ]


class TestInterimAssistantMessageConfig:
    """Test the explicit gateway interim-message config gate."""

    def test_default_config_enables_interim_assistant_messages(self):
        assert DEFAULT_CONFIG["display"]["interim_assistant_messages"] is True

    def test_migrate_to_v15_supplies_interim_message_gate_at_read_time(
        self, tmp_path, capsys
    ):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump({"_config_version": 14, "display": {"tool_progress": "off"}}),
            encoding="utf-8",
        )

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            results = migrate_config(interactive=False, quiet=False)
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            loaded = load_config()

        from hermes_cli.config import DEFAULT_CONFIG
        assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]
        # The user's explicit non-default value is preserved on disk.
        assert raw["display"]["tool_progress"] == "off"
        # interim_assistant_messages defaults to True and merges in transparently
        # at read time, so the migration must NOT materialise it to disk (that
        # was the config-bloat bug). It is still effective via load_config().
        assert "interim_assistant_messages" not in raw.get("display", {})
        assert loaded["display"]["interim_assistant_messages"] is True
        assert not any(
            "interim_assistant_messages" in item
            for item in results["config_added"]
        )
        assert "Added display.interim_assistant_messages" not in capsys.readouterr().out


class TestCliRefreshIntervalConfig:
    """Test the CLI refresh_interval config default (#45592 / #48309)."""

    def test_default_config_enables_cli_refresh_interval(self):
        """cli_refresh_interval defaults to 1.0 so the idle status-bar
        clock keeps ticking and the bottom chrome stays alive during
        idle (#45592). Users on emulators where the periodic redraw
        fights auto-scroll can set it to 0 (#48309)."""
        assert DEFAULT_CONFIG["display"]["cli_refresh_interval"] == 1.0


class TestDiscordChannelPromptsConfig:


    def test_migrate_preserves_custom_providers_and_no_defaults_dump(self, tmp_path):
        """Migration must not expand config.yaml to a defaults dump (#40821).

        Before the fix, migrations used load_config() which deep-merges
        DEFAULT_CONFIG, then save_config() wrote the full ~13KB expanded
        result — destroying comments and structure. Using read_raw_config()
        keeps the file small and preserves only the user's actual config.
        """
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump({
                "_config_version": 11,
                "model": {"default": "test-model", "provider": "openrouter"},
                "custom_providers": [
                    {"name": "local-llm", "base_url": "http://localhost:8080/v1",
                     "models": {"test": {}}}
                ],
            }),
            encoding="utf-8",
        )

        results = {"env_added": [], "config_added": [], "warnings": []}
        from hermes_cli.config_migrations import run_migrations
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            # Drive the ladder directly: migrate_config() refuses sub-v12
            # configs since the support floor, but the write-invariant this
            # test guards (#40821) lives in the steps themselves.
            run_migrations(11, results, quiet=True)
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        # custom_providers migrated to providers dict (by design, v11->v12)
        assert "custom_providers" not in raw
        assert "providers" in raw
        assert "local-llm" in raw["providers"]
        assert raw["providers"]["local-llm"]["api"] == "http://localhost:8080/v1"

        # File must NOT be a defaults dump — assert specific DEFAULT_CONFIG
        # top-level keys are absent (they should only appear via load_config's
        # deep-merge, not be written to the user's file by migration).
        for default_key in ("tts", "compression", "security", "whatsapp", "bedrock"):
            assert default_key not in raw, (
                f"{default_key} should not be in migrated config file — "
                f"migration should use read_raw_config() to avoid defaults dump"
            )


class TestEnvWriteDenylist:
    """``save_env_value`` refuses to persist env-var names that
    influence how subprocesses execute — ``LD_PRELOAD``, ``PYTHONPATH``,
    ``PATH``, ``EDITOR``, etc. — or selected Hermes runtime/security controls.

    The dashboard exposes ``PUT /api/env`` to any authed caller (and
    the session token lives in the SPA's HTML where any future plugin
    XSS or local process could exfiltrate it). Without this gate, an
    attacker who steals the token could plant
    ``LD_PRELOAD=/tmp/evil.so`` in ``.env`` and own the next Hermes
    process on next startup via the dotenv → ``os.environ`` chain in
    ``hermes_cli/env_loader.py``.

    Regression test for the dashboard pentest finding filed alongside
    the ``web-pentest`` skill (PR #32265 / issue #32267).
    """

    @pytest.fixture(autouse=True)
    def _hermes_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        ensure_hermes_home()


    @pytest.mark.parametrize(
        "allowed_key",
        [
            "HERMES_LANGFUSE_PUBLIC_KEY",
            "HERMES_SPOTIFY_CLIENT_ID",
            "HERMES_QWEN_BASE_URL",
            "HERMES_MAX_ITERATIONS",
        ],
    )
    def test_hermes_integration_keys_still_writable(self, allowed_key):
        """``HERMES_*`` overall is NOT blocked.

        Integration credentials following that convention must keep working
        or we'd regress provider setup flows (auth.py, Spotify, Langfuse, …).
        """
        save_env_value(allowed_key, "test-value-123")
        env = load_env()
        assert env[allowed_key] == "test-value-123"

    @pytest.mark.parametrize(
        "protected_key",
        [
            "HERMES_CONFIG_PATH",
            "HERMES_ENV_PATH",
            "HERMES_OPTIONAL_MCPS",
            "HERMES_COPILOT_ACP_COMMAND",
            "HERMES_COPILOT_ACP_ARGS",
            "HERMES_YOLO_MODE",
            "HERMES_ACCEPT_HOOKS",
            "HERMES_REDACT_SECRETS",
            "HERMES_INTERACTIVE",
            "HERMES_EXEC_ASK",
            "HERMES_GATEWAY_SESSION",
            "HERMES_CRON_SESSION",
            "HERMES_SINGLE_QUERY_SESSION",
            "HERMES_SESSION_KEY",
            "HERMES_SESSION_PLATFORM",
        ],
    )
    def test_hermes_security_control_keys_are_not_writable(self, protected_key):
        """Generic writers must not persist runtime or approval controls."""
        with pytest.raises(ValueError, match="denylist"):
            save_env_value(protected_key, "1")

        assert protected_key not in load_env()

    def test_preexisting_optional_mcps_override_still_loads(self, tmp_path):
        """The writer gate must not migrate or ignore operator-owned .env state."""
        from hermes_cli.config import invalidate_env_cache

        catalog = tmp_path / "custom-mcp-catalog"
        (tmp_path / ".env").write_text(
            f"HERMES_OPTIONAL_MCPS={catalog}\n",
            encoding="utf-8",
        )
        invalidate_env_cache()

        assert load_env()["HERMES_OPTIONAL_MCPS"] == str(catalog)

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("Path", "PATH"),
            ("Hermes_Yolo_Mode", "HERMES_YOLO_MODE"),
            ("Hermes_Optional_Mcps", "HERMES_OPTIONAL_MCPS"),
            ("Hermes_Copilot_Acp_Command", "HERMES_COPILOT_ACP_COMMAND"),
            ("Hermes_Copilot_Acp_Args", "HERMES_COPILOT_ACP_ARGS"),
        ],
    )
    def test_windows_policy_names_are_case_insensitive(self, key, expected):
        from hermes_cli.config import _env_var_policy_name

        assert _env_var_policy_name(key, is_windows=True) == expected

    def test_posix_policy_names_remain_case_sensitive(self):
        from hermes_cli.config import _env_var_policy_name

        assert _env_var_policy_name("Path", is_windows=False) == "Path"

    @pytest.mark.parametrize("prefix", ["", "export "])
    def test_windows_env_assignment_matching_is_case_insensitive(self, prefix):
        from hermes_cli.config import _env_line_defines_key

        line = f"{prefix}Path=C:\\Windows\\System32\n"
        assert _env_line_defines_key(line, "PATH", is_windows=True)
        assert not _env_line_defines_key(line, "PATH", is_windows=False)

    @pytest.mark.windows_only
    @pytest.mark.parametrize(
        "protected_key",
        [
            "Hermes_Yolo_Mode",
            "Hermes_Optional_Mcps",
            "Hermes_Copilot_Acp_Command",
            "Hermes_Copilot_Acp_Args",
        ],
    )
    def test_windows_writer_rejects_mixed_case_protected_name(self, protected_key):
        with pytest.raises(ValueError, match="denylist"):
            save_env_value(protected_key, "1")



    def test_save_env_value_secure_inherits_denylist(self):
        """The ``_secure`` variant goes through ``save_env_value`` so
        it inherits the gate — verify, don't assume."""
        with pytest.raises(ValueError, match="denylist"):
            save_env_value_secure("LD_PRELOAD", "/tmp/evil.so")



class TestWriteApprovalMigration:
    """Version 28→29 renames memory/skills write_mode → write_approval (bool).

    Only an explicit ``approve`` carried gating intent and maps to ``True``;
    ``on``/``off``/unset map to ``False`` (gate off). The old ``write_mode`` key
    is removed. Only a persisted key is rewritten — never invented.
    """

    def _write(self, tmp_path, body: str):
        (tmp_path / "config.yaml").write_text(body)

    def test_approve_maps_to_true(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            self._write(tmp_path,
                        "_config_version: 28\nmemory:\n  write_mode: approve\n"
                        "skills:\n  write_mode: approve\n")
            migrate_config(interactive=False, quiet=True)
            raw = yaml.safe_load((tmp_path / "config.yaml").read_text())
            assert raw["memory"]["write_approval"] is True
            assert raw["skills"]["write_approval"] is True
            assert "write_mode" not in raw["memory"]
            assert "write_mode" not in raw["skills"]

    def test_on_and_off_map_to_false(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            # YAML 1.1 parses bare on/off as bools — write_mode could be either
            # the string or the bool; both legacy "not gating" values → False.
            self._write(tmp_path,
                        "_config_version: 28\nmemory:\n  write_mode: 'on'\n"
                        "skills:\n  write_mode: 'off'\n")
            migrate_config(interactive=False, quiet=True)
            raw = yaml.safe_load((tmp_path / "config.yaml").read_text())
            loaded = load_config()
            # write_approval=False equals the schema default, so it is NOT
            # materialised to disk (lean-config invariant) — the legacy
            # write_mode key is gone and the effective value resolves to False
            # via load_config()'s deep-merge.
            assert "write_mode" not in raw.get("memory", {})
            assert "write_mode" not in raw.get("skills", {})
            assert loaded["memory"]["write_approval"] is False
            assert loaded["skills"]["write_approval"] is False


class TestMigrationWriteInvariant:
    """Architectural guard: every migration write routes through the single
    _persist_migration() chokepoint, which strips schema defaults so a lean
    config is never bloated into a DEFAULT_CONFIG dump on a version bump.

    These lock the centralised invariant so a future migration that calls
    save_config(...) directly (re-introducing the config-bloat bug class) is
    caught immediately.
    """


    @pytest.mark.parametrize("start_version", [12, "latest_minus_one"])
    def test_version_bump_keeps_config_lean(self, tmp_path, start_version):
        """A lean config migrated to the latest version must never be rewritten
        into a defaults dump — neither across the whole supported range
        (start=12, the auto-migration floor, where per-version seeds also
        fire) nor on a bare one-version bump (where only
        the catch-all finalizer runs). In both cases no default-only top-level
        section the user never wrote may land on disk, the merged view still
        exposes every default, and the user's explicit non-default value
        survives.
        """
        latest = DEFAULT_CONFIG["_config_version"]
        start = latest - 1 if start_version == "latest_minus_one" else start_version
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump({
                "_config_version": start,
                "model": {"default": "test-model", "provider": "openrouter"},
                "matrix": {"require_mention": False},
            }, sort_keys=False),
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            migrate_config(interactive=False, quiet=True)
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            loaded = load_config()

        assert raw["_config_version"] == latest
        # User's explicit non-default value preserved (not reset to True default).
        assert raw["matrix"]["require_mention"] is False
        assert loaded["matrix"]["require_mention"] is False
        # No default-only top-level section the user never wrote lands on disk —
        # neither from per-version seeds nor the catch-all finalizer.
        for default_key in (
            "timezone", "curator", "auxiliary", "tts", "compression",
            "whatsapp", "bedrock",
        ):
            assert default_key not in raw, (
                f"{default_key} was materialised into a lean config by the "
                f"version bump — the default-dump regression returned"
            )
        # Defaults still take effect transparently via the read-time merge.
        assert loaded["curator"]["enabled"] == DEFAULT_CONFIG["curator"]["enabled"]
        assert loaded["display"]["compact"] == DEFAULT_CONFIG["display"]["compact"]


class TestSaveConfigPartialWritePreservation:
    """Regression for #62723: partial migration writes must not drop unrelated sections."""

    def test_merge_existing_preserves_platforms_on_partial_write(self, tmp_path):
        body = """_config_version: 30
model:
  default: deepseek-v4-pro
  provider: deepseek
agent:
  max_turns: 60
platforms:
  feishu:
    enabled: true
    extra:
      app_id: cli_xxx
      app_secret: xxx
feishu:
  require_mention: true
"""
        (tmp_path / "config.yaml").write_text(body, encoding="utf-8")
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            save_config(
                {
                    "_config_version": 30,
                    "model": {"default": "deepseek-v4-pro", "provider": "deepseek"},
                    "agent": {"max_turns": 60, "verify_on_stop": False},
                },
                merge_existing=True,
            )
            raw = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
            merged = load_config()

        assert raw["platforms"]["feishu"]["extra"]["app_id"] == "cli_xxx"
        assert raw["feishu"]["require_mention"] is True
        # verify_on_stop=False now equals the schema default (opt-in), so
        # strip_defaults removes it from disk; deep-merge supplies it at read.
        assert "verify_on_stop" not in raw.get("agent", {})
        assert merged["agent"]["verify_on_stop"] is False


    def test_persist_migration_writes_full_read_raw_config(self, tmp_path):
        from hermes_cli.config import _persist_migration, read_raw_config

        body = """_config_version: 30
model:
  default: deepseek-v4-pro
  provider: deepseek
agent:
  max_turns: 60
platforms:
  feishu:
    enabled: true
    extra:
      app_id: cli_xxx
      app_secret: xxx
"""
        (tmp_path / "config.yaml").write_text(body, encoding="utf-8")
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config = read_raw_config()
            config.setdefault("agent", {})["verify_on_stop"] = False
            config["_config_version"] = 32
            _persist_migration(config)
            raw = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))

        assert raw["platforms"]["feishu"]["extra"]["app_id"] == "cli_xxx"
        # The migration-write invariant strips schema-default values, and
        # verify_on_stop=False now IS the default — so it must NOT be
        # materialised to disk by _persist_migration.
        assert "verify_on_stop" not in raw.get("agent", {})
        assert raw["agent"]["max_turns"] == 60
        assert raw["_config_version"] == 32

    def test_v30_to_latest_migration_keeps_platforms(self, tmp_path):
        """End-to-end: reporter's v30 feishu profile survives version bump."""
        body = """_config_version: 30
model:
  default: deepseek-v4-pro
  provider: deepseek
agent:
  max_turns: 60
platforms:
  feishu:
    enabled: true
    extra:
      app_id: cli_xxx
      app_secret: xxx
feishu:
  require_mention: true
"""
        (tmp_path / "config.yaml").write_text(body, encoding="utf-8")
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            migrate_config(interactive=False, quiet=True)
            raw = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))

        assert raw["platforms"]["feishu"]["extra"]["app_id"] == "cli_xxx"
        assert raw["feishu"]["require_mention"] is True


class TestVerifyOnStopMigration:
    """v30 → v31: switch verify_on_stop OFF once, preserving explicit choices."""

    def _write(self, tmp_path, body):
        (tmp_path / "config.yaml").write_text(body, encoding="utf-8")


class TestDelegationCapUnificationMigration:
    """v32 → v33: fold deprecated max_async_children into max_concurrent_children."""

    def _write(self, tmp_path, body):
        (tmp_path / "config.yaml").write_text(body, encoding="utf-8")


    def test_no_delegation_section_is_noop(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            self._write(tmp_path, "_config_version: 32\nmodel:\n  provider: openrouter\n")
            migrate_config(interactive=False, quiet=True)
            raw = yaml.safe_load((tmp_path / "config.yaml").read_text())
        # Migration must not materialize a delegation section it never had.
        assert "delegation" not in raw


class TestBackgroundNotificationsConciseMigration:
    """v34 → v35: move users on the old implicit default 'all' to 'concise'."""

    def _write(self, tmp_path, body):
        (tmp_path / "config.yaml").write_text(body, encoding="utf-8")

    def test_all_becomes_concise(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            self._write(
                tmp_path,
                "_config_version: 34\n"
                "display:\n"
                "  background_process_notifications: all\n",
            )
            migrate_config(interactive=False, quiet=True)
            raw = yaml.safe_load((tmp_path / "config.yaml").read_text())
        assert raw["display"]["background_process_notifications"] == "concise"

    def test_explicit_choices_preserved(self, tmp_path):
        # NOTE: bare `off` in YAML parses as boolean False — the gateway mode
        # loader maps False → "off", and the migration must leave it alone.
        for written, expected in (
            ("off", False), ("result", "result"),
            ("error", "error"), ("concise", "concise"),
        ):
            with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
                self._write(
                    tmp_path,
                    "_config_version: 34\n"
                    "display:\n"
                    f"  background_process_notifications: {written}\n",
                )
                migrate_config(interactive=False, quiet=True)
                raw = yaml.safe_load((tmp_path / "config.yaml").read_text())
            assert raw["display"]["background_process_notifications"] == expected

    def test_unset_key_is_not_materialized(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            self._write(tmp_path, "_config_version: 34\nmodel:\n  provider: openrouter\n")
            migrate_config(interactive=False, quiet=True)
            raw = yaml.safe_load((tmp_path / "config.yaml").read_text())
        # Unset users inherit the new default at read time; no write needed.
        assert "display" not in raw or "background_process_notifications" not in raw.get("display", {})

    def test_default_config_is_concise(self):
        assert DEFAULT_CONFIG["display"]["background_process_notifications"] == "concise"


class TestConfigNormalizationDoesNotOverwriteUserValues:
    """Regression tests for #27354."""

    def test_save_config_does_not_inject_max_turns_when_unset(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "_config_version": DEFAULT_CONFIG["_config_version"],
                    "memory": {"user_char_limit": 2200},
                }
            ),
            encoding="utf-8",
        )

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            save_config(load_config())
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        assert "max_turns" not in raw.get("agent", {})
        assert raw["memory"]["user_char_limit"] == 2200



    def test_normalize_max_turns_does_not_inject_default(self):
        result = _normalize_max_turns_config(
            {"_config_version": DEFAULT_CONFIG["_config_version"]}
        )
        assert "max_turns" not in result.get("agent", {})




class TestCodexAppServerAutoConfig:
    """codex_app_server_auto ships a default and survives migration untouched."""

    def _write(self, tmp_path, body):
        (tmp_path / "config.yaml").write_text(body, encoding="utf-8")

    def test_default_config_has_native_mode(self):
        assert DEFAULT_CONFIG["compression"]["codex_app_server_auto"] == "native"
        assert DEFAULT_CONFIG["compression"]["codex_gpt55_autoraise"] is True

    def test_preserves_existing_codex_app_server_auto_value(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            self._write(
                tmp_path,
                "_config_version: 31\n"
                "compression:\n"
                "  codex_app_server_auto: hermes\n",
            )

            migrate_config(interactive=False, quiet=True)

            raw = yaml.safe_load((tmp_path / "config.yaml").read_text())
            assert raw["compression"]["codex_app_server_auto"] == "hermes"


class TestIsProviderEnabled:
    """``is_provider_enabled`` gates ``providers.<name>`` blocks for the
    model picker, ``/models`` listings and the runtime resolver. Default
    must be ``True`` so existing configs keep working untouched."""

    def test_missing_flag_defaults_to_enabled(self):
        assert is_provider_enabled({"name": "Anthropic"}) is True


    @pytest.mark.parametrize("raw", ["true", "True", "yes", "on", "1", "anything-else"])
    def test_yaml_string_truthy_values_keep_it_enabled(self, raw):
        assert is_provider_enabled({"enabled": raw}) is True

    def test_non_dict_input_defaults_to_enabled(self):
        # Malformed entries (None, list, string) don't disappear silently —
        # the gate stays open and the existing validation paths will flag
        # them.
        assert is_provider_enabled(None) is True
        assert is_provider_enabled([]) is True
        assert is_provider_enabled("oops") is True


class TestProviderEnabledRuntimeGate:
    """Verify ``resolve_runtime_provider`` honours ``enabled: false`` for
    both custom-defined and built-in provider names. Smoke test only —
    full runtime resolution has its own fixture-heavy tests; here we
    only assert the early-exit raises a typed error."""

    def test_disabled_custom_provider_raises_valueerror(self, tmp_path, monkeypatch):
        cfg = {
            "model": {"default": "claude-sonnet-4-6", "provider": "claude-agent-sdk"},
            "providers": {
                "my-fork": {
                    "name": "my-fork",
                    "base_url": "http://127.0.0.1:9999",
                    "api_key": "not-needed",
                    "enabled": False,
                },
            },
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Bust the in-process config cache so the override picks up.
        from hermes_cli import config as cfg_mod
        cfg_mod._cached_config = None  # type: ignore[attr-defined]

        from hermes_cli.runtime_provider import resolve_runtime_provider
        with pytest.raises(ValueError, match="disabled"):
            resolve_runtime_provider(requested="my-fork")


# ---------------------------------------------------------------------------
# DEFAULT_CONFIG must not carry a duplicate "kanban" key
# ---------------------------------------------------------------------------

def test_default_config_kanban_block_not_dropped_by_duplicate_key():
    """DEFAULT_CONFIG previously declared ``"kanban"`` twice, so Python kept
    only the second literal and silently dropped the first — losing the
    ``auto_subscribe_on_create`` default. Both sets of defaults must survive.
    """
    kanban = DEFAULT_CONFIG["kanban"]
    # From the first (dropped) block:
    assert kanban.get("auto_subscribe_on_create") is True
    # From the second block:
    assert "dispatch_in_gateway" in kanban
    assert "auto_decompose" in kanban


def test_default_config_has_no_duplicate_top_level_keys():
    """Guard against any duplicate key silently shadowing a default."""
    import ast
    import hermes_cli.config as cfg_mod

    src = open(cfg_mod.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
            if "model" in keys and "kanban" in keys:  # the DEFAULT_CONFIG literal
                dupes = {k for k in keys if keys.count(k) > 1}
                assert not dupes, f"duplicate DEFAULT_CONFIG keys: {sorted(dupes)}"


class TestConfigCommandFailClosedSurface:
    """`hermes config set/unset` must exit cleanly (no traceback) when the
    fail-closed write guard refuses an unparseable config.yaml."""

    def _args(self, **kw):
        import argparse

        ns = argparse.Namespace()
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns

    def test_config_command_set_exits_cleanly_on_broken_yaml(self, tmp_path, capsys):
        from hermes_cli.config import config_command

        config_path = tmp_path / "config.yaml"
        original = "model:\n  default: keep\nbroken: [unterminated\n"
        config_path.write_text(original, encoding="utf-8")

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            with pytest.raises(SystemExit) as excinfo:
                config_command(
                    self._args(config_command="set", key="model.default",
                               value="gpt-4o", force=False)
                )

        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "not valid YAML" in err
        assert config_path.read_text(encoding="utf-8") == original

    def test_config_command_unset_exits_cleanly_on_broken_yaml(self, tmp_path, capsys):
        from hermes_cli.config import config_command

        config_path = tmp_path / "config.yaml"
        original = "model:\n  default: keep\nbroken: [unterminated\n"
        config_path.write_text(original, encoding="utf-8")

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            with pytest.raises(SystemExit) as excinfo:
                config_command(self._args(config_command="unset", key="model.default"))

        assert excinfo.value.code == 1
        assert "not valid YAML" in capsys.readouterr().err
        assert config_path.read_text(encoding="utf-8") == original
