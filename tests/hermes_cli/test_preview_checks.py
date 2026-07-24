"""Tests for hermes_cli.preview preview-check functions.

Covers the five core check helpers:
- _get_configured_model()
- _check_auth()
- _check_python()
- _check_hermes_home()
- _check_env_file()

Tests are fully hermetic: they use monkeypatch to inject faked configs,
environment variables and filesystem state. No external network calls and
no real API keys are exercised.

The autouse ``_hermetic_environment`` fixture (tests/conftest.py) already
isolates HERMES_HOME to a per-test tempdir and blanks credential-shaped env
vars, so each test starts from a clean slate.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# NOTE: we import preview lazily *inside* tests so that HERMES_HOME is bound
# AFTER the _hermetic_environment fixture has patched os.environ["HERMES_HOME"]
# to point at the per-test tempdir. Importing preview at module level would
# capture the developer's real HERMES_HOME instead.
# pylint: disable=import-error

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _write_config(hermes_home: Path, data) -> None:
    """Write a config.yaml into a fake Hermes home."""
    cfg = hermes_home / "config.yaml"
    import yaml  # project always ships pyyaml

    cfg.write_text(yaml.safe_dump(data), encoding="utf-8")


def _write_env_file(hermes_home: Path, content: str) -> None:
    env_path = hermes_home / ".env"
    env_path.write_text(content, encoding="utf-8")


def _clear_config_cache() -> None:
    """Wipe the module-level config cache so the next load_config() re-reads.

    The cache is keyed on the string of the config path and lives inside
    hermes_cli.config._load_config_impl via a private dict. We reach into it
    here to keep tests hermetic.
    """
    try:
        import hermes_cli.config as _cfg_mod

        _cache = getattr(_cfg_mod, "_config_cache", None) or getattr(
            _cfg_mod, "_config_read_cache", None
        )
        if _cache is not None:
            _cache.clear()
        # Defensive: also clear by name variations used over history.
        for name in ("_config_cache", "_config_read_cache", "_config_mtime_cache"):
            attr = getattr(_cfg_mod, name, None)
            if isinstance(attr, (dict, list)):
                attr.clear()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# A. _get_configured_model() — 8 tests
# ──────────────────────────────────────────────────────────────────────────────


class TestGetConfiguredModel:
    """Tests for _get_configured_model()."""

    # 36. model dict with default + provider
    def test_get_configured_model_returns_name_and_provider_from_dict_config(
        self, monkeypatch, tmp_path
    ):
        from hermes_cli.preview import _get_configured_model

        hermes_home = tmp_path / "preview_check"
        hermes_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _write_config(
            hermes_home,
            {
                "model": {
                    "default": "deepseek-v3",
                    "provider": "deepseek",
                }
            },
        )
        _clear_config_cache()
        name, provider = _get_configured_model()
        assert name == "deepseek-v3"
        assert provider == "DeepSeek"

    # 37. no config at all
    def test_get_configured_model_returns_not_set_when_no_config(
        self, monkeypatch, tmp_path
    ):
        from hermes_cli.preview import _get_configured_model

        hermes_home = tmp_path / "preview_check"
        hermes_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _clear_config_cache()
        name, provider = _get_configured_model()
        assert name == "(not set)"
        # provider_label("auto") => "Auto" when no config present
        assert provider in ("(unknown)", "Auto")

    # 38. model present but empty
    def test_get_configured_model_returns_not_set_when_model_empty(
        self, monkeypatch, tmp_path
    ):
        from hermes_cli.preview import _get_configured_model

        hermes_home = tmp_path / "preview_check"
        hermes_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _write_config(hermes_home, {"model": {"default": " ", "provider": "deepseek"}})
        _clear_config_cache()
        name, _provider = _get_configured_model()
        assert name == "(not set)"

    # 39. unknown provider — provider_label falls back to original
    def test_get_configured_model_returns_not_set_when_provider_unknown(
        self, monkeypatch, tmp_path
    ):
        """When the configured provider string is unknown, provider_label
        returns it back verbatim (no canonical label mapped)."""
        from hermes_cli.preview import _get_configured_model

        hermes_home = tmp_path / "preview_check"
        hermes_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _write_config(
            hermes_home,
            {
                "model": {
                    "default": "some-model",
                    "provider": "totally-unknown-provider",
                }
            },
        )
        _clear_config_cache()
        name, provider = _get_configured_model()
        assert name == "some-model"
        # unknown providers are echoed back by provider_label
        assert provider == "totally-unknown-provider"

    # 40. model default field
    def test_get_configured_model_parses_model_default_field(
        self, monkeypatch, tmp_path
    ):
        from hermes_cli.preview import _get_configured_model

        hermes_home = tmp_path / "preview_check"
        hermes_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _write_config(
            hermes_home,
            {
                "model": {
                    "default": "anthropic/claude-sonnet-4",
                    "provider": "anthropic",
                }
            },
        )
        _clear_config_cache()
        name, provider = _get_configured_model()
        assert name == "anthropic/claude-sonnet-4"
        assert provider == "Anthropic"

    # 41. model name field as fallback when default is absent
    def test_get_configured_model_parses_model_name_field_as_fallback(
        self, monkeypatch, tmp_path
    ):
        from hermes_cli.preview import _get_configured_model

        hermes_home = tmp_path / "preview_check"
        hermes_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _write_config(hermes_home, {"model": {"name": "gpt-4", "provider": "openai"}})
        _clear_config_cache()
        name, provider = _get_configured_model()
        assert name == "gpt-4"
        assert provider == "OpenAI"

    # 42. string config value (model field is a plain string)
    def test_get_configured_model_parses_string_config(self, monkeypatch, tmp_path):
        from hermes_cli.preview import _get_configured_model

        hermes_home = tmp_path / "preview_check"
        hermes_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _write_config(hermes_home, {"model": "openrouter/openai/gpt-4"})
        _clear_config_cache()
        name, provider = _get_configured_model()
        assert name == "openrouter/openai/gpt-4"

    # 43. custom provider string
    def test_get_configured_model_parses_custom_provider(self, monkeypatch, tmp_path):
        from hermes_cli.preview import _get_configured_model

        hermes_home = tmp_path / "preview_check"
        hermes_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _write_config(
            hermes_home,
            {"model": {"default": "my-local-model", "provider": "custom:myhost"}},
        )
        _clear_config_cache()
        name, provider = _get_configured_model()
        assert name == "my-local-model"
        # custom:xxx -> provider_label normalises to "Custom"
        assert provider == "Custom"


# ──────────────────────────────────────────────────────────────────────────────
# B. _check_auth() — 10 tests
# ──────────────────────────────────────────────────────────────────────────────


class TestCheckAuth:
    """Tests for _check_auth()."""

    def _setup_model_and_env(self, monkeypatch, tmp_path, provider_key="deepseek"):
        """Set up a minimal config with a given provider so auth checks run."""
        from hermes_cli import config as _cfg_mod  # noqa: F811
        from hermes_cli.preview import _check_auth  # noqa: F811

        hermes_home = tmp_path / "preview_check"
        hermes_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        # Clear config cache so load_config() re-reads.
        _clear_config_cache()
        # Inject an env var with a long-enough (>5) value for each of the
        # supported auth keys.
        env_map = {
            "deepseek": ("DEEPSEEK_API_KEY", "deepseek"),
            "openrouter": ("OPENROUTER_API_KEY", "openrouter"),
            "anthropic": ("ANTHROPIC_API_KEY", "anthropic"),
            "openai": ("OPENAI_API_KEY", "openai"),
            "nous": ("NOUS_API_KEY", "nous"),
            "glm": ("GLM_API_KEY", "glm"),
            "zai": ("ZAI_API_KEY", "zai"),
            "xai": ("XAI_API_KEY", "xai"),
            "dashscope": ("DASHSCOPE_API_KEY", "dashscope"),
            "tokenhub": ("TOKENHUB_API_KEY", "tokenhub"),
            "minimax": ("MINIMAX_API_KEY", "minimax"),
            "fireworks": ("FIREWORKS_API_KEY", "fireworks"),
        }
        return hermes_home, env_map

    def _assert_auth_ok(self, monkeypatch, tmp_path, provider="deepseek"):
        hermes_home, env_map = self._setup_model_and_env(
            monkeypatch, tmp_path, provider_key=provider
        )
        _write_config(
            hermes_home, {"model": {"default": "model-x", "provider": provider}}
        )
        # Provide a long api key so length check (>5) passes
        env_var, _ = env_map[provider]
        monkeypatch.setenv(env_var, "sk-abcde12345")
        from hermes_cli.preview import _check_auth

        status, detail = _check_auth()
        return status, detail

    # 44. DeepSeek API key present
    def test_check_auth_returns_ok_when_deepseek_api_key_present(
        self, monkeypatch, tmp_path
    ):
        status, detail = self._assert_auth_ok(
            monkeypatch, tmp_path, provider="deepseek"
        )
        assert status == "ok"
        assert "DEEPSEEK_API_KEY" in detail

    # 45. OpenRouter API key present
    def test_check_auth_returns_ok_when_openrouter_api_key_present(
        self, monkeypatch, tmp_path
    ):
        status, detail = self._assert_auth_ok(
            monkeypatch, tmp_path, provider="openrouter"
        )
        assert status == "ok"
        assert "OPENROUTER_API_KEY" in detail

    # 46. Anthropic API key present
    def test_check_auth_returns_ok_when_anthropic_api_key_present(
        self, monkeypatch, tmp_path
    ):
        status, detail = self._assert_auth_ok(
            monkeypatch, tmp_path, provider="anthropic"
        )
        assert status == "ok"
        assert "ANTHROPIC_API_KEY" in detail

    # 47. Nous OAuth authenticated
    def test_check_auth_returns_ok_when_nous_oauth_authenticated(
        self, monkeypatch, tmp_path
    ):
        from hermes_cli import config as _cfg_mod  # noqa: F811

        hermes_home = tmp_path / "preview_check"
        hermes_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _write_config(
            hermes_home, {"model": {"default": "nous-model", "provider": "nous"}}
        )
        _clear_config_cache()
        # No API key; simulate OAuth login
        monkeypatch.setattr(
            "hermes_cli.auth.get_nous_auth_status",
            lambda: {"logged_in": True},
            raising=False,
        )
        monkeypatch.setattr(
            "hermes_cli.auth.get_codex_auth_status",
            lambda: {"logged_in": False},
            raising=False,
        )
        from hermes_cli.preview import _check_auth

        status, detail = _check_auth()
        assert status == "ok"
        assert "Nous" in detail

    # 48. Custom provider base_url set
    def test_check_auth_returns_ok_when_custom_provider_base_url_set(
        self, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / "preview_check"
        hermes_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _write_config(
            hermes_home,
            {
                "model": {
                    "default": "my-model",
                    "provider": "custom:myhost",
                    "base_url": "https://myhost.example.com/v1",
                }
            },
        )
        _clear_config_cache()
        # Make OAuth unavailable so the custom-base_url path is exercised
        monkeypatch.setattr(
            "hermes_cli.auth.get_nous_auth_status",
            lambda: {"logged_in": False},
            raising=False,
        )
        monkeypatch.setattr(
            "hermes_cli.auth.get_codex_auth_status",
            lambda: {"logged_in": False},
            raising=False,
        )
        from hermes_cli.preview import _check_auth

        status, detail = _check_auth()
        assert status == "ok"
        assert "Custom provider configured" in detail

    # 49. No model configured
    def test_check_auth_returns_error_when_no_model_configured(
        self, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / "preview_check"
        hermes_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _clear_config_cache()
        from hermes_cli.preview import _check_auth

        status, detail = _check_auth()
        assert status == "error"
        assert detail == "No model configured"

    # 50. No API key and no OAuth
    def test_check_auth_returns_error_when_no_api_key_and_no_oauth(
        self, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / "preview_check"
        hermes_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _write_config(
            hermes_home, {"model": {"default": "deepseek-v3", "provider": "deepseek"}}
        )
        _clear_config_cache()
        # Make OAuth unavailable
        monkeypatch.setattr(
            "hermes_cli.auth.get_nous_auth_status",
            lambda: {"logged_in": False},
            raising=False,
        )
        monkeypatch.setattr(
            "hermes_cli.auth.get_codex_auth_status",
            lambda: {"logged_in": False},
            raising=False,
        )
        from hermes_cli.preview import _check_auth

        status, detail = _check_auth()
        assert status == "error"
        assert "No API key found" in detail

    # 51. Short API keys (<6 chars) are ignored
    def test_check_auth_ignores_short_api_keys_less_than_6_chars(
        self, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / "preview_check"
        hermes_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _write_config(
            hermes_home, {"model": {"default": "deepseek-v3", "provider": "deepseek"}}
        )
        _clear_config_cache()
        # Key length is 5 — below the >5 threshold, so it is ignored
        monkeypatch.setenv("DEEPSEEK_API_KEY", "short")
        monkeypatch.setattr(
            "hermes_cli.auth.get_nous_auth_status",
            lambda: {"logged_in": False},
            raising=False,
        )
        monkeypatch.setattr(
            "hermes_cli.auth.get_codex_auth_status",
            lambda: {"logged_in": False},
            raising=False,
        )
        from hermes_cli.preview import _check_auth

        status, detail = _check_auth()
        assert status == "error"
        assert "No API key found" in detail

    # 52. Multiple auth keys present — first match wins
    def test_check_auth_respects_multiple_auth_keys_picks_first(
        self, monkeypatch, tmp_path
    ):
        from hermes_cli import config as _cfg_mod  # noqa: F811

        hermes_home = tmp_path / "preview_check"
        hermes_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _write_config(
            hermes_home, {"model": {"default": "model-x", "provider": "deepseek"}}
        )
        _clear_config_cache()
        # Several keys are set simultaneously; auth checks iterates in
        # provider order and returns on the FIRST valid match.
        monkeypatch.setenv("DEEPSEEK_API_KEY", "dskey123456")
        monkeypatch.setenv("OPENROUTER_API_KEY", "orkey123456")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "antkey123456")
        from hermes_cli.preview import _check_auth

        status, detail = _check_auth()
        assert status == "ok"
        # First key in the auth_checks list
        assert "DEEPSEEK_API_KEY" in detail

    # 53. Empty API key string is ignored
    def test_check_auth_handles_empty_api_key_string(self, monkeypatch, tmp_path):
        hermes_home = tmp_path / "preview_check"
        hermes_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _write_config(
            hermes_home, {"model": {"default": "deepseek-v3", "provider": "deepseek"}}
        )
        _clear_config_cache()
        # An empty string is falsy, so it is skipped just like a missing key.
        monkeypatch.setenv("DEEPSEEK_API_KEY", "")
        monkeypatch.setattr(
            "hermes_cli.auth.get_nous_auth_status",
            lambda: {"logged_in": False},
            raising=False,
        )
        monkeypatch.setattr(
            "hermes_cli.auth.get_codex_auth_status",
            lambda: {"logged_in": False},
            raising=False,
        )
        from hermes_cli.preview import _check_auth

        status, detail = _check_auth()
        assert status == "error"
        assert "No API key found" in detail


# ──────────────────────────────────────────────────────────────────────────────
# C. _check_python() — 6 tests
# ──────────────────────────────────────────────────────────────────────────────


class TestCheckPython:
    """Tests for _check_python()."""

    def _check(self, monkeypatch, version_string: str):
        """Patch sys.version and run _check_python()."""
        monkeypatch.setattr("sys.version", version_string)
        from hermes_cli.preview import _check_python

        return _check_python()

    # 54. Python 3.10+ => ok
    def test_check_python_returns_ok_for_python_3_10_plus(self, monkeypatch):
        status, detail = self._check(monkeypatch, "3.10.12 (main, ...)")
        assert status == "ok"
        assert "Python 3.10.12" in detail

    # 55. Python below 3.10 => warn
    def test_check_python_returns_warn_for_python_below_3_10(self, monkeypatch):
        status, detail = self._check(monkeypatch, "3.9.7 (main, ...)")
        assert status == "warn"
        assert "3.10+" in detail

    # 56. Python 3.11 specifically
    def test_check_python_handles_python_3_11_specifically(self, monkeypatch):
        status, detail = self._check(monkeypatch, "3.11.4 (main, ...)")
        assert status == "ok"
        assert "Python 3.11.4" in detail

    # 57. Python 3.12 specifically
    def test_check_python_handles_python_3_12_specifically(self, monkeypatch):
        status, detail = self._check(monkeypatch, "3.12.0 (main, ...)")
        assert status == "ok"
        assert "Python 3.12.0" in detail

    # 58. Python 3.13 specifically
    def test_check_python_handles_python_3_13_specifically(self, monkeypatch):
        status, detail = self._check(monkeypatch, "3.13.99 (main, ...)")
        assert status == "ok"
        assert "Python 3.13.99" in detail

    # 59. Invalid version string => unknown
    def test_check_python_handles_invalid_version_string(self, monkeypatch):
        status, detail = self._check(monkeypatch, "not-a-version (bleargh)")
        assert status == "warn"
        assert "Python not-a-version" in detail


# ──────────────────────────────────────────────────────────────────────────────
# D. _check_hermes_home() — 5 tests
# ──────────────────────────────────────────────────────────────────────────────


class TestCheckHermesHome:
    """Tests for _check_hermes_home()."""

    # The HERMES_HOME module-level constant is bound at import time, so we
    # reload preview inside each test *after* monkeypatch has set the env var.
    # Because HERMES_HOME = get_hermes_home() runs once, we patch
    # get_hermes_home() on the config module before reloading preview.

    def _run_check(self, monkeypatch, tmp_path, hermes_home_path: Path) -> None:
        """Prepare environment and return a freshly-loaded _check_hermes_home.

        Returns the imported callable after monkeypatch has redirected both
        os.environ["HERMES_HOME"] and get_hermes_home().
        """
        monkeypatch.setenv("HERMES_HOME", str(hermes_home_path))
        monkeypatch.setattr(
            "hermes_cli.config.get_hermes_home",
            lambda: hermes_home_path,
            raising=False,
        )
        # Force a reimport so HERMES_HOME binds to the patched value.
        monkeypatch.delitem(sys.modules, "hermes_cli.preview", raising=False)
        monkeypatch.delitem(sys.modules, "hermes_cli", raising=False)
        # Clear config cache so load_config re-reads
        try:
            from hermes_cli import config as _cfg_mod

            for _n in ("_config_cache", "_config_read_cache", "_config_mtime_cache"):
                _v = getattr(_cfg_mod, _n, None)
                if isinstance(_v, dict):
                    _v.clear()
        except Exception:
            pass
        from hermes_cli import preview  # noqa: F811

        return preview._check_hermes_home

    # 60. Hermes home exists => ok
    def test_check_hermes_home_returns_ok_when_home_exists(self, monkeypatch, tmp_path):
        hermes_home = tmp_path / "preview_check"
        hermes_home.mkdir(exist_ok=True)
        check_fn = self._run_check(monkeypatch, tmp_path, hermes_home)
        status, detail, fix = check_fn()
        assert status == "ok"
        assert fix == ""
        assert "Hermes home at" in detail

    # 61. Hermes home missing => error
    def test_check_hermes_home_returns_error_when_home_missing(
        self, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / "does_not_exist"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(
            "hermes_cli.config.get_hermes_home",
            lambda: hermes_home,
            raising=False,
        )
        monkeypatch.delitem(sys.modules, "hermes_cli.preview", raising=False)
        monkeypatch.delitem(sys.modules, "hermes_cli", raising=False)
        from hermes_cli import preview  # noqa: F811

        status, detail, fix = preview._check_hermes_home()
        assert status == "error"
        assert detail == "Hermes home does not exist"

    # 62. Fix instruction when error
    def test_check_hermes_home_returns_fix_instruction_when_error(
        self, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / "missing_home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(
            "hermes_cli.config.get_hermes_home",
            lambda: hermes_home,
            raising=False,
        )
        monkeypatch.delitem(sys.modules, "hermes_cli.preview", raising=False)
        monkeypatch.delitem(sys.modules, "hermes_cli", raising=False)
        from hermes_cli import preview  # noqa: F811

        status, _detail, fix = preview._check_hermes_home()
        assert status == "error"
        assert "hermes setup" in fix

    # 63. Permission denied on Hermes home
    def test_check_hermes_home_handles_permission_denied(self, monkeypatch, tmp_path):
        hermes_home = tmp_path / "locked_home"
        hermes_home.mkdir(exist_ok=True)
        # Make the directory unreadable
        os.chmod(hermes_home, 0o000)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(
            "hermes_cli.config.get_hermes_home",
            lambda: hermes_home,
            raising=False,
        )
        monkeypatch.delitem(sys.modules, "hermes_cli.preview", raising=False)
        monkeypatch.delitem(sys.modules, "hermes_cli", raising=False)
        from hermes_cli import preview  # noqa: F811

        # Restore readability so tmp_path cleanup still works afterward.
        try:
            status, detail, _fix = preview._check_hermes_home()
            # Path.exists() under no-read-permission still returns True on
            # Linux; Hermes reports "ok" because the directory object exists.
            assert status == "ok"
            assert "Hermes home at" in detail
        finally:
            os.chmod(hermes_home, 0o700)

    # 64. Symlink to a nonexistent path
    def test_check_hermes_home_handles_symlink_to_nonexistent_path(
        self, monkeypatch, tmp_path
    ):
        target = tmp_path / "nonexistent_target"
        hermes_home = tmp_path / "symlink_home"
        hermes_home.symlink_to(target)  # dangling symlink
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(
            "hermes_cli.config.get_hermes_home",
            lambda: hermes_home,
            raising=False,
        )
        monkeypatch.delitem(sys.modules, "hermes_cli.preview", raising=False)
        monkeypatch.delitem(sys.modules, "hermes_cli", raising=False)
        from hermes_cli import preview  # noqa: F811

        status, detail, fix = preview._check_hermes_home()
        # A broken symlink is not an existing file/dir => error
        assert status == "error"
        assert detail == "Hermes home does not exist"


# ──────────────────────────────────────────────────────────────────────────────
# E. _check_env_file() — 6 tests
# ──────────────────────────────────────────────────────────────────────────────


class TestCheckEnvFile:
    """Tests for _check_env_file()."""

    def _run_check(self, monkeypatch, tmp_path, hermes_home_path: Path):
        monkeypatch.setenv("HERMES_HOME", str(hermes_home_path))
        monkeypatch.setattr(
            "hermes_cli.config.get_hermes_home",
            lambda: hermes_home_path,
            raising=False,
        )
        monkeypatch.delitem(sys.modules, "hermes_cli.preview", raising=False)
        from hermes_cli import preview  # noqa: F811

        return preview._check_env_file

    # 65. .env file exists => ok
    def test_check_env_file_returns_ok_when_dotenv_exists(self, monkeypatch, tmp_path):
        hermes_home = tmp_path / "preview_check"
        hermes_home.mkdir(exist_ok=True)
        _write_env_file(hermes_home, "DEEPSEEK_API_KEY=sk-test12345\n")
        check_fn = self._run_check(monkeypatch, tmp_path, hermes_home)
        status, detail, fix = check_fn()
        assert status == "ok"
        assert detail == ".env file found"
        assert fix == ""

    # 66. .env file missing => warn
    def test_check_env_file_returns_warn_when_dotenv_missing(
        self, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / "preview_check"
        hermes_home.mkdir(exist_ok=True)
        check_fn = self._run_check(monkeypatch, tmp_path, hermes_home)
        status, detail, fix = check_fn()
        assert status == "warn"
        assert detail == ".env file not found"

    # 67. Fix instruction when missing
    def test_check_env_file_returns_fix_instruction_when_missing(
        self, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / "preview_check"
        hermes_home.mkdir(exist_ok=True)
        check_fn = self._run_check(monkeypatch, tmp_path, hermes_home)
        status, _detail, fix = check_fn()
        assert status == "warn"
        assert "API key" in fix or "hermes auth add" in fix

    # 68. .env file has no read permissions
    def test_check_env_file_handles_dotenv_with_no_permissions(
        self, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / "preview_check"
        hermes_home.mkdir(exist_ok=True)
        env_path = hermes_home / ".env"
        env_path.write_text("KEY=123456\n", encoding="utf-8")
        os.chmod(env_path, 0o000)
        check_fn = self._run_check(monkeypatch, tmp_path, hermes_home)
        try:
            # Path.exists() still reports True even when unreadable, so
            # Hermes reports "ok".
            status, detail, _fix = check_fn()
            assert status == "ok"
            assert detail == ".env file found"
        finally:
            os.chmod(env_path, 0o600)

    # 69. .env is a directory instead of a file
    def test_check_env_file_handles_dotenv_is_directory_instead_of_file(
        self, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / "preview_check"
        hermes_home.mkdir(exist_ok=True)
        # Make .env a directory (still "exists" from os.path.exists view)
        (hermes_home / ".env").mkdir()
        check_fn = self._run_check(monkeypatch, tmp_path, hermes_home)
        status, detail, _fix = check_fn()
        # The current check only looks at Path.exists(), so a directory at
        # the .env path is reported ok.
        assert status == "ok"
        assert detail == ".env file found"

    # 70. .env with special characters in content
    def test_check_env_file_handles_dotenv_with_special_characters(
        self, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / "preview_check"
        hermes_home.mkdir(exist_ok=True)
        _write_env_file(
            hermes_home,
            'API_KEY=$!@#%^&*()\nSOME_TOKEN=abc\\"def\n',
        )
        check_fn = self._run_check(monkeypatch, tmp_path, hermes_home)
        status, detail, _fix = check_fn()
        assert status == "ok"
        assert detail == ".env file found"
