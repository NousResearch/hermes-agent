"""Tests for the clipboard auto-attach gate (#23984).

Pin the contract of ``hermes_cli.clipboard.is_clipboard_auto_attach_enabled``
plus the smaller env / config coercion helpers it composes. Also covers
the precedence rule (env override beats config beats default).

The companion end-to-end regression for #23984 lives in
``tests/tools/test_clipboard_auto_attach_wiring.py``; this file pins
the unit contract in isolation so behaviour does not regress silently
if the surrounding clipboard plumbing is refactored.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# _coerce_truthy_env
# ---------------------------------------------------------------------------

class TestCoerceTruthyEnv:
    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", " yes ", "on", "Y", "T"])
    def test_truthy_values_return_true(self, raw):
        from hermes_cli.clipboard import _coerce_truthy_env
        assert _coerce_truthy_env(raw) is True

    @pytest.mark.parametrize("raw", ["0", "false", "FALSE", " no ", "off", "N", "F"])
    def test_falsy_values_return_false(self, raw):
        from hermes_cli.clipboard import _coerce_truthy_env
        assert _coerce_truthy_env(raw) is False

    @pytest.mark.parametrize("raw", [None, "", "  ", "maybe", "2", "definitely"])
    def test_unknown_values_return_none(self, raw):
        from hermes_cli.clipboard import _coerce_truthy_env
        assert _coerce_truthy_env(raw) is None


# ---------------------------------------------------------------------------
# _coerce_config_bool
# ---------------------------------------------------------------------------

class TestCoerceConfigBool:
    @pytest.mark.parametrize("raw,expected", [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("yes", True),
        ("no", False),
        ("true", True),
        ("false", False),
    ])
    def test_recognised_values_coerce(self, raw, expected):
        from hermes_cli.clipboard import _coerce_config_bool
        assert _coerce_config_bool(raw) is expected

    @pytest.mark.parametrize("raw", [None, "", "  ", "maybe", [], {}, object()])
    def test_unknown_values_return_none(self, raw):
        from hermes_cli.clipboard import _coerce_config_bool
        assert _coerce_config_bool(raw) is None


# ---------------------------------------------------------------------------
# is_clipboard_auto_attach_enabled — env precedence
# ---------------------------------------------------------------------------

class TestEnvPrecedence:
    def test_env_disable_truthy_wins_over_config_enable(self):
        from hermes_cli.clipboard import is_clipboard_auto_attach_enabled

        cfg = {"clipboard": {"auto_attach_image": True}}
        env = {"HERMES_DISABLE_CLIPBOARD_AUTO_ATTACH": "1"}
        assert is_clipboard_auto_attach_enabled(cfg=cfg, env=env) is False

    def test_env_disable_zero_forces_enable_even_if_config_disables(self):
        """Explicit `HERMES_DISABLE_...=0` is a force-enable knob."""
        from hermes_cli.clipboard import is_clipboard_auto_attach_enabled

        cfg = {"clipboard": {"auto_attach_image": False}}
        env = {"HERMES_DISABLE_CLIPBOARD_AUTO_ATTACH": "0"}
        assert is_clipboard_auto_attach_enabled(cfg=cfg, env=env) is True

    @pytest.mark.parametrize("raw", ["1", "true", "yes", "on", "TRUE"])
    def test_env_disable_truthy_variants_disable(self, raw):
        from hermes_cli.clipboard import is_clipboard_auto_attach_enabled
        env = {"HERMES_DISABLE_CLIPBOARD_AUTO_ATTACH": raw}
        assert is_clipboard_auto_attach_enabled(cfg={}, env=env) is False

    @pytest.mark.parametrize("raw", ["", "maybe", "definitely", " "])
    def test_env_disable_unknown_falls_through_to_config(self, raw):
        from hermes_cli.clipboard import is_clipboard_auto_attach_enabled

        cfg = {"clipboard": {"auto_attach_image": False}}
        env = {"HERMES_DISABLE_CLIPBOARD_AUTO_ATTACH": raw}
        assert is_clipboard_auto_attach_enabled(cfg=cfg, env=env) is False


# ---------------------------------------------------------------------------
# is_clipboard_auto_attach_enabled — config-only path
# ---------------------------------------------------------------------------

class TestConfigOnlyPath:
    def test_config_enable_returns_true(self):
        from hermes_cli.clipboard import is_clipboard_auto_attach_enabled
        cfg = {"clipboard": {"auto_attach_image": True}}
        assert is_clipboard_auto_attach_enabled(cfg=cfg, env={}) is True

    def test_config_disable_returns_false(self):
        from hermes_cli.clipboard import is_clipboard_auto_attach_enabled
        cfg = {"clipboard": {"auto_attach_image": False}}
        assert is_clipboard_auto_attach_enabled(cfg=cfg, env={}) is False

    @pytest.mark.parametrize("cfg", [
        {},
        {"clipboard": {}},
        {"clipboard": {"unrelated_key": True}},
        {"other_section": {"auto_attach_image": False}},
    ])
    def test_missing_config_keys_default_to_enabled(self, cfg):
        from hermes_cli.clipboard import is_clipboard_auto_attach_enabled
        assert is_clipboard_auto_attach_enabled(cfg=cfg, env={}) is True

    def test_non_dict_clipboard_section_defaults_to_enabled(self):
        """A user typo (string instead of nested dict) must not crash."""
        from hermes_cli.clipboard import is_clipboard_auto_attach_enabled
        cfg = {"clipboard": "yes please"}
        assert is_clipboard_auto_attach_enabled(cfg=cfg, env={}) is True


# ---------------------------------------------------------------------------
# is_clipboard_auto_attach_enabled — defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_no_env_no_config_returns_true(self):
        from hermes_cli.clipboard import is_clipboard_auto_attach_enabled
        assert is_clipboard_auto_attach_enabled(cfg={}, env={}) is True

    def test_none_cfg_loads_lazily(self):
        from hermes_cli import clipboard as clip_mod

        with patch("hermes_cli.config.load_config",
                   return_value={"clipboard": {"auto_attach_image": False}}):
            assert clip_mod.is_clipboard_auto_attach_enabled(env={}) is False

    def test_none_cfg_with_failing_load_falls_back_to_default(self):
        from hermes_cli import clipboard as clip_mod

        with patch("hermes_cli.config.load_config",
                   side_effect=RuntimeError("config.yaml unreadable")):
            # Default → True (auto-attach stays on if config is broken).
            assert clip_mod.is_clipboard_auto_attach_enabled(env={}) is True

    def test_default_env_is_os_environ(self, monkeypatch):
        """When ``env=None`` we read os.environ — verify by setting it."""
        from hermes_cli.clipboard import is_clipboard_auto_attach_enabled

        monkeypatch.setenv("HERMES_DISABLE_CLIPBOARD_AUTO_ATTACH", "1")
        assert is_clipboard_auto_attach_enabled(cfg={}) is False

    def test_default_env_unset_returns_true(self, monkeypatch):
        from hermes_cli.clipboard import is_clipboard_auto_attach_enabled

        monkeypatch.delenv("HERMES_DISABLE_CLIPBOARD_AUTO_ATTACH", raising=False)
        assert is_clipboard_auto_attach_enabled(cfg={}) is True


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------

class TestModuleSurface:
    def test_helper_is_publicly_importable(self):
        from hermes_cli import clipboard as clip_mod

        assert callable(clip_mod.is_clipboard_auto_attach_enabled)

    def test_env_var_constant_matches_documented_name(self):
        """The env-var name is part of the user contract — pin it.

        If we ever rename it, this test fails on purpose so we update
        the docs / CHANGELOG.
        """
        from hermes_cli.clipboard import _AUTO_ATTACH_ENV_VAR

        assert _AUTO_ATTACH_ENV_VAR == "HERMES_DISABLE_CLIPBOARD_AUTO_ATTACH"

    def test_config_keys_constant_matches_documented_path(self):
        """The config path is part of the user contract — pin it."""
        from hermes_cli.clipboard import _AUTO_ATTACH_CONFIG_KEYS

        assert _AUTO_ATTACH_CONFIG_KEYS == ("clipboard", "auto_attach_image")
