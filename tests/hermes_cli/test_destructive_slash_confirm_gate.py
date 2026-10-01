"""Tests for the approvals.destructive_slash_confirm config gate.

Destructive session slash commands (/clear, /new, /reset, /undo) discard
conversation state.  This config key (default False) gates a three-option
confirmation prompt — "Always Approve" flips the key to False so future
destructive commands run silently.

See gateway/run.py::_maybe_confirm_destructive_slash and
cli.py::_confirm_destructive_slash for the runtime gate.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("key", ["destructive_slash_confirm", "mcp_reload_confirm"])
@pytest.mark.parametrize("config", [{}, {"approvals": {"mode": "off"}}])
def test_missing_confirmation_setting_skips_cli_prompt(key, config, monkeypatch):
    import cli
    from hermes_cli.cli_modal_mixin import _gated_confirm

    monkeypatch.setattr(cli, "load_cli_config", lambda: config)
    instance = SimpleNamespace(_prompt_text_input_modal=Mock())

    assert _gated_confirm(
        instance, "test", key, title="test", detail="test", choices=(),
        unchanged="", always_msg="", once_verb="",
    ) == "once"
    instance._prompt_text_input_modal.assert_not_called()


class TestUserConfigMerge:
    """If a user has a pre-existing config without this key, load_config
    should fill it in from DEFAULT_CONFIG (deep merge preserves keys the
    user didn't override)."""

    def test_existing_user_config_without_key_gets_default(self, tmp_path, monkeypatch):
        import hermes_yaml as yaml

        home = tmp_path / ".hermes"
        home.mkdir()
        cfg_path = home / "config.yaml"
        legacy = {
            "approvals": {"mode": "manual", "timeout": 60, "cron_mode": "deny"},
        }
        cfg_path.write_text(yaml.safe_dump(legacy))

        monkeypatch.setenv("HERMES_HOME", str(home))
        import importlib
        import hermes_cli.config as cfg_mod
        importlib.reload(cfg_mod)

        cfg = cfg_mod.load_config()
        assert cfg["approvals"]["destructive_slash_confirm"] is False
