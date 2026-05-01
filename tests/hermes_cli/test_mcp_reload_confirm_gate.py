"""Tests for the approvals.mcp_reload_confirm config gate.

When the user runs /reload-mcp, the MCP tool set is rebuilt which
invalidates the provider prompt cache for the active session.  That's
expensive on long-context / high-reasoning models.  The config gate
adds a three-option confirmation (Approve Once / Always Approve /
Cancel); "Always Approve" flips this key to false so subsequent reloads
run silently.
"""

from __future__ import annotations

from copy import deepcopy

from hermes_cli.config import DEFAULT_CONFIG


class TestMcpReloadConfirmDefault:
    def test_default_config_has_the_key(self):
        approvals = DEFAULT_CONFIG.get("approvals")
        assert isinstance(approvals, dict)
        assert "mcp_reload_confirm" in approvals

    def test_default_is_true(self):
        # New installs confirm by default — this is the safe behavior.
        assert DEFAULT_CONFIG["approvals"]["mcp_reload_confirm"] is True

    def test_shape_matches_other_approval_keys(self):
        # Same flat dict level as `mode` / `timeout` / `cron_mode`.
        approvals = DEFAULT_CONFIG["approvals"]
        assert isinstance(approvals.get("mode"), str)
        assert isinstance(approvals.get("timeout"), int)
        assert isinstance(approvals.get("cron_mode"), str)
        assert isinstance(approvals.get("mcp_reload_confirm"), bool)


class TestUserConfigMerge:
    """If a user has a pre-existing config without this key, load_config
    should fill it in from DEFAULT_CONFIG (deep merge preserves keys the
    user didn't override).
    """

    def test_existing_user_config_without_key_gets_default(self, tmp_path, monkeypatch):
        import yaml

        # Simulate a legacy user config without the new key.
        home = tmp_path / ".hermes"
        home.mkdir()
        cfg_path = home / "config.yaml"
        legacy = {
            "approvals": {"mode": "manual", "timeout": 60, "cron_mode": "deny"},
        }
        cfg_path.write_text(yaml.safe_dump(legacy))

        monkeypatch.setenv("HERMES_HOME", str(home))
        # Force a fresh reimport of config.py so the HERMES_HOME is honored.
        import importlib
        import hermes_cli.config as cfg_mod
        importlib.reload(cfg_mod)

        cfg = cfg_mod.load_config()
        assert cfg["approvals"]["mcp_reload_confirm"] is True

    def test_existing_user_config_with_false_key_survives_merge(
        self, tmp_path, monkeypatch,
    ):
        """A user who has clicked "Always Approve" (key=false) must keep
        that setting across reloads — the default_true value must not win.
        """
        import yaml

        home = tmp_path / ".hermes"
        home.mkdir()
        cfg_path = home / "config.yaml"
        user_cfg = {
            "approvals": {
                "mode": "manual",
                "timeout": 60,
                "cron_mode": "deny",
                "mcp_reload_confirm": False,
            },
        }
        cfg_path.write_text(yaml.safe_dump(user_cfg))

        monkeypatch.setenv("HERMES_HOME", str(home))
        import importlib
        import hermes_cli.config as cfg_mod
        importlib.reload(cfg_mod)

        cfg = cfg_mod.load_config()
        assert cfg["approvals"]["mcp_reload_confirm"] is False
