"""Tests for the approvals.destructive_slash_confirm config gate.

Destructive session slash commands (/clear, /new, /reset, /undo) discard
conversation state.  This config key (default True) gates a three-option
confirmation prompt — "Always Approve" flips the key to False so future
destructive commands run silently.

See gateway/run.py::_maybe_confirm_destructive_slash and
cli.py::_confirm_destructive_slash for the runtime gate.
"""

from __future__ import annotations

from hermes_cli.config import DEFAULT_CONFIG


class TestDestructiveSlashConfirmDefault:
    def test_default_config_has_the_key(self):
        approvals = DEFAULT_CONFIG.get("approvals")
        assert isinstance(approvals, dict)
        assert "destructive_slash_confirm" in approvals

    def test_default_is_true(self):
        # New installs confirm by default — destructive commands must not
        # silently wipe history without an explicit user "yes".
        assert DEFAULT_CONFIG["approvals"]["destructive_slash_confirm"] is True

    def test_shape_matches_other_approval_keys(self):
        approvals = DEFAULT_CONFIG["approvals"]
        assert isinstance(approvals.get("destructive_slash_confirm"), bool)
        # Sibling key shape sanity — same flat dict level as mcp_reload_confirm.
        assert isinstance(approvals.get("mcp_reload_confirm"), bool)


class TestUserConfigMerge:
    """If a user has a pre-existing config without this key, load_config
    should fill it in from DEFAULT_CONFIG (deep merge preserves keys the
    user didn't override)."""

    def test_existing_user_config_without_key_gets_default(self, tmp_path, monkeypatch):
        import yaml

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
        assert cfg["approvals"]["destructive_slash_confirm"] is True

    def test_existing_user_config_with_false_key_survives_merge(
        self, tmp_path, monkeypatch,
    ):
        """A user who clicked "Always Approve" (key=false) must keep that
        setting — the default-true value must not win on later loads.
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
                "destructive_slash_confirm": False,
            },
        }
        cfg_path.write_text(yaml.safe_dump(user_cfg))

        monkeypatch.setenv("HERMES_HOME", str(home))
        import importlib
        import hermes_cli.config as cfg_mod
        importlib.reload(cfg_mod)

        cfg = cfg_mod.load_config()
        assert cfg["approvals"]["destructive_slash_confirm"] is False
