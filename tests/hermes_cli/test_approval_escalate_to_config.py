"""Tests for the approvals.escalate_to config key."""

from __future__ import annotations

from hermes_cli.config import DEFAULT_CONFIG


class TestApprovalEscalateToDefault:
    def test_default_config_has_the_key(self):
        approvals = DEFAULT_CONFIG.get("approvals")
        assert isinstance(approvals, dict)
        assert "escalate_to" in approvals

    def test_default_is_empty_string(self):
        assert DEFAULT_CONFIG["approvals"]["escalate_to"] == ""

    def test_shape_matches_other_approval_keys(self):
        approvals = DEFAULT_CONFIG["approvals"]
        assert isinstance(approvals.get("mode"), str)
        assert isinstance(approvals.get("timeout"), int)
        assert isinstance(approvals.get("cron_mode"), str)
        assert isinstance(approvals.get("escalate_to"), str)


class TestUserConfigMerge:
    def test_existing_user_config_without_key_gets_default(
        self, tmp_path, monkeypatch
    ):
        import importlib
        import yaml

        home = tmp_path / ".hermes"
        home.mkdir()
        cfg_path = home / "config.yaml"
        legacy = {
            "approvals": {"mode": "manual", "timeout": 60, "cron_mode": "deny"},
        }
        cfg_path.write_text(yaml.safe_dump(legacy))

        monkeypatch.setenv("HERMES_HOME", str(home))
        import hermes_cli.config as cfg_mod

        importlib.reload(cfg_mod)

        cfg = cfg_mod.load_config()
        assert cfg["approvals"]["escalate_to"] == ""

    def test_existing_user_config_with_target_survives_merge(
        self, tmp_path, monkeypatch
    ):
        import importlib
        import yaml

        home = tmp_path / ".hermes"
        home.mkdir()
        cfg_path = home / "config.yaml"
        user_cfg = {
            "approvals": {
                "mode": "manual",
                "timeout": 60,
                "cron_mode": "deny",
                "escalate_to": "telegram:operator-chat",
            },
        }
        cfg_path.write_text(yaml.safe_dump(user_cfg))

        monkeypatch.setenv("HERMES_HOME", str(home))
        import hermes_cli.config as cfg_mod

        importlib.reload(cfg_mod)

        cfg = cfg_mod.load_config()
        assert cfg["approvals"]["escalate_to"] == "telegram:operator-chat"
