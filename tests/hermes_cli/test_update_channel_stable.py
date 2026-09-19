"""Source-update channel selection policy."""

import json
from unittest.mock import patch

import pytest

from hermes_cli.update_cmd import _source_update_channel


class _Args:
    def __init__(self, branch=None, channel=None):
        self.branch = branch
        self.channel = channel


class TestSourceUpdateChannel:
    def test_explicit_branch_always_wins(self):
        """--branch means main-style behavior regardless of channel config."""
        assert _source_update_channel(_Args(branch="bb/gui", channel="stable")) == "main"
        assert _source_update_channel(channel="canary", branch_explicit=True) == "main"

    def test_transient_channel_flag_wins(self):
        """--channel is the per-invocation override (--set-channel persists);
        no config read happens when it is present."""
        with patch("hermes_cli.config.require_readable_config_before_write") as load_config:
            for channel in ("stable", "main", "canary"):
                assert _source_update_channel(_Args(channel=channel)) == channel
                assert _source_update_channel(channel=channel) == channel
            load_config.assert_not_called()

    def test_per_install_record_activates(self, tmp_path, monkeypatch):
        from hermes_cli.update_channel import install_id

        root = tmp_path / "install"
        root.mkdir()
        (root / "install-stamp.json").write_text(
            '{"schemaVersion": 2, "updateMechanism": "self"}'
        )
        config = {
            "update": {"installs": {install_id(root): {"path": str(root), "channel": "stable"}}}
        }
        import hermes_cli.update_cmd as update_cmd

        monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", root)
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
        assert _source_update_channel(_Args()) == "stable"

    def test_no_record_stays_main(self, tmp_path, monkeypatch):
        root = tmp_path / "install"
        root.mkdir()
        (root / "install-stamp.json").write_text(
            '{"schemaVersion": 2, "updateMechanism": "self"}'
        )
        import hermes_cli.update_cmd as update_cmd

        monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", root)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        assert _source_update_channel(_Args()) == "main"

    def test_config_failure_never_changes_the_subscription(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("update: [broken", encoding="utf-8")
        with pytest.raises(RuntimeError, match="not valid YAML"):
            _source_update_channel(_Args())
