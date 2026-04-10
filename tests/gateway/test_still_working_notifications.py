"""Tests for configurable gateway still-working notifications.

The gateway emits periodic "Still working..." heartbeats for long-running agent
turns. These tests cover the config loader that controls the per-platform
interval or disables heartbeats entirely.
"""

import pytest

from gateway.run import GatewayRunner


class TestLoadStillWorkingInterval:
    def test_defaults_to_600_seconds(self, monkeypatch, tmp_path):
        import gateway.run as gw

        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        assert GatewayRunner._load_still_working_interval("telegram") == 600.0

    def test_reads_global_interval_from_config(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "display:\n  still_working_interval: 300\n",
            encoding="utf-8",
        )
        import gateway.run as gw

        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        assert GatewayRunner._load_still_working_interval("telegram") == 300.0

    def test_platform_override_wins(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "display:\n"
            "  still_working_interval: 600\n"
            "  still_working_overrides:\n"
            "    signal: off\n"
            "    telegram: 120\n",
            encoding="utf-8",
        )
        import gateway.run as gw

        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        assert GatewayRunner._load_still_working_interval("signal") is None
        assert GatewayRunner._load_still_working_interval("telegram") == 120.0

    @pytest.mark.parametrize(
        "raw_yaml",
        [
            "display:\n  still_working_interval: 0\n",
            "display:\n  still_working_interval: false\n",
            "display:\n  still_working_interval: off\n",
        ],
    )
    def test_zero_false_or_off_disables_globally(self, monkeypatch, tmp_path, raw_yaml):
        (tmp_path / "config.yaml").write_text(raw_yaml, encoding="utf-8")
        import gateway.run as gw

        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        assert GatewayRunner._load_still_working_interval("telegram") is None

    def test_invalid_value_defaults_to_600_seconds(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "display:\n  still_working_interval: banana\n",
            encoding="utf-8",
        )
        import gateway.run as gw

        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        assert GatewayRunner._load_still_working_interval("telegram") == 600.0

    def test_invalid_platform_override_inherits_global_interval(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "display:\n"
            "  still_working_interval: 300\n"
            "  still_working_overrides:\n"
            "    signal: banana\n",
            encoding="utf-8",
        )
        import gateway.run as gw

        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        assert GatewayRunner._load_still_working_interval("signal") == 300.0

    @pytest.mark.parametrize(
        "raw_yaml",
        [
            "display:\n  still_working_interval: true\n",
            "display:\n  still_working_interval: on\n",
        ],
    )
    def test_true_or_on_global_value_maps_to_default_interval(self, monkeypatch, tmp_path, raw_yaml):
        (tmp_path / "config.yaml").write_text(raw_yaml, encoding="utf-8")
        import gateway.run as gw

        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        assert GatewayRunner._load_still_working_interval("telegram") == 600.0

    @pytest.mark.parametrize(
        "raw_value",
        ["true", "on"],
    )
    def test_true_or_on_platform_override_inherits_global_interval(self, monkeypatch, tmp_path, raw_value):
        (tmp_path / "config.yaml").write_text(
            "display:\n"
            "  still_working_interval: 300\n"
            "  still_working_overrides:\n"
            f"    signal: {raw_value}\n",
            encoding="utf-8",
        )
        import gateway.run as gw

        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        assert GatewayRunner._load_still_working_interval("signal") == 300.0
