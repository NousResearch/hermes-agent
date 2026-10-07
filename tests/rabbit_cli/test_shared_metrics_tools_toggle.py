"""Tests for the `rabbit tools` shared-metrics collection toggle.

Collection is local-only and reachable from a config gate, the setup prompt,
AND `rabbit tools`. These cover the third surface.
"""

from __future__ import annotations


from rabbit_cli.tools_config import (
    _configure_shared_metrics_interactive,
    _shared_metrics_menu_label,
    _shared_metrics_state,
)


def _config(**shared):
    return {"telemetry": {"shared_metrics": shared}}


class TestState:
    def test_missing_telemetry_section_is_off(self):
        assert _shared_metrics_state({}) is False

    def test_malformed_section_does_not_raise(self):
        assert _shared_metrics_state({"telemetry": "nonsense"}) is False

    def test_reads_the_flag(self):
        assert _shared_metrics_state(_config(enabled=True)) is True


class TestMenuLabel:
    def test_label_names_the_local_state(self):
        assert "collecting locally" in _shared_metrics_menu_label(_config(enabled=True))
        assert "off" in _shared_metrics_menu_label(_config(enabled=False))


class TestToggle:
    def test_no_write_when_nothing_changed(self, monkeypatch):
        config = _config(enabled=False)
        saved = []
        monkeypatch.setattr(
            "rabbit_cli.setup.prompt_yes_no", lambda *_a, **_k: False
        )
        monkeypatch.setattr(
            "rabbit_cli.tools_config.save_config", lambda cfg: saved.append(cfg)
        )
        _configure_shared_metrics_interactive(config)
        assert saved == []

    def test_disabling_collection_persists(self, monkeypatch):
        config = _config(enabled=True)
        monkeypatch.setattr(
            "rabbit_cli.setup.prompt_yes_no", lambda *_a, **_k: False
        )
        monkeypatch.setattr(
            "rabbit_cli.tools_config.save_config", lambda cfg: None
        )
        _configure_shared_metrics_interactive(config)
        assert config["telemetry"]["shared_metrics"]["enabled"] is False
