"""Regression tests for #71047 (Problem A): per-platform display settings.

`hermes config set platforms.<name>.<display_setting> <value>` must write to
`display.platforms.<name>.<display_setting>` — the path the gateway actually
reads (gateway/display_config.py::resolve_display_setting). Writing to the
top-level `platforms.<name>` block is silently ignored by the runtime, so the
edit appeared to succeed while having no effect.
"""

from pathlib import Path

import pytest
import yaml


def _write_config(hermes_home: Path, data: dict) -> Path:
    hermes_home.mkdir(parents=True, exist_ok=True)
    config_path = hermes_home / "config.yaml"
    config_path.write_text(yaml.dump(data))
    return config_path


def _set(monkeypatch, hermes_home, key, value, force=False):
    """Isolated call to set_config_value against a temp HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # set_config_value resolves the home live via get_config_path()/get_hermes_home()
    from hermes_cli.config import set_config_value
    set_config_value(key, value, force=force)


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    # A config that already has a top-level platforms block (connection keys)
    # AND a display.platforms block, mirroring the real-world report.
    cfg = {
        "model": {"default": "test-model", "provider": "openrouter"},
        "platforms": {
            "telegram": {"token": "secret-bot-token"},
        },
        "display": {
            "skin": "default",
            "platforms": {
                "telegram": {"show_reasoning": True},
            },
        },
    }
    _write_config(home, cfg)
    return home


class TestPerPlatformDisplayRedirect:
    def test_streaming_redirects_to_display_platforms(self, hermes_home, monkeypatch):
        """platforms.telegram.streaming must land under display.platforms."""
        _set(monkeypatch, hermes_home, "platforms.telegram.streaming", "false")

        result = yaml.safe_load((hermes_home / "config.yaml").read_text())
        # Redirected target exists and is correct
        assert result["display"]["platforms"]["telegram"]["streaming"] is False
        # Top-level platforms.telegram must NOT gain a streaming key
        assert "streaming" not in result["platforms"]["telegram"]
        # Connection key untouched
        assert result["platforms"]["telegram"]["token"] == "secret-bot-token"

    def test_show_reasoning_redirects(self, hermes_home, monkeypatch):
        _set(monkeypatch, hermes_home, "platforms.telegram.show_reasoning", "false")
        result = yaml.safe_load((hermes_home / "config.yaml").read_text())
        assert result["display"]["platforms"]["telegram"]["show_reasoning"] is False

    def test_tool_progress_redirects(self, hermes_home, monkeypatch):
        # ``off`` is coerced to False by the bool-aware coercion in
        # set_config_value; gateway/display_config._normalise turns False back
        # into the canonical "off" string at read time, so the persisted value
        # is the bool.
        _set(monkeypatch, hermes_home, "platforms.discord.tool_progress", "off")
        result = yaml.safe_load((hermes_home / "config.yaml").read_text())
        assert result["display"]["platforms"]["discord"]["tool_progress"] is False

    def test_connection_key_not_redirected(self, hermes_home, monkeypatch):
        """A real connection key (token) stays in top-level platforms.<name>."""
        _set(monkeypatch, hermes_home, "platforms.telegram.token", "new-token")
        result = yaml.safe_load((hermes_home / "config.yaml").read_text())
        assert result["platforms"]["telegram"]["token"] == "new-token"
        # Nothing leaked into display.platforms.telegram.token
        assert "token" not in result["display"]["platforms"]["telegram"]

    def test_no_top_level_platforms_created_when_missing(self, tmp_path, monkeypatch):
        """When there is no pre-existing top-level platforms block, a display
        setting write must not invent one."""
        home = tmp_path / ".hermes"
        _write_config(home, {"model": {"default": "m"}})
        _set(monkeypatch, home, "platforms.telegram.streaming", "true")
        result = yaml.safe_load((home / "config.yaml").read_text())
        assert result["display"]["platforms"]["telegram"]["streaming"] is True
        assert "platforms" not in result  # no stray top-level platforms block
