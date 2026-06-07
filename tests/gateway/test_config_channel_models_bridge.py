"""channel_models bridges into config.extra for all platforms; Telegram gets channel_skill_bindings."""

from gateway.config import load_gateway_config
from gateway.platforms.base import Platform


def _write(tmp_path, yaml_text):
    (tmp_path / "config.yaml").write_text(yaml_text)


def test_channel_models_bridged_for_telegram(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write(
        tmp_path,
        """
telegram:
  channel_models:
    "100": "anthropic/claude-opus-4-8"
""",
    )
    cfg = load_gateway_config()
    assert cfg.platforms[Platform.TELEGRAM].extra["channel_models"]["100"] == "anthropic/claude-opus-4-8"


def test_channel_models_bridged_for_discord(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write(
        tmp_path,
        """
discord:
  channel_models:
    "200": "openai/gpt-5"
""",
    )
    cfg = load_gateway_config()
    assert cfg.platforms[Platform.DISCORD].extra["channel_models"]["200"] == "openai/gpt-5"


def test_channel_skill_bindings_bridged_for_telegram(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write(
        tmp_path,
        """
telegram:
  channel_skill_bindings:
    - id: "100"
      skills: ["a"]
""",
    )
    cfg = load_gateway_config()
    assert cfg.platforms[Platform.TELEGRAM].extra["channel_skill_bindings"][0]["id"] == "100"
