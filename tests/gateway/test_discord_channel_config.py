"""Tests for gateway config bridging of channel auto-skill settings."""


def test_top_level_discord_channel_skills_are_bridged(monkeypatch, tmp_path):
    from gateway.config import Platform, load_gateway_config

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        """
discord:
  enabled: true
  token: token
  channel_skills:
    123:
      - skill-a
      - skill-b
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)

    config = load_gateway_config()

    assert config.platforms[Platform.DISCORD].extra["channel_skills"] == {
        "123": ["skill-a", "skill-b"]
    }
