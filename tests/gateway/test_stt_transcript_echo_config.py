from pathlib import Path
from types import SimpleNamespace

from gateway.config import GatewayConfig, Platform, load_gateway_config
from gateway.run import GatewayRunner


def test_stt_echo_transcripts_defaults_on_for_backwards_compatibility():
    cfg = GatewayConfig.from_dict({})

    assert cfg.stt_enabled is True
    assert cfg.stt_echo_transcripts is True
    assert not hasattr(cfg, "stt_reply_to_transcript")
    assert cfg.to_dict()["stt_echo_transcripts"] is True
    assert "stt_reply_to_transcript" not in cfg.to_dict()


def test_telegram_reply_to_transcript_loads_as_platform_extra(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "platforms:\n"
        "  telegram:\n"
        "    enabled: true\n"
        "    token: test-token\n"
        "telegram:\n"
        "  reply_to_transcript: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    cfg = load_gateway_config()

    assert cfg.platforms[Platform.TELEGRAM].extra["reply_to_transcript"] is True


def test_platforms_telegram_reply_setting_beats_gateway_platforms(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "platforms:\n"
        "  telegram:\n"
        "    enabled: true\n"
        "    token: test-token\n"
        "    reply_to_transcript: true\n"
        "gateway:\n"
        "  platforms:\n"
        "    telegram:\n"
        "      reply_to_transcript: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    cfg = load_gateway_config()

    assert cfg.platforms[Platform.TELEGRAM].extra["reply_to_transcript"] is True


def test_telegram_reply_setting_coerces_quoted_false(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "platforms:\n"
        "  telegram:\n"
        "    enabled: true\n"
        "    token: test-token\n"
        "telegram:\n"
        "  reply_to_transcript: 'false'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    cfg = load_gateway_config()

    assert cfg.platforms[Platform.TELEGRAM].extra["reply_to_transcript"] is False


def test_top_level_stt_echo_transcripts_takes_precedence():
    cfg = GatewayConfig.from_dict({
        "stt_echo_transcripts": False,
        "stt": {"echo_transcripts": True},
    })

    assert cfg.stt_echo_transcripts is False


