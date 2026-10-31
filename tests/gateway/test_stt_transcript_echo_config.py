from pathlib import Path
from types import SimpleNamespace

from gateway.config import GatewayConfig, load_gateway_config
from gateway.run import GatewayRunner


def test_stt_echo_transcripts_defaults_on_for_backwards_compatibility():
    cfg = GatewayConfig.from_dict({})

    assert cfg.stt_enabled is True
    assert cfg.stt_echo_transcripts is True
    assert cfg.to_dict()["stt_echo_transcripts"] is True


def test_top_level_stt_echo_transcripts_takes_precedence():
    cfg = GatewayConfig.from_dict({
        "stt_echo_transcripts": False,
        "stt": {"echo_transcripts": True},
    })

    assert cfg.stt_echo_transcripts is False


