from pathlib import Path
from types import SimpleNamespace

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner


def test_stt_echo_transcripts_defaults_on_for_backwards_compatibility():
    cfg = GatewayConfig.from_dict({})

    assert cfg.stt_enabled is True
    assert cfg.stt_echo_transcripts is True
    assert cfg.to_dict()["stt_echo_transcripts"] is True


def test_stt_echo_transcripts_can_be_disabled_in_stt_section():
    cfg = GatewayConfig.from_dict({"stt": {"enabled": True, "echo_transcripts": False}})

    assert cfg.stt_enabled is True
    assert cfg.stt_echo_transcripts is False


def test_top_level_stt_echo_transcripts_takes_precedence():
    cfg = GatewayConfig.from_dict({
        "stt_echo_transcripts": False,
        "stt": {"echo_transcripts": True},
    })

    assert cfg.stt_echo_transcripts is False


def test_gateway_runner_uses_stt_echo_transcripts_flag():
    runner = GatewayRunner.__new__(GatewayRunner)

    runner.config = SimpleNamespace(stt_echo_transcripts=False)
    assert runner._should_echo_stt_transcripts() is False

    runner.config = SimpleNamespace(stt_echo_transcripts=True)
    assert runner._should_echo_stt_transcripts() is True

    runner.config = SimpleNamespace()
    assert runner._should_echo_stt_transcripts() is True


def test_all_gateway_transcript_echo_sends_are_gated():
    source = Path(__file__).resolve().parents[2] / "gateway" / "run.py"
    lines = source.read_text().splitlines()

    echo_send_lines = [
        index
        for index, line in enumerate(lines)
        if "f'🎙️" in line or 'f"🎙️' in line
    ]

    assert echo_send_lines
    for index in echo_send_lines:
        context = "\n".join(lines[max(0, index - 12): index + 1])
        assert "_should_echo_stt_transcripts()" in context
