"""Per-platform STT transcript echo configuration.

Covers the ``PlatformConfig.show_stt_transcription`` per-platform override and
the ``_should_echo_stt_transcripts(platform)`` resolution logic (including
string/string-enum platform normalization).
"""
from types import SimpleNamespace

from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner


# --- PlatformConfig.show_stt_transcription --------------------------------

def test_show_stt_transcription_default_is_none():
    assert PlatformConfig.from_dict({}).show_stt_transcription is None


def test_show_stt_transcription_parsed_from_top_level():
    pc = PlatformConfig.from_dict({"show_stt_transcription": False})
    assert pc.show_stt_transcription is False


def test_show_stt_transcription_parsed_from_extra():
    pc = PlatformConfig.from_dict({"extra": {"show_stt_transcription": False}})
    assert pc.show_stt_transcription is False


def test_show_stt_transcription_round_trips_through_to_dict():
    pc = PlatformConfig.from_dict({"show_stt_transcription": False})
    assert pc.to_dict()["show_stt_transcription"] is False
    # Omitted when None (backward compatible with older serialized output).
    assert "show_stt_transcription" not in PlatformConfig.from_dict({}).to_dict()


# --- _should_echo_stt_transcripts(platform) resolution --------------------

def _runner(echo_default=True, platforms=None):
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(
        stt_echo_transcripts=echo_default,
        platforms=platforms or {},
    )
    return runner


def test_echo_uses_global_default_when_no_override():
    runner = _runner(echo_default=True)
    assert runner._should_echo_stt_transcripts() is True
    assert runner._should_echo_stt_transcripts(Platform.SIGNAL) is True
    assert runner._should_echo_stt_transcripts(Platform.TELEGRAM) is True


def test_echo_global_off_applies_to_all_when_no_override():
    runner = _runner(echo_default=False)
    assert runner._should_echo_stt_transcripts(Platform.SIGNAL) is False


def test_echo_per_platform_false_overrides_global_true():
    runner = _runner(
        echo_default=True,
        platforms={Platform.SIGNAL: PlatformConfig.from_dict({"show_stt_transcription": False})},
    )
    assert runner._should_echo_stt_transcripts(Platform.SIGNAL) is False
    # Other platforms still use the global default.
    assert runner._should_echo_stt_transcripts(Platform.TELEGRAM) is True


def test_echo_per_platform_true_overrides_global_false():
    runner = _runner(
        echo_default=False,
        platforms={Platform.SIGNAL: PlatformConfig.from_dict({"show_stt_transcription": True})},
    )
    assert runner._should_echo_stt_transcripts(Platform.SIGNAL) is True
    assert runner._should_echo_stt_transcripts(Platform.TELEGRAM) is False


def test_echo_string_platform_id_normalizes_to_enum():
    runner = _runner(
        echo_default=True,
        platforms={Platform.SIGNAL: PlatformConfig.from_dict({"show_stt_transcription": False})},
    )
    assert runner._should_echo_stt_transcripts("signal") is False


def test_echo_unknown_string_platform_falls_back_to_global():
    runner = _runner(echo_default=True)
    assert runner._should_echo_stt_transcripts("not-a-real-platform") is True
