"""Fixture-driven unit tests for the voice platform plugin
(plugins/platforms/voice/). No live network — Daily / Deepgram / Cartesia are
all faked. Covers plugin registration + requirements, the Cartesia extra-config
parsers, the Flux event -> turn/barge-in logic in the turn loop (the
in-process barge-in is the plugin's whole point), and the 24k->16k resampler.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VOICE_DIR = _REPO_ROOT / "plugins" / "platforms" / "voice"


def _load_voice_module(name: str):
    """Load a sibling voice module (turn_loop, deepgram_flux_stt) the same
    isolated way load_plugin_adapter loads adapter.py — by explicit file path
    under a unique module name, no sys.path mutation."""
    mod_name = f"voice_mod_{name}"
    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(mod_name, _VOICE_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


_adapter = load_plugin_adapter("voice")
turn_loop = _load_voice_module("turn_loop")
flux = _load_voice_module("deepgram_flux_stt")
daily_transport = _load_voice_module("daily_transport")
cartesia = _load_voice_module("cartesia_tts")


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

class _FakeFluxSTT:
    """Yields a fixed list of normalized Flux events, then ends (so the
    turn loop's `async for` over events() returns)."""

    def __init__(self, events):
        self._events = events

    async def events(self):
        for ev in self._events:
            yield ev


class _FakeTransport:
    def __init__(self, playing: bool = False):
        self._playing = playing
        self.cleared = 0
        self.first_write_t = None

    def is_playing(self) -> bool:
        return self._playing

    def clear_output(self) -> None:
        self.cleared += 1

    def reset_write_mark(self) -> None:
        pass

    async def send_audio(self, pcm: bytes) -> None:
        pass


async def _noop_tts_factory(on_audio):  # never called: _start_turn is patched
    raise AssertionError("tts_factory should not run in _consume_flux tests")


def _make_loop(events, *, playing=False, extra=None):
    return turn_loop.VoiceTurnLoop(
        _FakeFluxSTT(events), _noop_tts_factory, _FakeTransport(playing=playing),
        extra=extra or {})


# --------------------------------------------------------------------------- #
# Plugin registration + requirements
# --------------------------------------------------------------------------- #

def test_register_registers_voice_platform():
    ctx = MagicMock()
    _adapter.register(ctx)
    ctx.register_platform.assert_called_once()
    kwargs = ctx.register_platform.call_args.kwargs
    assert kwargs["name"] == "voice"
    assert kwargs["label"] == "Voice"
    assert kwargs["required_env"] == [
        "DAILY_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY"]
    assert kwargs["pii_safe"] is True
    assert kwargs["allow_update_command"] is False
    for fn in ("check_fn", "validate_config", "is_connected", "adapter_factory"):
        assert callable(kwargs[fn])


def test_adapter_factory_returns_voice_adapter():
    ctx = MagicMock()
    _adapter.register(ctx)
    factory = ctx.register_platform.call_args.kwargs["adapter_factory"]
    adapter = factory(PlatformConfig(enabled=True, extra={}))
    assert isinstance(adapter, _adapter.VoiceAdapter)


def test_check_requirements_needs_all_three_keys(monkeypatch):
    monkeypatch.setattr(_adapter, "_daily_available", lambda: True)
    monkeypatch.setattr(_adapter, "_websockets_available", lambda: True)
    monkeypatch.setenv("DAILY_API_KEY", "dk")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg")
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    assert _adapter.check_requirements() is False
    monkeypatch.setenv("CARTESIA_API_KEY", "ck")
    assert _adapter.check_requirements() is True


def test_check_requirements_false_without_deps(monkeypatch):
    monkeypatch.setattr(_adapter, "_daily_available", lambda: False)
    monkeypatch.setattr(_adapter, "_websockets_available", lambda: True)
    monkeypatch.setenv("DAILY_API_KEY", "dk")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg")
    monkeypatch.setenv("CARTESIA_API_KEY", "ck")
    assert _adapter.check_requirements() is False


def test_validate_config_requires_daily_key(monkeypatch):
    cfg = PlatformConfig(enabled=True, extra={})
    monkeypatch.delenv("DAILY_API_KEY", raising=False)
    assert _adapter.validate_config(cfg) is False
    monkeypatch.setenv("DAILY_API_KEY", "dk")
    assert _adapter.validate_config(cfg) is True


# --------------------------------------------------------------------------- #
# Extra-config parsers
# --------------------------------------------------------------------------- #

def test_resolve_float_extra_default_and_parse():
    assert _adapter._resolve_float_extra({}, "eot_threshold", 0.7) == 0.7
    assert _adapter._resolve_float_extra({"eot_threshold": "0.5"}, "eot_threshold", 0.7) == 0.5
    # garbage falls back to the default, never raises
    assert _adapter._resolve_float_extra({"eot_threshold": "nope"}, "eot_threshold", 0.7) == 0.7


def test_resolve_opt_float_extra_returns_none_when_unset():
    assert _adapter._resolve_opt_float_extra({}, "tts_speed") is None
    assert _adapter._resolve_opt_float_extra({"tts_speed": ""}, "tts_speed") is None
    assert _adapter._resolve_opt_float_extra({"tts_speed": "1.2"}, "tts_speed") == 1.2


# --------------------------------------------------------------------------- #
# Flux event -> turn / barge-in logic (the moat)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_eager_then_end_starts_a_single_turn(monkeypatch):
    loop = _make_loop([
        {"event": "eager_end_of_turn", "transcript": "hello there"},
        {"event": "end_of_turn", "transcript": "hello there"},
    ])
    started = []
    monkeypatch.setattr(loop, "_start_turn",
                        lambda text, **k: started.append(text))
    await loop._consume_flux()
    # Eager starts the turn; the matching end_of_turn lets it stand (no restart).
    assert started == ["hello there"]


@pytest.mark.asyncio
async def test_end_of_turn_without_eager_starts_turn(monkeypatch):
    loop = _make_loop([
        {"event": "end_of_turn", "transcript": "what time is it"},
    ])
    started = []
    monkeypatch.setattr(loop, "_start_turn",
                        lambda text, **k: started.append(text))
    await loop._consume_flux()
    assert started == ["what time is it"]


@pytest.mark.asyncio
async def test_start_of_turn_barges_in_while_audio_playing(monkeypatch):
    # State is LISTENING but the transport is still playing the agent's tail —
    # barge-in must fire on that tail, which is the bug the is_playing() check
    # fixes.
    loop = _make_loop(
        [{"event": "start_of_turn", "transcript": "wait"}], playing=True)
    barged = []

    async def _fake_barge():
        barged.append(True)

    monkeypatch.setattr(loop, "_barge_in", _fake_barge)
    await loop._consume_flux()
    assert barged == [True]


@pytest.mark.asyncio
async def test_start_of_turn_idle_does_not_barge(monkeypatch):
    loop = _make_loop(
        [{"event": "start_of_turn", "transcript": "hi"}], playing=False)
    barged = []

    async def _fake_barge():
        barged.append(True)

    monkeypatch.setattr(loop, "_barge_in", _fake_barge)
    await loop._consume_flux()
    assert barged == []   # nothing playing -> pre-warm, not barge


@pytest.mark.asyncio
async def test_start_of_turn_respects_allow_interruptions_false(monkeypatch):
    loop = _make_loop(
        [{"event": "start_of_turn", "transcript": "wait"}],
        playing=True, extra={"allow_interruptions": False})
    barged = []

    async def _fake_barge():
        barged.append(True)

    monkeypatch.setattr(loop, "_barge_in", _fake_barge)
    await loop._consume_flux()
    assert barged == []


@pytest.mark.asyncio
async def test_turn_resumed_cancels_the_speculative_turn(monkeypatch):
    loop = _make_loop([
        {"event": "eager_end_of_turn", "transcript": "hello"},
        {"event": "turn_resumed", "transcript": ""},
    ])
    monkeypatch.setattr(loop, "_start_turn", lambda text, **k: None)
    barged = []

    async def _fake_barge():
        barged.append(True)

    monkeypatch.setattr(loop, "_barge_in", _fake_barge)
    await loop._consume_flux()
    assert barged == [True]   # the user kept talking -> kill the eager turn


def test_resolve_bool_extra():
    assert turn_loop._resolve_bool_extra({}, "k", True) is True
    assert turn_loop._resolve_bool_extra({"k": "false"}, "k", True) is False
    assert turn_loop._resolve_bool_extra({"k": "on"}, "k", False) is True
    assert turn_loop._resolve_bool_extra({"k": False}, "k", True) is False


# --------------------------------------------------------------------------- #
# Flux 24k -> 16k resampler (pure function)
# --------------------------------------------------------------------------- #

def test_resample_to_16k_reduces_sample_count():
    pcm = bytes(2400 * 2)   # 2400 s16le samples @ 24k (silence)
    out = flux._resample_to_16k(pcm, 24000)
    # 24k -> 16k is a 2/3 ratio: 2400 -> 1600 samples (3200 bytes).
    assert len(out) == 1600 * 2


def test_resample_to_16k_interpolates_a_ramp():
    import array
    src = array.array("h", [0, 300, 600, 900])   # 4 samples
    out = array.array("h")
    out.frombytes(flux._resample_to_16k(src.tobytes(), 24000))
    # 4 -> round(4*2/3)=3 samples, endpoints preserved, middle interpolated.
    assert len(out) == 3
    assert out[0] == 0 and out[-1] == 900
    assert 0 < out[1] < 900


def test_resample_to_16k_passthrough_at_16k():
    pcm = b"\x01\x00\x02\x00\x03\x00"
    assert flux._resample_to_16k(pcm, 16000) == pcm


def test_resample_to_16k_empty():
    assert flux._resample_to_16k(b"", 24000) == b""


# --------------------------------------------------------------------------- #
# Flux wire-message normalization (the event contract the turn loop consumes)
# --------------------------------------------------------------------------- #

def test_flux_normalize_start_of_turn():
    out = flux._normalize_flux_message(
        {"type": "TurnInfo", "event": "StartOfTurn", "transcript": "hi",
         "end_of_turn_confidence": 0.2})
    assert out["event"] == "start_of_turn"
    assert out["transcript"] == "hi"
    assert out["confidence"] == 0.2


def test_flux_normalize_eager_and_end_events():
    eager = flux._normalize_flux_message(
        {"type": "TurnInfo", "event": "EagerEndOfTurn"})
    end = flux._normalize_flux_message(
        {"type": "TurnInfo", "event": "EndOfTurn"})
    resumed = flux._normalize_flux_message(
        {"type": "TurnInfo", "event": "TurnResumed"})
    assert eager["event"] == "eager_end_of_turn"
    assert end["event"] == "end_of_turn"
    assert resumed["event"] == "turn_resumed"


def test_flux_normalize_missing_transcript_becomes_empty():
    out = flux._normalize_flux_message(
        {"type": "TurnInfo", "event": "StartOfTurn", "transcript": None})
    assert out["transcript"] == ""


def test_flux_normalize_drops_unknown_and_non_turninfo():
    assert flux._normalize_flux_message(
        {"type": "TurnInfo", "event": "Bogus"}) is None
    assert flux._normalize_flux_message({"type": "Metadata"}) is None
    assert flux._normalize_flux_message({"type": "Connected"}) is None


# --------------------------------------------------------------------------- #
# Daily transport queue logic (the barge-in audio path — no SDK needed)
# --------------------------------------------------------------------------- #

async def _noop_audio_in(pcm: bytes) -> None:
    pass


@pytest.mark.asyncio
async def test_transport_send_audio_splits_large_pcm():
    import asyncio
    t = daily_transport.DailyTransport(asyncio.get_running_loop(), _noop_audio_in)
    big = bytes(daily_transport.OUT_CHUNK_BYTES * 3 + 100)
    await t.send_audio(big)
    sizes = []
    while not t._out_q.empty():
        sizes.append(len(t._out_q.get_nowait()))
    assert len(sizes) == 4  # 3 full chunks + a 100-byte remainder
    assert all(s <= daily_transport.OUT_CHUNK_BYTES for s in sizes)
    assert sum(sizes) == len(big)


@pytest.mark.asyncio
async def test_transport_send_audio_small_stays_one_chunk():
    import asyncio
    t = daily_transport.DailyTransport(asyncio.get_running_loop(), _noop_audio_in)
    await t.send_audio(b"\x00" * 200)
    assert t._out_q.qsize() == 1


@pytest.mark.asyncio
async def test_transport_is_playing_and_clear_output():
    import asyncio
    t = daily_transport.DailyTransport(asyncio.get_running_loop(), _noop_audio_in)
    assert t.is_playing() is False
    await t.send_audio(b"\x00" * 200)
    assert t.is_playing() is True
    t.clear_output()                      # barge-in drops queued audio
    assert t.is_playing() is False


# --------------------------------------------------------------------------- #
# Cartesia request builder (pure) + the <flush> mapping
# --------------------------------------------------------------------------- #

def test_cartesia_request_basic_payload():
    c = cartesia.CartesiaTTSClient("k", "voice-1")
    req = c._request("ctx-1", "hello there", True)
    assert req["transcript"] == "hello there"
    assert req["voice"] == {"mode": "id", "id": "voice-1"}
    assert req["output_format"]["encoding"] == "pcm_s16le"
    assert req["output_format"]["sample_rate"] == cartesia.OUTPUT_SAMPLE_RATE
    assert req["context_id"] == "ctx-1"
    assert req["continue"] is True
    assert "flush" not in req
    assert "generation_config" not in req


def test_cartesia_request_flush_maps_to_flag():
    c = cartesia.CartesiaTTSClient("k", "voice-1")
    req = c._request("ctx-1", "", True, flush=True)
    assert req["flush"] is True
    assert req["transcript"] == ""


def test_cartesia_request_carries_generation_config():
    c = cartesia.CartesiaTTSClient("k", "voice-1", speed=1.2, emotion="calm")
    req = c._request("ctx-1", "hi", False)
    assert req["generation_config"] == {"speed": 1.2, "emotion": "calm"}
    assert req["continue"] is False


def test_cartesia_requires_a_voice_id():
    with pytest.raises(ValueError):
        cartesia.CartesiaTTSClient("k", "")
