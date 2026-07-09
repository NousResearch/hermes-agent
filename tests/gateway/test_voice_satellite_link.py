"""SatelliteLink integration tests against an in-process fake satellite."""

import asyncio

import pytest
import pytest_asyncio

from tests.gateway._fake_wyoming_satellite import FakeSatellite
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_adapter = load_plugin_adapter("voice_satellite")
link_mod = _adapter._import_sibling("satellite_link")


class Recorder:
    def __init__(self):
        self.pipeline_starts = []
        self.chunks = []
        self.played = []
        self.pipeline_started = asyncio.Event()
        self.got_audio = asyncio.Event()

    async def on_pipeline_start(self, name):
        self.pipeline_starts.append(name)
        self.pipeline_started.set()

    async def on_audio_chunk(self, name, pcm, seconds, rate):
        self.chunks.append((name, pcm, seconds, rate))
        self.got_audio.set()

    async def on_played(self, name):
        self.played.append(name)


@pytest_asyncio.fixture
async def fake_and_link():
    sat = FakeSatellite()
    await sat.start()
    rec = Recorder()
    link = link_mod.SatelliteLink(
        "kitchen", "127.0.0.1", sat.port,
        on_pipeline_start=rec.on_pipeline_start,
        on_audio_chunk=rec.on_audio_chunk,
        on_played=rec.on_played,
    )
    await link.start()
    await asyncio.wait_for(sat.run_satellite_received.wait(), timeout=5)
    yield sat, link, rec
    await link.stop()
    await sat.stop()


@pytest.mark.asyncio
async def test_handshake_describe_then_run_satellite(fake_and_link):
    sat, link, rec = fake_and_link
    types = [e.type for e in sat.received]
    assert types[0] == "describe"
    assert "run-satellite" in types
    assert link.connected is True


@pytest.mark.asyncio
async def test_pipeline_and_audio_reach_callbacks(fake_and_link):
    sat, link, rec = fake_and_link
    await sat.wake_and_stream(b"\x01\x00" * 16000)  # 1s of PCM
    await asyncio.wait_for(rec.got_audio.wait(), timeout=5)
    assert rec.pipeline_starts == ["kitchen"]
    name, pcm, seconds, rate = rec.chunks[0]
    assert (name, rate) == ("kitchen", 16000)
    assert seconds == pytest.approx(len(pcm) / (16000 * 2))


@pytest.mark.asyncio
async def test_ping_gets_pong_and_transcript_and_tts_flow(fake_and_link):
    sat, link, rec = fake_and_link
    await sat.send_ping()
    await asyncio.wait_for(sat.pong_received.wait(), timeout=5)

    await link.send_transcript("hello world")
    await asyncio.wait_for(sat.transcript_received.wait(), timeout=5)

    pcm = b"\x02\x00" * 22050  # 1s @22050
    await link.play_pcm(pcm, rate=22050)
    await asyncio.wait_for(sat.tts_done.wait(), timeout=5)
    assert bytes(sat.play_buffer) == pcm


@pytest.mark.asyncio
async def test_write_after_stop_raises_connection_error(fake_and_link):
    sat, link, rec = fake_and_link
    await link.stop()
    with pytest.raises(ConnectionError):
        await link.send_transcript("too late")


@pytest.mark.asyncio
async def test_handler_exception_does_not_drop_the_link(fake_and_link):
    sat, link, rec = fake_and_link

    async def exploding(name, pcm, seconds, rate):
        rec.chunks.append((name, pcm, seconds, rate))
        rec.got_audio.set()
        raise RuntimeError("handler bug")

    link._on_audio_chunk = exploding
    await sat.wake_and_stream(b"\x01\x00" * 1600)  # one 0.1s chunk
    await asyncio.wait_for(rec.got_audio.wait(), timeout=5)
    await asyncio.sleep(0.2)  # give a would-be reconnect time to happen
    assert link.connected is True
    # link still works: a transcript write succeeds after the handler error
    await link.send_transcript("still alive")
    await asyncio.wait_for(sat.transcript_received.wait(), timeout=5)
