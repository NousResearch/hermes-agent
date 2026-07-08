"""Minimal in-process Wyoming satellite for tests.

Speaks just enough of the protocol (describe/info handshake, run-pipeline,
audio streaming, transcript, TTS playback capture, ping) to exercise the
real SatelliteLink client socket path with no hardware.
"""

import asyncio

from wyoming.audio import AudioChunk
from wyoming.event import async_read_event, async_write_event
from wyoming.info import Attribution, Info, Satellite
from wyoming.pipeline import PipelineStage, RunPipeline
from wyoming.ping import Ping


def _fake_info() -> Info:
    return Info(
        satellite=Satellite(
            name="fake-satellite",
            attribution=Attribution(name="test", url="http://test"),
            installed=True,
            description="fake",
            version="1.0",
        )
    )


class FakeSatellite:
    def __init__(self):
        self.received: list = []  # every wyoming Event received from Hermes
        self.play_buffer = bytearray()  # concatenated TTS PCM
        self.run_satellite_received = asyncio.Event()
        self.transcript_received = asyncio.Event()
        self.tts_done = asyncio.Event()
        self.pong_received = asyncio.Event()
        self.port: int = 0
        self._server = None
        self._writer = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._on_conn, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._writer is not None:
            self._writer.close()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _on_conn(self, reader, writer) -> None:
        self._writer = writer
        while True:
            event = await async_read_event(reader)
            if event is None:
                break
            self.received.append(event)
            if event.type == "describe":
                await async_write_event(_fake_info().event(), writer)
            elif event.type == "run-satellite":
                self.run_satellite_received.set()
            elif event.type == "transcript":
                self.transcript_received.set()
            elif event.type == "audio-chunk":
                self.play_buffer.extend(event.payload or b"")
            elif event.type == "audio-stop":
                self.tts_done.set()
            elif event.type == "pong":
                self.pong_received.set()

    async def send_ping(self) -> None:
        await async_write_event(Ping().event(), self._writer)

    async def wake_only(self) -> None:
        """Send just the run-pipeline (local wake) with no mic audio."""
        await async_write_event(
            RunPipeline(
                start_stage=PipelineStage.ASR, end_stage=PipelineStage.TTS
            ).event(),
            self._writer,
        )

    async def wake_and_stream(
        self, pcm: bytes, rate: int = 16000, chunk_bytes: int = 3200
    ) -> None:
        """Simulate a local wake-word detection followed by mic audio."""
        await async_write_event(
            RunPipeline(
                start_stage=PipelineStage.ASR, end_stage=PipelineStage.TTS
            ).event(),
            self._writer,
        )
        for i in range(0, len(pcm), chunk_bytes):
            await async_write_event(
                AudioChunk(
                    rate=rate, width=2, channels=1, audio=pcm[i : i + chunk_bytes]
                ).event(),
                self._writer,
            )
