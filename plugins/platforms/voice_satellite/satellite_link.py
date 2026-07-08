"""Persistent Wyoming TCP link to one voice satellite.

Hermes is the dialing side: wyoming-satellite devices run a TCP server
(default port 10700) and expect the pipeline host to connect, send
`describe` then `run-satellite`, answer `ping` with `pong`, and exchange
pipeline/audio events over the same connection.
"""

import asyncio
import logging
import random
from typing import Awaitable, Callable, Optional

from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info
from wyoming.ping import Ping, Pong
from wyoming.pipeline import RunPipeline
from wyoming.satellite import RunSatellite
from wyoming.snd import Played

logger = logging.getLogger(__name__)

_TTS_CHUNK_BYTES = 4096


class SatelliteLink:
    def __init__(
        self,
        name: str,
        host: str,
        port: int,
        *,
        on_pipeline_start: Callable[[str], Awaitable[None]],
        on_audio_chunk: Callable[[str, bytes, float, int], Awaitable[None]],
        on_played: Callable[[str], Awaitable[None]],
        on_disconnect: Optional[Callable[[str], Awaitable[None]]] = None,
        tts_sample_rate: int = 22050,
        reconnect_max_delay: float = 60.0,
    ):
        self.name = name
        self.host = host
        self.port = port
        self.snd_rate = tts_sample_rate
        self.connected = False
        self._on_pipeline_start = on_pipeline_start
        self._on_audio_chunk = on_audio_chunk
        self._on_played = on_played
        self._on_disconnect = on_disconnect
        self._reconnect_max_delay = reconnect_max_delay
        self._client: Optional[AsyncTcpClient] = None
        self._task: Optional[asyncio.Task] = None
        self._write_lock = asyncio.Lock()

    async def start(self) -> None:
        self._task = asyncio.create_task(
            self._run_forever(), name=f"voice-satellite-{self.name}"
        )

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self._close_client()

    async def _close_client(self) -> None:
        self.connected = False
        async with self._write_lock:
            if self._client is not None:
                try:
                    await self._client.disconnect()
                except Exception:  # noqa: BLE001 - best-effort close
                    pass
                self._client = None

    async def _run_forever(self) -> None:
        delay = 1.0
        while True:
            try:
                await self._session()
                delay = 1.0  # clean disconnect: retry soon
            except asyncio.CancelledError:
                raise
            except (OSError, ConnectionError, asyncio.IncompleteReadError) as err:
                logger.debug("[voice_satellite:%s] link error: %s", self.name, err)
            except Exception:
                logger.exception(
                    "[voice_satellite:%s] unexpected link error", self.name
                )
            finally:
                await self._close_client()
                if self._on_disconnect is not None:
                    try:
                        await self._on_disconnect(self.name)
                    except Exception:
                        logger.exception(
                            "[voice_satellite:%s] on_disconnect callback error",
                            self.name,
                        )
            await asyncio.sleep(delay + random.uniform(0, delay / 2))
            delay = min(delay * 2, self._reconnect_max_delay)

    async def _session(self) -> None:
        client = AsyncTcpClient(self.host, self.port)
        await client.connect()
        self._client = client
        await self._write(Describe().event())
        while True:
            event = await client.read_event()
            if event is None:
                logger.info("[voice_satellite:%s] disconnected", self.name)
                return
            if Info.is_type(event.type):
                await self._write(RunSatellite().event())
                self.connected = True
                logger.info(
                    "[voice_satellite:%s] connected to %s:%s",
                    self.name, self.host, self.port,
                )
                continue
            try:
                await self._dispatch(event)
            except (ConnectionError, OSError):
                raise
            except Exception:
                logger.exception(
                    "[voice_satellite:%s] event handler error", self.name
                )

    async def _dispatch(self, event) -> None:
        if Ping.is_type(event.type):
            await self._write(Pong().event())
        elif AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            await self._on_audio_chunk(
                self.name, chunk.audio, chunk.seconds, chunk.rate
            )
        elif RunPipeline.is_type(event.type):
            # Older satellites (wyoming 1.5.x) send snd_format here; honor it.
            snd_format = (event.data or {}).get("snd_format") or {}
            if snd_format.get("rate"):
                self.snd_rate = int(snd_format["rate"])
            await self._on_pipeline_start(self.name)
        elif Played.is_type(event.type):
            await self._on_played(self.name)
        # detection / streaming-started / streaming-stopped etc.: no-op in M1

    async def _write(self, event) -> None:
        async with self._write_lock:
            client = self._client
            if client is None:
                raise ConnectionError(f"satellite {self.name} not connected")
            try:
                await client.write_event(event)
            except (OSError, ConnectionError) as err:
                raise ConnectionError(
                    f"satellite {self.name} write failed: {err}"
                ) from err

    async def send_transcript(self, text: str) -> None:
        """End the satellite's mic streaming (empty text aborts the turn)."""
        await self._write(Transcript(text=text).event())

    async def play_pcm(self, pcm: bytes, rate: int) -> None:
        """Stream s16le mono PCM to the satellite speaker.

        TCP backpressure (write_event drains) paces delivery; no manual
        sleep needed.
        """
        timestamp = 0
        await self._write(
            AudioStart(rate=rate, width=2, channels=1, timestamp=timestamp).event()
        )
        for i in range(0, len(pcm), _TTS_CHUNK_BYTES):
            chunk = pcm[i : i + _TTS_CHUNK_BYTES]
            await self._write(
                AudioChunk(
                    rate=rate, width=2, channels=1, audio=chunk, timestamp=timestamp
                ).event()
            )
            timestamp += int(len(chunk) / (rate * 2) * 1000)
        await self._write(AudioStop(timestamp=timestamp).event())
