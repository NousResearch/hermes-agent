"""Detached auto-TTS owns the encoder through cancellation, not just synthesis."""

import asyncio
import contextlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import psutil
import pytest
import pytest_asyncio

from gateway.config import Platform, PlatformConfig
from gateway.platforms import base_auto_tts
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.discord import ffmpeg_utils
from tests.waiting import async_wait_until


@pytest_asyncio.fixture
async def blocked_encoder(tmp_path, monkeypatch):
    """Substitute only the executable; exercise real spawn, waiting and cleanup."""
    ready = tmp_path / "encoder.json"
    run, spawn, which = subprocess.run, asyncio.create_subprocess_exec, shutil.which
    script = (
        "import json, os, pathlib, sys, threading; "
        "output = pathlib.Path(sys.argv[1]); "
        "os.write(1, b'partial-ogg') if sys.argv[2] == 'pipe:1' else output.write_bytes(b'partial-ogg'); "
        f"pathlib.Path({str(ready)!r}).write_text(json.dumps({{'pid': os.getpid(), 'output': str(output)}})); "
        "threading.Event().wait(30); output.write_bytes(b'late-ogg')"
    )

    def run_encoder(cmd, *args, **kwargs):
        if cmd[0] != "test-ffmpeg":
            return run(cmd, *args, **kwargs)
        return run([sys.executable, "-c", script, cmd[-1], cmd[-1]], *args, **kwargs)

    async def spawn_encoder(*args, **kwargs):
        if args[0] == "test-ffmpeg":
            output = str(tmp_path / "unused-output") if args[-1] == "pipe:1" else args[-1]
            args = (sys.executable, "-c", script, output, args[-1])
        return await spawn(*args, **kwargs)

    monkeypatch.setattr(shutil, "which", lambda name, *a, **kw: "test-ffmpeg" if name == "ffmpeg" else which(name, *a, **kw))
    monkeypatch.setattr(ffmpeg_utils, "resolve_ffmpeg_executable", lambda: "test-ffmpeg")
    monkeypatch.setattr(subprocess, "run", run_encoder)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn_encoder)
    yield ready
    # Reap the intentionally exposed pre-fix leak even when the assertion fails.
    if ready.exists():
        info = json.loads(ready.read_text())
        with contextlib.suppress(psutil.NoSuchProcess):
            child = psutil.Process(info["pid"])
            child.kill()
            child.wait(timeout=5)
        with contextlib.suppress(FileNotFoundError):
            Path(info["output"]).unlink()


def attachment_adapter():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake"))
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="text-1"))
    adapter._resolve_channel = AsyncMock(return_value=SimpleNamespace(id=123, send=AsyncMock()))
    adapter._reject_oversized_upload = AsyncMock(return_value=None)
    adapter._reply_reference_for_send = lambda *args: None
    adapter._is_forum_parent = lambda channel: False
    adapter._post_voice_message = AsyncMock(return_value=SendResult(success=True))

    async def synthesize(text, *, output_dir):
        audio = Path(output_dir) / "reply.mp3"
        audio.write_bytes(b"ID3-not-ogg")
        return [str(audio)], str(audio)

    adapter._synthesize_auto_tts = synthesize
    event = MessageEvent(text="hello", message_type=MessageType.VOICE,
                         source=SessionSource(platform=Platform.DISCORD, chat_id="123"))
    return adapter, event


@pytest.mark.platforms("any")
@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["cancel", "shutdown", "deadline", "slash"])
async def test_blocked_transcode_is_reaped_before_job_finishes(blocked_encoder, monkeypatch, stop):
    from gateway.run import GatewayRunner

    monkeypatch.setattr(base_auto_tts, "AUDIO_DEADLINE_SECONDS", 2)
    adapter, event = attachment_adapter()
    key = build_session_key(event.source)
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._background_tasks = set()
    runner.adapters = {Platform.DISCORD: adapter}
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(session_key=key, session_id="test"))
    adapter.gateway_runner = runner
    await base_auto_tts.schedule_auto_tts(adapter, event, key, "reply", "text-1", {})
    task, = adapter._background_tasks
    try:
        await async_wait_until(blocked_encoder.exists, timeout=5)
        info = json.loads(blocked_encoder.read_text())
        if stop == "cancel":
            task.cancel()
        elif stop == "shutdown":
            await adapter.cancel_background_tasks()
        elif stop == "slash":
            from agent.i18n import t
            assert not adapter._active_sessions
            assert await runner._handle_stop_command(event) == t("gateway.stop.stopped")
        done, _ = await asyncio.wait({task}, timeout=10)
        assert task in done, "deadline/cancellation failed to finish the job"
        assert not psutil.pid_exists(info["pid"]), "encoder outlived its audio job"
        assert not Path(info["output"]).exists(), "partial transcoded file leaked"
        assert not adapter._background_tasks
        assert not runner._background_tasks
        adapter._post_voice_message.assert_not_awaited()
        adapter._resolve_channel.return_value.send.assert_not_awaited()
        assert adapter.send.await_count == (1 if stop == "deadline" else 0)
        if stop == "deadline":
            assert adapter.send.await_args.args[1] == base_auto_tts.DEADLINE_NOTICE
            assert adapter.send.await_args.kwargs["reply_to"] == "text-1"
        else:
            assert task.cancelled()
    finally:
        await adapter.cancel_background_tasks()


@pytest.mark.platforms("any")
@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["spawn", "reap"])
async def test_repeated_cancellation_drains_encoder_ownership(blocked_encoder, monkeypatch, phase):
    entered, release = asyncio.Event(), asyncio.Event()
    spawn, finish = asyncio.create_subprocess_exec, ffmpeg_utils._finish_encoder

    async def delayed_spawn(*args, **kwargs):
        proc = await spawn(*args, **kwargs)
        entered.set()
        await release.wait()
        return proc

    async def delayed_finish(spawn_task):
        entered.set()
        await release.wait()
        await finish(spawn_task)

    if phase == "spawn":
        monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    else:
        monkeypatch.setattr(ffmpeg_utils, "_finish_encoder", delayed_finish)
    adapter, event = attachment_adapter()
    task = asyncio.create_task(base_auto_tts._deliver(adapter, event, "reply", "text-1", {}))
    try:
        await async_wait_until(blocked_encoder.exists, timeout=5)
        info = json.loads(blocked_encoder.read_text())
        task.cancel()
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "cancellation abandoned the owned encoder"
        release.set()
        done, _ = await asyncio.wait({task}, timeout=5)
        assert task in done and task.cancelled()
        assert not psutil.pid_exists(info["pid"])
        assert not Path(info["output"]).exists()
        adapter._post_voice_message.assert_not_awaited()
        adapter.send.assert_not_awaited()
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_real_ffmpeg_pipe_has_opus_duration_without_temp_output(tmp_path):
    import wave

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not installed")
    audio = tmp_path / "reply.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 4800)
    before = set(tmp_path.iterdir())
    data, duration = await DiscordAdapter._ogg_opus_voice_payload(str(audio))
    assert data.startswith(b"OggS")
    assert b"OpusHead" in data
    assert duration == pytest.approx(0.1, abs=0.01)
    assert set(tmp_path.iterdir()) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "nonzero", "invalid"])
async def test_encoder_failure_preserves_original_attachment(tmp_path, monkeypatch, failure):
    spawn = asyncio.create_subprocess_exec

    async def fail_encoder(*args, **kwargs):
        if failure == "missing":
            raise FileNotFoundError("no ffmpeg")
        script = "raise SystemExit(1)" if failure == "nonzero" else "print('not opus')"
        return await spawn(sys.executable, "-c", script, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_encoder)
    adapter, _ = attachment_adapter()
    channel = adapter._resolve_channel.return_value
    channel.send.return_value = SimpleNamespace(id=123)
    audio = tmp_path / "reply.mp3"
    audio.write_bytes(b"ID3-original")
    result = await adapter.send_voice("123", str(audio))
    assert result.success
    adapter._post_voice_message.assert_not_awaited()
    file = channel.send.await_args.kwargs["file"]
    assert file.filename == "reply.mp3"
    assert file.fp.read() == b"ID3-original"
    assert audio.read_bytes() == b"ID3-original"


@pytest.mark.asyncio
async def test_cancelled_failed_spawn_does_not_fall_back_to_upload(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()

    async def fail_spawn(*args, **kwargs):
        entered.set()
        await release.wait()
        raise FileNotFoundError("ffmpeg disappeared during spawn")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_spawn)
    adapter, event = attachment_adapter()
    task = asyncio.create_task(base_auto_tts._deliver(adapter, event, "reply", "text-1", {}))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        done, _ = await asyncio.wait({task}, timeout=5)
        assert task in done and task.cancelled()
        adapter._resolve_channel.return_value.send.assert_not_awaited()
        adapter.send.assert_not_awaited()
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
