"""Native BlueBubbles voice upload contracts (PR #30529)."""

import asyncio
import subprocess
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import build_auto_tts_output_path


def make_adapter():
    from plugins.platforms.bluebubbles.adapter import BlueBubblesAdapter

    adapter: Any = BlueBubblesAdapter(PlatformConfig(extra={
        "server_url": "https://example.invalid", "password": "test-password",
    }))
    adapter._resolve_chat_guid = AsyncMock(return_value="test-chat")
    adapter.client = SimpleNamespace(post=AsyncMock(return_value=httpx.Response(
        200, json={"status": 200, "data": {"guid": "test-message"}},
        request=httpx.Request("POST", "https://example.invalid"),
    )))
    return adapter


@pytest.mark.asyncio
async def test_default_mp3_prepares_off_loop_uploads_native_caf_and_cleans(monkeypatch, tmp_path):
    source = tmp_path / Path(build_auto_tts_output_path(Platform.BLUEBUBBLES)).name
    assert source.suffix == ".mp3"
    source.write_bytes(b"ID3-test-source")
    main_thread = threading.get_ident()
    calls = []
    created = []
    real_tempfile = tempfile.NamedTemporaryFile

    def temporary_file(*args, **kwargs):
        handle = real_tempfile(*args, dir=tmp_path, **kwargs)
        created.append(Path(handle.name))
        return handle

    def run(command, **kwargs):
        assert threading.get_ident() != main_thread
        calls.append((command, kwargs))
        Path(command[-1]).write_bytes(b"normalized" if command[0] == "ffmpeg" else b"caff-opus")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", temporary_file)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("hermes_platform.resolver.locate_command", lambda name: SimpleNamespace(command=(name,)))
    adapter = make_adapter()
    adapter._private_api_enabled = adapter._helper_connected = True

    result = await adapter.send_voice("test-chat", str(source))

    assert result.success and result.message_id == "test-message"
    upload = adapter.client.post.call_args.kwargs
    assert upload["files"]["attachment"] == ("Audio Message.caf", b"caff-opus", "audio/x-caf")
    assert upload["data"]["name"] == "Audio Message.caf"
    assert upload["data"]["isAudioMessage"] == "true"
    assert upload["data"]["method"] == "private-api"
    assert len(calls) == 2
    normalize, encode = (call[0] for call in calls)
    assert normalize[normalize.index("-ac") + 1] == "1"
    assert normalize[normalize.index("-ar") + 1] == "24000"
    assert encode[encode.index("-d") + 1] == "opus@24000"
    assert encode[encode.index("-c") + 1] == "1"
    assert all(options["timeout"] == 60 for _, options in calls)
    assert len(created) == 2 and all(not path.exists() for path in created)
    assert source.read_bytes() == b"ID3-test-source"


@pytest.mark.parametrize(("suffix", "payload", "expected_suffix", "mime"), [
    (".mp3", b"ID3-audio", ".mp3", "audio/mpeg"),
    (".m2a", b"unknown-audio", ".m2a", "audio/mpeg"),
    (".M2A", b"unknown-audio", ".m2a", "audio/mpeg"),
    (".wav", b"RIFF1234WAVE", ".wav", "audio/wav"),
    (".mp3", b"RIFF1234WAVE", ".wav", "audio/wav"),
    (".caf", b"ID3-audio", ".mp3", "audio/mpeg"),
    (".m4a", b"1234ftypM4A ", ".m4a", "audio/mp4"),
    (".aac", b"\xff\xf1-audio", ".aac", "audio/aac"),
    (".ogg", b"OggS-audio", ".ogg", "audio/ogg"),
    (".opus", b"unknown-audio", ".opus", "audio/opus"),
    (".flac", b"fLaC-audio", ".flac", "audio/flac"),
    (".bin", b"unknown-audio", ".bin", "application/octet-stream"),
])
@pytest.mark.asyncio
async def test_unavailable_conversion_uses_truthful_ordinary_audio(
    monkeypatch, tmp_path, suffix, payload, expected_suffix, mime,
):
    from plugins.platforms.bluebubbles import voice

    monkeypatch.setattr(voice, "convert_audio_to_caf", lambda _: None)
    source = tmp_path / f"voice{suffix}"
    source.write_bytes(payload)
    adapter = make_adapter()
    adapter._private_api_enabled = adapter._helper_connected = True

    result = await adapter.send_voice("test-chat", str(source))

    assert result.success
    upload = adapter.client.post.call_args.kwargs
    assert upload["files"]["attachment"] == (f"voice{expected_suffix}", payload, mime)
    assert "isAudioMessage" not in upload["data"]
    assert "method" not in upload["data"]
    assert source.exists()


@pytest.mark.parametrize(("private_api", "helper"), [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.asyncio
async def test_existing_caf_is_borrowed_with_conditional_private_api(monkeypatch, tmp_path, private_api, helper):
    from plugins.platforms.bluebubbles import voice

    def no_conversion(_):
        pytest.fail("Existing CAF should not require a converter")

    monkeypatch.setattr(voice, "convert_audio_to_caf", no_conversion)
    source = tmp_path / "existing.caf"
    source.write_bytes(b"caff-existing")
    adapter = make_adapter()
    adapter._private_api_enabled, adapter._helper_connected = private_api, helper

    result = await adapter.send_voice("test-chat", str(source))

    assert result.success
    upload = adapter.client.post.call_args.kwargs
    assert upload["files"]["attachment"] == ("Audio Message.caf", b"caff-existing", "audio/x-caf")
    assert upload["data"]["isAudioMessage"] == "true"
    assert (upload["data"].get("method") == "private-api") == (private_api and helper)
    assert source.exists()


@pytest.mark.asyncio
async def test_cancelled_preparation_cleans_after_worker_task_shutdown(monkeypatch, tmp_path):
    """A thread must release its CAF even if its asyncio wrapper is cancelled."""
    from plugins.platforms.bluebubbles import voice

    source = tmp_path / "source.mp3"
    source.write_bytes(b"ID3-source")
    generated = tmp_path / "generated.caf"
    started, release, cleaned = (threading.Event() for _ in range(3))
    original_cleanup = voice.cleanup_attachment

    def prepare(*_):
        started.set()
        assert release.wait(5)
        generated.write_bytes(b"caff-generated")
        return voice.PreparedAttachment(str(generated), "Audio Message.caf", "audio/x-caf", True, True)

    def cleanup(prepared):
        original_cleanup(prepared)
        cleaned.set()

    monkeypatch.setattr(voice, "prepare_voice_attachment", prepare)
    monkeypatch.setattr(voice, "cleanup_attachment", cleanup)
    adapter = make_adapter()
    send = asyncio.create_task(adapter.send_voice("test-chat", str(source)))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        workers = [task for task in asyncio.all_tasks() if getattr(task.get_coro(), "__qualname__", "") == "to_thread"]
        send.cancel()
        with pytest.raises(asyncio.CancelledError):
            await send
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    finally:
        release.set()
    assert await asyncio.to_thread(cleaned.wait, 5)
    assert not generated.exists()
    adapter.client.post.assert_not_awaited()


@pytest.mark.parametrize("error", [RuntimeError("unexpected preparation failure"), RuntimeError()])
@pytest.mark.asyncio
async def test_unexpected_preparation_failure_returns_send_result(monkeypatch, tmp_path, error):
    from plugins.platforms.bluebubbles import voice

    def fail(*_):
        raise error

    source = tmp_path / "voice.mp3"
    source.write_bytes(b"ID3-source")
    monkeypatch.setattr(voice, "prepare_voice_attachment", fail)
    adapter = make_adapter()
    result = await adapter.send_voice("test-chat", str(source))
    assert not result.success
    assert result.error == (str(error) or type(error).__name__)
    adapter.client.post.assert_not_awaited()


@pytest.mark.parametrize("failure", ["no-afconvert", "no-ffmpeg", "empty-wav", "empty-caf", "invalid-caf", "timeout", "tempfile"])
@pytest.mark.asyncio
async def test_conversion_failures_clean_owned_files_and_fall_back(monkeypatch, tmp_path, failure):
    from plugins.platforms.bluebubbles import voice

    source = tmp_path / "voice.mp3"
    source.write_bytes(b"ID3-source")
    created = []
    original_tempfile = tempfile.NamedTemporaryFile

    def temporary_file(*args, **kwargs):
        if failure == "tempfile":
            raise OSError("synthetic temporary directory failure")
        handle = original_tempfile(*args, dir=tmp_path, **kwargs)
        created.append(Path(handle.name))
        return handle

    def locate(name):
        missing = (failure == "no-afconvert" and name == "afconvert") or (failure == "no-ffmpeg" and name == "ffmpeg")
        return SimpleNamespace(command=None if missing else (name,))

    def run(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if (failure == "empty-wav" and command[0] == "ffmpeg") or (failure == "empty-caf" and command[0] == "afconvert"):
            return
        if failure == "invalid-caf" and command[0] == "afconvert":
            Path(command[-1]).write_bytes(b"invalid-output")
            return
        Path(command[-1]).write_bytes(b"normalized" if command[0] == "ffmpeg" else b"caff-generated")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", temporary_file)
    monkeypatch.setattr(voice.resolver, "locate_command", locate)
    monkeypatch.setattr(subprocess, "run", run)
    adapter = make_adapter()
    result = await adapter.send_voice("test-chat", str(source))
    assert result.success
    upload = adapter.client.post.call_args.kwargs
    assert upload["files"]["attachment"] == ("voice.mp3", b"ID3-source", "audio/mpeg")
    assert "isAudioMessage" not in upload["data"] and "method" not in upload["data"]
    assert all(not path.exists() for path in created)
    assert source.exists()


@pytest.mark.parametrize("upload_failure", ["http", "api", "cancel"])
@pytest.mark.asyncio
async def test_upload_failure_or_cancellation_has_no_owned_caf(monkeypatch, tmp_path, upload_failure):
    from plugins.platforms.bluebubbles import voice

    source, generated = tmp_path / "voice.mp3", tmp_path / "generated.caf"
    source.write_bytes(b"ID3-source")
    generated.write_bytes(b"caff-generated")
    monkeypatch.setattr(voice, "prepare_voice_attachment", lambda *_: voice.PreparedAttachment(
        str(generated), "Audio Message.caf", "audio/x-caf", True, True,
    ))
    adapter = make_adapter()
    if upload_failure == "api":
        adapter.client.post.return_value = httpx.Response(
            200, json={"status": 500, "message": "synthetic upload failure"},
            request=httpx.Request("POST", "https://example.invalid"),
        )
    else:
        adapter.client.post.side_effect = asyncio.CancelledError() if upload_failure == "cancel" else OSError("synthetic upload failure")
    if upload_failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await adapter.send_voice("test-chat", str(source))
    else:
        result = await adapter.send_voice("test-chat", str(source))
        assert not result.success and result.error == "synthetic upload failure"
    assert not generated.exists()
    assert source.exists()


@pytest.mark.parametrize("fail_read", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.asyncio
async def test_read_completion_owns_cleanup_and_observes_background_errors(monkeypatch, tmp_path, fail_read, cancel):
    from plugins.platforms.bluebubbles import voice

    source, generated = tmp_path / "voice.mp3", tmp_path / "generated.caf"
    source.write_bytes(b"ID3-source")
    generated.write_bytes(b"caff-generated")
    started, release, cleaned = (threading.Event() for _ in range(3))
    original_read, original_cleanup = Path.read_bytes, voice.cleanup_attachment
    main_thread = threading.get_ident()
    loop = asyncio.get_running_loop()
    background_errors = []
    old_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _, context: background_errors.append(context))

    def read(path):
        if path != generated:
            return original_read(path)
        assert threading.get_ident() != main_thread
        started.set()
        assert release.wait(5)
        if fail_read:
            raise OSError("synthetic worker read failure")
        return original_read(path)

    def cleanup(prepared):
        original_cleanup(prepared)
        cleaned.set()

    monkeypatch.setattr(voice, "prepare_voice_attachment", lambda *_: voice.PreparedAttachment(
        str(generated), "Audio Message.caf", "audio/x-caf", True, True,
    ))
    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(voice, "cleanup_attachment", cleanup)
    adapter = make_adapter()
    send = asyncio.create_task(adapter.send_voice("test-chat", str(source)))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        workers = [task for task in asyncio.all_tasks() if getattr(task.get_coro(), "__qualname__", "") == "to_thread"]
        if cancel:
            send.cancel()
            with pytest.raises(asyncio.CancelledError):
                await send
            assert generated.exists()
        release.set()
        assert await asyncio.to_thread(cleaned.wait, 5)
        if cancel:
            await asyncio.gather(*workers, return_exceptions=True)
            adapter.client.post.assert_not_awaited()
        else:
            result = await send
            assert result.success is not fail_read
            if fail_read:
                assert result.error == "synthetic worker read failure"
        # Drain result callbacks before checking the loop's exception handler.
        drained = loop.create_future()
        loop.call_soon(drained.set_result, None)
        await drained
        assert not background_errors
        assert not generated.exists() and source.exists()
    finally:
        release.set()
        loop.set_exception_handler(old_handler)


@pytest.mark.asyncio
async def test_plain_audio_stays_out_of_native_voice_preparation(monkeypatch, tmp_path):
    from plugins.platforms.bluebubbles import voice

    def no_preparation(*_):
        pytest.fail("Plain audio must not use the native voice path")

    monkeypatch.setattr(voice, "prepare_voice_attachment", no_preparation)
    source = tmp_path / "music.mp3"
    source.write_bytes(b"ID3-music")
    adapter = make_adapter()
    result = await adapter.send_voice("test-chat", str(source), is_voice=False)
    assert result.success
    upload = adapter.client.post.call_args.kwargs
    assert upload["files"]["attachment"] == ("music.mp3", b"ID3-music", "application/octet-stream")
    assert "isAudioMessage" not in upload["data"] and "method" not in upload["data"]
    assert source.exists()


@pytest.mark.platforms("macos")
def test_real_mp3_to_opus_caf_codec_smoke(monkeypatch, tmp_path):
    """Synthetic audio only; no messaging or inference services are contacted."""
    import json
    import struct
    from plugins.platforms.bluebubbles import voice

    ffmpeg = voice.resolver.locate_command("ffmpeg").command
    ffprobe = voice.resolver.locate_command("ffprobe").command
    afconvert = voice.resolver.locate_command("afconvert").command
    if not all((ffmpeg, ffprobe, afconvert)):
        pytest.skip("Existing ffmpeg, ffprobe and afconvert are required")
    source = tmp_path / "synthetic.mp3"
    subprocess.run([*ffmpeg, "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.2",
                    "-ac", "2", "-ar", "44100", str(source)], check=True, timeout=10,
                   stdin=subprocess.DEVNULL, capture_output=True)
    original_tempfile = tempfile.NamedTemporaryFile
    created = []

    def temporary_file(*args, **kwargs):
        handle = original_tempfile(*args, dir=tmp_path, **kwargs)
        created.append(Path(handle.name))
        return handle

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", temporary_file)
    prepared = voice.prepare_voice_attachment(str(source))
    try:
        assert prepared.native_voice and prepared.cleanup
        caf = Path(prepared.path).read_bytes()
        assert caf[:4] == b"caff" and caf[8:12] == b"desc"
        assert struct.unpack(">d4s", caf[20:32]) == (24000.0, b"opus")
        probe = subprocess.run([*ffprobe, "-v", "error", "-show_streams", "-of", "json", prepared.path],
                               check=True, capture_output=True, text=True, timeout=10)
        stream = json.loads(probe.stdout)["streams"][0]
        assert stream["codec_name"] == "opus"
        # ffprobe reports the decoder's 48 kHz output rate, not the CAF descriptor.
        assert stream["channels"] == 1 and stream["time_base"] == "1/24000"
    finally:
        voice.cleanup_attachment(prepared)
    assert len(created) == 2 and all(not path.exists() for path in created)
