"""Request cancellation must not remove audio still owned by a worker."""

import asyncio
import gc
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

import pytest
from aiohttp import ClientError, FormData, web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, body_limit_middleware


async def _cancel_upload(tmp_path, monkeypatch, stage, *, fail=False):
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    submitted = asyncio.Event()
    release = threading.Event()
    observed = []
    loop_errors = []
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))

    class ObservedExecutor(ThreadPoolExecutor):
        observe = False

        def submit(self, fn, /, *args, **kwargs):
            future = super().submit(fn, *args, **kwargs)
            if self.observe:
                loop.call_soon_threadsafe(submitted.set)
            return future

    pool = ObservedExecutor(max_workers=1)
    loop.set_default_executor(pool)
    blocker = None
    if stage == "queued":
        blocker = loop.run_in_executor(None, release.wait, 10)
        pool.observe = True

    def read_after_release(path):
        loop.call_soon_threadsafe(started.set)
        if not release.wait(10):
            raise TimeoutError("Test did not release the audio worker")
        observed.append(Path(path).read_bytes())
        if fail:
            raise RuntimeError("Controlled transcription failure after cancellation")

    def transcribe(path, *, model, language, prompt, source):
        if stage != "probe":
            read_after_release(path)
        return {"success": True, "transcript": "test transcript"}

    def probe(path):
        if stage == "probe":
            read_after_release(path)
        return 0.01

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("API_SERVER_KEY", "audio-lifetime-test-key-1234567890")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr("tools.transcription_tools.transcribe_audio", transcribe)
    monkeypatch.setattr("tools.transcription_audio._probe_audio_duration", probe)
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    handler_task = None

    async def route(request):
        nonlocal handler_task
        handler_task = asyncio.current_task()
        return await adapter._handle_audio_transcriptions(request)

    app = web.Application(middlewares=[body_limit_middleware])
    app.router.add_post("/v1/audio/transcriptions", route)
    audio = b"audio-lifetime-payload"
    form = FormData()
    form.add_field("file", audio, filename="voice.wav", content_type="audio/wav")
    form.add_field("model", "test-model")
    form.add_field("response_format", "verbose_json" if stage == "probe" else "json")

    async with TestClient(TestServer(app)) as client:
        request_task = asyncio.create_task(client.post(
            "/v1/audio/transcriptions", data=form,
            headers={"Authorization": "Bearer audio-lifetime-test-key-1234567890"},
        ))
        try:
            ready = submitted if stage == "queued" else started
            await asyncio.wait_for(ready.wait(), 5)
            uploads = list(tmp_path.glob("hermes-api-stt-*"))
            assert len(uploads) == 1
            upload = uploads[0]
            assert handler_task is not None
            handler_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await handler_task

            assert upload.exists(), "Cancellation removed audio before its worker finished"
            release.set()
            # One worker makes this sentinel a deterministic join, not a timing guess.
            await loop.run_in_executor(None, lambda: None)
            assert observed == [audio]
            assert not upload.exists()
            assert not list(tmp_path.glob("hermes-api-stt-*"))
        finally:
            release.set()
            if handler_task is not None and not handler_task.done():
                handler_task.cancel()
            if handler_task is not None:
                with suppress(asyncio.CancelledError, ClientError):
                    await handler_task
            if not request_task.done():
                request_task.cancel()
            with suppress(asyncio.CancelledError, ClientError):
                await request_task
            if blocker is not None:
                await blocker
            await loop.run_in_executor(None, lambda: None)
    gc.collect()
    await asyncio.sleep(0)
    assert not loop_errors, loop_errors


@pytest.mark.parametrize("stage", ["queued", "transcribe", "probe"])
def test_cancellation_preserves_audio_until_worker_finishes(tmp_path, monkeypatch, stage):
    asyncio.run(_cancel_upload(tmp_path, monkeypatch, stage))


def test_cancelled_worker_failure_cleans_audio_without_unobserved_exception(tmp_path, monkeypatch):
    asyncio.run(_cancel_upload(tmp_path, monkeypatch, "transcribe", fail=True))
