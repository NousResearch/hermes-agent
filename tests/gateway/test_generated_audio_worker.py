"""Queued or refused executor work must not strand a turn's audio directory."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import pytest

from gateway.generated_audio import GeneratedAudio


def test_cancelled_queued_synthesis_releases_directory(tmp_path):
    release = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    blocker = executor.submit(release.wait, 10)
    loop = asyncio.new_event_loop()
    loop.set_default_executor(executor)
    called = []
    audio = GeneratedAudio(str(tmp_path / "reply.mp3"))

    async def cancel():
        with audio:
            task = asyncio.create_task(audio.synthesize(lambda **kw: called.append(kw)))
            # A loop turn lets run_in_executor enqueue behind the blocked worker.
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    try:
        loop.run_until_complete(cancel())
        assert not Path(audio.output_path).parent.exists()
        assert called == []
    finally:
        loop.close()
        release.set()
        blocker.result(timeout=5)
        executor.shutdown(wait=True)
