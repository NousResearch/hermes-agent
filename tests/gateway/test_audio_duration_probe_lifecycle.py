import asyncio
import sys

import pytest

from gateway import run as gateway_run


@pytest.mark.asyncio
async def test_ffprobe_timeout_reaps_the_spawned_process(monkeypatch, tmp_path):
    """A timed-out best-effort duration probe must not leave ffprobe running."""
    spawned = []
    real_create_subprocess_exec = asyncio.create_subprocess_exec
    real_wait_for = asyncio.wait_for

    async def spawn_hanging_probe(*_args, **kwargs):
        proc = await real_create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            stdout=kwargs.get("stdout"),
            stderr=kwargs.get("stderr"),
        )
        spawned.append(proc)
        return proc

    async def expire_quickly(awaitable, _timeout):
        return await real_wait_for(awaitable, timeout=0.05)

    monkeypatch.setattr(gateway_run.asyncio, "create_subprocess_exec", spawn_hanging_probe)
    monkeypatch.setattr(gateway_run.asyncio, "wait_for", expire_quickly)

    try:
        assert await gateway_run._probe_audio_duration(str(tmp_path / "voice.mp3")) is None
        assert len(spawned) == 1
        assert spawned[0].returncode is not None, "timed-out ffprobe child was left running"
    finally:
        for proc in spawned:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
