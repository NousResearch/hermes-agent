"""``gateway.run._probe_audio_duration``: a timed-out ffprobe must not keep running under the gateway."""

import asyncio
import sys

import pytest


@pytest.mark.asyncio
async def test_timed_out_ffprobe_is_killed(monkeypatch):
    from gateway.run import _probe_audio_duration

    real_exec, real_wait_for = asyncio.create_subprocess_exec, asyncio.wait_for
    spawned = []

    async def _hung_ffprobe(*_argv, **kwargs):
        proc = await real_exec(sys.executable, "-c", "import time; time.sleep(120)", **kwargs)
        spawned.append(proc)
        return proc

    async def _expire(aw, timeout):
        # Behave like wait_for() hitting the 5 s cap, without waiting 5 s.
        if getattr(aw, "__qualname__", "") != "Process.communicate":
            return await real_wait_for(aw, timeout)
        aw.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _hung_ffprobe)
    monkeypatch.setattr(asyncio, "wait_for", _expire)

    assert await _probe_audio_duration("voice.mp3") is None
    [proc] = spawned
    try:
        await real_wait_for(proc.wait(), timeout=5)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        pytest.fail("timed-out ffprobe was left running")
