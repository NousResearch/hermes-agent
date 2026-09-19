"""Real process-death proof for the Hindsight retain journal (#88944).

Before the durable outbox, a retained turn lived only in an in-memory
queue.Queue / list (self._retain_queue / self._session_turns). A SIGKILL
between "the writer claimed the job" and "the server acknowledged it" lost
the turn permanently: nothing on disk knew it had ever existed, so a fresh
provider process started with no memory of the pending work.

This test proves the fix survives an actual OS-level crash, not just an
in-process mock: a child process is SIGKILLed while genuinely inside its
retain call, then a brand-new provider (a different Python object, loaded
fresh from disk) is shown to recover and complete that exact turn.

Run the same file against unmodified origin/main (outbox.py/reliability.py
absent, sync_turn() not journaled) to see it fail: recovery has nothing to
recover, so the fresh provider's client is never called at all.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from plugins.memory.hindsight import HindsightMemoryProvider

REPO_ROOT = Path(__file__).resolve().parents[3]

_CHILD_SCRIPT = '''
import asyncio
import os
import sys

sys.path.insert(0, {repo_root!r})

from plugins.memory.hindsight import HindsightMemoryProvider

MARKER = os.path.join(os.environ["HERMES_HOME"], "claimed.marker")


class _BlockingClient:
    """Stands in for the real Hindsight client: the send genuinely hangs, so a
    SIGKILL lands while the provider is truly mid-retain, not before or after."""

    async def aretain_batch(self, **kwargs):
        with open(MARKER, "w", encoding="utf-8") as f:
            f.write("claimed")
        await asyncio.sleep(3600)


def main():
    provider = HindsightMemoryProvider()
    provider._client = _BlockingClient()
    provider.initialize(session_id="crash-session", hermes_home=os.environ["HERMES_HOME"], platform="cli")
    provider.sync_turn("hello from before the crash", "ack, saving that now")
    provider._retain_queue.join()  # never returns; we get SIGKILLed first


if __name__ == "__main__":
    main()
'''


@pytest.mark.linux_only
def test_sigkill_mid_retain_is_recovered_by_a_fresh_provider(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    (home / "hindsight").mkdir(parents=True)
    config = {
        "mode": "cloud",
        "apiKey": "test-key",
        "api_url": "http://localhost:9999",
        "bank_id": "test-bank",
        "memory_mode": "hybrid",
        "retain_async": False,
    }
    (home / "hindsight" / "config.json").write_text(json.dumps(config))

    script_path = tmp_path / "child_retain.py"
    script_path.write_text(_CHILD_SCRIPT.format(repo_root=str(REPO_ROOT)))

    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    proc = subprocess.Popen([sys.executable, str(script_path)], cwd=str(REPO_ROOT), env=env)
    try:
        marker = home / "claimed.marker"
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.02)
        assert marker.exists(), "child process never reached its retain call before the deadline"

        # The child is genuinely inside the network call right now. Kill -9:
        # no atexit, no finally, no chance to persist anything further.
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    assert proc.returncode == -signal.SIGKILL

    # A brand-new provider object, loaded fresh from the same HERMES_HOME —
    # nothing shared with the dead process but the files on disk.
    monkeypatch.setenv("HERMES_HOME", str(home))
    fresh_client = AsyncMock()
    provider = HindsightMemoryProvider()
    provider._client = fresh_client
    provider.initialize(session_id="recovered-session", hermes_home=str(home), platform="cli")
    provider._retain_queue.join()

    assert fresh_client.aretain_batch.await_count >= 1, (
        "the fresh provider never resent the turn the killed process was mid-retain on"
    )
    sent_content = fresh_client.aretain_batch.await_args.kwargs["items"][0]["content"]
    assert "hello from before the crash" in sent_content
    assert "ack, saving that now" in sent_content
