"""Order Codex writer retirement before a gateway cache replacement starts."""

from concurrent.futures import Future
import threading


class CodexReleaseBarrier:
    """Register under the cache lock; close and wait only after leaving that lock.

    Only the owned Codex client gates replacement. Memory flushes and background
    work retain the gateway's existing asynchronous soft-release semantics.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._pending = {}
        self._agents = {}

    def track(self, key, agent):
        if getattr(agent, "_codex_session", None) is None:
            return
        with self._lock:
            if id(agent) not in self._agents:
                future = Future()
                self._agents[id(agent)] = (key, future)
                self._pending.setdefault(key, {})[id(agent)] = future

    def release(self, agent):
        with self._lock:
            entry = self._agents.get(id(agent))
            if entry is None:
                return
            key, future = entry
            owns_release = not future.running() and not future.done()
            if owns_release:
                future.set_running_or_notify_cancel()
        if not owns_release:
            future.result(timeout=30)
            return
        try:
            agent._close_codex_session()
        except BaseException as exc:
            # Keep a failed fence: resuming without proof of release loses continuity.
            future.set_exception(exc)
            raise
        else:
            with self._lock:
                future.set_result(None)
                del self._agents[id(agent)]
                del self._pending[key][id(agent)]
                if not self._pending[key]:
                    del self._pending[key]

    def wait(self, key):
        with self._lock:
            pending = list(self._pending.get(key, {}).values())
        for future in pending:
            # Fail the turn rather than seed a new thread while an old writer lives.
            future.result(timeout=30)


def codex_release_barrier(runner):
    # Also supports lightweight runners created with __new__ in integration tests.
    return runner.__dict__.setdefault("_codex_release_barrier", CodexReleaseBarrier())
