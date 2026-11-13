"""Which background processes run inside a terminal environment, for its teardown.

A live process belongs to the environment object it was spawned through (``env_ref``).
A handoff changes its ``task_id`` to the new owner's cache key without moving the process.
Only processes without an ``env_ref`` (local and host-recovered ones) fall back to their
``task_id``. ``owner_task_id`` is the raw spawning id, not an environment identity.
"""

from typing import Any


def _in_env(session, key: str, env: Any) -> bool:
    if session.env_ref is not None:
        return session.env_ref is env
    return session.task_id == key


class ProcessEnvScopeMixin:
    """Environment-scoped queries for ``ProcessRegistry`` (which supplies ``_lock``, ``_running``,
    ``_any_running`` and ``_finish_exited``)."""

    def has_running_in_env(self, key: str, env: Any) -> bool:
        """Whether a still-running process lives in the environment cached under ``key`` as
        ``env``. Tearing that environment down would kill or orphan it."""
        return self._any_running(lambda s: _in_env(s, key, env))

    def settle_env_closed(self, env: Any) -> int:
        """Settle every running process polled through ``env`` as lost, once ``env`` is closed.

        A sandbox process is observed only through its spawning environment object; once that
        object is closed the poller can no longer read its output or exit code, and some backends
        report the failure as output rather than raising, so the poller would retry forever.
        Marking the session exited stops the poller at its next check and publishes the result.
        A process that exits between the snapshot and its settlement keeps its own result. A
        backend whose processes outlive cleanup() (``cleanup_stops_processes = False``) is left
        alone: its poller still reaches them and collects the real exit.
        Returns the number of sessions settled as lost."""
        # ``None`` would match every local session (no env_ref). Duck-typed flag: plugin backends
        # and test doubles need not subclass BaseEnvironment.
        if env is None or not getattr(env, "cleanup_stops_processes", True):
            return 0
        with self._lock:
            sessions = [s for s in self._running.values() if s.env_ref is env and not s.exited]
        return sum(self._finish_exited(s, -1, reason="lost", source="env_closed") for s in sessions)
