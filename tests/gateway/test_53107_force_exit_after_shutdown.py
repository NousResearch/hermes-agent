"""Regression tests for #53107 — the gateway must force-exit after its graceful
shutdown has completed, so a wedged non-daemon worker thread can't block
interpreter finalization and hang the process.

The bug: ``main()`` ran ``success = asyncio.run(start_gateway(config))`` (which
performs the full graceful teardown — adapters disconnected, sessions saved,
cron stopped) and then, on failure, called bare ``sys.exit(1)``. ``SystemExit``
triggers CPython finalization (``Py_FinalizeEx`` → ``wait_for_thread_shutdown``),
which joins every non-daemon thread. A ``ThreadPoolExecutor`` tool/LLM worker
blocked on a no-timeout call never joins, so the process lingers with the
api_server + cron already down and the supervisor won't restart it.

The fix replaces the ``sys.exit(1)`` tail of ``main()`` with a log flush + an
``os._exit(0 if success else 1)`` backstop: teardown is already done by that
point, so there is nothing left that needs a clean interpreter shutdown. Exit
codes are preserved (0 success, 1 failure) so ``systemd Restart=on-failure``
still retries.

These are pure unit tests — no real process is spawned. ``os._exit`` is patched
to raise a sentinel so the call is observable instead of terminating the test
runner.
"""

from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture(autouse=True)
def _mock_dotenv(monkeypatch):
    """gateway.run imports dotenv at module load; stub so tests run bare."""
    fake = types.ModuleType("dotenv")
    fake.load_dotenv = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "dotenv", fake)


class _ExitCalled(Exception):
    """Sentinel raised in place of os._exit so we can observe the call without
    actually terminating the interpreter."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _drive_main(monkeypatch, success):
    """Run gateway.run.main() with start_gateway stubbed to return ``success``,
    os._exit patched to a sentinel, and the log flushes recorded. Returns the
    exit code os._exit was called with, plus the flush call counts."""
    import gateway.run as run

    monkeypatch.setattr(sys, "argv", ["hermes-gateway"])
    # Stub the whole start_gateway → asyncio.run hop: start_gateway is replaced
    # so no coroutine is created, and asyncio.run just returns ``success``.
    monkeypatch.setattr(run, "start_gateway", lambda config: None)
    monkeypatch.setattr(run.asyncio, "run", lambda coro: success)

    flushes = {"stdout": 0, "stderr": 0}
    monkeypatch.setattr(sys.stdout, "flush",
                        lambda: flushes.__setitem__("stdout", flushes["stdout"] + 1))
    monkeypatch.setattr(sys.stderr, "flush",
                        lambda: flushes.__setitem__("stderr", flushes["stderr"] + 1))

    def _fake_exit(code):
        raise _ExitCalled(code)

    monkeypatch.setattr(run.os, "_exit", _fake_exit)

    with pytest.raises(_ExitCalled) as excinfo:
        run.main()
    return excinfo.value.code, flushes


class TestForceExitAfterShutdown:
    def test_failure_force_exits_with_code_1(self, monkeypatch):
        """start_gateway() returning False → os._exit(1), after flushing both
        streams. This is the hang case: a stuck worker must not get a chance to
        block finalization."""
        code, flushes = _drive_main(monkeypatch, success=False)

        assert code == 1
        assert flushes["stdout"] == 1
        assert flushes["stderr"] == 1

    def test_success_force_exits_with_code_0(self, monkeypatch):
        """start_gateway() returning True → os._exit(0), so a clean run still
        terminates promptly (and systemd Restart=on-failure won't retry)."""
        code, flushes = _drive_main(monkeypatch, success=True)

        assert code == 0
        assert flushes["stdout"] == 1
        assert flushes["stderr"] == 1

    def test_main_does_not_raise_systemexit(self, monkeypatch):
        """The terminating call must be os._exit, NOT sys.exit — SystemExit is
        exactly what triggers the Py_FinalizeEx thread-join hang. If main()
        raised SystemExit it would propagate instead of our os._exit sentinel."""
        import gateway.run as run

        monkeypatch.setattr(sys, "argv", ["hermes-gateway"])
        monkeypatch.setattr(run, "start_gateway", lambda config: None)
        monkeypatch.setattr(run.asyncio, "run", lambda coro: False)
        monkeypatch.setattr(sys.stdout, "flush", lambda: None)
        monkeypatch.setattr(sys.stderr, "flush", lambda: None)
        monkeypatch.setattr(run.os, "_exit",
                            lambda code: (_ for _ in ()).throw(_ExitCalled(code)))

        # SystemExit must not be the thing that terminates main().
        with pytest.raises(_ExitCalled):
            run.main()
