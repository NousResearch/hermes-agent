"""Regression tests for zombie-process leaks and race conditions in the
codex app-server transport (the OpenAI/Codex route: agent/codex_runtime.py
-> agent/transports/codex_app_server_session.py ->
agent/transports/codex_app_server.py::CodexAppServerClient).

These tests were written after auditing ``CodexAppServerClient.close()``
and the gateway session-expiry watcher (``gateway/run.py:
_session_expiry_watcher``) for orphaned-subprocess / GC-unreachable-object
bugs. Each test class documents the exact defect it reproduces.

No real ``codex`` CLI is required: a tiny stand-in JSON-RPC script
(``_DUMMY_SERVER_SRC``) is spawned in its place via a ``subprocess.Popen``
monkeypatch, so these run as ordinary (non-``integration``) tests and are
part of the default ``pytest`` invocation.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time
from unittest.mock import Mock

import pytest

from agent.transports.codex_app_server import CodexAppServerClient

# ---------------------------------------------------------------------------
# Dummy "codex app-server" stand-in: reads newline-delimited JSON-RPC from
# stdin, answers `initialize` immediately, and otherwise only answers
# requests whose method is NOT "test/hang" (used to simulate a request the
# real server would never answer, e.g. because it died mid-turn).
# ---------------------------------------------------------------------------
_DUMMY_SERVER_SRC = textwrap.dedent(
    """
    import json
    import sys

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        mid = msg.get("id")
        method = msg.get("method")
        if mid is None:
            continue  # notification, nothing to reply to
        if method == "test/hang":
            continue  # deliberately never respond
        result = {"userAgent": "dummy", "codexHome": "", "platformOs": "test", "platformFamily": "test"}
        sys.stdout.write(json.dumps({"id": mid, "result": result}) + "\\n")
        sys.stdout.flush()
    """
)


@pytest.fixture()
def dummy_server_script(tmp_path):
    script = tmp_path / "dummy_codex_app_server.py"
    script.write_text(_DUMMY_SERVER_SRC, encoding="utf-8")
    return script


@pytest.fixture()
def spawn_client(monkeypatch, dummy_server_script):
    """Yields a factory that builds a real ``CodexAppServerClient`` whose
    subprocess is the dummy script above instead of the real ``codex``
    binary. Uses a real OS process/pipes/threads end to end."""

    real_popen = subprocess.Popen

    def _patched_popen(cmd, **kwargs):
        return real_popen([sys.executable, str(dummy_server_script)], **kwargs)

    monkeypatch.setattr(
        "agent.transports.codex_app_server.subprocess.Popen", _patched_popen
    )

    clients: list[CodexAppServerClient] = []

    def _factory() -> CodexAppServerClient:
        client = CodexAppServerClient(codex_bin="codex")
        clients.append(client)
        return client

    yield _factory

    # Test-harness safety net only — not part of any assertion. Guarantees
    # a broken test doesn't itself leak a real orphaned process on the
    # machine running the suite.
    for client in clients:
        try:
            client._proc.kill()
            client._proc.wait(timeout=5)
        except Exception:
            pass


class TestCloseLogsKillFailureInsteadOfSwallowingIt:
    """Fix: in ``CodexAppServerClient.close()`` (agent/transports/
    codex_app_server.py), if ``terminate()`` doesn't reap the child within
    ``timeout``, the code escalates to ``kill()`` + a second
    ``wait(timeout=1.0)``. If the child survives even
    ``SIGKILL``/``TerminateProcess`` past that final 1s wait, ``close()``
    now logs a warning (with the pid) instead of swallowing the failure
    silently. ``self._closed`` is still set to ``True`` before the
    terminate/kill dance starts, so a later ``close()`` call remains a
    no-op — the process can still be orphaned — but it is no longer
    invisible: this is the one signal that would exist anywhere in the
    stack for it.
    """

    def test_kill_wait_failure_is_logged_and_does_not_raise(self, caplog):
        client = object.__new__(CodexAppServerClient)  # skip __init__/Popen
        client._closed = False

        proc = Mock()
        proc.stdin = None
        proc.terminate = Mock()
        proc.kill = Mock()
        proc.pid = 4242
        proc.wait = Mock(
            side_effect=[
                subprocess.TimeoutExpired(cmd="dummy", timeout=3),
                subprocess.TimeoutExpired(cmd="dummy", timeout=1),
            ]
        )
        client._proc = proc

        with caplog.at_level("WARNING"):
            client.close()  # must not raise

        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        assert proc.wait.call_count == 2
        assert client._closed is True
        assert len(caplog.records) == 1
        assert "4242" in caplog.records[0].message
        assert "orphaned" in caplog.records[0].message.lower()

    def test_leak_is_permanent_because_closed_flag_short_circuits_retry(self):
        """Direct consequence of the above: once close() has swallowed a
        failed reap, a second call to close() returns immediately without
        attempting to kill/reap the child again — confirmed by
        ``proc.wait`` never being invoked a third time."""
        client = object.__new__(CodexAppServerClient)
        client._closed = False

        proc = Mock()
        proc.stdin = None
        proc.terminate = Mock()
        proc.kill = Mock()
        proc.wait = Mock(
            side_effect=[
                subprocess.TimeoutExpired(cmd="dummy", timeout=3),
                subprocess.TimeoutExpired(cmd="dummy", timeout=1),
            ]
        )
        client._proc = proc

        client.close()
        client.close()  # mirrors every real call site re-calling close()

        assert proc.wait.call_count == 2, (
            "close() attempted to reap the child again after already being "
            "marked closed — if this fails, the leak may have been fixed "
            "and this guard should be revisited"
        )


class TestClosePendingRequestsRace:
    """Bug: ``close()`` never touches ``self._pending``. A thread blocked
    in ``request()`` waiting on a reply has no idea the transport just
    died — it keeps waiting on its own ``timeout`` argument instead of
    failing fast.

    This is the concrete mechanism behind the observed "codex route hangs
    after session expiry" race: gateway/run.py's ``_session_expiry_watcher``
    explicitly falls back to ``_running_agents`` "in case the agent is
    still mid-turn" and calls ``AIAgent.close()`` -> ``codex_session.close()``
    on a session that may have an in-flight ``turn/start`` request. That
    caller thread does not get unblocked by the close() happening
    concurrently on another thread/task.
    """

    def test_blocked_request_does_not_notice_concurrent_close(self, spawn_client):
        client = spawn_client()
        client.initialize(timeout=5)

        outcome: dict[str, object] = {}

        def _blocked_request():
            start = time.monotonic()
            try:
                outcome["result"] = client.request("test/hang", timeout=2.0)
            except Exception as exc:  # TimeoutError from request()'s own budget
                outcome["error"] = exc
            outcome["elapsed"] = time.monotonic() - start

        t = threading.Thread(target=_blocked_request, daemon=True)
        t.start()
        time.sleep(0.2)  # let the request register in _pending

        close_started = time.monotonic()
        client.close()
        close_elapsed = time.monotonic() - close_started

        t.join(timeout=5)

        assert close_elapsed < 1.0, "close() itself should return quickly"
        assert not t.is_alive(), "blocked request() thread never returned"
        # The bug: the blocked call rides out its own ~2s timeout instead
        # of failing immediately when the transport it depends on closed.
        assert outcome.get("elapsed", 0) >= 1.5, (
            "request() returned suspiciously fast — if close() now cancels "
            "pending requests immediately, this test should be rewritten "
            "to assert the (fixed) fast-fail behavior instead"
        )
        assert "error" in outcome


class TestCloseReaping:
    """Regression guard (currently expected to pass): the happy-path close()
    does reap the real child process and does not leave stdout/stderr
    reader threads running forever, given a well-behaved child. This pins
    current good behavior so future refactors of close() don't regress the
    one thing it does get right."""

    def test_close_reaps_real_process_and_threads_exit(self, spawn_client):
        client = spawn_client()
        client.initialize(timeout=5)

        reader = client._reader
        stderr_reader = client._stderr_reader
        assert reader.is_alive()

        client.close()

        assert client._proc.poll() is not None, "child process not reaped"

        reader.join(timeout=3)
        stderr_reader.join(timeout=3)
        assert not reader.is_alive(), "stdout reader thread leaked past close()"
        assert not stderr_reader.is_alive(), "stderr reader thread leaked past close()"


class TestConcurrentCloseRace:
    """Race: nothing guards ``self._closed`` with a lock — it's a plain
    bool checked-then-set with no atomicity. Two threads racing
    ``close()`` (e.g. a turn-crash handler in agent/codex_runtime.py and
    the gateway expiry watcher both deciding to tear the same session
    down) can both observe ``_closed is False`` and both proceed into the
    terminate/kill/wait sequence concurrently.

    This test proves that at minimum no unhandled exception escapes either
    thread in the double-close case against a real process (double
    terminate/kill on an already-exited PID is normally safe on both POSIX
    and Windows) — but is written so that if a future change makes the
    race throw (e.g. more aggressive Windows handle reuse), the failure is
    caught here instead of in production telemetry.
    """

    def test_double_close_from_two_threads_does_not_raise(self, spawn_client):
        client = spawn_client()
        client.initialize(timeout=5)

        errors: list[BaseException] = []
        barrier = threading.Barrier(2)

        def _close():
            barrier.wait(timeout=5)
            try:
                client.close()
            except BaseException as exc:  # noqa: BLE001 - capturing for assertion
                errors.append(exc)

        threads = [threading.Thread(target=_close, daemon=True) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"concurrent close() raised: {errors}"
        assert client._proc.poll() is not None
