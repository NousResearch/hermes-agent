"""Kanban worker session-end emission on hard-exit paths (HM2).

A kanban worker's memory-provider ``on_session_end`` lifecycle event fires
from ``_run_cleanup`` → ``shutdown_memory_provider``. On a normal exit that
runs via the single-query ``finally:`` → ``_finalize_single_query`` — base
behaviour, not something these tests re-assert. The genuinely-missed exits
are the hard ``os._exit(0)`` paths that bypass ``finally:`` / ``atexit``
entirely:

  - ``_signal_handler_q`` (kanban SIGTERM/SIGHUP handler) — ``os._exit(0)``
  - the exit watchdog thread — ``os._exit(0)``

Both now call the shared ``_emit_session_end_before_hard_exit()`` helper
right before ``os._exit(0)``. These tests lock in that each hard-exit path
emits ``on_session_end`` exactly once, that a raising provider never prevents
the process exit, and that no double-emission is possible when the normal
``finally:`` also runs.

Interactive behaviour is unchanged: the interactive path already delivered
the memory session-end via the same ``shutdown_memory_provider`` helper and
is untouched here.
"""

import os
import signal
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import cli as cli_mod
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


class _RecordingProvider(MemoryProvider):
    """A memory provider that records its lifecycle calls."""

    def __init__(self, log, *, raise_on_end=False):
        self._log = log
        self._raise = raise_on_end

    @property
    def name(self):
        return "recording"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self._log.append("initialize")

    def on_session_end(self, messages):
        self._log.append(f"on_session_end:{len(messages)}")
        if self._raise:
            raise RuntimeError("provider on_session_end boom")

    def shutdown_all(self):
        self._log.append("shutdown_all")

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs):
        return "{}"


def _recording_agent(log, *, raise_on_end=False):
    """An agent whose shutdown_memory_provider records emissions to ``log``.

    Uses a real MemoryManager wired to a _RecordingProvider so a successful
    emission actually exercises the provider's on_session_end. Mirrors the
    surface cli._active_agent_ref / _run_cleanup expect.
    """
    mm = MemoryManager()
    mm._providers = [_RecordingProvider(log, raise_on_end=raise_on_end)]

    def shutdown(msgs=None):
        log.append("shutdown_memory_provider")
        mm.on_session_end(msgs if msgs is not None else [])
        mm.shutdown_all()

    agent = SimpleNamespace(
        session_id="worker-session",
        platform="cli",
        model="probe",
        _session_messages=[{"role": "user", "content": "hi"}],
        _memory_manager=mm,
        shutdown_memory_provider=shutdown,
    )
    return agent


def _count_ends(log):
    return sum(1 for c in log if c.startswith("on_session_end"))


# ---------------------------------------------------------------------------
# Shared helper — direct unit tests (the logic both hard-exit paths call)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    monkeypatch.setattr(cli_mod, "_cleanup_done", False)
    monkeypatch.setattr(cli_mod, "_active_agent_ref", None)


def test_helper_emits_exactly_once():
    log = []
    cli_mod._active_agent_ref = _recording_agent(log)
    cli_mod._emit_session_end_before_hard_exit()
    cli_mod._emit_session_end_before_hard_exit()  # second call must be a no-op
    assert log.count("shutdown_memory_provider") == 1
    assert _count_ends(log) == 1
    assert "on_session_end:1" in log  # forwarded the real transcript


def test_helper_is_noop_when_cleanup_already_ran():
    """If _run_cleanup already ran (set _cleanup_done), the helper must not
    double-emit — this is the no-double-emission-when-the-finally-also-runs
    guarantee."""
    log = []
    cli_mod._active_agent_ref = _recording_agent(log)
    cli_mod._run_cleanup()  # normal finally path emits once
    cli_mod._emit_session_end_before_hard_exit()  # hard-exit helper: no-op
    assert log.count("shutdown_memory_provider") == 1
    assert _count_ends(log) == 1


@patch("hermes_cli.plugins.invoke_hook")
def test_raising_provider_does_not_prevent_exit(mock_invoke_hook):
    """A memory provider on_session_end that raises is swallowed — the helper
    must not propagate it, so the process can still os._exit(0) after."""
    log = []
    cli_mod._active_agent_ref = _recording_agent(log, raise_on_end=True)
    # Must not raise.
    cli_mod._emit_session_end_before_hard_exit()
    # Provider was called despite raising; nothing escaped.
    assert _count_ends(log) == 1
    assert log.count("shutdown_memory_provider") == 1


def test_helper_noop_without_active_agent():
    cli_mod._emit_session_end_before_hard_exit()  # must not raise
    assert cli_mod._cleanup_done is False  # nothing to emit, shouldn't mark done


# ---------------------------------------------------------------------------
# Signal-handler path — subprocess synthetic-mirror (real helper invoked)
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# Mirrors _signal_handler_q's kanban (HERMES_KANBAN_TASK) branch: signal the
# handler, which calls the REAL cli._emit_session_end_before_hard_exit() right
# before os._exit(0) — recording the emission to a marker file — so we can
# assert the signal path delivered on_session_end exactly once.
_SIGNAL_PATH_SRC = """
import os, signal, sys, threading, time
sys.path.insert(0, {repo!r})
import cli

def recording_agent():
    class Rec:
        session_id = "sig-session"
        _session_messages = [{{"role": "user", "content": "hi"}}]
        def shutdown_memory_provider(self, messages=None):
            with open({marker!r}, "a") as f:
                f.write(f"shutdown_memory_provider:{{len(messages or [])}}\\n")
                f.flush()
    return Rec()

cli._active_agent_ref = recording_agent()
cli._cleanup_done = False

# Mirror the production handler's kanban branch shape (flat, no closure deps
# on main()'s locals besides cli._active_agent_ref).
def handler(signum, frame):
    if os.environ.get("HERMES_KANBAN_TASK"):
        try:
            if hasattr(signal, "SIGALRM"):
                signal.signal(signal.SIGALRM, lambda *_: os._exit(0))
                signal.alarm(5)
        except Exception:
            pass
        cli._emit_session_end_before_hard_exit()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    raise KeyboardInterrupt()

signal.signal(signal.SIGTERM, handler)
print("READY", flush=True)
stuck = threading.Event()
threading.Thread(target=stuck.wait, daemon=False).start()
try:
    stuck.wait()
except KeyboardInterrupt:
    sys.exit(0)
"""


def _read_marker(marker_path):
    if not os.path.exists(marker_path):
        return []
    with open(marker_path) as f:
        return [l for l in f.read().splitlines() if l.strip()]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_signal_handler_path_emits_exactly_once_before_exit(tmp_path):
    """A kanban worker's SIGTERM emits on_session_end exactly once before
    os._exit(0).

    Only the ``HERMES_KANBAN_TASK`` branch is tested: that is the code path we
    added (the hard ``os._exit(0)`` that would otherwise skip the memory
    session-end). The non-kanban branch just raises ``KeyboardInterrupt`` and
    falls through to normal ``finally``/``atexit`` cleanup — base behaviour,
    not ours, so it is not asserted here.
    """
    marker = tmp_path / "emit.log"
    src = _SIGNAL_PATH_SRC.format(repo=_REPO_ROOT, marker=str(marker))
    env = dict(os.environ)
    env["HERMES_KANBAN_TASK"] = "t_hm2"
    env.pop("PYTEST_CURRENT_TEST", None)
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", src],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "READY"
        os.kill(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("process did not exit after SIGTERM")
        assert _read_marker(marker) == ["shutdown_memory_provider:1"]
    finally:
        if proc.poll() is None:
            proc.kill()


# ---------------------------------------------------------------------------
# Watchdog path — subprocess (real _arm_exit_watchdog + real helper)
# ---------------------------------------------------------------------------

# Drives the REAL cli._arm_exit_watchdog: a recording agent is set on
# cli._active_agent_ref with _cleanup_done still False (no finally ran), so
# when the watchdog fires it must call the real helper and emit exactly once
# before os._exit(0).
_WATCHDOG_PATH_SRC = """
import os, sys, time
sys.path.insert(0, {repo!r})
import cli

def recording_agent():
    class Rec:
        session_id = "wd-session"
        _session_messages = [{{"role": "user", "content": "hi"}}]
        def shutdown_memory_provider(self, messages=None):
            with open({marker!r}, "a") as f:
                f.write(f"shutdown_memory_provider:{{len(messages or [])}}\\n")
                f.flush()
    return Rec()

cli._active_agent_ref = recording_agent()
cli._cleanup_done = False
cli._arm_exit_watchdog(timeout_s=0.3)
# Park; the watchdog thread must fire os._exit(0) after ~0.3s. If it never
# does (regression), we reach the 99 exit below.
time.sleep(5)
os._exit(99)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX watchdog")
def test_watchdog_path_emits_exactly_once_before_exit(tmp_path):
    """The exit watchdog, firing on a wedged process whose cleanup never ran,
    emits on_session_end exactly once before os._exit(0)."""
    marker = tmp_path / "emit.log"
    src = _WATCHDOG_PATH_SRC.format(repo=_REPO_ROOT, marker=str(marker))
    env = dict(os.environ)
    env.pop("PYTEST_CURRENT_TEST", None)  # _arm_exit_watchdog refuses under pytest
    env["HERMES_EXIT_WATCHDOG_S"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", src],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        rc = proc.wait(timeout=10)
        # The watchdog os._exit(0) exits 0; the fallback os._exit(99) is a
        # regression signal.
        assert rc == 0, f"watchdog exit code {rc}; expected 0"
        assert _read_marker(marker) == ["shutdown_memory_provider:1"]
    finally:
        if proc.poll() is None:
            proc.kill()
