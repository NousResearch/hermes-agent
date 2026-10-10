"""A SIGKILL inside the spawner's grace window must not take the session id with it.

`hermes chat -Q` is spawned by orchestrators that stop children with SIGTERM and then
SIGKILL the group after their own (often short) grace. The single-query signal handler
interrupts the agent and sleeps ``HERMES_SIGTERM_GRACE`` *before* raising
``KeyboardInterrupt`` — so on main the ``session_id:`` line only ever appears after that
sleep. A SIGKILL landing inside it kills the process with the line never printed, and
the wrapper that parses stderr for the id loses a session that is safely persisted.

The handler now publishes the id line first, before any interrupt-grace work.
"""

from __future__ import annotations

import io
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from hermes_cli import cli_single_query

_REPO_ROOT = Path(cli_single_query.__file__).resolve().parent.parent

# A standalone run mirroring the quiet path: handler first publishes the id line
# (only while a quiet turn is due), then sleeps the interrupt grace, then unwinds.
_CHILD_TEMPLATE = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {repo_root!r})
    from hermes_cli import cli_single_query as csq

    class Agent:
        session_id = "SESS-121890"
        def run_conversation(self, **kwargs):
            print("READY", flush=True)
            while True:
                time.sleep(0.05)

    class Cli:
        agent = Agent()
        session_id = "SESS-121890"

    def _grace(agent, signum):
        print("GRACE-START", flush=True)
        time.sleep({grace!r})
        print("GRACE-END", flush=True)

    csq._quiet_session_line_due = True
    cli = Cli()
    # The interrupt-grace seam the handler calls; the real one sleeps 1.5 s.
    csq._interrupt_agent_for_signal = _grace
    csq._install_single_query_signal_handlers(cli)
    try:
        cli.agent.run_conversation(user_message="q")
    except KeyboardInterrupt:
        csq._emit_interrupted_session_end = lambda *a, **k: None
        csq._emit_quiet_session_line(cli)
        sys.exit(17)
    """
)


def _spawn(grace: float):
    script = _CHILD_TEMPLATE.format(repo_root=str(_REPO_ROOT), grace=grace)
    return subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(_REPO_ROOT),
    )


def _await_ready(proc: subprocess.Popen) -> None:
    assert proc.stdout is not None
    for _ in range(200):
        line = proc.stdout.readline()
        if line.strip() == b"READY":
            return
    raise RuntimeError("mirrored quiet run never reported READY")


def _shutdown(proc: subprocess.Popen) -> None:
    if proc.stdout is not None:
        proc.stdout.close()
    if proc.stderr is not None:
        proc.stderr.close()
    proc.kill()
    proc.wait(timeout=10)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM semantics differ on Windows; the quiet spawner contract is POSIX-only")
def test_sigkill_inside_the_interrupt_grace_still_delivers_the_session_id():
    # SIGKILL lands mid-grace — on main the id line is never printed at all.
    proc = _spawn(grace=30.0)
    try:
        _await_ready(proc)
        proc.send_signal(signal.SIGTERM)
        deadline = time.perf_counter() + 10.0
        assert proc.stderr is not None
        # The line prints as "\nsession_id: …" — a lone newline may arrive first.
        received = b""
        while time.perf_counter() < deadline and b"session_id:" not in received:
            received += proc.stderr.readline()
        assert b"session_id: SESS-121890" in received, "the id line must precede the interrupt-grace sleep"
    finally:
        _shutdown(proc)


def test_the_handler_stays_silent_outside_a_quiet_turn():
    # Interactive mode and post-exit signals must not gain a stray stderr line.
    idle_cli = type("Cli", (), {"session_id": "SESS-IDLE"})()
    cli_single_query._quiet_session_line_due = False
    stderr = io.TextIOWrapper(io.BytesIO(), write_through=True)
    real_stderr, sys.stderr = sys.stderr, stderr
    try:
        cli_single_query._emit_quiet_session_line(idle_cli)
    finally:
        sys.stderr = real_stderr
    assert stderr.buffer.getvalue() == b"", "no id line may print outside a quiet one-shot turn"


def test_the_id_line_is_flushed_and_printed_once():
    cli = type("Cli", (), {"session_id": "SESS-ONCE"})()
    cli_single_query._quiet_session_line_due = True
    stderr = io.TextIOWrapper(io.BytesIO(), write_through=True)
    real_stderr, sys.stderr = sys.stderr, stderr
    try:
        cli_single_query._emit_quiet_session_line(cli)
        cli_single_query._emit_quiet_session_line(cli)  # normal exit after an early emit
    finally:
        sys.stderr = real_stderr
        cli_single_query._quiet_session_line_due = False
    assert stderr.buffer.getvalue() == b"\nsession_id: SESS-ONCE\n", "exactly one id line, bytes visible on the spot"
