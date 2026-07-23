"""P0-F M2 regression: formal-worker SIGTERM must not orphan an external CLI.

Before the fix, cli.py's single-query SIGTERM handler (`_signal_handler_q`)
ran `agent.interrupt()`, slept a grace window, then called `os._exit(0)`
whenever `HERMES_KANBAN_TASK` was set — regardless of whether an
`ExternalCliAgentAdapter` run was in flight. Because the external CLI child
starts in its own session/process group (`start_new_session=True`), the
handler's `os._exit(0)` terminated the formal worker before the adapter's
poll loop (paused in the same interrupted `time.sleep(0.05)` on the main
thread) ever got a chance to resume, detect cancellation, and kill/reap the
child — orphaning it.

This must run as a real, separate OS process: it installs real signal
handlers and, if the bug regresses, would call `os._exit(0)` on itself. That
must not be able to take down the pytest runner.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_WORKER_SCRIPT = textwrap.dedent(
    """
    import os
    import signal
    import subprocess
    import sys
    import threading
    import time
    from types import SimpleNamespace

    repo_root, marker_path = sys.argv[1], sys.argv[2]
    sys.path.insert(0, repo_root)

    os.environ["HERMES_KANBAN_TASK"] = "t_fake"
    os.environ.pop("HERMES_KANBAN_GOAL_MODE", None)

    import cli as cli_mod
    import hermes_cli.config as config_mod
    import hermes_cli.kanban_db as kb_mod
    import hermes_cli.external_cli_adapter as adapter_mod
    from hermes_cli.external_cli_adapter import ClaudeCliStrategy

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.console = SimpleNamespace(print=lambda *a, **k: None)
            self.session_id = "worker-session"
            self.agent = SimpleNamespace(
                session_id="worker-session", platform="cli",
                interrupt=lambda *_a, **_k: None,
            )
            self._last_chat_result = None

        def _claim_active_session(self, surface, *, stderr=False):
            return True

        def _show_security_advisories(self):
            pass

        def _print_exit_summary(self):
            pass

    cli_mod.HermesCLI = FakeCLI
    cli_mod._finalize_single_query = lambda _cli: None
    cli_mod.atexit.register = lambda *_a, **_k: None

    config_mod.load_config = lambda: {
        "worker": {
            "execution_backend": "external_cli",
            "external_cli": {
                "executable": "claude",
                "authentication_mode": "cli_managed_subscription",
                "output_mode": "structured",
            },
        }
    }

    class _FakeConn:
        def close(self):
            return None

    kb_mod.connect = lambda: _FakeConn()
    kb_mod.get_task = lambda _conn, _task_id: SimpleNamespace(
        id="t_fake", title="Ship it", body="Run tests",
        assignee="claude-coder", max_runtime_seconds=30,
    )

    def _unexpected(name):
        def _fail(*_a, **_k):
            raise AssertionError(f"{name} should not be called")
        return _fail

    kb_mod.complete_task = _unexpected("complete_task")
    kb_mod.block_task = _unexpected("block_task")

    # A real child that never reads stdin and never exits on its own; the
    # adapter (not this harness) is responsible for killing and reaping it.
    ClaudeCliStrategy.build_argv = lambda self, cfg, req: [
        sys.executable, "-c", "import time; time.sleep(30)",
    ]
    adapter_mod.shutil.which = lambda _n: "/usr/bin/claude"

    real_popen = subprocess.Popen

    def _capturing_popen(argv, **kwargs):
        proc = real_popen(argv, **kwargs)
        with open(marker_path, "w") as f:
            f.write(str(proc.pid))
        return proc

    adapter_mod.subprocess.Popen = _capturing_popen

    def _send_sigterm_soon():
        time.sleep(1.0)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_send_sigterm_soon, daemon=True).start()

    cli_mod.main(query="work kanban task t_fake", quiet=True, toolsets="terminal")
    """
)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@pytest.mark.skipif(os.name != "posix", reason="SIGTERM/process-group semantics are POSIX-specific")
def test_sigterm_during_external_cli_run_does_not_orphan_child(tmp_path):
    script_path = tmp_path / "worker_harness.py"
    marker_path = tmp_path / "child_pid.txt"
    script_path.write_text(_WORKER_SCRIPT)

    proc = subprocess.Popen(
        [sys.executable, str(script_path), str(REPO_ROOT), str(marker_path)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        stdout, _ = proc.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, _ = proc.communicate(timeout=5)
        pytest.fail(f"formal-worker harness did not exit within 15s after SIGTERM:\n{stdout}")

    deadline_msg = f"harness output:\n{stdout}"
    assert marker_path.exists(), f"adapter never Popen'd the external CLI child.\n{deadline_msg}"
    child_pid = int(marker_path.read_text().strip())

    # Bounded wait: the child may take a beat to actually be reaped after the
    # harness process itself has exited.
    import time as _time

    reap_deadline = _time.monotonic() + 5.0
    while _time.monotonic() < reap_deadline and _pid_alive(child_pid):
        _time.sleep(0.1)

    assert not _pid_alive(child_pid), (
        f"external CLI child pid={child_pid} survived formal-worker SIGTERM — "
        f"orphan bug regressed (P0-F M2).\n{deadline_msg}"
    )
