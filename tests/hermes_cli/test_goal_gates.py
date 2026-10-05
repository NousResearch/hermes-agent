"""Tests for /goal quality gates (GoalGate, run_gate, GoalManager gate flow)."""

import json
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import pytest

from hermes_cli.goals import (
    DEFAULT_GATE_MAX_RETRIES,
    DEFAULT_GATE_TIMEOUT_SECONDS,
    GoalGate,
    GoalManager,
    GoalState,
    run_gate,
)


# ──────────────────────────────────────────────────────────────────────
# GoalGate serialization
# ──────────────────────────────────────────────────────────────────────


def test_gate_roundtrip_through_goalstate_json():
    state = GoalState(goal="ship it", status="active")
    state.gates.append(GoalGate(command="echo ok", timeout_seconds=42, max_retries=7))
    raw = state.to_json()
    loaded = GoalState.from_json(raw)
    assert len(loaded.gates) == 1
    g = loaded.gates[0]
    assert g.command == "echo ok"
    assert g.timeout_seconds == 42
    assert g.max_retries == 7
    assert g.attempts == 0


def test_gate_from_dict_defaults_and_garbage():
    g = GoalGate.from_dict({"command": "true"})
    assert g.timeout_seconds == DEFAULT_GATE_TIMEOUT_SECONDS
    assert g.max_retries == DEFAULT_GATE_MAX_RETRIES
    assert GoalGate.from_dict(None).command == ""
    assert GoalGate.from_dict("nonsense").command == ""


def test_old_state_rows_without_gates_load_clean():
    """Backwards compatibility: pre-gates state_meta rows load with no gates."""
    old = {"goal": "legacy", "status": "active"}
    state = GoalState.from_json(json.dumps(old))
    assert state.gates == []


# ──────────────────────────────────────────────────────────────────────
# run_gate
# ──────────────────────────────────────────────────────────────────────


def test_run_gate_pass():
    passed, code, out = run_gate(GoalGate(command="echo hello"))
    assert passed is True
    assert code == 0
    assert "hello" in out


@pytest.mark.platforms("linux")
def test_run_gate_fail_captures_output():
    # POSIX shell syntax (`>&2`, `;`, `exit`) — cmd.exe (shell=True on Windows)
    # doesn't parse it, so the gate "passes" instead of failing.
    passed, code, out = run_gate(GoalGate(command="echo broken >&2; exit 3"))
    assert passed is False
    assert code == 3
    assert "broken" in out


def test_run_gate_timeout():
    passed, code, out = run_gate(GoalGate(command="sleep 5", timeout_seconds=1))
    assert passed is False
    assert code == -1
    assert "timed out" in out


def test_run_gate_keeps_diagnostics_when_a_byte_will_not_decode(tmp_path):
    """A gate's output tail must survive bytes the decoder rejects.

    A gate runs whatever the operator configured, so its output is arbitrary
    bytes — a test runner's checkmarks or CJK on a non-UTF-8 Windows console,
    or stray binary. Decoding strictly means one bad byte kills subprocess's
    reader thread, stdout comes back None, and the tail lands empty: the agent
    is told the gate failed with nothing to act on, so it burns every retry and
    the goal auto-pauses.
    """
    script = tmp_path / "gate.py"
    script.write_text(
        "import os, sys\n"
        "os.write(1, b'FAILED: 3 tests broken \\x90\\x8d rerun me\\n')\n"
        "sys.exit(1)\n",
        encoding="utf-8",
    )

    passed, code, out = run_gate(
        GoalGate(command=f'"{sys.executable}" "{script}"'),
    )

    assert passed is False
    assert code == 1
    assert "FAILED: 3 tests broken" in out, (
        f"gate diagnostics were lost to a decode failure (tail={out!r})"
    )


def _open_descendant(ready):
    """Open a handle on the descendant named by the ready file, only if its identity checks out.

    The handle is closed (never terminated) when validation fails, so a stale or reused PID is left alone.
    """
    import _winapi
    import psutil

    pid, created = json.loads(ready.read_text(encoding="utf-8-sig"))
    handle = _winapi.OpenProcess(0x100001, False, pid)  # SYNCHRONIZE | PROCESS_TERMINATE
    try:
        assert psutil.Process(pid).create_time() == created
    except BaseException:
        _winapi.CloseHandle(handle)
        raise
    return handle


def _release_descendant(handle):
    """Stop a validated descendant if it is still running, and always release the handle."""
    import _winapi

    try:
        if _winapi.WaitForSingleObject(handle, 0) == _winapi.WAIT_TIMEOUT:
            try:
                _winapi.TerminateProcess(handle, 1)
            except OSError:
                pass  # it exited between the check and the call
            _winapi.WaitForSingleObject(handle, 2000)
    finally:
        _winapi.CloseHandle(handle)


@pytest.mark.platforms("windows")
def test_run_gate_timeout_terminates_pipe_holding_descendant(tmp_path):
    """Regression for #132325: timeout must release pipes and terminate their owner."""
    import _winapi
    import psutil

    ready = tmp_path / "ready.json"
    script = tmp_path / "finite gate.py"
    script.write_text(
        "import json, os, pathlib, psutil, sys, time\n"
        "os.write(1, b'BEFORE_STDOUT\\n'); os.write(2, b'BEFORE_STDERR\\n')\n"
        "p = psutil.Process()\n"
        "ready = pathlib.Path(sys.argv[1]); staging = ready.with_suffix('.tmp')\n"
        "staging.write_text(json.dumps([p.pid, p.create_time()]), encoding='utf-8'); staging.replace(ready)\n"
        "time.sleep(6)\n"
        "os.write(1, b'AFTER_DEADLINE\\n')\n",
        encoding="utf-8",
    )
    results = []
    elapsed = []

    def run():
        start = time.monotonic()
        results.append(run_gate(GoalGate(
            command=f'echo SHELL_MARKER & "{sys.executable}" "{script}" "{ready}"',
            timeout_seconds=1,
        ), cwd=str(tmp_path)))
        elapsed.append(time.monotonic() - start)

    worker = threading.Thread(target=run, daemon=True)
    handle = None
    worker.start()
    try:
        deadline = time.monotonic() + 4
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "descendant never became ready"
        handle = _open_descendant(ready)
        worker.join(timeout=7)
        assert not worker.is_alive(), "run_gate did not finish after the finite fixture"
        assert elapsed[0] < 4, f"1s gate took {elapsed[0]:.3f}s waiting for descendant pipes"
        passed, code, output = results[0]
        assert (passed, code) == (False, -1)
        assert all(marker in output for marker in ("SHELL_MARKER", "BEFORE_STDOUT", "BEFORE_STDERR", "timed out"))
        assert "AFTER_DEADLINE" not in output
        assert _winapi.WaitForSingleObject(handle, 2000) == _winapi.WAIT_OBJECT_0
    finally:
        try:
            if handle is not None:
                _release_descendant(handle)
        finally:
            worker.join(timeout=2)


def _descendant_script(tmp_path):
    """A descendant that reports its identity, then lives far longer than any permitted cleanup window."""
    ready = tmp_path / "descendant-ready.json"
    script = tmp_path / "descendant.py"
    script.write_text(
        "import json, os, pathlib, psutil, sys, time\n"
        "p = psutil.Process()\n"
        "ready = pathlib.Path(sys.argv[1]); staging = ready.with_suffix('.tmp')\n"
        "staging.write_text(json.dumps([p.pid, p.create_time()]), encoding='utf-8'); staging.replace(ready)\n"
        "print('CHILD_BEFORE', flush=True)\n"
        "time.sleep(60)\n", encoding="utf-8")
    return script, ready


def _run_gate_with_failing_job_close(command, tmp_path, ready, *, no_job_terminate=False):
    """run_gate whose Windows job fails its first close().

    Returns (verdict, killed pids, seconds, (close attempts, handle released, descendant exited)).
    The descendant is observed through a process handle, so there is no timing race with its own lifetime.
    The handle state is captured before the real job is closed here, so a leak by the gate is visible.
    """
    import _winapi

    from hermes_cli import _subprocess_compat as compat
    from hermes_cli.local_runtime import processes

    real_spawn = processes.spawn_server
    real_kill = compat.kill_process_tree
    killed, jobs, results, elapsed = [], [], [], []
    handle = None

    class FailFirstClose:
        def __init__(self, job):
            self.real = job
            self._handle = job._handle
            self.attempts = 0

        def close(self):
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("CloseHandle failed")
            self.real.close()

    def spawn(*args, **kwargs):
        proc, job = real_spawn(*args, **kwargs)
        jobs.append(FailFirstClose(job))
        return proc, jobs[-1]

    def kill(proc):
        killed.append(proc.pid)
        real_kill(proc)

    def run():
        start = time.monotonic()
        results.append(run_gate(GoalGate(command=command, timeout_seconds=1), cwd=str(tmp_path)))
        elapsed.append(time.monotonic() - start)

    terminate = (lambda job: False) if no_job_terminate else compat._terminate_job
    try:
        with patch.object(processes, "spawn_server", spawn), \
                patch.object(compat, "kill_process_tree", kill), \
                patch.object(compat, "_terminate_job", terminate):
            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            deadline = time.monotonic() + 8
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.exists(), "descendant never became ready"
            handle = _open_descendant(ready)
            worker.join(timeout=30)
            assert not worker.is_alive(), "run_gate did not finish"
        # Generous bound that is still far below the descendant's own 60s lifetime.
        exited = _winapi.WaitForSingleObject(handle, 20000) == _winapi.WAIT_OBJECT_0
        # Snapshot before the cleanup below, so a handle the gate failed to release is not masked.
        released = jobs[0].real._handle is None
        return results[0], killed, elapsed[0], (jobs[0].attempts, released, exited)
    finally:
        try:
            if handle is not None:
                _release_descendant(handle)
        finally:
            for job in jobs:
                job.real.close()


@pytest.mark.platforms("windows")
def test_run_gate_timeout_survives_a_failed_job_close(tmp_path):
    """A CloseHandle failure must not replace the timeout verdict, leave the tree running or leak the job."""
    script, ready = _descendant_script(tmp_path)
    (passed, code, output), killed, elapsed, job = _run_gate_with_failing_job_close(
        f'echo BEFORE_CLOSE & "{sys.executable}" "{script}" "{ready}"', tmp_path, ready)
    assert (passed, code) == (False, -1)
    assert "BEFORE_CLOSE" in output and "CHILD_BEFORE" in output and "timed out" in output
    assert "could not run" not in output
    assert elapsed < 10
    assert not killed, "job termination should not need the process-tree walk"
    assert job == (2, True, True), "the failed close was not retried, or the descendant survived"


@pytest.mark.platforms("windows")
def test_run_gate_failed_job_close_falls_back_to_the_process_tree(tmp_path):
    script, ready = _descendant_script(tmp_path)
    (passed, code, output), killed, _, job = _run_gate_with_failing_job_close(
        f'echo BEFORE_CLOSE & "{sys.executable}" "{script}" "{ready}"', tmp_path, ready, no_job_terminate=True)
    assert (passed, code) == (False, -1)
    assert "BEFORE_CLOSE" in output and "CHILD_BEFORE" in output and "timed out" in output
    assert killed
    assert job == (2, True, True), "the failed close was not retried, or the descendant survived"


@pytest.mark.platforms("windows")
def test_run_gate_failed_job_close_reaches_descendants_of_an_exited_shell(tmp_path):
    """taskkill /T cannot find orphans of a shell that already exited; the owned job still can."""
    script, ready = _descendant_script(tmp_path)
    (passed, code, output), killed, elapsed, job = _run_gate_with_failing_job_close(
        f'echo BEFORE_EXIT & start "" /b "{sys.executable}" "{script}" "{ready}" & exit /b 0', tmp_path, ready)
    assert (passed, code) == (False, -1)
    assert "BEFORE_EXIT" in output and "CHILD_BEFORE" in output and "timed out" in output
    assert elapsed < 10
    assert job == (2, True, True), "the failed close was not retried, or the orphan survived"


@pytest.mark.platforms("posix")
def test_run_gate_timeout_keeps_output_on_posix(tmp_path):
    start = time.monotonic()
    passed, code, output = run_gate(GoalGate(command="echo BEFORE_TIMEOUT; sleep 6; echo AFTER_DEADLINE", timeout_seconds=1),
                                    cwd=str(tmp_path))
    assert (passed, code) == (False, -1)
    assert "BEFORE_TIMEOUT" in output and "timed out" in output and "AFTER_DEADLINE" not in output
    assert time.monotonic() - start < 5


@pytest.mark.parametrize("mode", ["success", "nonzero", "bytes", "tail", "cwd", "missing-cwd"])
def test_run_gate_shell_diagnostic_contract(tmp_path, mode):
    script = tmp_path / "diagnostic gate.py"
    script.write_text(
        "import os, pathlib, sys\n"
        "mode = sys.argv[1]\n"
        "if mode == 'bytes': os.write(1, 'UNICODE: \\U0001f642 \\u4e2d\\n'.encode() + b'INVALID: \\xff\\n')\n"
        "elif mode == 'tail': os.write(1, b'A' * 6000 + b'TAIL_SENTINEL\\n')\n"
        "elif mode == 'cwd': print(pathlib.Path.cwd()); print(pathlib.Path('marker').read_text(encoding='utf-8-sig'))\n"
        "else: print('STDOUT'); print('STDERR', file=sys.stderr)\n"
        "sys.exit(3 if mode in ('nonzero', 'bytes') else 0)\n",
        encoding="utf-8",
    )
    (tmp_path / "marker").write_text("OWNED_CWD", encoding="utf-8")
    cwd = tmp_path / "absent" if mode == "missing-cwd" else tmp_path
    # A real shell operator exercises the shell=True parsing contract on each host.
    separator = "&" if sys.platform == "win32" else ";"
    result = run_gate(GoalGate(command=f'echo SHELL_MARKER {separator} "{sys.executable}" "{script}" {mode}'), cwd=str(cwd))
    passed, code, output = result
    if mode == "missing-cwd":
        assert (passed, code) == (False, -1)
        assert "gate could not run" in output and "timed out" not in output
    elif mode == "tail":
        assert (passed, code) == (True, 0)
        assert len(output) == 3000 and output.endswith("TAIL_SENTINEL\n")
    else:
        assert (passed, code) == (mode not in ("nonzero", "bytes"), 3 if mode in ("nonzero", "bytes") else 0)
        assert "SHELL_MARKER" in output
        if mode == "bytes":
            assert "UNICODE: \U0001f642 \u4e2d" in output and "INVALID: \ufffd" in output
        elif mode == "cwd":
            assert str(tmp_path) in output and "OWNED_CWD" in output
        else:
            assert "STDOUT" in output and "STDERR" in output


# ──────────────────────────────────────────────────────────────────────
# GoalManager gate management
# ──────────────────────────────────────────────────────────────────────


def _mgr_with_goal(session_id="gate-test-sid"):
    mgr = GoalManager(session_id=session_id)
    mgr.set("test goal")
    return mgr


def test_add_remove_clear_gates():
    mgr = _mgr_with_goal("gate-mgmt-sid")
    mgr.add_gate("echo one")
    mgr.add_gate("echo two")
    assert len(mgr.state.gates) == 2
    assert "echo one" in mgr.render_gates()

    removed = mgr.remove_gate(1)
    assert removed == "echo one"
    assert len(mgr.state.gates) == 1

    assert mgr.clear_gates() == 1
    assert mgr.state.gates == []


def test_add_gate_requires_active_goal():
    mgr = GoalManager(session_id="gate-nogoal-sid")
    with pytest.raises(RuntimeError):
        mgr.add_gate("echo nope")


def test_gates_persist_and_reload():
    mgr = _mgr_with_goal("gate-persist-sid")
    mgr.add_gate("echo persisted")
    reloaded = GoalManager(session_id="gate-persist-sid")
    assert len(reloaded.state.gates) == 1
    assert reloaded.state.gates[0].command == "echo persisted"




# ──────────────────────────────────────────────────────────────────────
# evaluate_after_turn integration
# ──────────────────────────────────────────────────────────────────────


def test_failing_gate_short_circuits_judge():
    mgr = _mgr_with_goal("gate-fail-sid")
    mgr.add_gate("exit 5")
    with patch("hermes_cli.goals.judge_goal") as mock_judge:
        decision = mgr.evaluate_after_turn("I think it's done!")
    mock_judge.assert_not_called()
    assert decision["verdict"] == "gate_failed"
    assert decision["should_continue"] is True
    assert "exit 5" in decision["continuation_prompt"]
    assert "quality gate" in decision["continuation_prompt"].lower()


def test_passing_gates_fall_through_to_judge():
    mgr = _mgr_with_goal("gate-pass-sid")
    mgr.add_gate("true")
    with patch(
        "hermes_cli.goals.judge_goal",
        return_value=("done", "all good", False, None, False),
    ) as mock_judge:
        decision = mgr.evaluate_after_turn("finished")
    mock_judge.assert_called_once()
    assert decision["verdict"] == "done"
    # Passing run resets attempt bookkeeping.
    assert mgr.state.gates[0].attempts == 0
    assert mgr.state.gates[0].last_exit_code == 0


def test_gate_retry_exhaustion_pauses_goal():
    mgr = _mgr_with_goal("gate-exhaust-sid")
    mgr.add_gate("exit 1")
    mgr.state.gates[0].max_retries = 2
    with patch("hermes_cli.goals.judge_goal") as mock_judge:
        d1 = mgr.evaluate_after_turn("attempt one")
        d2 = mgr.evaluate_after_turn("attempt two")
        d3 = mgr.evaluate_after_turn("attempt three")
    mock_judge.assert_not_called()
    assert d1["should_continue"] is True
    assert d2["should_continue"] is True
    assert d3["status"] == "paused"
    assert d3["should_continue"] is False
    assert mgr.state.status == "paused"
    assert "gate" in (mgr.state.paused_reason or "")


def test_failed_gate_reruns_when_untracked_file_content_changes(tmp_path, monkeypatch):
    """#110649: `git status --porcelain` reports the same `?? untracked/` for `before` and
    `after`, so a status-based cache replayed the stale failure; the gate must execute again."""
    for argv in (["init", "-q"], ["config", "user.email", "t@example.com"], ["config", "user.name", "T"],
                 ["commit", "-q", "--allow-empty", "-m", "baseline"]):
        subprocess.run(["git", *argv], cwd=tmp_path, check=True, capture_output=True)
    result = tmp_path / "untracked" / "result.txt"
    result.parent.mkdir()
    result.write_text("before", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    mgr = _mgr_with_goal("gate-content-sid")
    mgr.add_gate(f"grep -q after {result}")
    with patch("hermes_cli.goals.judge_goal", return_value=("done", "ok", False, None, False)) as judge:
        d1 = mgr.evaluate_after_turn("turn 1")
        result.write_text("after", encoding="utf-8")
        d2 = mgr.evaluate_after_turn("turn 2")
    assert d1["verdict"] == "gate_failed"
    assert d2["verdict"] == "done"
    judge.assert_called_once()
    assert mgr.state.gates[0].attempts == 0


def test_gate_continuation_respects_turn_budget():
    mgr = GoalManager(session_id="gate-budget-sid", default_max_turns=1)
    mgr.set("budget goal")
    mgr.add_gate("exit 1")
    with patch("hermes_cli.goals.judge_goal"):
        decision = mgr.evaluate_after_turn("only turn")
    assert decision["status"] == "paused"
    assert decision["should_continue"] is False
    assert "turns used" in decision["message"]


def test_no_gates_behaves_exactly_as_before():
    mgr = _mgr_with_goal("gate-none-sid")
    with patch(
        "hermes_cli.goals.judge_goal",
        return_value=("continue", "keep going", False, None, False),
    ) as mock_judge:
        decision = mgr.evaluate_after_turn("wip")
    mock_judge.assert_called_once()
    assert decision["verdict"] == "continue"
    assert decision["should_continue"] is True


# ──────────────────────────────────────────────────────────────────────
# Gate working directory (#125369)
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def backend_and_session(tmp_path, monkeypatch):
    """A backend started in a project whose check passes, serving a session whose check fails."""
    from agent.runtime_cwd import reset_session_cwd, set_session_cwd

    backend, session = tmp_path / "backend", tmp_path / "session"
    for folder, code in ((backend, 0), (session, 1)):
        folder.mkdir()
        (folder / "check.sh").write_text(f"pwd\nexit {code}\n", encoding="utf-8")
    monkeypatch.chdir(backend)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    def bind(cwd):
        token = set_session_cwd(str(cwd))
        return lambda: reset_session_cwd(token)

    return backend, session, bind


def _evaluate_with_done_judge(mgr):
    with patch("hermes_cli.goals.judge_goal", return_value=("done", "all good", False, None, False)) as judge:
        return mgr.evaluate_after_turn("ready"), judge


def test_gate_runs_in_the_session_workspace_not_the_backend_directory(backend_and_session):
    backend, session, bind = backend_and_session
    mgr = _mgr_with_goal("gate-cwd-sid")
    mgr.add_gate("sh check.sh")
    unbind = bind(session)
    try:
        decision, judge = _evaluate_with_done_judge(mgr)
    finally:
        unbind()
    judge.assert_not_called()
    assert decision["verdict"] == "gate_failed"
    assert mgr.state.gates[0].last_exit_code == 1
    assert str(session.resolve()) in mgr.state.gates[0].last_output_tail


def test_missing_session_workspace_pauses_instead_of_running_elsewhere(backend_and_session, tmp_path):
    # A deleted, remote or container workspace: the backend's passing check must not stand in for it,
    # and no retry can fix it, so the goal pauses on the first check with the reason and no attempt charged.
    backend, _session, bind = backend_and_session
    missing = tmp_path / "gone"
    mgr = _mgr_with_goal("gate-missing-cwd-sid")
    mgr.add_gate("sh check.sh")
    unbind = bind(missing)
    try:
        with patch("hermes_cli.goals.run_gate") as run:
            decision, judge = _evaluate_with_done_judge(mgr)
    finally:
        unbind()
    run.assert_not_called()
    judge.assert_not_called()
    assert decision["status"] == "paused" and decision["should_continue"] is False
    assert str(missing) in decision["message"] and "still failing" not in decision["message"]
    assert str(missing) in (mgr.state.paused_reason or "")
    gate = mgr.state.gates[0]
    assert (gate.attempts, gate.last_exit_code) == (0, None)


def test_gate_without_a_session_workspace_keeps_the_launch_directory(backend_and_session):
    from agent.runtime_cwd import clear_session_cwd

    backend, _session, _bind = backend_and_session
    clear_session_cwd()
    mgr = _mgr_with_goal("gate-launch-cwd-sid")
    mgr.add_gate("sh check.sh")
    decision, judge = _evaluate_with_done_judge(mgr)
    judge.assert_called_once()
    assert decision["verdict"] == "done"
    assert str(backend.resolve()) in mgr.state.gates[0].last_output_tail
