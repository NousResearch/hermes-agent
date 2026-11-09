"""Regression tests for #127089: post-update respawn inside a Windows Job Object.

The failure: ``hermes update`` (or the desktop hand-off) ran inside a
kill-on-close Windows Job Object, ``CREATE_BREAKAWAY_FROM_JOB`` was denied, and
the plain-``Popen`` fallback left the respawned gateway in that job. The
updater exited, the job tore down, and the gateway died with it.

The fix: when the caller is inside a job, hand the spawn to a transient Task
Scheduler task (or an already-installed gateway task), which starts outside the
caller's job; if the handoff fails, fail closed instead of silently respawning
in-job.

These tests cover the decision logic and one native proof that a process
started by the transient task survives ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import _subprocess_compat, gateway, gateway_windows

pytestmark = pytest.mark.platforms("windows")

REPO_ROOT = Path(__file__).resolve().parents[2]


def _enable_handoff(monkeypatch) -> None:
    monkeypatch.delenv("HERMES_GATEWAY_DISABLE_JOB_HANDOFF", raising=False)
    monkeypatch.setattr(_subprocess_compat, "process_is_in_job", lambda: True)


def _fake_spec(argv, home=None):
    return list(argv), r"C:\cwd", {"HERMES_HOME": r"C:\home"}


def test_process_is_in_job_returns_bool():
    assert isinstance(_subprocess_compat.process_is_in_job(), bool)


def test_in_job_restart_uses_transient_task(monkeypatch):
    _enable_handoff(monkeypatch)
    monkeypatch.setattr(gateway_windows, "windowless_gateway_restart_spec", _fake_spec)
    monkeypatch.setattr(gateway, "_host_gateway_watcher_env", dict)
    captured: dict = {}

    def fake_handoff(old_pid, run_argv, **kwargs):
        captured.update(old_pid=old_pid, run_argv=list(run_argv), **kwargs)
        return True

    monkeypatch.setattr(gateway_windows, "_spawn_gateway_via_transient_task", fake_handoff)
    monkeypatch.setattr(
        gateway.subprocess, "Popen",
        lambda *a, **k: pytest.fail("in-job Popen must not be used"),
    )

    assert gateway._spawn_gateway_restart_watcher(
        1234, ["python", "-m", "hermes_cli.main", "gateway", "run"], host=True,
    )
    assert captured["old_pid"] == 1234
    assert captured["run_argv"][-2:] == ["gateway", "run"]
    assert captured["respawn_cwd"] == r"C:\cwd"


def test_in_job_restart_fails_closed_when_handoff_fails(monkeypatch):
    _enable_handoff(monkeypatch)
    monkeypatch.setattr(gateway_windows, "windowless_gateway_restart_spec", _fake_spec)
    monkeypatch.setattr(gateway, "_host_gateway_watcher_env", dict)

    def fail_handoff(*a, **k):
        raise RuntimeError("schtasks /Create failed: access denied")

    monkeypatch.setattr(gateway_windows, "_spawn_gateway_via_transient_task", fail_handoff)
    monkeypatch.setattr(
        gateway.subprocess, "Popen",
        lambda *a, **k: pytest.fail("in-job Popen must not be used after a failed handoff"),
    )

    assert not gateway._spawn_gateway_restart_watcher(
        1234, ["python", "-m", "hermes_cli.main", "gateway", "run"], host=True,
    )


def test_not_in_job_keeps_legacy_watcher(monkeypatch):
    monkeypatch.delenv("HERMES_GATEWAY_DISABLE_JOB_HANDOFF", raising=False)
    monkeypatch.setattr(_subprocess_compat, "process_is_in_job", lambda: False)
    monkeypatch.setattr(gateway_windows, "windowless_gateway_restart_spec", _fake_spec)
    captured: dict = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs

        class _Proc:
            pid = 4242

        return _Proc()

    monkeypatch.setattr(gateway.subprocess, "Popen", fake_popen)
    assert gateway._spawn_gateway_restart_watcher(
        1234, ["python", "-m", "hermes_cli.main", "gateway", "run"], host=True,
    )
    assert captured["argv"][1] == "-c"


def test_transient_launcher_renders_none_env_and_compiles(tmp_path):
    src = gateway_windows._build_transient_task_launcher(
        task_name="Hermes_GW_Restart_test",
        temp_dir=str(tmp_path),
        old_pid=0,
        gateway_cmd=[sys.executable, "-c", "pass"],
        respawn_cwd="",
        respawn_env_overlay={},
        watcher_env=None,
        project_root=str(REPO_ROOT),
        watcher_timeout_s=1.0,
    )
    # ``json.dumps(None)`` would render ``null`` and raise NameError at runtime.
    assert "_WATCHER_ENV = None" in src
    assert "_WATCHER_ENV = null" not in src
    assert "HERMES_GATEWAY_TASK_LAUNCH" in src
    compile(src, "<transient-launcher>", "exec")


def test_scheduler_job_denied_breakaway_never_retries_in_job(tmp_path, monkeypatch):
    """The Scheduler launcher must not repeat the updater's doomed fallback."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(_subprocess_compat, "process_is_in_job", lambda: True)
    attempted_flags = []

    def denied_breakaway(*args, **kwargs):
        attempted_flags.append(kwargs["creationflags"])
        raise OSError(5, "breakaway denied")

    monkeypatch.setattr(subprocess, "Popen", denied_breakaway)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0),
    )
    launcher = gateway_windows._build_transient_task_launcher(
        task_name="Hermes_GW_Restart_test",
        temp_dir=str(tmp_path / "transient-task"),
        old_pid=0,
        gateway_cmd=[sys.executable, "-c", "pass"],
        respawn_cwd="",
        respawn_env_overlay={},
        watcher_env=None,
        project_root=str(REPO_ROOT),
        watcher_timeout_s=1.0,
    )

    with pytest.raises(OSError, match="breakaway denied"):
        exec(compile(launcher, "<transient-launcher>", "exec"), {"__name__": "__main__"})
    assert len(attempted_flags) == 1, "never retry without breakaway inside a Scheduler job"


def test_transient_task_xml_is_triggerless_hidden():
    xml = gateway_windows._build_transient_task_xml("Hermes_GW_Restart_test", "cmd.exe", "/c echo ok", None)
    assert "<Triggers />" in xml
    assert "<Hidden>true</Hidden>" in xml
    assert "<ExecutionTimeLimit>PT0S</ExecutionTimeLimit>" in xml


def test_spawn_detached_delegates_when_in_job(monkeypatch):
    _enable_handoff(monkeypatch)
    monkeypatch.setattr(
        gateway_windows, "_build_gateway_argv",
        lambda home=None: (["python", "-m", "hermes_cli.main", "gateway", "run"], r"C:\cwd", {"HERMES_HOME": r"C:\home"}),
    )
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway")
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda name=None: True)
    captured: dict = {}

    def fake_start(argv, **kwargs):
        captured.update(argv=list(argv), **kwargs)
        return True

    monkeypatch.setattr(gateway_windows, "_start_gateway_outside_job", fake_start)
    monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda **kwargs: [9999])
    monkeypatch.setattr(
        gateway_windows.subprocess, "Popen",
        lambda *a, **k: pytest.fail("in-job gateway spawn must not be used"),
    )

    assert gateway_windows._spawn_detached() == 9999
    assert captured["argv"][-2:] == ["gateway", "run"]
    assert captured["task_name"] == "Hermes_Gateway"
    assert captured["use_registered"] is True


def test_spawn_detached_fails_closed_when_handoff_rejected(monkeypatch):
    _enable_handoff(monkeypatch)
    monkeypatch.setattr(
        gateway_windows, "_build_gateway_argv",
        lambda home=None: (["python", "-m", "hermes_cli.main", "gateway", "run"], r"C:\cwd", {}),
    )
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway")
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda name=None: False)

    def fail_start(*a, **k):
        raise RuntimeError("schtasks /Create failed")

    monkeypatch.setattr(gateway_windows, "_start_gateway_outside_job", fail_start)
    monkeypatch.setattr(
        gateway_windows.subprocess, "Popen",
        lambda *a, **k: pytest.fail("in-job gateway spawn must not be used after a failed handoff"),
    )
    with pytest.raises(RuntimeError, match="refusing to start the gateway"):
        gateway_windows._spawn_detached()


# ---------------------------------------------------------------------------
# Native proof: the task-started process survives a kill-on-close job
# ---------------------------------------------------------------------------


class _BasicLimit(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimit),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _make_kill_on_close_job() -> int:
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = ctypes.c_void_p
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    k32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    job = k32.CreateJobObjectW(None, None)
    info = _ExtendedLimit()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not job or not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        raise OSError(f"could not create kill-on-close job: {ctypes.get_last_error()}")
    return job, k32


def test_native_transient_task_survives_kill_on_close_job(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_GATEWAY_DISABLE_JOB_HANDOFF", raising=False)
    if shutil.which("schtasks") is None:
        pytest.skip("schtasks.exe unavailable")

    handoff = tmp_path / "handoff.txt"
    survived = tmp_path / "survived.txt"
    error = tmp_path / "error.txt"
    child_script = tmp_path / "child.py"
    child_script.write_text(
        "\n".join([
            "import os, sys, time",
            f"sys.path.insert(0, {str(REPO_ROOT)!r})",
            "from hermes_cli.gateway_windows import _spawn_gateway_via_transient_task",
            f"marker = {str(survived)!r}",
            f"handoff = {str(handoff)!r}",
            f"error = {str(error)!r}",
            "cmd = [sys.executable, '-c', \"import time; time.sleep(1.5); open(%r, 'w').write('survived')\" % marker]",
            "try:",
            "    _spawn_gateway_via_transient_task(0, cmd)",
            "except BaseException as exc:",
            "    open(error, 'w').write(repr(exc))",
            "    raise",
            "open(handoff, 'w').write('accepted')",
            "time.sleep(30)",
        ]),
        encoding="utf-8",
    )

    job, k32 = _make_kill_on_close_job()
    env = {k: v for k, v in os.environ.items() if k != "HERMES_GATEWAY_DISABLE_JOB_HANDOFF"}
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    child = subprocess.Popen([sys.executable, str(child_script)], env=env)
    try:
        if not k32.AssignProcessToJobObject(ctypes.c_void_p(job), ctypes.c_void_p(int(child._handle))):
            pytest.skip(f"could not assign child to job: {ctypes.get_last_error()}")

        deadline = time.time() + 30
        while time.time() < deadline and not handoff.exists() and not error.exists():
            time.sleep(0.2)
        if error.exists():
            detail = error.read_text(encoding="utf-8", errors="replace")
            if "denied" in detail.lower() or "access" in detail.lower():
                pytest.skip(f"Task Scheduler handoff not permitted here: {detail}")
            pytest.fail(f"transient task handoff failed: {detail}")
        assert handoff.exists(), "child never reached the transient task handoff"

        # Tear down the updater's job while the task-started process is alive.
        k32.CloseHandle(ctypes.c_void_p(job))
        job = None

        deadline = time.time() + 15
        while time.time() < deadline and not survived.exists():
            time.sleep(0.2)
        assert survived.exists(), "task-started process died with the caller's Job Object"
        assert survived.read_text(encoding="utf-8").strip() == "survived"
    finally:
        if job is not None:
            k32.CloseHandle(ctypes.c_void_p(job))
        if child.poll() is None:
            child.kill()
