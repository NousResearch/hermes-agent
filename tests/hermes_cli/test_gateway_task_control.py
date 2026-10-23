"""Bounded Task Scheduler operations and UAC result/handle propagation (#63743)."""

import ctypes
import subprocess
from types import SimpleNamespace
import uuid

import pytest

from hermes_cli import gateway_windows_task_control as native
from hermes_cli import update_windows_tasks as tasks


@pytest.mark.platforms("windows")
@pytest.mark.parametrize(
    "waits,expected,terminated",
    [([0], 7, False), ([258, 0], 124, True), ([0xFFFFFFFF, 0], 124, True)],
)
def test_elevated_operation_waits_returns_exit_and_closes_its_handle(
    monkeypatch, waits, expected, terminated
):
    calls = []

    class Function:
        def __init__(self, call):
            self.call = call

        def __call__(self, *args):
            return self.call(*args)

    def execute(pointer):
        pointer._obj.hProcess = 123
        return True

    def exit_code(_handle, pointer):
        pointer._obj.value = 7
        return True

    outcomes = iter(waits)
    kernel = SimpleNamespace(
        WaitForSingleObject=Function(lambda *_: next(outcomes)),
        TerminateProcess=Function(lambda *_: calls.append("terminate") or True),
        GetExitCodeProcess=Function(exit_code),
        CloseHandle=Function(lambda handle: calls.append(("close", handle)) or True),
    )
    monkeypatch.setattr(
        ctypes,
        "windll",
        SimpleNamespace(
            shell32=SimpleNamespace(ShellExecuteExW=Function(execute)), kernel32=kernel
        ),
    )
    code, _detail = native._shell_execute_elevated_and_wait(
        "fixture.exe", "", cwd=".", timeout_s=2
    )
    assert code == expected
    assert ("terminate" in calls) is terminated
    assert calls.count(("close", 123)) == 1


@pytest.mark.platforms("windows")
@pytest.mark.parametrize(
    "initial,elevated", [(0, None), (5, 0), (5, 1223), (13, None), (124, None)]
)
def test_task_operation_elevates_only_denial_and_propagates_failure(
    monkeypatch, tmp_path, initial, elevated
):
    from hermes_platform import resolver
    from hermes_platform.resolver import Candidate, Resolution

    class Scratch:
        def __enter__(self):
            return str(tmp_path)

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(tasks.tempfile, "TemporaryDirectory", lambda **_: Scratch())
    monkeypatch.setattr(
        resolver,
        "locate_command",
        lambda _: Resolution(
            "path_executable", (Candidate("fixture.exe", "test", True),)
        ),
    )
    monkeypatch.setattr(
        tasks.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=initial)
    )
    (tmp_path / "task.xml").write_text("<Task/>", encoding="utf-8")
    elevated_calls = []

    def elevate(*args, **kwargs):
        elevated_calls.append((args, kwargs))
        return elevated, "fixture result"

    monkeypatch.setattr(native, "_shell_execute_elevated_and_wait", elevate)
    if initial == 0 or (initial == 5 and elevated == 0):
        assert tasks.task_operation("Hermes_Gateway_ops", "stop", "a" * 64) == "<Task/>"
    else:
        with pytest.raises(RuntimeError, match="failed"):
            tasks.task_operation("Hermes_Gateway_ops", "stop", "a" * 64)
    assert len(elevated_calls) == (1 if initial == 5 else 0)


@pytest.mark.platforms("windows")
def test_task_timeout_remains_uncertain_and_does_not_retry_or_elevate(monkeypatch):
    attempts = []

    def timeout(command, **kwargs):
        attempts.append(command)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    def no_prompt(*_a, **_k):
        pytest.fail("A timed-out task operation attempted UAC")

    monkeypatch.setattr(tasks.subprocess, "run", timeout)
    monkeypatch.setattr(native, "_shell_execute_elevated_and_wait", no_prompt)
    with pytest.raises(tasks.TaskOperationError, match="timed out") as failure:
        tasks.task_operation("Hermes_Gateway_timeout", "stop", "a" * 64)
    assert not failure.value.not_started
    assert len(attempts) == 1


@pytest.mark.platforms("windows")
def test_real_task_query_handles_an_absent_task_without_elevation(monkeypatch):
    def no_prompt(*_a, **_k):
        pytest.fail("An absent task query attempted UAC")

    monkeypatch.setattr(native, "_shell_execute_elevated_and_wait", no_prompt)
    assert (
        tasks.task_operation("Hermes_Gateway_test_absent_" + uuid.uuid4().hex, "query")
        is None
    )
