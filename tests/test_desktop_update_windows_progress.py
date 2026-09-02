"""The Windows hand-off keeps serving progress while its main thread blocks.

windows.ps1 answers /progress from a dedicated runspace precisely so the
window keeps moving through the long silent stretches (`hermes update`, pip,
the desktop rebuild) that made an 18-minute update look hung. This drives the
real script and polls the real listener; the posix half of the same contract
is covered in test_desktop_update_shim_progress.py.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import ctypes
from ctypes import wintypes
from pathlib import Path
from urllib.request import urlopen

import pytest

pytestmark = pytest.mark.windows_only

REPO_ROOT = Path(__file__).resolve().parent.parent
WINDOWS_UPDATE_PS1 = REPO_ROOT / "scripts" / "desktop-update" / "windows.ps1"


def _windows_child_env(tmp_path: Path) -> dict[str, str]:
    """Preserve the Windows location variables the hermetic runner omits."""
    env = os.environ.copy()
    env["TEMP"] = str(tmp_path)
    env["TMP"] = str(tmp_path)
    if "SystemDrive" not in env and (system_root := env.get("SYSTEMROOT")):
        env["SystemDrive"] = Path(system_root).drive
    return env


def _user32() -> ctypes.WinDLL:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(_RECT)]
    user32.GetClientRect.restype = wintypes.BOOL
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetClassNameW.restype = ctypes.c_int
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.SendMessageTimeoutW.argtypes = [
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
        wintypes.UINT,
        wintypes.UINT,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    user32.SendMessageTimeoutW.restype = ctypes.c_size_t
    return user32


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


def _windows_for_pid(pid: int) -> list[int]:
    """Return the visible top-level windows owned by exactly ``pid``."""
    user32 = _user32()
    windows: list[int] = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL

    @callback_type
    def visit(hwnd: int, _lparam: int) -> bool:
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            windows.append(hwnd)
        return True

    assert user32.EnumWindows(visit, 0), ctypes.get_last_error()
    return windows


def _window_text(user32: ctypes.WinDLL, hwnd: int) -> str:
    text_length = user32.GetWindowTextLengthW(hwnd)
    text = ctypes.create_unicode_buffer(text_length + 1)
    user32.GetWindowTextW(hwnd, text, len(text))
    return text.value


def _child_controls(hwnd: int) -> list[tuple[str, str, bool]]:
    """Read the real WinForms child-control tree, not script text."""
    user32 = _user32()
    controls: list[tuple[str, str, bool]] = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumChildWindows.argtypes = [wintypes.HWND, callback_type, wintypes.LPARAM]
    user32.EnumChildWindows.restype = wintypes.BOOL

    @callback_type
    def visit(child: int, _lparam: int) -> bool:
        class_name = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(child, class_name, len(class_name))
        text_length = user32.GetWindowTextLengthW(child)
        text = ctypes.create_unicode_buffer(text_length + 1)
        user32.GetWindowTextW(child, text, len(text))
        controls.append((class_name.value, text.value, bool(user32.IsWindowVisible(child))))
        return True

    assert user32.EnumChildWindows(hwnd, visit, 0), ctypes.get_last_error()
    return controls


def _kernel32() -> ctypes.WinDLL:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def _open_self_test_process(
    kernel32: ctypes.WinDLL,
    pid: int,
    expected_creation_filetime: int,
) -> wintypes.HANDLE:
    """Open the published child only when its kernel creation token matches."""
    synchronize_query_terminate = 0x00100000 | 0x1000 | 0x0001
    process = kernel32.OpenProcess(synchronize_query_terminate, False, pid)
    if not process:
        raise AssertionError(
            f"could not open self-test PID {pid}: {ctypes.get_last_error()}"
        )
    try:
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel_time = wintypes.FILETIME()
        user_time = wintypes.FILETIME()
        if not kernel32.GetProcessTimes(
            process,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            raise AssertionError(ctypes.get_last_error())
        actual_creation_filetime = (
            int(creation.dwHighDateTime) << 32
        ) | int(creation.dwLowDateTime)
        if actual_creation_filetime != expected_creation_filetime:
            raise AssertionError(
                f"self-test PID {pid} was reused before its process handle was opened"
            )
        return process
    except Exception:
        kernel32.CloseHandle(process)
        raise


def _wait_for_process_exit(kernel32: ctypes.WinDLL, process: wintypes.HANDLE, timeout_ms: int) -> int | None:
    """Return the exit code from the already-opened child process handle."""
    if kernel32.WaitForSingleObject(process, timeout_ms) != 0:
        return None
    exit_code = wintypes.DWORD()
    if not kernel32.GetExitCodeProcess(process, ctypes.byref(exit_code)):
        return None
    return int(exit_code.value)


def _read_progress(url: str, deadline: float) -> dict[str, object]:
    """Poll /progress, retrying transient socket stalls until ``deadline``.

    A single slow answer from the PS runspace listener is NOT the bug this
    test guards (the listener can lose the CPU for seconds on a loaded CI
    runner while it still serves fine a moment later). One raw
    ``urlopen(timeout=5)`` propagating TimeoutError was exactly the Aug 2026
    flake (run 32440286339). Only a listener that stays unresponsive until
    the deadline fails the test.

    Per-attempt timeout is 1s, not 5s: a connection the kernel accepted into
    the backlog before the runspace was serving never gets answered, and a 5s
    wait on it burned half the readiness budget per attempt (two stale
    attempts = red, run 33591547099). The script's own readiness handshake
    now keeps that gap from reaching us, but the probe should not be able to
    lose the whole budget to one dead socket either way.
    """
    last_exc: Exception | None = None
    attempted = False
    while not attempted or time.monotonic() < deadline:
        attempted = True
        try:
            with urlopen(f"{url}progress", timeout=1) as response:
                return json.loads(response.read().decode("utf-8"))
        except (TimeoutError, OSError) as exc:  # transient stall — retry
            last_exc = exc
            time.sleep(0.1)
    raise AssertionError(
        f"/progress unresponsive until deadline (last error: {last_exc!r})"
    )


def test_progress_advances_while_the_orchestrator_blocks(tmp_path: Path) -> None:
    powershell = shutil.which("powershell.exe")
    assert powershell, "Windows updater tests require Windows PowerShell."

    output_path = tmp_path / "self-test-output.log"
    env = _windows_child_env(tmp_path)
    # Generous hold: the assertions below must both land INSIDE it. 4s was
    # too tight for a slow runner — the second sample slid past the hold,
    # caught the cleared terminal state, and failed '' == 'Testing quiet
    # update' (PR #90358 rerun, Aug 2026). 10s left no headroom once
    # transient /progress retries entered the budget (publish wait ≤10s +
    # stability window + retry sleeps), so: 30s, and every sampling deadline
    # below is derived from the moment the held stage lands, keeping the
    # whole window comfortably inside the hold.
    env["HERMES_SELFTEST_HOLD_SECONDS"] = "30"

    with output_path.open("wb") as output:
        process = subprocess.Popen(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(WINDOWS_UPDATE_PS1),
                "-SelfTestUi",
                "-NoUi",
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
        )

    try:
        deadline = time.monotonic() + 20
        shim_url = None
        while time.monotonic() < deadline:
            text = output_path.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"SELF-TEST: shim at (http://127\.0\.0\.1:\d+/)", text)
            if match:
                shim_url = match.group(1)
                break
            if process.poll() is not None:
                break
            time.sleep(0.1)

        assert shim_url, output_path.read_text(encoding="utf-8", errors="replace")

        # The URL prints BEFORE the orchestrator publishes its held stage —
        # sampling immediately races the publish and can catch the page's
        # boot default instead ('Hermes will open once done.' ==
        # 'Testing quiet update', PR #90358 first run). Wait for the held
        # stage to actually land, THEN start the stability window.
        held_stage = "Testing quiet update"
        publish_deadline = time.monotonic() + 10
        first = _read_progress(shim_url, publish_deadline)
        while first.get("message") != held_stage and time.monotonic() < publish_deadline:
            time.sleep(0.1)
            first = _read_progress(shim_url, publish_deadline)
        assert first["message"] == held_stage, first

        time.sleep(1.5)
        second = _read_progress(shim_url, time.monotonic() + 10)

        # The stage is whatever the orchestrator last published -- it must
        # reach the page verbatim and must not churn on its own.
        assert first["status"] == "running"
        assert first["message"]
        assert second["message"] == first["message"]
        # The main thread is asleep for the whole window above. If elapsed
        # only moved when the orchestrator published, it would be frozen here
        # -- which is what a stalled update looks like to the user.
        assert int(second["elapsed_seconds"]) > int(first["elapsed_seconds"])

        assert process.wait(timeout=60) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_minimized_production_handoff_restores_a_responsive_winforms_progress_window(
    tmp_path: Path,
) -> None:
    """The real ``cmd start /min`` topology must not hide or freeze the fallback card."""
    powershell = shutil.which("powershell.exe")
    cmd = shutil.which("cmd.exe")
    assert powershell and cmd, "Windows updater tests require cmd.exe and Windows PowerShell."

    identity_path = tmp_path / "self-test-process.json"
    env = _windows_child_env(tmp_path)
    env["HERMES_SELFTEST_HOLD_SECONDS"] = "6"
    env["HERMES_SELFTEST_IDENTITY_PATH"] = str(identity_path)

    wrapper = subprocess.Popen(
        [
            cmd,
            "/d",
            "/s",
            "/c",
            "start",
            "",
            "/min",
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(WINDOWS_UPDATE_PS1),
            "-SelfTestUi",
            "-ForceWinForms",
        ],
        env=env,
    )
    child_pid: int | None = None
    child_creation_filetime: int | None = None
    child_process: wintypes.HANDLE | None = None
    child_exited = False
    try:
        assert wrapper.wait(timeout=10) == 0

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if identity_path.exists():
                try:
                    identity = json.loads(identity_path.read_text(encoding="ascii"))
                    child_pid = int(identity["pid"])
                    child_creation_filetime = int(identity["creation_filetime"])
                    if child_pid <= 0 or child_creation_filetime <= 0:
                        raise ValueError("invalid process identity")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    child_pid = None
                    child_creation_filetime = None
                    pass
                else:
                    break
            time.sleep(0.1)
        assert child_pid is not None and child_creation_filetime is not None, (
            "self-test did not publish its PowerShell process identity"
        )
        kernel32 = _kernel32()
        child_process = _open_self_test_process(
            kernel32, child_pid, child_creation_filetime
        )

        deadline = time.monotonic() + 15
        windows: list[int] = []
        while time.monotonic() < deadline:
            user32 = _user32()
            windows = [hwnd for hwnd in _windows_for_pid(child_pid) if _window_text(user32, hwnd) == "Hermes"]
            if windows:
                break
            time.sleep(0.1)
        assert windows, f"no visible Hermes window owned by self-test PID {child_pid}"

        user32 = _user32()
        hwnd = windows[0]
        assert _window_text(user32, hwnd) == "Hermes"
        assert not user32.IsIconic(hwnd), "cmd start /min left the progress card minimized"
        client = _RECT()
        assert user32.GetClientRect(hwnd, ctypes.byref(client)), ctypes.get_last_error()
        assert client.right > client.left and client.bottom > client.top

        result = ctypes.c_size_t()
        assert user32.SendMessageTimeoutW(hwnd, 0, 0, 0, 2, 2_000, ctypes.byref(result)), (
            "Hermes progress card stopped processing window messages"
        )

        controls = _child_controls(hwnd)
        visible_text = {text for _class_name, text, visible in controls if visible and text}
        visible_classes = {class_name.lower() for class_name, _text, visible in controls if visible}
        assert "Updating Hermes" in visible_text
        assert any("Testing quiet update" in text for text in visible_text)
        assert any("progress" in class_name for class_name in visible_classes)
        exit_code = _wait_for_process_exit(kernel32, child_process, 15_000)
        if exit_code is not None:
            child_exited = True
        assert exit_code == 0
    finally:
        if child_process:
            try:
                if not child_exited:
                    kernel32.TerminateProcess(child_process, 1)
                    kernel32.WaitForSingleObject(child_process, 5_000)
            finally:
                kernel32.CloseHandle(child_process)
