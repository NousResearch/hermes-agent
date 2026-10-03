"""Native subprocess helper; never attach pytest or the live Gateway to a job."""
from __future__ import annotations

import asyncio
import ctypes
import importlib.util
import json
import os
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hermes_cli._subprocess_compat import windows_detach_flags
from plugins.platforms.whatsapp import adapter as whatsapp


def restricted_job():
    """Own process only: nested kill-on-close job with no breakaway permission."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    kernel.IsProcessInJob.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]

    class Basic(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD),
        ]

    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IO), *((name, ctypes.c_size_t) for name in (
            "ProcessMemoryLimit", "JobMemoryLimit", "PeakProcessMemoryUsed", "PeakJobMemoryUsed",
        ))]

    job = kernel.CreateJobObjectW(None, None)
    assert job, ctypes.get_last_error()
    limits = Extended()
    limits.BasicLimitInformation.LimitFlags = 0x2000
    assert kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)), ctypes.get_last_error()
    assert kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()), ctypes.get_last_error()
    return kernel, job


async def main():
    work = Path(sys.argv[1]).resolve()
    assert Path(os.environ["HERMES_HOME"]).resolve().is_relative_to(work)
    kernel, job = restricted_job()
    assert job  # keep the job alive until helper exit; do not close it under ourselves
    executable = sys._base_executable
    try:
        subprocess.run([executable, "--version"], stdin=subprocess.DEVNULL,
                       capture_output=True, creationflags=windows_detach_flags(), timeout=10, check=True)
    except PermissionError as exc:
        assert exc.winerror == 5
    else:
        raise AssertionError("The real Windows job did not reject breakaway")

    spec = importlib.util.spec_from_file_location("whatsapp_test_fixture", ROOT / "tests/gateway/test_whatsapp_connect.py")
    assert spec and spec.loader
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    managed = fixtures._make_adapter()
    marker = work / "child.json"
    bridge = work / "child.py"
    bridge.write_text(
        "import ctypes,json,os,time\n"
        "from pathlib import Path\n"
        "k=ctypes.WinDLL('kernel32'); k.GetConsoleWindow.restype=ctypes.c_void_p\n"
        "u=ctypes.WinDLL('user32'); u.IsWindowVisible.argtypes=[ctypes.c_void_p]\n"
        "receipt={'pid':os.getpid(),'public':os.getenv('PUBLIC_TEST'),"
        "'visible':bool(u.IsWindowVisible(k.GetConsoleWindow()))}\n"
        "Path(os.environ['NATIVE_TEST_MARKER']).write_text(json.dumps(receipt),encoding='utf-8')\n"
        "time.sleep(60)\n", encoding="utf-8",
    )
    managed._bridge_script = str(bridge)
    managed._session_path = work / "session"
    managed._bridge_port = 0
    # Authentication and package provisioning are outside this process-creation
    # invariant. No actual session, credentials, QR or network bridge is used.
    managed._preflight = lambda: True
    managed._ensure_bridge_deps = lambda _: True
    managed._bridge_env = lambda: {**os.environ, "PUBLIC_TEST": "native-owned-child", "NATIVE_TEST_MARKER": str(marker)}
    whatsapp.find_node_executable = lambda _: executable
    managed._attach_to_bridge = lambda _: None
    managed._wire_plugin_handlers = lambda _: None

    async def wait_for_child():
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if marker.exists():
                return True
            if managed._bridge_process.poll() is not None:
                return False
            await asyncio.sleep(0.05)
        return False

    managed._wait_for_bridge = wait_for_child
    try:
        assert await managed.connect(), "Adapter failed the real restricted-job launch"
        process = managed._bridge_process
        assert process.poll() is None
        observed = json.loads(marker.read_text(encoding="utf-8"))
        assert observed["pid"] == process.pid
        assert observed["public"] == "native-owned-child" and not observed["visible"]
        handle = kernel.OpenProcess(0x1000, False, process.pid)
        assert handle, ctypes.get_last_error()
        try:
            in_job = wintypes.BOOL()
            assert kernel.IsProcessInJob(handle, job, ctypes.byref(in_job))
            assert in_job.value
        finally:
            kernel.CloseHandle(handle)
    finally:
        if managed._bridge_process is not None:
            managed._bridge_process.terminate()
            managed._bridge_process.wait(timeout=10)
        managed._release_platform_lock()
        managed._close_bridge_log()
    assert managed._platform_lock_identity is None and managed._bridge_log_fh is None
    print(json.dumps({
        "native_adapter_fallback": True, "initial_winerror": 5,
        "child_in_job": True, "child_pid_matched": True,
        "console_visible": False, "public_environment_preserved": True,
        "lock_released": True, "log_closed": True,
    }), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
