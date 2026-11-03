"""Wait for UAC task operations. Preserves the synchronous control from PR #69743."""

import ctypes
from importlib import import_module


def process_is_system(pid: int) -> bool:
    """Compare the token SID, not a localized account name or a domain user's name."""
    win32api = import_module("win32api")
    import win32con

    win32security = import_module("win32security")

    process = win32api.OpenProcess(
        win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid
    )
    try:
        token = win32security.OpenProcessToken(process, win32con.TOKEN_QUERY)
        try:
            sid = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
            return win32security.ConvertSidToStringSid(sid) == "S-1-5-18"
        finally:
            token.Close()
    finally:
        process.Close()


def _shell_execute_elevated_and_wait(
    executable: str,
    parameters: str,
    *,
    cwd: str,
    timeout_s: float,
) -> tuple[int, str]:
    """Run one executable through UAC, wait, and return its exit code.

    ``ShellExecuteW`` only reports whether the handoff started. Update-time
    Scheduled Task control needs a stronger contract: the parent must know
    that the task operation finished before it mutates the checkout or venv. Use
    ``ShellExecuteExW`` with a process handle so cancellation, timeout, and a
    nonzero child exit all remain visible to the caller.
    """
    from hermes_cli.gateway_windows import _assert_windows

    _assert_windows()
    from ctypes import wintypes

    class _ShellExecuteInfoW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("fMask", wintypes.ULONG),
            ("hwnd", wintypes.HWND),
            ("lpVerb", wintypes.LPCWSTR),
            ("lpFile", wintypes.LPCWSTR),
            ("lpParameters", wintypes.LPCWSTR),
            ("lpDirectory", wintypes.LPCWSTR),
            ("nShow", ctypes.c_int),
            ("hInstApp", wintypes.HINSTANCE),
            ("lpIDList", wintypes.LPVOID),
            ("lpClass", wintypes.LPCWSTR),
            ("hkeyClass", wintypes.HKEY),
            ("dwHotKey", wintypes.DWORD),
            ("hIconOrMonitor", wintypes.HANDLE),
            ("hProcess", wintypes.HANDLE),
        ]

    shell32 = ctypes.windll.shell32
    kernel32 = ctypes.windll.kernel32
    execute = shell32.ShellExecuteExW
    execute.argtypes = [ctypes.POINTER(_ShellExecuteInfoW)]
    execute.restype = wintypes.BOOL
    wait_for_process = kernel32.WaitForSingleObject
    wait_for_process.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    wait_for_process.restype = wintypes.DWORD
    terminate_process = kernel32.TerminateProcess
    terminate_process.argtypes = [wintypes.HANDLE, wintypes.UINT]
    terminate_process.restype = wintypes.BOOL
    get_exit_code = kernel32.GetExitCodeProcess
    get_exit_code.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    get_exit_code.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL

    info = _ShellExecuteInfoW()
    info.cbSize = ctypes.sizeof(info)
    info.fMask = 0x00000040  # SEE_MASK_NOCLOSEPROCESS
    info.lpVerb = "runas"
    info.lpFile = executable
    info.lpParameters = parameters
    info.lpDirectory = cwd
    info.nShow = 0  # SW_HIDE

    if not execute(ctypes.byref(info)):
        error_code = int(kernel32.GetLastError())
        if error_code == 1223:  # ERROR_CANCELLED
            return (1223, "UAC prompt was cancelled")
        return (error_code or 1, f"ShellExecuteExW failed with error {error_code}")
    if not info.hProcess:
        return (1, "ShellExecuteExW returned no process handle")

    try:
        timeout_ms = max(1, int(max(timeout_s, 0.001) * 1000))
        wait_result = int(wait_for_process(info.hProcess, timeout_ms))
        if wait_result != 0x00000000:  # WAIT_OBJECT_0
            if not terminate_process(info.hProcess, 124):
                raise RuntimeError(
                    "Could not stop the uncertain elevated task operation"
                )
            if wait_for_process(info.hProcess, 1000) != 0:
                raise RuntimeError("Elevated task operation exit could not be verified")
            return (
                124,
                f"elevated process did not complete (wait result {wait_result})",
            )

        exit_code = wintypes.DWORD()
        if not get_exit_code(info.hProcess, ctypes.byref(exit_code)):
            error_code = int(kernel32.GetLastError())
            return (
                error_code or 1,
                f"GetExitCodeProcess failed with error {error_code}",
            )
        return (int(exit_code.value), "")
    finally:
        if not close_handle(info.hProcess):
            raise ctypes.WinError()
