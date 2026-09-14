"""Native local-only bootstrap pipes with cancellable overlapped I/O.

The single pipe thread does not own session state. Handlers marshal authority
operations to the gateway loop; the ticket store itself is thread-safe.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import threading
import time


def _native():
    if os.name != 'nt':
        raise OSError('native Windows bootstrap requires Windows')
    import _winapi
    return _winapi


def _api(dll, name, result, args):
    fn = getattr(dll, name)
    fn.restype, fn.argtypes = result, args
    return fn


def _process_sid(pid: int) -> str:
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    a = ctypes.WinDLL('advapi32', use_last_error=True)
    open_process = _api(k, 'OpenProcess', wintypes.HANDLE, [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD])
    close = _api(k, 'CloseHandle', wintypes.BOOL, [wintypes.HANDLE])
    process = open_process(0x1000, False, pid)
    if not process:
        raise PermissionError('cannot verify bootstrap peer process')
    token = wintypes.HANDLE()
    try:
        open_token = _api(a, 'OpenProcessToken', wintypes.BOOL,
                          [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)])
        if not open_token(process, 8, ctypes.byref(token)):
            raise PermissionError('cannot verify bootstrap peer token')
        length = wintypes.DWORD()
        token_info = _api(a, 'GetTokenInformation', wintypes.BOOL,
                          [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)])
        token_info(token, 1, None, 0, ctypes.byref(length))
        buf = ctypes.create_string_buffer(length.value)
        if not token_info(token, 1, buf, length, ctypes.byref(length)):
            raise PermissionError('cannot read bootstrap peer identity')
        sid = ctypes.cast(buf, ctypes.POINTER(ctypes.c_void_p))[0]
        text = wintypes.LPWSTR()
        convert = _api(a, 'ConvertSidToStringSidW', wintypes.BOOL,
                       [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)])
        if not convert(sid, ctypes.byref(text)):
            raise PermissionError('cannot encode bootstrap peer identity')
        try:
            return text.value
        finally:
            _api(k, 'LocalFree', ctypes.c_void_p, [ctypes.c_void_p])(ctypes.cast(text, ctypes.c_void_p))
    finally:
        if token:
            close(token)
        close(process)


def _peer_subject(handle, *, server: bool) -> str:
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    name = 'GetNamedPipeServerProcessId' if server else 'GetNamedPipeClientProcessId'
    get_pid = _api(k, name, wintypes.BOOL, [wintypes.HANDLE, ctypes.POINTER(wintypes.ULONG)])
    pid = wintypes.ULONG()
    if not get_pid(handle, ctypes.byref(pid)):
        raise PermissionError('cannot identify named-pipe peer')
    sid = _process_sid(pid.value)
    if sid != _process_sid(os.getpid()):
        raise PermissionError('named-pipe peer belongs to another user')
    return 'sid:' + sid


def _disconnect_pipe(handle) -> None:
    """``DisconnectNamedPipe`` via kernel32: CPython's ``_winapi`` does not export it, and the
    server thread must outlive its first client (a dead pipe leaves the gateway stuck in
    ``starting`` for every later ``identify``)."""
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    disconnect = _api(k, 'DisconnectNamedPipe', wintypes.BOOL, [wintypes.HANDLE])
    if not disconnect(handle):
        error = ctypes.get_last_error()
        raise OSError(None, 'DisconnectNamedPipe failed', None, error)


def _remaining_ms(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('runtime control deadline exceeded')
    return max(1, int(remaining * 1000))


def _complete(ov, deadline):
    win = _native()
    try:
        if win.WaitForSingleObject(ov.event, _remaining_ms(deadline)) != win.WAIT_OBJECT_0:
            raise TimeoutError('runtime control deadline exceeded')
        transferred, error = ov.GetOverlappedResult(False)
        if error:
            raise OSError(error, 'runtime pipe I/O failed')
        return transferred
    except BaseException:
        # Cancel the native operation, not merely its caller's wait. Completion is
        # reaped before the buffer/OVERLAPPED can be released.
        ov.cancel()
        ov.GetOverlappedResult(True)
        raise


def _read_line(handle, deadline, maximum):
    win = _native()
    chunks = bytearray()
    while b'\n' not in chunks:
        ov, _ = win.ReadFile(handle, min(4096, maximum + 1 - len(chunks)), overlapped=True)
        count = _complete(ov, deadline)
        if not count:
            raise ConnectionError('runtime pipe closed before response')
        chunks.extend(ov.getbuffer()[:count])
        if len(chunks) > maximum:
            raise PermissionError('runtime control message too large')
    return bytes(chunks).split(b'\n', 1)[0]


def _write(handle, data, deadline):
    win = _native()
    while data:
        ov, _ = win.WriteFile(handle, data, overlapped=True)
        sent = _complete(ov, deadline)
        if not sent:
            raise ConnectionError('runtime pipe closed during write')
        data = data[sent:]


def query_runtime_control(home: Path, request: bytes, timeout: float) -> bytes:
    from gateway.control_socket import windows_pipe_name
    win = _native()
    deadline = time.monotonic() + timeout
    name = windows_pipe_name(home)
    _remaining_ms(deadline)
    while True:
        try:
            handle = win.CreateFile(name, 0xC0000000, 0, win.NULL, win.OPEN_EXISTING,
                                    win.FILE_FLAG_OVERLAPPED | 0x00100000, win.NULL)
            break
        except OSError as exc:
            if exc.winerror == 2:
                raise FileNotFoundError('runtime control pipe absent') from None
            if exc.winerror == 5:
                raise PermissionError('runtime control pipe inaccessible') from None
            if exc.winerror != 231:
                raise
            try:
                win.WaitNamedPipe(name, _remaining_ms(deadline))
            except OSError as wait_error:
                if wait_error.winerror == 121:
                    raise TimeoutError('runtime control deadline exceeded') from None
                raise
    try:
        _peer_subject(handle, server=True)
        _write(handle, request.rstrip(b'\n') + b'\n', deadline)
        return _read_line(handle, deadline, 512 * 1024) + b'\n'
    finally:
        win.CloseHandle(handle)


class NativeControlServer:
    """One bounded native worker, independent of asyncio event-loop policy."""
    def __init__(self, home, handler):
        self.home, self.handler = home, handler
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._run, name='runtime-control-pipe', daemon=True)

    def start(self):
        self._thread.start()
        if not self._ready.wait(5):
            self.close()
            raise TimeoutError('runtime control pipe startup deadline exceeded')
        if self._error:
            raise self._error

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise TimeoutError('runtime control pipe did not stop')

    def _create_pipe(self):
        from gateway.control_socket import windows_pipe_name
        win = _native()
        class SecurityAttributes(ctypes.Structure):
            _fields_ = [('length', wintypes.DWORD), ('descriptor', ctypes.c_void_p), ('inherit', wintypes.BOOL)]
        a = ctypes.WinDLL('advapi32', use_last_error=True)
        descriptor = ctypes.c_void_p()
        convert = _api(a, 'ConvertStringSecurityDescriptorToSecurityDescriptorW', wintypes.BOOL,
                       [wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p])
        # Protected DACL: exactly the current user, no inherited Everyone/Users ACEs.
        if not convert('D:P(A;;GA;;;' + _process_sid(os.getpid()) + ')', 1, ctypes.byref(descriptor), None):
            raise PermissionError('cannot create private pipe DACL')
        attrs = SecurityAttributes(ctypes.sizeof(SecurityAttributes), descriptor, False)
        try:
            return win.CreateNamedPipe(windows_pipe_name(self.home),
                                       3 | win.FILE_FLAG_OVERLAPPED | 0x00080000,
                                       8, 1, 65536, 65536, 2000, ctypes.addressof(attrs))
        finally:
            k = ctypes.WinDLL('kernel32', use_last_error=True)
            _api(k, 'LocalFree', ctypes.c_void_p, [ctypes.c_void_p])(descriptor)

    def _run(self):
        win = _native()
        handle = None
        try:
            handle = self._create_pipe()
            self._ready.set()
            while not self._stop.is_set():
                try:
                    ov = win.ConnectNamedPipe(handle, overlapped=True)
                    _complete(ov, time.monotonic() + 0.5)
                    subject = _peer_subject(handle, server=False)
                    deadline = time.monotonic() + 2
                    raw = _read_line(handle, deadline, 64 * 1024)
                    _write(handle, self.handler(raw, subject), deadline)
                    # DisconnectNamedPipe discards unread output. Let the client
                    # consume and close, bounded by the same I/O deadline.
                    _read_line(handle, deadline, 64 * 1024)
                except (OSError, TimeoutError, ConnectionError):
                    pass
                finally:
                    try:
                        _disconnect_pipe(handle)
                    except OSError as exc:
                        if exc.winerror != 233:  # ERROR_PIPE_NOT_CONNECTED after a cancelled accept
                            raise
        except BaseException as exc:
            self._error = exc
            self._ready.set()
        finally:
            if handle is not None:
                win.CloseHandle(handle)
