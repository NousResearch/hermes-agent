"""Native local-only bootstrap pipes with cancellable overlapped I/O.

The pipe worker threads do not own session state. Handlers marshal authority
operations to the gateway loop; the ticket store itself is thread-safe.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import logging
import os
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)


def _native():
    if os.name != 'nt':
        raise OSError('native Windows bootstrap requires Windows')
    import _winapi
    return _winapi


def _api(dll, name, result, args):
    fn = getattr(dll, name)
    fn.restype, fn.argtypes = result, args
    return fn


def _token_sid(token) -> str:
    """String SID of a token's user (``TokenUser``)."""
    a = ctypes.WinDLL('advapi32', use_last_error=True)
    length = wintypes.DWORD()
    token_info = _api(a, 'GetTokenInformation', wintypes.BOOL,
                      [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)])
    token_info(token, 1, None, 0, ctypes.byref(length))
    buf = ctypes.create_string_buffer(length.value)
    if not token_info(token, 1, buf, length, ctypes.byref(length)):
        raise PermissionError('cannot read bootstrap peer identity')
    return _sid_text(ctypes.cast(buf, ctypes.POINTER(ctypes.c_void_p))[0])


def _sid_text(sid) -> str:
    """String form (``S-1-...``) of the SID at *sid*."""
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    a = ctypes.WinDLL('advapi32', use_last_error=True)
    text = wintypes.LPWSTR()
    convert = _api(a, 'ConvertSidToStringSidW', wintypes.BOOL,
                   [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)])
    if not convert(sid, ctypes.byref(text)):
        raise PermissionError('cannot encode bootstrap peer identity')
    try:
        return text.value
    finally:
        _api(k, 'LocalFree', ctypes.c_void_p, [ctypes.c_void_p])(ctypes.cast(text, ctypes.c_void_p))


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
        return _token_sid(token)
    finally:
        if token:
            close(token)
        close(process)


class _AclHeader(ctypes.Structure):
    _fields_ = [('revision', ctypes.c_ubyte), ('sbz1', ctypes.c_ubyte), ('size', wintypes.WORD),
                ('count', wintypes.WORD), ('sbz2', wintypes.WORD)]


def _pipe_security(handle) -> tuple[str, tuple[tuple[int, str], ...]]:
    """The pipe object's owner SID and its DACL as ``(ace_type, sid)`` pairs, read from *handle*
    (``GetSecurityInfo``; the client's GENERIC_READ carries READ_CONTROL). Unlike ``OpenProcess`` on
    the server's PID this needs no access to the server process, so it still answers when the
    server is elevated and this client is not (#140706)."""
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    a = ctypes.WinDLL('advapi32', use_last_error=True)
    get_info = _api(a, 'GetSecurityInfo', wintypes.DWORD,
                    [wintypes.HANDLE, ctypes.c_int, wintypes.DWORD] + [ctypes.POINTER(ctypes.c_void_p)] * 5)
    get_ace = _api(a, 'GetAce', wintypes.BOOL, [ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p)])
    owner, dacl, descriptor = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    # SE_KERNEL_OBJECT; OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION
    if get_info(handle, 6, 0x1 | 0x4, ctypes.byref(owner), None, ctypes.byref(dacl), None,
                ctypes.byref(descriptor)) or not owner:
        raise PermissionError('cannot read named-pipe owner')
    try:
        aces = []
        # A NULL DACL (no entries to read) grants everyone: it is reported as no ACEs at all.
        for index in range(_AclHeader.from_address(dacl.value).count if dacl else 0):
            ace = ctypes.c_void_p()
            if not get_ace(dacl, index, ctypes.byref(ace)):
                raise PermissionError('cannot read named-pipe DACL')
            kind = ctypes.c_ubyte.from_address(ace.value).value
            # ACCESS_ALLOWED_ACE / ACCESS_DENIED_ACE: 4-byte ACE_HEADER, 4-byte mask, then the SID.
            aces.append((kind, _sid_text(ace.value + 8) if kind in (0, 1) else ''))
        return _sid_text(owner.value), tuple(aces)
    finally:
        _api(k, 'LocalFree', ctypes.c_void_p, [ctypes.c_void_p])(descriptor)


def _pipe_server_session(handle) -> int | None:
    """The pipe server's logon session, from the pipe (no access to the server process needed)."""
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    session = wintypes.ULONG()
    get = _api(k, 'GetNamedPipeServerSessionId', wintypes.BOOL, [wintypes.HANDLE, ctypes.POINTER(wintypes.ULONG)])
    return session.value if get(handle, ctypes.byref(session)) else None


class _ImpersonationLeak(RuntimeError):
    """RevertToSelf failed: this thread may still carry the client's identity. Deliberately not an
    OSError, so the serving loop does not swallow it and the worker stops instead."""


def _pipe_client_sid(handle) -> str:
    """The client's SID from the identification token the kernel captured when it opened the pipe
    (the client dials with ``SECURITY_SQOS_PRESENT``). Unlike ``OpenProcess`` on the client's PID,
    this still answers after a short-lived client (a one-shot ``gateway ensure``) has exited.
    Windows only allows it once data has been read from the pipe."""
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    a = ctypes.WinDLL('advapi32', use_last_error=True)
    impersonate = _api(a, 'ImpersonateNamedPipeClient', wintypes.BOOL, [wintypes.HANDLE])
    revert = _api(a, 'RevertToSelf', wintypes.BOOL, [])
    open_thread_token = _api(a, 'OpenThreadToken', wintypes.BOOL,
                             [wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL, ctypes.POINTER(wintypes.HANDLE)])
    current_thread = _api(k, 'GetCurrentThread', wintypes.HANDLE, [])
    close = _api(k, 'CloseHandle', wintypes.BOOL, [wintypes.HANDLE])
    if not impersonate(handle):
        raise PermissionError('cannot impersonate named-pipe peer')
    token = wintypes.HANDLE()
    try:
        try:
            # OpenAsSelf: the access check runs against the gateway's own token, not the client's.
            if not open_thread_token(current_thread(), 8, True, ctypes.byref(token)):
                raise PermissionError('cannot read named-pipe peer token')
        finally:
            if not revert():
                raise _ImpersonationLeak('RevertToSelf failed after reading a named-pipe peer')
        return _token_sid(token)
    finally:
        if token:
            close(token)


def _peer_subject(handle, *, server: bool) -> str:
    """``sid:<owner>`` for a same-user peer; PermissionError for another user or an unidentified one.

    The serving side (``server=False``) reads the identity from the pipe's client token first and
    falls back to the client's process; an unreadable peer is refused, never served as anonymous.
    The client side (``server=True``) is judged by :func:`judge_pipe_server`."""
    sid = None
    if not server:
        try:
            sid = _pipe_client_sid(handle)
        except PermissionError:
            sid = None
    if sid is None:
        k = ctypes.WinDLL('kernel32', use_last_error=True)
        name = 'GetNamedPipeServerProcessId' if server else 'GetNamedPipeClientProcessId'
        get_pid = _api(k, name, wintypes.BOOL, [wintypes.HANDLE, ctypes.POINTER(wintypes.ULONG)])
        pid = wintypes.ULONG()
        if not get_pid(handle, ctypes.byref(pid)):
            raise PermissionError('cannot identify named-pipe peer')
        if server:
            return _server_subject(handle, pid.value)
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


def _accept(handle, budget) -> bool:
    """Wait up to ``budget`` seconds for a client; True once one is connected.

    A client can connect between the wait timing out and the cancel reaching the kernel; the
    reaped result then reports success and that client is served. ERROR_OPERATION_ABORTED (995)
    is the only idle outcome. A client that is already connected completes the accept at once.
    """
    win = _native()
    ov = win.ConnectNamedPipe(handle, overlapped=True)
    if win.WaitForSingleObject(ov.event, max(1, int(budget * 1000))) != win.WAIT_OBJECT_0:
        ov.cancel()
    _, error = ov.GetOverlappedResult(True)
    if error == 995:
        return False
    if error:
        raise OSError(None, 'runtime pipe accept failed', None, error)
    return True


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


_PIPE_CLIENT_SQOS = 0x00100000 | 0x00010000


def query_runtime_control(home: Path, request: bytes, timeout: float) -> bytes:
    from gateway.control_socket import windows_pipe_name
    win = _native()
    deadline = time.monotonic() + timeout
    name = windows_pipe_name(home)
    _remaining_ms(deadline)
    while True:
        try:
            # SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION: the server may read (never act as)
            # this client's token, captured now, so it can identify us even after we exit.
            handle = win.CreateFile(name, 0xC0000000, 0, win.NULL, win.OPEN_EXISTING,
                                    win.FILE_FLAG_OVERLAPPED | _PIPE_CLIENT_SQOS, win.NULL)
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


# Listening pipe instances, one worker each. Requests are served concurrently, as the POSIX
# socket does through its executor: a handler may dial this same pipe (a hosted room whose
# source and member share one multiplexer re-enters it via hosted-producer -> hosted-attest).
_PIPE_INSTANCES = 8


class NativeControlServer:
    """Bounded native workers, independent of asyncio event-loop policy."""
    def __init__(self, home, handler):
        self.home, self.handler = home, handler
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error = None
        self._threads = [threading.Thread(target=self._run, args=(index == 0,), daemon=True,
                                          name=f'runtime-control-pipe-{index}')
                         for index in range(_PIPE_INSTANCES)]
        self._thread = self._threads[0]

    def start(self):
        # The first instance claims the name (FILE_FLAG_FIRST_PIPE_INSTANCE) before any other.
        self._thread.start()
        if not self._ready.wait(5):
            self.close()
            raise TimeoutError('runtime control pipe startup deadline exceeded')
        if self._error:
            raise self._error
        for thread in self._threads[1:]:
            thread.start()

    def close(self):
        self._stop.set()
        for thread in self._threads:
            if thread.ident is not None:
                thread.join(timeout=5)
        if any(thread.is_alive() for thread in self._threads):
            raise TimeoutError('runtime control pipe did not stop')

    def _create_pipe(self, first=True):
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
                                       3 | win.FILE_FLAG_OVERLAPPED | (0x00080000 if first else 0),
                                       8, _PIPE_INSTANCES, 65536, 65536, 2000, ctypes.addressof(attrs))
        finally:
            k = ctypes.WinDLL('kernel32', use_last_error=True)
            _api(k, 'LocalFree', ctypes.c_void_p, [ctypes.c_void_p])(descriptor)

    def _run(self, first=True):
        win = _native()
        handle = None
        try:
            handle = self._create_pipe(first)
            self._ready.set()
            while not self._stop.is_set():
                try:
                    connected = _accept(handle, 0.5)
                except OSError:
                    # e.g. ERROR_NO_DATA: a client came and went. Fall through so the serve path
                    # fails fast and the instance is recycled by DisconnectNamedPipe below.
                    connected = True
                if not connected:
                    # Idle: the instance is still listening and a client may have opened it after
                    # the cancel. DisconnectNamedPipe here dropped that client mid-request (233).
                    continue
                try:
                    # Read first: ImpersonateNamedPipeClient needs data read from the pipe, and the
                    # handler runs only once the peer is identified (bounded: 64 KiB, 2 s).
                    raw = _read_line(handle, time.monotonic() + 2, 64 * 1024)
                    subject = _peer_subject(handle, server=False)
                    response = self.handler(raw, subject)
                    # The I/O budget bounds the peer, not the handler's own (client-bounded) work.
                    deadline = time.monotonic() + 2
                    _write(handle, response, deadline)
                    # DisconnectNamedPipe discards unread output. Let the client
                    # consume and close, bounded by the same I/O deadline.
                    _read_line(handle, deadline, 64 * 1024)
                except (OSError, TimeoutError, ConnectionError):
                    pass
                finally:
                    try:
                        _disconnect_pipe(handle)
                    except OSError as exc:
                        if exc.winerror != 233:  # ERROR_PIPE_NOT_CONNECTED: the client already left
                            raise
        except BaseException as exc:
            # Hand-off to the owning thread: start() re-raises the stored error.
            logger.debug('runtime control pipe stopped', exc_info=True)
            self._error = exc
            self._ready.set()
        finally:
            if handle is not None:
                win.CloseHandle(handle)


# Why a same-user gateway's control pipe or process refused this client (#139845, #140252). The
# pipe DACL names the user's SID, so a refusal between two processes of one user means a boundary
# the SID does not cross: an elevated owner (High integrity) seen from a filtered token, or an
# owner in another logon session (a session-0 S4U boot task). ``authorization`` reads as a
# credential problem and names neither.
GATEWAY_ELEVATED = "gateway_elevated"
GATEWAY_OTHER_SESSION = "gateway_other_session"


def classify_refused_owner(*, own_session: int | None, owner_session: int | None,
                           own_elevated: bool, owner_elevated: bool | None) -> str | None:
    """Pure decision over probed facts; ``None`` keeps the generic ``authorization`` code.

    ``owner_elevated`` is None when the owner's token could not be opened; from a non-elevated
    client in the same session that refusal is itself the elevation boundary."""
    if own_session is not None and owner_session is not None and own_session != owner_session:
        return GATEWAY_OTHER_SESSION
    if own_elevated:
        return None
    if owner_elevated or (owner_elevated is None and owner_session is not None):
        return GATEWAY_ELEVATED
    return None


def refused_owner_remedy(reason: str, pid: int, owner_session: int | None) -> str:
    if reason == GATEWAY_ELEVATED:
        return (f"The running gateway (PID {pid}) was started as administrator, so this non-elevated "
                "app cannot reach it. Stop it from an administrator prompt (hermes gateway stop), then "
                "start Hermes without elevation")
    where = f"session {owner_session}" if owner_session is not None else "another session"
    return (f"The running gateway (PID {pid}) runs in another Windows logon session ({where}, e.g. a "
            "boot-time scheduled task), so this app cannot reach it. Stop it (hermes gateway stop from an "
            "administrator prompt) and let only one auto-start task launch the gateway")


_ADMINISTRATORS_SID = 'S-1-5-32-544'  # BUILTIN\Administrators: default owner of what an elevated admin creates
_ACCESS_ALLOWED_ACE = 0


class RefusedOwner(PermissionError):
    """This user's gateway, behind a boundary this client must not cross. ``str()`` is the remedy
    (the refused-owner wording above); ``reason_code`` is the bounded code discovery reports."""

    def __init__(self, reason_code: str, pid: int, remedy: str):
        super().__init__(remedy)
        self.reason_code, self.pid = reason_code, pid


def judge_pipe_server(*, own_sid: str, process_sid: str | None, pipe: tuple[str, tuple] | None, pid: int,
                      own_elevated: bool, own_session: int | None, server_session: int | None) -> str:
    """``sid:<own>`` when this client may talk to the pipe's server; raises otherwise. Pure: the
    Win32 facts are passed in (#140706).

    *process_sid* is the server process's token user, None when this client cannot open the
    process; *pipe* is the pipe object's ``(owner, dacl)`` (:func:`_pipe_security`), None when
    unreadable.

    * Server process readable: its token user must be this user. A pipe owned by
      BUILTIN\\Administrators was made by an elevated process, and a non-elevated client never
      hands its requests to one (its tools would run with the full administrator token).
    * Server process unreadable (an elevated server seen from a filtered token): never attached.
      The pipe object says who made it: owner this user or BUILTIN\\Administrators, and a DACL
      granting exactly this user, the gateway's own private shape (``_create_pipe``). Another
      non-administrator account can set neither owner, so that is this user's elevated gateway
      (or an administrator, who already controls this account) and the refusal names it. Any
      other pipe stays the generic refusal."""
    owner, aces = pipe if pipe is not None else (None, ())
    if process_sid is not None:
        if process_sid != own_sid:
            raise PermissionError('named-pipe peer belongs to another user')
        if owner != _ADMINISTRATORS_SID or own_elevated:
            return 'sid:' + own_sid
        owner_elevated = True
    else:
        if pipe is None:
            raise PermissionError('cannot verify bootstrap peer process')
        if owner not in (own_sid, _ADMINISTRATORS_SID) or tuple(aces) != ((_ACCESS_ALLOWED_ACE, own_sid),):
            raise PermissionError('named-pipe peer belongs to another user')
        owner_elevated = True if owner == _ADMINISTRATORS_SID else None
    reason = classify_refused_owner(own_session=own_session, owner_session=server_session,
                                    own_elevated=own_elevated, owner_elevated=owner_elevated)
    if reason is None:
        raise PermissionError('cannot verify bootstrap peer process')
    logger.info('control pipe server PID %s refused as %s: pipe owner %s, server process %s, '
                'server session %s, own session %s', pid, reason, owner,
                'readable' if process_sid else 'unreadable', server_session, own_session)
    raise RefusedOwner(reason, pid, refused_owner_remedy(reason, pid, server_session))


def _server_subject(handle, pid: int) -> str:
    """The client half of the pipe identity check: the Win32 facts for :func:`judge_pipe_server`."""
    from hermes_dir_acl import process_is_elevated
    try:
        process_sid = _process_sid(pid)
    except PermissionError:
        process_sid = None
    try:
        pipe = _pipe_security(handle)
    except PermissionError:
        pipe = None
    return judge_pipe_server(own_sid=_process_sid(os.getpid()), process_sid=process_sid, pipe=pipe, pid=pid,
                             own_elevated=process_is_elevated(), own_session=_session_of(os.getpid()),
                             server_session=_pipe_server_session(handle))


def _session_of(pid: int) -> int | None:
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    session = wintypes.DWORD()
    to_session = _api(k, 'ProcessIdToSessionId', wintypes.BOOL, [wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)])
    return session.value if to_session(pid, ctypes.byref(session)) else None


def probe_refused_owner(pid: int) -> tuple[str, str] | None:
    """``(reason_code, remedy)`` naming why *pid*'s gateway refused this client, or None."""
    from hermes_dir_acl import process_elevation, process_is_elevated
    own_session, owner_session = _session_of(os.getpid()), _session_of(pid)
    reason = classify_refused_owner(own_session=own_session, owner_session=owner_session,
                                    own_elevated=process_is_elevated(), owner_elevated=process_elevation(pid))
    return (reason, refused_owner_remedy(reason, pid, owner_session)) if reason else None
