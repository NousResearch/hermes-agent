"""Desktop-application resolver over an `AppDef` (parsed from an MCP manifest's `app:` block).

`locate` stats the executable or bundle. `inspect` reads the version source in-process.
`probe` re-reads the vendor's runtime file on every call; the bearer token in it never
leaves this module.
"""

from __future__ import annotations

import json
import os
import plistlib
import sys
import time
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

from hermes_platform.resolver.base import Effort, Inspection, Probe
from hermes_platform.resolver.core import (
    Candidate,
    CheckState,
    LookupContext,
    Observation,
    Resolution,
)

PresenceKind = Literal["executable", "bundle"]
VersionKind = Literal["pe_resource", "plist", "uninstall_registry", "none"]
LivenessKind = Literal["server_json", "none"]

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


@dataclass(frozen=True)
class AppDef:
    """One application on one OS. Paths use `%VAR%` and `~`; expansion happens at lookup."""

    app_id: str
    os_family: str
    presence: PresenceKind
    location: str
    version_kind: VersionKind = "none"
    version_arg: str = ""
    liveness_kind: LivenessKind = "none"
    liveness_path: str = ""
    liveness_pid_key: str = "pid"
    liveness_url_key: str = "http"
    liveness_token_key: str = "token"
    endpoint_path: str = "/mcp"


def _expand(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))


@dataclass(frozen=True)
class Endpoint:
    url: str
    token: str = ""

    def __repr__(self) -> str:
        return f"Endpoint(url={self.url!r}, token=<redacted>)"


@dataclass(frozen=True)
class AppResolver:
    definition: AppDef

    @property
    def name(self) -> str:
        return self.definition.app_id

    # ---- locate: stat only -------------------------------------------------------------

    def locate(self, ctx: LookupContext | None = None) -> Resolution:
        d = self.definition
        target = _expand(d.location)
        if not os.path.isabs(target) or "%" in target or "$" in target:
            return Resolution("missing", (Candidate(target, f"app:{d.app_id}", False),))
        if d.presence == "bundle":
            present = os.path.isdir(target) and os.path.isfile(os.path.join(target, "Contents", "Info.plist"))
        else:
            present = os.path.isfile(target)
        cand = Candidate(target, f"app:{d.app_id}", present)
        return Resolution("known_path" if present else "missing", (cand,))

    # ---- inspect: bounded file reads and in-process OS APIs -----------------------------

    def inspect(self, res: Resolution, ctx: LookupContext | None = None) -> Inspection:
        if not res.found:
            return Inspection(Observation.not_checked(), Observation.not_checked())
        return Inspection(version=self._version(res.command[0]), signer=Observation.not_checked())

    def _version(self, path: str) -> Observation[str]:
        kind = self.definition.version_kind
        try:
            if kind == "none":
                return Observation.not_checked()
            if kind == "plist":
                return _plist_version(path)
            if kind == "pe_resource":
                return _pe_version(path)
            if kind == "uninstall_registry":
                return _uninstall_registry_version(self.definition.version_arg)
        except Exception as exc:  # a vendor's plist/PE/registry entry is untrusted input; never abort the caller
            return Observation(CheckState.ERROR, detail=exc.__class__.__name__)
        return Observation(CheckState.UNAVAILABLE, detail=f"unknown version kind {kind}")

    # ---- probe: fresh, never cached ------------------------------------------------------

    def endpoint(self) -> Endpoint | None:
        """Read and validate the current runtime endpoint."""
        d = self.definition
        if d.liveness_kind != "server_json":
            return None
        session = _read_server_json(_expand(d.liveness_path), d)
        if session is None or _pid_alive(session.pid).value is not True:
            return None
        endpoint = _endpoint_observation(session.url, d.endpoint_path)
        if endpoint.state is not CheckState.PRESENT or not endpoint.value:
            return None
        return Endpoint(endpoint.value, session.token)

    def probe(self, res: Resolution, *, effort: Effort, deadline_s: float = 3.0) -> Probe:
        d = self.definition
        nc: Observation = Observation.not_checked()
        if d.liveness_kind != "server_json":
            return Probe(running=nc, answering=nc, endpoint=nc)
        session = _read_server_json(_expand(d.liveness_path), d)
        if session is None:
            absent = Observation(CheckState.ABSENT, False, "runtime file missing or unreadable")
            return Probe(running=absent, answering=nc, endpoint=Observation(CheckState.ABSENT))
        running = _pid_alive(session.pid)
        endpoint_obs = _endpoint_observation(session.url, d.endpoint_path)
        if effort is Effort.LOCAL or running.value is not True or endpoint_obs.state is not CheckState.PRESENT:
            return Probe(running=running, answering=nc, endpoint=endpoint_obs)
        answering = _mcp_initialize(session, endpoint_obs.value or "", deadline_s)
        return Probe(running=running, answering=answering, endpoint=endpoint_obs)


# ---- version sources -------------------------------------------------------------------


def _plist_version(bundle: str) -> Observation[str]:
    with open(os.path.join(bundle, "Contents", "Info.plist"), "rb") as fh:  # windows-footgun: ok — binary mode
        info = plistlib.load(fh)
    value = info.get("CFBundleShortVersionString") or info.get("CFBundleVersion")
    if not value:
        return Observation(CheckState.UNAVAILABLE, detail="no version key in Info.plist")
    return Observation(CheckState.PRESENT, str(value))


def _pe_version(path: str) -> Observation[str]:
    if sys.platform != "win32":
        return Observation(CheckState.UNAVAILABLE, detail="pe_resource needs Windows")
    import ctypes
    from ctypes import wintypes

    ver = ctypes.windll.version  # type: ignore[attr-defined]
    ver.GetFileVersionInfoSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    ver.GetFileVersionInfoSizeW.restype = wintypes.DWORD
    size = ver.GetFileVersionInfoSizeW(path, None)
    if not size:
        return Observation(CheckState.UNAVAILABLE, detail="no version resource")
    buf = ctypes.create_string_buffer(size)
    ver.GetFileVersionInfoW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    ver.GetFileVersionInfoW.restype = wintypes.BOOL
    if not ver.GetFileVersionInfoW(path, 0, size, buf):
        return Observation(CheckState.ERROR, detail="GetFileVersionInfoW failed")
    ptr = ctypes.c_void_p()
    length = wintypes.UINT()
    ver.VerQueryValueW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.UINT)]
    ver.VerQueryValueW.restype = wintypes.BOOL
    if not ver.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(length)) or not ptr.value:
        return Observation(CheckState.UNAVAILABLE, detail="no fixed file info")
    # VS_FIXEDFILEINFO: dwFileVersionMS at offset 8, dwFileVersionLS at offset 12.
    ms = ctypes.cast(ptr.value + 8, ctypes.POINTER(wintypes.DWORD)).contents.value
    ls = ctypes.cast(ptr.value + 12, ctypes.POINTER(wintypes.DWORD)).contents.value
    return Observation(CheckState.PRESENT, f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}")


def _uninstall_registry_version(display_name_prefix: str) -> Observation[str]:
    if sys.platform != "win32":
        return Observation(CheckState.UNAVAILABLE, detail="uninstall_registry needs Windows")
    import winreg

    roots = (
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
    )
    for hive, root in roots:
        try:
            with winreg.OpenKey(hive, root) as key:
                count = winreg.QueryInfoKey(key)[0]
                for i in range(count):
                    sub = winreg.EnumKey(key, i)
                    with winreg.OpenKey(key, sub) as entry:
                        try:
                            name, _ = winreg.QueryValueEx(entry, "DisplayName")
                        except OSError:
                            continue
                        if str(name).startswith(display_name_prefix):
                            try:
                                version, _ = winreg.QueryValueEx(entry, "DisplayVersion")
                            except OSError:
                                return Observation(CheckState.UNAVAILABLE, detail="entry has no DisplayVersion")
                            return Observation(CheckState.PRESENT, str(version))
        except OSError:
            continue
    return Observation(CheckState.ABSENT, detail="no uninstall entry")


# ---- liveness: the token stays inside this section -------------------------------------


@dataclass(frozen=True)
class _Session:
    pid: int | None
    url: str
    token: str

    def __repr__(self) -> str:
        return f"_Session(pid={self.pid}, url={self.url!r}, token=<redacted>)"


def _read_server_json(path: str, d: AppDef) -> _Session | None:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    pid = data.get(d.liveness_pid_key)
    url = data.get(d.liveness_url_key)
    token = data.get(d.liveness_token_key)
    return _Session(
        pid=pid if isinstance(pid, int) else None,
        url=url if isinstance(url, str) else "",
        token=token if isinstance(token, str) else "",
    )


def _pid_alive(pid: int | None) -> Observation[bool]:
    if pid is None or pid <= 0:
        return Observation(CheckState.UNAVAILABLE, detail="no pid in runtime file")
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.OpenProcess.restype = wintypes.HANDLE
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return Observation(CheckState.ABSENT, False)
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle(handle)
        return Observation(CheckState.PRESENT, True)
    try:
        os.kill(pid, 0)  # windows-footgun: ok — POSIX only, the win32 branch returned above
    except ProcessLookupError:
        return Observation(CheckState.ABSENT, False)
    except PermissionError:
        return Observation(CheckState.PRESENT, True)
    return Observation(CheckState.PRESENT, True)


def _endpoint_observation(raw_url: str, fixed_path: str) -> Observation[str]:
    """Accept only a loopback http URL with a numeric port and no userinfo; the path is ours."""
    if not raw_url:
        return Observation(CheckState.UNAVAILABLE, detail="no url in runtime file")
    try:
        parts = urlsplit(raw_url)
        hostname, port, username, password = parts.hostname, parts.port, parts.username, parts.password
    except ValueError:
        return Observation(CheckState.UNAVAILABLE, detail="malformed endpoint")
    if parts.scheme != "http" or username or password:
        return Observation(CheckState.UNAVAILABLE, detail="endpoint must be plain http without userinfo")
    if hostname not in _LOOPBACK_HOSTS:
        return Observation(CheckState.UNAVAILABLE, detail="endpoint must be loopback")
    if port is None or not (1 <= port <= 65535):
        return Observation(CheckState.UNAVAILABLE, detail="endpoint needs a numeric port")
    return Observation(CheckState.PRESENT, f"http://{hostname}:{port}{fixed_path}")


def _mcp_initialize(session: _Session, endpoint: str, deadline_s: float) -> Observation[bool]:
    """One MCP `initialize` POST under one absolute deadline covering connect, headers, and body."""
    import http.client

    parts = urlsplit(endpoint)
    body = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "hermes", "version": "probe"}},
    }).encode()
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if session.token:
        headers["Authorization"] = f"Bearer {session.token}"
    deadline = time.monotonic() + max(0.1, deadline_s)

    def remaining() -> float:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError
        return left

    conn = http.client.HTTPConnection(parts.hostname or "127.0.0.1", parts.port or 80, timeout=remaining())
    try:
        conn.request("POST", parts.path or "/", body=body, headers=headers)
        conn.sock.settimeout(remaining())
        status = conn.getresponse().status
    except TimeoutError:
        return Observation(CheckState.ABSENT, False, f"no answer within {deadline_s:g} s")
    except (OSError, http.client.HTTPException):
        return Observation(CheckState.ABSENT, False, "connection refused or timed out")
    finally:
        conn.close()
    if status in (200, 401, 403):
        return Observation(CheckState.PRESENT, True, f"http {status}")
    return Observation(CheckState.ABSENT, False, f"http {status}")
