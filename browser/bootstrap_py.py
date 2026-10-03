"""browser.bootstrap_py — runs inside Pyodide BEFORE importing tui_gateway.

Stubs native-only modules so upstream files import unmodified, points
HERMES_HOME at the browser FS, and installs the page-mediated httpx/urllib3/
urllib transport so model calls leave via fetch() — TLS and CORS are the
browser's own, and no raw socket can wedge the interpreter.
"""
import os
import sys
import types

# --- 1. HERMES_HOME → in-browser FS --------------------------------------
# MEMFS for the spike; swapped for an OPFS-backed mount once we persist.
os.environ.setdefault("HERMES_HOME", "/hermes-home")
os.environ.setdefault("HERMES_SKIP_UPDATE_CHECKS", "1")
os.makedirs(os.environ["HERMES_HOME"], exist_ok=True)

# stdin: no TTY in a browser worker. EOF immediately so interactive prompts
# (input()/sys.stdin.read) raise EOFError instead of spinning on EAGAIN.
import io  # noqa: E402


class _EOFStdin(io.StringIO):
    def isatty(self):
        return False

    def readable(self):
        return True

    def read(self, *a):
        return ""

    def readline(self, *a):
        return ""


sys.stdin = _EOFStdin("")

# --- 2. Native-module stubs ------------------------------------------------
def _stub(name: str, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    mod.__browser_stub__ = True
    sys.modules[name] = mod
    return mod


def _install_native_stubs():
    """Replace modules with no wasm build. Registry's discover_builtin_tools
    already skips import-failures, but stubs keep shared utilities importable
    (e.g. modules that read psutil constants at import). Only stub what the
    wasm build genuinely lacks — signal/mmap/socket exist in Pyodide and
    stubbing them breaks asyncio/anyio/starlette."""
    import importlib.util

    def _stub_if_missing(name, **attrs):
        try:
            if importlib.util.find_spec(name) is not None:
                return None
        except Exception:
            pass
        return _stub(name, **attrs)

    psutil = _stub_if_missing("psutil")
    if psutil is not None:
        psutil.Process = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("psutil unavailable in browser"))
        psutil.pid_exists = lambda pid: False
        psutil.boot_time = lambda: 0.0
        psutil.cpu_count = lambda *a, **k: 1
        psutil.virtual_memory = lambda: types.SimpleNamespace(total=0, available=0, percent=0)
        psutil.disk_usage = lambda p: types.SimpleNamespace(total=0, used=0, free=0, percent=0)

    for name in ("ptyprocess", "ptyprocess.PtyProcess", "pexpect", "pty",
                 "nemo_relay", "pillow_heif", "winpty", "pywinpty",
                 "termios", "tty", "resource", "readline", "mmap",
                 "grp", "pwd"):
        _stub_if_missing(name)

    fcntl = _stub_if_missing("fcntl")
    if fcntl is not None:
        # DB layers flock() the sqlite file; single-writer cooperative runtime
        # makes it a real no-op rather than a capability gap.
        fcntl.flock = lambda fd, op: None
        fcntl.lockf = lambda fd, op, *a, **k: None
        fcntl.LOCK_EX = 2
        fcntl.LOCK_SH = 1
        fcntl.LOCK_UN = 8
        fcntl.LOCK_NB = 4

    # PIL ships in Pyodide (pillow wheel exists) — leave it; only stub if absent.
    try:
        import PIL  # noqa: F401
    except ImportError:
        _stub("PIL")
        _stub("PIL.Image")

    # subprocess exists in Pyodide but cannot spawn — any terminal tool that
    # reaches Popen must fail loudly, never hang. check_fn-style gating keeps
    # most of those tools unregistered anyway.
    import subprocess  # noqa: F401

    def _enoent(*a, **k):
        # Same exception the OS raises for a missing binary: upstream probe
        # paths (git_probe, which-style checks) treat ENOENT as "absent".
        raise FileNotFoundError(2, "No such file or directory (browser runtime)",
                                (a[0] if a else k.get("args") or [""])[0]
                                if isinstance(a[0] if a else k.get("args"), (list, tuple)) else (a[0] if a else k.get("args", "")))

    class _NoSpawnPopen(subprocess.Popen):
        """Popen stays a type (upstream annotates ``subprocess.Popen | None``)
        but instantiation always raises ENOENT — wasm cannot spawn."""

        def __init__(self, *a, **k):
            _enoent(*a, **k)

    subprocess.Popen = _NoSpawnPopen
    subprocess.run = _enoent
    subprocess.check_output = _enoent
    subprocess.call = _enoent

    _install_socket_guards()


def _install_socket_guards():
    """Raw TCP does not exist in a browser. Emscripten sockets attempt a
    WebSocket-backed proxy connect that can block inside a C call forever —
    wedging the cooperative interpreter with no pump. Every connect-family
    entry point must fail fast instead; all real egress goes through the
    page-mediated fetch path."""
    import errno
    import socket

    def _clog(msg):
        try:
            import js
            js.hermesBridge.log("i", "[sock] " + msg)
        except Exception:
            pass

    def _refuse(name):
        def _blocked(*a, **k):
            _clog(f"{name} refused: {str(a)[:120]}")
            raise OSError(errno.EHOSTUNREACH,
                          f"{name}: raw sockets unavailable in browser runtime")
        return _blocked

    socket.create_connection = _refuse("socket.create_connection")
    socket.getaddrinfo = _refuse("socket.getaddrinfo")
    socket.gethostbyname = _refuse("socket.gethostbyname")
    socket.gethostbyname_ex = _refuse("socket.gethostbyname_ex")
    socket.getfqdn = lambda *a, **k: "localhost"

    _Sock = socket.socket

    class _NoConnectSock(_Sock):
        def connect(self, *a, **k):
            raise OSError(errno.EHOSTUNREACH,
                          "connect: raw sockets unavailable in browser runtime")

        def connect_ex(self, *a, **k):
            return errno.EHOSTUNREACH

        def bind(self, *a, **k):
            raise OSError(errno.EHOSTUNREACH,
                          "bind: raw sockets unavailable in browser runtime")

        def listen(self, *a, **k):
            raise OSError(errno.EHOSTUNREACH,
                          "listen: raw sockets unavailable in browser runtime")

        def accept(self, *a, **k):
            raise OSError(errno.EHOSTUNREACH,
                          "accept: raw sockets unavailable in browser runtime")

    socket.socket = _NoConnectSock


# --- 3. Page-mediated httpx transport -------------------------------------
def _install_http_transport():
    """Model calls route through the page: the Python side hands the request
    to the worker bridge, which Atomics-waits while the page performs
    fetch() on the main thread.

    Response bodies are buffered (no incremental SSE streaming in v1 — the
    bytes are complete and the model SDK still parses them; incremental
    streaming needs the chunked ring extension)."""
    try:
        import httpx  # noqa: F401
    except ImportError:
        return

    class _BufferedStream(httpx.SyncByteStream):
        def __init__(self, data: bytes):
            self._data = data

        def __iter__(self):
            if self._data:
                yield self._data
            self._data = b""

    def _fetch_request(request):
        import base64

        from browser import gateway as py_gateway

        body_b64 = base64.b64encode(request.content).decode() if request.content else None
        resp = py_gateway.fetch_blocking(
            str(request.url), request.method,
            {k: v for k, v in request.headers.items()}, body_b64)
        status = int(resp.get("status") or 599)
        raw = base64.b64decode(resp.get("body") or "")
        if resp.get("error"):
            raise httpx.TransportError(str(resp["error"]), request=request)
        return httpx.Response(status, headers=resp.get("headers") or {},
                              stream=_BufferedStream(raw), request=request)

    class BrowserFetchTransport(httpx.BaseTransport):
        def handle_request(self, request):  # sync httpx; worker blocks on SAB
            return _fetch_request(request)

    # The real choke point is the default transport class itself: upstream
    # build_keepalive_http_client passes an explicit shared
    # httpx.HTTPTransport into httpx.Client and the openai SDK, so a
    # Client-subclass setdefault("transport") is bypassed. Patch
    # HTTPTransport.handle_request (and the async twin) so every transport —
    # shared, per-client, or default — routes through the page bridge.
    async def _fetch_async(self, request):
        # Atomics.wait inside a coroutine serializes the wasm loop, but the
        # fetch resolves on the page thread — same blocking shape as sync.
        return _fetch_request(request)

    # Patch through httpx._transports' re-exported names, not a submodule:
    # on Emscripten HTTPTransport aliases jsfetch.JavascriptFetchTransport
    # (the default.py module is never imported), on every other platform it
    # is HTTPCoreTransport — patching the alias covers both.
    try:
        import httpx._transports as _httpx_transports
    except ImportError:
        _httpx_transports = None
    if _httpx_transports is not None:
        _httpx_transports.HTTPTransport.handle_request = (
            lambda self, request: _fetch_request(request))
        _httpx_transports.AsyncHTTPTransport.handle_async_request = _fetch_async

    _orig = httpx.Client

    class PageFetchClient(_orig):
        def __init__(self, *a, **k):
            k.setdefault("transport", BrowserFetchTransport())
            super().__init__(*a, **k)

    httpx.Client = PageFetchClient

    # urllib3 is patched at urlopen, the single choke point requests'
    # HTTPAdapter.send and direct PoolManager users both funnel through.
    # On Emscripten urllib3's contrib backend falls back to a synchronous
    # XMLHttpRequest — a JS-level block no Python interrupt can reach.
    # Route it through the same page-side fetch as httpx.
    try:
        import urllib3.connectionpool as _ucp
        import urllib3.response as _ures
        from urllib3._collections import HTTPHeaderDict as _HHD
    except ImportError:
        return

    def _timeout_s(timeout, default=120.0):
        if timeout is None:
            return default
        for attr in ("read_timeout", "total"):
            v = getattr(timeout, attr, None)
            if isinstance(v, (int, float)) and v:
                return float(v)
        if isinstance(timeout, (int, float)):
            return float(timeout)
        return default

    def _fetch_urlopen(self, method, url, body=None, headers=None,
                       retries=None, redirect=True, assert_same_host=True,
                       timeout=None, pool_timeout=None, release_conn=None,
                       chunked=False, body_pos=None, preload_content=True,
                       decode_content=True, **response_kw):
        import base64
        import urllib.parse

        from browser import gateway as py_gateway

        if not urllib.parse.urlsplit(str(url)).scheme:
            host = self.host if self.port in (None, 80, 443) else f"{self.host}:{self.port}"
            url = f"{self.scheme}://{host}{url}"
        if isinstance(body, str):
            body = body.encode()
        body_b64 = base64.b64encode(body).decode() if body else None
        resp = py_gateway.fetch_blocking(
            str(url), str(method).upper(), dict(headers or {}), body_b64,
            timeout_s=_timeout_s(timeout))
        raw = base64.b64decode(resp.get("body") or "")
        if resp.get("error"):
            raise OSError(f"page fetch failed: {resp['error']}")
        return _ures.HTTPResponse(
            body=raw, status=int(resp.get("status") or 599),
            headers=_HHD(resp.get("headers") or {}), reason=None,
            preload_content=False, decode_content=False,
            original_response=None, request_method=str(method),
            request_url=str(url), version=11,
            enforce_content_length=False)

    _ucp.HTTPConnectionPool.urlopen = _fetch_urlopen

    # Third stdlib transport: urllib.request.urlopen / OpenerDirector.open go
    # through http.client -> raw socket. Route them through the same
    # page-side fetch; HTTP semantics (error raise, addinfourl shape) are
    # preserved.
    try:
        import email.message
        import urllib.error
        import urllib.request as _urq
        import urllib.response
    except ImportError:
        return

    def _fetch_urllib_open(url, data=None, timeout=None):
        import base64
        import io

        from browser import gateway as py_gateway

        is_req = hasattr(url, "full_url")
        full = url.full_url if is_req else str(url)
        method = (url.get_method() if is_req else None) or ("POST" if data else "GET")
        headers = dict(getattr(url, "headers", {}) or {})
        if data is None and is_req:
            data = getattr(url, "data", None)
        if isinstance(data, str):
            data = data.encode()
        body_b64 = base64.b64encode(data).decode() if data else None
        resp = py_gateway.fetch_blocking(
            full, method, headers, body_b64,
            timeout_s=float(timeout) if isinstance(timeout, (int, float)) else 120.0)
        if resp.get("error"):
            raise urllib.error.URLError(resp["error"])
        status = int(resp.get("status") or 599)
        msg = email.message.Message()
        for hk, hv in (resp.get("headers") or {}).items():
            msg[hk] = hv
        fp = urllib.response.addinfourl(io.BytesIO(base64.b64decode(resp.get("body") or "")),
                                      msg, full, code=status)
        fp.status = status
        if status >= 400:
            raise urllib.error.HTTPError(full, status, "", msg, fp)
        return fp

    _urq.urlopen = _fetch_urllib_open
    _urq.OpenerDirector.open = lambda self, url, data=None, timeout=None: _fetch_urllib_open(url, data, timeout)


def _install_exit_guards():
    """os._exit -> proc_exit under wasm = silent process death. Raise instead
    so exit paths surface as catchable errors during bring-up."""
    def _no_exit(code=0):
        raise SystemExit(f"os._exit({code}) intercepted in browser runtime")
    os._exit = _no_exit
    os.abort = lambda: (_ for _ in ()).throw(RuntimeError("os.abort intercepted"))


def install():
    _install_exit_guards()
    _install_native_stubs()
    _install_http_transport()
    sys.path.insert(0, "/hermes-py")  # unpacked vendored tree
    return True
