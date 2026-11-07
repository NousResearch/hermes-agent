"""Authenticated, SSRF-safe sidecar request channel on ``MediaRuntimeController``.

Extends the lifecycle fakes from ``test_media_runtime.py`` (pieces copied, that
file untouched) with a fake backend implementing the REAL upstream auth
contract read from the installed backend:

* ``POST /api/auth/login`` with ``{"username": "unsloth", "password": <stdin>}``
  -> ``Token{access_token, refresh_token, token_type: "bearer"}``
  (studio/backend/main.py:1906 prefix ``/api/auth``; routes/auth.py:445-499;
  models/auth.py:17-20 request shape; models/users.py:11-29 response shape;
  username 'unsloth' = auth/storage.py:28 DEFAULT_ADMIN_USERNAME).
* ``POST /api/auth/refresh`` with ``{"refresh_token": ...}`` -> new Token,
  single-use rotation; invalid/expired -> 401 (routes/auth.py:578-608;
  models/auth.py:30-33).
* bearer-protected routes answer ``401`` ``{"detail": "Not authenticated"}``
  without one (auth/authentication.py:128-179 ``_BearerOrKeyless(HTTPBearer)``
  fall-through, matching the Phase-1 probe).

Everything runs against real subprocesses on ephemeral 127.0.0.1 ports only.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

import psutil
import pytest

from hermes_cli.local_runtime import media_runtime as mr
from hermes_cli.local_runtime.admission import AdmissionStore

pytestmark = pytest.mark.platforms("posix")

_BOOT = "test-boot"
# Raw SQL in these tests targets the row directly, so it must name the slot canonically; the store's alias maps names, not WHERE clauses.
_RESOURCE = mr.DEFAULT_RESOURCE

# ---------------------------------------------------------------------------
# fake authenticated backend (copied launch contract from the lifecycle tests,
# extended with the upstream auth routes above)
# ---------------------------------------------------------------------------

_BACKEND = '''
import json
import os
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

MODE = "__MODE__"
RECEIPT = "__RECEIPT__"
HITS_FILE = RECEIPT + ".redirect-hits"

args = sys.argv[1:]


def _flag(name):
    return args[args.index(name) + 1]


# Mirror the REAL studio/backend/run.py argparse contract (it has no -H/-p; exits 2 on unknown flags).
_ACCEPTED = {"--api-only": 0, "--host": 1, "--port": 1, "--password": 1, "--no-cloudflare": 0}
_i = 0
while _i < len(args):
    if args[_i] not in _ACCEPTED:
        sys.stderr.write("run.py: error: unrecognized arguments: " + " ".join(args[_i:]) + "\\n")
        sys.exit(2)
    _i += 1 + _ACCEPTED[args[_i]]

assert "--api-only" in args, "api-only flag required"
assert "--no-cloudflare" in args, "a public tunnel must be explicitly off"
HOST = _flag("--host")
PORT = int(_flag("--port"))
assert _flag("--password") == "-", "password must be '-' (stdin)"
PASSWORD = sys.stdin.readline().strip()

TOKENS = {"access": "acc-" + uuid.uuid4().hex, "refresh": "ref-" + uuid.uuid4().hex}
STATE = {"hits": 0, "logins": 0, "refreshes": 0, "last_auth": None,
         "last_host": None, "expire_next": MODE == "rotate"}


def _write_receipt():
    Path(RECEIPT).write_text(json.dumps({
        "pid": os.getpid(), "port": PORT, "password": PASSWORD,
        "access": TOKENS["access"], "refresh": TOKENS["refresh"],
        "redirect_port": REDIRECT_PORT,
    }), encoding="utf-8")


REDIRECT_PORT = 0
if MODE != "legacy":  # second loopback server: the 302 redirect target

    class _Target(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            with Path(HITS_FILE).open("a", encoding="utf-8") as fh:
                fh.write(f"{self.command} {self.path}\\n")
            body = b"stolen"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    _target = ThreadingHTTPServer((HOST, 0), _Target)
    REDIRECT_PORT = _target.server_address[1]
    threading.Thread(target=_target.serve_forever, daemon=True).start()

_write_receipt()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _record(self, observe=True):
        STATE["hits"] += 1
        if observe:  # /_debug reads must not clobber the observation itself
            STATE["last_auth"] = self.headers.get("Authorization")
            STATE["last_host"] = self.headers.get("Host")

    def _json(self, code, payload, extra=()):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in extra:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            return {}

    def _token(self):
        return {"access_token": TOKENS["access"], "refresh_token": TOKENS["refresh"],
                "token_type": "bearer", "must_change_password": False,
                "account_id": None}

    def do_POST(self):
        if MODE == "legacy":  # like the pre-auth fakes: no POST handler at all
            self.send_error(501, "Unsupported method ('POST')")
            return
        path = urlsplit(self.path).path
        self._record(observe=not path.startswith("/_debug"))
        if path == "/api/auth/login":
            STATE["logins"] += 1
            body = self._body()
            ok = (MODE != "bad_login" and body.get("username") == "unsloth"
                  and body.get("password") == PASSWORD)
            if not ok:
                # upstream shape: 401 + reset-password prose (routes/auth.py:468-471)
                self._json(401, {"detail": "Incorrect password. To reset it, run ..."})
                return
            self._json(200, self._token())
            return
        if path == "/api/auth/refresh":
            STATE["refreshes"] += 1
            body = self._body()
            if (MODE == "refresh_dead"
                    or body.get("refresh_token") != TOKENS["refresh"]):
                self._json(401, {"detail": "Invalid or expired refresh token"})
                return
            # single-use rotation (routes/auth.py:578-608)
            TOKENS["access"] = "acc-" + uuid.uuid4().hex
            TOKENS["refresh"] = "ref-" + uuid.uuid4().hex
            STATE["expire_next"] = False
            _write_receipt()
            self._json(200, self._token())
            return
        if path == "/api/inference/audio/inputs":
            # upstream shape: RAW bytes body, ?name=, Content-Type = the clip's mime
            if STATE["expire_next"]:
                STATE["expire_next"] = False
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self._json(401, {"detail": "Token has been expired or revoked."})
                return
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            if self.headers.get("Authorization") != "Bearer " + TOKENS["access"]:
                self._json(401, {"detail": "Not authenticated"})
                return
            import hashlib
            name = parse_qs(urlsplit(self.path).query).get("name", [""])[0]
            self._json(200, {"id": "in-1", "name": name, "size": len(raw),
                             "sha256": hashlib.sha256(raw).hexdigest(),
                             "content_type": self.headers.get("Content-Type"),
                             "url": "/api/inference/audio/inputs/in-1/file"})
            return
        self._json(404, {"detail": "Not Found"})

    def do_GET(self):
        split = urlsplit(self.path)
        path = split.path
        self._record(observe=not path.startswith("/_debug"))
        if path == "/api/health":
            self._json(200, {"status": "ok"})
        elif path == "/api/models/list":
            if MODE == "refresh_dead" or STATE["expire_next"]:
                STATE["expire_next"] = False
                self._json(401, {"detail": "Token has been expired or revoked."})
                return
            if self.headers.get("Authorization") != "Bearer " + TOKENS["access"]:
                missing = not self.headers.get("Authorization")
                self._json(401, {"detail": "Not authenticated" if missing
                                 else "Could not validate credentials"})
                return
            self._json(200, {"models": [{"id": "m1"}],
                             "query": {k: v[0] for k, v in parse_qs(split.query).items()}},
                     extra=(("X-Test", "yes"), ("Set-Cookie", "sid=SECRETCOOKIE; Path=/")))
        elif path == "/api/redirect-me":
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{REDIRECT_PORT}/stolen")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif path == "/api/stream":
            data = b"".join(f"chunk{i:04d}".encode("ascii") * 8 for i in range(64))
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif path == "/api/stream/sse":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(4):
                self.wfile.write(f"data: event-{i}\\n\\n".encode("ascii"))
                self.wfile.flush()
        elif path == "/_debug/state":  # NOT under /api/: test reads it directly
            self._json(200, {"hits": STATE["hits"], "logins": STATE["logins"],
                             "refreshes": STATE["refreshes"],
                             "last_auth": STATE["last_auth"],
                             "last_host": STATE["last_host"]})
        else:
            self._json(404, {"detail": "Not Found"})


server = ThreadingHTTPServer((HOST, PORT), Handler)
print(f"Unsloth Studio running on http://{HOST}:{PORT}", flush=True)
print(f"TAURI_PORT={PORT}", flush=True)
server.serve_forever()
'''


def _write_backend(tmp_path: Path, mode: str) -> tuple[Path, Path]:
    script = tmp_path / f"backend_{mode}.py"
    receipt = script.with_suffix(".receipt.json")
    script.write_text(_BACKEND.replace("__MODE__", mode).replace(
        "__RECEIPT__", str(receipt)), encoding="utf-8")
    return script, receipt


def _controller(tmp_path: Path, store: AdmissionStore, **over):
    kwargs = dict(owner_scope="scope-a", data_root=tmp_path / "runtime",
                  admission=store, boot_id_fn=lambda: _BOOT,
                  health_timeout=5.0, lease_seconds=30.0)
    kwargs.update(over)
    return mr.MediaRuntimeController(**kwargs)


def _wait(predicate, *, timeout=15.0, what="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    pytest.fail(f"timed out waiting for {what}")


def _gone(pid: int) -> bool:
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return True
    try:
        return proc.status() == psutil.STATUS_ZOMBIE
    except psutil.Error:
        return True


def _http_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as resp:
        assert resp.status == 200
        return json.loads(resp.read().decode("utf-8"))


@pytest.fixture(autouse=True)
def _isolate_spawn_ledger(tmp_path, monkeypatch):
    from hermes_cli import process_identity as pi
    monkeypatch.setattr(pi, "_ledger_path", lambda: tmp_path / "spawn-ledger.jsonl")


@pytest.fixture
def live():
    controllers: list = []

    def track(obj):
        controllers.append(obj)
        return obj

    yield track
    for ctl in controllers:
        with contextlib.suppress(Exception):
            ctl.stop(timeout=10)


def _started(tmp_path, live, mode="auth"):
    """Start a controller against the fake backend and return (ctl, status, receipt)."""
    store = AdmissionStore(tmp_path / "adm")
    script, receipt = _write_backend(tmp_path, mode)
    ctl = live(_controller(tmp_path, store))
    status = ctl.start(python=sys.executable, run_py=str(script))
    return ctl, status, receipt


def _debug(ctl) -> dict:
    """Backend-side observations, fetched DIRECTLY (bypasses the controller channel)."""
    return _http_json(ctl.status().base_url + "/_debug/state")


def _strings(obj, depth=0):
    """Every string reachable from ``obj`` (recursive vars() scan)."""
    if depth > 6:
        return
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from _strings(key, depth + 1)
            yield from _strings(value, depth + 1)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for value in obj:
            yield from _strings(value, depth + 1)


# ---------------------------------------------------------------------------
# state gate: only 'running' serves
# ---------------------------------------------------------------------------

def test_request_and_stream_require_running_state(tmp_path, live):
    ctl = live(_controller(tmp_path, AdmissionStore(tmp_path / "adm")))
    for bad in (ctl.request,):
        with pytest.raises(mr.MediaRuntimeUnavailable):
            bad("GET", "/api/models/list")
    with pytest.raises(mr.MediaRuntimeUnavailable):
        ctl.stream_request("GET", "/api/stream")
    ctl, status, _ = _started(tmp_path, live)
    assert status.state == "running", status.detail
    assert ctl.stop().state == "stopped"
    with pytest.raises(mr.MediaRuntimeUnavailable):
        ctl.request("GET", "/api/models/list")


# ---------------------------------------------------------------------------
# start() logs in ONCE; the password is discarded, only tokens remain
# ---------------------------------------------------------------------------

def test_start_logs_in_once_and_password_is_discarded(tmp_path, live):
    ctl, status, receipt = _started(tmp_path, live)
    assert status.state == "running", status.detail
    facts = json.loads(receipt.read_text(encoding="utf-8"))
    password, access = facts["password"], facts["access"]
    assert password and access

    # the fake saw exactly one login, and the controller kept the tokens
    assert _debug(ctl)["logins"] == 1
    assert ctl._access_token == access
    assert ctl._refresh_token == facts["refresh"]

    # the password is nowhere on the controller, recursively
    for surface in _strings(vars(ctl)):
        assert password not in surface, "password retained as a controller attribute"

    # and on every public surface (the private token attributes themselves are
    # allowed only in vars(ctl), which the password scan above already covers)
    surfaces = [repr(ctl), str(ctl), repr(ctl.status()),
                str(ctl.status()), ctl.log_tail()]
    for text in surfaces + [repr(vars(ctl))]:
        assert password not in text
    for text in surfaces:
        assert access not in text  # tokens too must not leak into repr/status/log
    assert "password" not in ctl.status().detail.lower()


def test_login_failure_fails_start_through_teardown(tmp_path, live):
    store = AdmissionStore(tmp_path / "adm")
    script, _ = _write_backend(tmp_path, "bad_login")
    ctl = live(_controller(tmp_path, store))
    status = ctl.start(python=sys.executable, run_py=str(script))
    assert status.state == "failed"
    assert "login" in status.detail.lower()
    assert status.pid is not None
    _wait(lambda: _gone(status.pid), what="rejected-login sidecar to die")
    _wait(lambda: store.status(_RESOURCE).residency == "released",
          what="grant released after login failure")
    assert ctl._access_token is None and ctl._refresh_token is None
    for surface in _strings(vars(ctl)):
        assert "acc-" not in surface


def test_sidecar_without_an_auth_api_fails_the_start_closed(tmp_path, live):
    """Default is fail-closed: a sidecar whose auth API is missing (HTTP 501
    here) never ends up 'running' with a channel that silently refuses calls."""
    store = AdmissionStore(tmp_path / "adm")
    script, _ = _write_backend(tmp_path, "legacy")
    ctl = live(_controller(tmp_path, store))
    status = ctl.start(python=sys.executable, run_py=str(script))
    assert status.state == "failed", status.detail
    assert "auth" in status.detail.lower() or "login" in status.detail.lower()
    assert status.pid is not None
    _wait(lambda: _gone(status.pid), what="auth-less sidecar to die")
    _wait(lambda: store.status(_RESOURCE).residency == "released",
          what="grant released after the failed start")


def test_unreachable_auth_endpoint_fails_the_start_closed(tmp_path, live, monkeypatch):
    """A transient connection error during login on a healthy sidecar must not
    produce a 'running' runtime whose channel can never work."""
    import urllib.error

    real_open = mr._NO_REDIRECT_OPENER.open

    class _Flaky:
        def open(self, request, *args, **kwargs):
            if str(getattr(request, "full_url", "")).endswith(mr.LOGIN_PATH):
                raise urllib.error.URLError("connection refused")
            return real_open(request, *args, **kwargs)

    monkeypatch.setattr(mr, "_NO_REDIRECT_OPENER", _Flaky())
    store = AdmissionStore(tmp_path / "adm")
    script, _ = _write_backend(tmp_path, "auth")
    ctl = live(_controller(tmp_path, store))
    status = ctl.start(python=sys.executable, run_py=str(script))
    assert status.state == "failed", status.detail
    _wait(lambda: _gone(status.pid), what="sidecar torn down after login failure")
    assert ctl._access_token is None and ctl._refresh_token is None


def test_explicit_opt_out_runs_a_lifecycle_only_sidecar_without_a_channel(tmp_path, live):
    """require_channel_auth=False is the explicit lifecycle-only mode: it runs,
    but the channel refuses every call and never sends a bearer anywhere."""
    store = AdmissionStore(tmp_path / "adm")
    script, _ = _write_backend(tmp_path, "legacy")
    ctl = live(_controller(tmp_path, store, require_channel_auth=False))
    status = ctl.start(python=sys.executable, run_py=str(script))
    assert status.state == "running", status.detail
    with pytest.raises(mr.MediaRuntimeUnavailable) as exc:
        ctl.request("GET", "/api/models/list")
    assert "authentication" in str(exc.value).lower()
    with pytest.raises(mr.MediaRuntimeUnavailable):
        ctl.stream_request("GET", "/api/stream")
    assert ctl._access_token is None
    assert _debug(ctl)["last_auth"] is None  # no bearer ever attached to anything


@pytest.mark.parametrize("path", [
    "/api/\u2025/x",          # two-dot leader
    "/api/\uff0e\uff0e/x",   # fullwidth dots
    "/api/a b",               # raw space
    "/api/a\u00a0b",          # no-break space
    "/api/\u202ex",           # bidi override
    "/api/caf\u00e9",         # any non-ASCII: Studio's API paths are ASCII
])
def test_guard_is_ascii_only_and_rejects_whitespace_itself(path):
    """The guard must not depend on http.client happening to reject these."""
    with pytest.raises(ValueError):
        mr._validate_channel_path(path)


# ---------------------------------------------------------------------------
# request(): bearer attached, caller cannot override it, response sanitized
# ---------------------------------------------------------------------------

def test_request_attaches_controller_bearer_and_sanitizes_response(tmp_path, live):
    ctl, _status, receipt = _started(tmp_path, live)
    tokens = json.loads(receipt.read_text(encoding="utf-8"))
    resp = ctl.request("GET", "/api/models/list", headers={"X-Trace": "t1"})
    assert isinstance(resp, mr.ChannelResponse)
    assert resp.status == 200
    assert json.loads(resp.body)["models"][0]["id"] == "m1"
    # lowercase keys, hop-by-hop and set-cookie removed (token material stripped)
    assert resp.headers["x-test"] == "yes"
    assert "set-cookie" not in resp.headers
    assert "transfer-encoding" not in resp.headers
    for header_value in resp.headers.values():
        assert tokens["access"] not in header_value
    # the backend saw the controller's own bearer and the loopback Host
    debug = _debug(ctl)
    assert debug["last_auth"] == "Bearer " + tokens["access"]
    assert debug["last_host"].startswith("127.0.0.1:")


@pytest.mark.parametrize("forged", [
    {"Authorization": "***"},
    {"authorization": "Bearer evil"},
    {"Cookie": "sid=evil"},
    {"Host": "evil.example"},
    {"Content-Length": "999999"},
    # request FRAMING belongs to the transport alone: a caller-set Transfer-Encoding next to the transport's Content-Length is the
    # classic CL.TE smuggling pair, and the hop-by-hop headers change how the connection itself is handled
    {"Transfer-Encoding": "chunked"},
    {"transfer-encoding": "identity"},
    {"Connection": "close"},
    {"Keep-Alive": "timeout=999"},
    {"Upgrade": "websocket"},
    {"TE": "trailers"},
    {"Trailer": "Authorization"},
    {"Proxy-Authorization": "Basic eDp5"},
    {"Proxy-Connection": "keep-alive"},
])
def test_caller_cannot_override_credential_headers(tmp_path, live, forged):
    ctl, _status, _ = _started(tmp_path, live)
    before = _debug(ctl)["hits"]
    with pytest.raises(ValueError):
        ctl.request("POST", "/api/models/list", json_body={"a": 1}, headers=forged)
    # the only new hit is this second /_debug read itself
    assert _debug(ctl)["hits"] == before + 1  # nothing else was sent


# ---------------------------------------------------------------------------
# SSRF / path guard: ValueError raised, nothing sent, bearer never attached
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "http://127.0.0.1:9/api/models/list",   # absolute URL
    "https://evil.example/api/x",
    "//evil.example/api/x",                 # protocol-relative
    "file:///etc/passwd",
    "HTTP://127.0.0.1:1/api/x",
    "/api/../../etc/passwd",
    "/api/../secret",
    "/api/%2e%2e/secret",                   # encoded '..'
    "/api/%2E%2E/secret",                   # any case
    "/api/x%2fy",                           # encoded slash
    "/api/x%5Cy",                           # encoded backslash
    "/api/x%00y",                           # encoded NUL
    "/api/x\ty",                            # control char
    "/api/x\nY: 1",                         # newline / header split
    "/api/x\r\nZ: 1",
    "/api/x\x00y",                          # raw NUL
    "/api/x@evil.example/",                 # userinfo
    "/api/x#frag",                          # fragment
    "/api/x?y=1",                           # query smuggled into path
    "/api/" + "a" * 3000,                   # over-long path
    "api/models/list",                      # not absolute-relative
    "/other/path",                          # outside /api/
    "/apix/models",
    "/api\\..\\secret",                     # backslash traversal
    "/api/a//b",                            # an empty segment: a router that collapses '//' may route it elsewhere
    "/api//models/list",
    "/api/models//",
    "/api///x",
])
def test_ssrf_path_guard_rejects_everything_but_relative_api_paths(tmp_path, live, path):
    ctl, _status, _ = _started(tmp_path, live)
    before = _debug(ctl)["hits"]
    with pytest.raises(ValueError):
        ctl.request("GET", path)
    with pytest.raises(ValueError):
        ctl.stream_request("GET", path)
    # the only new hit is this second /_debug read itself
    assert _debug(ctl)["hits"] == before + 1, "guard must reject BEFORE anything is sent"


def test_query_dict_is_the_only_query_channel(tmp_path, live):
    ctl, _status, _ = _started(tmp_path, live)
    resp = ctl.request("GET", "/api/models/list", query={"q": "hello world", "n": "1"})
    assert resp.status == 200
    assert json.loads(resp.body)["query"] == {"q": "hello world", "n": "1"}


# ---------------------------------------------------------------------------
# redirects: returned as-is, never followed, bearer never re-sent anywhere
# ---------------------------------------------------------------------------

class _Listener:
    """A loopback TCP listener that records the first request line of every connection; ``reply`` makes it answer 200 so it can be a target."""

    def __init__(self, reply: bool = False):
        import socket
        import threading
        self.requests: list[str] = []
        self.bodies: list[str] = []
        self._reply = reply
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(16)
        self.port = self._sock.getsockname()[1]
        self._stop = False
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            try:
                conn.settimeout(1.0)
                data = self._read_request(conn)
                self.requests.append(data.splitlines()[0] if data else "")
                self.bodies.append(data)
                if self._reply:
                    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
            except OSError:
                pass
            finally:
                conn.close()

    @staticmethod
    def _read_request(conn) -> str:
        """The whole request: urllib may send the headers and the body in separate segments, so one recv() sometimes misses the body."""
        raw = b""
        while b"\r\n\r\n" not in raw:
            chunk = conn.recv(8192)
            if not chunk:
                break
            raw += chunk
        head, _, rest = raw.partition(b"\r\n\r\n")
        length = 0
        for line in head.split(b"\r\n")[1:]:
            name, _, value = line.partition(b":")
            if name.strip().lower() == b"content-length":
                length = int(value.strip() or 0)
        while len(rest) < length:
            chunk = conn.recv(8192)
            if not chunk:
                break
            rest += chunk
        return (head + b"\r\n\r\n" + rest).decode("latin-1", "replace")

    def close(self):
        self._stop = True
        try:
            self._sock.close()
        except OSError:
            pass


_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Runs in a FRESH interpreter whose proxy variables are set before anything is imported: the production condition (Hermes started from a
#: shell that exports http_proxy). In-process this cannot be tested reliably: urllib builds its global opener once, at first use.
_PROXY_PROBE = """
import json, sys, urllib.request
mode, target = sys.argv[1], sys.argv[2]
out = {}
if mode == "default":
    with urllib.request.build_opener().open(f"http://127.0.0.1:{target}/api/health", timeout=3) as r:
        out["default_opener_status"] = r.status
else:
    from hermes_cli.local_runtime import media_runtime as mr
    out["health"] = mr._probe_health(f"http://127.0.0.1:{target}/api/health")
    request = urllib.request.Request(f"http://127.0.0.1:{target}/api/auth/login", data=b'{"password":"SECRET-PASSWORD"}',
                                     headers={"Authorization": "Bearer SECRET-TOKEN", "Content-Type": "application/json"}, method="POST")
    with mr._NO_REDIRECT_OPENER.open(request, timeout=3) as r:
        out["login_status"] = r.status
print(json.dumps(out))
"""


def _run_probe(mode: str, proxy_port: int, target_port: int) -> dict:
    import os
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(_REPO_ROOT), "HOME": os.environ.get("HOME", ""),
           "http_proxy": f"http://127.0.0.1:{proxy_port}", "HTTP_PROXY": f"http://127.0.0.1:{proxy_port}",
           "https_proxy": f"http://127.0.0.1:{proxy_port}", "HTTPS_PROXY": f"http://127.0.0.1:{proxy_port}",
           "all_proxy": f"http://127.0.0.1:{proxy_port}", "ALL_PROXY": f"http://127.0.0.1:{proxy_port}",
           "no_proxy": "localhost", "NO_PROXY": "localhost"}
    done = subprocess.run([sys.executable, "-c", _PROXY_PROBE, mode, str(target_port)], capture_output=True, text=True, env=env,
                          cwd=str(_REPO_ROOT), timeout=60, check=False)
    assert done.returncode == 0, done.stderr[-800:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.fixture
def proxy_and_target():
    proxy, target = _Listener(), _Listener(reply=True)
    yield proxy, target
    proxy.close()
    target.close()


def test_the_proxy_environment_would_catch_a_leak(proxy_and_target):
    """Positive control: urllib's DEFAULT opener sends a 127.0.0.1 request to the proxy under this environment (NO_PROXY=localhost does not
    cover the 127.0.0.1 literal). If this ever stops being true the two tests below prove nothing."""
    proxy, target = proxy_and_target
    with pytest.raises(AssertionError):                           # the stand-in proxy answers nothing, so the default opener fails
        _run_probe("default", proxy.port, target.port)
    assert proxy.requests and proxy.requests[0].startswith(f"GET http://127.0.0.1:{target.port}/api/health")
    assert target.requests == []


def test_loopback_health_and_login_never_go_through_a_proxy(proxy_and_target):
    """The login POST carries the sidecar password and every channel request carries the bearer: neither may ride an ambient proxy,
    and the health probe must reach the sidecar itself (with a proxy in the way the managed runtime could not even start)."""
    proxy, target = proxy_and_target
    result = _run_probe("controller", proxy.port, target.port)
    assert result == {"health": True, "login_status": 200}
    assert proxy.requests == [], f"loopback traffic reached the proxy: {proxy.requests[:3]}"
    assert len(target.requests) == 2
    assert any("SECRET-PASSWORD" in body and "SECRET-TOKEN" in body for body in target.bodies), "the login reached the sidecar directly"


def test_ordinary_caller_headers_still_pass_after_the_denylist_grew(tmp_path, live):
    """Control: the denylist must not eat the headers a plugin legitimately sends."""
    ctl, _status, _ = _started(tmp_path, live)
    resp = ctl.request("GET", "/api/models/list", headers={"X-Trace": "t1", "Accept": "application/json", "Range": "bytes=0-9",
                                                         "Content-Type": "application/json", "If-None-Match": "abc"})
    assert resp.status == 200


def test_single_slashes_and_a_trailing_slash_are_still_valid_paths(tmp_path, live):
    """Control for the empty-segment rule."""
    assert mr._validate_channel_path("/api/models/list") == "/api/models/list"
    assert mr._validate_channel_path("/api/models/") == "/api/models/"
    assert mr._validate_channel_path("/api/inference/audio/gallery/abc/file") == "/api/inference/audio/gallery/abc/file"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_the_raw_sidecar_log_is_private_to_the_user(tmp_path, live):
    """The log is captured before any scrubbing (sanitize_credentials runs only when the tail is READ), so it must not be world-readable."""
    ctl, _status, _ = _started(tmp_path, live)
    log = ctl._log_path
    assert log is not None and log.exists()
    assert (log.stat().st_mode & 0o077) == 0, oct(log.stat().st_mode)


def test_a_proxy_environment_does_not_stop_a_real_sidecar_from_starting_and_serving(tmp_path, live, proxy_and_target, monkeypatch):
    """The whole lifecycle with proxy variables exported to the test process (a smoke test; the subprocess tests above are the regression guard)."""
    proxy, _target = proxy_and_target
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.setenv(name, f"http://127.0.0.1:{proxy.port}")
    monkeypatch.setenv("no_proxy", "localhost")
    ctl, status, _receipt = _started(tmp_path, live)
    assert status.state == "running", status.detail
    assert ctl.request("GET", "/api/models/list").status == 200
    stream = ctl.stream_request("GET", "/api/models/list")
    assert b"models" in b"".join(stream)
    stream.close()
    assert proxy.requests == []


def test_redirect_not_followed_and_second_server_sees_zero(tmp_path, live):
    ctl, _status, receipt = _started(tmp_path, live)
    facts = json.loads(receipt.read_text(encoding="utf-8"))
    resp = ctl.request("GET", "/api/redirect-me")
    assert resp.status == 302
    assert resp.headers["location"] == f"http://127.0.0.1:{facts['redirect_port']}/stolen"
    hits_file = Path(str(receipt) + ".redirect-hits")
    assert not hits_file.exists(), "redirect target server received requests"
    stream = ctl.stream_request("GET", "/api/redirect-me")
    assert stream.status == 302  # surfaced as-is, never followed
    assert b"".join(stream) == b""
    assert not hits_file.exists()


# ---------------------------------------------------------------------------
# refresh-once-then-retry; expired refresh => 'auth lost'
# ---------------------------------------------------------------------------

def test_401_triggers_single_refresh_then_retry(tmp_path, live):
    ctl, _status, receipt = _started(tmp_path, live, mode="rotate")
    old = json.loads(receipt.read_text(encoding="utf-8"))
    resp = ctl.request("GET", "/api/models/list")  # first call 401s server-side
    assert resp.status == 200
    fresh = json.loads(receipt.read_text(encoding="utf-8"))
    assert fresh["access"] != old["access"], "backend did not rotate"
    debug = _debug(ctl)
    assert debug["refreshes"] == 1, "must refresh exactly once"
    assert debug["last_auth"] == "Bearer " + fresh["access"]
    assert ctl._access_token == fresh["access"]
    assert ctl._refresh_token == fresh["refresh"]
    assert ctl.status().state == "running"
    # tokens never leak into the response headers or public surfaces
    text = repr(ctl) + repr(ctl.status()) + ctl.log_tail() + json.dumps(resp.headers)
    for token in (fresh["access"], fresh["refresh"], old["access"]):
        assert token not in text


def test_expired_refresh_marks_auth_lost(tmp_path, live):
    ctl, _status, receipt = _started(tmp_path, live, mode="refresh_dead")
    tokens = json.loads(receipt.read_text(encoding="utf-8"))
    with pytest.raises(mr.MediaRuntimeUnavailable) as exc:
        ctl.request("GET", "/api/models/list")
    assert "auth lost" in str(exc.value)
    assert ctl.status().state == "running"  # sidecar alive; auth is what died
    assert ctl.status().detail == "auth lost - restart the runtime"
    # tokens are gone (the password is long gone): nothing authentic remains
    for surface in _strings(vars(ctl)):
        assert tokens["access"] not in surface
        assert tokens["refresh"] not in surface
    assert ctl._access_token is None and ctl._refresh_token is None
    # never falls back to the password: subsequent calls fail fast, zero hits
    before = _debug(ctl)["hits"]
    with pytest.raises(mr.MediaRuntimeUnavailable):
        ctl.request("GET", "/api/models/list")
    with pytest.raises(mr.MediaRuntimeUnavailable):
        ctl.stream_request("GET", "/api/stream")
    assert _debug(ctl)["hits"] == before + 1  # only the second /_debug read itself
    assert _debug(ctl)["refreshes"] == 1, "must not retry refresh after auth is lost"


# ---------------------------------------------------------------------------
# stream_request(): chunks + status/headers, same guards, closes properly
# ---------------------------------------------------------------------------

def test_stream_reads_all_bytes_and_closes_when_exhausted(tmp_path, live):
    ctl, _status, _ = _started(tmp_path, live)
    stream = ctl.stream_request("GET", "/api/stream", chunk_size=300)
    assert stream.status == 200
    assert stream.headers["content-type"] == "application/octet-stream"
    chunks = list(stream)
    expected = b"".join(f"chunk{i:04d}".encode("ascii") * 8 for i in range(64))
    assert b"".join(chunks) == expected
    assert len(chunks) > 1
    assert stream._response.closed, "connection must close when the iterator ends"


def test_stream_close_midway_releases_connection(tmp_path, live):
    ctl, _status, _ = _started(tmp_path, live)
    stream = ctl.stream_request("GET", "/api/stream", chunk_size=128)
    it = iter(stream)
    first = next(it)
    assert first.startswith(b"chunk0000")
    it.close()  # generator close -> finally -> response close
    assert stream._response.closed
    assert "MediaStream" in repr(stream)
    assert stream.status == 200


def test_stream_sse_yields_incrementally(tmp_path, live):
    ctl, _status, _ = _started(tmp_path, live)
    stream = ctl.stream_request("GET", "/api/stream/sse")
    events = list(stream)
    assert len(events) == 4, "SSE lines must surface per event, not one blob"
    assert events[0].strip() == b"data: event-0"
    assert stream.headers["content-type"] == "text/event-stream"


# ---------------------------------------------------------------------------
# raw request bodies (audio clip upload): additive to the JSON-only channel
# ---------------------------------------------------------------------------

_CLIP = bytes(range(256)) * 600          # 153 600 bytes, NUL and non-UTF-8 included
_UPLOAD = "/api/inference/audio/inputs"


def test_raw_body_is_uploaded_verbatim_with_the_callers_content_type(tmp_path, live):
    ctl, status, _ = _started(tmp_path, live)
    assert status.state == "running", status.detail
    resp = ctl.request("POST", _UPLOAD, query={"name": "my clip.wav"}, raw_body=_CLIP,
                       headers={"Content-Type": "audio/wav"})
    assert resp.status == 200
    rec = json.loads(resp.body)
    assert rec["size"] == len(_CLIP)
    assert rec["sha256"] == hashlib.sha256(_CLIP).hexdigest()
    assert rec["content_type"] == "audio/wav"
    assert rec["name"] == "my clip.wav"
    assert _debug(ctl)["last_auth"].startswith("Bearer ")


def test_raw_body_defaults_to_octet_stream(tmp_path, live):
    ctl, status, _ = _started(tmp_path, live)
    assert status.state == "running", status.detail
    rec = json.loads(ctl.request("POST", _UPLOAD, raw_body=b"abc").body)
    assert rec["content_type"] == "application/octet-stream"


def test_raw_body_and_json_body_are_mutually_exclusive_and_nothing_is_sent(tmp_path, live):
    ctl, status, _ = _started(tmp_path, live)
    assert status.state == "running", status.detail
    hits = _debug(ctl)["hits"]
    with pytest.raises(ValueError):
        ctl.request("POST", _UPLOAD, json_body={"a": 1}, raw_body=b"x")
    assert _debug(ctl)["hits"] == hits + 1      # only the debug read itself: nothing was sent


@pytest.mark.parametrize("bad", ["a string", 12, {"k": "v"}, [1, 2]])
def test_raw_body_must_be_bytes_like(tmp_path, live, bad):
    ctl, status, _ = _started(tmp_path, live)
    assert status.state == "running", status.detail
    with pytest.raises(ValueError):
        ctl.request("POST", _UPLOAD, raw_body=bad)


def test_raw_body_over_the_cap_is_refused_before_anything_is_sent(tmp_path, live, monkeypatch):
    ctl, status, _ = _started(tmp_path, live)
    assert status.state == "running", status.detail
    monkeypatch.setattr(mr, "CHANNEL_BODY_MAX", 1024)
    before = _debug(ctl)["hits"]
    with pytest.raises(ValueError):
        ctl.request("POST", _UPLOAD, raw_body=b"x" * 1025)
    assert _debug(ctl)["hits"] == before + 1      # the debug read itself, no upload
    assert json.loads(ctl.request("POST", _UPLOAD, raw_body=b"x" * 1024).body)["size"] == 1024


def test_raw_upload_survives_a_401_refresh_and_retry_with_identical_bytes(tmp_path, live):
    ctl, status, _ = _started(tmp_path, live, mode="rotate")   # first protected call 401s
    assert status.state == "running", status.detail
    resp = ctl.request("POST", _UPLOAD, raw_body=_CLIP, headers={"Content-Type": "audio/wav"})
    assert resp.status == 200
    rec = json.loads(resp.body)
    assert rec["sha256"] == hashlib.sha256(_CLIP).hexdigest()
    assert _debug(ctl)["refreshes"] == 1


def test_caller_still_cannot_forge_framing_headers_on_a_raw_upload(tmp_path, live):
    ctl, status, _ = _started(tmp_path, live)
    assert status.state == "running", status.detail
    for forged in ({"Content-Length": "1"}, {"Authorization": "Bearer x"}, {"Host": "evil"}):
        with pytest.raises(ValueError):
            ctl.request("POST", _UPLOAD, raw_body=b"abc", headers=forged)


def test_stream_request_accepts_a_raw_body_too(tmp_path, live):
    ctl, status, _ = _started(tmp_path, live)
    assert status.state == "running", status.detail
    stream = ctl.stream_request("POST", _UPLOAD, raw_body=_CLIP,
                                headers={"Content-Type": "audio/mpeg"})
    try:
        assert stream.status == 200
        rec = json.loads(b"".join(stream))
    finally:
        stream.close()
    assert rec["sha256"] == hashlib.sha256(_CLIP).hexdigest()
    assert rec["content_type"] == "audio/mpeg"
