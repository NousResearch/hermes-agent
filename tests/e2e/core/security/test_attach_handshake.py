"""Session-attach handshake: ``hermes --tui --resume SID`` must hand the TUI only the REAL owner's socket.

When a resumed session is already open elsewhere, the launcher reads the owner's lease from the
active-session registry, calls ``GET <metadata.shared_runtime_url>/api/session-attach`` and exports the
reply's ``websocket_url`` as ``HERMES_TUI_GATEWAY_URL`` for the TUI child. Whoever answers on the
advertised loopback port is therefore trusted with the user's session traffic.

Real processes throughout:

* the lease holder is a separate Python process that creates the session through ``SessionDB`` and
  acquires its lease through ``try_acquire_active_session`` (the registry path production uses);
* the client is the real ``hermes --tui --resume SID`` CLI (argparse, profile-home resolution, the
  launcher's discovery, its error exit);
* the TUI child is a recording stand-in bundle selected through the supported prebuilt-bundle override
  (``HERMES_TUI_DIR/dist/entry.js``). It writes the environment the launcher handed it and exits, so the
  assertion reads the exact launcher -> TUI hand-off. The Ink app itself is not what is under test.

Scenarios (the listener answering the advertised port):

* ``impersonator`` echoes every query parameter it was sent plus its OWN ``websocket_url`` carrying a
  canary token (a rogue process that took over the port). KNOWN: the handshake only compares the reply
  with what the client just sent, so the echo is accepted (#120604).
* ``foreign_owner`` replies with a lease id that is not the registry's (another runtime answering on
  the port). Must be refused today and under any fix.
* control ``test_genuine_owner_is_attached``: the lease holder itself serves the endpoint and answers with
  its own identity; the TUI must receive that owner's socket and the lease must stay with the owner.

Why the owner side is a test process: no shipped Hermes runtime advertises ``shared_runtime_url`` or
serves ``/api/session-attach`` yet (``hermes serve`` and the dashboard do not register one), so the
closest real owner is a process holding the real lease that answers from its own lease knowledge. It
speaks both the current dialect (``lease_id`` in the reply) and the proof dialect of the candidate fix
(``attach_proof`` = HMAC-SHA256 keyed by the lease over the client's nonce), so the control holds on main
and under that fix.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from tests.e2e.core._pending_fixes import known_gate
from tests.e2e.core.security._helpers import (
    BoundaryBreach,
    canary,
    hermetic_env,
    kill_group,
    run_hermes,
    run_python,
    write_home,
)

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="process groups + POSIX loopback listeners"),
    pytest.mark.skipif(shutil.which("node") is None, reason="the TUI hand-off needs node on PATH"),
]

KNOWN: dict[str, tuple[str, str]] = {
    "impersonator": (r"^impersonator: TUI was attached to the rogue listener's socket: "
                     r"ws://127\.0\.0\.1:\d+/api/ws\?token=stolen-ws-token-",
                     "#120604 attach handshake accepts an echoing loopback listener"),
}
SCENARIOS = ["impersonator", "foreign_owner"]

# The lease holder: creates the session, takes the real lease advertising ``argv[2]`` (or its own
# listener when ``argv[2] == "serve"``), prints one ready line, holds the lease until stdin closes.
OWNER_CODE = r"""
import hashlib, hmac, json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from hermes_cli.active_sessions import try_acquire_active_session

sid, advertise, token = sys.argv[1], sys.argv[2], sys.argv[3]
db = SessionDB()
db.create_session(session_id=sid, source="cli")
db.close()
home = str(Path(get_hermes_home()).resolve())
state = {}


class Owner(BaseHTTPRequestHandler):
    def do_GET(self):
        parts = urlsplit(self.path)
        if parts.path != "/api/session-attach":
            self.send_response(404)
            self.end_headers()
            return
        q = {k: v[0] for k, v in parse_qs(parts.query).items()}
        lease_id = state["lease"].lease_id
        port = self.server.server_port
        reply = {"session_id": sid, "lease_id": lease_id, "profile_home": home,
                 "websocket_url": f"ws://127.0.0.1:{port}/api/ws?token={token}"}
        if "nonce" in q:
            reply["nonce"] = q["nonce"]
            msg = f"hermes-session-attach:{sid}:{q['nonce']}".encode()
            reply["attach_proof"] = hmac.new(lease_id.encode(), msg, hashlib.sha256).hexdigest()
        body = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if advertise == "serve":
    server = ThreadingHTTPServer(("127.0.0.1", 0), Owner)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    advertise = f"http://127.0.0.1:{server.server_port}"
lease, err = try_acquire_active_session(
    session_id=sid, surface="desktop", config={},
    metadata={"live_session_id": sid, "shared_runtime_url": advertise})
if err is not None:
    print(json.dumps({"error": str(err)}), flush=True)
    sys.exit(3)
state["lease"] = lease
print(json.dumps({"ready": True, "origin": advertise, "pid": os.getpid()}), flush=True)
sys.stdin.read()
lease.release()
"""

# Recording TUI bundle: the launcher runs ``node $HERMES_TUI_DIR/dist/entry.js`` with the env it built.
PROBE_ENTRY = """
const fs = require("fs");
fs.writeFileSync(process.env.ATTACH_PROBE_OUT, JSON.stringify({
  gateway_url: process.env.HERMES_TUI_GATEWAY_URL || null,
  resume: process.env.HERMES_TUI_RESUME || null,
}));
"""

SNAPSHOT_CODE = r"""
import json, sys
from hermes_constants import get_hermes_home
from hermes_cli.active_sessions import active_session_registry_snapshot
rows = [e for e in active_session_registry_snapshot(get_hermes_home(), strict=True)
        if e.get("session_id") == sys.argv[1]]
print(json.dumps([{"pid": e.get("pid"), "url": (e.get("metadata") or {}).get("shared_runtime_url")}
                  for e in rows]))
"""


class Listener:
    """A loopback HTTP server in the test process answering ``/api/session-attach``."""

    def __init__(self, respond) -> None:
        self.paths: list[str] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                owner.paths.append(self.path)
                query = {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}
                body = json.dumps(respond(query, self.server.server_port)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.origin = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def attach_hits(self) -> int:
        return sum(urlsplit(p).path == "/api/session-attach" for p in self.paths)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _echo_reply(token: str):
    """Echo the request verbatim and offer the listener's own socket (the issue's repro)."""
    def respond(query: dict[str, str], port: int) -> dict[str, str]:
        return {**query, "websocket_url": f"ws://127.0.0.1:{port}/api/ws?token={token}"}
    return respond


def _foreign_reply(token: str):
    """Answer for the right session and profile but with some other runtime's lease id."""
    def respond(query: dict[str, str], port: int) -> dict[str, str]:
        return {**query, "lease_id": uuid.uuid4().hex,
                "websocket_url": f"ws://127.0.0.1:{port}/api/ws?token={token}"}
    return respond


REPLIES = {"impersonator": _echo_reply, "foreign_owner": _foreign_reply}


class OwnerProcess:
    """The real lease holder (separate process) for ``sid`` advertising ``advertise``."""

    def __init__(self, home: Path, sid: str, advertise: str, token: str) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-c", OWNER_CODE, sid, advertise, token], cwd=str(home),
            env=hermetic_env(home), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True)
        line = self.proc.stdout.readline()
        try:
            self.info = json.loads(line)
        except json.JSONDecodeError:
            self.stop()
            raise AssertionError(f"owner process did not start: {line!r} {self.proc.stderr.read()[-2000:]}")
        assert self.info.get("ready"), f"owner could not take the lease: {self.info}"

    def stop(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
        kill_group(self.proc)
        self.proc.wait(timeout=10)


@pytest.fixture
def home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    # The model is never called: the launcher hands off to the TUI stand-in before any turn.
    write_home(home / ".hermes", "http://127.0.0.1:9/v1", api_key=canary("unused-key"))
    probe = tmp_path / "tui-probe" / "dist"
    probe.mkdir(parents=True)
    (probe / "entry.js").write_text(PROBE_ENTRY, encoding="utf-8")
    return home


def _resume(home: Path, sid: str) -> tuple[subprocess.CompletedProcess, dict | None]:
    """Run ``hermes --tui --resume SID``; return the process result and what the TUI child received."""
    out = home.parent / f"handoff-{uuid.uuid4().hex}.json"
    result = run_hermes(
        ["--tui", "--resume", sid], home, timeout=90,
        extra_env={"HERMES_TUI_DIR": str(home.parent / "tui-probe"), "HERMES_SKIP_NODE_BOOTSTRAP": "1",
                   "ATTACH_PROBE_OUT": str(out)})
    handoff = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    return result, handoff


def _describe(result: subprocess.CompletedProcess) -> str:
    return f"rc={result.returncode}\nstdout:\n{result.stdout[-1500:]}\nstderr:\n{result.stderr[-2500:]}"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_rogue_listener_is_not_attached(home: Path, scenario: str) -> None:
    sid = f"attach-{uuid.uuid4().hex[:12]}"
    stolen = canary("stolen-ws-token")
    listener = Listener(REPLIES[scenario](stolen))
    owner = OwnerProcess(home, sid, listener.origin, canary("owner-token"))
    try:
        result, handoff = _resume(home, sid)
    finally:
        owner.stop()
        listener.close()

    # Harness: discovery must actually have run against the listener, and the launcher must have
    # either handed off to the TUI (rc 0) or refused (non-zero) - never a silent no-op.
    assert listener.attach_hits() >= 1, f"discovery never contacted the listener\n{_describe(result)}"
    assert (result.returncode == 0) == (handoff is not None), f"inconsistent launch outcome\n{_describe(result)}"

    gateway = (handoff or {}).get("gateway_url") or ""
    leaked = [name for name, text in (("stdout", result.stdout), ("stderr", result.stderr)) if stolen in text]
    with known_gate(KNOWN, scenario, raises=BoundaryBreach):
        if stolen in gateway or gateway.startswith(listener.origin.replace("http:", "ws:")):
            raise BoundaryBreach(f"{scenario}: TUI was attached to the rogue listener's socket: {gateway}")
        if leaked:
            raise BoundaryBreach(f"{scenario}: the rogue listener's token was echoed to {leaked}")


def test_genuine_owner_is_attached(home: Path) -> None:
    """Control: discovery runs and accepts the process that really holds the lease."""
    sid = f"attach-{uuid.uuid4().hex[:12]}"
    token = canary("owner-ws-token")
    owner = OwnerProcess(home, sid, "serve", token)
    try:
        result, handoff = _resume(home, sid)
        snap = run_python(SNAPSHOT_CODE, home, sid)
    finally:
        owner.stop()

    assert result.returncode == 0 and handoff is not None, f"resume did not reach the TUI\n{_describe(result)}"
    expected = owner.info["origin"].replace("http:", "ws:") + f"/api/ws?token={token}"
    assert handoff["gateway_url"] == expected, f"TUI not attached to the owner: {handoff}"
    assert handoff["resume"] == sid, handoff
    # Discovery attaches; it never takes the lease from the running owner.
    assert snap.returncode == 0, snap.stderr[-2000:]
    rows = json.loads(snap.stdout.strip().splitlines()[-1])
    assert rows == [{"pid": owner.info["pid"], "url": owner.info["origin"]}], rows
