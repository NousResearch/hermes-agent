"""``hermes acp`` as a thin relay to the gateway's hosted ACP socket.

The gateway owns every ACP session; the per-editor ``hermes acp`` process only translated ACP to the
gateway WebSocket, yet each one was a full interpreter with the agent's import graph (~150-190 MB
per open editor session). When the running gateway hosts ACP itself (``<home>/acp.sock``, served by
``acp_adapter.hosted``), ``hermes acp`` re-execs into this module: a stdlib-only ``python -I -S``
process (~10 MB) that pipes the editor's stdio to that socket.

The relay survives a gateway restart the way the in-process viewer did: requests in flight fail
with an "outcome unknown" error (never resent), and once the gateway answers again it replays the
editor's ``initialize`` and re-attaches every session it holds (``session/resume`` without a
transcript replay) before forwarding anything the editor sent meanwhile. Requests the gateway sends
the editor (permission cards) are renumbered so an answer to a card from a dead connection can never
land on a different card of the new one.

When no hosted listener answers (no socket, a stale one, Windows, a sticky profile,
``HERMES_ACP_HOSTED=0``), ``exec_relay_if_hosted`` returns and the classic viewer starts as before.
"""

from __future__ import annotations

import json
import os
import selectors
import socket
import sys
import time

SOCKET_NAME = "acp.sock"
HEADER_VERSION = 1
REATTACH_META = "hermesRelayReattach"
_PREFIX = "hermes-relay:"
_CHUNK = 64 * 1024
_MAX_QUEUED_BYTES = 50 * 1024 * 1024
LOST = ("The Hermes gateway restarted while this request was in flight; its outcome is unknown. "
        "Check the session before resending the work.")


def disabled(environ=None) -> bool:
    value = (os.environ if environ is None else environ).get("HERMES_ACP_HOSTED", "")
    return value.strip().lower() in {"0", "false", "no", "off"}


def connect(path: str) -> socket.socket | None:
    """A socket connected to the hosted ACP listener at ``path``, or None when nothing serves there."""
    if os.name == "nt" or not os.path.exists(path):
        return None
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(2)
        sock.connect(path)
        from_peer = _peer_uid(sock)
        if from_peer != os.getuid():  # windows-footgun: ok — connect returns above on Windows
            raise PermissionError("ACP listener uid mismatch")
        sock.settimeout(None)
    except OSError:  # a stale file from a stopped gateway, or a socket we may not use
        sock.close()
        return None
    return sock


def _peer_uid(sock):
    # Keep the re-exec relay stdlib-only; use the same kernel credentials as control discovery.
    if hasattr(socket, "SO_PEERCRED"):
        import struct
        return struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[1]
    if hasattr(sock, "getpeereid"):
        return sock.getpeereid()[0]
    if sys.platform == "darwin":
        import ctypes
        uid, gid = ctypes.c_uint(), ctypes.c_uint()
        if ctypes.CDLL(None).getpeereid(sock.fileno(), ctypes.byref(uid), ctypes.byref(gid)) == 0:
            return uid.value
    return None


def _parse(line: bytes):
    try:
        message = json.loads(line)
    except ValueError:
        return None
    return message if isinstance(message, dict) else None


def _frame(message: dict) -> bytes:
    return json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"


def _key(ident) -> str:
    return json.dumps(ident)


class Relay:
    """Line-oriented ACP relay between the editor (``stdin_fd``/``stdout_fd``) and the gateway socket."""

    def __init__(self, path: str, sock: socket.socket | None, *, environ=None, stdin_fd: int = 0,
                 stdout_fd: int = 1, reconnect_seconds: float = 300.0):
        env = os.environ if environ is None else environ
        self.header = _frame({"hermes_acp_relay": HEADER_VERSION, "cwd": os.getcwd(),
                              "env": {k: v for k, v in env.items() if k.startswith("HERMES_")}})
        self.path, self.sock, self.first = path, None, sock
        self.stdin_fd, self.stdout_fd = stdin_fd, stdout_fd
        self.reconnect_seconds = reconnect_seconds
        self.initialize = None   # the editor's initialize params, replayed on reconnect
        self.sessions = {}       # sessionId -> {"cwd", "mcpServers"} the editor attached with
        self.pending = {}        # editor request key -> [method, params, sent to the gateway?]
        self.replaying = set()   # relay request ids the current gateway connection has not answered
        self.backlog = []        # (editor request key | None, line) held until the (re)attach settles
        self.disconnected_at = None
        self.recovery_deadline = None
        self.queued_bytes = 0
        self.generation = 0      # gateway connection count
        self.next_agent_id = 0   # editor-facing ids of gateway -> editor requests
        self.agent_requests = {}  # editor-facing id -> (generation, the gateway's own id)
        self.editor_buf, self.gateway_buf = bytearray(), bytearray()
        self.selector = selectors.DefaultSelector()
        self.selector.register(stdin_fd, selectors.EVENT_READ, "editor")

    # -- io -----------------------------------------------------------------------------------
    def _to_editor(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            view = view[os.write(self.stdout_fd, view):]

    @staticmethod
    def _lines(buf: bytearray, chunk: bytes):
        buf += chunk
        while (end := buf.find(b"\n")) >= 0:
            line = bytes(buf[:end + 1])
            del buf[:end + 1]
            yield line

    def _send(self, key, line: bytes) -> None:
        if self.sock is None or self.replaying:
            if self.queued_bytes + len(line) > _MAX_QUEUED_BYTES:
                raise RuntimeError("ACP reconnect queue exceeded its byte limit")
            self.backlog.append((key, line))
            self.queued_bytes += len(line)
            return
        if key in self.pending:
            self.pending[key][2] = True
        try:
            self.sock.sendall(line)
        except OSError:
            # A partial write is ambiguous too: fail sent work, never put it back in the queue.
            self.lost()

    # -- editor -> gateway --------------------------------------------------------------------
    def on_editor_line(self, line: bytes) -> None:
        message = _parse(line)
        key = None
        if message is not None and "id" in message:
            if "method" in message:  # an editor request
                key = _key(message["id"])
                params = message.get("params") if isinstance(message.get("params"), dict) else {}
                self.pending[key] = [message["method"], params, False]
                if message["method"] == "initialize":
                    self.initialize = params
            else:  # the editor's answer to a gateway request: restore the gateway's id
                origin = self.agent_requests.pop(_key(message["id"]), None)
                if origin is None or origin[0] != self.generation:
                    return  # answers a card of a connection that is gone
                line = _frame({**message, "id": origin[1]})
                if self.sock is not None:
                    try:
                        self.sock.sendall(line)
                    except OSError:
                        self.lost()
                return  # responses belong to this connection, never the reconnect backlog
        self._send(key, line)

    # -- gateway -> editor --------------------------------------------------------------------
    def on_gateway_line(self, line: bytes) -> None:
        message = _parse(line)
        if message is not None and "id" in message:
            ident = message["id"]
            if "method" in message:  # a gateway request: give it an editor-facing id
                self.next_agent_id += 1
                self.agent_requests[_key(self.next_agent_id)] = (self.generation, ident)
                line = _frame({**message, "id": self.next_agent_id})
            elif isinstance(ident, str) and ident in self.replaying:
                self._settle_replay(ident, message)
                return
            else:
                method, params, _ = self.pending.pop(_key(ident), (None, None, None))
                if method and isinstance(message.get("result"), dict):
                    self._track(method, params, message["result"])
        self._to_editor(line)

    def _track(self, method: str, params: dict, result: dict) -> None:
        if method == "session/close":
            self.sessions.pop(params.get("sessionId"), None)
            return
        if method in ("session/new", "session/fork"):
            session_id = result.get("sessionId")
        elif method in ("session/load", "session/resume"):
            session_id = params.get("sessionId")
        else:
            return
        if isinstance(session_id, str) and session_id:
            self.sessions[session_id] = {"cwd": params.get("cwd") or os.getcwd(),
                                         "mcpServers": params.get("mcpServers") or []}

    def _settle_replay(self, ident: str, message: dict) -> None:
        self.replaying.discard(ident)
        if "error" in message:
            raise RuntimeError("Gateway refused ACP reattachment; work was not resubmitted")
        if ident == _PREFIX + "init":
            replay = [(f"{_PREFIX}resume:{sid}", "session/resume",
                       {"sessionId": sid, "cwd": held["cwd"], "mcpServers": held["mcpServers"],
                        "_meta": {REATTACH_META: True}}) for sid, held in self.sessions.items()]
            self.replaying.update(replay_id for replay_id, _, _ in replay)
            try:
                for replay_id, method, params in replay:
                    self.sock.sendall(_frame({"jsonrpc": "2.0", "id": replay_id, "method": method, "params": params}))
            except OSError:
                self.lost()
                return
        if not self.replaying:
            self.recovery_deadline = None
            self._flush()

    def _flush(self) -> None:
        backlog, self.backlog = self.backlog, []
        self.queued_bytes = 0
        for key, line in backlog:
            self._send(key, line)

    # -- gateway (re)attach -------------------------------------------------------------------
    def lost(self) -> None:
        """The gateway connection ended: fail what it was given, keep what it never saw."""
        self.selector.unregister(self.sock)
        self.sock.close()
        self.sock = None
        self.disconnected_at = time.monotonic()
        if self.recovery_deadline is None:
            self.recovery_deadline = self.disconnected_at + self.reconnect_seconds
        self.gateway_buf.clear()
        self.replaying.clear()
        self.agent_requests.clear()
        for key in [key for key, (_, _, sent) in self.pending.items() if sent]:
            del self.pending[key]
            self._to_editor(_frame({"jsonrpc": "2.0", "id": json.loads(key),
                                    "error": {"code": -32603, "message": LOST}}))

    def attach(self, sock: socket.socket) -> None:
        self.disconnected_at = None
        self.sock = sock
        self.generation += 1
        self.selector.register(sock, selectors.EVENT_READ, "gateway")
        try:
            sock.sendall(self.header)
        except OSError:
            self.lost()
            return
        replay = []
        # Nothing to restore before the editor's own initialize (first attach, or one still queued).
        queued_init = any(method == "initialize" for method, _, sent in self.pending.values() if not sent)
        if self.initialize is not None and not queued_init:
            replay.append((_PREFIX + "init", "initialize", self.initialize))
        self.replaying = {ident for ident, _, _ in replay}
        try:
            for ident, method, params in replay:
                sock.sendall(_frame({"jsonrpc": "2.0", "id": ident, "method": method, "params": params}))
        except OSError:
            self.lost()
            return
        if not replay:
            self._flush()

    def _check_recovery_deadline(self):
        if self.recovery_deadline is not None and time.monotonic() >= self.recovery_deadline:
            raise RuntimeError("ACP gateway reattachment timed out; work was not resubmitted")

    # -- loop ---------------------------------------------------------------------------------
    def run(self) -> int:
        self.disconnected_at = time.monotonic()
        if self.first is not None:
            self.attach(self.first)
        while True:
            self._check_recovery_deadline()
            if self.sock is None:
                if time.monotonic() - self.disconnected_at > self.reconnect_seconds:
                    print("hermes acp: the Hermes gateway did not come back; closing this ACP session",
                          file=sys.stderr)
                    return 1
                if (sock := connect(self.path)) is not None:
                    self.attach(sock)
            for key, _ in self.selector.select(timeout=1.0 if self.recovery_deadline is not None or not self.sock else None):
                if key.data == "editor":
                    chunk = os.read(self.stdin_fd, _CHUNK)
                    if not chunk:  # the editor closed the session
                        if self.sock is not None:
                            self.sock.close()
                        return 0
                    for line in self._lines(self.editor_buf, chunk):
                        self.on_editor_line(line)
                else:
                    if self.sock is None:
                        break
                    try:
                        chunk = self.sock.recv(_CHUNK)
                    except OSError:
                        chunk = b""
                    if not chunk:
                        self.lost()
                        break
                    for line in self._lines(self.gateway_buf, chunk):
                        self.on_gateway_line(line)
                        if self.sock is None:
                            break


def relay_home() -> str | None:
    """The Hermes home a bare ``hermes acp`` serves when it is knowable without the CLI's profile
    machinery: an explicit profile home, or the default home with no sticky profile and no container
    routing. None otherwise (the classic viewer resolves the rest)."""
    from hermes_cli import _startup_fast as fast
    home = os.environ.get("HERMES_HOME", "").strip()
    if home and os.path.basename(os.path.dirname(os.path.normpath(home))) == "profiles":
        return home
    if fast.container_mode_may_be_active():
        return None
    root = home or os.path.join(os.path.expanduser("~"), ".hermes")
    return None if fast.active_profile_may_override_home(root) else root


def exec_relay_if_hosted() -> None:
    """Replace this ``hermes acp`` process with the stdlib-only relay when the gateway hosts ACP for
    its home (the connected socket rides the exec); return and let the classic viewer start otherwise."""
    home = None if disabled() or os.environ.get("HERMES_TUI_GATEWAY_URL", "").strip() else relay_home()
    if not home:
        return
    path = os.path.join(home, SOCKET_NAME)
    sock = connect(path)
    if sock is None:
        return
    sock.set_inheritable(True)
    os.execv(sys.executable, [sys.executable, "-I", "-S", os.path.abspath(__file__), path, str(sock.fileno())])


def main(argv: list[str]) -> int:
    path = argv[0]
    sock = socket.socket(fileno=int(argv[1])) if len(argv) > 1 else connect(path)
    try:
        return Relay(path, sock).run()
    except (BrokenPipeError, KeyboardInterrupt):  # the editor went away mid-write
        return 0
    except RuntimeError as exc:
        print(f"hermes acp: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
