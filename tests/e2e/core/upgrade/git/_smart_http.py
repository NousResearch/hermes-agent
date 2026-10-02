"""A real git smart-HTTP origin on loopback, with faults the test arms on command.

``git http-backend`` (the CGI git itself ships) runs behind a small threaded HTTP/1.1 server, so
the client under test speaks the real wire protocol (v2 over HTTP, partial-clone filters, shallow
negotiation, thin packs) exactly as it does against GitHub; a ``file://`` or ``--shared`` origin
never exercises any of that. The bare repositories it serves get ``uploadpack.allowFilter`` and
``uploadpack.allowAnySHA1InWant``.

Faults (``Fault``) are matched per request, in arming order, and consumed:

* ``status``: answer with that HTTP status (429/500/502/503...) before the backend runs;
* ``cut_after``: stream that many body bytes of the real response, then drop the connection
  without the terminating chunk (curl: "transfer closed with outstanding read data");
* ``stall_after``: stream that many body bytes, then go silent until released or ``stall_max``.

A byte offset only means something at a fixed pkt-line framing. ``upload-pack`` relays whatever
``pack-objects`` has buffered, so the same pack arrives as 8 KiB sideband packets when it keeps up
and as 65520-byte ones when it lags (a loaded CI runner). The client acts only on whole pkt-lines:
a cut inside the first, coalesced packet means it never sees the ``PACK`` header and never starts
``index-pack``. So a cut/stall response has its sideband-1 pack data re-framed to the unhurried
``_FRAME``-byte packets first (a valid framing; the pack bytes are untouched), and the request log
records how many pack bytes the client got in whole packets before the drop.

Every request is recorded (``requests``): method, path, v2 command, status, body bytes sent and
whether a fault fired, so a cell can bound what a retry downloaded.
"""

from __future__ import annotations

import gzip
import os
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

_CHUNK = 64 * 1024
_FRAME = 8192  # sideband-1 payload per pkt-line that an unhurried upload-pack emits


@dataclass
class Fault:
    """One armed fault. ``path_has`` / ``command`` select the request; ``times`` how often it fires."""

    status: int | None = None
    cut_after: int | None = None
    stall_after: int | None = None
    stall_max: float = 600.0
    path_has: str = "git-upload-pack"
    method: str | None = None
    command: str | None = None  # protocol-v2 command in the request body ("fetch", "ls-refs")
    times: int = 1
    skip: int = 0  # let this many matching requests through untouched first
    fired: int = 0
    seen: int = 0

    def matches(self, method: str, path: str, command: str | None) -> bool:
        if self.fired >= self.times:
            return False
        if self.path_has and self.path_has not in path:
            return False
        if self.method and self.method != method:
            return False
        return self.command is None or self.command == command


@dataclass
class Request:
    method: str
    path: str
    command: str | None
    status: int = 0
    body_bytes: int = 0
    fault: str = ""
    pack_bytes: int = -1  # faulted responses: pack payload delivered in whole pkt-lines before the drop
    at: float = field(default_factory=time.monotonic)


def _v2_command(body: bytes) -> str | None:
    """First ``command=<name>`` pkt-line of a protocol-v2 request body."""
    i = 0
    while i + 4 <= len(body):
        try:
            n = int(body[i:i + 4], 16)
        except ValueError:
            return None
        if n < 4:
            i += 4
            continue
        line = body[i + 4:i + n].rstrip(b"\n")
        if line.startswith(b"command="):
            return line[len(b"command="):].decode("ascii", "replace")
        i += n
    return None


class _Reframer:
    """Re-frame a streamed upload-pack response so every sideband-1 pkt-line after the v2
    ``packfile`` section header carries at most ``_FRAME`` bytes. Everything else (section
    headers, progress, flush/delim packets) passes through unchanged; a body that is not pkt-lines
    passes through raw. Records where each pack pkt ends in the output, for ``pack_bytes_before``."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._in_pack = False
        self._raw = False
        self._emitted = 0
        self._pack_ends: list[tuple[int, int]] = []  # (end offset in output, pack payload bytes)

    def feed(self, data: bytes) -> bytes:
        if self._raw:
            self._emitted += len(data)
            return data
        self._buf += data
        out = bytearray()
        while len(self._buf) >= 4:
            try:
                n = int(bytes(self._buf[:4]), 16)
            except ValueError:
                self._raw = True
                out += self._buf
                self._buf.clear()
                break
            if n < 4:  # flush / delim / response-end
                out += self._buf[:4]
                del self._buf[:4]
                continue
            if len(self._buf) < n:
                break
            pkt = bytes(self._buf[4:n])
            del self._buf[:n]
            if self._in_pack and pkt[:1] == b"\x01":
                for i in range(1, len(pkt), _FRAME):
                    piece = pkt[i:i + _FRAME]
                    out += b"%04x\x01" % (len(piece) + 5) + piece
                    self._pack_ends.append((self._emitted + len(out), len(piece)))
            else:
                if pkt.rstrip(b"\n") == b"packfile":
                    self._in_pack = True
                out += b"%04x" % n + pkt
        self._emitted += len(out)
        return bytes(out)

    def tail(self) -> bytes:
        """An incomplete trailing pkt at EOF, passed through as the backend sent it."""
        rest = bytes(self._buf)
        self._buf.clear()
        self._emitted += len(rest)
        return rest

    def pack_bytes_before(self, offset: int) -> int:
        return sum(size for end, size in self._pack_ends if end <= offset)


class GitHTTPServer:
    """Serve bare repositories under ``root`` at ``http://127.0.0.1:<port>/<name>``."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.faults: list[Fault] = []
        self.requests: list[Request] = []
        self._lock = threading.Lock()
        self._release = threading.Event()
        self._git = shutil.which("git") or "git"
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_a):  # quiet
                pass

            def do_GET(self):
                server._handle(self, b"")

            def do_POST(self):
                server._handle(self, self._read_body())

            def _read_body(self) -> bytes:
                if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                    out = bytearray()
                    while True:
                        size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                        if size == 0:
                            self.rfile.readline()
                            return bytes(out)
                        out += self.rfile.read(size)
                        self.rfile.readline()
                return self.rfile.read(int(self.headers.get("Content-Length") or 0))

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    # -- lifecycle -------------------------------------------------------------------------
    def __enter__(self) -> GitHTTPServer:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._release.set()
        self._httpd.shutdown()
        self._httpd.server_close()

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    def url(self, name: str) -> str:
        return f"http://127.0.0.1:{self.port}/{name}"

    # -- test controls ---------------------------------------------------------------------
    def arm(self, fault: Fault) -> Fault:
        with self._lock:
            self.faults.append(fault)
        self._release.clear()
        return fault

    def clear_faults(self) -> None:
        with self._lock:
            self.faults.clear()
        self._release.set()

    def mark(self) -> int:
        """Index into ``requests`` to slice what happened after this point."""
        with self._lock:
            return len(self.requests)

    def since(self, mark: int) -> list[Request]:
        with self._lock:
            return list(self.requests[mark:])

    def describe(self, mark: int = 0) -> str:
        return "\n".join(f"  {r.method} {r.path} cmd={r.command} -> {r.status} {r.body_bytes}B {r.fault}"
                         + (f" (pack bytes in whole pkt-lines: {r.pack_bytes})" if r.pack_bytes >= 0 else "")
                         for r in self.since(mark)) or "  (no requests)"

    # -- request handling ------------------------------------------------------------------
    def _take_fault(self, method: str, path: str, command: str | None) -> Fault | None:
        with self._lock:
            for f in self.faults:
                if f.matches(method, path, command):
                    f.seen += 1
                    if f.seen <= f.skip:
                        continue
                    f.fired += 1
                    return f
        return None

    def _handle(self, h: BaseHTTPRequestHandler, raw_body: bytes) -> None:
        parts = urlsplit(h.path)
        body = raw_body
        if h.headers.get("Content-Encoding", "").lower() == "gzip" and raw_body:
            try:
                body = gzip.decompress(raw_body)
            except OSError:
                body = raw_body
        command = _v2_command(body) if h.command == "POST" else None
        rec = Request(h.command, parts.path, command)
        with self._lock:
            self.requests.append(rec)
        fault = self._take_fault(h.command, h.path, command)
        if fault is not None and fault.status is not None:
            rec.status, rec.fault = fault.status, f"status {fault.status}"
            msg = f"injected {fault.status}\n".encode()
            h.send_response(fault.status)
            h.send_header("Content-Type", "text/plain")
            h.send_header("Content-Length", str(len(msg)))
            if fault.status == 429:
                h.send_header("Retry-After", "1")
            h.end_headers()
            h.wfile.write(msg)
            return
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "GIT_PROJECT_ROOT": str(self.root),
            "GIT_HTTP_EXPORT_ALL": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": str(self.root),
            "REQUEST_METHOD": h.command,
            "PATH_INFO": parts.path,
            "QUERY_STRING": parts.query,
            "CONTENT_TYPE": h.headers.get("Content-Type", ""),
            "CONTENT_LENGTH": str(len(raw_body)),
            "REMOTE_ADDR": "127.0.0.1",
            "SERVER_PROTOCOL": "HTTP/1.1",
        }
        if h.headers.get("Git-Protocol"):
            env["GIT_PROTOCOL"] = h.headers["Git-Protocol"]
            env["HTTP_GIT_PROTOCOL"] = h.headers["Git-Protocol"]
        if h.headers.get("Content-Encoding"):
            env["HTTP_CONTENT_ENCODING"] = h.headers["Content-Encoding"]
        proc = subprocess.Popen([self._git, "http-backend"], env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        feeder = threading.Thread(target=self._feed, args=(proc, raw_body), daemon=True)
        feeder.start()
        try:
            self._relay(h, proc, rec, fault)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            feeder.join(timeout=5)

    @staticmethod
    def _feed(proc: subprocess.Popen, body: bytes) -> None:
        try:
            proc.stdin.write(body)
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    def _relay(self, h: BaseHTTPRequestHandler, proc: subprocess.Popen, rec: Request, fault: Fault | None) -> None:
        out = proc.stdout
        status, headers = 200, []
        while True:
            line = out.readline()
            if not line or line in (b"\r\n", b"\n"):
                break
            name, _, value = line.decode("latin-1").partition(":")
            name, value = name.strip(), value.strip()
            if name.lower() == "status":
                status = int(value.split()[0])
            else:
                headers.append((name, value))
        rec.status = status
        h.send_response(status)
        for name, value in headers:
            if name.lower() not in ("content-length", "transfer-encoding", "connection"):
                h.send_header(name, value)
        h.send_header("Transfer-Encoding", "chunked")
        h.end_headers()
        limit = None
        if fault is not None:
            limit = fault.cut_after if fault.cut_after is not None else fault.stall_after
        framer = _Reframer() if limit is not None else None
        sent, pending = 0, b""
        try:
            while True:
                if not pending:
                    data = out.read1(_CHUNK)
                    if not data:
                        pending = framer.tail() if framer else b""
                        if not pending:
                            break
                    else:
                        pending = framer.feed(data) if framer else data
                        if not pending:
                            continue
                take = pending if limit is None else pending[:max(0, limit - sent)]
                pending = pending[len(take):]
                if take:
                    h.wfile.write(b"%x\r\n%s\r\n" % (len(take), take))
                    sent += len(take)
                if limit is not None and sent >= limit:
                    h.wfile.flush()
                    rec.body_bytes = sent
                    rec.pack_bytes = framer.pack_bytes_before(sent)
                    if fault.cut_after is not None:
                        rec.fault = f"cut after {sent}B"
                        self._abort(h)
                        return
                    rec.fault = f"stalled after {sent}B"
                    self._release.wait(timeout=fault.stall_max)
                    self._abort(h)
                    return
            h.wfile.write(b"0\r\n\r\n")
            h.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            rec.fault = rec.fault or "client went away"
        rec.body_bytes = sent
        if framer is not None:
            rec.pack_bytes = framer.pack_bytes_before(sent)

    @staticmethod
    def _abort(h: BaseHTTPRequestHandler) -> None:
        h.close_connection = True
        try:
            h.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def serve_bare(root: Path, name: str, source: Path, ref: str) -> Path:
    """Create ``root/name``: a bare origin with only ``main`` (at ``ref``) and the release tags
    already merged into it, like the official repository looked when ``ref`` was its tip. It is
    configured for partial clone and any-SHA wants. The server borrows ``source``'s objects
    (``--shared``: no copy); the client still receives real packs over HTTP.

    ``HERMES_E2E_GIT_OBJECTS`` may name an extra complete object store: a developer checkout that
    is itself a partial clone lacks old blobs a full clone needs (CI's full checkout does not)."""
    bare = root / name
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True, capture_output=True)
    stores = [Path(subprocess.run(["git", "-C", str(source), "rev-parse", "--path-format=absolute",
                                   "--git-common-dir"], check=True, capture_output=True, text=True)
                   .stdout.strip()) / "objects"]
    extra = os.environ.get("HERMES_E2E_GIT_OBJECTS")
    if extra:
        stores.append(Path(extra))
    (bare / "objects" / "info" / "alternates").write_text("".join(f"{s}\n" for s in stores), encoding="utf-8")
    tags = subprocess.run(["git", "-C", str(source), "for-each-ref", "--merged", ref,
                           "--format=%(objectname) %(refname)", "refs/tags/v*"],
                          check=True, capture_output=True, text=True).stdout.splitlines()
    stdin = f"update refs/heads/main {ref}\n" + "".join(
        f"update {line.split()[1]} {line.split()[0]}\n" for line in tags)
    subprocess.run(["git", "-C", str(bare), "update-ref", "--stdin"], input=stdin, text=True, check=True,
                   capture_output=True)
    for args in (["symbolic-ref", "HEAD", "refs/heads/main"],
                 ["config", "uploadpack.allowFilter", "true"],
                 ["config", "uploadpack.allowAnySHA1InWant", "true"],
                 ["config", "http.receivepack", "false"]):
        subprocess.run(["git", "-C", str(bare), *args], check=True, capture_output=True)
    return bare
