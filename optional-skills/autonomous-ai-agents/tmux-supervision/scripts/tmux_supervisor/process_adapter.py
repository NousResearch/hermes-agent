"""Observe real command start/exit without intercepting its terminal or input."""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import selectors
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import supervision

MAX_COMMAND = 1024 * 1024
MAX_CLIENTS = 8
MAX_EVENTS = 128
MAX_PENDING = supervision.MAX_FRAME * (MAX_EVENTS + 1)
HANDSHAKE_TIMEOUT = 5.0
DRAIN_TIMEOUT = 1.0


def _command(argv, *, executable=None, workspace=None):
    supervision._require(
        isinstance(argv, list)
        and argv
        and all(isinstance(arg, str) and "\0" not in arg for arg in argv),
        "invalid_command",
    )
    try:
        size = sum(len(os.fsencode(arg)) + 1 for arg in argv)
    except UnicodeError as exc:
        raise supervision.TUIError("invalid_command") from exc
    supervision._require(size <= MAX_COMMAND, "command_too_large")
    name = argv[0]
    supervision._require(supervision._text(name) and name.strip(), "invalid_executable")
    if executable is not None:
        supervision._require(Path(executable).is_absolute(), "invalid_executable")
        name = executable
    elif workspace is not None and not os.path.isabs(name):
        if "/" in name:
            name = str(Path(workspace) / name)
        else:
            search_path = os.pathsep.join(
                str(Path(workspace) / entry) for entry in os.get_exec_path()
            )
            name = shutil.which(name, path=search_path)
            supervision._require(name is not None, "invalid_executable")
    return [supervision._executable(name), *argv[1:]]


def _command_bytes(command_file):
    try:
        return supervision._read_bytes(
            Path(command_file), private=False, limit=MAX_COMMAND
        )
    except OSError as exc:
        raise supervision.TUIError("invalid_command_file") from exc


def _parse_command(content, *, executable=None, workspace=None):
    try:
        argv = json.loads(content)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise supervision.TUIError("invalid_command") from exc
    return _command(argv, executable=executable, workspace=workspace)


def read_command(command_file, expected_sha256=None, *, executable=None) -> list[str]:
    """Read bounded argv; reject a changed launch file before parsing or resolving it."""
    content = _command_bytes(command_file)
    if expected_sha256 is not None:
        supervision._require(
            isinstance(expected_sha256, str)
            and len(expected_sha256) == 64
            and all(character in "0123456789abcdef" for character in expected_sha256),
            "invalid_command_digest",
        )
        supervision._require(
            hmac.compare_digest(hashlib.sha256(content).hexdigest(), expected_sha256),
            "command_changed",
        )
    return _parse_command(content, executable=executable)


def launch(run_dir, command_file, tmux_executable="tmux"):
    run, binding = supervision._binding(run_dir)
    supervision._require(binding["adapter"] == "command", "adapter_mismatch")
    content = _command_bytes(command_file)
    argv = _parse_command(content, workspace=binding["workspace"])
    try:
        command_file = Path(command_file).resolve(strict=True)
    except OSError as exc:
        raise supervision.TUIError("invalid_command_file") from exc
    runner = Path(__file__).resolve().parents[1] / "tmux_supervise.py"
    return supervision.launch(
        run,
        [
            sys.executable,
            str(runner),
            "_run-command",
            "--run-dir",
            str(run),
            "--command-file",
            str(command_file),
            "--executable",
            argv[0],
            "--command-sha256",
            hashlib.sha256(content).hexdigest(),
        ],
        tmux_executable=tmux_executable,
    )


def _frame(value):
    encoded = (json.dumps(value, separators=(",", ":")) + "\n").encode()
    supervision._require(len(encoded) <= supervision.MAX_FRAME, "frame_too_large")
    return encoded


@dataclass
class _Subscriber:
    socket: socket.socket
    deadline: float = field(
        default_factory=lambda: time.monotonic() + HANDSHAKE_TIMEOUT
    )
    incoming: bytearray = field(default_factory=bytearray)
    outgoing: bytearray = field(default_factory=bytearray)
    after_seq: int | None = None
    rejected: bool = False


class _Bridge:
    """Bounded, observation-only transport; failures never own the child lifetime."""

    def __init__(self, run):
        self.run = run
        self.path = run / "bridge.sock"
        supervision._require(
            not os.path.lexists(self.path)
            and not os.path.lexists(run / "journal.json"),
            "producer_already_exists",
        )
        self.journal = {
            "version": 2,
            "run_id": run.name,
            "epoch": uuid.uuid4().hex,
            "app_session_id": uuid.uuid4().hex,
            "pid": os.getpid(),
            "seq": 0,
            "state": "idle",
            "events": [],
        }
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.clients = {}
        self.worker = None
        self.identity = None
        self.resources = contextlib.ExitStack()
        try:
            self.selector = self.resources.enter_context(selectors.DefaultSelector())
            self.listener = self.resources.enter_context(
                socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            )
            self.listener.bind(str(self.path))
            info = self.path.lstat()
            self.identity = (info.st_dev, info.st_ino)
            self.path.chmod(0o600)
            self.listener.setblocking(False)
            self.listener.listen(MAX_CLIENTS)
            reader, writer = socket.socketpair()
            self.reader = self.resources.enter_context(reader)
            self.writer = self.resources.enter_context(writer)
            reader.setblocking(False)
            writer.setblocking(False)
            self.selector.register(self.listener, selectors.EVENT_READ)
            self.selector.register(reader, selectors.EVENT_READ)
            supervision._write_json(run / "journal.json", self.journal, exclusive=True)
        except BaseException:
            self._cleanup()
            raise

    def start(self):
        self.worker = threading.Thread(target=self._serve, daemon=True)
        self.worker.start()

    def _wake(self):
        # Wakeups coalesce; the durable snapshot, not the socket, holds events.
        with contextlib.suppress(OSError):
            self.writer.send(b"\0")

    def publish(self, kind, *, exit_code=None):
        with self.lock:
            event = {
                key: self.journal[key]
                for key in ("version", "run_id", "epoch", "app_session_id")
            }
            event.update(
                type="event",
                seq=self.journal["seq"] + 1,
                kind=kind,
                at_ms=int(time.time() * 1000),
            )
            if kind == "process_exited":
                supervision._require(
                    type(exit_code) is int and -255 <= exit_code <= 255,
                    "invalid_exit_code",
                )
                event["exit_code"] = exit_code
            journal = {
                **self.journal,
                "seq": event["seq"],
                "state": "closed" if kind == "process_exited" else "busy",
                "events": [*self.journal["events"], event][-MAX_EVENTS:],
            }
            # No client can see a transition until its journal commit succeeds.
            supervision._write_json(self.run / "journal.json", journal)
            self.journal = journal
        self._wake()

    def _drop(self, client):
        self.clients.pop(client.socket, None)
        self.selector.unregister(client.socket)
        client.socket.close()

    def _queue(self, client, frames):
        encoded = b"".join(_frame(value) for value in frames)
        if len(client.outgoing) + len(encoded) > MAX_PENDING:
            self._drop(client)
            return
        client.outgoing.extend(encoded)
        if client.outgoing:
            self.selector.modify(
                client.socket, selectors.EVENT_READ | selectors.EVENT_WRITE, client
            )

    def _accept(self):
        connection, _ = self.listener.accept()
        try:
            _, uid, _ = struct.unpack(
                "3i",
                connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12),
            )
            if uid != os.getuid() or len(self.clients) >= MAX_CLIENTS:
                connection.close()
                return
            connection.setblocking(False)
            client = _Subscriber(connection)
            self.selector.register(connection, selectors.EVENT_READ, client)
            self.clients[connection] = client
        except OSError:
            connection.close()

    def _reject(self, client, reason):
        client.rejected = True
        self._queue(
            client,
            [
                {
                    "version": 2,
                    "type": "rejected",
                    "run_id": self.run.name,
                    "reason": reason,
                }
            ],
        )

    def _receive(self, client):
        chunk = client.socket.recv(supervision.MAX_FRAME + 1 - len(client.incoming))
        if not chunk or client.after_seq is not None or client.rejected:
            self._drop(client)
            return
        client.incoming.extend(chunk)
        if len(client.incoming) > supervision.MAX_FRAME:
            self._drop(client)
            return
        if b"\n" not in client.incoming:
            return
        line, _, extra = client.incoming.partition(b"\n")
        if extra:
            self._drop(client)
            return
        request = supervision._decode(line)
        if (
            set(request)
            not in (
                {"version", "type", "run_id", "after_seq"},
                {"version", "type", "run_id", "after_seq", "epoch"},
            )
            or type(request["version"]) is not int
            or request["version"] != 2
            or request["type"] != "observe"
            or request["run_id"] != self.run.name
        ):
            self._drop(client)
            return
        with self.lock:
            journal = self.journal
        hello = {key: value for key, value in journal.items() if key != "events"}
        self._queue(client, [{**hello, "type": "hello"}])
        if client.socket not in self.clients:
            return
        if (
            not supervision._integer(request["after_seq"])
            or request["after_seq"] > journal["seq"]
            or (request["after_seq"] > 0 and "epoch" not in request)
        ):
            self._reject(client, "invalid_cursor")
            return
        if "epoch" in request and request["epoch"] != journal["epoch"]:
            self._reject(client, "epoch_mismatch")
            return
        if journal["events"] and request["after_seq"] < journal["events"][0]["seq"] - 1:
            self._reject(client, "invalid_cursor")
            return
        self._queue(
            client,
            [
                event
                for event in journal["events"]
                if event["seq"] > request["after_seq"]
            ],
        )
        client.incoming.clear()
        client.after_seq = journal["seq"]

    def _send(self, client):
        sent = client.socket.send(client.outgoing)
        if not sent:
            self._drop(client)
            return
        del client.outgoing[:sent]
        if not client.outgoing:
            if client.rejected:
                self._drop(client)
            else:
                self.selector.modify(client.socket, selectors.EVENT_READ, client)

    def _serve(self):
        drain_deadline = None
        try:
            while True:
                now = time.monotonic()
                with self.lock:
                    journal = self.journal
                    stopping = self.stopping.is_set()
                for client in list(self.clients.values()):
                    if client.after_seq is None:
                        if now >= client.deadline:
                            self._drop(client)
                    elif client.after_seq < journal["seq"]:
                        events = [
                            event
                            for event in journal["events"]
                            if event["seq"] > client.after_seq
                        ]
                        if events[0]["seq"] != client.after_seq + 1:
                            self._drop(client)
                        else:
                            self._queue(client, events)
                            client.after_seq = journal["seq"]
                if stopping and drain_deadline is None:
                    drain_deadline = now + DRAIN_TIMEOUT
                    self.selector.unregister(self.listener)
                    self.listener.close()
                if drain_deadline is not None and (
                    now >= drain_deadline
                    or not any(c.outgoing for c in self.clients.values())
                ):
                    break
                deadlines = [
                    c.deadline for c in self.clients.values() if c.after_seq is None
                ]
                if drain_deadline is not None:
                    deadlines.append(drain_deadline)
                timeout = max(0, min(deadlines) - now) if deadlines else None
                for key, mask in self.selector.select(timeout):
                    if key.fileobj is self.reader:
                        self.reader.recv(4096)
                    elif key.fileobj is self.listener:
                        self._accept()
                    else:
                        client = key.data
                        try:
                            if mask & selectors.EVENT_READ:
                                self._receive(client)
                            if (
                                mask & selectors.EVENT_WRITE
                                and client.socket in self.clients
                            ):
                                self._send(client)
                        except BlockingIOError:
                            continue
                        except (OSError, supervision.TUIError, RecursionError):
                            if client.socket in self.clients:
                                self._drop(client)
        except Exception:
            # Socket/persistence supervision is not authority to stop the command.
            # The observer can recover committed events from the journal.
            pass
        finally:
            self._cleanup()

    def _cleanup(self):
        for client in self.clients.values():
            client.socket.close()
        self.clients.clear()
        self.resources.close()
        if self.identity is not None:
            with contextlib.suppress(OSError):
                info = self.path.lstat()
                if (info.st_dev, info.st_ino) == self.identity:
                    self.path.unlink()

    def close(self):
        self.stopping.set()
        self._wake()
        if self.worker is not None and self.worker.ident is not None:
            self.worker.join(DRAIN_TIMEOUT + 1)
        else:
            self._cleanup()


@contextlib.contextmanager
def _terminal_signals():
    # A caught (not ignored) disposition resets at exec. The terminal delivers
    # keyboard signals directly to the child in our shared foreground group;
    # Python must not unwind the wrapper while that child handles Ctrl-C/Ctrl-\.
    previous = {}
    try:
        for number in (signal.SIGINT, signal.SIGQUIT):
            previous[number] = signal.getsignal(number)
            if previous[number] != signal.SIG_IGN:
                signal.signal(number, lambda *_: None)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def run_command(run_dir, argv) -> int:
    run, binding = supervision._read_binding(run_dir)
    supervision._require(binding["adapter"] == "command", "adapter_mismatch")
    argv = _command(argv, workspace=binding["workspace"])
    with _terminal_signals():
        bridge = _Bridge(run)
        try:
            bridge.start()
            try:
                child = subprocess.Popen(
                    argv,
                    cwd=binding["workspace"],
                    stdin=None,
                    stdout=None,
                    stderr=None,
                    close_fds=True,
                )
            except OSError as exc:
                raise supervision.TUIError("command_start_failed") from exc
            try:
                bridge.publish("started")
            except Exception:
                # Never use Popen as a context manager here: its cleanup can
                # abandon the wait if monitoring raises during an interactive run.
                with contextlib.suppress(Exception):
                    bridge.close()
                bridge = None
            while True:
                try:
                    exit_code = child.wait()
                    break
                except KeyboardInterrupt:
                    continue
            if bridge is not None:
                with contextlib.suppress(Exception):
                    bridge.publish("process_exited", exit_code=exit_code)
            return exit_code
        finally:
            if bridge is not None:
                with contextlib.suppress(Exception):
                    bridge.close()
