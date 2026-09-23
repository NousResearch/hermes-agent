"""Session-owned, one-wake observation of an explicitly launched OMP TUI.

The cursor records readiness for Hermes' native background notification, never
platform delivery. No operation sends input to, or terminates, an OMP process.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
import re
import shlex
import shutil
import socket
import stat
import struct
import subprocess
import sys
import time
import uuid
from pathlib import Path

EXTENSION = Path(__file__).with_name("tui_extension.ts")
THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max", "auto")
MAX_FRAME = 4096
MAX_JSON = 65536
MAX_RECONNECTS = 64
KINDS = frozenset({
    "started",
    "turn_settled",
    "needs_input",
    "error",
    "session_revoked",
    "shutdown",
    "observation_lost",
})
STATES = frozenset({"idle", "busy", "revoked", "closed"})
HEX = re.compile(r"[0-9a-f]{32}\Z")
SESSION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
PLATFORM = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


class TUIError(Exception):
    """A sanitized diagnostic, safe to emit to the owning conversation."""


def _require(condition, code):
    if not condition:
        raise TUIError(code)


def _integer(value, minimum=0):
    return type(value) is int and minimum <= value <= 2**53 - 1


def _text(value):
    return (
        isinstance(value, str)
        and 0 < len(value) <= 1024
        and not any(ord(c) < 32 for c in value)
    )


def _directory(path, *, private=False):
    path = Path(path).absolute()
    info = path.lstat()
    _require(stat.S_ISDIR(info.st_mode) and path == path.resolve(), "unsafe_directory")
    if private:
        _require(
            info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700,
            "unsafe_directory",
        )
    return path


def _file_info(info, *, private=True):
    _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "unsafe_file")
    if private:
        _require(
            info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o600,
            "unsafe_file",
        )


def _read_bytes(path, *, private=True, limit=MAX_JSON):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        _file_info(info, private=private)
        _require(info.st_size <= limit, "file_too_large")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            content = stream.read(limit + 1)
        _require(len(content) <= limit, "file_too_large")
        return content
    finally:
        os.close(fd)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate_json_field")
        result[key] = value
    return result


def _decode(content):
    try:
        result = json.loads(
            content,
            object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(TUIError("invalid_json")),
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise TUIError("invalid_json") from exc
    _require(isinstance(result, dict), "invalid_json")
    return result


def _read_json(path, *, optional=False):
    try:
        return _decode(_read_bytes(path))
    except FileNotFoundError:
        if optional:
            return None
        raise


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json(path, value, *, exclusive=False):
    """Atomic durable replace, or atomic create with no overwrite."""
    path = Path(path)
    _directory(path.parent, private=True)
    if not exclusive:
        try:
            _file_info(path.lstat())
        except FileNotFoundError:
            pass
    encoded = (
        json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    _require(len(encoded) <= MAX_JSON, "file_too_large")
    temporary = path.with_name("." + uuid.uuid4().hex + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path, follow_symlinks=False)
            temporary.unlink()
        else:
            os.replace(temporary, path)
        _sync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def current_owner():
    names = {
        "hermes_home": "HERMES_HOME",
        "platform": "HERMES_SESSION_PLATFORM",
        "session_key": "HERMES_SESSION_KEY",
        "session_id": "HERMES_SESSION_ID",
        "chat_id": "HERMES_SESSION_CHAT_ID",
        "thread_id": "HERMES_SESSION_THREAD_ID",
    }
    owner = {key: os.environ.get(name, "") for key, name in names.items()}
    _require(
        all(_text(value) for key, value in owner.items() if key != "thread_id"),
        "missing_native_scope",
    )
    _require(PLATFORM.fullmatch(owner["platform"]), "unsupported_platform")
    _require(
        owner["thread_id"] == "" or _text(owner["thread_id"]), "missing_native_scope"
    )
    _require("HERMES_SESSION_THREAD_ID" in os.environ, "missing_native_scope")
    home = Path(owner["hermes_home"])
    _require(home.is_absolute(), "missing_native_scope")
    owner["hermes_home"] = str(home.resolve(strict=True))
    _require(home.is_dir(), "missing_native_scope")
    return owner


def prepare(workspace, tmux_session, state_root=None, run_id=None):
    owner = current_owner()
    workspace = Path(workspace).resolve(strict=True)
    _require(workspace.is_dir(), "invalid_workspace")
    _require(
        isinstance(tmux_session, str) and SESSION_NAME.fullmatch(tmux_session),
        "invalid_tmux_session",
    )
    run_id = uuid.uuid4().hex if run_id is None else run_id
    _require(isinstance(run_id, str) and HEX.fullmatch(run_id), "invalid_run_id")
    root = (
        Path(owner["hermes_home"]) / "omp-supervisor"
        if state_root is None
        else Path(state_root).absolute()
    )
    if not root.exists():
        root.mkdir(mode=0o700)
    _directory(root, private=True)
    run = root / run_id
    _require(len(os.fsencode(run / "bridge.sock")) <= 107, "socket_path_too_long")
    try:
        run.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise TUIError("binding_already_exists") from exc
    binding = {
        "version": 1,
        "mode": "tui",
        "run_id": run_id,
        "workspace": str(workspace),
        "tmux_session": tmux_session,
        "owner": owner,
        "created_at": time.time(),
    }
    _write_json(run / "binding.json", binding, exclusive=True)
    _sync_dir(root)
    return {"run_id": run_id, "run_dir": str(run), "status": "prepared"}


def _binding(run_dir):
    owner = current_owner()
    run = _directory(run_dir, private=True)
    _directory(run.parent, private=True)
    _require(HEX.fullmatch(run.name), "invalid_run_id")
    _require(len(os.fsencode(run / "bridge.sock")) <= 107, "socket_path_too_long")
    binding = _read_json(run / "binding.json")
    _require(
        binding.get("version") == 1
        and type(binding.get("version")) is int
        and binding.get("mode") == "tui"
        and binding.get("run_id") == run.name,
        "invalid_binding",
    )
    _require(binding.get("owner") == owner, "scope_denied")
    _require(
        isinstance(binding.get("tmux_session"), str)
        and SESSION_NAME.fullmatch(binding["tmux_session"]),
        "invalid_binding",
    )
    _require(
        isinstance(binding.get("workspace"), str)
        and Path(binding["workspace"]).is_absolute(),
        "invalid_binding",
    )
    _require(
        str(_directory(binding["workspace"])) == binding["workspace"], "invalid_binding"
    )
    return run, binding


@contextlib.contextmanager
def observation_lock(run):
    fd = _open_lock(Path(run) / "observation.lock")
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise TUIError("observer_already_running") from exc
        yield
    finally:
        os.close(fd)


def _open_lock(path):
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        _file_info(os.fstat(fd))
    except BaseException:
        os.close(fd)
        raise
    return fd


def _observer_active(run):
    fd = _open_lock(run / "observation.lock")
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False
    finally:
        os.close(fd)


def _validate_message(message, run_id, kind):
    common = {"version", "type", "run_id", "epoch", "omp_session_id", "seq"}
    fields = common | ({"pid", "state"} if kind == "hello" else {"kind", "at_ms"})
    _require(set(message) == fields, "invalid_frame_fields")
    _require(
        type(message["version"]) is int
        and message["version"] == 1
        and message["type"] == kind
        and message["run_id"] == run_id,
        "invalid_frame",
    )
    _require(
        isinstance(message["epoch"], str)
        and HEX.fullmatch(message["epoch"])
        and _text(message["omp_session_id"])
        and _integer(message["seq"]),
        "invalid_frame",
    )
    if kind == "hello":
        _require(
            _integer(message["pid"], 1)
            and isinstance(message["state"], str)
            and message["state"] in STATES,
            "invalid_frame",
        )
    else:
        _require(
            isinstance(message["kind"], str)
            and message["kind"] in KINDS
            and _integer(message["seq"], 1)
            and _integer(message["at_ms"]),
            "invalid_frame",
        )
    return message


def _journal(run):
    journal = _read_json(run / "journal.json", optional=True)
    if journal is None:
        return None
    _require(
        set(journal)
        == {
            "version",
            "run_id",
            "epoch",
            "omp_session_id",
            "pid",
            "seq",
            "state",
            "events",
        },
        "invalid_journal",
    )
    hello = {key: value for key, value in journal.items() if key != "events"}
    _validate_message({**hello, "type": "hello"}, run.name, "hello")
    events = journal["events"]
    _require(isinstance(events, list) and len(events) <= 128, "invalid_journal")
    previous = None
    for event in events:
        _require(isinstance(event, dict), "invalid_journal")
        _validate_message(event, run.name, "event")
        _require(
            event["epoch"] == journal["epoch"]
            and event["omp_session_id"] == journal["omp_session_id"]
            and event["seq"] <= journal["seq"],
            "invalid_journal",
        )
        _require(previous is None or event["seq"] == previous + 1, "invalid_journal")
        previous = event["seq"]
    _require(
        (previous == journal["seq"]) if events else journal["seq"] == 0,
        "invalid_journal",
    )
    return journal


def _cursor(run):
    cursor = _read_json(run / "cursor.json", optional=True)
    if cursor is None:
        return {
            "version": 1,
            "run_id": run.name,
            "epoch": None,
            "omp_session_id": None,
            "seq": 0,
            "status": "unobserved",
            "terminal": False,
            "receipt": None,
        }
    _require(
        set(cursor)
        == {
            "version",
            "run_id",
            "epoch",
            "omp_session_id",
            "seq",
            "status",
            "terminal",
            "receipt",
        },
        "invalid_cursor",
    )
    _require(
        type(cursor["version"]) is int
        and cursor["version"] == 1
        and cursor["run_id"] == run.name
        and _integer(cursor["seq"])
        and type(cursor["terminal"]) is bool,
        "invalid_cursor",
    )
    _require(
        isinstance(cursor["status"], str)
        and cursor["status"]
        in {"unobserved", "observing", "ready_for_native_notification"},
        "invalid_cursor",
    )
    if cursor["epoch"] is None:
        _require(
            cursor["omp_session_id"] is None and cursor["seq"] == 0, "invalid_cursor"
        )
    else:
        _require(
            isinstance(cursor["epoch"], str)
            and HEX.fullmatch(cursor["epoch"])
            and _text(cursor["omp_session_id"]),
            "invalid_cursor",
        )
    receipt = cursor["receipt"]
    if receipt is not None:
        _require(
            isinstance(receipt, dict)
            and set(receipt)
            == {
                "version",
                "run_id",
                "status",
                "kind",
                "epoch",
                "seq",
                "at_ms",
                "reason",
            },
            "invalid_receipt",
        )
        _require(
            type(receipt["version"]) is int
            and receipt["version"] == 1
            and receipt["run_id"] == run.name
            and receipt["status"] == "ready_for_native_notification"
            and isinstance(receipt["kind"], str)
            and receipt["kind"] in KINDS
            and receipt["epoch"] == cursor["epoch"]
            and _integer(receipt["seq"])
            and receipt["seq"] <= cursor["seq"]
            and _integer(receipt["at_ms"]),
            "invalid_receipt",
        )
        _require(
            receipt["reason"] is None
            or (
                isinstance(receipt["reason"], str)
                and receipt["reason"]
                in {
                    "timeout",
                    "reconnect_exhausted",
                    "epoch_changed",
                    "session_changed",
                    "event_gap",
                    "sequence_regressed",
                    "invalid_protocol",
                    "observer_interrupted",
                    "closed_without_event",
                }
            ),
            "invalid_receipt",
        )
    return cursor


def _identity(cursor, message):
    if cursor["epoch"] is not None:
        _require(cursor["epoch"] == message["epoch"], "epoch_changed")
        _require(
            cursor["omp_session_id"] == message["omp_session_id"], "session_changed"
        )
    else:
        cursor["epoch"] = message["epoch"]
        cursor["omp_session_id"] = message["omp_session_id"]


def _receipt(run, cursor, kind, reason=None):
    receipt = {
        "version": 1,
        "run_id": run.name,
        "status": "ready_for_native_notification",
        "kind": kind,
        "epoch": cursor["epoch"],
        "seq": cursor["seq"],
        "at_ms": int(time.time() * 1000),
        "reason": reason,
    }
    cursor.update(
        status="ready_for_native_notification",
        receipt=receipt,
        # A finite deadline does not invalidate the captured identity or sequence.
        terminal=kind in {"session_revoked", "shutdown"}
        or (kind == "observation_lost" and reason != "timeout"),
    )
    _write_json(run / "cursor.json", cursor)
    return receipt


def _consume(run, cursor, event):
    _identity(cursor, event)
    if event["seq"] <= cursor["seq"]:
        return None
    _require(event["seq"] == cursor["seq"] + 1, "event_gap")
    cursor["seq"] = event["seq"]
    if event["kind"] != "started":
        return _receipt(run, cursor, event["kind"])
    _write_json(run / "cursor.json", cursor)
    return None


def _replay(run, cursor):
    journal = _journal(run)
    if journal is None:
        return None
    _identity(cursor, journal)
    _require(journal["seq"] >= cursor["seq"], "sequence_regressed")
    _write_json(run / "cursor.json", cursor)
    for event in journal["events"]:
        receipt = _consume(run, cursor, event)
        if receipt:
            return receipt
    _require(journal["state"] not in {"closed", "revoked"}, "closed_without_event")
    return None


def _frames(client, deadline):
    pending = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TUIError("timeout")
        client.settimeout(remaining)
        chunk = client.recv(MAX_FRAME + 1 - len(pending))
        if not chunk:
            raise ConnectionError("disconnected")
        pending.extend(chunk)
        while b"\n" in pending:
            line, _, rest = pending.partition(b"\n")
            _require(len(line) + 1 <= MAX_FRAME, "frame_too_large")
            pending = bytearray(rest)
            yield _decode(line)
        _require(len(pending) < MAX_FRAME, "frame_too_large")


def _connect(run, deadline):
    path = run / "bridge.sock"
    info = path.lstat()
    _require(
        stat.S_ISSOCK(info.st_mode)
        and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) == 0o600,
        "unsafe_socket",
    )
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.settimeout(max(0.001, deadline - time.monotonic()))
        client.connect(str(path))
        _, uid, _ = struct.unpack(
            "3i",
            client.getsockopt(
                socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
            ),
        )
        _require(uid == os.getuid(), "unsafe_socket_peer")
        return client
    except BaseException:
        client.close()
        raise


def watch(run_dir, timeout=300):
    _require(
        isinstance(timeout, (int, float))
        and not isinstance(timeout, bool)
        and math.isfinite(timeout)
        and 0 < timeout <= 86400,
        "invalid_timeout",
    )
    run, _ = _binding(run_dir)
    with observation_lock(run):
        cursor = _cursor(run)
        _require(not cursor["terminal"], "observation_closed")
        cursor["status"] = "observing"
        _write_json(run / "cursor.json", cursor)
        deadline = time.monotonic() + timeout
        attempts = 0
        try:
            while True:
                receipt = _replay(run, cursor)
                if receipt:
                    return receipt
                if time.monotonic() >= deadline:
                    raise TUIError("timeout")
                if attempts >= MAX_RECONNECTS:
                    raise TUIError("reconnect_exhausted")
                attempts += 1
                try:
                    with _connect(run, deadline) as client:
                        request = {
                            "version": 1,
                            "type": "observe",
                            "run_id": run.name,
                            "after_seq": cursor["seq"],
                        }
                        if cursor["epoch"] is not None:
                            request["epoch"] = cursor["epoch"]
                        client.sendall(
                            (json.dumps(request, separators=(",", ":")) + "\n").encode()
                        )
                        frames = _frames(client, deadline)
                        hello = _validate_message(next(frames), run.name, "hello")
                        _identity(cursor, hello)
                        _require(hello["seq"] >= cursor["seq"], "sequence_regressed")
                        _write_json(run / "cursor.json", cursor)
                        for message in frames:
                            event = _validate_message(message, run.name, "event")
                            receipt = _consume(run, cursor, event)
                            if receipt:
                                return receipt
                except (ConnectionError, FileNotFoundError, TimeoutError):
                    time.sleep(
                        min(0.1 * attempts, 1.0, max(0, deadline - time.monotonic()))
                    )
        except KeyboardInterrupt:
            return _receipt(run, cursor, "observation_lost", "observer_interrupted")
        except (TUIError, OSError) as exc:
            reason = (
                str(exc)
                if isinstance(exc, TUIError)
                and str(exc)
                in {
                    "timeout",
                    "reconnect_exhausted",
                    "epoch_changed",
                    "session_changed",
                    "event_gap",
                    "sequence_regressed",
                    "closed_without_event",
                }
                else "invalid_protocol"
            )
            return _receipt(run, cursor, "observation_lost", reason)


def _launch_record(run):
    record = _read_json(run / "launch.json", optional=True)
    if record is not None:
        _require(
            set(record) == {"version", "run_id", "status", "created_at"}
            and record["version"] == 1
            and record["run_id"] == run.name
            and record["status"] == "launch_intent_committed"
            and isinstance(record["created_at"], (int, float)),
            "invalid_launch_record",
        )
    return record


def status(run_dir):
    run, _ = _binding(run_dir)
    journal = _journal(run)
    cursor = _cursor(run)
    if journal is not None and cursor["epoch"] is not None:
        _identity(cursor, journal)
    return {
        "run_id": run.name,
        "journal": journal,
        "cursor": cursor,
        "observer_active": _observer_active(run),
        "launch": _launch_record(run),
    }


def _executable(path):
    _require(isinstance(path, (str, os.PathLike)), "invalid_executable")
    name = os.fspath(path)
    _require(_text(name) and name.strip(), "invalid_executable")
    resolved = shutil.which(name)
    _require(resolved is not None, "invalid_executable")
    path = Path(resolved).absolute()
    _file_info(path.stat(), private=False)
    _require(os.access(path, os.X_OK), "invalid_executable")
    return str(path)


def launch(
    run_dir,
    prompt_file,
    omp_executable="omp",
    tmux_executable="tmux",
    canary=False,
    *,
    model=None,
    thinking=None,
    append_system_prompt=None,
):
    run, binding = _binding(run_dir)
    prompt = Path(prompt_file).absolute()
    _require(
        _read_bytes(prompt, private=False, limit=1024 * 1024).strip(), "empty_prompt"
    )
    options = []
    if model is not None:
        _require(_text(model) and model.strip(), "invalid_model")
        options.append("--model=" + model)
    if thinking is not None:
        _require(
            isinstance(thinking, str) and thinking in THINKING_LEVELS,
            "invalid_thinking",
        )
        options.append("--thinking=" + thinking)
    if append_system_prompt is not None:
        _require(
            isinstance(append_system_prompt, (str, os.PathLike)),
            "invalid_system_prompt",
        )
        _require(_text(os.fspath(append_system_prompt)), "invalid_system_prompt")
        system_prompt = Path(append_system_prompt).absolute()
        _require(system_prompt == system_prompt.resolve(), "invalid_system_prompt")
        _require(
            _read_bytes(system_prompt, private=False, limit=1024 * 1024).strip(),
            "empty_system_prompt",
        )
        options += ["--append-system-prompt", str(system_prompt)]
    _require(
        _read_bytes(EXTENSION, private=False, limit=1024 * 1024).strip(),
        "missing_extension",
    )
    omp = _executable(omp_executable)
    tmux = _executable(tmux_executable)
    _require(_observer_active(run), "native_observer_required")
    _require(not _cursor(run)["terminal"], "observation_closed")
    _require(_launch_record(run) is None, "launch_already_attempted")
    argv = [
        "env",
        "OMP_HERMES_BINDING_FILE=" + str(run / "binding.json"),
        omp,
        *options,
        "-e",
        str(EXTENSION),
    ]
    if canary:
        argv += [
            "--no-tools",
            "--no-skills",
            "--no-rules",
            "--no-extensions",
            "--no-title",
            "--no-session",
            "--no-lsp",
        ]
    argv.append("@" + str(prompt))
    # The durable intent survives failed probes, ambiguous tmux timeouts and crashes.
    intent = {
        "version": 1,
        "run_id": run.name,
        "status": "launch_intent_committed",
        "created_at": time.time(),
    }
    try:
        _write_json(run / "launch.json", intent, exclusive=True)
    except FileExistsError as exc:
        raise TUIError("launch_already_attempted") from exc
    try:
        probe = subprocess.run(
            [tmux, "has-session", "-t", "=" + binding["tmux_session"]],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        _require(
            probe.returncode == 1,
            "tmux_session_exists" if probe.returncode == 0 else "tmux_probe_failed",
        )
        result = subprocess.run(
            [
                tmux,
                "new-session",
                "-d",
                "-s",
                binding["tmux_session"],
                "-c",
                binding["workspace"],
                shlex.join(argv),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
        _require(result.returncode == 0, "launch_ambiguous")
        verify = subprocess.run(
            [tmux, "has-session", "-t", "=" + binding["tmux_session"]],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        _require(verify.returncode == 0, "launch_ambiguous")
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TUIError("launch_ambiguous") from exc
    return {
        "run_id": run.name,
        "status": "tmux_session_present",
        "launch": "launch_intent_committed",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="omp-supervise", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--workspace", required=True)
    prepare_parser.add_argument("--tmux-session", required=True)
    prepare_parser.add_argument("--state-root")
    prepare_parser.add_argument("--run-id")
    launch_parser = commands.add_parser("launch")
    launch_parser.add_argument("--run-dir", required=True)
    launch_parser.add_argument("--prompt-file", required=True)
    launch_parser.add_argument("--omp-executable", default="omp")
    launch_parser.add_argument("--tmux-executable", default="tmux")
    launch_parser.add_argument("--model")
    launch_parser.add_argument("--thinking", choices=THINKING_LEVELS)
    launch_parser.add_argument("--append-system-prompt", metavar="FILE")
    launch_parser.add_argument("--canary", action="store_true")
    watch_parser = commands.add_parser("watch")
    watch_parser.add_argument("--run-dir", required=True)
    watch_parser.add_argument("--timeout", type=float, default=300)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--run-dir", required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        result = {
            "prepare": prepare,
            "launch": launch,
            "watch": watch,
            "status": status,
        }[command](**args)
    except (TUIError, OSError) as exc:
        print(
            json.dumps({
                "error": str(exc)
                if isinstance(exc, TUIError)
                else "filesystem_or_process_error"
            }),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
