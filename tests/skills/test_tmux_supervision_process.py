"""Real subprocess/PTY and local-socket contracts for generic tmux supervision."""

import sys

import pytest

# Skip before importing the Linux-only producer or POSIX terminal modules.
if sys.platform != "linux":
    pytest.skip("Linux-only skill", allow_module_level=True)

import contextlib
import hashlib
import json
import os
import pty
import select
import signal
import socket
import stat
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

pytestmark = pytest.mark.linux_only
SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "optional-skills/autonomous-ai-agents/tmux-supervision/scripts"
)
sys.path.insert(0, str(SCRIPTS))
from tmux_supervisor import process_adapter, supervision  # noqa: E402


@pytest.fixture
def run(monkeypatch):
    with tempfile.TemporaryDirectory(
        prefix="p", dir=os.environ.get("TMPDIR")
    ) as directory:
        root = Path(directory).resolve()
        (root / "h").mkdir()
        (root / "w").mkdir()
        for name, value in {
            "HERMES_HOME": str(root / "h"),
            "HERMES_SESSION_PLATFORM": "discord",
            "HERMES_SESSION_KEY": "process-test-key",
            "HERMES_SESSION_ID": "process-test-generation",
            "HERMES_SESSION_CHAT_ID": "test-chat",
            "HERMES_SESSION_THREAD_ID": "test-thread",
        }.items():
            monkeypatch.setenv(name, value)
        prepared = supervision.prepare(
            root / "w", "command-test", root, adapter="command"
        )
        yield Path(prepared["run_dir"])


def command_file(run, value):
    path = run.parent / "argv.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def journal(run):
    return json.loads((run / "journal.json").read_bytes())


@contextlib.contextmanager
def bridge(run):
    producer = process_adapter._Bridge(run)
    producer.start()
    try:
        producer.publish("started")
        yield producer
    finally:
        producer.close()


@contextlib.contextmanager
def subscriber(run, after_seq=0, **extra):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(5)
        connection.connect(str(run / "bridge.sock"))
        connection.sendall(
            json.dumps({
                "version": 2,
                "type": "observe",
                "run_id": run.name,
                "after_seq": after_seq,
                **extra,
            }).encode()
            + b"\n"
        )
        with connection.makefile("rb") as stream:
            yield stream


def receive(stream):
    value = stream.readline(supervision.MAX_FRAME + 1)
    assert value.endswith(b"\n") and len(value) <= supervision.MAX_FRAME
    return json.loads(value)


def read_line(stream):
    fd = stream if isinstance(stream, int) else stream.fileno()
    deadline = time.monotonic() + 5
    result = bytearray()
    while not result.endswith(b"\n"):
        readable, _, _ = select.select(
            [fd], [], [], max(0, deadline - time.monotonic())
        )
        assert readable, "child did not produce the expected line"
        chunk = os.read(fd, 1)
        assert chunk, "child closed its output before the expected line"
        result.extend(chunk)
        assert len(result) <= 16384
    return bytes(result).rstrip(b"\r\n")


@contextlib.contextmanager
def child_process(run, source, *, bootstrap="", terminal=False, extra_args=()):
    if bootstrap or terminal:
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(SCRIPTS)!r}); "
            "from tmux_supervisor import process_adapter as adapter; "
            + (
                "import fcntl, termios; fcntl.ioctl(0, termios.TIOCSCTTY, 0); "
                if terminal
                else ""
            )
            + "\n"
            + bootstrap
            + "\nraise SystemExit(adapter.run_command(sys.argv[1], sys.argv[2:]))"
        )
        runner = [
            sys.executable,
            "-c",
            code,
            str(run),
            sys.executable,
            "-c",
            source,
            *extra_args,
        ]
    else:
        path = command_file(run, [sys.executable, "-c", source, *extra_args])
        runner = [
            sys.executable,
            str(SCRIPTS / "tmux_supervise.py"),
            "_run-command",
            "--run-dir",
            str(run),
            "--command-file",
            str(path),
            "--command-sha256",
            hashlib.sha256(path.read_bytes()).hexdigest(),
        ]
    # tmux may retain a previous Hermes turn's environment. The explicit
    # immutable binding, not this stale environment, identifies the producer.
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("HERMES_SESSION_")
    }
    env["HERMES_SESSION_ID"] = "stale-tmux-generation"
    master = slave = None
    if terminal:
        master, slave = pty.openpty()
    process = subprocess.Popen(
        runner,
        stdin=slave if terminal else subprocess.PIPE,
        stdout=slave if terminal else subprocess.PIPE,
        stderr=slave if terminal else subprocess.PIPE,
        start_new_session=True,
        env=env,
    )
    if slave is not None:
        os.close(slave)
    try:
        yield process, master
    finally:
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Only this fixture's new session is eligible for test cleanup.
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        if master is not None:
            os.close(master)


def test_launch_resolves_argv_without_persisting_or_shell_parsing(run, monkeypatch):
    arguments = [sys.executable, "", "a b", "$(not-a-command);", "PRIVATE_PROMPT\nline"]
    path = command_file(run, arguments)
    launch = Mock(return_value={"status": "launched"})
    monkeypatch.setattr(supervision, "launch", launch)
    assert process_adapter.launch(run, path, tmux_executable="test-tmux") == {
        "status": "launched"
    }
    launch.assert_called_once_with(
        run,
        [
            sys.executable,
            str(SCRIPTS / "tmux_supervise.py"),
            "_run-command",
            "--run-dir",
            str(run),
            "--command-file",
            str(path),
            "--executable",
            supervision._executable(sys.executable),
            "--command-sha256",
            hashlib.sha256(path.read_bytes()).hexdigest(),
        ],
        tmux_executable="test-tmux",
    )
    assert {path.name for path in run.iterdir()} == {"binding.json"}


def test_command_digest_binds_exact_validated_bytes_before_parsing(run):
    path = command_file(run, [sys.executable, "-c", "pass", "PRIVATE_PROMPT"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert process_adapter.read_command(path, digest) == [
        supervision._executable(sys.executable),
        "-c",
        "pass",
        "PRIVATE_PROMPT",
    ]
    path.write_bytes(b"not JSON: PRIVATE_CHANGED_PROMPT")
    with pytest.raises(supervision.TUIError, match="^command_changed$"):
        process_adapter.read_command(path, digest)
    with pytest.raises(supervision.TUIError, match="^invalid_command_digest$"):
        process_adapter.read_command(path, digest.upper())


def test_launch_pins_executable_despite_stale_tmux_path(run, monkeypatch):
    original = run.parent / "original"
    stale = run.parent / "stale"
    for directory, message in ((original, "ORIGINAL_APP"), (stale, "WRONG_APP")):
        directory.mkdir()
        executable = directory / "test-app"
        executable.write_text(f"#!{sys.executable}\nprint({message!r})\n")
        executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(original))
    launch = Mock(return_value={"status": "launched"})
    monkeypatch.setattr(supervision, "launch", launch)
    path = command_file(run, ["test-app"])
    process_adapter.launch(run, path)
    result = subprocess.run(
        launch.call_args.args[1],
        env={**os.environ, "PATH": str(stale)},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "ORIGINAL_APP\n"


def test_large_command_stays_out_of_tmux_argv(run, monkeypatch):
    arguments = [sys.executable, "-c", "pass", *("x" * 100000 for _ in range(8))]
    path = command_file(run, arguments)
    launch = Mock(return_value={"status": "launched"})
    monkeypatch.setattr(supervision, "launch", launch)
    process_adapter.launch(run, path)
    wrapper = launch.call_args.args[1]
    assert sum(len(arg) for arg in wrapper) < 4096
    assert process_adapter.read_command(path, wrapper[-1]) == arguments


@pytest.mark.parametrize("name", ["./workspace-app", "workspace-app"])
@pytest.mark.parametrize("caller_collision", [False, True])
def test_relative_executable_resolution_uses_enrolled_workspace(
    run, monkeypatch, name, caller_collision
):
    executable = run.parent / "w" / "workspace-app"
    executable.write_text(f"#!{sys.executable}\nprint('WORKSPACE_APP')\n")
    executable.chmod(0o700)
    if caller_collision:
        wrong_executable = run.parent / "workspace-app"
        wrong_executable.write_text(f"#!{sys.executable}\nprint('WRONG_CALLER_APP')\n")
        wrong_executable.chmod(0o700)
    monkeypatch.chdir(run.parent)
    monkeypatch.setenv("PATH", ".")
    launch = Mock(return_value={"status": "launched"})
    monkeypatch.setattr(supervision, "launch", launch)
    process_adapter.launch(run, command_file(run, [name]))
    result = subprocess.run(
        launch.call_args.args[1],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "WORKSPACE_APP\n"


def test_cli_rejects_changed_command_before_starting_process(run):
    path = command_file(run, [sys.executable, "-c", "print('MUST_NOT_START')"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(b"PRIVATE_CHANGED_PROMPT")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "tmux_supervise.py"),
            "_run-command",
            "--run-dir",
            str(run),
            "--command-file",
            str(path),
            "--command-sha256",
            digest,
        ],
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 2
    assert result.stdout == b""
    assert json.loads(result.stderr) == {"error": "command_changed"}
    assert not (run / "journal.json").exists()


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        "python",
        [],
        [1],
        [True],
        [["python"]],
        [""],
        ["a\0b"],
        [sys.executable, "\0"],
        [sys.executable, "\ud800"],
    ],
)
def test_launch_rejects_invalid_argv(run, monkeypatch, value):
    launch = Mock()
    monkeypatch.setattr(supervision, "launch", launch)
    with pytest.raises(supervision.TUIError):
        process_adapter.launch(run, command_file(run, value))
    launch.assert_not_called()
    assert not (run / "launch.json").exists()


@pytest.mark.parametrize(
    "content",
    [
        b"not-json PRIVATE_PROMPT",
        b"\xff",
        b"[" * 1500 + b"]" * 1500,
        b" " * (process_adapter.MAX_COMMAND + 1),
    ],
)
def test_invalid_command_files_are_bounded_and_sanitized(run, content):
    path = run.parent / "bad.json"
    path.write_bytes(content)
    with pytest.raises(supervision.TUIError) as error:
        process_adapter.launch(run, path)
    assert "PRIVATE_PROMPT" not in str(error.value)
    assert not (run / "launch.json").exists()


def test_missing_command_and_wrong_owner_do_not_launch(run, monkeypatch):
    with pytest.raises(supervision.TUIError, match="invalid_command_file"):
        process_adapter.launch(run, run.parent / "missing")
    monkeypatch.setenv("HERMES_SESSION_ID", "not-owner")
    with pytest.raises(supervision.TUIError, match="scope_denied"):
        process_adapter.launch(run, command_file(run, [sys.executable]))


@pytest.mark.parametrize("entry", ["launch", "run_command"])
def test_omp_binding_cannot_be_used_as_command_adapter(run, entry):
    binding = json.loads((run / "binding.json").read_bytes())
    binding["adapter"] = "omp"
    supervision._write_json(run / "binding.json", binding)
    arguments = (
        command_file(run, [sys.executable]) if entry == "launch" else [sys.executable]
    )
    with pytest.raises(supervision.TUIError, match="adapter_mismatch"):
        getattr(process_adapter, entry)(run, arguments)
    assert not (run / "journal.json").exists()


@pytest.mark.parametrize("exit_code", [0, 7, 255, -signal.SIGTERM])
def test_real_exit_is_durable_terminal_and_replayed_to_core(
    run, exit_code, monkeypatch
):
    monkeypatch.setattr(
        subprocess.Popen,
        "poll",
        lambda *_: pytest.fail("completion must use wait, not poll"),
    )
    code = (
        f"import sys; sys.exit({exit_code})"
        if exit_code >= 0
        else "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"
    )
    assert process_adapter.run_command(run, [sys.executable, "-c", code]) == exit_code
    state = journal(run)
    assert state["state"] == "closed"
    assert state["pid"] == os.getpid()
    assert len(state["epoch"]) == 32 and int(state["epoch"], 16)
    assert state["app_session_id"]
    assert [(event["seq"], event["kind"]) for event in state["events"]] == [
        (1, "started"),
        (2, "process_exited"),
    ]
    assert state["events"][-1]["exit_code"] == exit_code
    assert stat.S_IMODE((run / "journal.json").stat().st_mode) == 0o600
    assert not (run / "bridge.sock").exists()
    receipt = supervision.watch(run, timeout=2)
    assert (receipt["kind"], receipt["exit_code"]) == ("process_exited", exit_code)
    assert supervision.status(run)["cursor"]["terminal"] is True
    with pytest.raises(supervision.TUIError, match="producer_already_exists"):
        process_adapter.run_command(run, [sys.executable, "-c", "pass"])


def test_real_stdin_stdout_stderr_are_inherited_and_not_journaled(run):
    source = (
        "import os, sys; "
        f"assert os.getcwd() == {str(run.parent / 'w')!r}; "
        "print('READY', flush=True); "
        "line = input(); "
        "print('OUT:' + line, flush=True); "
        "print('PRIVATE_STDERR', file=sys.stderr, flush=True)"
    )
    with child_process(run, source, extra_args=("PRIVATE_ARGV",)) as (process, _):
        assert process.stdin is not None
        assert read_line(process.stdout) == b"READY"
        with subscriber(run) as stream:
            hello = receive(stream)
            started = receive(stream)
            assert hello["type"] == "hello" and "events" not in hello
            assert started["kind"] == "started"
            assert started == journal(run)["events"][0]
            assert journal(run)["state"] == "busy"
            assert stat.S_IMODE((run / "bridge.sock").stat().st_mode) == 0o600
            process.stdin.write(b"PRIVATE_INPUT\n")
            process.stdin.flush()
            assert read_line(process.stdout) == b"OUT:PRIVATE_INPUT"
            assert read_line(process.stderr) == b"PRIVATE_STDERR"
            exited = receive(stream)
            assert (exited["kind"], exited["exit_code"]) == ("process_exited", 0)
            assert exited == journal(run)["events"][-1]
        assert process.wait(timeout=5) == 0
    state = (run / "journal.json").read_bytes()
    for secret in (
        b"PRIVATE_INPUT",
        b"PRIVATE_STDERR",
        b"PRIVATE_ARGV",
        source.encode(),
    ):
        assert secret not in state
    assert {path.name for path in run.iterdir()} == {"binding.json", "journal.json"}


def test_tty_flags_and_terminal_keyboard_signals_reach_only_real_child_behavior(run):
    source = """
import json, os, signal
assert signal.getsignal(signal.SIGINT) != signal.SIG_IGN
assert signal.getsignal(signal.SIGQUIT) == signal.SIG_DFL
print(json.dumps([os.isatty(fd) for fd in (0, 1, 2)]), flush=True)
signal.signal(signal.SIGINT, lambda *_: print('CHILD_INTERRUPT', flush=True))
signal.signal(signal.SIGQUIT, lambda *_: print('CHILD_QUIT', flush=True))
print('READY', flush=True)
print('CHILD_INPUT:' + input(), flush=True)
"""
    with child_process(run, source, terminal=True) as (process, master):
        assert master is not None
        assert json.loads(read_line(master)) == [True, True, True]
        assert read_line(master) == b"READY"
        with subscriber(run) as stream:
            receive(stream)
            assert receive(stream)["kind"] == "started"
            os.write(master, b"\x03")
            assert b"CHILD_INTERRUPT" in read_line(master)
            os.write(master, b"\x1c")
            assert b"CHILD_QUIT" in read_line(master)
            assert journal(run)["seq"] == 1
            os.write(master, b"kept-running\n")
            assert read_line(master) == b"kept-running"
            assert read_line(master) == b"CHILD_INPUT:kept-running"
            assert receive(stream)["exit_code"] == 0
        assert process.wait(timeout=5) == 0


@pytest.mark.parametrize("failure", ["started", "process_exited", "transport"])
def test_monitoring_failure_does_not_terminate_or_restart_child(run, failure):
    if failure == "transport":
        bootstrap = """
def fail_select(self, timeout=None):
    print('MONITOR_FAILURE_INJECTED', file=sys.stderr, flush=True)
    raise OSError('PRIVATE_MONITOR_ERROR')
adapter.selectors.EpollSelector.select = fail_select
"""
    else:
        seq = 1 if failure == "started" else 2
        bootstrap = f"""
write = adapter.supervision._write_json
def fail_write(path, value, **kwargs):
    if value.get('seq') == {seq}:
        print('MONITOR_FAILURE_INJECTED', file=sys.stderr, flush=True)
        raise OSError('PRIVATE_MONITOR_ERROR')
    return write(path, value, **kwargs)
adapter.supervision._write_json = fail_write
"""
    source = "import sys; print('READY', flush=True); print('DONE:' + input(), flush=True); sys.exit(9)"
    with child_process(run, source, bootstrap=bootstrap) as (process, _):
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        assert read_line(process.stdout) == b"READY"
        if failure != "process_exited":
            assert read_line(process.stderr) == b"MONITOR_FAILURE_INJECTED"
        process.stdin.write(b"still-alive\n")
        process.stdin.flush()
        assert read_line(process.stdout) == b"DONE:still-alive"
        if failure == "process_exited":
            assert read_line(process.stderr) == b"MONITOR_FAILURE_INJECTED"
        assert process.wait(timeout=5) == 9
        assert process.stdout.read() == b""
        assert process.stderr.read() == b""
    state = journal(run)
    assert "PRIVATE_MONITOR_ERROR" not in json.dumps(state)
    if failure == "transport":
        assert state["state"] == "closed"
        assert state["events"][-1]["exit_code"] == 9
    else:
        assert not any(event["kind"] == "process_exited" for event in state["events"])


def test_exec_failure_does_not_invent_process_lifecycle(run):
    executable = run.parent / "PRIVATE_EXECUTABLE"
    executable.write_bytes(b"executable without a shebang")
    executable.chmod(0o700)
    with pytest.raises(supervision.TUIError, match="^command_start_failed$"):
        process_adapter.run_command(run, [str(executable)])
    assert journal(run)["seq"] == 0
    assert journal(run)["events"] == []
    assert not (run / "bridge.sock").exists()


@pytest.mark.parametrize("path_name", ["bridge.sock", "journal.json"])
@pytest.mark.parametrize("kind", ["file", "symlink", "directory"])
def test_existing_monitoring_paths_are_never_adopted_or_removed(run, path_name, kind):
    target = run / path_name
    if kind == "file":
        target.write_bytes(b"existing")
    elif kind == "symlink":
        target.symlink_to(run.parent / "missing-target")
    else:
        target.mkdir()
    before = target.lstat()
    with pytest.raises(supervision.TUIError, match="producer_already_exists"):
        process_adapter.run_command(
            run, [sys.executable, "-c", "raise RuntimeError('must not start')"]
        )
    assert target.lstat().st_ino == before.st_ino


@pytest.mark.parametrize("cursor", [-1, True, 1, 2, 2**64, None, "0"])
def test_invalid_cursor_rejects_connection_without_poisoning_journal(run, cursor):
    with bridge(run):
        before = (run / "journal.json").read_bytes()
        with subscriber(run, after_seq=cursor) as stream:
            assert receive(stream)["type"] == "hello"
            assert receive(stream) == {
                "version": 2,
                "type": "rejected",
                "run_id": run.name,
                "reason": "invalid_cursor",
            }
            assert stream.readline() == b""
        assert (run / "journal.json").read_bytes() == before
        with subscriber(run) as stream:
            assert receive(stream)["type"] == "hello"
            assert receive(stream)["kind"] == "started"


def test_reconnect_cursor_and_epoch_are_strict_and_do_not_replay_seen_events(run):
    with bridge(run) as producer:
        epoch = journal(run)["epoch"]
        with subscriber(run, epoch="wrong-epoch") as stream:
            assert receive(stream)["type"] == "hello"
            assert receive(stream)["reason"] == "epoch_mismatch"
            assert stream.readline() == b""
        with subscriber(run, after_seq=1, epoch=epoch) as stream:
            assert receive(stream)["seq"] == 1
            producer.publish("process_exited", exit_code=3)
            event = receive(stream)
            assert (event["seq"], event["kind"], event["exit_code"]) == (
                2,
                "process_exited",
                3,
            )
            assert event == journal(run)["events"][-1]


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json\n",
        b"[]\n",
        b"{}\n",
        b"\xff\n",
        b"[" * 1500 + b"]" * 1500 + b"\n",
        b"x" * (supervision.MAX_FRAME + 1),
        b'{"type":"input","text":"PRIVATE_CONTROL"}\n',
        b'{"type":"kill","pid":1}\n',
    ],
)
def test_malformed_and_control_requests_only_drop_the_client(run, payload):
    with bridge(run):
        before = (run / "journal.json").read_bytes()
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(5)
            connection.connect(str(run / "bridge.sock"))
            connection.sendall(payload)
            try:
                assert connection.recv(1) == b""
            except ConnectionResetError:
                pass
        assert (run / "journal.json").read_bytes() == before
        with subscriber(run) as stream:
            assert receive(stream)["type"] == "hello"
            assert receive(stream)["kind"] == "started"


def test_peer_uid_is_enforced_before_any_frames_are_sent(run, monkeypatch):
    real_getsockopt = socket.socket.getsockopt

    def other_uid(connection, level, option, *args):
        if (level, option) == (socket.SOL_SOCKET, socket.SO_PEERCRED):
            return struct.pack("3i", os.getpid(), os.getuid() + 1, os.getgid())
        return real_getsockopt(connection, level, option, *args)

    with bridge(run):
        with monkeypatch.context() as patch:
            patch.setattr(socket.socket, "getsockopt", other_uid)
            with socket.socket(socket.AF_UNIX) as connection:
                connection.settimeout(5)
                connection.connect(str(run / "bridge.sock"))
                assert connection.recv(1) == b""
        with subscriber(run) as stream:
            assert receive(stream)["type"] == "hello"


def test_subscriber_count_and_handshake_time_are_bounded(run, monkeypatch):
    monkeypatch.setattr(process_adapter, "HANDSHAKE_TIMEOUT", 2)
    with bridge(run), contextlib.ExitStack() as clients:
        for _ in range(process_adapter.MAX_CLIENTS):
            stream = clients.enter_context(subscriber(run))
            receive(stream)
            receive(stream)
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(5)
            connection.connect(str(run / "bridge.sock"))
            assert connection.recv(1) == b""
    # A separate enrollment tests silent handshakes, without connection retries.
    prepared = supervision.prepare(run.parent / "w", "handshake", run.parent)
    other_run = Path(prepared["run_dir"])
    with bridge(other_run), socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(5)
        connection.connect(str(other_run / "bridge.sock"))
        connection.sendall(b'{"version":')
        assert connection.recv(1) == b""


def test_per_subscriber_output_is_bounded_without_poisoning_the_bridge(
    run, monkeypatch
):
    with bridge(run):
        with monkeypatch.context() as patch:
            patch.setattr(process_adapter, "MAX_PENDING", 1)
            with subscriber(run) as stream:
                assert stream.readline() == b""
        with subscriber(run) as stream:
            assert receive(stream)["type"] == "hello"
            assert receive(stream)["kind"] == "started"
        assert journal(run)["seq"] == 1
