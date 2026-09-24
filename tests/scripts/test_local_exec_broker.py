"""Behavior tests for the packaged local execution broker (``tools.local_exec_broker``).

Regression cover for #59293. The argv-only ``sudo -u`` carrier closed the same-UID policy
escape but broke ``execute_code`` three ways at once, all of them the same root shape —
*implicit* inheritance across a uid boundary:

  * ``sudo``'s ``env_reset`` discards the explicit ``env=`` dict ``code_kernel._spawn``
    passes to ``Popen``;
  * ``sudo`` closes inherited non-std descriptors, so the parent-death pipe
    (``HERMES_KERNEL_PARENT_DEATH_FD``) never reaches the kernel runner;
  * the kernel staging dir is ``tempfile.mkdtemp()`` (0700), which the new uid cannot
    traverse to reach ``hermes_kernel_runner.py``.

Two tests, one vertical contract each:

  1. **Transport + protocol.** What the broker accepts, what it refuses, and who owns a
     descriptor that arrived over ``SCM_RIGHTS``. Every refusal path is also an fd-ownership
     path: the broker must close what it received, or the peer's pipe never reaches EOF.
  2. **Lease lifetime.** Nothing outlives its lease (child, descendants, broker shutdown) and
     nothing pins a worker thread forever (a child that exits on its own, a silent peer).

Real processes throughout: a separately spawned broker, a real ``AF_UNIX`` connection, real
children and grandchildren. Nothing here is mocked — the point is the boundary.
"""

from __future__ import annotations

import array
import contextlib
import errno
import fcntl
import json
import os
import selectors
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Wall-clock ceilings. Generous on purpose: these bound a FAILURE (the thing under test never
# happened) rather than pace a success, and the suite runner is not a quiet machine.
DEADLINE = 15.0
HANDSHAKE_TIMEOUT = 2.0


def _load_broker():
    """Import the packaged broker module used by installed Hermes builds."""
    import tools.local_exec_broker as broker

    return broker


def _socket_path():
    """A short-named socket path.

    Not under ``tmp_path``: pytest's per-test directory names push an ``AF_UNIX`` path past
    the 108-byte ``sun_path`` limit. ``code_kernel._bind_rpc_socket`` binds short names in
    ``gettempdir()`` for the same reason.
    """
    return os.path.join(tempfile.mkdtemp(prefix="hbrk_"), "b.sock")


def _start_broker(
    host_only_value,
    staging_root,
    *,
    sock_path=None,
    expect_ready=True,
    allowed_uids=None,
    socket_mode=None,
    systemd_cgroup=False,
):
    """Spawn the broker as its own process; return (proc, socket path).

    ``host_only_value`` lands in the BROKER's environment only. A child that can see it
    inherited the broker's env instead of receiving the approved one.
    """
    sock_path = sock_path or _socket_path()
    argv = [
        sys.executable,
        "-m",
        "tools.local_exec_broker",
        "--socket",
        sock_path,
        "--staging-root",
        str(staging_root),
        "--handshake-timeout",
        str(HANDSHAKE_TIMEOUT),
    ]
    for uid in allowed_uids or []:
        argv.extend(["--allow-uid", str(uid)])
    if socket_mode is not None:
        argv.extend(["--socket-mode", socket_mode])
    if systemd_cgroup:
        argv.append("--systemd-cgroup")
    broker_env = {
        "PATH": os.environ.get("PATH", ""),
        "BROKER_PROBE_HOST_ONLY": host_only_value,
    }
    if systemd_cgroup:
        runtime_dir = f"/run/user/{os.geteuid()}"
        broker_env.update(
            XDG_RUNTIME_DIR=runtime_dir,
            DBUS_SESSION_BUS_ADDRESS=f"unix:path={runtime_dir}/bus",
        )
    proc = subprocess.Popen(
        argv,
        env=broker_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if not expect_ready:
        return proc, sock_path
    ready = proc.stdout.readline()
    if not ready:
        proc.terminate()
        pytest.fail(f"broker never became ready; stderr:\n{proc.stderr.read()}")
    assert json.loads(ready)["ready"] is True, f"broker never became ready: {ready!r}"
    return proc, sock_path


def _stop_broker(proc):
    proc.terminate()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=10)


def _pid_running(pid: int) -> bool:
    """True while *pid* is a LIVE process. A killed-but-unreaped child is still a ``/proc``
    entry that answers ``kill(pid, 0)``, so excluding the zombie state makes the broker's own
    reaping part of the contract rather than an implementation detail."""
    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    return stat_line.rsplit(") ", 1)[1].split(" ", 1)[0] != "Z"


def _wait_until_gone(pid: int, timeout: float = DEADLINE) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_running(pid):
            return True
        time.sleep(0.05)
    return not _pid_running(pid)


def _fd_count(pid: int) -> int:
    return len(os.listdir(f"/proc/{pid}/fd"))


def _wait_for_fd_count(pid: int, ceiling: int, timeout: float = DEADLINE) -> int:
    """Poll the broker's open-descriptor count back down to at most *ceiling*.

    A ceiling rather than equality: worker threads close asynchronously, so an earlier lease
    may still be unwinding when the baseline is sampled. A leak GROWS the count, so the
    inequality still catches it, without turning teardown timing into a flake.
    """
    deadline = time.monotonic() + timeout
    count = _fd_count(pid)
    while time.monotonic() < deadline and count > ceiling:
        time.sleep(0.05)
        count = _fd_count(pid)
    return count


def _read_with_deadline(
    read_fd: int, *, until_eof: bool, timeout: float = DEADLINE
) -> bytes:
    """Read until EOF (or one newline), failing on a deadline instead of hanging.

    A bare blocking ``os.read`` here would turn the regression this file exists to catch —
    the broker retaining a descriptor it forwarded, leaving a live writer on the peer's pipe —
    into an indefinite hang instead of a failure. A hang is a much weaker CI signal.
    """
    os.set_blocking(read_fd, False)
    chunks: list[bytes] = []
    deadline = time.monotonic() + timeout
    with selectors.DefaultSelector() as sel:
        sel.register(read_fd, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                pytest.fail(
                    "pipe never reached EOF: the broker retained a descriptor it was "
                    f"handed (read so far: {b''.join(chunks)!r})"
                )
            if not sel.select(remaining):
                continue
            chunk = os.read(read_fd, 4096)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            if not until_eof and b"\n" in chunk:
                return b"".join(chunks).split(b"\n", 1)[0]


def _child_report(read_fd: int) -> dict:
    """Read the child's JSON report off the SCM_RIGHTS pipe.

    The empty-read guard matters: a child that never ran (or died before writing) closes its
    copy and yields EOF, and a bare ``json.loads`` would report that as an opaque decode
    error instead of naming what actually happened.
    """
    raw = _read_with_deadline(read_fd, until_eof=True)
    assert raw, "the child produced no report: it never ran, or died before writing"
    return json.loads(raw)


def _stage_runner(root, name, source):
    """Write *source* into the staging root, mirroring ``code_kernel._spawn``'s mkdtemp."""
    runner = Path(root) / name
    runner.write_text(textwrap.dedent(source), encoding="utf-8")
    return runner


def _staging_root(tmp_path):
    root = tmp_path / "staging"
    root.mkdir(mode=0o700)
    return root


def _require_systemd_user_scopes():
    runtime_dir = f"/run/user/{os.geteuid()}"
    probe_env = dict(os.environ)
    probe_env.update(
        XDG_RUNTIME_DIR=runtime_dir,
        DBUS_SESSION_BUS_ADDRESS=f"unix:path={runtime_dir}/bus",
    )
    probe = subprocess.run(
        [
            "systemd-run",
            "--user",
            "--scope",
            "--quiet",
            "--collect",
            "--",
            "/bin/true",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        env=probe_env,
    )
    if probe.returncode != 0:
        pytest.skip(f"systemd user scopes unavailable: {probe.stderr.strip()}")


@pytest.mark.linux_only
def test_user_service_definition_enables_required_cgroup_containment():
    broker = _load_broker()
    broker_path = "/opt/hermes checkout/scripts/local_exec_broker.py"

    service = broker._render_user_service(
        python_path="/opt/hermes venv/bin/python",
        import_root="/opt/hermes %checkout",
    )

    assert "ExecStart=" in service
    assert "-m" in service
    assert "tools.local_exec_broker" in service
    assert broker_path not in service
    assert "--systemd-cgroup" in service
    assert "--socket=%t/hermes-local-exec-broker/broker.sock" in service
    assert "--staging-root=%t/hermes-local-exec-broker" in service
    assert "RuntimeDirectory=hermes-local-exec-broker" in service
    assert "RuntimeDirectoryMode=0700" in service
    assert "WorkingDirectory=/opt/hermes %%checkout" in service
    assert 'WorkingDirectory="' not in service
    assert "KillMode=control-group" in service
    assert "StartLimitIntervalSec=60" in service
    assert "StartLimitBurst=3" in service
    assert "WantedBy=default.target" in service
    with pytest.raises(ValueError, match="line break"):
        broker._render_user_service(
            python_path="/usr/bin/python3", import_root="/tmp/bad\npath"
        )
    with pytest.raises(ValueError, match="line break"):
        broker._render_user_service(
            python_path="/tmp/bad\npython", import_root="/tmp/hermes"
        )


@pytest.mark.linux_only
def test_user_service_definition_launches_module_from_source_checkout(tmp_path):
    broker = _load_broker()
    import_root = broker._verified_import_root()
    service = broker._render_user_service(
        python_path=sys.executable,
        import_root=import_root,
    )
    directives = dict(
        line.split("=", 1) for line in service.splitlines() if "=" in line
    )
    command = shlex.split(directives["ExecStart"])
    assert directives["WorkingDirectory"] == import_root

    result = subprocess.run(
        [*command[:3], "--help"],
        cwd=directives["WorkingDirectory"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Local execution broker" in result.stdout


@pytest.mark.linux_only
def test_install_user_service_writes_unit_and_enables_it(tmp_path, monkeypatch):
    broker = _load_broker()
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(broker.subprocess, "run", fake_run)

    unit_path = broker._install_user_service(
        service_dir=tmp_path / "systemd" / "user",
        python_path="/usr/bin/python3",
        import_root=str(REPO_ROOT),
    )

    assert unit_path == tmp_path / "systemd" / "user" / broker.USER_SERVICE_NAME
    assert "--systemd-cgroup" in unit_path.read_text(encoding="utf-8")
    assert [call[0] for call in calls] == [
        ["systemctl", "--user", "show-environment"],
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", broker.USER_SERVICE_NAME],
    ]
    assert all(call[1]["check"] is True for call in calls)
    probe_kwargs = calls[0][1]
    assert probe_kwargs["stdin"] is subprocess.DEVNULL
    assert probe_kwargs["stdout"] is subprocess.DEVNULL
    assert probe_kwargs["stderr"] is subprocess.PIPE


@pytest.mark.linux_only
def test_install_user_service_does_not_publish_unit_without_user_manager(
    tmp_path, monkeypatch
):
    broker = _load_broker()
    service_dir = tmp_path / "systemd" / "user"

    def unavailable(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv, stderr="Failed to connect to bus")

    monkeypatch.setattr(broker.subprocess, "run", unavailable)

    with pytest.raises(subprocess.CalledProcessError):
        broker._install_user_service(
            service_dir=service_dir,
            python_path="/usr/bin/python3",
            import_root=str(REPO_ROOT),
        )

    assert not (service_dir / broker.USER_SERVICE_NAME).exists()


@pytest.mark.linux_only
def test_cgroup_v2_kill_targets_the_process_membership(tmp_path):
    broker = _load_broker()
    proc_root = tmp_path / "proc"
    cgroup_root = tmp_path / "cgroup"
    (proc_root / "4321").mkdir(parents=True)
    (proc_root / "4321" / "cgroup").write_text(
        "0::/user.slice/test.scope\n", encoding="utf-8"
    )
    (proc_root / "1234").mkdir(parents=True)
    (proc_root / "1234" / "cgroup").write_text(
        "0::/user.slice/hermes-broker.service\n", encoding="utf-8"
    )
    scope = cgroup_root / "user.slice" / "test.scope"
    scope.mkdir(parents=True)
    kill_file = scope / "cgroup.kill"
    kill_file.write_text("", encoding="utf-8")

    assert broker._kill_cgroup_v2(
        4321,
        broker_pid=1234,
        proc_root=proc_root,
        cgroup_root=cgroup_root,
    )
    assert kill_file.read_text(encoding="utf-8") == "1"


@pytest.mark.linux_only
def test_cgroup_v2_kill_refuses_the_broker_membership(tmp_path):
    broker = _load_broker()
    proc_root = tmp_path / "proc"
    cgroup_root = tmp_path / "cgroup"
    for pid in (1234, 4321):
        (proc_root / str(pid)).mkdir(parents=True)
        (proc_root / str(pid) / "cgroup").write_text(
            "0::/user.slice/hermes-broker.service\n", encoding="utf-8"
        )
    broker_cgroup = cgroup_root / "user.slice" / "hermes-broker.service"
    broker_cgroup.mkdir(parents=True)
    kill_file = broker_cgroup / "cgroup.kill"
    kill_file.write_text("", encoding="utf-8")

    assert not broker._kill_cgroup_v2(
        4321,
        broker_pid=1234,
        proc_root=proc_root,
        cgroup_root=cgroup_root,
    )
    assert kill_file.read_text(encoding="utf-8") == ""


@pytest.mark.linux_only
def test_cgroup_v2_kill_refuses_ancestor_of_broker_membership(tmp_path):
    broker = _load_broker()
    proc_root = tmp_path / "proc"
    cgroup_root = tmp_path / "cgroup"
    memberships = {
        1234: "/user.slice/hermes-broker.service/worker.scope",
        4321: "/user.slice/hermes-broker.service",
    }
    for pid, membership in memberships.items():
        (proc_root / str(pid)).mkdir(parents=True)
        (proc_root / str(pid) / "cgroup").write_text(
            f"0::{membership}\n", encoding="utf-8"
        )
    ancestor = cgroup_root / "user.slice" / "hermes-broker.service"
    ancestor.mkdir(parents=True)
    kill_file = ancestor / "cgroup.kill"
    kill_file.write_text("", encoding="utf-8")

    assert not broker._kill_cgroup_v2(
        4321,
        broker_pid=1234,
        proc_root=proc_root,
        cgroup_root=cgroup_root,
    )
    assert kill_file.read_text(encoding="utf-8") == ""


@pytest.mark.linux_only
def test_systemd_teardown_falls_back_when_systemctl_kill_fails(monkeypatch):
    broker = _load_broker()

    class FakeProcess:
        pid = 4321
        _hermes_systemd_unit = "test.scope"

    group_calls = []
    cgroup_calls = []
    monkeypatch.setattr(
        broker.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1),
    )
    monkeypatch.setattr(
        broker, "_signal_group", lambda pid, sig: group_calls.append((pid, sig))
    )
    monkeypatch.setattr(
        broker,
        "_kill_cgroup_v2",
        lambda pid: cgroup_calls.append(pid) or True,
    )

    broker._signal_process_tree(FakeProcess(), signal.SIGTERM)
    broker._signal_process_tree(FakeProcess(), signal.SIGKILL)

    assert group_calls == [(4321, signal.SIGTERM)]
    assert cgroup_calls == [4321]


@pytest.mark.linux_only
def test_systemd_contained_launch_and_teardown_target_the_whole_scope(monkeypatch):
    broker = _load_broker()
    popen_calls = []
    systemctl_calls = []

    class FakeProcess:
        pid = 4321
        returncode = None

        def wait(self, timeout=None):
            self.returncode = -signal.SIGKILL
            return self.returncode

    def fake_popen(argv, **kwargs):
        popen_calls.append((argv, kwargs))
        bootstrap = argv.index(broker._ENV_EXEC_BOOTSTRAP)
        os.write(int(argv[bootstrap + 2]), b"ready\n")
        return FakeProcess()

    def fake_run(argv, **kwargs):
        systemctl_calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(broker.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(broker.subprocess, "run", fake_run)
    monkeypatch.setattr(broker, "_wait_unreaped", lambda *_args: None)
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus")
    monkeypatch.setenv("BROKER_ONLY_SECRET", "must-not-cross")

    resolved = {
        "systemd-run": "/opt/systemd/bin/systemd-run",
        "env": "/opt/coreutils/bin/env",
    }
    monkeypatch.setattr(broker.shutil, "which", lambda name: resolved.get(name))
    containment = broker._resolve_systemd_containment()

    peer_env = {
        "PATH": "/usr/bin",
        "APP_SECRET": "approved-child-secret",
        "XDG_RUNTIME_DIR": "/peer/runtime",
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/peer/bus",
    }

    proc = broker._launch(
        None,
        peer_env,
        [],
        argv=["/bin/sh", "-c", "sleep 300"],
        cwd="/work",
        systemd_containment=containment,
    )
    broker._terminate(proc)

    launched_argv = popen_calls[0][0]
    assert launched_argv[:4] == [
        resolved["systemd-run"],
        "--user",
        "--scope",
        "--quiet",
    ]
    assert "--collect" in launched_argv
    unit_arg = next(arg for arg in launched_argv if arg.startswith("--unit="))
    separator = launched_argv.index("--")
    assert launched_argv[separator + 1] == resolved["env"]
    bootstrap = launched_argv.index(broker._ENV_EXEC_BOOTSTRAP)
    assert launched_argv[bootstrap - 3 : bootstrap] == [sys.executable, "-I", "-c"]
    assert all(value not in launched_argv for value in peer_env.values())
    launch_env = popen_calls[0][1]["env"]
    assert "APP_SECRET" not in launch_env
    assert launch_env["XDG_RUNTIME_DIR"] == "/run/user/1000"
    assert launch_env["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/run/user/1000/bus"
    assert "BROKER_ONLY_SECRET" not in launch_env
    assert launched_argv[-3:] == ["/bin/sh", "-c", "sleep 300"]
    unit = unit_arg.removeprefix("--unit=") + ".scope"
    assert [call[0] for call in systemctl_calls] == [
        [
            "systemctl",
            "--user",
            "kill",
            "--signal=SIGTERM",
            unit,
        ],
        [
            "systemctl",
            "--user",
            "kill",
            "--signal=SIGKILL",
            unit,
        ],
    ]
    assert all(call[1]["check"] is False for call in systemctl_calls)


@pytest.mark.linux_only
def test_systemd_bootstrap_closes_requested_stderr_source_fd():
    broker = _load_broker()
    env_fd = broker._child_env_memfd({})
    status_r, status_w = os.pipe2(os.O_CLOEXEC)
    stderr_r, stderr_w = os.pipe2(os.O_CLOEXEC)
    probe = """
import json
import os

target = os.readlink('/proc/self/fd/2')
duplicates = []
for name in os.listdir('/proc/self/fd'):
    try:
        if os.readlink(f'/proc/self/fd/{name}') == target:
            duplicates.append(int(name))
    except OSError:
        pass
print(json.dumps(sorted(duplicates)))
"""
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                broker._ENV_EXEC_BOOTSTRAP,
                str(env_fd),
                str(status_w),
                str(stderr_w),
                sys.executable,
                "-c",
                probe,
            ],
            pass_fds=(env_fd, status_w, stderr_w),
            stdout=subprocess.PIPE,
            check=True,
            text=True,
        )
        assert json.loads(result.stdout) == [2]
    finally:
        for fd in (env_fd, status_r, status_w, stderr_r, stderr_w):
            with contextlib.suppress(OSError):
                os.close(fd)


@pytest.mark.linux_only
def test_systemd_launcher_never_receives_peer_loader_environment(monkeypatch):
    broker = _load_broker()
    captured = {}

    class FakeProcess:
        pid = 4321

    def fake_popen(argv, **kwargs):
        captured.update(argv=argv, kwargs=kwargs)
        payload_fds = kwargs["pass_fds"]
        assert len(payload_fds) == 2
        captured["child_env"] = json.loads(os.pread(payload_fds[0], 1 << 20, 0))
        bootstrap = argv.index(broker._ENV_EXEC_BOOTSTRAP)
        os.write(int(argv[bootstrap + 2]), b"ready\n")
        return FakeProcess()

    monkeypatch.setattr(broker.subprocess, "Popen", fake_popen)
    containment = broker._SystemdContainment(
        "/opt/systemd/bin/systemd-run",
        "/opt/coreutils/bin/env",
        {"XDG_RUNTIME_DIR": "/run/user/1000"},
    )
    peer_env = {
        "PATH": "/peer/bin",
        "LD_PRELOAD": "/peer/attack.so",
        "LD_AUDIT": "/peer/audit.so",
        "LD_LIBRARY_PATH": "/peer/lib",
    }

    broker._launch(
        None,
        peer_env,
        [],
        argv=["/bin/true"],
        systemd_containment=containment,
    )

    assert captured["kwargs"]["env"] == {"XDG_RUNTIME_DIR": "/run/user/1000"}
    wrapper = captured["argv"].index(containment.env)
    assert captured["argv"][wrapper : wrapper + 2] == [containment.env, "-i"]
    assert captured["child_env"] == {
        **peer_env,
        broker.FDS_ENV: "",
    }
    for value in peer_env.values():
        assert value not in captured["argv"]


@pytest.mark.linux_only
def test_systemd_env_payload_round_trips_surrogateescaped_bytes(monkeypatch):
    broker = _load_broker()
    captured = {}

    class FakeProcess:
        pid = 4321

    def fake_popen(argv, **kwargs):
        env_fd = kwargs["pass_fds"][-2]
        captured["child_env"] = json.loads(os.pread(env_fd, 1 << 20, 0))
        bootstrap = argv.index(broker._ENV_EXEC_BOOTSTRAP)
        os.write(int(argv[bootstrap + 2]), b"ready\n")
        return FakeProcess()

    monkeypatch.setattr(broker.subprocess, "Popen", fake_popen)
    containment = broker._SystemdContainment("/systemd-run", "/env", {})

    broker._launch(
        None,
        {"RAW_BYTE": "\udcff"},
        [],
        argv=["/bin/true"],
        systemd_containment=containment,
    )

    assert captured["child_env"]["RAW_BYTE"] == "\udcff"


@pytest.mark.linux_only
def test_systemd_env_payload_fd_closes_when_serialization_fails(monkeypatch):
    broker = _load_broker()
    baseline = _fd_count(os.getpid())
    monkeypatch.setattr(
        broker.json,
        "dump",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("serialization failed")
        ),
    )

    with pytest.raises(RuntimeError, match="serialization failed"):
        broker._launch(
            None,
            {},
            [],
            argv=["/bin/true"],
            systemd_containment=broker._SystemdContainment("/systemd-run", "/env", {}),
        )

    assert _fd_count(os.getpid()) == baseline


@pytest.mark.linux_only
def test_contained_launcher_failure_is_a_structured_refusal(tmp_path, monkeypatch):
    broker = _load_broker()
    launcher = tmp_path / "failing-systemd-run"
    launcher.write_text(
        "#!/bin/sh\necho scope setup failed >&2\nexit 1\n", encoding="utf-8"
    )
    launcher.chmod(0o700)

    class Connection:
        def __init__(self):
            self.replies = []

        def getsockopt(self, *_args):
            return broker._UCRED.pack(os.getpid(), os.geteuid(), os.getegid())

        def sendall(self, payload):
            self.replies.append(json.loads(payload))

        def close(self):
            pass

    conn = Connection()
    leases = broker._Leases()
    assert leases.register(conn, threading.current_thread())
    monkeypatch.setattr(
        broker,
        "_recv_request",
        lambda *_args: (
            None,
            False,
            None,
            ["/bin/true"],
            str(tmp_path),
            {},
            {},
            False,
            [],
        ),
    )
    monkeypatch.setattr(broker, "_await_lease_end", lambda *_args: None)
    monkeypatch.setattr(broker, "_unreaped_returncode", lambda _pid: 1)

    broker._serve_connection(
        conn,
        str(_staging_root(tmp_path)),
        HANDSHAKE_TIMEOUT,
        leases,
        systemd_containment=broker._SystemdContainment(
            str(launcher), shutil.which("env") or "/usr/bin/env", {}
        ),
    )

    assert conn.replies == [
        {"ok": False, "error": "launch_failed", "message": "scope setup failed"}
    ]


@pytest.mark.linux_only
def test_scopes_bind_only_to_the_current_service(tmp_path, monkeypatch):
    broker = _load_broker()
    proc_root = tmp_path / "proc"
    cgroup = proc_root / "123" / "cgroup"
    cgroup.parent.mkdir(parents=True)
    cgroup.write_text(
        "malformed\n"
        "11:pids:/user.slice/user-1000.slice/user@1000.service\n"
        f"0::/user.slice/user-{os.geteuid()}.slice/"
        f"user@{os.geteuid()}.service/app.slice/"
        "hermes-local-exec-broker.service\n",
        encoding="utf-8",
    )
    assert broker._current_service_unit(pid=123, proc_root=proc_root) == (
        "hermes-local-exec-broker.service"
    )

    cgroup.write_text(
        "0::/system.slice/hermes-local-exec-broker.service\n",
        encoding="utf-8",
    )
    assert broker._current_service_unit(pid=123, proc_root=proc_root) is None

    launches = []

    class FakeProcess:
        pid = 4321

    def fake_popen(argv, **_kwargs):
        launches.append(argv)
        bootstrap = argv.index(broker._ENV_EXEC_BOOTSTRAP)
        os.write(int(argv[bootstrap + 2]), b"ready\n")
        return FakeProcess()

    monkeypatch.setattr(broker.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        broker, "_current_service_unit", lambda: "hermes-local-exec-broker.service"
    )
    containment = broker._SystemdContainment("/systemd-run", "/env", {})
    broker._launch(None, {}, [], argv=["/bin/true"], systemd_containment=containment)
    assert "--property=BindsTo=hermes-local-exec-broker.service" in launches[-1]
    assert "--property=After=hermes-local-exec-broker.service" in launches[-1]

    monkeypatch.setattr(broker, "_current_service_unit", lambda: None)
    broker._launch(None, {}, [], argv=["/bin/true"], systemd_containment=containment)
    assert not any(arg.startswith("--property=BindsTo=") for arg in launches[-1])
    assert not any(arg.startswith("--property=After=") for arg in launches[-1])


@pytest.mark.linux_only
def test_scope_kill_uses_cross_version_systemctl_command(monkeypatch):
    broker = _load_broker()
    calls = []

    class FakeProcess:
        pid = 4321
        _hermes_systemd_unit = "hermes-local-exec-test.scope"

    def successful(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(broker.subprocess, "run", successful)
    broker._signal_process_tree(FakeProcess(), signal.SIGTERM)

    assert calls[0][0] == [
        "systemctl",
        "--user",
        "kill",
        "--signal=SIGTERM",
        "hermes-local-exec-test.scope",
    ]
    assert calls[0][1]["timeout"] == broker._SYSTEMCTL_TIMEOUT_SECONDS

    groups = []
    monkeypatch.setattr(
        broker.subprocess,
        "run",
        lambda argv, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(argv, kwargs["timeout"])
        ),
    )
    monkeypatch.setattr(
        broker, "_signal_group", lambda pid, sig: groups.append((pid, sig))
    )
    broker._signal_process_tree(FakeProcess(), signal.SIGTERM)
    assert groups == [(4321, signal.SIGTERM)]


@pytest.mark.linux_only
def test_requested_systemd_containment_fails_closed_without_user_manager(monkeypatch):
    broker = _load_broker()

    def unavailable(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv, stderr="Failed to connect to bus")

    monkeypatch.setattr(broker.subprocess, "run", unavailable)

    with pytest.raises(SystemExit, match="systemd cgroup containment is unavailable"):
        broker._require_systemd_cgroup(
            broker._SystemdContainment("/custom/systemd-run", "/custom/env", {})
        )


@pytest.mark.linux_only
def test_systemd_containment_probe_uses_a_valid_scope_invocation(monkeypatch):
    broker = _load_broker()
    captured = []

    def fake_run(argv, **kwargs):
        captured.append((argv, kwargs))
        if "--scope" in argv and "--wait" in argv:
            raise subprocess.CalledProcessError(
                1, argv, stderr="--wait may not be combined with --scope"
            )
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(broker.subprocess, "run", fake_run)

    broker._require_systemd_cgroup(
        broker._SystemdContainment("/custom/systemd-run", "/custom/env", {})
    )

    assert "--scope" in captured[0][0]
    assert "--wait" not in captured[0][0]
    assert captured[0][0][0] == "/custom/systemd-run"


@pytest.mark.linux_only
def test_systemd_cgroup_option_reaches_broker_server(tmp_path, monkeypatch):
    broker = _load_broker()
    captured = {}

    def fake_serve(socket_path, staging_root, **kwargs):
        captured.update(
            socket_path=socket_path,
            staging_root=staging_root,
            **kwargs,
        )

    monkeypatch.setattr(broker, "serve", fake_serve)

    assert (
        broker.main([
            "--socket",
            str(tmp_path / "broker.sock"),
            "--staging-root",
            str(tmp_path),
            "--systemd-cgroup",
        ])
        == 0
    )
    assert captured["systemd_cgroup"] is True


@pytest.mark.linux_only
def test_install_user_service_cli_uses_operator_owned_unit_directory(
    tmp_path, monkeypatch
):
    broker = _load_broker()
    config_home = tmp_path / "config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    captured = {}

    def fake_install(**kwargs):
        captured.update(kwargs)
        return kwargs["service_dir"] / broker.USER_SERVICE_NAME

    monkeypatch.setattr(broker, "_install_user_service", fake_install)
    monkeypatch.setattr(
        broker,
        "serve",
        lambda *_args, **_kwargs: pytest.fail("install mode must not start inline"),
    )

    assert broker.main(["--install-user-service"]) == 0
    assert captured["service_dir"] == config_home / "systemd" / "user"
    assert captured["python_path"] == sys.executable
    assert captured["allowed_uids"] == frozenset({os.geteuid()})
    assert captured["socket_mode"] == 0o600


@pytest.mark.linux_only
def test_install_user_service_rejects_broader_socket_mode(monkeypatch):
    broker = _load_broker()
    monkeypatch.setattr(
        broker,
        "_install_user_service",
        lambda **_kwargs: pytest.fail("invalid install options must not write a unit"),
    )

    with pytest.raises(SystemExit):
        broker.main(["--install-user-service", "--socket-mode", "0666"])


@pytest.mark.linux_only
def test_install_user_service_rejects_cross_uid_allowlist(monkeypatch):
    broker = _load_broker()
    monkeypatch.setattr(
        broker,
        "_install_user_service",
        lambda **_kwargs: pytest.fail("invalid install options must not write a unit"),
    )

    with pytest.raises(SystemExit):
        broker.main(["--install-user-service", "--allow-uid", str(os.geteuid() + 1)])


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
def test_systemd_cgroup_lease_kills_setsid_descendant(tmp_path, monkeypatch):
    """A new session escapes a process group, but not the command's systemd scope."""
    _require_systemd_user_scopes()

    root = _staging_root(tmp_path)
    proc, sock_path = _start_broker(
        "broker-only-secret",
        root,
        allowed_uids=[os.getuid()],
        systemd_cgroup=True,
    )
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        f"terminal:\n  local_exec_broker:\n    socket: {sock_path}\n    uid: {os.geteuid()}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("LOCAL_EXEC_VISIBLE", "approved")

    env = None
    escaped_pid = None
    try:
        from tools.environments.local import LocalEnvironment

        env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
        result = env.execute(
            "setsid sleep 300 </dev/null >/dev/null 2>&1 & "
            "printf '__setsid__%s\\n' \"$!\"; "
            "if printenv BROKER_PROBE_HOST_ONLY >/dev/null; then a=present; else a=unset; fi; "
            "if printenv DBUS_SESSION_BUS_ADDRESS >/dev/null; then b=present; else b=unset; fi; "
            'printf \'__transport__%s:%s:%s\\n\' "$a" "$b" "$LOCAL_EXEC_VISIBLE"'
        )
        assert "__setsid__" in result["output"], result
        assert "__transport__unset:unset:approved" in result["output"], result
        escaped_pid = int(
            next(
                line.removeprefix("__setsid__")
                for line in result["output"].splitlines()
                if line.startswith("__setsid__")
            )
        )

        assert result["returncode"] == 0
        assert _wait_until_gone(escaped_pid)
    finally:
        if env is not None:
            env.cleanup()
        if escaped_pid is not None and _pid_running(escaped_pid):
            os.kill(escaped_pid, signal.SIGKILL)
        _stop_broker(proc)


@pytest.mark.linux_only
def test_local_environment_opt_in_executes_through_broker(tmp_path, monkeypatch):
    """The configured local terminal path is a real broker lease, not a fallback."""
    from tools.environments.local import LocalEnvironment

    root = _staging_root(tmp_path)
    proc, sock_path = _start_broker(
        "broker-only-secret",
        root,
        allowed_uids=[os.getuid()],
    )
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        f"terminal:\n  local_exec_broker:\n    socket: {sock_path}\n    uid: {os.geteuid()}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("LOCAL_EXEC_VISIBLE", "approved")
    monkeypatch.setenv("OPENAI_API_KEY", "must-be-scrubbed")

    env = None
    try:
        env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
        result = env.execute(
            "read value; printf 'out:%s uid:%s ppid:%s cwd:%s env:%s secret:%s fds:%s\\n' "
            '"$value" "$(id -u)" "$PPID" "$PWD" "$LOCAL_EXEC_VISIBLE" '
            '"${OPENAI_API_KEY-unset}" "${HERMES_BROKER_FDS:-empty}"; '
            'printf "err\\n" >&2; exit 7',
            stdin_data="input payload\n",
        )

        assert result["returncode"] == 7
        assert (
            f"out:input payload uid:{os.getuid()} ppid:{proc.pid} cwd:{tmp_path} "
            "env:approved secret:unset fds:empty\nerr\n"
        ) in result["output"]
    finally:
        if env is not None:
            env.cleanup()
        _stop_broker(proc)


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
def test_execute_code_session_kernel_uses_broker_lease_and_project_runtime(
    tmp_path, monkeypatch
):
    """The persistent project kernel crosses the real broker boundary without semantic drift."""
    from tools.code_execution_tool import execute_code
    from tools.code_kernel import _KERNELS, shutdown_all_kernels

    root = _staging_root(tmp_path)
    proc, sock_path = _start_broker(
        "broker-only-secret", root, allowed_uids=[os.getuid()]
    )
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "code_execution:\n"
        "  mode: project\n"
        "  timeout: 30\n"
        "terminal:\n"
        "  local_exec_broker:\n"
        f"    socket: {sock_path}\n"
        f"    uid: {os.geteuid()}\n",
        encoding="utf-8",
    )
    project = tmp_path / "project"
    project.mkdir()
    venv = tmp_path / "project-venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("VIRTUAL_ENV", str(venv))
    monkeypatch.chdir(project)

    kernel_pid = None
    try:
        first = json.loads(
            execute_code(
                "import os, sys\n"
                "value = 41\n"
                "print(os.getpid(), os.getppid(), os.getcwd(), sys.prefix)",
                task_id="broker-kernel",
            )
        )
        assert first["status"] == "success", first
        report = first["output"].strip().split()
        kernel_pid = int(report[0])
        assert int(report[1]) == proc.pid
        assert report[2] == str(project)
        assert os.path.realpath(report[3]) == os.path.realpath(venv)

        second = json.loads(execute_code("print(value + 1)", task_id="broker-kernel"))
        assert second["status"] == "success", second
        assert second["output"].strip() == "42"
        assert second["kernel"]["reused"] is True

        shutdown_all_kernels()
        assert _wait_until_gone(kernel_pid)
        assert not _KERNELS
        kernel_pid = None

        live = json.loads(
            execute_code("import os; print(os.getpid())", task_id="broker-loss")
        )
        assert live["status"] == "success", live
        lost_pid = int(live["output"].strip())
        lost_kernel = next(iter(_KERNELS.values()))
        _stop_broker(proc)
        proc = None
        assert _wait_until_gone(lost_pid)
        assert lost_kernel.alive() is False
        shutdown_all_kernels()
        assert not _KERNELS

        forbidden = tmp_path / "direct-fallback-ran"
        failed = json.loads(
            execute_code(
                f"from pathlib import Path; Path({str(forbidden)!r}).touch()",
                task_id="broker-refusal",
            )
        )
        assert failed["status"] == "error", failed
        assert not forbidden.exists(), "configured broker failure launched directly"
    finally:
        shutdown_all_kernels()
        if proc is not None:
            _stop_broker(proc)
        if kernel_pid is not None:
            assert _wait_until_gone(kernel_pid)


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
def test_process_registry_background_survives_foreground_broker_lease(
    tmp_path, monkeypatch
):
    """The real hermes_bg entry point escapes only its worker from the command lease."""
    from tools.environments.local import LocalEnvironment
    from tools.process_registry import ProcessRegistry

    _require_systemd_user_scopes()
    root = _staging_root(tmp_path)
    proc, sock_path = _start_broker(
        "broker-only-secret",
        root,
        allowed_uids=[os.getuid()],
        systemd_cgroup=True,
    )
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        f"terminal:\n  local_exec_broker:\n    socket: {sock_path}\n    uid: {os.geteuid()}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    env = None
    session = None
    escaped_pid = None
    try:
        env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
        registry = ProcessRegistry()
        escaped_path = tmp_path / "escaped.pid"
        session = registry.spawn_via_env(
            env,
            f"setsid sleep 300 </dev/null >/dev/null 2>&1 & "
            f"echo $! > {shlex.quote(str(escaped_path))}; sleep 300",
        )

        assert session.pid is not None
        assert _pid_running(session.pid)
        assert os.getsid(session.pid) == session.pid
        time.sleep(0.2)
        assert _pid_running(session.pid), (
            "systemd lease teardown killed the background worker"
        )
        deadline = time.monotonic() + DEADLINE
        while not escaped_path.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert escaped_path.exists()
        escaped_pid = int(escaped_path.read_text(encoding="utf-8"))
        assert _pid_running(escaped_pid)
        monkeypatch.setattr(
            "tools.process_registry._stop_systemd_unit",
            lambda _unit: pytest.fail("broker-owned scope was stopped locally"),
        )

        result = registry.kill_process(session.id)

        assert result["status"] == "killed"
        assert _wait_until_gone(escaped_pid)
    finally:
        if session is not None and session.pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(session.pid, signal.SIGKILL)
        if escaped_pid is not None and _pid_running(escaped_pid):
            os.kill(escaped_pid, signal.SIGKILL)
        if env is not None:
            env.cleanup()
        _stop_broker(proc)


@pytest.mark.linux_only
def test_local_environment_broker_accepts_kernel_sized_command_payload(
    tmp_path, monkeypatch
):
    """A normal command below Linux's per-argument limit crosses the full terminal path."""
    from tools.environments.local import LocalEnvironment

    root = _staging_root(tmp_path)
    proc, sock_path = _start_broker(
        "broker-only-secret", root, allowed_uids=[os.getuid()]
    )
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        f"terminal:\n  local_exec_broker:\n    socket: {sock_path}\n    uid: {os.geteuid()}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    env = None
    try:
        env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
        result = env.execute(": #" + "x" * 100_000)
        assert result["returncode"] == 0
    finally:
        if env is not None:
            env.cleanup()
        _stop_broker(proc)


@pytest.mark.linux_only
@pytest.mark.parametrize("systemd_cgroup", (False, True))
def test_local_environment_reports_execve_oversize_as_payload_error(
    tmp_path, monkeypatch, systemd_cgroup
):
    """An E2BIG request is a payload refusal; it does not poison the broker."""
    from tools.environments.base import EnvironmentConnectionError
    from tools.environments.local import LocalEnvironment

    root = _staging_root(tmp_path)
    if systemd_cgroup:
        _require_systemd_user_scopes()
    proc, sock_path = _start_broker(
        "broker-only-secret",
        root,
        allowed_uids=[os.getuid()],
        systemd_cgroup=systemd_cgroup,
    )
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        f"terminal:\n  local_exec_broker:\n    socket: {sock_path}\n    uid: {os.geteuid()}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    env = None
    try:
        env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
        env.env["HERMES_OVERSIZED_ENV"] = "x" * 4_000_000
        with pytest.raises(EnvironmentConnectionError) as excinfo:
            env.execute(":")

        assert "request is too large" in excinfo.value.reason
        assert "broker is unavailable" not in excinfo.value.reason
        assert "Reduce the command or environment payload" in excinfo.value.retry_hint

        env.env.pop("HERMES_OVERSIZED_ENV")
        result = env.execute("printf ok")
        assert result["returncode"] == 0
        assert result["output"] == "ok"
    finally:
        if env is not None:
            env.cleanup()
        _stop_broker(proc)


@pytest.mark.linux_only
@pytest.mark.parametrize("expected_peer_uid", (None, -1, True, "1000"))
def test_request_launch_rejects_invalid_expected_peer_uid_before_socket(
    monkeypatch, expected_peer_uid
):
    """Direct callers cannot weaken the configured broker identity type."""
    broker = _load_broker()

    def fail_socket(*_args):
        pytest.fail("invalid broker uid must be rejected before opening a socket")

    monkeypatch.setattr(broker.socket, "socket", fail_socket)

    with pytest.raises(broker.BrokerError) as excinfo:
        broker.request_launch(
            "test-only",
            expected_peer_uid=expected_peer_uid,
            argv=["/bin/true"],
            cwd="/",
            env={},
            fds=[],
        )

    assert excinfo.value.code == "peer_authentication_failed"


@pytest.mark.linux_only
def test_request_launch_rejects_oversized_frame_before_sending_payload(monkeypatch):
    """The client reports the deliberate cap without leaking a partial request."""
    broker = _load_broker()

    class Connection:
        closed = False

        def settimeout(self, _timeout):
            pass

        def connect(self, _path):
            pass

        def sendmsg(self, _buffers, _ancillary):
            pytest.fail(
                "an oversized request must be rejected before payload transmission"
            )

        def close(self):
            self.closed = True

    conn = Connection()
    monkeypatch.setattr(broker, "MAX_REQUEST_BYTES", 64)
    monkeypatch.setattr(broker.socket, "socket", lambda *_args: conn)

    with pytest.raises(broker.BrokerError) as excinfo:
        broker.request_launch(
            "test-only",
            expected_peer_uid=os.geteuid(),
            argv=["/bin/bash", "-c", "x" * 64],
            cwd="/",
            env={},
            fds=[],
        )

    assert excinfo.value.code == "request_too_large"
    assert "64-byte frame limit" in excinfo.value.message
    assert conn.closed


@pytest.mark.linux_only
def test_request_launch_authenticates_peer_before_sending_payload(
    tmp_path, monkeypatch
):
    """Socket ownership and SO_PEERCRED must agree before env or stdin crosses."""
    broker = _load_broker()
    socket_dir = tmp_path / "broker"
    socket_dir.mkdir(mode=0o700)
    sock_path = str(socket_dir / "broker.sock")
    real_lstat = os.lstat
    socket_info = os.stat_result((
        stat.S_IFSOCK | 0o600,
        4242,
        1,
        1,
        os.geteuid(),
        os.getegid(),
        0,
        0,
        0,
        0,
    ))

    def fake_lstat(path):
        return socket_info if os.fspath(path) == sock_path else real_lstat(path)

    class ImpostorConnection:
        closed = False

        def settimeout(self, _timeout):
            pass

        def connect(self, _path):
            pass

        def getsockopt(self, _level, _kind, _size):
            return broker._UCRED.pack(os.getpid(), os.geteuid() + 1, os.getegid())

        def sendmsg(self, _buffers, _ancillary):
            pytest.fail("peer authentication must happen before payload transmission")

        def close(self):
            self.closed = True

    conn = ImpostorConnection()
    monkeypatch.setattr(broker.os, "lstat", fake_lstat)
    monkeypatch.setattr(broker.socket, "socket", lambda *_args: conn)
    with pytest.raises(broker.BrokerError) as excinfo:
        broker.request_launch(
            sock_path,
            expected_peer_uid=os.geteuid(),
            argv=["/bin/true"],
            cwd="/",
            env={"APPROVED_SECRET": "must-not-cross"},
            fds=[],
        )

    assert excinfo.value.code == "peer_authentication_failed"
    assert conn.closed


@pytest.mark.linux_only
def test_request_launch_refuses_unexpected_socket_owner_before_sending_payload(
    tmp_path, monkeypatch
):
    """Configured broker identity must be checked before env or stdin crosses."""
    broker = _load_broker()
    socket_dir = tmp_path / "broker"
    socket_dir.mkdir(mode=0o700)
    sock_path = str(socket_dir / "broker.sock")
    real_lstat = os.lstat
    socket_info = os.stat_result((
        stat.S_IFSOCK | 0o600,
        4242,
        1,
        1,
        os.geteuid(),
        os.getegid(),
        0,
        0,
        0,
        0,
    ))

    def fake_lstat(path):
        return socket_info if os.fspath(path) == sock_path else real_lstat(path)

    class UnexpectedOwnerConnection:
        closed = False

        def settimeout(self, _timeout):
            pass

        def connect(self, _path):
            pass

        def getsockopt(self, _level, _kind, _size):
            return broker._UCRED.pack(os.getpid(), os.geteuid(), os.getegid())

        def sendmsg(self, _buffers, _ancillary):
            pytest.fail(
                "unexpected broker identity must be rejected before transmission"
            )

        def close(self):
            self.closed = True

    conn = UnexpectedOwnerConnection()
    monkeypatch.setattr(broker.os, "lstat", fake_lstat)
    monkeypatch.setattr(broker.socket, "socket", lambda *_args: conn)

    with pytest.raises(broker.BrokerError) as excinfo:
        broker.request_launch(
            sock_path,
            expected_peer_uid=os.geteuid() + 1,
            argv=["/bin/true"],
            cwd="/",
            env={"APPROVED_SECRET": "must-not-cross"},
            fds=[],
        )

    assert excinfo.value.code == "peer_authentication_failed"
    assert conn.closed


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
def test_local_environment_broker_sweeps_descendants_after_shell_exit(
    tmp_path, monkeypatch
):
    """A completed shell must not let a background descendant outlive its broker lease."""
    from tools.environments.local import LocalEnvironment

    root = _staging_root(tmp_path)
    proc, sock_path = _start_broker(
        "broker-only-secret",
        root,
        allowed_uids=[os.getuid()],
    )
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        f"terminal:\n  local_exec_broker:\n    socket: {sock_path}\n    uid: {os.geteuid()}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    env = None
    child_pid = None
    try:
        env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
        result = env.execute(
            "sleep 300 </dev/null >/dev/null 2>&1 & printf '__child__%s\\n' \"$!\""
        )
        child_pid = int(
            next(
                line.removeprefix("__child__")
                for line in result["output"].splitlines()
                if line.startswith("__child__")
            )
        )

        assert result["returncode"] == 0
        assert _wait_until_gone(child_pid)
    finally:
        if env is not None:
            env.cleanup()
        if child_pid is not None and _pid_running(child_pid):
            os.kill(child_pid, signal.SIGKILL)
        _stop_broker(proc)


@pytest.mark.linux_only
def test_local_environment_reports_midflight_broker_death(tmp_path, monkeypatch):
    """Lease EOF without an exit frame is a broker failure, not command return code -1."""
    from tools.environments.base import EnvironmentConnectionError
    from tools.environments.local import LocalEnvironment

    root = _staging_root(tmp_path)
    proc, sock_path = _start_broker(
        "broker-only-secret",
        root,
        allowed_uids=[os.getuid()],
    )
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        f"terminal:\n  local_exec_broker:\n    socket: {sock_path}\n    uid: {os.geteuid()}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    marker = tmp_path / "child-started"

    env = LocalEnvironment(cwd=str(tmp_path), timeout=10)

    def kill_broker_after_launch():
        deadline = time.monotonic() + DEADLINE
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        proc.kill()

    killer = threading.Thread(target=kill_broker_after_launch)
    killer.start()
    try:
        with pytest.raises(
            EnvironmentConnectionError,
            match="configured local execution broker failed during command execution",
        ):
            env.execute(
                f"touch {shlex.quote(str(marker))}; "
                'while kill -0 "$PPID" 2>/dev/null; do sleep 0.05; done'
            )
    finally:
        killer.join(timeout=DEADLINE)
        env.cleanup()
        _stop_broker(proc)
    assert not killer.is_alive()


@pytest.mark.linux_only
def test_local_environment_closes_lease_when_broker_reply_is_malformed(
    tmp_path, monkeypatch
):
    """Handle-construction failure must close the lease and every locally owned descriptor."""
    from tools.environments.base import EnvironmentConnectionError
    from tools.environments.local import LocalEnvironment
    import tools.local_exec_broker as broker

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    env = LocalEnvironment(cwd=str(tmp_path), timeout=1)
    env._local_exec_broker_socket = "test-only"
    env._local_exec_broker_uid = os.geteuid()
    client, peer = socket.socketpair()
    baseline = _fd_count(os.getpid())

    def malformed_launch(*_args, **_kwargs):
        return client, {"ok": True}, b""

    monkeypatch.setattr(broker, "request_launch", malformed_launch)
    try:
        with pytest.raises(
            EnvironmentConnectionError,
            match="returned an invalid launch reply",
        ):
            env._run_bash("true")
        peer.settimeout(DEADLINE)
        assert peer.recv(1) == b""
        assert _fd_count(os.getpid()) == baseline - 1
    finally:
        env.cleanup()
        client.close()
        peer.close()


@pytest.mark.linux_only
@pytest.mark.parametrize(
    "broker_yaml",
    ("{}", "false", "{socket: ''}", "{uid: 0}"),
)
def test_present_invalid_local_exec_broker_config_fails_closed(
    tmp_path, monkeypatch, broker_yaml
):
    """Once the broker section exists, an invalid socket cannot opt back into direct Popen."""
    from tools.environments.base import EnvironmentConnectionError
    from tools.environments.local import LocalEnvironment

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        f"terminal:\n  local_exec_broker: {broker_yaml}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with pytest.raises(
        EnvironmentConnectionError,
        match="terminal.local_exec_broker requires a non-empty string socket",
    ):
        LocalEnvironment(cwd=str(tmp_path), timeout=1)


def test_nonlinux_local_environment_ignores_broker_config(tmp_path, monkeypatch):
    import tools.environments.local as local

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "terminal:\n"
        "  local_exec_broker:\n"
        "    socket: /run/hermes-broker/broker.sock\n"
        f"    uid: {os.geteuid()}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(local.sys, "platform", "darwin")

    env = local.LocalEnvironment(cwd=str(tmp_path), timeout=1)
    try:
        assert env._local_exec_broker_socket is None
        assert env._local_exec_broker_uid is None
    finally:
        env.cleanup()


@pytest.mark.linux_only
@pytest.mark.parametrize("broker_uid", ("null", "-1", "true", "'1000'"))
def test_local_exec_broker_config_requires_non_negative_integer_uid(
    tmp_path, monkeypatch, broker_uid
):
    """Missing, negative, bool, and string UIDs cannot weaken broker identity."""
    from tools.environments.base import EnvironmentConnectionError
    from tools.environments.local import LocalEnvironment

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "terminal:\n"
        "  local_exec_broker:\n"
        "    socket: /run/hermes-broker/broker.sock\n"
        f"    uid: {broker_uid}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with pytest.raises(
        EnvironmentConnectionError,
        match="terminal.local_exec_broker requires a non-negative integer uid",
    ):
        LocalEnvironment(cwd=str(tmp_path), timeout=1)


@pytest.mark.linux_only
def test_local_environment_configured_broker_failure_never_falls_back(
    tmp_path, monkeypatch
):
    """A configured broker is a required boundary, not a best-effort launch path."""
    from tools.environments.base import EnvironmentConnectionError
    from tools.environments.local import LocalEnvironment

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    missing_socket = _socket_path()
    (hermes_home / "config.yaml").write_text(
        f"terminal:\n  local_exec_broker:\n    socket: {missing_socket}\n    uid: {os.geteuid()}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    env = LocalEnvironment(cwd=str(tmp_path), timeout=1)
    try:
        with pytest.raises(
            EnvironmentConnectionError,
            match="configured local execution broker is unavailable",
        ):
            env.execute("printf should-not-run")
    finally:
        env.cleanup()
        shutil.rmtree(Path(missing_socket).parent)


@pytest.mark.linux_only
def test_request_launch_preserves_coalesced_exit_frame(monkeypatch):
    """A launch reply and exit frame from one recv must cross into the process handle."""
    broker = _load_broker()
    from tools.environments.local import _BrokerProcessHandle

    class CoalescedConnection:
        closed = False

        def settimeout(self, _timeout):
            pass

        def connect(self, _path):
            pass

        def sendmsg(self, buffers, _ancillary):
            return len(buffers[0])

        def recv(self, _size):
            return b'{"ok": true, "pid": 123}\n{"exit": 7}\n'

        def setblocking(self, _blocking):
            pass

        def close(self):
            self.closed = True

    conn = CoalescedConnection()
    monkeypatch.setattr(broker.socket, "socket", lambda *_args: conn)
    monkeypatch.setattr(
        broker, "_validated_client_socket", lambda _path, _uid: object()
    )
    monkeypatch.setattr(
        broker,
        "_authenticate_broker_peer",
        lambda _conn, _path, _info, _uid: None,
    )

    returned_conn, reply, remainder = broker.request_launch(
        "test-only",
        expected_peer_uid=os.geteuid(),
        argv=["true"],
        cwd="/",
        env={},
        fds=[],
    )

    assert returned_conn is conn
    assert reply == {"ok": True, "pid": 123}
    assert remainder == b'{"exit": 7}\n'
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    handle = _BrokerProcessHandle(conn, reply["pid"], read_fd, None, remainder)
    try:
        assert handle.poll() == 7
        assert conn.closed
    finally:
        handle.stdout.close()


@pytest.mark.linux_only
def test_broker_process_poll_normalizes_connection_reset():
    """A reset lease is the same typed broker failure as a clean unexpected EOF."""
    from tools.environments.base import EnvironmentConnectionError
    from tools.environments.local import _BrokerProcessHandle

    class ResetConnection:
        closed = False

        def setblocking(self, _blocking):
            pass

        def recv(self, _size):
            raise ConnectionResetError(errno.ECONNRESET, "reset by broker")

        def close(self):
            self.closed = True

    conn = ResetConnection()
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    stdin_r, stdin_w = os.pipe()
    handle = _BrokerProcessHandle(conn, 123, read_fd, stdin_w)
    try:
        with pytest.raises(
            EnvironmentConnectionError,
            match="configured local execution broker failed during command execution",
        ):
            handle.poll()
        assert conn.closed
        assert handle.stdin is not None and handle.stdin.closed
    finally:
        os.close(stdin_r)
        if not handle.stdout.closed:
            handle.stdout.close()
        if handle.stdin is not None and not handle.stdin.closed:
            handle.stdin.close()


@pytest.mark.linux_only
def test_local_environment_does_not_reclose_transferred_pipe_fd_when_handle_init_fails(
    tmp_path, monkeypatch
):
    """Once the handle owns a pipe, outer launch cleanup cannot close that fd again."""
    import tools.environments.local as local_module
    from tools.environments.local import LocalEnvironment

    class Connection:
        def close(self):
            pass

    reused = []

    def fake_request_launch(*_args, **_kwargs):
        return Connection(), {"pid": 123}, b""

    def failing_handle(_conn, _pid, stdout_fd, _stdin_fd, _remainder):
        os.close(stdout_fd)
        reused.append(os.open("/dev/null", os.O_RDONLY))
        raise RuntimeError("constructor failed after taking descriptor ownership")

    monkeypatch.setattr("tools.local_exec_broker.request_launch", fake_request_launch)
    monkeypatch.setattr(local_module, "_BrokerProcessHandle", failing_handle)
    env = LocalEnvironment.__new__(LocalEnvironment)
    env.cwd = str(tmp_path)
    env.env = {}
    env._local_exec_broker_socket = "/run/test-broker.sock"
    env._local_exec_broker_uid = os.geteuid()

    try:
        with pytest.raises(RuntimeError, match="constructor failed"):
            env._run_bash("true")
        os.fstat(reused[0])
    finally:
        if reused:
            with contextlib.suppress(OSError):
                os.close(reused[0])


@pytest.mark.linux_only
def test_broker_process_kill_preserves_buffered_stdout_for_caller():
    """A timeout kills the lease without discarding output already in the pipe."""
    from tools.environments.local import _BrokerProcessHandle

    conn, peer = socket.socketpair()
    stdout_r, stdout_w = os.pipe()
    stdin_r, stdin_w = os.pipe()
    handle = _BrokerProcessHandle(conn, 123, stdout_r, stdin_w)
    try:
        os.write(stdout_w, b"partial output\n")
        handle.kill()
        os.close(stdout_w)
        stdout_w = -1

        assert handle.stdout.read() == "partial output\n"
        assert not handle.stdout.closed
        assert handle.stdin is not None and handle.stdin.closed
    finally:
        peer.close()
        os.close(stdin_r)
        if stdout_w != -1:
            os.close(stdout_w)
        if not handle.stdout.closed:
            handle.stdout.close()
        if handle.stdin is not None and not handle.stdin.closed:
            handle.stdin.close()


@pytest.mark.linux_only
def test_local_environment_kills_broker_handle_through_lease(monkeypatch):
    from tools.environments import local as local_module
    from tools.environments.local import LocalEnvironment, _BrokerProcessHandle

    conn, peer = socket.socketpair()
    stdout_r, stdout_w = os.pipe()
    os.close(stdout_w)
    handle = _BrokerProcessHandle(conn, 123, stdout_r, None)
    monkeypatch.setattr(
        local_module,
        "_kill_process_group_posix",
        lambda _proc: pytest.fail("broker handles must not use PID-based teardown"),
    )
    try:
        LocalEnvironment.__new__(LocalEnvironment)._kill_process(handle)
        assert handle.poll() == -signal.SIGKILL
    finally:
        peer.close()
        handle.stdout.close()


@pytest.mark.linux_only
def test_local_environment_force_kills_broker_handle_through_lease(monkeypatch):
    from tools.environments import local as local_module
    from tools.environments.local import LocalEnvironment, _BrokerProcessHandle

    conn, peer = socket.socketpair()
    stdout_r, stdout_w = os.pipe()
    os.close(stdout_w)
    handle = _BrokerProcessHandle(conn, 123, stdout_r, None)
    monkeypatch.setattr(
        local_module.os,
        "getpgid",
        lambda _pid: pytest.fail("broker handles must not inspect a host process group"),
    )
    try:
        LocalEnvironment.__new__(LocalEnvironment)._force_kill_process(handle)  # type: ignore[arg-type]
        assert handle.poll() == -signal.SIGKILL
    finally:
        peer.close()
        handle.stdout.close()


@pytest.mark.linux_only
@pytest.mark.parametrize(
    "remainder",
    (
        b"not-json\n",
        b'{"status": 7}\n',
        b'{"exit": true}\n',
        b'{"exit": "7"}\n',
        b'{"exit": 7.9}\n',
    ),
)
def test_broker_process_poll_rejects_malformed_buffered_exit_frame(remainder):
    """Malformed buffered status is a typed broker failure that closes its lease."""
    from tools.environments.base import EnvironmentConnectionError
    from tools.environments.local import _BrokerProcessHandle

    class Connection:
        closed = False

        def setblocking(self, _blocking):
            pass

        def recv(self, _size):
            pytest.fail("a complete buffered frame must be parsed before recv")

        def close(self):
            self.closed = True

    conn = Connection()
    read_fd, write_fd = os.pipe()
    stdin_r, stdin_w = os.pipe()
    os.write(write_fd, b"partial output\n")
    os.close(write_fd)
    handle = _BrokerProcessHandle(conn, 123, read_fd, stdin_w, remainder)
    try:
        with pytest.raises(
            EnvironmentConnectionError,
            match="invalid exit reply",
        ):
            handle.poll()
        assert conn.closed
        assert handle.stdout.read() == "partial output\n"
        assert not handle.stdout.closed
        assert handle.stdin is not None and handle.stdin.closed
    finally:
        os.close(stdin_r)
        if not handle.stdout.closed:
            handle.stdout.close()
        if handle.stdin is not None and not handle.stdin.closed:
            handle.stdin.close()


@pytest.mark.linux_only
def test_broker_process_poll_rejects_oversized_exit_frame_without_buffering():
    """A newline-free peer reply is bounded and closes the broker lease."""
    broker = _load_broker()
    from tools.environments.base import EnvironmentConnectionError
    from tools.environments.local import _BrokerProcessHandle

    class Connection:
        closed = False

        def setblocking(self, _blocking):
            pass

        def recv(self, _size):
            return b"x" * (broker.MAX_REPLY_BYTES + 1)

        def close(self):
            self.closed = True

    conn = Connection()
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    handle = _BrokerProcessHandle(conn, 123, read_fd, None)
    try:
        with pytest.raises(
            EnvironmentConnectionError,
            match="oversized exit reply",
        ):
            handle.poll()
        assert conn.closed
        assert handle._reply == b""
    finally:
        handle.stdout.close()


@pytest.mark.linux_only
def test_broker_process_poll_does_not_overwrite_exit_status_during_concurrent_poll():
    """The wait loop and stdout drainer may poll together; EOF cannot erase the exit frame."""
    from tools.environments.local import _BrokerProcessHandle

    class RacingConnection:
        def __init__(self):
            self._calls = 0
            self._lock = threading.Lock()
            self._first_receiving = threading.Event()
            self._second_poll_started = threading.Event()
            self._frame_consumed = threading.Event()

        def setblocking(self, _blocking):
            pass

        def recv(self, _size):
            with self._lock:
                call = self._calls
                self._calls += 1
            if call == 0:
                self._first_receiving.set()
                assert self._second_poll_started.wait(timeout=DEADLINE)
                return b'{"exit": 7}\n'
            assert self._frame_consumed.wait(timeout=DEADLINE)
            return b""

        def close(self):
            self._frame_consumed.set()

    conn = RacingConnection()
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    handle = _BrokerProcessHandle(conn, 123, read_fd, None)
    results = []
    first = threading.Thread(target=lambda: results.append(handle.poll()))

    def second_poll():
        conn._second_poll_started.set()
        results.append(handle.poll())

    second = threading.Thread(target=second_poll)
    first.start()
    assert conn._first_receiving.wait(timeout=DEADLINE)
    second.start()
    threads = [first, second]
    for thread in threads:
        thread.join(timeout=DEADLINE)

    handle.stdout.close()
    assert not any(thread.is_alive() for thread in threads)
    assert handle.returncode == 7
    assert 7 in results


@pytest.mark.linux_only
def test_runner_fd_launch_routes_explicit_stdout_descriptor(tmp_path):
    """stdio indices remain valid when the runner descriptor occupies wire slot zero."""
    broker = _load_broker()
    root = _staging_root(tmp_path)
    runner = _stage_runner(
        root, "stdout_runner.py", "print('runner-stdout', flush=True)\n"
    )
    proc, sock_path = _start_broker("broker-only-secret", root)
    runner_fd = os.open(runner, os.O_RDONLY | os.O_CLOEXEC)
    read_fd, write_fd = os.pipe()
    conn = None
    try:
        conn, reply, _remainder = broker.request_launch(
            sock_path,
            expected_peer_uid=os.geteuid(),
            runner_fd=runner_fd,
            env={},
            fds=[],
            stdout_fd=write_fd,
        )
        assert reply["ok"] is True
        os.close(write_fd)
        write_fd = -1
        assert _read_with_deadline(read_fd, until_eof=True) == b"runner-stdout\n"
    finally:
        if conn is not None:
            conn.close()
        os.close(runner_fd)
        os.close(read_fd)
        if write_fd != -1:
            os.close(write_fd)
        _stop_broker(proc)
        assert "Traceback" not in proc.stderr.read()


@pytest.mark.linux_only
def test_runner_fd_launch_accepts_max_child_fds_with_explicit_stdio(tmp_path):
    broker = _load_broker()
    root = _staging_root(tmp_path)
    runner = _stage_runner(root, "fd_budget_runner.py", "pass\n")
    proc, sock_path = _start_broker("broker-only-secret", root)
    runner_fd = os.open(runner, os.O_RDONLY | os.O_CLOEXEC)
    null_fd = os.open(os.devnull, os.O_RDWR | os.O_CLOEXEC)
    conn = None
    try:
        conn, reply, remainder = broker.request_launch(
            sock_path,
            expected_peer_uid=os.geteuid(),
            runner_fd=runner_fd,
            env={},
            fds=[null_fd] * broker.MAX_FDS,
            stdin_fd=null_fd,
            stdout_fd=null_fd,
            stderr_fd=null_fd,
        )
        assert reply["ok"] is True
        assert isinstance(remainder, bytes)
    finally:
        if conn is not None:
            conn.close()
        os.close(runner_fd)
        os.close(null_fd)
        _stop_broker(proc)
        assert "Traceback" not in proc.stderr.read()


@pytest.mark.linux_only
def test_runner_fd_launch_requires_allowed_peer_uid(tmp_path):
    """Runner transport and peer authority are one boundary, enforced before launch."""
    broker = _load_broker()
    root = _staging_root(tmp_path)
    runner = _stage_runner(
        root,
        "runner.py",
        """
        import os
        fd = int(os.environ["HERMES_BROKER_FDS"])
        os.write(fd, b"launched")
        """,
    )
    runner_fd = os.open(runner, os.O_RDONLY | os.O_CLOEXEC)
    writable_runner_fd = os.open(runner, os.O_RDWR | os.O_CLOEXEC)
    # SCM_RIGHTS carries the open description, including this deliberately hostile offset.
    # The broker must execute the whole runner without mutating the client's cursor.
    runner_offset = os.lseek(runner_fd, 0, os.SEEK_END)
    # The descriptor is the authority. After a cross-UID handoff the worker may read this
    # already-open file but cannot reopen its inode through /proc/self/fd.
    runner.chmod(0o000)
    runner.unlink()
    current_uid = os.getuid()  # windows-footgun: ok — linux_only test

    allowed_proc = denied_proc = None
    conn = None
    read_fd, write_fd = os.pipe()
    try:
        allowed_proc, allowed_socket = _start_broker(
            "host-only-secret",
            root,
            allowed_uids=[current_uid],
            socket_mode="0666",
        )
        assert stat.S_IMODE(os.lstat(allowed_socket).st_mode) == 0o666
        with pytest.raises(broker.BrokerError) as excinfo:
            broker.request_launch(
                allowed_socket,
                expected_peer_uid=os.geteuid(),
                runner_fd=writable_runner_fd,
                env={},
                fds=[],
            )
        assert excinfo.value.code == "runner_not_readable"
        conn, reply, _remainder = broker.request_launch(
            allowed_socket,
            expected_peer_uid=os.geteuid(),
            runner_fd=runner_fd,
            env={},
            fds=[write_fd],
        )
        os.close(write_fd)
        write_fd = -1
        assert reply["ok"] is True
        assert _read_with_deadline(read_fd, until_eof=True) == b"launched"
        assert os.lseek(runner_fd, 0, os.SEEK_CUR) == runner_offset
        conn.close()
        conn = None
        _stop_broker(allowed_proc)
        assert "Traceback" not in allowed_proc.stderr.read()
        allowed_proc = None

        denied_proc, denied_socket = _start_broker(
            "host-only-secret", root, allowed_uids=[current_uid + 1]
        )
        with pytest.raises(broker.BrokerError) as excinfo:
            broker.request_launch(
                denied_socket,
                expected_peer_uid=os.geteuid(),
                runner_fd=runner_fd,
                env={},
                fds=[],
            )
        assert excinfo.value.code == "peer_uid_not_allowed"
    finally:
        if conn is not None:
            conn.close()
        if write_fd != -1:
            os.close(write_fd)
        os.close(read_fd)
        os.close(runner_fd)
        os.close(writable_runner_fd)
        if allowed_proc is not None:
            _stop_broker(allowed_proc)
        if denied_proc is not None:
            _stop_broker(denied_proc)


@pytest.mark.linux_only
def test_empty_uid_allowlist_denies_same_uid_peer(tmp_path):
    broker = _load_broker()
    root = _staging_root(tmp_path)
    client, accepted = socket.socketpair()
    leases = broker._Leases()
    worker = threading.Thread(
        target=broker._serve_connection,
        args=(accepted, str(root), 0.05, leases, frozenset()),
    )
    assert leases.register(accepted, worker)
    worker.start()
    try:
        reply = json.loads(client.recv(4096).split(b"\n", 1)[0])
        assert reply["error"] == "peer_uid_not_allowed"
    finally:
        client.close()
        worker.join(DEADLINE)
        assert not worker.is_alive()


def _launch_body(path):
    return json.dumps({"op": "launch", "runner": str(path), "env": {}}).encode() + b"\n"


def _raw_request(sock_path: str, body: bytes, fds: list):
    """Send a request the client helper would never construct; return the parsed reply.

    Returns ``None`` if the broker closed the connection without saying anything — which is
    itself a finding: every refusal must be a structured reply, not a silent hangup.
    """
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(DEADLINE)
    try:
        conn.connect(sock_path)
        ancillary = (
            [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds))]
            if fds
            else []
        )
        _sendmsg_all(conn, body, ancillary)
        buf = b""
        while b"\n" not in buf:
            data = conn.recv(4096)
            if not data:
                break
            buf += data
        return json.loads(buf.split(b"\n", 1)[0]) if b"\n" in buf else None
    finally:
        conn.close()


def _sendmsg_all(conn, body: bytes, ancillary) -> None:
    sent = 0
    first = True
    while sent < len(body):
        written = conn.sendmsg([body[sent:]], ancillary if first else [])
        if written <= 0:
            raise ConnectionError("sendmsg made no progress")
        sent += written
        first = False


@pytest.mark.linux_only
def test_departed_peer_during_exit_frame_does_not_escape_connection_worker(
    monkeypatch, tmp_path
):
    """A client departure precisely at exit-frame delivery is routine lease teardown."""
    broker = _load_broker()
    root = _staging_root(tmp_path)

    class DepartingConnection:
        def __init__(self):
            self.replies = 0
            self.closed = False

        def getsockopt(self, *_args):
            return broker._UCRED.pack(os.getpid(), os.geteuid(), os.getegid())

        def sendall(self, _payload):
            self.replies += 1
            if self.replies == 2:
                raise BrokenPipeError(errno.EPIPE, "peer departed before exit frame")

        def close(self):
            self.closed = True

    class ExitedProcess:
        pid = 123

    conn = DepartingConnection()
    exited = ExitedProcess()
    leases = broker._Leases()
    assert leases.register(conn, threading.current_thread())
    terminated = []
    monkeypatch.setattr(
        broker,
        "_recv_request",
        lambda *_args: (
            None,
            False,
            None,
            ["/bin/true"],
            str(tmp_path),
            {},
            {},
            False,
            [],
        ),
    )
    monkeypatch.setattr(broker, "_launch", lambda *_args, **_kwargs: exited)
    monkeypatch.setattr(broker, "_await_lease_end", lambda *_args: None)
    monkeypatch.setattr(broker, "_unreaped_returncode", lambda _pid: 0)
    monkeypatch.setattr(broker, "_terminate", terminated.append)

    broker._serve_connection(conn, str(root), HANDSHAKE_TIMEOUT, leases)

    assert conn.replies == 2
    assert conn.closed is True
    assert terminated == [exited]


@pytest.mark.linux_only
def test_legacy_runner_observes_lease_eof_before_process_tree_cleanup(
    monkeypatch, tmp_path
):
    broker = _load_broker()
    root = _staging_root(tmp_path)
    runner = _stage_runner(root, "runner.py", "pass\n")
    events = []

    class Connection:
        def getsockopt(self, *_args):
            return broker._UCRED.pack(os.getpid(), os.geteuid(), os.getegid())

        def sendall(self, _payload):
            pass

        def close(self):
            events.append("close")

    class Process:
        pid = 123

    conn = Connection()
    process = Process()
    leases = broker._Leases()
    assert leases.register(conn, threading.current_thread())
    monkeypatch.setattr(
        broker,
        "_recv_request",
        lambda *_args: (
            str(runner),
            False,
            None,
            None,
            None,
            {},
            {},
            False,
            False,
        ),
    )
    monkeypatch.setattr(broker, "_launch", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(broker, "_process_start_time", lambda _pid: 9876)
    monkeypatch.setattr(broker, "_await_lease_end", lambda *_args: None)
    monkeypatch.setattr(broker, "_terminate", lambda _proc: events.append("terminate"))

    broker._serve_connection(conn, str(root), HANDSHAKE_TIMEOUT, leases)

    assert events == ["close", "terminate"]


@pytest.mark.linux_only
def test_detached_launch_is_terminated_when_client_misses_acknowledgement(
    monkeypatch, tmp_path
):
    broker = _load_broker()
    root = _staging_root(tmp_path)

    class DepartingConnection:
        def getsockopt(self, *_args):
            return broker._UCRED.pack(os.getpid(), os.geteuid(), os.getegid())

        def sendall(self, _payload):
            raise BrokenPipeError(errno.EPIPE, "peer departed before launch reply")

        def shutdown(self, *_args):
            pass

        def close(self):
            pass

    class RunningProcess:
        pid = 123
        returncode = None

    conn = DepartingConnection()
    process = RunningProcess()
    leases = broker._Leases()
    assert leases.register(conn, threading.current_thread())
    terminated = []
    monkeypatch.setattr(
        broker,
        "_recv_request",
        lambda *_args: (
            None,
            False,
            None,
            ["/bin/true"],
            str(tmp_path),
            {},
            {},
            True,
            [],
        ),
    )
    monkeypatch.setattr(broker, "_launch", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(broker, "_process_start_time", lambda _pid: 9876)
    monkeypatch.setattr(
        broker,
        "_wait_unreaped",
        lambda *_args: pytest.fail("an unacknowledged launch must not detach"),
    )
    monkeypatch.setattr(broker, "_exited_unreaped", lambda _pid: False)
    monkeypatch.setattr(broker, "_terminate", terminated.append)

    broker._serve_connection(conn, str(root), HANDSHAKE_TIMEOUT, leases)

    assert terminated == [process]


@pytest.mark.linux_only
def test_detached_launch_start_time_failure_terminates_child(monkeypatch, tmp_path):
    broker = _load_broker()
    root = _staging_root(tmp_path)

    class Connection:
        def __init__(self):
            self.replies = []

        def getsockopt(self, *_args):
            return broker._UCRED.pack(os.getpid(), os.geteuid(), os.getegid())

        def sendall(self, payload):
            self.replies.append(json.loads(payload))

        def close(self):
            pass

    class RunningProcess:
        pid = 123
        returncode = None

    conn = Connection()
    process = RunningProcess()
    leases = broker._Leases()
    assert leases.register(conn, threading.current_thread())
    terminated = []
    monkeypatch.setattr(
        broker,
        "_recv_request",
        lambda *_args: (
            None,
            False,
            None,
            ["/bin/true"],
            str(tmp_path),
            {},
            {},
            True,
            [],
        ),
    )
    monkeypatch.setattr(broker, "_launch", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        broker,
        "_process_start_time",
        lambda _pid: (_ for _ in ()).throw(OSError(errno.EMFILE, "fd limit")),
    )
    monkeypatch.setattr(broker, "_terminate", terminated.append)

    broker._serve_connection(conn, str(root), HANDSHAKE_TIMEOUT, leases)

    assert conn.replies == [
        {"ok": False, "error": "launch_failed", "message": "[Errno 24] fd limit"}
    ]
    assert terminated == [process]


@pytest.mark.linux_only
def test_detached_launch_reply_acknowledges_protocol_and_process_identity(
    monkeypatch, tmp_path
):
    broker = _load_broker()
    root = _staging_root(tmp_path)

    class Connection:
        def __init__(self):
            self.replies = []
            self.recv_calls = 0

        def getsockopt(self, *_args):
            return broker._UCRED.pack(os.getpid(), os.geteuid(), os.getegid())

        def sendall(self, payload):
            self.replies.append(json.loads(payload))

        def recv(self, _size):
            self.recv_calls += 1
            return b'{"op":"detach_ack"}\n'

        def settimeout(self, _timeout):
            pass

        def shutdown(self, *_args):
            pass

        def close(self):
            pass

    class ExitedProcess:
        pid = 123
        returncode = 0
        _hermes_systemd_unit = "hermes-local-exec-test.scope"

    conn = Connection()
    process = ExitedProcess()
    leases = broker._Leases()
    assert leases.register(conn, threading.current_thread())
    monkeypatch.setattr(
        broker,
        "_recv_request",
        lambda *_args: (
            None,
            False,
            None,
            ["/bin/true"],
            str(tmp_path),
            {},
            {},
            True,
            [],
        ),
    )
    launched = {}

    def fake_launch(*_args, **kwargs):
        launched.update(kwargs)
        return process

    monkeypatch.setattr(broker, "_launch", fake_launch)
    monkeypatch.setattr(broker, "_process_start_time", lambda _pid: 9876)
    monkeypatch.setattr(broker, "_exited_unreaped", lambda _pid: True)
    monkeypatch.setattr(broker, "_terminate", lambda _proc: None)

    broker._serve_connection(conn, str(root), 0.25, leases)

    assert conn.replies[0] == {
        "ok": True,
        "pid": 123,
        "detached": True,
        "start_time": 9876,
        "systemd_unit": "hermes-local-exec-test.scope",
    }
    assert conn.recv_calls == 1
    assert launched["exec_timeout"] == broker.DEFAULT_EXEC_TIMEOUT


@pytest.mark.linux_only
def test_background_client_rejects_broker_without_detached_ack(tmp_path, monkeypatch):
    from tools.environments.local import LocalEnvironment

    broker = _load_broker()
    connections = []

    class Connection:
        def __init__(self):
            self.sent = []
            connections.append(self)

        def sendall(self, payload):
            self.sent.append(json.loads(payload))

        def close(self):
            pass

    env = LocalEnvironment(cwd=str(tmp_path), timeout=5)
    env._local_exec_broker_socket = "/run/hermes-broker/broker.sock"
    env._local_exec_broker_uid = os.geteuid()
    replies = iter((
        {"ok": True, "pid": 123},
        {
            "ok": True,
            "pid": 124,
            "detached": True,
            "start_time": 9877,
            "systemd_unit": "hermes-local-exec-test.scope",
        },
    ))
    monkeypatch.setattr(
        broker,
        "request_launch",
        lambda *_args, **_kwargs: (Connection(), next(replies), b""),
    )

    with pytest.raises(broker.BrokerError, match="detached"):
        env._start_broker_background("sleep 300")
    assert connections[0].sent == []
    assert env._start_broker_background("sleep 300") == (
        124,
        9877,
        "hermes-local-exec-test.scope",
    )
    assert connections[1].sent == [{"op": "detach_ack"}]


@pytest.mark.linux_only
def test_argv_launch_rejects_text_the_os_cannot_encode():
    broker = _load_broker()

    with pytest.raises(broker.BrokerError) as excinfo:
        broker._validated({
            "op": "launch",
            "argv": ["/bin/printf", "\ud800"],
            "cwd": "/tmp",
            "env": {},
        })

    assert excinfo.value.code == "bad_request"
    assert "argv" in excinfo.value.message
    assert "encodable" in excinfo.value.message


@pytest.mark.linux_only
def test_runner_fd_scratch_finishes_cleanup_before_exit_reply(tmp_path):
    broker = _load_broker()
    root = _staging_root(tmp_path)
    report = tmp_path / "scratch-path"
    runner = _stage_runner(
        root,
        "scratch_runner.py",
        """
        import os
        from pathlib import Path

        scratch = Path(os.environ["TMPDIR"])
        nested = scratch / "worker-owned" / "nested"
        nested.mkdir(parents=True, mode=0o700)
        (nested / "payload").write_text("worker-owned", encoding="utf-8")
        Path(os.environ["REPORT"]).write_text(str(scratch), encoding="utf-8")
        """,
    )
    proc, sock_path = _start_broker(
        "secret",
        root,
        allowed_uids=[os.geteuid()],
    )
    runner_fd = os.open(runner, os.O_RDONLY | os.O_CLOEXEC)

    try:
        conn, reply, remainder = broker.request_launch(
            sock_path,
            expected_peer_uid=os.geteuid(),
            runner_fd=runner_fd,
            runner_python=sys.executable,
            cwd=None,
            env={"REPORT": str(report)},
            fds=[],
            scratch=True,
        )
        try:
            assert reply["ok"] is True
            exit_frame = remainder or _read_with_deadline(
                conn.fileno(), until_eof=False
            )
            assert json.loads(exit_frame.split(b"\n", 1)[0]) == {"exit": 0}
        finally:
            conn.close()
        scratch = Path(report.read_text(encoding="utf-8"))
        assert scratch.parent == root
        assert not scratch.exists()
    finally:
        os.close(runner_fd)
        _stop_broker(proc)


@pytest.mark.linux_only
def test_runner_fd_does_not_prepend_project_cwd_to_python_imports(tmp_path):
    broker = _load_broker()
    root = _staging_root(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    marker = tmp_path / "cwd-imported"
    report = tmp_path / "json-module"
    (project / "json.py").write_text(
        "import os\nopen(os.environ['MARKER'], 'w').write('ran')\n",
        encoding="utf-8",
    )
    runner = _stage_runner(
        root,
        "import_runner.py",
        """
        import json
        import os
        from pathlib import Path

        Path(os.environ["REPORT"]).write_text(json.__file__, encoding="utf-8")
        """,
    )
    proc, sock_path = _start_broker(
        "secret",
        root,
        allowed_uids=[os.geteuid()],
    )
    runner_fd = os.open(runner, os.O_RDONLY | os.O_CLOEXEC)

    try:
        conn, reply, remainder = broker.request_launch(
            sock_path,
            expected_peer_uid=os.geteuid(),
            runner_fd=runner_fd,
            runner_python=sys.executable,
            cwd=str(project),
            env={"MARKER": str(marker), "REPORT": str(report)},
            fds=[],
            scratch=True,
        )
        try:
            assert reply["ok"] is True
            exit_frame = remainder or _read_with_deadline(
                conn.fileno(), until_eof=False
            )
            assert json.loads(exit_frame.split(b"\n", 1)[0]) == {"exit": 0}
        finally:
            conn.close()
        assert not marker.exists()
        assert Path(report.read_text(encoding="utf-8")).resolve() != (
            project / "json.py"
        ).resolve()
    finally:
        os.close(runner_fd)
        _stop_broker(proc)


@pytest.mark.linux_only
def test_runner_fd_keeps_stderr_out_of_stdout_protocol_pipe(tmp_path):
    broker = _load_broker()
    root = _staging_root(tmp_path)
    report = tmp_path / "stderr-fds.json"
    runner = _stage_runner(
        root,
        "stdio_runner.py",
        """
        import json
        import os

        stderr_target = os.readlink("/proc/self/fd/2")
        duplicates = []
        for name in os.listdir("/proc/self/fd"):
            try:
                if os.readlink(f"/proc/self/fd/{name}") == stderr_target:
                    duplicates.append(int(name))
            except OSError:
                pass
        with open(os.environ["REPORT"], "w", encoding="utf-8") as stream:
            json.dump(sorted(duplicates), stream)
        os.write(1, b"stdout-only")
        os.write(2, b"stderr-only")
        """,
    )
    proc, sock_path = _start_broker(
        "secret",
        root,
        allowed_uids=[os.geteuid()],
    )
    runner_fd = os.open(runner, os.O_RDONLY | os.O_CLOEXEC)
    stdout_r, stdout_w = os.pipe()
    stderr_r, stderr_w = os.pipe()

    try:
        conn, reply, remainder = broker.request_launch(
            sock_path,
            expected_peer_uid=os.geteuid(),
            runner_fd=runner_fd,
            runner_python=sys.executable,
            cwd=None,
            env={"REPORT": str(report)},
            fds=[],
            stdout_fd=stdout_w,
            stderr_fd=stderr_w,
            scratch=True,
        )
        os.close(stdout_w)
        stdout_w = -1
        os.close(stderr_w)
        stderr_w = -1
        try:
            assert reply["ok"] is True
            exit_frame = remainder or _read_with_deadline(
                conn.fileno(), until_eof=False
            )
            assert json.loads(exit_frame.split(b"\n", 1)[0]) == {"exit": 0}
        finally:
            conn.close()
        assert _read_with_deadline(stdout_r, until_eof=True) == b"stdout-only"
        assert _read_with_deadline(stderr_r, until_eof=True) == b"stderr-only"
        assert json.loads(report.read_text(encoding="utf-8")) == [2]
    finally:
        for fd in (runner_fd, stdout_r, stdout_w, stderr_r, stderr_w):
            if fd >= 0:
                os.close(fd)
        _stop_broker(proc)


@pytest.mark.linux_only
def test_stdio_index_guards_return_bad_request_without_leaking_fds(tmp_path):
    """Malformed stdio indexes are refused before descriptor ownership can be corrupted."""
    broker = _load_broker()
    root = _staging_root(tmp_path)
    requests = (
        {
            "op": "launch",
            "argv": ["/bin/true"],
            "cwd": str(tmp_path),
            "env": {},
            "stdin_fd": 0,
            "stdout_fd": 0,
        },
        {
            "op": "launch",
            "argv": ["/bin/true"],
            "cwd": str(tmp_path),
            "env": {},
            "stdout_fd": 1,
        },
        {
            "op": "launch",
            "runner_fd": True,
            "env": {},
            "stdin_fd": 0,
        },
    )
    baseline = _fd_count(os.getpid())

    for request in requests:
        read_fd, write_fd = os.pipe()
        replies = []

        class Connection:
            def getsockopt(self, *_args):
                return broker._UCRED.pack(os.getpid(), os.geteuid(), os.getegid())

            def sendall(self, payload):
                replies.append(json.loads(payload))

            def close(self):
                pass

        conn = Connection()
        leases = broker._Leases()
        assert leases.register(conn, threading.current_thread())
        real_recv_request = broker._recv_request

        def receive(_conn, fds, _timeout):
            fds.append(os.dup(write_fd))
            return broker._validated(request)

        broker._recv_request = receive
        try:
            broker._serve_connection(conn, str(root), HANDSHAKE_TIMEOUT, leases)
            reply = replies[0]
            assert reply["ok"] is False
            assert reply["error"] == "bad_request"
            os.close(write_fd)
            write_fd = -1
            assert _read_with_deadline(read_fd, until_eof=True) == b""
        finally:
            broker._recv_request = real_recv_request
            os.close(read_fd)
            if write_fd != -1:
                os.close(write_fd)

    assert _fd_count(os.getpid()) == baseline, (
        "rejected stdio indexes leaked or over-closed a controller-side descriptor"
    )


@pytest.mark.linux_only
def test_teardown_failure_does_not_strand_lease(monkeypatch, tmp_path):
    broker = _load_broker()
    root = _staging_root(tmp_path)
    runner = _stage_runner(root, "runner.py", "import time; time.sleep(600)\n")
    client, conn = socket.socketpair()
    leases = broker._Leases()
    thread_errors = []

    def serve():
        try:
            broker._serve_connection(conn, str(root), HANDSHAKE_TIMEOUT, leases)
        except OSError as exc:
            thread_errors.append(exc)

    worker = threading.Thread(target=serve)
    assert leases.register(conn, worker)
    worker.start()
    client.sendall(_launch_body(runner))
    reply = json.loads(client.recv(4096).split(b"\n", 1)[0])
    assert reply["ok"] is True

    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                broker, "_terminate", lambda _proc: (_ for _ in ()).throw(OSError())
            )
            client.close()
            worker.join(DEADLINE)
        assert len(thread_errors) == 1
        drain = threading.Thread(target=leases.drain, daemon=True)
        drain.start()
        drain.join(DEADLINE)
        assert not drain.is_alive(), (
            "teardown failure stranded the lease registry entry"
        )

        monkeypatch.setattr(
            broker.os, "pidfd_open", lambda _pid: (_ for _ in ()).throw(OSError())
        )
        broker._wait_unreaped(reply["pid"], 0)
    finally:
        broker._signal_group(
            reply["pid"],
            signal.SIGKILL,  # windows-footgun: ok — linux_only test
        )
        with contextlib.suppress(ChildProcessError, ProcessLookupError):
            os.waitpid(reply["pid"], 0)


@pytest.mark.linux_only
def test_socket_is_private_when_first_published(monkeypatch, tmp_path):
    broker = _load_broker()
    root = _staging_root(tmp_path)
    sock_path = _socket_path()
    published_modes = []
    real_link = os.link

    def observe_link(source, target):
        assert target == sock_path
        assert stat.S_IMODE(os.lstat(source).st_mode) == 0o600
        real_link(source, target)
        published_modes.append(stat.S_IMODE(os.lstat(target).st_mode))

    monkeypatch.setattr(broker.os, "link", observe_link)
    monkeypatch.setattr(broker, "_install_shutdown", lambda listener: listener.close())

    broker.serve(sock_path, str(root))

    assert published_modes == [0o600]


@pytest.mark.linux_only
def test_broker_launches_from_long_socket_directory(tmp_path):
    root = _staging_root(tmp_path)
    runner = _stage_runner(root, "runner.py", "import time; time.sleep(600)\n")
    path_root = tempfile.mkdtemp(prefix="hbrk-long-")
    socket_dir = os.path.join(
        path_root,
        "d" * (107 - len(os.fsencode(path_root)) - len(os.fsencode("/b.sock")) - 1),
    )
    os.mkdir(socket_dir, 0o700)
    sock_path = os.path.join(socket_dir, "b.sock")
    assert len(os.fsencode(sock_path)) == 107
    assert (
        len(os.fsencode(os.path.join(socket_dir, ".hermes-broker-XXXXXXXX", "socket")))
        > 107
    )

    proc = None
    conn = None
    try:
        proc, _ = _start_broker("host-only-secret", root, sock_path=sock_path)
        broker = _load_broker()
        conn, reply, _remainder = broker.request_launch(
            sock_path,
            expected_peer_uid=os.geteuid(),
            runner=str(runner),
            env={},
            fds=[],
        )
        assert reply["ok"] is True
    finally:
        if conn is not None:
            conn.close()
        if proc is not None:
            _stop_broker(proc)
        with contextlib.suppress(FileNotFoundError):
            os.rmdir(socket_dir)
        with contextlib.suppress(FileNotFoundError):
            os.rmdir(path_root)


@pytest.mark.linux_only
def test_broker_launches_when_socket_basename_is_publication_suffix(tmp_path):
    root = _staging_root(tmp_path)
    runner = _stage_runner(root, "runner.py", "import time; time.sleep(600)\n")
    socket_dir = tempfile.mkdtemp(prefix="hbrk_")
    sock_path = os.path.join(socket_dir, "0")

    proc = None
    conn = None
    try:
        proc, _ = _start_broker("host-only-secret", root, sock_path=sock_path)
        broker = _load_broker()
        conn, reply, _remainder = broker.request_launch(
            sock_path,
            expected_peer_uid=os.geteuid(),
            runner=str(runner),
            env={},
            fds=[],
        )
        assert reply["ok"] is True
    finally:
        if conn is not None:
            conn.close()
        if proc is not None:
            _stop_broker(proc)
        with contextlib.suppress(FileNotFoundError):
            os.rmdir(socket_dir)


@pytest.mark.linux_only
def test_broker_publishes_107_byte_socket_with_one_character_basename(tmp_path):
    root = _staging_root(tmp_path)
    runner = _stage_runner(root, "runner.py", "import time; time.sleep(600)\n")
    path_root = tempfile.mkdtemp(prefix="hbrk-boundary-")
    socket_dir = os.path.join(
        path_root,
        "d" * (107 - len(os.fsencode(path_root)) - len(os.fsencode("/0")) - 1),
    )
    os.mkdir(socket_dir, 0o700)
    sock_path = os.path.join(socket_dir, "0")
    assert len(os.fsencode(sock_path)) == 107

    proc = None
    conn = None
    try:
        proc, _ = _start_broker("host-only-secret", root, sock_path=sock_path)
        broker = _load_broker()
        conn, reply, _remainder = broker.request_launch(
            sock_path,
            expected_peer_uid=os.geteuid(),
            runner=str(runner),
            env={},
            fds=[],
        )
        assert reply["ok"] is True
    finally:
        if conn is not None:
            conn.close()
        if proc is not None:
            _stop_broker(proc)
        with contextlib.suppress(FileNotFoundError):
            os.rmdir(socket_dir)
        with contextlib.suppress(FileNotFoundError):
            os.rmdir(path_root)


@pytest.mark.linux_only
def test_broker_refuses_unreachable_socket_path(tmp_path):
    root = _staging_root(tmp_path)
    path_root = tempfile.mkdtemp(prefix="hbrk-overlong-")
    socket_dir = os.path.join(
        path_root,
        "d" * (108 - len(os.fsencode(path_root)) - len(os.fsencode("/b.sock")) - 1),
    )
    os.mkdir(socket_dir, 0o700)
    sock_path = os.path.join(socket_dir, "b.sock")
    assert len(os.fsencode(sock_path)) == 108

    proc, _ = _start_broker(
        "host-only-secret", root, sock_path=sock_path, expect_ready=False
    )
    try:
        assert proc.wait(timeout=DEADLINE) != 0
        diagnostic = proc.stderr.read()
        assert "AF_UNIX socket path must be at most 107 bytes" in diagnostic
        assert "Traceback" not in diagnostic
    finally:
        if proc.poll() is None:
            _stop_broker(proc)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(sock_path)
        with contextlib.suppress(OSError):
            os.rmdir(socket_dir)
        with contextlib.suppress(OSError):
            os.rmdir(path_root)


@pytest.mark.linux_only
def test_broker_refuses_writable_socket_directory(tmp_path):
    root = _staging_root(tmp_path)
    socket_dir = tempfile.mkdtemp(prefix="hbrk-writable-")
    os.chmod(socket_dir, 0o777)
    sock_path = os.path.join(socket_dir, "broker.sock")

    proc, _ = _start_broker(
        "host-only-secret", root, sock_path=sock_path, expect_ready=False
    )
    try:
        assert proc.wait(timeout=DEADLINE) != 0
        diagnostic = proc.stderr.read()
        assert "socket directory must be broker-owned and not writable" in diagnostic
        assert "Traceback" not in diagnostic
    finally:
        if proc.poll() is None:
            _stop_broker(proc)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(sock_path)
        os.chmod(socket_dir, 0o700)
        with contextlib.suppress(OSError):
            os.rmdir(socket_dir)


@pytest.mark.linux_only
def test_empty_publication_slot_is_reclaimable(tmp_path):
    broker = _load_broker()
    slot = tmp_path / broker._PUBLISH_SUFFIXES[0]
    slot.mkdir(mode=0o700)

    assert broker._reclaim_stale_publish_dir(str(slot)) is True
    assert not slot.exists()


@pytest.mark.linux_only
def test_publication_slots_reclaim_only_stale_broker_sockets(tmp_path, request):
    broker = _load_broker()
    root = _staging_root(tmp_path)
    runner = _stage_runner(root, "runner.py", "import time; time.sleep(600)\n")
    socket_dir = tempfile.mkdtemp(prefix="hbrk-slots-")
    request.addfinalizer(lambda: shutil.rmtree(socket_dir, ignore_errors=True))
    live_dir = os.path.join(socket_dir, broker._PUBLISH_SUFFIXES[0])
    unrelated_dir = os.path.join(socket_dir, broker._PUBLISH_SUFFIXES[1])
    os.mkdir(live_dir, 0o700)
    os.mkdir(unrelated_dir, 0o700)
    live = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    live.bind(os.path.join(live_dir, "s"))
    live.listen(1)
    unrelated = Path(unrelated_dir) / "keep"
    unrelated.write_text("not broker state\n", encoding="utf-8")
    for suffix in broker._PUBLISH_SUFFIXES[2:]:
        candidate = os.path.join(socket_dir, suffix)
        os.mkdir(candidate, 0o700)
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(os.path.join(candidate, "s"))
        stale.close()

    proc = None
    conn = None
    sock_path = os.path.join(socket_dir, "broker.sock")
    try:
        proc, _ = _start_broker("host-only-secret", root, sock_path=sock_path)
        conn, reply, _remainder = broker.request_launch(
            sock_path,
            expected_peer_uid=os.geteuid(),
            runner=str(runner),
            env={},
            fds=[],
        )
        assert reply["ok"] is True
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(os.path.join(live_dir, "s"))
        finally:
            probe.close()
        assert unrelated.read_text(encoding="utf-8") == "not broker state\n"
    finally:
        if conn is not None:
            conn.close()
        if proc is not None:
            _stop_broker(proc)
        live.close()

    exhausted_dir = tempfile.mkdtemp(prefix="hbrk-exhausted-")
    request.addfinalizer(lambda: shutil.rmtree(exhausted_dir, ignore_errors=True))
    for suffix in broker._PUBLISH_SUFFIXES:
        candidate = Path(exhausted_dir) / suffix
        candidate.mkdir(mode=0o700)
        (candidate / "keep").write_text("unrelated\n", encoding="utf-8")
    refused, _ = _start_broker(
        "host-only-secret",
        root,
        sock_path=os.path.join(exhausted_dir, "broker.sock"),
        expect_ready=False,
    )
    assert refused.wait(timeout=DEADLINE) != 0
    diagnostic = refused.stderr.read()
    assert "no compact private socket publication directory was available" in diagnostic
    assert "Traceback" not in diagnostic


@pytest.mark.linux_only
def test_fifo_runner_is_refused_without_blocking_shutdown(tmp_path):
    root = _staging_root(tmp_path)
    fifo = root / "runner.py"
    os.mkfifo(fifo)
    proc, sock_path = _start_broker("host-only-secret", root)
    read_fd, write_fd = os.pipe()
    try:
        reply = _raw_request(sock_path, _launch_body(fifo), [write_fd])
        os.close(write_fd)
        write_fd = -1
        assert reply == {
            "ok": False,
            "error": "runner_not_regular",
            "message": "runner is not a regular file",
        }
        assert _read_with_deadline(read_fd, until_eof=True) == b""

        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=DEADLINE) == 0
        assert not os.path.exists(sock_path)
    finally:
        os.close(read_fd)
        if write_fd != -1:
            os.close(write_fd)
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=DEADLINE)


@pytest.mark.linux_only
@pytest.mark.parametrize("accept_errno", [errno.EMFILE, errno.ENOBUFS, errno.ENOMEM])
def test_transient_accept_failure_preserves_live_lease(
    monkeypatch, tmp_path, accept_errno
):
    broker = _load_broker()
    root = _staging_root(tmp_path)
    runner = _stage_runner(root, "runner.py", "import time; time.sleep(600)\n")
    sock_path = _socket_path()
    real_socket = socket.socket
    retry_accept = threading.Event()
    release_failure = threading.Event()
    finish_accept = threading.Event()
    listener_box = []

    class OneTransientFailureSocket(real_socket):
        accept_calls = 0

        def listen(self, backlog):
            listener_box.append(self)
            return super().listen(backlog)

        def accept(self):
            type(self).accept_calls += 1
            if type(self).accept_calls == 2:
                assert release_failure.wait(DEADLINE)
                raise OSError(accept_errno, "injected transient accept failure")
            if type(self).accept_calls == 3:
                retry_accept.set()
                assert finish_accept.wait(DEADLINE)
                raise OSError(errno.EBADF, "listener closed by test")
            return super().accept()

    monkeypatch.setattr(broker.socket, "socket", OneTransientFailureSocket)
    monkeypatch.setattr(broker, "_install_shutdown", lambda _listener: None)
    server = threading.Thread(
        target=broker.serve, args=(sock_path, str(root)), daemon=True
    )
    server.start()
    deadline = time.monotonic() + DEADLINE
    while not os.path.exists(sock_path) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert os.path.exists(sock_path), "broker never published its socket"

    conn = None
    child_pid = None
    try:
        conn, reply, _remainder = broker.request_launch(
            sock_path,
            expected_peer_uid=os.geteuid(),
            runner=str(runner),
            env={},
            fds=[],
        )
        child_pid = reply["pid"]
        assert _pid_running(child_pid)
        release_failure.set()
        assert retry_accept.wait(DEADLINE), (
            f"serve stopped after transient accept errno {accept_errno}"
        )
        assert server.is_alive()
        assert _pid_running(child_pid), "transient accept failure drained a live lease"
    finally:
        release_failure.set()
        if conn is not None:
            conn.close()
        if listener_box:
            listener_box[0].close()
        finish_accept.set()
        server.join(DEADLINE)
        assert not server.is_alive(), "serve did not stop after its listener closed"
        if child_pid is not None and _pid_running(child_pid):
            os.kill(child_pid, signal.SIGKILL)  # windows-footgun: ok — linux_only test
        with contextlib.suppress(FileNotFoundError):
            os.unlink(sock_path)
        with contextlib.suppress(OSError):
            os.rmdir(os.path.dirname(sock_path))


@pytest.mark.linux_only
def test_shutdown_cleanup_cannot_unlink_replacement_socket(monkeypatch):
    broker = _load_broker()
    sock_path = _socket_path()
    socket_dir = os.path.dirname(sock_path)
    temporary_path = os.path.join(socket_dir, "replacement.sock")
    first = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    replacement = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    first.bind(sock_path)
    owned_fd = os.open(
        sock_path,
        os.O_PATH
        | os.O_NOFOLLOW
        | os.O_CLOEXEC,  # windows-footgun: ok — linux_only test
    )
    publisher_holds_lock = threading.Event()
    publish_replacement = threading.Event()
    replacement_published = threading.Event()
    thread_errors = []
    real_flock = fcntl.flock

    def coordinated_flock(fd, operation):
        if threading.current_thread().name == "cleanup":
            publish_replacement.set()
        return real_flock(fd, operation)

    def publish():
        lock_fd = os.open(socket_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            real_flock(lock_fd, fcntl.LOCK_EX)
            publisher_holds_lock.set()
            assert publish_replacement.wait(DEADLINE)
            os.unlink(sock_path)
            replacement.bind(temporary_path)
            replacement.listen(1)
            os.link(temporary_path, sock_path)
            os.unlink(temporary_path)
            replacement_published.set()
        except BaseException as exc:
            thread_errors.append(exc)
            replacement_published.set()
        finally:
            os.close(lock_fd)

    publisher = threading.Thread(target=publish, name="publisher")
    cleanup = threading.Thread(
        target=broker._unlink_owned_socket,
        args=(sock_path, owned_fd),
        name="cleanup",
    )
    monkeypatch.setattr(broker.fcntl, "flock", coordinated_flock)
    publisher.start()
    assert publisher_holds_lock.wait(DEADLINE)
    cleanup.start()
    try:
        publisher.join(DEADLINE)
        cleanup.join(DEADLINE)
        assert not publisher.is_alive() and not cleanup.is_alive()
        assert not thread_errors
        with pytest.raises(OSError) as closed:
            os.fstat(owned_fd)
        assert closed.value.errno == errno.EBADF
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(sock_path)
        finally:
            probe.close()
    finally:
        publish_replacement.set()
        replacement_published.set()
        publisher.join(DEADLINE)
        cleanup.join(DEADLINE)
        with contextlib.suppress(OSError):
            os.close(owned_fd)
        first.close()
        replacement.close()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_path)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(sock_path)
        with contextlib.suppress(OSError):
            os.rmdir(socket_dir)


@pytest.mark.linux_only
def test_broker_transports_approved_resources_and_refuses_invalid_requests_without_leaking_fds(
    tmp_path,
):
    """Transport + protocol: what crosses the boundary, what is refused, and who owns an fd.

    The accepted launch witnesses all three transports the ``sudo`` carrier severed, and
    witnesses them by MECHANISM, not by side effect: the child reports the inode behind its
    own ``/proc/self/fd`` argv, so a broker that handed over a plain filesystem path instead
    would fail. Every refusal then has to close the descriptors it was handed — the peer's
    pipe reaching EOF is the only proof the broker is not holding its channel open.
    """
    broker = _load_broker()
    root = _staging_root(tmp_path)
    runner = _stage_runner(
        root,
        "runner.py",
        """
        import json, os, sys
        fd = int(os.environ["HERMES_BROKER_FDS"].split(",")[0])
        os.write(fd, json.dumps({
            "approved": os.environ.get("BROKER_PROBE_APPROVED", "<missing>"),
            "host_only_leaked": "BROKER_PROBE_HOST_ONLY" in os.environ,
            "argv0": sys.argv[0],
            "runner_inode": os.stat(sys.argv[0]).st_ino,
            "stderr_target": os.readlink("/proc/self/fd/2"),
        }).encode())
    """,
    )
    outside = tmp_path / "outside.py"
    outside.write_text("raise SystemExit(0)\n", encoding="utf-8")
    escape = root / "escape.py"
    escape.symlink_to(outside)

    proc, sock_path = _start_broker("host-only-secret", root)
    try:
        # Short writes resend only payload bytes; SCM_RIGHTS is attached exactly once.
        class ShortSender:
            def __init__(self, target):
                self.target = target
                self.ancillary_calls = 0

            def sendmsg(self, buffers, ancillary):
                self.ancillary_calls += bool(ancillary)
                return self.target.sendmsg([buffers[0][:3]], ancillary)

        send_sock, recv_sock = socket.socketpair()
        read_fd, write_fd = os.pipe()
        try:
            sender = ShortSender(send_sock)
            broker._sendmsg_all(
                sender, b"short-write-frame", broker._ancillary([write_fd])
            )
            received = b""
            while len(received) < len(b"short-write-frame"):
                received += recv_sock.recv(128)
            assert received == b"short-write-frame"
            assert sender.ancillary_calls == 1
        finally:
            send_sock.close()
            recv_sock.close()
            os.close(read_fd)
            os.close(write_fd)

        # --- accepted launch: env, descriptor and runner all arrive explicitly -----------
        read_fd, write_fd = os.pipe()
        try:
            conn, reply, _remainder = broker.request_launch(
                sock_path,
                expected_peer_uid=os.geteuid(),
                runner=str(runner),
                env={"BROKER_PROBE_APPROVED": "approved-value-42"},
                fds=[write_fd],
            )
            try:
                assert reply["ok"] is True
                os.close(write_fd)
                write_fd = -1
                # The broker resolved and opened the runner before replying, so sealing the
                # staging dir now witnesses the /proc/self/fd hand-off: a child that had to
                # traverse the directory itself could no longer reach the file.
                os.chmod(root, 0o000)
                report = _child_report(read_fd)
            finally:
                os.chmod(root, 0o700)
                conn.close()
        finally:
            os.close(read_fd)
            if write_fd != -1:
                os.close(write_fd)

        # (a) the approved value arrived; the broker's own environment did not.
        assert report["approved"] == "approved-value-42"
        assert report["host_only_leaked"] is False
        # (b) the runner crossed as a descriptor, not as a path: argv[0] is the broker's own
        #     open fd, and it resolves to the staged file's inode.
        assert report["argv0"].startswith("/proc/self/fd/")
        assert report["runner_inode"] == runner.stat().st_ino
        # (c) stderr is transported explicitly like stdin/stdout, not inherited from the
        #     broker. An inherited stderr is both a log-forgery channel for a future
        #     lower-privileged child and an undrained pipe the child can wedge itself on.
        assert report["stderr_target"] == "/dev/null"

        # --- refusals: each must be structured, and must close what it was handed --------
        baseline = _wait_for_fd_count(proc.pid, _fd_count(proc.pid))
        refusals = [
            (
                "runner outside the staging root",
                _launch_body(outside),
                1,
                "runner_outside_root",
            ),
            (
                "symlink escaping the staging root",
                _launch_body(escape),
                1,
                "runner_outside_root",
            ),
            (
                "runner that does not exist",
                _launch_body(root / "gone.py"),
                1,
                "launch_failed",
            ),
            ("body that is not JSON", b"{definitely not json\n", 1, "bad_request"),
            ("body with invalid UTF-8", b"\xff\xff\xff\xff\n", 1, "bad_request"),
            (
                "deeply nested body",
                b"[" * 30000 + b"\n",
                1,
                "bad_request",
            ),
            (
                "body that is not a launch request",
                b'{"op": "nope"}\n',
                1,
                "bad_request",
            ),
            (
                "argv launch carrying relative cwd",
                json.dumps({
                    "op": "launch",
                    "argv": ["/bin/true"],
                    "cwd": "relative",
                    "env": {},
                }).encode()
                + b"\n",
                1,
                "bad_request",
            ),
            (
                "argv launch carrying runner_python",
                json.dumps({
                    "op": "launch",
                    "argv": ["/bin/true"],
                    "cwd": str(tmp_path),
                    "runner_python": "/usr/bin/python3",
                    "env": {},
                }).encode()
                + b"\n",
                1,
                "bad_request",
            ),
            (
                "legacy runner request carrying cwd",
                json.dumps({
                    "op": "launch",
                    "runner": str(runner),
                    "cwd": 5,
                    "env": {},
                }).encode()
                + b"\n",
                1,
                "bad_request",
            ),
            (
                "environment key containing equals",
                json.dumps({
                    "op": "launch",
                    "runner": str(runner),
                    "env": {"A=B": "c"},
                }).encode()
                + b"\n",
                1,
                "bad_request",
            ),
            (
                "environment value containing an unencodable surrogate",
                json.dumps({
                    "op": "launch",
                    "runner": str(runner),
                    "env": {"A": "\ud800"},
                }).encode()
                + b"\n",
                1,
                "bad_request",
            ),
            (
                "environment that is not an object",
                json.dumps({"op": "launch", "runner": str(runner), "env": []}).encode()
                + b"\n",
                1,
                "bad_request",
            ),
            (
                "runner containing NUL",
                json.dumps({"op": "launch", "runner": "/x/a\0b", "env": {}}).encode()
                + b"\n",
                1,
                "bad_request",
            ),
            (
                "runner containing an unencodable surrogate",
                json.dumps({"op": "launch", "runner": "/x/\ud800", "env": {}}).encode()
                + b"\n",
                1,
                "bad_request",
            ),
            # More descriptors than one control buffer holds: the kernel truncates and sets
            # MSG_CTRUNC, so the broker must refuse rather than launch a child whose
            # HERMES_BROKER_FDS is silently short of what the client passed.
            (
                "more descriptors than the request cap",
                _launch_body(runner),
                broker.MAX_RECEIVED_FDS + 1,
                "truncated_ancillary",
            ),
            # A complete frame larger than the cap: the broker drains through the newline
            # before replying, so a well-behaved sender gets a structured refusal instead
            # of EPIPE while still bounding retained attacker-controlled bytes.
            (
                "body larger than the request cap",
                b"x" * (broker.MAX_REQUEST_BYTES + 4096) + b"\n",
                1,
                "request_too_large",
            ),
        ]
        for label, body, fd_copies, error in refusals:
            read_fd, write_fd = os.pipe()
            try:
                reply = _raw_request(sock_path, body, [write_fd] * fd_copies)
                os.close(write_fd)
                write_fd = -1
                assert reply is not None, f"{label}: broker closed without a reply"
                assert reply["ok"] is False, f"{label}: broker accepted the request"
                assert reply["error"] == error, label
                assert reply["message"], f"{label}: reply carries no message"
                assert _read_with_deadline(read_fd, until_eof=True) == b"", label
            finally:
                os.close(read_fd)
                if write_fd != -1:
                    os.close(write_fd)

        # Descriptor limits apply to the whole request, not one ancillary message.
        read_fd, write_fd = os.pipe()
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            conn.settimeout(DEADLINE)
            conn.connect(sock_path)
            first = broker.MAX_RECEIVED_FDS // 2 + 1
            conn.sendmsg(
                [b'{"op":'],
                _ancillary := [
                    (
                        socket.SOL_SOCKET,
                        socket.SCM_RIGHTS,
                        array.array("i", [write_fd] * first),
                    )
                ],
            )
            conn.sendmsg([b'"launch"}\n'], _ancillary)
            reply = json.loads(conn.recv(4096).split(b"\n", 1)[0])
            os.close(write_fd)
            write_fd = -1
            assert reply["error"] == "too_many_fds"
            assert _read_with_deadline(read_fd, until_eof=True) == b""
        finally:
            conn.close()
            os.close(read_fd)
            if write_fd != -1:
                os.close(write_fd)

        # Repeating the cheapest refusal exercises the accumulation the per-arm EOF check
        # cannot see: a per-request fd leak ends at RLIMIT_NOFILE, not at one bad request.
        for _ in range(40):
            _raw_request(sock_path, _launch_body(root / "gone.py"), [])
        assert _wait_for_fd_count(proc.pid, baseline) <= baseline

        # --- and the broker still works -------------------------------------------------
        read_fd, write_fd = os.pipe()
        try:
            conn, reply, _remainder = broker.request_launch(
                sock_path,
                expected_peer_uid=os.geteuid(),
                runner=str(runner),
                env={},
                fds=[write_fd],
            )
            os.close(write_fd)
            write_fd = -1
            assert reply["ok"] is True
            report = _child_report(read_fd)
            assert report["approved"] == "<missing>"
            conn.close()
        finally:
            os.close(read_fd)
            if write_fd != -1:
                os.close(write_fd)

        # A refused request must also be a typed failure on the client, not a JSONDecodeError
        # escaping from inside the helper (which would also strand the client's own socket).
        with pytest.raises(broker.BrokerError) as excinfo:
            broker.request_launch(
                sock_path,
                expected_peer_uid=os.geteuid(),
                runner=str(outside),
                env={},
                fds=[],
            )
        assert excinfo.value.code == "runner_outside_root"
    finally:
        _stop_broker(proc)
        assert "Traceback" not in proc.stderr.read()


@pytest.mark.linux_only
def test_brokered_child_is_terminated_when_the_client_connection_disappears(
    tmp_path, monkeypatch
):
    """Lease lifetime: nothing outlives its lease, and nothing pins a worker forever.

    The client connection is the broker-owned replacement for the inherited parent-death pipe
    that ``sudo`` closes — the liveness signal crosses the boundary explicitly instead of by
    inheritance. That only holds if it holds for the whole process GROUP (a code kernel or a
    bash runner spawns descendants), for the broker's own shutdown (a lease whose owner dies
    still has to reap), and in the other direction too: a child that exits on its own, or a
    peer that never sends a request, must not park a worker thread forever.
    """
    broker = _load_broker()
    root = _staging_root(tmp_path)
    # Spawns a grandchild in the SAME process group, then reports both pids. A broker that
    # signalled only proc.pid instead of the group would leak the grandchild.
    group_runner = _stage_runner(
        root,
        "group_runner.py",
        """
        import json, os, subprocess, sys, time
        fd = int(os.environ["HERMES_BROKER_FDS"].split(",")[0])
        kid = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        os.write(fd, (json.dumps({"child": os.getpid(), "grandchild": kid.pid}) + "\\n").encode())
        time.sleep(600)
    """,
    )
    quick_runner = _stage_runner(
        root,
        "quick_runner.py",
        """
        import os
        fd = int(os.environ["HERMES_BROKER_FDS"].split(",")[0])
        os.write(fd, b"bye\\n")
    """,
    )
    term_runner = _stage_runner(
        root,
        "term_runner.py",
        """
        import os, signal, time
        fd = int(os.environ["HERMES_BROKER_FDS"].split(",")[0])
        def stop(_signum, _frame):
            os.write(fd, f"term {time.monotonic()}\\n".encode())
            time.sleep(600)
        signal.signal(signal.SIGTERM, stop)
        os.write(fd, b"ready\\n")
        time.sleep(600)
    """,
    )
    proc, sock_path = _start_broker("host-only-secret", root)
    doomed: list = []
    try:
        # --- a live broker owns its socket; a second broker must not steal it -------------
        duplicate, _ = _start_broker(
            "host-only-secret",
            root,
            sock_path=sock_path,
            expect_ready=False,
        )
        try:
            assert duplicate.wait(timeout=DEADLINE) != 0, (
                "a second broker replaced the live broker's socket"
            )
        finally:
            if duplicate.poll() is None:
                _stop_broker(duplicate)

        # --- the lease governs the whole process group ----------------------------------
        read_fd, write_fd = os.pipe()
        try:
            conn, reply, _remainder = broker.request_launch(
                sock_path,
                expected_peer_uid=os.geteuid(),
                runner=str(group_runner),
                env={},
                fds=[write_fd],
            )
            os.close(write_fd)
            write_fd = -1
            pids = json.loads(_read_with_deadline(read_fd, until_eof=False))
            doomed.extend(pids.values())
            assert reply["pid"] == pids["child"]
            assert _pid_running(pids["child"]) and _pid_running(pids["grandchild"])

            conn.close()  # the lease disappears

            assert _wait_until_gone(pids["child"]), (
                f"child {pids['child']} outlived the client connection"
            )
            assert _wait_until_gone(pids["grandchild"]), (
                f"grandchild {pids['grandchild']} survived the lease: the broker tore down "
                "the process it spawned, not the process GROUP it leads"
            )
        finally:
            os.close(read_fd)
            if write_fd != -1:
                os.close(write_fd)

        # --- a child that exits on its own releases the lease ---------------------------
        # Otherwise the worker thread sits in recv until the client happens to disconnect,
        # holding an unreaped child, and a caller polling the connection never learns the
        # child is gone.
        read_fd, write_fd = os.pipe()
        try:
            conn, reply, _remainder = broker.request_launch(
                sock_path,
                expected_peer_uid=os.geteuid(),
                runner=str(quick_runner),
                env={},
                fds=[write_fd],
            )
            os.close(write_fd)
            write_fd = -1
            assert _read_with_deadline(read_fd, until_eof=False) == b"bye"
            conn.settimeout(DEADLINE)
            try:
                assert conn.recv(4096) == b""
            except TimeoutError:
                pytest.fail(
                    "the broker held the lease open after the child exited: the worker "
                    "thread is pinned until the client disconnects"
                )
            assert _wait_until_gone(reply["pid"]), "the exited child was never reaped"
            conn.close()
        finally:
            os.close(read_fd)
            if write_fd != -1:
                os.close(write_fd)

        # --- a peer that never sends a request does not park a worker forever -----------
        silent = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        silent.settimeout(DEADLINE)
        try:
            started = time.monotonic()
            silent.connect(sock_path)
            handshake = silent.recv(4096)
            elapsed = time.monotonic() - started
            assert handshake, "the broker closed a silent peer without saying why"
            assert (
                json.loads(handshake.split(b"\n", 1)[0])["error"] == "handshake_timeout"
            )
            assert elapsed >= HANDSHAKE_TIMEOUT, (
                f"the broker gave up after {elapsed:.2f}s, before the handshake window"
            )
        finally:
            silent.close()

        # The handshake window is cumulative: traffic cannot renew it one byte at a time.
        drip = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        drip.settimeout(DEADLINE)
        try:
            drip.connect(sock_path)
            started = time.monotonic()
            for byte in b'{"op":':
                time.sleep(HANDSHAKE_TIMEOUT / 5)
                try:
                    drip.sendall(bytes([byte]))
                except BrokenPipeError:
                    break
            reply = drip.recv(4096)
            elapsed = time.monotonic() - started
            assert json.loads(reply.split(b"\n", 1)[0])["error"] == "handshake_timeout"
            assert elapsed < HANDSHAKE_TIMEOUT * 1.5, (
                f"slow-drip bytes renewed the handshake deadline for {elapsed:.2f}s"
            )
        finally:
            drip.close()

        # --- the broker's own SIGTERM tears down every lease it is still holding --------
        read_fd, write_fd = os.pipe()
        held = None
        term_leases = []
        term_reads = []
        try:
            held, reply, _remainder = broker.request_launch(
                sock_path,
                expected_peer_uid=os.geteuid(),
                runner=str(group_runner),
                env={},
                fds=[write_fd],
            )
            os.close(write_fd)
            write_fd = -1
            pids = json.loads(_read_with_deadline(read_fd, until_eof=False))
            doomed.extend(pids.values())

            for _ in range(2):
                term_read, term_write = os.pipe()
                term_conn, term_reply, _remainder = broker.request_launch(
                    sock_path,
                    expected_peer_uid=os.geteuid(),
                    runner=str(term_runner),
                    env={},
                    fds=[term_write],
                )
                os.close(term_write)
                assert _read_with_deadline(term_read, until_eof=False) == b"ready"
                term_leases.append(term_conn)
                term_reads.append(term_read)
                doomed.append(term_reply["pid"])

            # The lease is still HELD: only the broker's death can end it.
            proc.send_signal(signal.SIGTERM)
            term_times = [
                float(_read_with_deadline(fd, until_eof=False).split()[1])
                for fd in term_reads
            ]
            assert max(term_times) - min(term_times) < 0.5, (
                "shutdown waited on one lease before notifying the next"
            )
            assert proc.wait(timeout=DEADLINE) is not None

            assert _wait_until_gone(pids["child"]), (
                f"child {pids['child']} was orphaned by the broker's shutdown; "
                "start_new_session detached it, so nothing else will ever reap it"
            )
            assert _wait_until_gone(pids["grandchild"]), (
                f"grandchild {pids['grandchild']} was orphaned by the broker's shutdown"
            )
            assert not os.path.exists(sock_path), (
                "the broker left its socket behind on a clean shutdown"
            )
        finally:
            if held is not None:
                held.close()
            for term_conn in term_leases:
                term_conn.close()
            for term_read in term_reads:
                os.close(term_read)
            os.close(read_fd)
            if write_fd != -1:
                os.close(write_fd)

        # --- restart over a STALE socket, but never over something that is not one ------
        stale_path = _socket_path()
        killed, _ = _start_broker("host-only-secret", root, sock_path=stale_path)
        killed.kill()
        killed.wait(timeout=DEADLINE)
        assert os.path.exists(stale_path), "expected an uncleaned socket after SIGKILL"
        contenders = [
            _start_broker(
                "host-only-secret", root, sock_path=stale_path, expect_ready=False
            )[0]
            for _ in range(2)
        ]
        deadline = time.monotonic() + DEADLINE
        while time.monotonic() < deadline and all(p.poll() is None for p in contenders):
            time.sleep(0.05)
        losers = [p for p in contenders if p.poll() is not None]
        winners = [p for p in contenders if p.poll() is None]
        assert len(losers) == len(winners) == 1
        winner = winners[0]
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(stale_path)
        finally:
            probe.close()
        assert os.path.exists(stale_path), "losing startup unlinked the winner's socket"
        _stop_broker(winner)

        # Cleanup is conditional on the bound pathname still naming this broker's socket.
        owned_path = _socket_path()
        first = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        replacement = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            first.bind(owned_path)
            owned_fd = os.open(
                owned_path,
                os.O_PATH
                | os.O_NOFOLLOW
                | os.O_CLOEXEC,  # windows-footgun: ok — linux_only test
            )
            os.unlink(owned_path)
            replacement.bind(owned_path)
            broker._unlink_owned_socket(owned_path, owned_fd)
            assert os.path.exists(owned_path), "cleanup unlinked a replacement socket"
        finally:
            first.close()
            replacement.close()
            with contextlib.suppress(FileNotFoundError):
                os.unlink(owned_path)

        occupied = Path(_socket_path())
        occupied.write_text("not a socket\n", encoding="utf-8")
        refused, _ = _start_broker(
            "host-only-secret", root, sock_path=str(occupied), expect_ready=False
        )
        assert refused.wait(timeout=DEADLINE) != 0
        assert occupied.read_text(encoding="utf-8") == "not a socket\n", (
            "the broker unlinked a path that was not a socket"
        )

        shared_root = tmp_path / "shared-staging"
        shared_root.mkdir()
        os.chmod(shared_root, 0o750)
        assert stat.S_IMODE(shared_root.stat().st_mode) == 0o750
        refused_root, refused_path = _start_broker(
            "host-only-secret", shared_root, expect_ready=False
        )
        try:
            assert refused_root.wait(timeout=DEADLINE) != 0
            assert not os.path.exists(refused_path)
        finally:
            if refused_root.poll() is None:
                _stop_broker(refused_root)

        # Teardown has exactly one owner, including a child spawned while shutdown is
        # taking its registry snapshot. A late registration is handed back to its worker;
        # a registered lease can be claimed only once.
        leases = broker._Leases()
        registry_client, registry_conn = socket.socketpair()
        registered = object()
        try:
            assert leases.register(registry_conn, threading.current_thread()) is True
            assert leases.add(registry_conn, registered) is True
            assert leases.claim(registry_conn, registered) is True
            assert leases.claim(registry_conn, registered) is False
            leases.finished(registry_conn)
            leases.drain()
            late_client, late_conn = socket.socketpair()
            try:
                assert leases.register(late_conn, threading.current_thread()) is False
            finally:
                late_client.close()
                late_conn.close()
        finally:
            registry_client.close()
            registry_conn.close()

        # A child still unreaped at the kill deadline retains a waiter that reaps it later.
        class DelayedExit:
            pid = 999_999_999
            returncode = None

            def __init__(self):
                self.waits = 0

            def wait(self, timeout=None):
                self.waits += 1
                if timeout is not None:
                    raise subprocess.TimeoutExpired("delayed", timeout)
                self.returncode = 0

        delayed = DelayedExit()
        waiter_daemon = None

        class ImmediateThread:
            def __init__(self, *, target, daemon):
                nonlocal waiter_daemon
                waiter_daemon = daemon
                self.target = target

            def start(self):
                self.target()

        monkeypatch.setattr(broker, "_signal_group", lambda *_args: None)
        monkeypatch.setattr(broker, "_wait_unreaped", lambda *_args: None)
        monkeypatch.setattr(broker.threading, "Thread", ImmediateThread)
        broker._terminate_many([delayed])
        assert delayed.returncode == 0
        assert delayed.waits == 2
        assert waiter_daemon is True
    finally:
        # Best-effort sweep for a run that already failed. RuntimeError is suppressed
        # alongside OSError because ``tests/conftest.py``'s live-system guard refuses
        # ``os.kill`` outside the test subtree — which is precisely where an orphaned
        # grandchild ends up. Letting it raise here would bury the assertion that failed.
        for pid in doomed:
            if _pid_running(pid):
                with contextlib.suppress(OSError, RuntimeError):
                    os.kill(
                        pid,
                        signal.SIGKILL,  # windows-footgun: ok — linux_only test
                    )
        _stop_broker(proc)


@pytest.mark.linux_only
def test_shutdown_waits_for_a_worker_spawning_before_process_registration(
    tmp_path, monkeypatch
):
    broker = _load_broker()
    root = _staging_root(tmp_path)
    runner = _stage_runner(root, "blocked_launch.py", "import time; time.sleep(600)\n")
    client, accepted = socket.socketpair()
    leases = broker._Leases()
    launched = threading.Event()
    release_launch = threading.Event()
    child_pid = None
    real_launch = broker._launch

    def launch_then_block(*args, **kwargs):
        nonlocal child_pid
        proc = real_launch(*args, **kwargs)
        child_pid = proc.pid
        launched.set()
        assert release_launch.wait(DEADLINE)
        return proc

    monkeypatch.setattr(broker, "_launch", launch_then_block)
    worker = threading.Thread(
        target=broker._serve_connection,
        args=(accepted, str(root), HANDSHAKE_TIMEOUT, leases),
        daemon=True,
    )
    leases.register(accepted, worker)
    worker.start()
    drain = threading.Thread(target=leases.drain)
    try:
        client.sendall(_launch_body(runner))
        assert launched.wait(DEADLINE), "the real child was not spawned"
        drain.start()
        drain.join(0.2)
        assert drain.is_alive(), (
            "drain returned while a registered worker's spawned child was still "
            "between Popen and process registration"
        )
        release_launch.set()
        drain.join(DEADLINE)
        assert not drain.is_alive(), "drain did not wait for the registered worker"
        assert child_pid is not None and _wait_until_gone(child_pid)

        failed_client, failed_conn = socket.socketpair()

        class FailingThread:
            def __init__(self, **_kwargs):
                pass

            def start(self):
                raise RuntimeError("thread start failed")

        monkeypatch.setattr(broker.threading, "Thread", FailingThread)
        failed_leases = broker._Leases()
        broker._start_worker(failed_conn, str(root), HANDSHAKE_TIMEOUT, failed_leases)
        failed_client.settimeout(DEADLINE)
        assert failed_client.recv(1) == b""
        failed_leases.drain()
        failed_client.close()
    finally:
        release_launch.set()
        client.close()
        accepted.close()
        worker.join(DEADLINE)
        drain.join(DEADLINE)
        if child_pid is not None and _pid_running(child_pid):
            os.kill(
                child_pid,
                signal.SIGKILL,  # windows-footgun: ok — linux_only test
            )


@pytest.mark.linux_only
def test_thread_start_failure_rejects_only_that_connection(tmp_path, monkeypatch):
    broker = _load_broker()
    root = _staging_root(tmp_path)
    runner = _stage_runner(root, "runner.py", "import time; time.sleep(600)\n")
    leases = broker._Leases()
    live_client, live_conn = socket.socketpair()
    rejected_client, rejected_conn = socket.socketpair()
    later_client, later_conn = socket.socketpair()
    real_thread = threading.Thread

    broker._start_worker(live_conn, str(root), HANDSHAKE_TIMEOUT, leases)
    live_client.sendall(_launch_body(runner))
    live_reply = json.loads(live_client.recv(4096).split(b"\n", 1)[0])

    class FailingThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("thread start failed")

    try:
        monkeypatch.setattr(broker.threading, "Thread", FailingThread)
        broker._start_worker(rejected_conn, str(root), HANDSHAKE_TIMEOUT, leases)
        rejected_client.settimeout(DEADLINE)
        assert rejected_client.recv(1) == b""
        assert _pid_running(live_reply["pid"]), (
            "a rejected worker drained an unrelated live lease"
        )

        monkeypatch.setattr(broker.threading, "Thread", real_thread)
        broker._start_worker(later_conn, str(root), HANDSHAKE_TIMEOUT, leases)
        later_client.sendall(_launch_body(runner))
        later_reply = json.loads(later_client.recv(4096).split(b"\n", 1)[0])
        assert later_reply["ok"] is True
        assert _pid_running(later_reply["pid"])
    finally:
        live_client.close()
        rejected_client.close()
        later_client.close()
        leases.drain()


@pytest.mark.linux_only
@pytest.mark.parametrize("pidfd_errno", [errno.EMFILE, errno.ENOSYS])
def test_pidfd_failure_keeps_live_client_lease_open(monkeypatch, pidfd_errno):
    broker = _load_broker()
    client, conn = socket.socketpair()
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    waiter = threading.Thread(target=broker._await_lease_end, args=(conn, proc))
    monkeypatch.setattr(
        broker.os,
        "pidfd_open",
        lambda _pid: (_ for _ in ()).throw(OSError(pidfd_errno, "injected")),
    )
    try:
        waiter.start()
        waiter.join(0.2)
        assert waiter.is_alive(), "pidfd failure was mistaken for child exit"
        assert proc.poll() is None
        client.close()
        waiter.join(DEADLINE)
        assert not waiter.is_alive(), "connection EOF did not end the fallback wait"
    finally:
        client.close()
        conn.close()
        proc.kill()
        proc.wait(timeout=DEADLINE)


@pytest.mark.linux_only
@pytest.mark.parametrize("pidfd_fails", [False, True])
def test_selector_emfile_keeps_live_client_lease_open(monkeypatch, pidfd_fails):
    broker = _load_broker()
    client, conn = socket.socketpair()
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    waiter = threading.Thread(target=broker._await_lease_end, args=(conn, proc))
    if pidfd_fails:
        monkeypatch.setattr(
            broker.os,
            "pidfd_open",
            lambda _pid: (_ for _ in ()).throw(OSError(errno.EMFILE, "injected")),
        )
    monkeypatch.setattr(
        broker.selectors,
        "DefaultSelector",
        lambda: (_ for _ in ()).throw(OSError(errno.EMFILE, "injected")),
    )
    try:
        waiter.start()
        waiter.join(0.2)
        assert waiter.is_alive(), "selector exhaustion was mistaken for lease EOF"
        assert proc.poll() is None
        client.close()
        waiter.join(DEADLINE)
        assert not waiter.is_alive(), "connection EOF did not end descriptor-free wait"
    finally:
        client.close()
        conn.close()
        proc.kill()
        proc.wait(timeout=DEADLINE)


@pytest.mark.linux_only
def test_pidfd_enosys_preserves_sigterm_grace(monkeypatch):
    broker = _load_broker()
    read_fd, write_fd = os.pipe()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import os, signal, time
                def stop(_signum, _frame):
                    os.write({write_fd}, b"term\\n")
                    time.sleep(0.2)
                    os.write({write_fd}, b"done\\n")
                    raise SystemExit(0)
                signal.signal(signal.SIGTERM, stop)
                os.write({write_fd}, b"ready\\n")
                time.sleep(600)
                """
            ),
        ],
        pass_fds=(write_fd,),
        start_new_session=True,
    )
    os.close(write_fd)
    monkeypatch.setattr(
        broker.os,
        "pidfd_open",
        lambda _pid: (_ for _ in ()).throw(OSError(errno.ENOSYS, "injected")),
    )
    try:
        assert _read_with_deadline(read_fd, until_eof=False) == b"ready"
        broker._terminate(proc)
        assert _read_with_deadline(read_fd, until_eof=True) == b"term\ndone\n"
    finally:
        os.close(read_fd)
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=DEADLINE)
