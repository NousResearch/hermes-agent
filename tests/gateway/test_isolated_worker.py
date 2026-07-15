from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

import pytest

import gateway.isolated_worker as worker_module

from gateway.isolated_worker import (
    IsolatedWorkerClient,
    IsolatedWorkerServer,
    MAX_FRAME_BYTES,
    PROTOCOL,
    ProtocolError,
    ReadOnlyBind,
    REQUEST_SCHEMA,
    WorkerPolicy,
    canonical_lease_id,
    canonical_bytes,
    parse_request,
)


def _worker_policy(lease_base: Path, **overrides) -> WorkerPolicy:
    shell = Path("/bin/bash")
    values = {
        "expected_peer_uid": os.getuid(),
        "expected_peer_gid": os.getgid(),
        "socket_uid": os.getuid(),
        "socket_gid": os.getgid(),
        "lease_base": lease_base,
        "lease_uid": os.getuid(),
        "lease_gid": os.getgid(),
        "network_isolated": True,
        "bwrap_path": shell,
        "bwrap_sha256": hashlib.sha256(shell.read_bytes()).hexdigest(),
        "bwrap_uid": os.lstat(shell).st_uid,
        "shell": shell,
        "shell_sha256": hashlib.sha256(shell.read_bytes()).hexdigest(),
        "shell_uid": os.lstat(shell).st_uid,
    }
    values.update(overrides)
    return WorkerPolicy(**values)


@pytest.fixture
def worker(tmp_path: Path, monkeypatch):
    lease_ids = {
        "lease-alpha": canonical_lease_id("session-alpha"),
        "lease-bravo": canonical_lease_id("session-bravo"),
    }
    lease_base = tmp_path / "leases"
    lease_base.mkdir(mode=0o700)
    os.chown(lease_base, os.getuid(), os.getgid())
    os.chmod(lease_base, 0o700)
    roots = {
        name: lease_base / lease_id for name, lease_id in lease_ids.items()
    }
    bwrap_test_path = Path("/bin/bash")
    policy = WorkerPolicy(
        expected_peer_uid=os.getuid(),
        expected_peer_gid=os.getgid(),
        socket_uid=os.getuid(),
        socket_gid=os.getgid(),
        lease_base=lease_base,
        lease_uid=os.getuid(),
        lease_gid=os.getgid(),
        network_isolated=True,
        bwrap_path=bwrap_test_path,
        bwrap_sha256=hashlib.sha256(bwrap_test_path.read_bytes()).hexdigest(),
        shell_sha256=hashlib.sha256(Path("/bin/bash").read_bytes()).hexdigest(),
        maximum_timeout_seconds=3,
    )
    socket_root = Path(tempfile.mkdtemp(prefix="iw-", dir="/tmp"))
    socket_path = socket_root / "worker.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    os.chown(socket_path, -1, os.getgid())
    os.chmod(socket_path, 0o660)
    socket_state = os.lstat(socket_path)
    assert (
        socket_state.st_uid,
        socket_state.st_gid,
        socket_state.st_mode & 0o777,
    ) == (os.getuid(), os.getgid(), 0o660)
    listener.listen(8)
    stop = threading.Event()
    server = IsolatedWorkerServer(policy)

    # macOS has neither Linux SO_PEERCRED nor Python getpeereid(). Production
    # remains fail-closed; this fixture supplies the kernel fact so protocol
    # and lifecycle behavior can be exercised cross-platform.
    monkeypatch.setattr(
        worker_module,
        "_peer_credentials",
        lambda _connection: (os.getuid(), os.getgid()),
    )

    # Unit-test process adapter only. The production method always executes
    # the exact digest-bound bwrap inode and has no raw fallback.
    def test_spawn(*, lease, virtual_cwd, command, environment):
        relative = virtual_cwd.relative_to(Path("/workspace"))
        return worker_module.subprocess.Popen(
            ["/bin/bash", "--noprofile", "--norc", "-c", command],
            cwd=lease.root / relative,
            env=dict(environment),
            stdin=worker_module.subprocess.PIPE,
            stdout=worker_module.subprocess.PIPE,
            stderr=worker_module.subprocess.PIPE,
            start_new_session=True,
        )

    monkeypatch.setattr(server, "_spawn_sandboxed", test_spawn)
    thread = threading.Thread(target=server.serve, args=(listener, stop), daemon=True)
    thread.start()
    try:
        yield socket_path, roots, lease_ids, server
    finally:
        stop.set()
        thread.join(timeout=2)
        listener.close()
        server.close()
        shutil.rmtree(socket_root, ignore_errors=True)


def _client(socket_path: Path, lease_id: str) -> IsolatedWorkerClient:
    return IsolatedWorkerClient(
        socket_path,
        lease_id=lease_id,
        expected_server_uid=os.getuid(),
        expected_server_gid=os.getgid(),
        expected_socket_uid=os.getuid(),
        expected_socket_gid=os.getgid(),
    )


def _collect(client: IsolatedWorkerClient, session_id: str) -> tuple[str, str, dict]:
    stdout = bytearray()
    stderr = bytearray()
    deadline = time.monotonic() + 5
    final: dict = {}
    while time.monotonic() < deadline:
        final = dict(client.poll(session_id, wait_milliseconds=100))
        stdout.extend(base64.b64decode(final["stdout_b64"], validate=True))
        stderr.extend(base64.b64decode(final["stderr_b64"], validate=True))
        if (
            final["state"] != "running"
            and final["drained"]
            and final["complete"]
        ):
            break
    else:  # pragma: no cover - makes a hang fail with a useful assertion
        pytest.fail("worker job did not finish")
    return stdout.decode(), stderr.decode(), final


def _request(**changes):
    value = {
        "schema": REQUEST_SCHEMA,
        "protocol": PROTOCOL,
        "request_id": uuid.uuid4().hex,
        "lease_id": canonical_lease_id("session-alpha"),
        "operation": "exec.start",
        "parameters": {
            "command": "true",
            "cwd": "/tmp",
            "stdin_b64": "",
            "timeout_seconds": 1,
        },
    }
    value.update(changes)
    return value


def test_protocol_requires_exact_canonical_bounded_frames() -> None:
    value = _request()
    assert parse_request(canonical_bytes(value))["protocol"] == PROTOCOL

    pretty = json.dumps(value, sort_keys=True).encode("ascii")
    assert pretty != canonical_bytes(value)
    with pytest.raises(ProtocolError, match="request_not_canonical"):
        parse_request(pretty)

    with pytest.raises(ProtocolError, match="request_fields_not_exact"):
        parse_request(canonical_bytes({**value, "unexpected": True}))

    with pytest.raises(ProtocolError, match="request_frame_invalid"):
        parse_request(b"x" * (MAX_FRAME_BYTES + 1))

    bad_params = dict(value)
    bad_params["parameters"] = {**value["parameters"], "env": {"TOKEN": "x"}}
    with pytest.raises(ProtocolError, match="request_parameters_fields_not_exact"):
        parse_request(canonical_bytes(bad_params))

    duplicate = canonical_bytes(value).replace(
        b'"protocol":"muncho.isolated-worker.v1"',
        b'"protocol":"muncho.isolated-worker.v1","protocol":"muncho.isolated-worker.v1"',
    )
    with pytest.raises(ProtocolError, match="request_json_duplicate_key"):
        parse_request(duplicate)

    nonfinite = canonical_bytes(value).replace(b'"timeout_seconds":1', b'"timeout_seconds":NaN')
    with pytest.raises(ProtocolError, match="request_json_invalid"):
        parse_request(nonfinite)


def test_default_spawn_is_exact_bwrap_only(worker, monkeypatch) -> None:
    _socket_path, roots, lease_ids, server = worker
    captured: dict = {}

    class DummyProcess:
        pass

    def capture(arguments, **kwargs):
        shell_bind_index = arguments.index("--ro-bind-fd")
        shell_descriptor = int(arguments[shell_bind_index + 1])
        shell_state = os.fstat(shell_descriptor)
        configured_shell_state = os.lstat(server.policy.shell)
        assert (shell_state.st_dev, shell_state.st_ino) == (
            configured_shell_state.st_dev,
            configured_shell_state.st_ino,
        )
        assert arguments[shell_bind_index + 2] == "/run/hermes-shell"
        assert shell_descriptor in kwargs["pass_fds"]
        captured["arguments"] = arguments
        captured["kwargs"] = kwargs
        return DummyProcess()

    monkeypatch.setattr(worker_module.subprocess, "Popen", capture)
    lease = server._ensure_lease(lease_ids["lease-alpha"])
    result = IsolatedWorkerServer._spawn_sandboxed(
        server,
        lease=lease,
        virtual_cwd=Path("/workspace"),
        command="printf safe",
        environment={
            "HOME": str(roots["lease-alpha"]),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "TMPDIR": str(roots["lease-alpha"]),
        },
    )
    assert isinstance(result, DummyProcess)
    arguments = captured["arguments"]
    assert arguments[0].startswith("/proc/self/fd/")
    assert arguments[1:6] == [
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--cap-drop",
        "ALL",
    ]
    bind_sources = [
        arguments[index + 1]
        for index, value in enumerate(arguments)
        if value == "--bind"
    ]
    assert len(bind_sources) == 1
    assert bind_sources[0].startswith("/proc/self/fd/")
    assert str(roots["lease-bravo"]) not in arguments
    assert "/run/credentials" not in arguments
    assert arguments[arguments.index("--dir") + 1] == "/run"
    assert "--tmpfs" in arguments and arguments[arguments.index("--tmpfs") + 1] == "/tmp"
    command_index = arguments.index("--") + 1
    assert arguments[command_index] == "/run/hermes-shell"
    assert str(server.policy.shell) not in arguments[command_index:]
    assert captured["kwargs"]["env"] == {}
    assert len(captured["kwargs"]["pass_fds"]) == 3


def test_bwrap_identity_or_digest_drift_fails_closed(tmp_path: Path) -> None:
    lease_base = tmp_path / "leases"
    lease_base.mkdir(mode=0o700)
    os.chown(lease_base, os.getuid(), os.getgid())
    os.chmod(lease_base, 0o700)
    with pytest.raises(ValueError, match="executable_digest_mismatch"):
        WorkerPolicy(
            expected_peer_uid=os.getuid(),
            expected_peer_gid=os.getgid(),
            socket_uid=os.getuid(),
            socket_gid=os.getgid(),
            lease_base=lease_base,
            lease_uid=os.getuid(),
            lease_gid=os.getgid(),
            network_isolated=True,
            bwrap_path=Path("/bin/bash"),
            bwrap_sha256="0" * 64,
            shell_sha256=hashlib.sha256(Path("/bin/bash").read_bytes()).hexdigest(),
        )


def test_two_leases_have_no_cwd_file_or_output_bleed(worker) -> None:
    socket_path, roots, lease_ids, _server = worker
    alpha = _client(socket_path, lease_ids["lease-alpha"])
    bravo = _client(socket_path, lease_ids["lease-bravo"])
    try:
        alpha_id = alpha.start(
            "printf alpha > marker; printf 'alpha:%s' \"$(pwd)\"",
            cwd=Path("/workspace"),
            timeout_seconds=2,
        )
        bravo_id = bravo.start(
            "printf bravo > marker; printf 'bravo:%s' \"$(pwd)\"",
            cwd=Path("/workspace"),
            timeout_seconds=2,
        )
        alpha_out, alpha_err, alpha_final = _collect(alpha, alpha_id)
        bravo_out, bravo_err, bravo_final = _collect(bravo, bravo_id)

        assert alpha_final["state"] == bravo_final["state"] == "exited"
        assert alpha_final["returncode"] == bravo_final["returncode"] == 0
        assert alpha_err == bravo_err == ""
        assert alpha_out == f"alpha:{roots['lease-alpha']}"
        assert bravo_out == f"bravo:{roots['lease-bravo']}"
        assert (roots["lease-alpha"] / "marker").read_text() == "alpha"
        assert (roots["lease-bravo"] / "marker").read_text() == "bravo"

        with pytest.raises(ProtocolError, match="cwd_invalid|cwd_outside_lease"):
            alpha.start(
                "true",
                cwd=Path("/workspace/../outside"),
                timeout_seconds=1,
            )
        with pytest.raises(ProtocolError, match="session_not_authorized"):
            bravo.poll(alpha_id)
    finally:
        alpha.close()
        bravo.close()


def test_worker_forwards_exact_allowlisted_environment_only(worker, monkeypatch) -> None:
    socket_path, roots, lease_ids, _server = worker
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-cross")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/must-not-cross.so")
    client = _client(socket_path, lease_ids["lease-alpha"])
    try:
        session_id = client.start(
            "printf '%s|%s|%s' \"${OPENAI_API_KEY-unset}\" "
            "\"${LD_PRELOAD-unset}\" \"$PATH\"",
            cwd=Path("/workspace"),
            timeout_seconds=2,
        )
        stdout, stderr, final = _collect(client, session_id)
        assert final["returncode"] == 0
        assert stderr == ""
        assert stdout == "unset|unset|/usr/bin:/bin"
        assert "must-not-cross" not in stdout
    finally:
        client.close()


def test_timeout_and_cancel_are_session_bound(worker) -> None:
    socket_path, roots, lease_ids, _server = worker
    client = _client(socket_path, lease_ids["lease-alpha"])
    try:
        timed = client.start(
            "sleep 5", cwd=Path("/workspace"), timeout_seconds=1
        )
        _stdout, _stderr, timed_final = _collect(client, timed)
        assert timed_final["state"] == "timed_out"
        assert timed_final["returncode"] is not None

        cancelled = client.start(
            "sleep 5", cwd=Path("/workspace"), timeout_seconds=3
        )
        receipt = client.cancel(cancelled)
        assert receipt == {"session_id": cancelled, "state": "cancelled"}
        _stdout, _stderr, cancelled_final = _collect(client, cancelled)
        assert cancelled_final["state"] == "cancelled"
        assert cancelled_final["returncode"] is not None
    finally:
        client.close()


def test_disconnect_kills_lease_job(worker) -> None:
    socket_path, roots, lease_ids, _server = worker
    client = _client(socket_path, lease_ids["lease-alpha"])
    session_id = client.start(
        "printf '%s' $$ > pid; sleep 30",
        cwd=Path("/workspace"),
        timeout_seconds=3,
    )
    assert session_id.startswith("job-")
    pid_path = roots["lease-alpha"] / "pid"
    deadline = time.monotonic() + 2
    while not pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    pid = int(pid_path.read_text())
    client.close()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        pytest.fail("worker child survived client disconnect")


def test_accept_loop_closes_connections_above_global_cap(worker, monkeypatch) -> None:
    socket_path, _roots, _lease_ids, server = worker
    monkeypatch.setattr(worker_module, "MAX_ACTIVE_CONNECTIONS", 1)

    first = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    second = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    first.connect(str(socket_path))
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with server._threads_lock:
            if len(server._threads) == 1:
                break
        time.sleep(0.01)
    else:
        pytest.fail("first connection was not accepted")

    try:
        second.connect(str(socket_path))
        second.settimeout(0.1)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                if second.recv(1) == b"":
                    break
            except socket.timeout:
                continue
        else:
            pytest.fail("connection above cap was not closed")
    finally:
        first.close()
        second.close()


def test_symlink_cwd_and_peer_uid_spoof_fail_closed(worker, tmp_path: Path) -> None:
    socket_path, roots, lease_ids, server = worker
    server._ensure_lease(lease_ids["lease-alpha"])
    link = roots["lease-alpha"] / "escape"
    link.symlink_to(tmp_path)
    client = _client(socket_path, lease_ids["lease-alpha"])
    try:
        with pytest.raises(
            ProtocolError,
            match="cwd_symlink_or_not_directory|lease_contains_symlink",
        ):
            client.start("true", cwd=Path("/workspace/escape"), timeout_seconds=1)
    finally:
        client.close()

    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    original = worker_module._peer_credentials
    worker_module._peer_credentials = lambda _connection: (os.getuid() + 1, os.getgid())
    try:
        with pytest.raises(ProtocolError, match="peer_uid_not_authorized"):
            server.serve_connection(left)
    finally:
        worker_module._peer_credentials = original
        right.close()


def test_rejected_start_releases_lease_job_reservation(worker, monkeypatch) -> None:
    _socket_path, _roots, lease_ids, server = worker
    lease = server._ensure_lease(lease_ids["lease-alpha"])
    monkeypatch.setattr(
        server,
        "_validate_cwd",
        lambda _lease, _cwd: (_ for _ in ()).throw(ProtocolError("cwd_invalid")),
    )

    with pytest.raises(ProtocolError, match="cwd_invalid"):
        server._start(
            lease.lease_id,
            {
                "command": "true",
                "cwd": "/workspace",
                "stdin_b64": "",
                "timeout_seconds": 1,
            },
            {},
        )

    assert lease.jobs == 0


def test_workspace_scan_failure_kills_child_instead_of_losing_monitor(
    worker,
    monkeypatch,
) -> None:
    socket_path, _roots, lease_ids, server = worker
    process_started = threading.Event()
    original_spawn = server._spawn_sandboxed
    original_usage = server._global_usage

    def mark_started(**kwargs):
        process = original_spawn(**kwargs)
        process_started.set()
        return process

    def fail_after_start():
        if process_started.is_set():
            raise OSError("workspace changed during scan")
        return original_usage()

    monkeypatch.setattr(server, "_spawn_sandboxed", mark_started)
    monkeypatch.setattr(server, "_global_usage", fail_after_start)
    client = _client(socket_path, lease_ids["lease-alpha"])
    try:
        session_id = client.start(
            "sleep 5",
            cwd=Path("/workspace"),
            timeout_seconds=3,
        )
        _stdout, _stderr, final = _collect(client, session_id)
        assert final["state"] == "quota_exceeded"
        assert final["returncode"] is not None
    finally:
        client.close()


def test_global_byte_quota_blocks_aggregate_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease_base = tmp_path / "leases"
    lease_base.mkdir(mode=0o700)
    os.chown(lease_base, os.getuid(), os.getgid())
    os.chmod(lease_base, 0o700)
    policy = _worker_policy(
        lease_base,
        lease_quota_bytes=4096,
        lease_quota_entries=4,
        global_quota_bytes=6000,
        global_quota_entries=10,
    )
    server = IsolatedWorkerServer(policy)
    alpha = server._ensure_lease(canonical_lease_id("aggregate-alpha"))
    bravo = server._ensure_lease(canonical_lease_id("aggregate-bravo"))
    try:
        (alpha.root / "payload").write_bytes(b"a" * 3000)
        (bravo.root / "payload").write_bytes(b"b" * 3000)
        assert server._global_usage() == (4, 6000)

        (bravo.root / "overage").write_bytes(b"x")
        assert server._lease_usage(alpha) == (1, 3000)
        assert server._lease_usage(bravo) == (2, 3001)
        with pytest.raises(ProtocolError, match="global_quota_exceeded"):
            server._global_usage()

        monkeypatch.setattr(
            server,
            "_spawn_sandboxed",
            lambda **_kwargs: pytest.fail("aggregate-over-quota job was spawned"),
        )
        with pytest.raises(ProtocolError, match="global_quota_exceeded"):
            server._start(
                alpha.lease_id,
                {
                    "command": "true",
                    "cwd": "/workspace",
                    "stdin_b64": "",
                    "timeout_seconds": 1,
                },
                {},
            )
    finally:
        server.close()


def test_global_entry_quota_counts_all_lease_roots(tmp_path: Path) -> None:
    lease_base = tmp_path / "leases"
    lease_base.mkdir(mode=0o700)
    os.chown(lease_base, os.getuid(), os.getgid())
    os.chmod(lease_base, 0o700)
    policy = _worker_policy(
        lease_base,
        lease_quota_bytes=4096,
        lease_quota_entries=4,
        global_quota_bytes=8192,
        global_quota_entries=5,
    )
    server = IsolatedWorkerServer(policy)
    alpha = server._ensure_lease(canonical_lease_id("entries-alpha"))
    bravo = server._ensure_lease(canonical_lease_id("entries-bravo"))
    try:
        (alpha.root / "one").mkdir()
        (alpha.root / "two").mkdir()
        (bravo.root / "one").mkdir()
        assert server._global_usage() == (5, 0)

        (bravo.root / "two").mkdir()
        assert server._lease_usage(alpha) == (2, 0)
        assert server._lease_usage(bravo) == (2, 0)
        with pytest.raises(ProtocolError, match="global_quota_exceeded"):
            server._global_usage()
    finally:
        server.close()


def test_dynamic_lease_cap_ttl_quota_and_canonical_ids(tmp_path: Path) -> None:
    lease_base = tmp_path / "leases"
    lease_base.mkdir(mode=0o700)
    os.chown(lease_base, os.getuid(), os.getgid())
    os.chmod(lease_base, 0o700)
    policy = _worker_policy(
        lease_base,
        maximum_active_leases=1,
        lease_ttl_seconds=10,
        lease_quota_bytes=4096,
        lease_quota_entries=4,
    )
    server = IsolatedWorkerServer(policy)
    alpha_id = canonical_lease_id("session-alpha")
    bravo_id = canonical_lease_id("session-bravo")
    try:
        alpha = server._ensure_lease(alpha_id)
        assert alpha.root == lease_base / alpha_id
        assert alpha.root.is_dir()

        with pytest.raises(ProtocolError, match="lease_id_not_canonical"):
            server._ensure_lease("../caller-path")
        with pytest.raises(ProtocolError, match="lease_capacity_exhausted"):
            server._ensure_lease(bravo_id)

        (alpha.root / "oversized").write_bytes(b"x" * 4097)
        with pytest.raises(ProtocolError, match="lease_quota_exceeded"):
            server._lease_usage(alpha)
        (alpha.root / "oversized").unlink()

        removed = server.reap_expired(
            now_monotonic=alpha.last_used_monotonic + policy.lease_ttl_seconds + 1
        )
        assert removed == (alpha_id,)
        assert not alpha.root.exists()
        assert server._ensure_lease(bravo_id).root == lease_base / bravo_id
    finally:
        server.close()


def test_existing_lease_ttl_survives_server_restart(tmp_path: Path) -> None:
    lease_base = tmp_path / "leases"
    lease_base.mkdir(mode=0o700)
    os.chown(lease_base, os.getuid(), os.getgid())
    os.chmod(lease_base, 0o700)
    policy = _worker_policy(lease_base, lease_ttl_seconds=1)
    lease_id = canonical_lease_id("stale-session")
    root = lease_base / lease_id
    root.mkdir(mode=0o700)
    stale = time.time() - 60
    os.utime(root, (stale, stale))

    server = IsolatedWorkerServer(policy)
    try:
        assert server.reap_expired() == (lease_id,)
        assert not root.exists()
    finally:
        server.close()


def test_read_only_bind_is_config_only_and_not_worker_mutable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(worker_module, "HOST_READ_ONLY_ROOT", tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o700)
    os.chown(shared, os.getuid(), os.getgid())
    reference = shared / "reference.txt"
    reference.write_text("sealed", encoding="utf-8")
    os.chown(reference, os.getuid(), os.getgid())
    reference.chmod(0o400)
    shared.chmod(0o500)
    bind = ReadOnlyBind(
        source=shared,
        destination=Path("/opt/hermes-shared/reference"),
        source_uid=os.getuid(),
        source_gid=os.getgid(),
    )
    lease_base = tmp_path / "leases"
    lease_base.mkdir(mode=0o700)
    os.chown(lease_base, os.getuid(), os.getgid())
    os.chmod(lease_base, 0o700)
    with pytest.raises(ValueError, match="read_only_bind_mutable_by_worker"):
        _worker_policy(lease_base, read_only_binds=(bind,))

    nested = tmp_path / "nested" / "reference"
    nested.mkdir(parents=True, mode=0o500)
    with pytest.raises(ValueError, match="read_only_bind_source_namespace_invalid"):
        ReadOnlyBind(
            source=nested,
            destination=Path("/opt/hermes-shared/nested"),
            source_uid=os.getuid(),
            source_gid=os.getgid(),
        )

    forbidden = tmp_path / "skills"
    forbidden.mkdir(mode=0o500)
    with pytest.raises(ValueError, match="read_only_bind_source_forbidden"):
        ReadOnlyBind(
            source=forbidden,
            destination=Path("/opt/hermes-shared/skills"),
            source_uid=os.getuid(),
            source_gid=os.getgid(),
        )
