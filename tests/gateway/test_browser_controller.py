from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import signal
import shutil
import socket
import stat
import struct
import subprocess
import tempfile
import threading
import time
import ipaddress
from pathlib import Path

import pytest

from gateway import browser_controller as browser_controller_module
from gateway.browser_controller import (
    AgentBrowserExecutor,
    BrowserControllerConfig,
    BrowserControllerError,
    BrowserControllerServer,
    BrowserSession,
    ExecutableAttestor,
    PeerCredentials,
    PublicURLPolicy,
    SYSTEMD_READY_STATUS,
    notify_systemd_ready,
)
from gateway.browser_controller_protocol import (
    BrowserCommand,
    BrowserControllerProtocolError,
    MAX_ARTIFACT_BYTES,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    MAX_RESULT_BYTES,
    PROTOCOL_VERSION,
    canonical_json,
    normalize_command,
    receive_frame,
    send_frame,
)
from tools import browser_controller_client as browser_controller_client_module
from tools.browser_controller_client import (
    BrowserControllerClient,
    BrowserControllerClientConfig,
    BrowserControllerClientError,
    CLIENT_CONFIG_SCHEMA,
)


PUBLIC_IP = "93.184.216.34"
PUBLIC_URL = "https://example.com/"
PNG = b"\x89PNG\r\n\x1a\n"


def test_systemd_ready_notification_is_exact_and_optional() -> None:
    assert notify_systemd_ready(_notify_socket="") is False

    root = Path(tempfile.mkdtemp(prefix="hbc-notify-", dir="/tmp")).resolve()
    path = root / "notify.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    listener.bind(str(path))
    listener.settimeout(1)
    try:
        assert notify_systemd_ready(_notify_socket=str(path)) is True
        assert listener.recv(1024) == (
            f"READY=1\nSTATUS={SYSTEMD_READY_STATUS}\n"
        ).encode("ascii")
    finally:
        listener.close()
        shutil.rmtree(root)


@pytest.mark.parametrize("value", ["relative", "\x00bad", "bad\npath"])
def test_systemd_ready_notification_rejects_invalid_socket(value: str) -> None:
    with pytest.raises(
        BrowserControllerError,
        match="browser_controller_notify_socket_invalid",
    ):
        notify_systemd_ready(_notify_socket=value)


@pytest.fixture(autouse=True)
def _test_release_config_owner(monkeypatch):
    # Production hard-codes UID 0. Tests run unprivileged, so substitute the
    # test process owner while exercising the same exact-owner invariant.
    monkeypatch.setattr(
        browser_controller_module,
        "_TRUSTED_AGENT_BROWSER_CONFIG_UID",
        os.geteuid(),
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _controller_config(tmp_path: Path, *, allowed_uid: int | None = None):
    root = tmp_path.resolve()
    root.mkdir(parents=True, exist_ok=True)
    release = root / "release"
    release.mkdir()
    paths = {}
    for name, payload in (
        ("node", b"fixed-node"),
        ("wrapper", b"fixed-wrapper"),
        ("native", b"fixed-native"),
        ("chrome", b"fixed-chrome"),
    ):
        path = release / name
        path.write_bytes(payload)
        path.chmod(0o755)
        paths[name] = path
    agent_browser_config = release / "agent-browser.empty.json"
    agent_browser_config.write_bytes(b"{}\n")
    agent_browser_config.chmod(0o444)
    paths["agent_browser_config"] = agent_browser_config
    # macOS limits AF_UNIX paths to ~104 bytes; pytest's per-test path is much
    # longer, so exercise the real socket through a short private runtime root.
    socket_runtime = Path(tempfile.mkdtemp(prefix="hbc-", dir="/tmp")).resolve()
    socket_runtime.chmod(0o750)
    session_root = root / "session-state"
    session_root.mkdir(mode=0o700)
    artifact_root = root / "gateway-artifacts"
    artifact_root.mkdir(mode=0o700)
    raw = {
        "schema": "hermes-browser-controller-service.v1",
        "socket_path": str(socket_runtime / "controller.sock"),
        "socket_runtime_root": str(socket_runtime),
        # macOS may inherit a directory group that differs from the process's
        # effective primary group.  The service contract pins the configured
        # client group to the actual owner group of the pre-created runtime
        # directory, so mirror that production invariant here.
        "socket_gid": socket_runtime.stat().st_gid,
        "allowed_client_uid": os.geteuid() if allowed_uid is None else allowed_uid,
        "session_root": str(session_root),
        "release_root": str(release),
        "node_path": str(paths["node"]),
        "node_sha256": _sha(paths["node"]),
        "wrapper_path": str(paths["wrapper"]),
        "wrapper_sha256": _sha(paths["wrapper"]),
        "native_path": str(paths["native"]),
        "native_sha256": _sha(paths["native"]),
        "chrome_path": str(paths["chrome"]),
        "chrome_sha256": _sha(paths["chrome"]),
        "agent_browser_config_path": str(agent_browser_config),
        "agent_browser_config_sha256": _sha(agent_browser_config),
        "command_timeout_seconds": 5,
        "idle_timeout_seconds": 30,
        "max_connections": 8,
        "max_sessions": 8,
        "session_quota_bytes": 128 * 1024 * 1024,
        "session_quota_entries": 10_000,
    }
    config = BrowserControllerConfig.from_mapping(raw)
    client = BrowserControllerClientConfig.from_mapping(
        {
            "schema": CLIENT_CONFIG_SCHEMA,
            "socket_path": str(config.socket_path),
            "server_uid": os.geteuid(),
            "artifact_root": str(artifact_root),
            "connect_timeout_seconds": 2,
            "request_timeout_seconds": 10,
        }
    )
    return config, client, paths


def _resolver(host: str, port: int, *_args):
    try:
        address = str(ipaddress.ip_address(host))
    except ValueError:
        address = PUBLIC_IP
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    sockaddr = (address, port, 0, 0) if family == socket.AF_INET6 else (address, port)
    return [(family, socket.SOCK_STREAM, 6, "", sockaddr)]


class RecordingExecutor:
    def __init__(self) -> None:
        self.urls: dict[str, str] = {}
        self.commands: list[tuple[str, str, tuple[str, ...]]] = []
        self.closed: list[str] = []
        self.cleanup_calls = 0

    def execute(self, session: BrowserSession, command: BrowserCommand):
        self.commands.append(
            (session.agent_session_name, command.name, command.argv)
        )
        current = self.urls.setdefault(session.agent_session_name, "about:blank")
        if command.name in {"open", "internal.blank"}:
            current = command.argv[0]
            self.urls[session.agent_session_name] = current
            return {"success": True, "data": {"url": current, "title": "ok"}}
        if command.name == "current_url":
            return {"success": True, "data": {"result": current}}
        if command.name == "snapshot":
            return {
                "success": True,
                "data": {
                    "snapshot": f"session={session.agent_session_name};url={current}",
                    "refs": {},
                },
            }
        if command.name == "screenshot":
            return {
                "success": True,
                "data": {},
                "artifact": {
                    "encoding": "base64",
                    "media_type": "image/png",
                    "sha256": hashlib.sha256(PNG).hexdigest(),
                    "size": len(PNG),
                    "data": base64.b64encode(PNG).decode("ascii"),
                },
            }
        return {"success": True, "data": {}}

    def close(self, session: BrowserSession) -> None:
        self.closed.append(session.agent_session_name)
        shutil.rmtree(session.root, ignore_errors=True)

    def cleanup_stale_sessions(self) -> None:
        self.cleanup_calls += 1


def _start_server(config, executor):
    policy = PublicURLPolicy(resolver=_resolver, website_checker=lambda _url: None)
    server = BrowserControllerServer(config, executor=executor, url_policy=policy)
    # Bind synchronously so a client cannot observe the filesystem socket in
    # the small interval before its ownership/mode/listen contract is ready.
    server.bind()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    assert config.socket_path.exists()
    return server, thread


def _stop_server(server, thread):
    server.stop()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_protocol_is_structured_and_has_no_raw_escape_surface() -> None:
    assert normalize_command("open", [PUBLIC_URL]).argv == (PUBLIC_URL,)
    assert normalize_command("snapshot", ["-c"]).argv == ("-c",)
    assert normalize_command("fill", ["@e1", "hello"]).argv == (
        "@e1",
        "hello",
    )
    for command, args in (
        ("eval", ["fetch('http://169.254.169.254/latest/meta-data')"]),
        ("cdp", ["Runtime.evaluate"]),
        ("cookies", []),
        ("tabs", []),
        ("profile", ["/tmp/other"]),
        ("download", ["/etc/passwd"]),
        ("install", []),
    ):
        with pytest.raises(
            BrowserControllerProtocolError,
            match="browser_controller_command_forbidden",
        ):
            normalize_command(command, args)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://10.1.2.3/",
        "http://172.16.4.5/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://[::1]/",
    ],
)
def test_public_url_policy_blocks_private_loopback_rfc1918_and_metadata(url) -> None:
    policy = PublicURLPolicy(resolver=_resolver, website_checker=lambda _url: None)
    with pytest.raises(BrowserControllerError, match="url_not_public"):
        policy.validate(url)


def test_public_url_policy_allows_public_and_fails_closed_on_probe_exception() -> None:
    policy = PublicURLPolicy(resolver=_resolver, website_checker=lambda _url: None)
    assert policy.validate(PUBLIC_URL) == PUBLIC_URL

    def broken(*_args):
        raise socket.gaierror("offline")

    with pytest.raises(BrowserControllerError, match="url_probe_failed"):
        PublicURLPolicy(
            resolver=broken, website_checker=lambda _url: None
        ).validate(PUBLIC_URL)
    with pytest.raises(BrowserControllerError, match="app_policy_failed"):
        PublicURLPolicy(
            resolver=_resolver,
            website_checker=lambda _url: (_ for _ in ()).throw(RuntimeError()),
        ).validate(PUBLIC_URL)


def test_exact_executor_ignores_path_npx_and_inherited_secrets_and_detects_drift(
    tmp_path, monkeypatch
) -> None:
    config, _client, paths = _controller_config(tmp_path)
    captured = {}
    malicious_home = tmp_path / "malicious-home"
    malicious_home.mkdir()
    (malicious_home / ".agent-browser").mkdir()
    (malicious_home / ".agent-browser" / "config.json").write_text(
        '{"headed":true,"cdp":"http://127.0.0.1:9222"}',
        encoding="utf-8",
    )
    (config.release_root / "agent-browser.json").write_text(
        '{"headed":true,"extension":"/tmp/evil"}',
        encoding="utf-8",
    )

    class Proc:
        returncode = 0
        pid = 999999

        def __init__(self, argv, **kwargs):
            captured["argv"] = list(argv)
            captured["env"] = dict(kwargs["env"])
            captured["cwd"] = kwargs["cwd"]
            assert list(Path(kwargs["cwd"]).iterdir()) == []
            os.write(
                kwargs["stdout"],
                json.dumps(
                    {"success": True, "data": {"url": PUBLIC_URL}}
                ).encode(),
            )

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return self.returncode

    monkeypatch.setenv("PATH", "/tmp/evil-path")
    monkeypatch.setenv("HOME", str(malicious_home))
    monkeypatch.setenv("NPM_CONFIG_PREFIX", "/tmp/evil-npm")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-cross")
    attestor = ExecutableAttestor(config)
    executor = AgentBrowserExecutor(config, attestor, popen=Proc)
    session_root = config.session_root / "session-executor"
    session_root.mkdir(mode=0o700)
    session = BrowserSession("a" * 64, "hbc_fixed", session_root, time.monotonic())
    result = executor.execute(
        session, BrowserCommand("open", "open", (PUBLIC_URL,))
    )
    assert result["success"] is True
    assert captured["argv"][:2] == [str(config.node_path), str(config.wrapper_path)]
    config_index = captured["argv"].index("--config")
    assert captured["argv"][config_index + 1] == str(
        config.agent_browser_config_path
    )
    assert "npx" not in captured["argv"]
    assert captured["env"]["PATH"] == "/usr/bin:/bin"
    assert "OPENAI_API_KEY" not in captured["env"]
    assert "NPM_CONFIG_PREFIX" not in captured["env"]
    assert captured["env"]["AGENT_BROWSER_EXECUTABLE_PATH"] == str(
        config.chrome_path
    )
    assert captured["env"]["AGENT_BROWSER_CONFIG"] == str(
        config.agent_browser_config_path
    )
    assert captured["env"]["HOME"] != str(malicious_home)
    assert Path(captured["cwd"]).parent == session.root / "work"
    assert Path(captured["cwd"]).is_relative_to(session.root)
    assert "--session" in captured["argv"]
    assert "--cdp" not in captured["argv"]

    paths["native"].write_bytes(b"drifted-native")
    paths["native"].chmod(0o755)
    with pytest.raises(BrowserControllerError, match="executable_drifted"):
        executor.execute(session, BrowserCommand("snapshot", "snapshot", ("-c",)))


def test_agent_browser_config_must_be_root_owned_release_local_exact_empty_json(
    tmp_path, monkeypatch
) -> None:
    config, _client, paths = _controller_config(tmp_path)
    path = paths["agent_browser_config"]
    path.chmod(0o644)
    path.write_bytes(b'{"headed":true}\n')
    malicious = dataclasses.replace(
        config,
        agent_browser_config_sha256=_sha(path),
    )
    with pytest.raises(BrowserControllerError, match="config_invalid"):
        ExecutableAttestor(malicious)

    path.write_bytes(b"{}\n")
    path.chmod(0o444)
    if os.geteuid() != 0:
        monkeypatch.setattr(
            browser_controller_module,
            "_TRUSTED_AGENT_BROWSER_CONFIG_UID",
            0,
        )
        with pytest.raises(BrowserControllerError, match="config_invalid"):
            ExecutableAttestor(config)


def test_running_output_flood_is_killed_before_read(tmp_path) -> None:
    config, _client, _paths = _controller_config(tmp_path)

    class FloodProc:
        pid = 999_997

        def __init__(self, _argv, **kwargs):
            self.returncode = None
            self.killed = False
            os.write(kwargs["stdout"], b"x" * (MAX_RESULT_BYTES + 1))

        def poll(self):
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -signal.SIGKILL

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("agent-browser", timeout)
            return self.returncode

    attestor = ExecutableAttestor(config)
    holder = {}

    def spawn(*args, **kwargs):
        holder["proc"] = FloodProc(*args, **kwargs)
        return holder["proc"]

    executor = AgentBrowserExecutor(config, attestor, popen=spawn)
    session_root = config.session_root / "session-output-flood"
    session_root.mkdir(mode=0o700)
    session = BrowserSession("a" * 64, "hbc_output_flood", session_root, time.monotonic())
    with pytest.raises(BrowserControllerError, match="command_output_oversized"):
        executor.execute(session, BrowserCommand("snapshot", "snapshot", ()))
    assert holder["proc"].killed is True


@pytest.mark.parametrize("flood_kind", ["entries", "bytes", "symlink"])
def test_continuous_session_quota_closes_idle_profile_flood(
    tmp_path, flood_kind
) -> None:
    config, client_config, _paths = _controller_config(tmp_path)
    if flood_kind == "entries":
        config = dataclasses.replace(config, session_quota_entries=64)
    elif flood_kind == "bytes":
        config = dataclasses.replace(
            config,
            session_quota_bytes=(
                MAX_ARTIFACT_BYTES + MAX_RESULT_BYTES + 64 * 1024
            ),
        )
    server, thread = _start_server(config, None)
    client = BrowserControllerClient(client_config, "7" * 64)
    try:
        client.connect()
        with server._lock:
            session = next(iter(server._sessions.values()))
        profile = session.root / "profile-flood"
        profile.mkdir(mode=0o700)
        if flood_kind == "entries":
            for index in range(70):
                (profile / f"entry-{index}").write_bytes(b"x")
            expected = "browser_controller_session_entry_quota_exceeded"
        elif flood_kind == "bytes":
            with (profile / "large-profile").open("wb") as handle:
                handle.truncate(config.session_quota_bytes + 1)
            expected = "browser_controller_session_byte_quota_exceeded"
        else:
            (profile / "escape").symlink_to("/etc")
            expected = "browser_controller_session_entry_invalid"
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with server._lock:
                if not server._sessions:
                    break
            time.sleep(0.01)
        with server._lock:
            assert not server._sessions
        assert session.quota_error == expected
        assert not session.root.exists()
    finally:
        client.close()
        _stop_server(server, thread)


def test_quota_scan_rejects_sparse_flood_without_reading_file_bytes(
    tmp_path, monkeypatch
) -> None:
    config, _client, _paths = _controller_config(tmp_path)
    config = dataclasses.replace(
        config,
        session_quota_bytes=MAX_ARTIFACT_BYTES + MAX_RESULT_BYTES + 64 * 1024,
    )
    executor = AgentBrowserExecutor(config, ExecutableAttestor(config))
    root = config.session_root / "session-sparse-quota"
    root.mkdir(mode=0o700)
    session = BrowserSession("a" * 64, "hbc_sparse", root, time.monotonic())
    with (root / "sparse").open("wb") as handle:
        handle.truncate(config.session_quota_bytes + 1)
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _path: (_ for _ in ()).throw(AssertionError("unbounded read")),
    )

    with pytest.raises(BrowserControllerError, match="byte_quota_exceeded"):
        executor._scan_session_tree(session)


def test_bounded_output_read_rejects_path_swap_race(tmp_path, monkeypatch) -> None:
    config, _client, _paths = _controller_config(tmp_path)
    executor = AgentBrowserExecutor(config, ExecutableAttestor(config))
    output = tmp_path / "output"
    output.write_bytes(b"safe")
    protected = tmp_path / "protected-output"
    protected.write_bytes(b"must-not-read")
    real_open = os.open
    swapped = False

    def raced_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if Path(path) == output and not swapped:
            swapped = True
            output.unlink()
            output.symlink_to(protected)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", raced_open)
    with pytest.raises(BrowserControllerError, match="output_invalid"):
        executor._read_bounded_regular(
            output,
            maximum=32,
            code="browser_controller_command_output_invalid",
        )
    assert protected.read_bytes() == b"must-not-read"


def test_verified_daemon_cleanup_targets_process_group(tmp_path, monkeypatch) -> None:
    config, _client, _paths = _controller_config(tmp_path)
    executor = AgentBrowserExecutor(config, ExecutableAttestor(config))
    root = config.session_root / "session-daemon-group"
    socket_dir = root / "socket"
    socket_dir.mkdir(mode=0o700, parents=True)
    session = BrowserSession("a" * 64, "hbc_group", root, time.monotonic())
    (socket_dir / "hbc_group.pid").write_text("4242", encoding="ascii")
    calls = []
    monkeypatch.setattr(executor, "_pid_bound_to_session", lambda *_args: True)
    monkeypatch.setattr(os, "getpgid", lambda _pid: 5151)
    monkeypatch.setattr(os, "getpgrp", lambda: 6161)

    def killpg(pgid, sig):
        calls.append((pgid, sig))
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(
        os,
        "kill",
        lambda *_args: (_ for _ in ()).throw(AssertionError("single PID kill")),
    )
    executor._kill_daemon(session)
    assert calls == [(5151, signal.SIGTERM), (5151, 0)]


def test_daemon_cleanup_refuses_sigkill_after_process_group_identity_loss(
    tmp_path, monkeypatch
) -> None:
    config, _client, _paths = _controller_config(tmp_path)
    executor = AgentBrowserExecutor(config, ExecutableAttestor(config))
    root = config.session_root / "session-reused-group"
    socket_dir = root / "socket"
    socket_dir.mkdir(mode=0o700, parents=True)
    session = BrowserSession("a" * 64, "hbc_reused", root, time.monotonic())
    (socket_dir / "hbc_reused.pid").write_text("4242", encoding="ascii")
    calls = []
    monkeypatch.setattr(executor, "_pid_bound_to_session", lambda *_args: True)
    monkeypatch.setattr(
        executor,
        "_process_group_bound_to_session",
        lambda *_args: False,
    )
    monkeypatch.setattr(os, "getpgid", lambda _pid: 5151)
    monkeypatch.setattr(os, "getpgrp", lambda: 6161)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    clock = iter((0.0, 3.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))

    executor._kill_daemon(session)

    assert calls == [(5151, signal.SIGTERM)]


def test_public_socket_and_private_session_roots_have_exact_access_contract(
    tmp_path,
) -> None:
    config, _client, _paths = _controller_config(tmp_path)
    executor = RecordingExecutor()

    config.socket_runtime_root.chmod(0o700)
    with pytest.raises(BrowserControllerError, match="socket_runtime_root_invalid"):
        BrowserControllerServer(config, executor=executor).bind()

    config.socket_runtime_root.chmod(0o750)
    config.session_root.chmod(0o750)
    with pytest.raises(BrowserControllerError, match="session_root_invalid"):
        BrowserControllerServer(config, executor=executor).bind()

    config.session_root.chmod(0o700)
    wrong_group = dataclasses.replace(config, socket_gid=config.socket_gid + 1)
    with pytest.raises(BrowserControllerError, match="socket_runtime_root_invalid"):
        BrowserControllerServer(wrong_group, executor=executor).bind()


def test_pre_handshake_connection_cap_rejects_socket_flood(tmp_path) -> None:
    config, _client, _paths = _controller_config(tmp_path)
    config = dataclasses.replace(config, max_connections=1, max_sessions=1)
    executor = RecordingExecutor()
    server, thread = _start_server(config, executor)
    first = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    second = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        first.connect(str(config.socket_path))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with server._lock:
                if len(server._connections) == 1:
                    break
            time.sleep(0.01)
        second.settimeout(2)
        second.connect(str(config.socket_path))
        try:
            payload = second.recv(1)
        except (ConnectionResetError, BrokenPipeError):
            payload = b""
        assert payload == b""
        with server._lock:
            assert len(server._connections) == 1
            assert len(server._threads) == 1
    finally:
        first.close()
        second.close()
        _stop_server(server, thread)


def test_peer_uid_is_checked_on_both_ends(tmp_path) -> None:
    config, client_config, _paths = _controller_config(
        tmp_path, allowed_uid=os.geteuid() + 1
    )
    executor = RecordingExecutor()
    server, thread = _start_server(config, executor)
    try:
        denied = BrowserControllerClient(client_config, "a" * 64)
        with pytest.raises(Exception):
            denied.connect()
    finally:
        _stop_server(server, thread)

    # A client must also reject a socket whose authenticated server UID drifts.
    config2, client_config2, _paths = _controller_config(tmp_path / "second")
    executor2 = RecordingExecutor()
    server2, thread2 = _start_server(config2, executor2)
    try:
        spoofed = BrowserControllerClient(
            client_config2,
            "b" * 64,
            peer_getter=lambda _sock: PeerCredentials(
                pid=1, uid=client_config2.server_uid + 1, gid=0
            ),
        )
        with pytest.raises(Exception, match="server_peer_forbidden"):
            spoofed.connect()
    finally:
        _stop_server(server2, thread2)


def test_two_connections_get_private_sessions_and_forbidden_state_surfaces(
    tmp_path,
) -> None:
    config, client_config, _paths = _controller_config(tmp_path)
    executor = RecordingExecutor()
    server, thread = _start_server(config, executor)
    first = BrowserControllerClient(client_config, "1" * 64)
    second = BrowserControllerClient(client_config, "2" * 64)
    try:
        assert first.command("open", [PUBLIC_URL])["success"] is True
        other_url = "https://www.iana.org/"
        assert second.command("open", [other_url])["success"] is True
        first_snapshot = first.command("snapshot", ["-c"])
        second_snapshot = second.command("snapshot", ["-c"])
        first_text = first_snapshot["data"]["snapshot"]
        second_text = second_snapshot["data"]["snapshot"]
        assert PUBLIC_URL in first_text and other_url not in first_text
        assert other_url in second_text and PUBLIC_URL not in second_text
        first_name = first_text.split(";", 1)[0]
        second_name = second_text.split(";", 1)[0]
        assert first_name != second_name
        roots = list(config.session_root.glob("session-*"))
        assert len(roots) == 2 and roots[0] != roots[1]
        for forbidden in ("cookies", "tabs", "profile", "download", "eval"):
            result = first.command(forbidden, ["document.cookie"] if forbidden == "eval" else [])
            assert result == {
                "success": False,
                "error": "browser_controller_command_forbidden",
            }
    finally:
        first.close()
        second.close()
        _stop_server(server, thread)
    assert len(executor.closed) == 2


def test_fragmented_frames_preserve_framing_and_request_ids_cannot_replay(
    tmp_path,
) -> None:
    config, _client_config, _paths = _controller_config(tmp_path)
    executor = RecordingExecutor()
    server, thread = _start_server(config, executor)
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        conn.settimeout(3)
        conn.connect(str(config.socket_path))
        request_id = "11111111-1111-4111-8111-111111111111"
        opening = canonical_json(
            {
                "version": PROTOCOL_VERSION,
                "request_id": request_id,
                "op": "session.open",
                "session_id_sha256": "f" * 64,
            }
        )
        frame = struct.pack("!I", len(opening)) + opening
        conn.sendall(frame[:2])
        # Cross at least one controller polling interval.  The partial header
        # must be retained rather than treated as the start of a new frame.
        time.sleep(0.3)
        conn.sendall(frame[2:])
        ready = receive_frame(conn, maximum=MAX_RESPONSE_BYTES)
        assert ready["status"] == "ok"

        send_frame(
            conn,
            {
                "version": PROTOCOL_VERSION,
                "request_id": request_id,
                "op": "command",
                "command": "snapshot",
                "args": [],
            },
            maximum=MAX_REQUEST_BYTES,
        )
        replayed = receive_frame(conn, maximum=MAX_RESPONSE_BYTES)
        assert replayed == {
            "version": PROTOCOL_VERSION,
            "request_id": request_id,
            "status": "error",
            "error": "browser_controller_request_replayed",
        }
        assert not any(item[1] == "snapshot" for item in executor.commands)
    finally:
        conn.close()
        _stop_server(server, thread)


def test_failed_navigation_cannot_leave_extractable_private_page(tmp_path) -> None:
    config, client_config, _paths = _controller_config(tmp_path)

    class FailedNavigationExecutor(RecordingExecutor):
        def execute(self, session, command):
            if command.name == "open":
                self.commands.append(
                    (session.agent_session_name, command.name, command.argv)
                )
                self.urls[session.agent_session_name] = "http://169.254.169.254/"
                return {"success": False, "error": "navigation_failed"}
            return super().execute(session, command)

    executor = FailedNavigationExecutor()
    server, thread = _start_server(config, executor)
    client = BrowserControllerClient(client_config, "9" * 64)
    try:
        assert client.command("open", [PUBLIC_URL]) == {
            "success": False,
            "error": "navigation_failed",
        }
        assert client.command("snapshot", []) == {
            "success": False,
            "error": "browser_controller_url_not_public",
        }
        assert not any(item[1] == "snapshot" for item in executor.commands)
        assert any(item[1] == "internal.blank" for item in executor.commands)
    finally:
        client.close()
        _stop_server(server, thread)


def test_server_blocks_private_open_and_materializes_bounded_screenshot(tmp_path) -> None:
    config, client_config, _paths = _controller_config(tmp_path)
    executor = RecordingExecutor()
    server, thread = _start_server(config, executor)
    client = BrowserControllerClient(client_config, "c" * 64)
    try:
        blocked = client.command("open", ["http://127.0.0.1/admin"])
        assert blocked == {
            "success": False,
            "error": "browser_controller_url_not_public",
        }
        assert not any(item[1] == "open" for item in executor.commands)
        assert client.command("open", [PUBLIC_URL])["success"] is True
        shot = client.command("screenshot", ["--full", "/etc/passwd"])
        assert shot["success"] is True
        path = Path(shot["data"]["path"])
        assert path.parent == client_config.artifact_root
        assert path.read_bytes() == PNG
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        screenshot_commands = [item for item in executor.commands if item[1] == "screenshot"]
        assert screenshot_commands[-1][2] == ("--full",)
    finally:
        client.close()
        _stop_server(server, thread)


def test_client_artifact_count_cap_and_close_cleanup_are_symlink_safe(
    tmp_path, monkeypatch
) -> None:
    _config, client_config, _paths = _controller_config(tmp_path)
    monkeypatch.setattr(browser_controller_client_module, "MAX_CLIENT_ARTIFACTS", 2)
    client = BrowserControllerClient(client_config, "8" * 64)
    artifact = {
        "encoding": "base64",
        "media_type": "image/png",
        "sha256": hashlib.sha256(PNG).hexdigest(),
        "size": len(PNG),
        "data": base64.b64encode(PNG).decode("ascii"),
    }
    first = client._materialize_artifact(artifact)
    second = client._materialize_artifact(artifact)
    with pytest.raises(BrowserControllerClientError, match="artifact_quota_exceeded"):
        client._materialize_artifact(artifact)

    protected = tmp_path / "protected"
    protected.write_bytes(b"keep")
    first.unlink()
    first.symlink_to(protected)
    client.close()
    assert not first.exists()
    assert not second.exists()
    assert protected.read_bytes() == b"keep"

    monkeypatch.setattr(
        browser_controller_client_module,
        "MAX_CLIENT_ARTIFACT_BYTES",
        len(PNG),
    )
    byte_capped = BrowserControllerClient(client_config, "6" * 64)
    only = byte_capped._materialize_artifact(artifact)
    with pytest.raises(BrowserControllerClientError, match="artifact_quota_exceeded"):
        byte_capped._materialize_artifact(artifact)
    byte_capped.close()
    assert not only.exists()


def test_disconnect_and_controller_restart_cleanup_without_replay(tmp_path) -> None:
    config, client_config, _paths = _controller_config(tmp_path)
    executor = RecordingExecutor()
    server, thread = _start_server(config, executor)
    client = BrowserControllerClient(client_config, "d" * 64)
    assert client.command("open", [PUBLIC_URL])["success"] is True
    sock = client._socket
    assert sock is not None
    sock.shutdown(socket.SHUT_RDWR)
    sock.close()
    client._socket = None
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not executor.closed:
        time.sleep(0.01)
    assert len(executor.closed) == 1
    _stop_server(server, thread)

    executor2 = RecordingExecutor()
    server2, thread2 = _start_server(config, executor2)
    replacement = BrowserControllerClient(client_config, "d" * 64)
    try:
        assert replacement.command("open", [PUBLIC_URL])["success"] is True
        assert executor2.cleanup_calls == 1
        assert executor2.commands[0][0] != executor.commands[0][0]
    finally:
        replacement.close()
        _stop_server(server2, thread2)


def test_server_stop_cleans_live_session(tmp_path) -> None:
    config, client_config, _paths = _controller_config(tmp_path)
    executor = RecordingExecutor()
    server, thread = _start_server(config, executor)
    client = BrowserControllerClient(client_config, "e" * 64)
    assert client.command("open", [PUBLIC_URL])["success"] is True
    _stop_server(server, thread)
    assert len(executor.closed) == 1
    assert client.command("snapshot", []) == {
        "success": False,
        "error": "browser_controller_transport_failed",
    }
