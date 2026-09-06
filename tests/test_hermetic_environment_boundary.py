"""Regression proof for the runner-level Hermes hermetic boundary."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading

import pytest


def test_runner_uses_synthetic_home_and_os_sandbox():
    home = Path(os.environ["HOME"]).resolve()
    hermes_home = Path(os.environ["HERMES_HOME"]).resolve()
    real_home = Path(os.environ["HERMES_TEST_REAL_HOME"]).resolve()
    assert home != real_home
    assert not hermes_home.is_relative_to(real_home)
    assert os.environ["HERMES_TEST_OS_SANDBOX"] in {
        "linux-bwrap",
        "macos-sandbox-exec",
        "windows-ephemeral-ci",
    }


@pytest.mark.parametrize(
    "command",
    [
        ["launchctl", "kickstart", "-k", "gui/501/ai.hermes.gateway"],
        ["systemctl", "--user", "restart", "hermes-gateway"],
        ["pkill", "-f", "python"],
        ["security", "find-generic-password", "-s", "hermes"],
        ["op", "read", "op://vault/item/password"],
        ["cmdkey.exe", "/list"],
        ["curl", "https://example.invalid/"],
        ["docker", "info"],
        ["netsh", "advfirewall", "set", "allprofiles", "state", "off"],
        ["git", "ls-remote", "https://example.invalid/repo.git"],
    ],
)
def test_forbidden_subprocess_classes_fail_before_exec(command):
    with pytest.raises(RuntimeError, match="guard"):
        subprocess.run(command, check=False)


def test_shell_wrapped_forbidden_operation_is_blocked():
    with pytest.raises(RuntimeError, match="guard"):
        subprocess.run(
            ["bash", "-c", "launchctl kickstart -k gui/501/ai.hermes.gateway"],
            check=False,
        )


def test_child_python_inherits_guard_before_user_code():
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import subprocess; "
            "subprocess.run(['launchctl', 'print', "
            "'gui/501/ai.hermes.gateway'])",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode != 0
    assert "hermetic-test guard blocked" in (probe.stdout + probe.stderr)


def test_direct_pytest_without_boundary_refuses_before_collection():
    env = os.environ.copy()
    env.pop("HERMES_TEST_OS_SANDBOX", None)
    env.pop("HERMES_TEST_GUARD_ACTIVE", None)
    env.pop("PYTHONPATH", None)
    probe = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_live_system_guard.py",
            "--collect-only",
            "-q",
        ],
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode in {3, 4}
    assert "refusing an unsandboxed pytest process" in (probe.stdout + probe.stderr)


def test_external_network_and_unowned_loopback_are_blocked():
    with pytest.raises(RuntimeError, match="external (network|DNS)"):
        socket.create_connection(("192.0.2.1", 443), timeout=0.01)
    with pytest.raises(RuntimeError, match="non-test-owned loopback"):
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)


def test_native_shell_network_is_blocked_by_os_boundary():
    probe = subprocess.run(
        ["bash", "-c", "exec 3<>/dev/tcp/192.0.2.1/443"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode != 0


@pytest.mark.windows_only
def test_windows_restricted_principal_cannot_remove_firewall_boundary():
    import base64
    import ctypes

    assert ctypes.windll.shell32.IsUserAnAdmin() == 0
    command = "Set-NetFirewallProfile -Profile Domain,Public,Private -DefaultOutboundAction Allow"
    encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
    probe = subprocess.run(
        ["powershell.exe", "-NoProfile", "-EncodedCommand", encoded],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode != 0


@pytest.mark.windows_only
def test_windows_kernel_blocks_unguarded_python_network():
    probe = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import socket,sys; s=socket.socket(); "
            "sys.exit(9 if s.connect_ex(('192.0.2.1',443)) == 0 else 0)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux bwrap proof")
def test_generic_js_rust_runner_has_the_same_kernel_boundary():
    repo = Path(__file__).resolve().parent.parent
    probe = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "run_hermetic_command.py"),
            "--",
            sys.executable,
            "-S",
            "-c",
            "import socket,sys; s=socket.socket(); "
            "sys.exit(0 if s.connect_ex(('192.0.2.1',443)) else 9)",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux bwrap proof")
def test_generic_runner_rejects_arbitrary_writable_cache_bind():
    repo = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["HERMES_TEST_CARGO_HOME"] = str(
        Path(env["HERMES_TEST_REAL_HOME"]) / ".config" / "systemd" / "user"
    )
    probe = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "run_hermetic_command.py"),
            "--",
            sys.executable,
            "-c",
            "print('must not run')",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode != 0
    assert "trusted runner temp root" in (probe.stdout + probe.stderr)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux bwrap proof")
def test_generic_runner_rejects_spoofed_home():
    repo = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env.pop("HERMES_TEST_REAL_HOME", None)
    env["HOME"] = str(Path(env["HERMES_TEST_SANDBOX_ROOT"]) / "spoofed-home")
    probe = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "run_hermetic_command.py"),
            "--",
            sys.executable,
            "-c",
            "print('must not run')",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode != 0
    assert "OS account home" in (probe.stdout + probe.stderr)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux bwrap proof")
def test_python_without_sitecustomize_cannot_read_host_canary_or_unix_socket():
    canary = os.environ["HERMES_TEST_HOST_CANARY"]
    host_socket = os.environ["HERMES_TEST_HOST_SOCKET"]
    read_probe = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import pathlib,sys; sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())",
            canary,
        ],
        capture_output=True,
        check=False,
    )
    assert b"HERMES_HOST_CANARY" not in read_probe.stdout

    socket_probe = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import socket,sys; s=socket.socket(socket.AF_UNIX); s.connect(sys.argv[1])",
            host_socket,
        ],
        capture_output=True,
        check=False,
    )
    assert socket_probe.returncode != 0


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Seatbelt proof")
def test_macos_kernel_blocks_signal_syscall_and_launchctl_copy(tmp_path):
    import ctypes
    import errno

    libc = ctypes.CDLL(None, use_errno=True)
    assert libc.kill(os.getpid(), 0) == -1
    assert ctypes.get_errno() == errno.EPERM

    launchctl = shutil.which("launchctl")
    assert launchctl is not None
    copied = tmp_path / "launchctl-copy"
    copy_probe = subprocess.run(
        ["cp", launchctl, copied],
        capture_output=True,
        text=True,
        check=False,
    )
    assert copy_probe.returncode != 0
    assert not copied.exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Seatbelt proof")
def test_macos_kernel_blocks_native_keychain_broker():
    import ctypes

    security = ctypes.CDLL(
        "/System/Library/Frameworks/Security.framework/Security"
    )
    security.SecKeychainCopyDefault.argtypes = [
        ctypes.POINTER(ctypes.c_void_p)
    ]
    security.SecKeychainCopyDefault.restype = ctypes.c_int32
    keychain = ctypes.c_void_p()
    status = security.SecKeychainCopyDefault(ctypes.byref(keychain))
    # errSecNotAvailable is the Security.framework result when Seatbelt
    # prevents the securityd Mach lookup. Other errors (locked/no default/no
    # item) would not prove that the broker boundary caused the failure.
    assert status == -25291
    assert not keychain.value


def test_test_owned_loopback_listener_is_allowed():
    if os.environ["HERMES_TEST_OS_SANDBOX"] == "macos-sandbox-exec":
        with pytest.raises(OSError):
            socket.socket().bind(("127.0.0.1", 0))
        return
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    accepted: list[bytes] = []

    def serve():
        conn, _ = server.accept()
        with conn:
            accepted.append(conn.recv(4))

    thread = threading.Thread(target=serve)
    thread.start()
    with socket.create_connection(server.getsockname(), timeout=1) as client:
        client.sendall(b"test")
    thread.join(timeout=1)
    server.close()
    assert accepted == [b"test"]


def test_real_credential_and_hermes_paths_are_unreadable_and_unwritable():
    real_home = Path(os.environ["HERMES_TEST_REAL_HOME"])
    for relative in (
        ".claude/.credentials.json",
        ".codex/auth.json",
        ".copilot/config.json",
        ".config/github-copilot/hosts.json",
        ".minimax/credentials.json",
        ".future-provider/credentials.json",
    ):
        with pytest.raises(RuntimeError, match="credential/Hermes"):
            (real_home / relative).read_text()
    with pytest.raises(RuntimeError, match="credential/Hermes"):
        (real_home / ".hermes" / "config.yaml").write_text("forbidden")


def test_monkeypatch_undo_cannot_remove_process_boundary(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="hermetic-test guard"):
        subprocess.run(["security", "find-generic-password", "-s", "hermes"])
