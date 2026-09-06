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

from scripts.run_tests_parallel import _trusted_sandbox_executable


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
    for raw_entry in os.environ["PATH"].split(os.pathsep):
        entry = Path(raw_entry).resolve()
        assert entry != real_home
        assert not entry.is_relative_to(real_home)


def test_os_sandbox_launcher_ignores_ambient_path(monkeypatch):
    fake = Path(os.environ["HOME"]) / "bin" / "sandbox-exec"
    monkeypatch.setattr(shutil, "which", lambda *_args, **_kwargs: str(fake))
    if sys.platform.startswith("linux"):
        resolved = _trusted_sandbox_executable(Path("/usr/bin/bwrap"), "bubblewrap")
    elif sys.platform == "darwin":
        resolved = _trusted_sandbox_executable(
            Path("/usr/bin/sandbox-exec"), "sandbox-exec"
        )
    else:
        pytest.skip("fixed launcher attestation is POSIX-only")
    assert Path(resolved).is_absolute()
    assert Path(resolved) != fake


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
        ["env", "git", "fetch", "https://example.invalid/repo.git"],
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
    with pytest.raises(RuntimeError, match="git remote network operation"):
        subprocess.run(
            ["bash", "-c", "echo preflight; git fetch https://example.invalid/live.git"],
            check=False,
        )
    for command in (
        ["env", "bash", "-c", "git fetch https://example.invalid/live.git"],
        ["bash", "--norc", "-c", "git fetch https://example.invalid/live.git"],
        ["bash", "-c", '"$@"', "_", "launchctl", "kickstart", "ai.hermes.gateway"],
        ["env", "-S", "git fetch https://example.invalid/live.git"],
        ["env", "-Sgit fetch https://example.invalid/live.git"],
        ["env", "-C", str(Path.cwd()), "git", "fetch", "https://example.invalid/live.git"],
        ["xargs", "git", "fetch"],
    ):
        with pytest.raises(RuntimeError, match="guard"):
            subprocess.run(
                command,
                input="https://example.invalid/live.git\n",
                text=True,
                encoding="utf-8",
                errors="replace",
            )


def test_git_local_fixture_operations_are_allowed_but_remote_alias_is_blocked(
    tmp_path,
):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    origin = tmp_path / "origin"
    clone = tmp_path / "clone"
    origin.mkdir()

    def git(cwd, *args):
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    git(origin, "init", "-q")
    git(origin, "config", "user.email", "test@example.invalid")
    git(origin, "config", "user.name", "Hermetic Test")
    (origin / "tracked.txt").write_text("one\n", encoding="utf-8")
    git(origin, "add", "tracked.txt")
    git(origin, "commit", "-qm", "initial")
    git(tmp_path, "clone", "-q", str(origin), str(clone))
    (clone / "tracked.txt").write_text("two\n", encoding="utf-8")
    git(clone, "stash", "push", "-m", "local-only")
    git(clone, "fetch", "-q", "origin")
    subprocess.run(["git", "push", "-u", "origin", "HEAD"], cwd=clone, check=False)
    with pytest.raises(RuntimeError, match="git remote network operation"):
        subprocess.run(["git", "fetch"], cwd=clone, check=False)
    (clone / ".gitmodules").write_text(
        '[submodule "live"]\n\tpath = live\n\turl = https://example.invalid/live.git\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="git remote network operation"):
        subprocess.run(["git", "fetch", "origin"], cwd=clone, check=False)
    subprocess.run(
        ["git", "fetch", "--no-recurse-submodules", "origin"],
        cwd=clone,
        check=True,
    )
    (clone / ".gitmodules").unlink()

    env = os.environ.copy()
    env["GIT_DIR"] = str(clone / ".git")
    with pytest.raises(RuntimeError, match="git remote network operation"):
        subprocess.run(["git", "fetch", "origin"], cwd=clone, env=env, check=False)
    unsafe_env = os.environ.copy()
    unsafe_env.pop("GIT_CONFIG_NOSYSTEM")
    unsafe_env.pop("GIT_CONFIG_GLOBAL")
    with pytest.raises(RuntimeError, match="git remote network operation"):
        subprocess.run(
            ["git", "fetch", str(origin)], cwd=clone, env=unsafe_env, check=False
        )
    with pytest.raises(RuntimeError, match="guard"):
        subprocess.run(
            [
                "env",
                "-u",
                "GIT_CONFIG_NOSYSTEM",
                "-u",
                "GIT_CONFIG_GLOBAL",
                "git",
                "fetch",
                str(origin),
            ],
            cwd=clone,
            check=False,
        )
    with pytest.raises(RuntimeError, match="git remote network operation"):
        subprocess.run(
            ["git", "fetch", "--upload-pack=/bin/false", str(origin)],
            cwd=clone,
            check=False,
        )
    with pytest.raises(RuntimeError, match="git remote network operation"):
        subprocess.run(
            ["git", "clone", "-u/bin/false", str(origin), str(tmp_path / "bad-clone")],
            cwd=tmp_path,
            check=False,
        )
    with pytest.raises(RuntimeError, match="git remote network operation"):
        subprocess.run(
            ["git", "clone", "-qu/bin/false", str(origin), str(tmp_path / "bad-cluster")],
            cwd=tmp_path,
            check=False,
        )
    for secondary in (
        "--recurse-submodules",
        "--bundle-uri=https://example.invalid/bundle",
        "--config=url.https://example.invalid/.insteadOf=/",
    ):
        with pytest.raises(RuntimeError, match="git remote network operation"):
            subprocess.run(
                ["git", "clone", secondary, str(origin), str(tmp_path / "secondary")],
                cwd=tmp_path,
                check=False,
            )
    exec_env = os.environ.copy()
    exec_env["GIT_EXEC_PATH"] = str(tmp_path)
    with pytest.raises(RuntimeError, match="git remote network operation"):
        subprocess.run(
            ["git", "fetch", str(origin)], cwd=clone, env=exec_env, check=False
        )
    with pytest.raises(RuntimeError, match="git remote network operation"):
        subprocess.run(
            ["git", f"--exec-path={tmp_path}", "fetch", str(origin)],
            cwd=clone,
            check=False,
        )
    with pytest.raises(RuntimeError, match="guard"):
        subprocess.run(
            [
                "env",
                "-uGIT_CONFIG_NOSYSTEM",
                "-uGIT_CONFIG_GLOBAL",
                "git",
                "fetch",
                str(origin),
            ],
            cwd=clone,
            check=False,
        )

    git(clone, "remote", "add", "live", "https://example.invalid/live.git")
    with pytest.raises(RuntimeError, match="git remote network operation"):
        subprocess.run(["git", "fetch", "live"], cwd=clone, check=False)
    with pytest.raises(RuntimeError, match="git remote network operation"):
        subprocess.run(
            [
                "git",
                "-c",
                "remote.origin.url=https://example.invalid/override.git",
                "fetch",
                "origin",
            ],
            cwd=clone,
            check=False,
        )
    git(
        clone,
        "config",
        "remote.origin.pushurl",
        "https://example.invalid/push.git",
    )
    with pytest.raises(RuntimeError, match="git remote network operation"):
        subprocess.run(["git", "push", "origin"], cwd=clone, check=False)
    with pytest.raises(RuntimeError, match="git remote network operation"):
        subprocess.run(
            ["git", "remote", "add", "-f", "eager", "https://example.invalid/eager.git"],
            cwd=clone,
            check=False,
        )
    with pytest.raises(RuntimeError, match="git remote network operation"):
        subprocess.run(
            ["git", "remote", "set-head", "live", "-a"], cwd=clone, check=False
        )
    git(clone, "config", "url.https://example.invalid/rewrite/.insteadOf", str(origin))
    with pytest.raises(RuntimeError, match="git remote network operation"):
        subprocess.run(["git", "fetch", "origin"], cwd=clone, check=False)
    with pytest.raises(RuntimeError, match="git remote network operation"):
        subprocess.run(["git", "fetch", str(origin)], cwd=clone, check=False)
    with pytest.raises(RuntimeError, match="git remote network operation"):
        subprocess.run(
            ["bash", "-c", "echo sh; git fetch https://example.invalid/live.git"],
            cwd=clone,
            check=False,
        )


@pytest.mark.windows_only
def test_windows_git_drive_remote_stays_inside_restricted_test_root(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    sandbox = Path(os.environ["HERMES_TEST_SANDBOX_ROOT"]).resolve()
    assert tmp_path.resolve().is_relative_to(sandbox)
    origin = tmp_path / "windows-origin"
    clone = tmp_path / "windows-clone"
    origin.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=origin, check=True)
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    with pytest.raises(RuntimeError, match="git remote network operation"):
        subprocess.run(
            ["git", "ls-remote", r"\\live-host\hermes"], check=False
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
def test_generic_runner_mutates_only_disposable_repo_copy():
    repo = Path(__file__).resolve().parent.parent
    source = repo / "scripts" / "hermetic_command_entry.py"
    original = source.read_bytes()
    common_raw = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--git-common-dir"], text=True
    ).strip()
    common_dir = Path(common_raw)
    if not common_dir.is_absolute():
        common_dir = repo / common_dir
    host_hook = common_dir.resolve() / "hooks" / "post-checkout"
    original_hook = host_hook.read_bytes() if host_hook.is_file() else None
    probe_code = (
        "import os,pathlib,subprocess; "
        "root=pathlib.Path(os.environ['HERMES_TEST_REPO_ROOT']); "
        "assert (root/'.env.example').is_file(); "
        "assert (root/'.npmrc').is_file(); "
        "assert (root/'.envrc').is_file(); "
        "assert (root/'website/.npmrc').is_file(); "
        "subprocess.run(['git','-C',str(root),'cat-file','-e','HEAD'],check=True); "
        "status=subprocess.run(['git','-C',str(root),'status','--porcelain'],"
        "check=True,capture_output=True); assert status.stdout == b''; "
        "source=root/'scripts/hermetic_command_entry.py'; "
        "source.write_text('disposable mutation'); "
        "hook=root/'.git/hooks/post-checkout'; "
        "hook.parent.mkdir(parents=True,exist_ok=True); "
        "hook.write_text('#!/bin/sh\\nexit 99\\n')"
    )
    probe = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "run_hermetic_command.py"),
            "--",
            sys.executable,
            "-S",
            "-c",
            probe_code,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert source.read_bytes() == original
    assert (host_hook.read_bytes() if host_hook.is_file() else None) == original_hook


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
    parent_pid = int(os.environ["HERMES_TEST_SANDBOX_PARENT_PID"])
    assert parent_pid > 1
    assert parent_pid != os.getpid()
    assert libc.kill(parent_pid, 0) == -1
    assert ctypes.get_errno() == errno.EPERM

    # The Python guard hides service-control executables from shutil.which;
    # use the immutable system path to exercise the independent Seatbelt
    # file/exec boundary without weakening that defense-in-depth layer.
    launchctl = Path("/bin/launchctl")
    assert launchctl.is_file()
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
def test_macos_kernel_cannot_modify_or_unlink_host_canary():
    canary = Path(os.environ["HERMES_TEST_HOST_CANARY"])
    original = canary.read_bytes()
    assert original == b"HERMES_HOST_CANARY"
    with pytest.raises(OSError):
        canary.write_bytes(b"changed")
    with pytest.raises(OSError):
        canary.unlink()
    assert canary.read_bytes() == original
    with open(os.devnull, "wb") as sink:
        assert sink.write(b"safe sink") == len(b"safe sink")


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Seatbelt proof")
def test_macos_kernel_blocks_native_keychain_broker():
    import ctypes

    libc = ctypes.CDLL(None)
    bootstrap_port = ctypes.c_uint32.in_dll(libc, "bootstrap_port").value
    libc.bootstrap_look_up.argtypes = [
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    libc.bootstrap_look_up.restype = ctypes.c_int
    service_port = ctypes.c_uint32()
    broker = os.environ["HERMES_TEST_ATTESTED_MACH_BROKER"].encode("utf-8")
    status = libc.bootstrap_look_up(
        bootstrap_port,
        broker,
        ctypes.byref(service_port),
    )
    # The unsandboxed runner resolved this exact broker successfully before it
    # entered Seatbelt. Its failure here therefore attests the profile's
    # deny mach-lookup boundary, not an absent service or headless host state.
    assert status != 0
    assert service_port.value == 0


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
