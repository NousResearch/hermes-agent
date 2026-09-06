"""Process-wide guard used by every canonical Hermes test process.

The guard is installed by ``sitecustomize`` before pytest imports plugins or
test modules, then inherited by nested Python subprocesses.  It is deliberately
not implemented as a pytest ``monkeypatch`` fixture: tests that call
``monkeypatch.undo()`` must not be able to remove the safety boundary.

The OS sandbox installed by ``run_tests_parallel.py`` remains the outer
boundary.  This module supplies actionable failures and protects direct Python
operations inside that sandbox.
"""

from __future__ import annotations

import asyncio
import builtins
import io
import ipaddress
import os
from pathlib import Path
import shlex
import shutil
import socket
import sqlite3
import subprocess
import sys
from typing import Any


class HermeticTestViolation(RuntimeError):
    """A test attempted to cross the live-host isolation boundary."""


_INSTALLED = False
_BOUND_INET: set[tuple[str, int]] = set()


def _violation(kind: str, detail: object) -> HermeticTestViolation:
    return HermeticTestViolation(
        f"Hermes hermetic-test guard blocked {kind}: {detail!r}. "
        "Use a temp HOME/HERMES_HOME, a test-owned listener, and injected "
        "subprocess/provider doubles; no broad bypass exists."
    )


def _root_pid() -> int:
    raw = os.environ.get("HERMES_TEST_GUARD_ROOT_PID", "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            raise _violation("invalid guard root PID", raw) from None
    root = os.getpid()
    os.environ["HERMES_TEST_GUARD_ROOT_PID"] = str(root)
    return root


def _is_test_process(pid: int) -> bool:
    root = _root_pid()
    if pid in {root, os.getpid()}:
        return True
    try:
        import psutil

        return any(parent.pid == root for parent in psutil.Process(pid).parents())
    except Exception:
        return False


def _resolved(path: object) -> Path | None:
    if isinstance(path, int):
        return None
    try:
        return Path(os.fsdecode(path)).expanduser().resolve(strict=False)
    except (TypeError, ValueError, OSError):
        return None


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _sensitive_real_path(path: object) -> bool:
    candidate = _resolved(path)
    if candidate is None:
        return False
    real_home_raw = os.environ.get("HERMES_TEST_REAL_HOME", "").strip()
    if not real_home_raw:
        return False
    real_home = Path(real_home_raw).expanduser().resolve(strict=False)
    repo_raw = os.environ.get("HERMES_TEST_REPO_ROOT", "").strip()
    repo = Path(repo_raw).resolve(strict=False) if repo_raw else None
    prefix = Path(sys.prefix).resolve(strict=False)
    base_prefix = Path(sys.base_prefix).resolve(strict=False)
    if (
        _under(candidate, prefix)
        or _under(candidate, base_prefix)
        or (repo is not None and _under(candidate, repo))
    ):
        return False

    # Any path in the operator's real home is sensitive unless it is the
    # explicitly restored source tree or test interpreter. Provider clients
    # add credential locations over time; fail closed for unknown stores.
    if _under(candidate, real_home):
        return True

    sensitive_roots = (
        real_home / ".credentials",
        real_home / ".ssh",
        real_home / ".aws",
        real_home / ".gnupg",
        real_home / ".claude",
        real_home / ".codex",
        real_home / ".copilot",
        real_home / ".minimax",
        real_home / ".config" / "github-copilot",
        real_home / ".config" / "gh",
        real_home / ".config" / "gcloud",
        real_home / ".config" / "op",
        real_home / ".config" / "openai",
        real_home / ".config" / "anthropic",
        real_home / ".config" / "huggingface",
        real_home / ".config" / "Bitwarden CLI",
        real_home / ".cache" / "huggingface",
        real_home / ".azure",
        real_home / ".kube",
        real_home / ".docker",
        real_home / ".local" / "share" / "keyrings",
        real_home / "Library" / "Keychains",
    )
    if any(_under(candidate, root) for root in sensitive_roots):
        return True
    sensitive_files = (
        real_home / ".netrc",
        real_home / ".gitconfig",
        real_home / ".git-credentials",
        real_home / ".npmrc",
        real_home / ".pypirc",
        real_home / ".anthropic_oauth.json",
        real_home / ".cargo" / "credentials",
        real_home / ".cargo" / "credentials.toml",
    )
    if candidate in sensitive_files:
        return True

    hermes_root = real_home / ".hermes"
    if not _under(candidate, hermes_root):
        return (
            candidate
            == real_home / "Library" / "LaunchAgents" / "ai.hermes.gateway.plist"
        )

    return True


def _install_file_guards() -> None:
    real_open = builtins.open
    real_io_open = io.open
    real_os_open = os.open
    real_sqlite_connect = sqlite3.connect

    def guarded_open(file, *args, **kwargs):
        if _sensitive_real_path(file):
            raise _violation("real credential/Hermes file access", file)
        return real_open(file, *args, **kwargs)

    def guarded_io_open(file, *args, **kwargs):
        if _sensitive_real_path(file):
            raise _violation("real credential/Hermes file access", file)
        return real_io_open(file, *args, **kwargs)

    def guarded_os_open(path, flags, *args, **kwargs):
        if _sensitive_real_path(path):
            raise _violation("real credential/Hermes file access", path)
        return real_os_open(path, flags, *args, **kwargs)

    def guarded_sqlite_connect(database, *args, **kwargs):
        if _sensitive_real_path(database):
            raise _violation("real Hermes database access", database)
        return real_sqlite_connect(database, *args, **kwargs)

    builtins.open = guarded_open
    io.open = guarded_io_open
    os.open = guarded_os_open
    sqlite3.connect = guarded_sqlite_connect

    for name in (
        "remove",
        "unlink",
        "rmdir",
        "mkdir",
        "chmod",
        "chown",
        "lchown",
        "truncate",
    ):
        real = getattr(os, name, None)
        if real is None:
            continue

        def guarded(path, *args, __real=real, __name=name, **kwargs):
            if _sensitive_real_path(path):
                raise _violation(f"real credential/Hermes os.{__name}", path)
            return __real(path, *args, **kwargs)

        setattr(os, name, guarded)

    for name in ("rename", "replace", "link", "symlink"):
        real = getattr(os, name, None)
        if real is None:
            continue

        def guarded(src, dst, *args, __real=real, __name=name, **kwargs):
            if _sensitive_real_path(src) or _sensitive_real_path(dst):
                raise _violation(f"real credential/Hermes os.{__name}", (src, dst))
            return __real(src, dst, *args, **kwargs)

        setattr(os, name, guarded)

    real_rmtree = shutil.rmtree
    real_move = shutil.move

    def guarded_rmtree(path, *args, **kwargs):
        if _sensitive_real_path(path):
            raise _violation("real credential/Hermes shutil.rmtree", path)
        return real_rmtree(path, *args, **kwargs)

    def guarded_move(src, dst, *args, **kwargs):
        if _sensitive_real_path(src) or _sensitive_real_path(dst):
            raise _violation("real credential/Hermes shutil.move", (src, dst))
        return real_move(src, dst, *args, **kwargs)

    shutil.rmtree = guarded_rmtree
    shutil.move = guarded_move


def _loopback_host(host: object) -> bool:
    if isinstance(host, bytes):
        host = host.decode(errors="replace")
    if not isinstance(host, str):
        return False
    normalized = host.strip().strip("[]").lower()
    if normalized in {"localhost", "ip6-localhost"}:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _test_owned_listener(host: str, port: int, sock_type: int) -> bool:
    normalized = host.strip().strip("[]").lower()
    if (normalized, port) in _BOUND_INET:
        return True
    if normalized == "localhost":
        aliases = {"127.0.0.1", "::1", "localhost"}
        if any((alias, port) in _BOUND_INET for alias in aliases):
            return True
    try:
        import psutil

        kind = "udp" if sock_type == socket.SOCK_DGRAM else "tcp"
        for conn in psutil.net_connections(kind=kind):
            if not conn.laddr or int(conn.laddr.port) != port:
                continue
            address = str(conn.laddr.ip).strip("[]").lower()
            if address not in {"0.0.0.0", "::"} and not _loopback_host(address):
                continue
            if conn.pid is not None and _is_test_process(int(conn.pid)):
                return True
    except Exception:
        pass
    return False


def _check_network(sock: socket.socket, address: object, operation: str) -> None:
    if sock.family == socket.AF_UNIX:
        if not isinstance(address, (str, bytes, os.PathLike)):
            raise _violation(f"{operation} Unix socket", address)
        candidate = _resolved(address)
        allowed_roots = [
            Path(value).resolve(strict=False)
            for key in ("HOME", "HERMES_HOME", "TMPDIR", "TEMP", "TMP")
            if (value := os.environ.get(key, "")).strip()
        ]
        if candidate is not None and any(
            _under(candidate, root) for root in allowed_roots
        ):
            return
        raise _violation(f"{operation} non-test Unix socket", address)

    if not isinstance(address, tuple) or len(address) < 2:
        raise _violation(f"{operation} unknown network address", address)
    host, port = address[0], address[1]
    if not _loopback_host(host):
        raise _violation(f"{operation} external network", address)
    try:
        numeric_port = int(port)
    except (TypeError, ValueError):
        raise _violation(f"{operation} invalid port", address) from None
    if not _test_owned_listener(str(host), numeric_port, sock.type & 0xF):
        raise _violation(f"{operation} non-test-owned loopback service", address)


def _install_network_guards() -> None:
    real_bind = socket.socket.bind
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_sendto = socket.socket.sendto
    real_getaddrinfo = socket.getaddrinfo

    def guarded_bind(sock, address):
        if sock.family in {socket.AF_INET, socket.AF_INET6}:
            if not isinstance(address, tuple) or not _loopback_host(address[0]):
                raise _violation("non-loopback network bind", address)
        result = real_bind(sock, address)
        if sock.family in {socket.AF_INET, socket.AF_INET6}:
            bound = sock.getsockname()
            _BOUND_INET.add((str(bound[0]).strip("[]").lower(), int(bound[1])))
        return result

    def guarded_connect(sock, address):
        _check_network(sock, address, "socket.connect")
        return real_connect(sock, address)

    def guarded_connect_ex(sock, address):
        _check_network(sock, address, "socket.connect_ex")
        return real_connect_ex(sock, address)

    def guarded_sendto(sock, data, *args):
        address = args[-1] if args else None
        _check_network(sock, address, "socket.sendto")
        return real_sendto(sock, data, *args)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if host is not None and not _loopback_host(host):
            raise _violation("external DNS resolution", host)
        return real_getaddrinfo(host, *args, **kwargs)

    socket.socket.bind = guarded_bind
    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.socket.sendto = guarded_sendto
    socket.getaddrinfo = guarded_getaddrinfo


_SERVICE_COMMANDS = {"launchctl", "systemctl", "service", "sc", "sc.exe"}
_PROCESS_COMMANDS = {
    "kill",
    "pkill",
    "killall",
    "taskkill",
    "taskkill.exe",
    "skill",
    "fuser",
}
_CREDENTIAL_COMMANDS = {
    "security",
    "secret-tool",
    "keyring",
    "op",
    "bw",
    "bws",
    "pass",
    "cmdkey",
    "cmdkey.exe",
    "vaultcmd",
    "vaultcmd.exe",
}
_NETWORK_COMMANDS = {
    "curl",
    "wget",
    "http",
    "https",
    "ssh",
    "scp",
    "sftp",
    "telnet",
    "nc",
    "ncat",
    "openai",
    "claude",
    "codex",
    "ollama",
    "netsh",
    "podman",
}
_WRAPPERS = {
    "sh",
    "bash",
    "zsh",
    "dash",
    "env",
    "nohup",
    "setsid",
    "timeout",
    "sudo",
    "xargs",
}


def _tokens(command: object) -> list[str]:
    if isinstance(command, (list, tuple)):
        return [os.fsdecode(token) for token in command]
    if isinstance(command, bytes):
        command = command.decode(errors="replace")
    if not isinstance(command, str):
        command = str(command)
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _basename(token: str) -> str:
    return token.replace("\\", "/").rsplit("/", 1)[-1].lower()


def _process_command_targets_test_tree(tokens: list[str], direct: str) -> bool:
    """Allow only PID-addressed control of a process the test owns."""
    raw_pids: list[str] = []
    if direct == "kill":
        raw_pids = [token for token in tokens[1:] if not token.startswith("-")]
    elif direct in {"taskkill", "taskkill.exe"}:
        for index, token in enumerate(tokens[:-1]):
            if token.lower() == "/pid":
                raw_pids.append(tokens[index + 1])
    else:
        return False
    if not raw_pids:
        return False
    try:
        pids = [int(value) for value in raw_pids]
    except ValueError:
        return False
    return all(pid > 0 and _is_test_process(pid) for pid in pids)


def _check_command(command: object, operation: str) -> None:
    tokens = _tokens(command)
    if not tokens:
        return
    bases = [_basename(token) for token in tokens]
    direct = bases[0]
    if direct == "docker":
        shim_raw = os.environ.get("HERMES_TEST_DOCKER_SHIM", "").strip()
        docker_host = os.environ.get("DOCKER_HOST", "")
        sandbox_raw = os.environ.get("HERMES_TEST_SANDBOX_ROOT", "").strip()
        command_path = (
            shutil.which(tokens[0])
            if "/" not in tokens[0] and "\\" not in tokens[0]
            else tokens[0]
        )
        executable = _resolved(command_path) if command_path else None
        shim = _resolved(shim_raw) if shim_raw else None
        socket_path = _resolved(docker_host.removeprefix("unix://"))
        sandbox = _resolved(sandbox_raw) if sandbox_raw else None
        shared_raw = os.environ.get("HERMES_TEST_DIND_SHARED_ROOT", "").strip()
        shared_root = _resolved(shared_raw) if shared_raw else None
        if not (
            os.environ.get("HERMES_TEST_OS_SANDBOX") == "linux-bwrap"
            and os.environ.get("HERMES_TEST_EPHEMERAL_DOCKER") == "1"
            and docker_host.startswith("unix://")
            and executable is not None
            and shim is not None
            and executable == shim
            and socket_path is not None
            and sandbox is not None
            and shared_root is not None
            and _under(socket_path, shared_root)
            and sandbox.parent == shared_root / "pytest-roots"
        ):
            raise _violation(f"{operation} non-hermetic Docker daemon", command)
    scan = set(bases if direct in _WRAPPERS else bases[:1])
    if scan & _SERVICE_COMMANDS:
        raise _violation(f"{operation} host service control", command)
    if scan & _PROCESS_COMMANDS and not _process_command_targets_test_tree(
        tokens, direct
    ):
        raise _violation(f"{operation} process control", command)
    if scan & _CREDENTIAL_COMMANDS:
        raise _violation(f"{operation} credential store", command)
    if scan & _NETWORK_COMMANDS:
        raise _violation(f"{operation} provider/network executable", command)

    low = " ".join(tokens).lower()
    if direct in {"powershell", "powershell.exe", "pwsh", "cmd", "cmd.exe"} and any(
        phrase in low
        for phrase in (
            "stop-process",
            "restart-service",
            "stop-service",
            "start-service",
            "invoke-webrequest",
            "invoke-restmethod",
            "taskkill",
            " sc ",
            " net stop ",
            "new-netfirewallrule",
            "set-netfirewallprofile",
            "set-netfirewallrule",
            "remove-netfirewallrule",
        )
    ):
        raise _violation(f"{operation} PowerShell/cmd live operation", command)
    if direct == "git" and any(
        verb in tokens[1:] for verb in ("clone", "fetch", "pull", "push", "ls-remote")
    ):
        raise _violation(f"{operation} git remote network operation", command)
    if direct in {"pip", "pip3"} and any(
        verb in tokens[1:] for verb in ("install", "download", "wheel")
    ):
        raise _violation(f"{operation} package network operation", command)
    if direct == "uv" and any(
        verb in tokens[1:] for verb in ("install", "pip", "tool", "sync")
    ):
        raise _violation(f"{operation} package network operation", command)
    if direct in {"npm", "pnpm", "yarn", "bun"} and any(
        verb in tokens[1:] for verb in ("install", "add", "update", "upgrade", "dlx")
    ):
        raise _violation(f"{operation} package network operation", command)


def _install_process_guards() -> None:
    real_kill = os.kill

    def guarded_kill(pid, sig, *args, **kwargs):
        numeric_pid = int(pid)
        numeric_sig = int(sig)
        if numeric_sig == 0:
            return real_kill(pid, sig, *args, **kwargs)
        if numeric_pid > 0 and _is_test_process(numeric_pid):
            return real_kill(pid, sig, *args, **kwargs)
        if numeric_pid == 0 and os.getpgrp() == _root_pid():
            return real_kill(pid, sig, *args, **kwargs)
        if numeric_pid < 0 and abs(numeric_pid) == os.getpgrp() == _root_pid():
            return real_kill(pid, sig, *args, **kwargs)
        raise _violation("os.kill outside test process tree", (pid, sig))

    os.kill = guarded_kill
    if hasattr(os, "killpg"):
        real_killpg = os.killpg

        def guarded_killpg(pgid, sig, *args, **kwargs):
            numeric_pgid = abs(int(pgid))
            if (
                (
                    os.environ.get("HERMES_TEST_OS_SANDBOX") == "linux-bwrap"
                    and numeric_pgid > 1
                )
                # A private PID namespace makes every visible process
                # test-owned at the kernel boundary. This also lets nested
                # runners reap an already-exited group leader safely. PID 1
                # remains reserved so the deliberate foreign-PGID regression
                # proves the Python guard also fails closed.
                or int(sig) == 0
                or int(pgid) == os.getpgrp() == _root_pid()
                or _is_test_process(numeric_pgid)
            ):
                return real_killpg(pgid, sig, *args, **kwargs)
            raise _violation("os.killpg outside test process group", (pgid, sig))

        os.killpg = guarded_killpg

    real_popen = subprocess.Popen

    class GuardedPopen(real_popen):  # type: ignore[misc, valid-type]
        def __init__(self, args, *pargs, **kwargs):
            _check_command(args, "subprocess.Popen")
            super().__init__(args, *pargs, **kwargs)

    GuardedPopen.__name__ = "Popen"
    GuardedPopen.__qualname__ = "Popen"
    subprocess.Popen = GuardedPopen

    for name in (
        "run",
        "call",
        "check_call",
        "check_output",
        "getoutput",
        "getstatusoutput",
    ):
        real = getattr(subprocess, name)

        def guarded(command, *args, __real=real, __name=name, **kwargs):
            _check_command(command, f"subprocess.{__name}")
            return __real(command, *args, **kwargs)

        setattr(subprocess, name, guarded)

    real_system = os.system
    real_popen_fn = os.popen

    def guarded_system(command):
        _check_command(command, "os.system")
        return real_system(command)

    def guarded_popen_fn(command, *args, **kwargs):
        _check_command(command, "os.popen")
        return real_popen_fn(command, *args, **kwargs)

    os.system = guarded_system
    os.popen = guarded_popen_fn

    try:
        import pty

        real_pty_spawn = pty.spawn

        def guarded_pty_spawn(argv, *args, **kwargs):
            _check_command(argv, "pty.spawn")
            return real_pty_spawn(argv, *args, **kwargs)

        pty.spawn = guarded_pty_spawn
    except (ImportError, AttributeError):
        pass

    real_async_exec = asyncio.create_subprocess_exec
    real_async_shell = asyncio.create_subprocess_shell

    async def guarded_async_exec(program, *args, **kwargs):
        _check_command([program, *args], "asyncio.create_subprocess_exec")
        return await real_async_exec(program, *args, **kwargs)

    async def guarded_async_shell(command, *args, **kwargs):
        _check_command(command, "asyncio.create_subprocess_shell")
        return await real_async_shell(command, *args, **kwargs)

    asyncio.create_subprocess_exec = guarded_async_exec
    asyncio.create_subprocess_shell = guarded_async_shell

    try:
        import psutil

        for name in ("kill", "terminate", "send_signal", "suspend", "resume"):
            real = getattr(psutil.Process, name)

            def guarded_process(proc, *args, __real=real, __name=name, **kwargs):
                signal_value = args[0] if __name == "send_signal" and args else None
                if signal_value == 0 or _is_test_process(int(proc.pid)):
                    return __real(proc, *args, **kwargs)
                raise _violation(f"psutil.Process.{__name} outside test tree", proc.pid)

            setattr(psutil.Process, name, guarded_process)
    except (ImportError, AttributeError):
        pass


def install() -> None:
    """Install the guard once in this interpreter."""
    global _INSTALLED
    if _INSTALLED:
        return
    if os.environ.get("HERMES_TEST_GUARD_ACTIVE") != "1":
        return
    _root_pid()
    _install_file_guards()
    _install_network_guards()
    _install_process_guards()
    _INSTALLED = True


def installed() -> bool:
    return _INSTALLED
