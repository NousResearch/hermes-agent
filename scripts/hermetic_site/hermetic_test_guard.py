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
from collections.abc import Mapping
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
from urllib.parse import unquote, urlparse


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


def _git_subcommand(tokens: list[str]) -> tuple[str | None, list[str]]:
    """Return Git's actual subcommand and its remaining arguments.

    Git accepts global options before the subcommand.  Looking for words such
    as ``push`` anywhere in argv incorrectly classifies the entirely local
    ``git stash push`` operation as a network push.
    """
    options_with_values = {
        "-C",
        "-c",
        "--config-env",
        "--exec-path",
        "--git-dir",
        "--namespace",
        "--super-prefix",
        "--work-tree",
    }
    index = 1
    while index < len(tokens):
        token = tokens[index]
        option = token.split("=", 1)[0]
        if option in options_with_values:
            index += 1 if "=" in token else 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return token.lower(), tokens[index + 1 :]
    return None, []


def _first_git_positional(args: list[str], *, subcommand: str) -> str | None:
    """Find the repository operand for a Git network-capable subcommand."""
    value_options = {
        "clone": {
            "-b", "--branch", "-j", "--jobs", "-o", "--origin",
            "-u", "--upload-pack", "--depth", "--filter", "--reference",
            "--reference-if-able", "--separate-git-dir", "--server-option",
            "--shallow-exclude", "--shallow-since",
        },
        "fetch": {
            "-j", "--jobs", "--depth", "--deepen", "--filter", "--negotiation-tip",
            "--server-option", "--shallow-exclude", "--shallow-since", "--upload-pack",
        },
        "pull": {
            "-j", "--jobs", "--depth", "--deepen", "--filter", "--server-option",
            "--shallow-exclude", "--shallow-since", "--upload-pack",
        },
        "push": {
            "--exec", "--push-option", "--receive-pack", "--repo",
        },
        "ls-remote": {"--server-option", "--sort", "--upload-pack"},
    }.get(subcommand, set())
    index = 0
    while index < len(args):
        token = args[index]
        option = token.split("=", 1)[0]
        if option in value_options:
            index += 1 if "=" in token else 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return token
    return None


def _git_working_directory(tokens: list[str], cwd: object | None) -> Path:
    base = _resolved(cwd) if cwd is not None else Path.cwd().resolve(strict=False)
    assert base is not None
    for index, token in enumerate(tokens[1:-1], start=1):
        if token == "-C":
            candidate = Path(tokens[index + 1]).expanduser()
            base = (
                candidate.resolve(strict=False)
                if candidate.is_absolute()
                else (base / candidate).resolve(strict=False)
            )
        elif token.startswith("-C") and len(token) > 2:
            candidate = Path(token[2:]).expanduser()
            base = (
                candidate.resolve(strict=False)
                if candidate.is_absolute()
                else (base / candidate).resolve(strict=False)
            )
    return base


def _git_repository_config_paths(cwd: Path) -> list[Path]:
    current = cwd
    while True:
        marker = current / ".git"
        if marker.is_dir():
            return [marker / "config", marker / "config.worktree"]
        if marker.is_file():
            try:
                declaration = marker.read_text(encoding="utf-8").strip()
            except OSError:
                return []
            if not declaration.lower().startswith("gitdir:"):
                return []
            gitdir = Path(declaration.split(":", 1)[1].strip()).expanduser()
            if not gitdir.is_absolute():
                gitdir = (current / gitdir).resolve(strict=False)
            common_gitdir = gitdir
            common = gitdir / "commondir"
            if common.is_file():
                try:
                    common_dir = Path(common.read_text(encoding="utf-8").strip())
                except OSError:
                    return []
                common_gitdir = (
                    common_dir.resolve(strict=False)
                    if common_dir.is_absolute()
                    else (gitdir / common_dir).resolve(strict=False)
                )
            return [common_gitdir / "config", gitdir / "config.worktree"]
        if current.parent == current:
            return []
        current = current.parent


def _git_repository_config(cwd: Path) -> list[str] | None:
    lines: list[str] = []
    for config_path in _git_repository_config_paths(cwd):
        if not config_path.exists():
            continue
        try:
            lines.extend(config_path.read_text(encoding="utf-8").splitlines())
        except OSError:
            return None
    return lines


def _configured_git_remote(
    lines: list[str], name: str, *, for_push: bool = False
) -> list[str] | None:
    """Resolve a simple remote URL from already-audited repository config."""
    wanted = f'remote "{name}"'.lower()
    in_remote = False
    urls: list[str] = []
    push_urls: list[str] = []
    for raw in lines:
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            if section == "include" or section.startswith(
                ("includeif ", "url ", "http ", "credential ")
            ):
                return None
            in_remote = section == wanted
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            normalized_key = key.strip().lower()
            if normalized_key in {
                "gitproxy",
                "insteadof",
                "proxy",
                "pushinsteadof",
                "receivepack",
                "sshcommand",
                "uploadpack",
                "vcs",
            }:
                return None
            if not in_remote:
                continue
            if normalized_key == "url":
                urls.append(value.strip())
            elif normalized_key == "pushurl":
                push_urls.append(value.strip())
    return push_urls if for_push and push_urls else urls


def _local_git_remote(remote: str, cwd: Path) -> bool:
    """Permit only filesystem remotes contained by the disposable sandbox."""
    windows_drive = (
        len(remote) >= 3
        and remote[0].isalpha()
        and remote[1] == ":"
        and remote[2] in {"/", "\\"}
    )
    if remote.startswith(("\\\\", "//")):
        return False
    parsed = urlparse(remote) if not windows_drive else None
    if parsed is not None and parsed.scheme:
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            return False
        raw_path = unquote(parsed.path)
    else:
        # Git's scp-like syntax is a network destination, not a local path.
        if ":" in remote and not remote.startswith(("./", "../", "/")):
            return False
        raw_path = remote
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = cwd / candidate
    candidate = candidate.resolve(strict=False)
    sandbox_raw = os.environ.get("HERMES_TEST_SANDBOX_ROOT", "").strip()
    sandbox = _resolved(sandbox_raw) if sandbox_raw else None
    return sandbox is not None and candidate.exists() and _under(candidate, sandbox)


def _git_operand_is_literal(remote: str) -> bool:
    return (
        remote.startswith((".", "/", "\\", "~"))
        or "/" in remote
        or "\\" in remote
        or ":" in remote
    )


_LOCAL_GIT_SUBCOMMANDS = {
    "add", "am", "apply", "bisect", "blame", "branch", "bundle",
    "cat-file", "checkout", "cherry", "cherry-pick", "clean", "commit",
    "config", "describe", "diff", "diff-tree", "for-each-ref",
    "format-patch", "fsck", "gc", "grep", "hash-object", "init", "log",
    "ls-files", "ls-tree", "merge", "merge-base", "mv", "name-rev",
    "notes", "prune", "read-tree", "reflog", "reset", "restore",
    "rev-list", "rev-parse", "rm", "show", "show-ref", "sparse-checkout",
    "stash", "status", "switch", "symbolic-ref", "tag", "update-index",
    "update-ref", "verify-commit", "verify-tag", "worktree", "write-tree",
}


def _git_remote_overrides_present(
    tokens: list[str], child_env: object | None
) -> bool:
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "-c" and index + 1 < len(tokens):
            value = tokens[index + 1]
            index += 2
            key = value.split("=", 1)[0].strip().lower()
            if key.startswith(
                ("alias.", "credential.", "http.", "remote.", "url.")
            ) or key in {"core.gitproxy", "core.sshcommand"}:
                return True
            continue
        if token.startswith("-c") and len(token) > 2:
            key = token[2:].split("=", 1)[0].strip().lower()
            if key.startswith(
                ("alias.", "credential.", "http.", "remote.", "url.")
            ) or key in {"core.gitproxy", "core.sshcommand"}:
                return True
        if token == "--config-env" or token.startswith("--config-env="):
            return True
        index += 1
    effective_env = child_env if isinstance(child_env, Mapping) else os.environ
    normalized_env = {str(key).upper(): value for key, value in effective_env.items()}
    if normalized_env.get("GIT_CONFIG_NOSYSTEM") != "1" or str(
        normalized_env.get("GIT_CONFIG_GLOBAL", "")
    ).lower() != os.devnull.lower():
        return True
    repository_selectors = {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_DIR",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        "GIT_OBJECT_DIRECTORY",
        "GIT_WORK_TREE",
    }
    if any(name in normalized_env for name in repository_selectors):
        return True
    for name, value in normalized_env.items():
        if not name.startswith("GIT_CONFIG_"):
            continue
        if name == "GIT_CONFIG_NOSYSTEM" and str(value) == "1":
            continue
        if name == "GIT_CONFIG_GLOBAL" and str(value).lower() == os.devnull.lower():
            continue
        return True
    return False


def _git_remote_is_test_local(
    tokens: list[str], cwd: object | None, child_env: object | None
) -> bool:
    subcommand, args = _git_subcommand(tokens)
    if subcommand in _LOCAL_GIT_SUBCOMMANDS:
        return True
    if subcommand == "remote":
        action = next((arg for arg in args if not arg.startswith("-")), None)
        if action == "add" and any(arg in {"-f", "--fetch"} for arg in args):
            return False
        if action == "set-head" and any(arg in {"-a", "--auto"} for arg in args):
            return False
        return action in {
            None,
            "add",
            "get-url",
            "remove",
            "rename",
            "set-head",
            "set-url",
        }
    if subcommand == "archive":
        return not any(arg == "--remote" or arg.startswith("--remote=") for arg in args)
    if subcommand not in {"clone", "fetch", "pull", "push", "ls-remote"}:
        # Unknown subcommands may be aliases that launch arbitrary helpers.
        return False
    # Config and repository-path overrides can redirect an apparently local
    # remote after this guard has resolved it. Fail closed instead of trying
    # to duplicate Git's full configuration precedence language.
    if _git_remote_overrides_present(tokens, child_env) or any(
        token == "--git-dir" or token.startswith("--git-dir=")
        for token in tokens[1:]
    ):
        return False
    if subcommand == "fetch" and any(
        token in {"--all", "--multiple"} for token in args
    ):
        return False
    if subcommand == "push" and any(
        token == "--repo" or token.startswith("--repo=") for token in args
    ):
        return False
    transport_helper_options = {
        "-u",
        "--exec",
        "--receive-pack",
        "--upload-pack",
    }
    if any(
        token.split("=", 1)[0] in transport_helper_options for token in args
    ):
        return False
    working_directory = _git_working_directory(tokens, cwd)
    config_lines = _git_repository_config(working_directory)
    if config_lines is None:
        return False
    if _configured_git_remote(config_lines, "__hermetic_audit__") is None:
        return False
    operand = _first_git_positional(args, subcommand=subcommand)
    if subcommand in {"fetch", "pull", "push"}:
        operand = operand or "origin"
        if _git_operand_is_literal(operand):
            return _local_git_remote(operand, working_directory)
        configured = _configured_git_remote(
            config_lines, operand, for_push=subcommand == "push"
        )
        if configured:
            return all(
                _local_git_remote(remote, working_directory)
                for remote in configured
            )
        # A bare unresolved name could be supplied by an includeIf/global
        # config the intentionally small parser did not load.
        return False
    return operand is not None and _local_git_remote(operand, working_directory)


def _shell_payload_tokens(payload: str) -> list[str]:
    try:
        lexer = shlex.shlex(payload, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        return list(lexer)
    except ValueError:
        return payload.split()


def _check_wrapped_commands(
    tokens: list[str],
    *,
    operation: str,
    cwd: object | None,
    child_env: object | None,
) -> None:
    direct = _basename(tokens[0])
    if direct == "env":
        inherited = dict(child_env if isinstance(child_env, Mapping) else os.environ)
        index = 1
        while index < len(tokens):
            token = tokens[index]
            if token in {"-i", "--ignore-environment"}:
                inherited.clear()
                index += 1
                continue
            if token in {"-S", "--split-string"} or token.startswith(
                "--split-string="
            ):
                raise _violation(f"{operation} env split-string wrapper", tokens)
            if token in {"-u", "--unset"}:
                if index + 1 < len(tokens):
                    unwanted = tokens[index + 1].upper()
                    inherited = {
                        key: value
                        for key, value in inherited.items()
                        if str(key).upper() != unwanted
                    }
                index += 2
                continue
            if token.startswith("--unset="):
                unwanted = token.split("=", 1)[1].upper()
                inherited = {
                    key: value
                    for key, value in inherited.items()
                    if str(key).upper() != unwanted
                }
                index += 1
                continue
            if token.startswith("-"):
                index += 1
                continue
            if "=" in token and not token.startswith("="):
                key, value = token.split("=", 1)
                inherited[key] = value
                index += 1
                continue
            break
        if index < len(tokens):
            _check_command(
                tokens[index:],
                operation,
                cwd=cwd,
                child_env=inherited,
            )
        return

    inspected = tokens
    if direct in {"sh", "bash", "zsh", "dash"}:
        payload: str | None = None
        for index, token in enumerate(tokens[1:-1], start=1):
            if token == "-c" or (
                token.startswith("-")
                and not token.startswith("--")
                and "c" in token[1:]
            ):
                payload = tokens[index + 1]
                break
        if payload is None:
            return
        if any(marker in payload for marker in ("$", "`", "<(" , ">(")):
            raise _violation(f"{operation} dynamic shell wrapper", tokens)
        inspected = _shell_payload_tokens(payload)

    bases = [_basename(token) for token in inspected]
    forbidden = _SERVICE_COMMANDS | _CREDENTIAL_COMMANDS | _NETWORK_COMMANDS
    match = next((base for base in bases if base in forbidden), None)
    if match is not None:
        raise _violation(f"{operation} wrapped forbidden executable", tokens)
    # Shell/process-wrapper indirection obscures process identity and PID
    # parsing. Direct kill/taskkill remains permitted only for the test tree.
    if any(base in _PROCESS_COMMANDS for base in bases):
        raise _violation(f"{operation} wrapped process control", tokens)
    for index, base in enumerate(bases):
        if direct == "xargs" and base in _WRAPPERS | {"git"}:
            raise _violation(f"{operation} data-driven wrapped command", tokens)
        if base == "git" or (base in _WRAPPERS and index > 0):
            _check_command(
                inspected[index:],
                operation,
                cwd=cwd,
                child_env=child_env,
            )
            return


def _check_command(
    command: object,
    operation: str,
    *,
    cwd: object | None = None,
    child_env: object | None = None,
) -> None:
    tokens = _tokens(command)
    if not tokens:
        return
    bases = [_basename(token) for token in tokens]
    direct = bases[0]
    if direct in _WRAPPERS:
        _check_wrapped_commands(
            tokens,
            operation=operation,
            cwd=cwd,
            child_env=child_env,
        )
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
    if direct == "git" and not _git_remote_is_test_local(tokens, cwd, child_env):
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
            _check_command(
                args,
                "subprocess.Popen",
                cwd=kwargs.get("cwd"),
                child_env=kwargs.get("env"),
            )
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
            _check_command(
                command,
                f"subprocess.{__name}",
                cwd=kwargs.get("cwd"),
                child_env=kwargs.get("env"),
            )
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
        _check_command(
            [program, *args],
            "asyncio.create_subprocess_exec",
            cwd=kwargs.get("cwd"),
            child_env=kwargs.get("env"),
        )
        return await real_async_exec(program, *args, **kwargs)

    async def guarded_async_shell(command, *args, **kwargs):
        _check_command(
            command,
            "asyncio.create_subprocess_shell",
            cwd=kwargs.get("cwd"),
            child_env=kwargs.get("env"),
        )
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
