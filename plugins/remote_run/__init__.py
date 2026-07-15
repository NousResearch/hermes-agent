"""Per-call SSH command execution plugin."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import threading
import time
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

SSH_CONNECT_TIMEOUT = 15
MAX_RESULT_SIZE_CHARS = 100_000
_READ_CHUNK_BYTES = 32_768
_HOST_KEY_LOCK = threading.Lock()

REMOTE_RUN_SCHEMA = {
    "name": "remote_run",
    "description": (
        "Execute one command on an explicitly selected SSH host. Returns structured "
        "stdout, stderr, and exit_code. Each call opens a new connection."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "host": {"type": "string", "description": "Remote hostname or IP address."},
            "command": {"type": "string", "description": "Command to execute."},
            "user": {"type": "string", "description": "SSH username."},
            "port": {"type": "integer", "default": 22},
            "key_file": {"type": "string", "description": "SSH private-key path."},
            "password": {
                "type": "string",
                "description": "SSH password; also supplied to sudo when sudo=true.",
            },
            "sudo": {"type": "boolean", "default": False},
            "workdir": {"type": "string", "description": "Remote working directory."},
            "env": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Remote environment variables.",
            },
            "timeout": {
                "type": "integer",
                "default": 60,
                "description": "Command timeout in seconds (1-3600).",
            },
        },
        "required": ["host", "command"],
    },
}


def check_remote_run_requirements() -> bool:
    try:
        import paramiko  # noqa: F401
        return True
    except ImportError:
        return False


def _build_env_export(env: dict[str, str] | None) -> str:
    if not env:
        return ""
    parts = []
    for key, value in env.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid environment variable key: {key!r}")
        parts.append(f"export {key}={shlex.quote(value)}")
    return "; ".join(parts) + "; "


def _build_command(
    command: str,
    workdir: str | None,
    env: dict[str, str] | None,
    sudo: bool,
) -> str:
    prefix = f"cd {shlex.quote(workdir)} && " if workdir else ""
    full_command = f"{prefix}{_build_env_export(env)}{command}"
    if sudo:
        return f"sudo -S -p '' -- bash -c {shlex.quote(full_command)}"
    return full_command


def _known_hosts_path() -> Path:
    return get_hermes_home() / "ssh" / "known_hosts"


def _configure_host_keys(client: Any, paramiko: Any, path: Path) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    client.load_system_host_keys()
    if path.exists():
        client.load_host_keys(str(path))

    class _AcceptNewAndPersist(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self, ssh_client: Any, hostname: str, key: Any) -> None:
            ssh_client.get_host_keys().add(hostname, key.get_name(), key)
            temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            ssh_client.save_host_keys(str(temp_path))
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, path)

    client.set_missing_host_key_policy(_AcceptNewAndPersist())


def _append_bounded(target: bytearray, chunk: bytes, remaining: int) -> tuple[int, bool]:
    kept = chunk[:remaining]
    target.extend(kept)
    return remaining - len(kept), len(kept) < len(chunk)


def _drain_channel(channel: Any, timeout: int) -> tuple[str, str, int, bool]:
    """Drain stdout and stderr concurrently while retaining at most 100 KB."""
    stdout = bytearray()
    stderr = bytearray()
    remaining = MAX_RESULT_SIZE_CHARS
    truncated = False
    deadline = time.monotonic() + timeout

    while True:
        progressed = False
        if channel.recv_ready():
            chunk = channel.recv(_READ_CHUNK_BYTES)
            remaining, clipped = _append_bounded(stdout, chunk, remaining)
            truncated |= clipped
            progressed = True
        if channel.recv_stderr_ready():
            chunk = channel.recv_stderr(_READ_CHUNK_BYTES)
            remaining, clipped = _append_bounded(stderr, chunk, remaining)
            truncated |= clipped
            progressed = True
        if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
            break
        if time.monotonic() >= deadline:
            channel.close()
            raise TimeoutError(f"Command timed out after {timeout} seconds")
        if not progressed:
            time.sleep(0.01)

    return (
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
        channel.recv_exit_status(),
        truncated,
    )


def _blocked_result(approval: dict[str, Any], command: str) -> str:
    return json.dumps(
        {
            "stdout": "",
            "stderr": approval.get("message") or "Command blocked by approval guard.",
            "exit_code": -1,
            "status": approval.get("status", "blocked"),
            "command": approval.get("command", command),
        }
    )


def remote_run_handler(args: dict[str, Any], **_: Any) -> str:
    import paramiko
    from tools.terminal_tool import check_command_guards

    host = str(args.get("host", "")).strip()
    command = str(args.get("command", "")).strip()
    port = args.get("port", 22)
    timeout = args.get("timeout", 60)
    if not host or not command:
        return json.dumps({"error": "host and command are required", "exit_code": 2})
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        return json.dumps({"error": "port must be between 1 and 65535", "exit_code": 2})
    if not isinstance(timeout, int) or isinstance(timeout, bool) or not 1 <= timeout <= 3600:
        return json.dumps({"error": "timeout must be between 1 and 3600 seconds", "exit_code": 2})

    approval = check_command_guards(command, "ssh")
    if not approval.get("approved", False):
        return _blocked_result(approval, command)

    client = paramiko.SSHClient()
    try:
        connect_args: dict[str, Any] = {
            "hostname": host,
            "port": port,
            "timeout": SSH_CONNECT_TIMEOUT,
            "allow_agent": True,
            "look_for_keys": True,
        }
        if args.get("user"):
            connect_args["username"] = args["user"]
        if args.get("key_file"):
            connect_args["key_filename"] = os.path.expanduser(args["key_file"])
        if args.get("password"):
            connect_args["password"] = args["password"]

        # ponytail: one process-wide lock prevents concurrent trust-on-first-use races;
        # use per-known-hosts locks only if connection setup throughput becomes material.
        with _HOST_KEY_LOCK:
            _configure_host_keys(client, paramiko, _known_hosts_path())
            client.connect(**connect_args)

        full_command = _build_command(
            command,
            args.get("workdir"),
            args.get("env"),
            bool(args.get("sudo", False)),
        )
        stdin, stdout, _stderr = client.exec_command(
            full_command,
            timeout=timeout,
            get_pty=bool(args.get("sudo", False)),
        )
        if args.get("sudo") and args.get("password"):
            stdin.write(args["password"] + "\n")
            stdin.flush()
            stdin.channel.shutdown_write()

        out_text, err_text, exit_code, truncated = _drain_channel(stdout.channel, timeout)
        if truncated:
            err_text += ("\n" if err_text else "") + "... [output truncated at 100000 bytes]"
        return json.dumps(
            {"stdout": out_text, "stderr": err_text, "exit_code": exit_code, "host": host}
        )
    except TimeoutError as exc:
        return json.dumps({"error": str(exc), "host": host, "exit_code": 124})
    except paramiko.AuthenticationException as exc:
        return json.dumps({"error": f"Authentication failed: {exc}", "host": host, "exit_code": 1})
    except paramiko.BadHostKeyException as exc:
        return json.dumps({"error": f"Host key verification failed: {exc}", "host": host, "exit_code": 1})
    except (paramiko.SSHException, OSError, ValueError) as exc:
        return json.dumps({"error": str(exc), "host": host, "exit_code": 1})
    finally:
        client.close()


def register(ctx: Any) -> None:
    ctx.register_tool(
        name="remote_run",
        toolset="ssh",
        schema=REMOTE_RUN_SCHEMA,
        handler=remote_run_handler,
        check_fn=check_remote_run_requirements,
        emoji="🔗",
        max_result_size_chars=MAX_RESULT_SIZE_CHARS,
    )
