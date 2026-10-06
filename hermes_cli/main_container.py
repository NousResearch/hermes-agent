"""NixOS container mode: route a ``hermes`` invocation into the managed container (``main()``)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys


def _probe_container(cmd: list, backend: str, via_sudo: bool = False):
    """Run a container inspect probe, returning the CompletedProcess.

    Catches TimeoutExpired specifically for a human-readable message;
    all other exceptions propagate naturally.
    """
    try:
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
    except subprocess.TimeoutExpired:
        label = f"sudo {backend}" if via_sudo else backend
        print(
            f"Error: timed out waiting for {label} to respond.\n"
            f"The {backend} daemon may be unresponsive or starting up.",
            file=sys.stderr,
        )
        sys.exit(1)


def exec_in_container(container_info: dict, cli_args: list):
    """Replace the current process with a command inside the managed container.

    Probes whether sudo is needed (rootful containers), then os.execvp
    into the container. On success the Python process is replaced entirely
    and the container's exit code becomes the process exit code (OS semantics).
    On failure, OSError propagates naturally.

    Args:
        container_info: dict with backend, container_name, exec_user, hermes_bin
        cli_args: the original CLI arguments (everything after 'hermes')
    """

    backend = container_info["backend"]
    container_name = container_info["container_name"]
    exec_user = container_info["exec_user"]
    hermes_bin = container_info["hermes_bin"]

    runtime = shutil.which(backend)
    if not runtime:
        print(
            f"Error: {backend} not found on PATH. Cannot route to container.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Rootful containers (NixOS systemd service) are invisible to unprivileged
    # users — Podman uses per-user namespaces, Docker needs group access.
    # Probe whether the runtime can see the container; if not, try via sudo.
    inspect_cmd = [runtime, "inspect", "--format", "ok", container_name]
    cmd_prefix = [runtime]
    if _probe_container(inspect_cmd, backend).returncode != 0:
        sudo_path = shutil.which("sudo")
        if not sudo_path:
            print(
                f"Error: container '{container_name}' not found via {backend}.\n"
                f"The container may be running under root. Try: sudo hermes {' '.join(cli_args)}",
                file=sys.stderr,
            )
            sys.exit(1)
        cmd_prefix = [sudo_path, "-n", runtime]
        if _probe_container(cmd_prefix[:2] + inspect_cmd, backend, via_sudo=True).returncode != 0:
            print(
                f"Error: container '{container_name}' not found via {backend}.\n"
                f"\n"
                f"The container is likely running as root. Your user cannot see it\n"
                f"because {backend} uses per-user namespaces. Grant passwordless\n"
                f"sudo for {backend} — the -n (non-interactive) flag is required\n"
                f"because a password prompt would hang or break piped commands.\n"
                f"\n"
                f"On NixOS:\n"
                f"\n"
                f"  security.sudo.extraRules = [{{\n"
                f'    users = [ "{os.getenv("USER", "your-user")}" ];\n'
                f'    commands = [{{ command = "{runtime}"; options = [ "NOPASSWD" ]; }}];\n'
                f"  }}];\n"
                f"\n"
                f"Or run: sudo hermes {' '.join(cli_args)}",
                file=sys.stderr,
            )
            sys.exit(1)

    env_flags = []
    for var in ("TERM", "COLORTERM", "LANG", "LC_ALL"):
        val = os.environ.get(var)
        if val:
            env_flags.extend(["-e", f"{var}={val}"])

    exec_cmd = (
        cmd_prefix
        + ["exec", "-it" if sys.stdin.isatty() else "-i", "-u", exec_user]
        + env_flags
        + [container_name, hermes_bin]
        + cli_args
    )
    os.execvp(exec_cmd[0], exec_cmd)
