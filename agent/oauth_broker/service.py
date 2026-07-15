"""launchd rendering and lifecycle helpers for the OAuth broker.

Rendering uses plistlib and atomic 0600 writes. launchctl interactions are
explicit argv lists targeting the ``gui/<uid>`` domain; nothing here shells
out on its own — `install`/`uninstall` default to render-only, and an
``apply=True`` call demands an explicit runner so a live launchctl action is
always a deliberate CLI-confirmed step, never a side effect.
"""

from __future__ import annotations

import os
import plistlib
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, TypedDict

from hermes_constants import get_hermes_home

from agent.oauth_broker.server import DEFAULT_BROKER_PORT, validate_bind_host

BROKER_LAUNCHD_LABEL = "ai.hermes.oauth-broker"


class InstallResult(TypedDict):
    plist_path: Path
    bootstrap: List[str]
    kickstart: List[str]
    executed: bool


class UninstallResult(TypedDict):
    plist_path: Path
    bootout: List[str]
    executed: bool


def broker_launchd_plist_path(
    *, launch_agents_dir: Optional[Path] = None
) -> Path:
    base = launch_agents_dir or (Path.home() / "Library" / "LaunchAgents")
    return base / f"{BROKER_LAUNCHD_LABEL}.plist"


def render_broker_launchd_plist(
    *,
    python_executable: Optional[str] = None,
    hermes_home: Optional[Path] = None,
    host: str = "127.0.0.1",
    port: int = DEFAULT_BROKER_PORT,
) -> bytes:
    bind_host = validate_bind_host(host)
    python_path = python_executable or sys.executable
    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    log_dir = home / "logs"
    # launchd needs the log target to exist; owner-only like the plist.
    log_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(log_dir, 0o700)
    payload = {
        "Label": BROKER_LAUNCHD_LABEL,
        "ProgramArguments": [
            python_path,
            "-m",
            "hermes_cli.main",
            "oauth-broker",
            "run",
            "--host",
            bind_host,
            "--port",
            str(int(port)),
        ],
        "RunAtLoad": True,
        # Restart on crash; a clean exit (deliberate bootout/stop) stays down.
        "KeepAlive": {"SuccessfulExit": False},
        # Damp crash loops: launchd waits this many seconds between respawns.
        "ThrottleInterval": 5,
        "StandardOutPath": str(log_dir / "oauth-broker.log"),
        "StandardErrorPath": str(log_dir / "oauth-broker.error.log"),
        # Exactly one environment variable. Tokens and the client key live in
        # the Keychain; the plist must never carry secret material.
        "EnvironmentVariables": {"HERMES_HOME": str(home)},
    }
    return plistlib.dumps(payload)


def write_broker_launchd_plist(*, plist_path: Path, content: bytes) -> Path:
    """Atomically write the plist with owner-only permissions and durability."""
    plist_path = Path(plist_path).expanduser()
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    fd, staging_name = tempfile.mkstemp(
        prefix=f".{plist_path.name}.",
        suffix=".tmp",
        dir=plist_path.parent,
    )
    staging = Path(staging_name)
    try:
        os.fchmod(fd, 0o600)
        remaining = memoryview(content)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError("short write while staging launchd plist")
            remaining = remaining[written:]
        os.fsync(fd)
    except Exception:
        os.close(fd)
        try:
            staging.unlink()
        except FileNotFoundError:
            pass
        raise
    else:
        os.close(fd)

    try:
        os.replace(staging, plist_path)
        directory_fd = os.open(plist_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass
        raise
    return plist_path


def launchd_domain(uid: Optional[int] = None) -> str:
    return f"gui/{os.getuid() if uid is None else uid}"


def _service_target(uid: Optional[int]) -> str:
    return f"{launchd_domain(uid)}/{BROKER_LAUNCHD_LABEL}"


def launchctl_bootstrap_argv(
    plist_path: Path, uid: Optional[int] = None
) -> List[str]:
    return ["launchctl", "bootstrap", launchd_domain(uid), str(plist_path)]


def launchctl_bootout_argv(uid: Optional[int] = None) -> List[str]:
    return ["launchctl", "bootout", _service_target(uid)]


def launchctl_kickstart_argv(uid: Optional[int] = None) -> List[str]:
    return ["launchctl", "kickstart", "-k", _service_target(uid)]


def launchctl_print_argv(uid: Optional[int] = None) -> List[str]:
    return ["launchctl", "print", _service_target(uid)]


def install_broker_service(
    *,
    plist_path: Path,
    content: bytes,
    apply: bool = False,
    runner: Optional[Callable[..., object]] = None,
    uid: Optional[int] = None,
) -> InstallResult:
    """Write the plist and (only with ``apply`` + explicit runner) load it."""
    if apply and runner is None:
        raise ValueError(
            "apply=True requires an explicit runner — live launchctl actions "
            "must come from a confirmed CLI invocation"
        )
    plist_path = Path(plist_path)
    previous_content: Optional[bytes] = None
    if apply and plist_path.exists():
        if not plist_path.is_file():
            raise ValueError("existing launchd plist path is not a regular file")
        previous_content = plist_path.read_bytes()

    written = write_broker_launchd_plist(plist_path=plist_path, content=content)
    bootstrap = launchctl_bootstrap_argv(written, uid)
    kickstart = launchctl_kickstart_argv(uid)
    executed = False
    if apply:
        assert runner is not None
        bootstrapped = False
        try:
            runner(bootstrap, check=True)
            bootstrapped = True
            runner(kickstart, check=True)
        except BaseException as install_error:
            recovery_errors = []
            if bootstrapped:
                try:
                    runner(launchctl_bootout_argv(uid), check=True)
                except BaseException as exc:
                    recovery_errors.append(exc)
            try:
                if previous_content is None:
                    written.unlink(missing_ok=True)
                    directory_fd = os.open(written.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                else:
                    write_broker_launchd_plist(
                        plist_path=written, content=previous_content
                    )
            except BaseException as exc:
                recovery_errors.append(exc)
            if recovery_errors:
                raise RuntimeError(
                    "launchd install failed and filesystem/service recovery was incomplete"
                ) from install_error
            raise
        executed = True
    return {
        "plist_path": written,
        "bootstrap": bootstrap,
        "kickstart": kickstart,
        "executed": executed,
    }


def uninstall_broker_service(
    *,
    plist_path: Path,
    apply: bool = False,
    runner: Optional[Callable[..., object]] = None,
    uid: Optional[int] = None,
) -> UninstallResult:
    """Report (and only with ``apply`` + runner, perform) service removal.

    Never touches Keychain grants — logout is a separate, explicitly
    confirmed command.
    """
    if apply and runner is None:
        raise ValueError(
            "apply=True requires an explicit runner — live launchctl actions "
            "must come from a confirmed CLI invocation"
        )
    bootout = launchctl_bootout_argv(uid)
    executed = False
    if apply:
        assert runner is not None
        runner(bootout, check=True)
        if plist_path.exists():
            plist_path.unlink()
        executed = True
    return {"plist_path": plist_path, "bootout": bootout, "executed": executed}


__all__ = [
    "BROKER_LAUNCHD_LABEL",
    "broker_launchd_plist_path",
    "install_broker_service",
    "launchctl_bootout_argv",
    "launchctl_bootstrap_argv",
    "launchctl_kickstart_argv",
    "launchctl_print_argv",
    "launchd_domain",
    "render_broker_launchd_plist",
    "uninstall_broker_service",
    "write_broker_launchd_plist",
]
