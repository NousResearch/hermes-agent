"""Start or retire a named profile's gateway under s6."""

import os
from pathlib import Path
from typing import Literal

from hermes_cli.service_manager import (
    S6ServiceManager,
    S6_SERVICE_PREFIX,
    register_unregistered_profile_gateway,
)
from utils.files import atomic_write_text

_INTENT_FILE = "gateway-lifecycle-intent"


class GatewayRetiredError(Exception):
    """The managed endpoint cannot start a retired gateway."""


def profile_gateway_retired(profile_dir: Path) -> bool:
    try:
        return (profile_dir / _INTENT_FILE).read_text(
            encoding="utf-8"
        ).strip() == "retired"
    except FileNotFoundError:
        return False


def reconcile_profile_gateway(
    manager: S6ServiceManager,
    profile: str,
    profile_dir: Path,
    desired: Literal["running", "retired"],
) -> bool:
    """Return True when s6 confirms the requested state."""
    import fcntl

    name = f"{S6_SERVICE_PREFIX}{profile}"
    intent_file = profile_dir / _INTENT_FILE
    # Lock a separate file so replacing the state file doesn't release the lock.
    fd = os.open(
        profile_dir / f".{_INTENT_FILE}.lock",
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
        0o600,
    )
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        try:
            intent = intent_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            intent = None

        if desired == "running" and intent == "retired":
            raise GatewayRetiredError
        if intent not in (None, "starting", "running", "retired"):
            raise ValueError("Invalid persisted S6 gateway lifecycle intent")

        if not (manager.scandir / name).is_dir():
            if desired == "retired" and intent is None:
                atomic_write_text(intent_file, "retired\n", mode=0o600, fsync_dir=True)
                return True
            if not register_unregistered_profile_gateway(manager, profile):
                return False

        fields = manager.gateway_supervisor_state(name)
        if fields is None:
            return False
        up, wanted_up = fields
        if intent == "starting" and not wanted_up:
            # This sample may precede the queued 'up'. A new 'up' would make
            # it ambiguous which request the next sample acknowledges.
            return False
        if desired == "running":
            if not wanted_up:
                atomic_write_text(intent_file, "starting\n", mode=0o600, fsync_dir=True)
                manager.start(name, lock_fd=fd)
                return False
            if intent != "running":
                atomic_write_text(intent_file, "running\n", mode=0o600, fsync_dir=True)
            return up

        if intent != "retired":
            atomic_write_text(intent_file, "retired\n", mode=0o600, fsync_dir=True)
        if wanted_up or not (manager.scandir / name / "down").is_file():
            manager.stop(name, persistent=True, lock_fd=fd)
            return False
        return not up and manager.gateway_fully_down(name)
    finally:
        os.close(fd)
