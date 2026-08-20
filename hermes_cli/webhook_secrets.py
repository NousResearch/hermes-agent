"""Canonical webhook secret-reference persistence and resolution.

All webhook writers and readers use this seam so profile resolution, fallback,
and migration locking cannot drift between CLI and gateway surfaces.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_LOCK_TIMEOUT_SECONDS = 10.0
_LOCK_STALE_SECONDS = 60.0


def resolve_webhook_secret(secret_ref: object) -> str:
    """Resolve one non-empty reference under the active Hermes profile."""
    if not isinstance(secret_ref, str) or not secret_ref.strip():
        return ""
    ref = secret_ref.strip()
    try:
        from agent.secret_scope import get_secret

        value = get_secret(ref, "")
        if value:
            return str(value)
    except Exception:
        pass
    try:
        from hermes_cli.config import get_env_value_prefer_dotenv

        return str(get_env_value_prefer_dotenv(ref) or "")
    except Exception:
        return ""


@contextmanager
def webhook_secret_write_lock() -> Iterator[None]:
    """Serialize webhook secret writers across CLI/gateway processes.

    The lock uses atomic O_EXCL publication under the active Hermes home. A
    stale lock older than one minute is reclaimed; live contention is bounded
    and fails closed rather than racing two read-modify-write .env updates.
    """
    from hermes_constants import get_hermes_home

    home = Path(get_hermes_home())
    home.mkdir(parents=True, exist_ok=True)
    lock_path = home / ".webhook-secrets.lock"
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.write(fd, f"{os.getpid()}\n".encode())
            os.fsync(fd)
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > _LOCK_STALE_SECONDS:
                    lock_path.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for webhook secret writer lock")
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            os.close(fd)
        finally:
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass


def store_webhook_secret_unlocked(secret_ref: str, value: str) -> None:
    """Persist while the caller already owns :func:`webhook_secret_write_lock`."""
    if not isinstance(secret_ref, str) or not secret_ref.strip():
        raise ValueError("webhook secret reference must be non-empty")
    if not isinstance(value, str) or not value:
        raise ValueError("webhook secret value must be non-empty")
    from hermes_cli.config import save_env_value

    save_env_value(secret_ref.strip(), value)


def store_webhook_secret(secret_ref: str, value: str) -> None:
    """Persist one webhook secret through the profile .env owner."""
    with webhook_secret_write_lock():
        store_webhook_secret_unlocked(secret_ref, value)


__all__ = [
    "resolve_webhook_secret",
    "store_webhook_secret",
    "store_webhook_secret_unlocked",
    "webhook_secret_write_lock",
]
