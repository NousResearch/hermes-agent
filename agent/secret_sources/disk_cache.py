"""Shared disk-cache helpers for external secret sources.

Secret-source caches are plaintext-equivalent to ``~/.hermes/.env``. Keep
them under ``<hermes_home>/cache``, write atomically, and avoid letting cache
I/O failures block startup.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Optional


def disk_cache_path(cache_basename: str, home_path: Optional[Path] = None) -> Path:
    """Return a secret-source disk cache path under hermes_home/cache/."""

    if home_path is None:
        home_path = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    return home_path / "cache" / cache_basename


def read_secret_cache(
    *,
    cache_basename: str,
    cache_key: str,
    ttl_seconds: float,
    home_path: Optional[Path] = None,
) -> tuple[dict[str, str], float] | None:
    """Return ``(secrets, fetched_at)`` for a fresh cache entry, else None."""

    if ttl_seconds <= 0:
        return None
    path = disk_cache_path(cache_basename, home_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("key") != cache_key:
        return None
    secrets = payload.get("secrets")
    fetched_at = payload.get("fetched_at")
    if not isinstance(secrets, dict) or not isinstance(fetched_at, (int, float)):
        return None
    typed_secrets = {
        key: value
        for key, value in secrets.items()
        if isinstance(key, str) and isinstance(value, str)
    }
    fetched_at_float = float(fetched_at)
    if (time.time() - fetched_at_float) >= ttl_seconds:
        return None
    return typed_secrets, fetched_at_float


def write_secret_cache(
    *,
    cache_basename: str,
    cache_key: str,
    secrets: dict[str, str],
    fetched_at: float,
    temp_prefix: str,
    home_path: Optional[Path] = None,
) -> None:
    """Persist a cache entry atomically with cache dir mode 0700 and file 0600."""

    path = disk_cache_path(cache_basename, home_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(path.parent, 0o700)
        except OSError:
            pass
        payload = {
            "key": cache_key,
            "secrets": secrets,
            "fetched_at": fetched_at,
        }
        fd, tmp = tempfile.mkstemp(
            prefix=temp_prefix, suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError:
        pass
