"""Pressure-driven Ollama release, fenced against complete conversation turns."""

from __future__ import annotations

import errno
import json
import logging
import os
import time
import urllib.request
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator
from urllib.parse import urlsplit, urlunsplit

from hermes_cli.active_sessions import _FileLock

logger = logging.getLogger(__name__)


def _try_lock(handle) -> bool:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            return False
        raise


class ActivityGate:
    """Admission lock plus kernel-held turn records shared by all profiles.

    Turns hold separate records so parallel conversations and delegated children
    do not serialize. The admission lock covers only record changes and an idle
    release. A process exit drops its kernel locks; stale records reset the quiet
    period when collected. No age-based expiry can mistake a long tool for idle.
    """

    def __init__(self, directory: Path, *, now: Callable[[], float] = time.monotonic):
        self.directory = directory
        self.now = now
        self.admission = directory / "admission.lock"
        self.quiet = directory / "quiet-since"

    @contextmanager
    def turn(self) -> Iterator[None]:
        path = self.directory / f"turn-{uuid.uuid4().hex}"
        with _FileLock(self.admission):
            handle = path.open("x+b")
            try:
                handle.write(b"0")
                handle.flush()
                if not _try_lock(handle):
                    raise RuntimeError("New activity record could not be locked")
            except BaseException:
                handle.close()
                path.unlink(missing_ok=True)
                raise
        try:
            yield
        finally:
            try:
                with _FileLock(self.admission):
                    self.quiet.write_text(str(self.now()), encoding="utf-8")
                    handle.close()
                    path.unlink(missing_ok=True)
            finally:
                handle.close()

    @contextmanager
    def idle(self, seconds: float) -> Iterator[float | None]:
        """Yield the quiet-period identity, or None; fence new turns until exit."""
        with _FileLock(self.admission):
            busy = False
            stale = False
            for path in self.directory.glob("turn-*"):
                with path.open("r+b") as handle:
                    if not _try_lock(handle):
                        busy = True
                        break
                path.unlink()
                stale = True
            if stale:
                self.quiet.write_text(str(self.now()), encoding="utf-8")
            if busy or not self.quiet.exists():
                yield None
                return
            since = float(self.quiet.read_text(encoding="utf-8"))
            elapsed = self.now() - since
            yield since if elapsed >= seconds else None


def _local_root(base_url: str) -> str | None:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in ("http", "https")
        or parsed.hostname not in ("localhost", "127.0.0.1", "::1")
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") not in ("", "/v1")
    ):
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


class IdleOllamaPolicy:
    """Release only an explicitly opted-in, resident local model while idle."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        enabled: bool,
        gate: ActivityGate,
        pressure: Callable[[], bool | None],
        idle_seconds: float = 60,
        api_key: str = "",
    ):
        self.root = _local_root(base_url)
        self.model = model
        self.enabled = enabled is True
        self.gate = gate
        self.pressure = pressure
        self.idle_seconds = idle_seconds
        self.api_key = api_key
        self._released_period: float | None = None

    def _request(self, path: str, body: dict | None = None):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.root}{path}",
            headers=headers,
            data=json.dumps(body).encode() if body is not None else None,
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.loads(response.read(1 << 20))

    def tick(self) -> bool:
        if not self.enabled or self.root is None:
            return False
        try:
            if self.pressure() is not True:
                return False
            with self.gate.idle(self.idle_seconds) as period:
                if period is None or period == self._released_period:
                    return False
                models = self._request("/api/ps").get("models", [])
                names = {self.model}
                if ":" not in self.model.rsplit("/", 1)[-1]:
                    names.add(f"{self.model}:latest")
                resident = next(
                    (
                        row["name"]
                        for row in models
                        if isinstance(row, dict)
                        and row.get("name") in names
                        and isinstance(row.get("size_vram"), (int, float))
                        and row["size_vram"] > 0
                    ),
                    None,
                )
                if resident is None:
                    return False
                response = self._request(
                    "/api/generate",
                    {"model": resident, "keep_alive": 0},
                )
                if (
                    response.get("done") is not True
                    or response.get("done_reason") != "unload"
                ):
                    return False
                self._released_period = period
                logger.info(
                    "Released idle Ollama model %s due to memory pressure; "
                    "the next request will load it normally",
                    self.model,
                )
                return True
        except Exception:
            logger.debug(
                "Idle Ollama release skipped: pressure or activity could not be verified",
                exc_info=True,
            )
            return False
