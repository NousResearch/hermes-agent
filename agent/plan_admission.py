"""Host-wide one-at-a-time admission for the shared Z.ai GLM Coding Plan route.

Owner rule: one GLM-backed bot actively working at a time. Dispatcher-local
locks do not reach CLI one-shots, cron children or helper launches, so
admission is enforced at the chat-completions transport seam: every request
whose base_url is an official Z.ai coding endpoint holds an exclusive
host-wide flock for the duration of the request / stream attempt. Non-plan
routes pass through untouched. Fail-closed: a caller that cannot acquire
within the bounded wait raises PlanAdmissionBusy instead of proceeding
unlocked. Re-entry inside the same process (auxiliary calls during a turn)
reuses the outer hold. The append-only log records acquisition timings and
route metadata only - never message content, keys or responses.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import pwd
import threading
import time
import urllib.parse

_PLAN_HOSTS = frozenset({"api.z.ai", "open.bigmodel.cn"})
_PLAN_PATH_PREFIX = "/api/coding/paas"
DEFAULT_WAIT_SECONDS = 1800.0
_RELEASE_MARKER = object()


def _admission_dir() -> str:
    # pwd, not $HOME: the lock must be identical for every process of this
    # user regardless of profile/child HOME overrides.
    return os.path.join(pwd.getpwuid(os.getuid()).pw_dir, ".hermes", "admission")


def lock_path() -> str:
    return os.path.join(_admission_dir(), "zai-plan.lock")


def log_path() -> str:
    return os.path.join(_admission_dir(), "log.jsonl")


def is_plan_route(base_url) -> bool:
    """True for the official Z.ai GLM Coding Plan chat endpoints only."""
    if not base_url or not isinstance(base_url, str):
        return False
    try:
        parsed = urllib.parse.urlparse(base_url if "//" in base_url else "https://" + base_url)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    path = parsed.path or "/"
    return host in _PLAN_HOSTS and path.startswith(_PLAN_PATH_PREFIX)


class PlanAdmissionBusy(RuntimeError):
    """Another process holds plan admission; fail closed, never proceed unlocked."""


_local = threading.local()


def _log(event: str, *, wait_s: float, base_url, model, purpose: str) -> None:
    try:
        host = urllib.parse.urlparse(base_url or "").hostname or ""
        os.makedirs(_admission_dir(), mode=0o700, exist_ok=True)
        row = {
            "ts": round(time.time(), 3),
            "event": event,
            "pid": os.getpid(),
            "thread": threading.current_thread().name,
            "wait_s": round(wait_s, 3),
            "host": host,
            "model": str(model or ""),
            "purpose": purpose,
        }
        with open(log_path(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError:
        # Admission logging must never break inference; the flock is the gate.
        pass


@contextlib.contextmanager
def plan_admission(base_url, *, model=None, purpose: str = "chat", wait_seconds: float | None = None):
    """Serialize plan-route requests host-wide for the duration of the block."""
    if not is_plan_route(base_url):
        yield None
        return
    depth = getattr(_local, "depth", 0)
    if depth > 0:
        # Re-entry inside the same process reuses the outer hold.
        _local.depth = depth + 1
        try:
            yield None
        finally:
            _local.depth = depth
        return
    os.makedirs(_admission_dir(), mode=0o700, exist_ok=True)
    fd = os.open(lock_path(), os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    t0 = time.monotonic()
    waited = 0.0
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                waited = time.monotonic() - t0
                if waited >= (DEFAULT_WAIT_SECONDS if wait_seconds is None else wait_seconds):
                    _log("busy", wait_s=waited, base_url=base_url, model=model, purpose=purpose)
                    raise PlanAdmissionBusy(
                        "plan admission busy: another process holds the Z.ai GLM Coding Plan slot"
                    ) from None
                time.sleep(0.25)
        waited = time.monotonic() - t0
        _log("acquire", wait_s=waited, base_url=base_url, model=model, purpose=purpose)
        _local.depth = 1
        try:
            yield None
        finally:
            _local.depth = 0
            _log("release", wait_s=0.0, base_url=base_url, model=model, purpose=purpose)
    finally:
        if acquired:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)
