"""Shared plumbing for the extracted dashboard routers — thin wrappers over the
late-binding seam in :mod:`hermes_cli.web_deps` (web_server owns helpers/state;
every access resolves at call time so ``monkeypatch.setattr(<owning module>, ...)`` wins)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
import time
from typing import Any, Callable, Dict, Optional

from fastapi import HTTPException

from hermes_cli.web_deps import LateState, late
from hermes_cli.web_server_profiles import _profile_cli_args

# Same logger the handlers used before extraction (identical logger object).
log = logging.getLogger("hermes_cli.web_server")

_profile_scope = late("_profile_scope", "hermes_cli.web_server_profiles")
_spawn_hermes_action = late("_spawn_hermes_action", "hermes_cli.web_server_gateway")
# Config read-modify-write serialization for off-loop handlers (live lock —
# LateState supports ``with``-blocks).
_CONFIG_MUTATION_LOCK = LateState("_CONFIG_MUTATION_LOCK")


@contextlib.contextmanager
def config_write_scope(profile: Optional[str]):
    """Profile scope, then the config mutation lock — the write-path nesting
    every config-mutating handler uses."""
    with _profile_scope(profile):
        with _CONFIG_MUTATION_LOCK:
            yield


async def scoped_to_thread(profile: Optional[str], fn: Callable[[], Any]) -> Any:
    """Run ``fn()`` inside ``_profile_scope(profile)`` on a worker thread."""

    def _run():
        with _profile_scope(profile):
            return fn()

    return await asyncio.to_thread(_run)


@contextlib.contextmanager
def http_failure(log_msg: str, status: int, prefix: Optional[str] = None, *, detail: Optional[str] = None):
    """Map unexpected exceptions to an ``HTTPException``.

    ``HTTPException`` passes through; anything else is logged with ``log_msg`` (traceback),
    then re-raised as ``HTTPException(status, f"{prefix}: {exc}")`` — or ``detail`` when given
    (fixed message, exception text only in the log).
    """
    try:
        yield
    except HTTPException:
        raise
    except Exception as exc:
        log.exception(log_msg)
        raise HTTPException(status_code=status, detail=detail if detail is not None else f"{prefix}: {exc}")


def spawn_profile_action(
    profile: Optional[str], argv: list, name: str, *, log_msg: str, prefix: str
) -> dict:
    """Spawn a background ``hermes -p <profile> <argv>`` action; a spawn
    failure is logged and becomes ``500 "<prefix>: <exc>"``."""
    with http_failure(log_msg, 500, prefix):
        proc = _spawn_hermes_action(_profile_cli_args(profile) + argv, name)
    return {"ok": True, "pid": proc.pid, "name": name}


def require(value: Optional[str], detail: str) -> str:
    """Strip ``value``; 400 with ``detail`` when empty."""
    stripped = (value or "").strip()
    if not stripped:
        raise HTTPException(status_code=400, detail=detail)
    return stripped


# Corrupt-store reporting for polled read endpoints. The dashboard polls analytics every few
# seconds; a persistently malformed state.db once produced ~520K identical tracebacks in 24 h
# (#96591). One WARNING per store per interval, then debug; the caller gets an explicit status
# instead of a 500. The file is never quarantined or renamed from here — that is `hermes doctor`'s job.
_CORRUPT_STORE_WARN_INTERVAL_S = 300.0
_corrupt_store_warned_at: Dict[str, float] = {}  # {db path: monotonic}

CORRUPT_STORE_DETAIL = {
    "error": "state_db_corrupt",
}


def _corrupt_store_detail(db_path):
    """Return recovery guidance pinned to the profile that owns ``db_path``."""
    from hermes_constants import profile_name_for_home

    profile = profile_name_for_home(db_path.parent)
    profile_arg = f"-p {profile} " if profile else ""
    return {
        **CORRUPT_STORE_DETAIL,
        "message": (
            f"state.db corrupt — run `hermes {profile_arg}doctor` "
            f"(then `hermes {profile_arg}doctor --fix` or "
            f"`hermes {profile_arg}sessions repair`)."
        ),
    }


@contextlib.contextmanager
def corrupt_store_as_status(db_path):
    """Map a corrupt-image ``sqlite3.DatabaseError`` from a state.db read to a 503 status
    payload, warning once per store per :data:`_CORRUPT_STORE_WARN_INTERVAL_S`.
    Busy/locked and every other error propagate unchanged."""
    from hermes_state_errors import is_malformed_db_error

    try:
        yield
    except sqlite3.DatabaseError as exc:
        if not is_malformed_db_error(exc):
            raise
        key, now = str(db_path), time.monotonic()
        detail = _corrupt_store_detail(db_path)
        last = _corrupt_store_warned_at.get(key)
        if last is None or now - last >= _CORRUPT_STORE_WARN_INTERVAL_S:
            _corrupt_store_warned_at[key] = now
            log.warning("state.db at %s is corrupt (%s); dashboard reads return a status payload until it is "
                        "repaired; %s", db_path, exc, detail["message"])
        else:
            log.debug("state.db at %s still corrupt: %s", db_path, exc)
        raise HTTPException(status_code=503, detail={**detail, "path": key}) from exc
