"""Shared plumbing for the extracted dashboard routers — thin wrappers over the
late-binding seam in :mod:`hermes_cli.web_deps` (web_server owns helpers/state;
every access resolves at call time so ``monkeypatch.setattr(<owning module>, ...)`` wins)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import sqlite3
import time
from typing import Any, Callable, Dict, Optional

from fastapi import HTTPException

from hermes_cli.web_deps import LateState, late
from hermes_cli.web_server_profiles import _profile_cli_args

# Same logger the handlers used before extraction (identical logger object).
log = logging.getLogger("hermes_cli.web_server")

_profile_scope = late("_profile_scope", "hermes_cli.web_server_profiles")
_config_profile_scope = late("_config_profile_scope", "hermes_cli.web_server_profiles")
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


async def config_scoped_to_thread(profile: Optional[str], fn: Callable[[], Any]) -> Any:
    """Run ``fn()`` inside ``_config_profile_scope(profile)`` on a worker thread —
    home + secret scope without the process-global skills-module swap."""

    def _run():
        with _config_profile_scope(profile):
            return fn()

    return await asyncio.to_thread(_run)


def destructive_profile(profile: Optional[str], route: str) -> Optional[str]:
    """The profile a DESTRUCTIVE or PRIVILEGED route acts on, or 400 when it is ambiguous.

    One backend serves every profile, so an omitted ``profile`` on a route that deletes,
    overwrites or privileges profile-owned data is not a default — it silently meant
    "whichever home this process launched with". Named profile: honoured. Omitted:
    rejected as soon as the process hosts more than one profile
    (``is_multiplex_active()``, decided once at boot by
    ``activate_multi_profile_hosting_eagerly``). A genuinely single-profile host has
    nothing to confuse, so there an omitted profile keeps meaning the launch profile
    and `curl` against a plain ``hermes serve`` is unchanged.

    "Privileged" is the same class as "destructive": arming an auto-approved shell hook
    in the wrong profile is at least as bad as removing one from it.
    """
    if (profile or "").strip():
        return profile
    from agent.secret_scope import is_multiplex_active
    if is_multiplex_active():
        raise HTTPException(
            status_code=400,
            detail=f"{route} requires an explicit profile: this backend serves several profiles, "
                   "so an unnamed target would act on the wrong profile's data.")
    return profile


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


REDACTED_CREDENTIAL_WRITE_DETAIL = (
    "Refusing to save a redacted credential preview; re-enter the full secret to replace it."
)


def redacted_credential_preview(value: Any) -> Optional[str]:
    """Return a display-only credential sentinel that can never gain write authority."""
    if not value:
        return None
    from hermes_cli.config import redact_key
    return f"«redacted:{redact_key(str(value))}»"


# Legacy bare masks (pre-sentinel pages, older Desktop builds) are recognised by the
# producer shape of ``agent.redact.mask_secret`` — never by equality to the current
# secret, which would authorise a stale preview after a rotation (#121002).
_LEGACY_MASK_RE = re.compile(r".{4}\.\.\..{4}")


def is_redacted_credential_preview(submitted: Any) -> bool:
    """Recognize current, stale and legacy dashboard previews by shape alone."""
    value = str(submitted or "")
    # Any ``«redacted…`` value is already-masked output (the same test agent.redact uses
    # to skip re-masking): our ``«redacted:…»`` sentinel, ``«redacted-secret»`` and the
    # vault marker ``«redacted-vault-secret»``. Then the legacy bare mask shapes.
    if value.startswith("«redacted"):
        return True
    return value == "***" or _LEGACY_MASK_RE.fullmatch(value) is not None


# Corrupt-store reporting for polled read endpoints. The dashboard polls analytics every few
# seconds; a persistently malformed state.db once produced ~520K identical tracebacks in 24 h
# (#96591). One WARNING per store per interval, then debug; the caller gets an explicit status
# instead of a 500. The file is never quarantined or renamed from here — that is `hermes doctor`'s job.
_CORRUPT_STORE_WARN_INTERVAL_S = 300.0
_corrupt_store_warned_at: Dict[str, float] = {}  # {db path: monotonic}

CORRUPT_STORE_DETAIL = {
    "error": "state_db_corrupt",
    "message": "state.db corrupt — run `hermes doctor` (then `hermes doctor --fix` or `hermes sessions repair`).",
}
# One payload per persistence-cause bucket (``classify_persistence_error``); ``error`` codes
# all follow the ``state_db_*`` scheme of the pre-existing ``state_db_corrupt``.
# Same guidance as the deleted_wal / replaced turn explainers: `doctor --fix` while a holder
# lives would repair the wrong generation in place, so it is deliberately NOT suggested here.
DELETED_WAL_DETAIL = {
    "error": "state_db_deleted_wal",
    "message": "another Hermes process still holds an old copy of the session database's write-ahead log — "
               "quit every Hermes process on this profile, run `hermes doctor` (it names the holders), "
               "then start Hermes again. Do not run `hermes doctor --fix` while they run.",
}
STATE_DB_REPLACED_DETAIL = {
    "error": "state_db_replaced",
    "message": "state.db was replaced while Hermes was running — stop Hermes, run `hermes doctor`, "
               "then start it again. Do not run `hermes doctor --fix`, which would repair the wrong file in place.",
}
# Every other bucket a malformed image can classify as ("corrupt", "fts_index") is the corrupt payload.
_STORE_STATUS_DETAIL_BY_CAUSE = {"deleted_wal": DELETED_WAL_DETAIL, "replaced": STATE_DB_REPLACED_DETAIL}


@contextlib.contextmanager
def corrupt_store_as_status(db_path):
    """Map a corrupt-image ``sqlite3.DatabaseError`` or ``StateDbReplacedError`` from a state.db read to a 503 status
    payload, warning once per store per :data:`_CORRUPT_STORE_WARN_INTERVAL_S`.
    Busy/locked and every other error propagate unchanged."""
    from hermes_state_errors import StateDbReplacedError, classify_persistence_error, is_malformed_db_error

    try:
        yield
    except (sqlite3.DatabaseError, StateDbReplacedError) as exc:
        if not isinstance(exc, StateDbReplacedError) and not is_malformed_db_error(exc):
            raise
        key, now = str(db_path), time.monotonic()
        last = _corrupt_store_warned_at.get(key)
        detail = _STORE_STATUS_DETAIL_BY_CAUSE.get(classify_persistence_error(exc), CORRUPT_STORE_DETAIL)
        if last is None or now - last >= _CORRUPT_STORE_WARN_INTERVAL_S:
            _corrupt_store_warned_at[key] = now
            log.warning("state.db at %s is unreadable (%s); dashboard reads return a status payload until it is "
                        "repaired — run `hermes doctor`", db_path, exc)
        else:
            log.debug("state.db at %s still has error: %s", db_path, exc)
        raise HTTPException(status_code=503, detail={**detail, "path": key}) from exc
