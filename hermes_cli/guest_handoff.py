"""Guest handoff: after a Nous account replaces the free tier, move the free tier's apps into it.

The account service moves a free-tier identity's connected apps (and its preview cohort) into a
signed-in account with one call, ``POST /api/anonymous/claim {"token": "<anon_...>"}``, authorized by
the account's own ``hermes-cli`` Bearer token. Hermes never asks first: the move just happens after
the account is saved, and the result (the account email, what moved, what needs reconnecting) is
shown afterwards. Two halves:

- :func:`note_replacement` runs at the lowest write seams (``auth._store_provider_state`` and
  ``auth_nous._write_shared_nous_state``), under the caller's auth-store locks, whenever a non-guest
  Nous state replaces a guest. It writes a pending record and does no network. Records sit beside
  the shared Nous store, keyed by a hash of the ``anon_`` credential.
- :func:`drain` makes one attempt per due record, outside every lock: each sign-in surface calls it
  after its save, and the retry worker, ``hermes auth status`` and ``hermes auth upgrade`` call it
  for a move a transient error left retrying.

User copy never says guest / anonymous / claim.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger("hermes_cli.auth")

IDLE, PENDING, RETRYING, CLAIMED, GONE, FAILED = "idle", "pending", "retrying", "claimed", "gone", "failed"
_ACTIVE = frozenset({PENDING, RETRYING})
HANDOFF_FILENAME = "guest_handoff.json"
CLAIM_TIMEOUT_SECONDS = 10.0
# While an attempt is in flight its record is not due, so a second process never sends it too.
_LEASE_SECONDS = CLAIM_TIMEOUT_SECONDS + 20.0
# Wait before each retry of a transient failure; the attempt after the last one is final.
_RETRY_DELAYS_SECONDS = (60.0, 600.0, 3600.0, 6 * 3600.0)
# ``_is_expiring`` treats any token as expiring inside this skew, so the account token refreshes now.
_FORCE_REFRESH_SKEW = 10**9

# Outcome of one claim response.
_CLAIMED, _GONE, _RETRY, _FAIL = "claimed", "gone", "retry", "fail"
# A plain reason per final code; any other code shows only the generic sentence, never the code.
_FAILURE_COPY = {
    "client_not_allowed": "This sign-in can't receive them. Sign in again with `hermes auth add nous`.",
    "account_locked": "This Nous account is locked.",
}
# The service's refusal codes; anything unlisted retries (``_classify``).
_REFUSALS = {
    "unknown_token": _GONE, "already_retired": _GONE,
    "client_not_allowed": _FAIL, "account_locked": _FAIL, "invalid_request": _FAIL,
}

_records_lock_holder = threading.local()


def _now() -> float:
    return time.time()


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat()


def _token_key(anon_token: str) -> str:
    return hashlib.sha256(anon_token.encode("utf-8")).hexdigest()[:32]


def _home_key() -> str:
    from hermes_constants import hermes_home_key
    return hermes_home_key()


def _records_path() -> Path:
    from hermes_cli.auth_nous import _nous_shared_store_path
    return _nous_shared_store_path().with_name(HANDOFF_FILENAME)


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {"_schema": 1, "records": {}}


def _records() -> dict[str, Any]:
    try:
        return _load(_records_path())["records"]
    except RuntimeError:
        return {}  # the test seat belt on the shared store path


def _update(path: Path, key: str, change) -> Optional[dict[str, Any]]:
    """Apply *change(record) -> record | None* to one record under the records lock and return what
    was written (None: nothing written). The innermost lock: the hook takes it under the auth-store
    locks, and it never holds another lock itself."""
    from hermes_cli.auth import _file_lock, _save_private_json
    with _file_lock(path.with_suffix(".lock"), _records_lock_holder, 15.0, f"Timed out waiting for {path.name}"):
        data = _load(path)
        record = change(data["records"].get(key))
        if record is not None:
            record["updated_at"] = _iso(_now())
            data["records"][key] = record
            _save_private_json(path, data, sort_keys=True)
        return record


# --- The hook: no network, runs under the writer's locks ------------------------------------------

def note_store_write(auth_store: dict[str, Any], provider_id: str, state: Any) -> None:
    """``auth._store_provider_state``'s hook: the Nous state the store holds now is the one replaced
    (the profile's own, else the root store's it falls back to)."""
    if provider_id == "nous":
        from hermes_cli.auth import _load_provider_state
        note_replacement(_load_provider_state(auth_store, "nous"), state)


def note_replacement(previous: Any, new: Any) -> None:
    """Record that a non-guest Nous state replaced the guest *previous*. Idempotent per ``anon_``
    credential; a finished record is never reopened."""
    from hermes_cli.anon_auth import is_guest_state
    if not is_guest_state(previous) or is_guest_state(new) or not isinstance(new, dict):
        return
    anon_token = previous.get("anon_token")
    if not isinstance(anon_token, str) or not anon_token:
        return
    try:
        path = _records_path()
    except RuntimeError:
        return
    home = _home_key()

    def change(record: Optional[dict[str, Any]]) -> dict[str, Any]:
        if record is None:
            record = {"state": PENDING, "anon_token": anon_token, "attempts": 0, "profiles": [],
                      "created_at": _iso(_now())}
        if home not in record["profiles"]:
            record["profiles"].append(home)
        return record

    record = _update(path, _token_key(anon_token), change)
    logger.info("Nous sign-in replaced the free tier; handoff %s (%s)", record["state"],
                _token_key(anon_token)[:8])


# --- The runner: outside every lock ---------------------------------------------------------------

def _due(record: dict[str, Any], now: float) -> bool:
    return record.get("state") in _ACTIVE and float(record.get("next_attempt_at") or 0) <= now


def drain() -> list[dict[str, Any]]:
    """One attempt for each of this profile's due records; returns their public views. Call it only
    outside the auth-store locks: it sends a request."""
    home, now = _home_key(), _now()
    due = [key for key, record in _records().items() if home in record.get("profiles", ()) and _due(record, now)]
    return [view for key in due if (view := _run_one(_records_path(), key)) is not None]


def _run_one(path: Path, key: str) -> Optional[dict[str, Any]]:
    from hermes_cli.auth import get_provider_auth_state
    from hermes_cli.auth_constants import AuthError
    try:
        bearer = _bearer()
    except (AuthError, TimeoutError) as exc:  # a busy store lock defers the move like a missing token
        logger.info("Free-tier handoff (%s): account token unavailable (%s)", key[:8],
                    getattr(exc, "code", None) or type(exc).__name__)
        return None
    record = _update(path, key, _reserve)
    if record is None:
        return None  # another run holds it, or it settled meanwhile
    prior = record["attempts"] - 1
    outcome, payload, retry_after = _post(record["anon_token"], bearer, get_provider_auth_state("nous") or {})
    logger.info("Free-tier handoff attempt %d (%s): %s", prior + 1, key[:8], payload.get("error") or outcome)
    record = _update(path, key, lambda r: _apply(r, outcome, payload, retry_after, prior))
    if outcome == _CLAIMED and payload.get("cohort_carried") is True:
        _refresh_account_token()
    return public_view(record) if record else None


def _reserve(record: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Count the attempt and lease the record before the request leaves: a lost answer still counts,
    and no other run sends it meanwhile."""
    now = _now()
    if record is None or not _due(record, now):
        return None
    return {**record, "attempts": int(record.get("attempts") or 0) + 1, "last_attempt_at": _iso(now),
            "next_attempt_at": now + _LEASE_SECONDS}


def _bearer(*, force: bool = False) -> str:
    from hermes_cli import auth
    # The resolver's 5 s memo can still hold the guest's token from just before the save.
    with auth._RESOLVE_TOKEN_CACHE_LOCK:
        auth._RESOLVE_TOKEN_CACHE.pop(_home_key(), None)
    if force:
        return auth.resolve_nous_access_token(timeout_seconds=CLAIM_TIMEOUT_SECONDS,
                                              refresh_skew_seconds=_FORCE_REFRESH_SKEW)
    return auth.resolve_nous_access_token(timeout_seconds=CLAIM_TIMEOUT_SECONDS)


def _refresh_account_token() -> None:
    """The account's token picks up the carried cohort only on its next mint, so mint it now."""
    from hermes_cli.auth_constants import AuthError
    try:
        _bearer(force=True)
    except AuthError as exc:
        logger.info("Free-tier handoff: account token refresh after the move failed: %s", exc)


def _post(anon_token: str, bearer: str, account: dict[str, Any]) -> tuple[str, dict[str, Any], Optional[float]]:
    from agent.retry_utils import parse_retry_after_seconds
    from hermes_cli.anon_auth import _anon_headers
    from hermes_cli.auth import _nous_portal_base_url, _resolve_verify
    from hermes_cli.auth_constants import httpx
    from hermes_cli.auth_nous import _nous_http_client
    verify = _resolve_verify(insecure=None, ca_bundle=None, auth_state=account)
    try:
        with _nous_http_client(CLAIM_TIMEOUT_SECONDS, verify) as client:
            response = client.post(
                f"{_nous_portal_base_url(account).rstrip('/')}/api/anonymous/claim",
                headers={**_anon_headers(), "authorization": f"Bearer {bearer}"}, json={"token": anon_token})
    except httpx.HTTPError as exc:
        return _RETRY, {"error": "unreachable", "detail": type(exc).__name__}, None
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    payload = payload if isinstance(payload, dict) else {}
    return _classify(response.status_code, payload), payload, parse_retry_after_seconds(response.headers)


def _classify(status: int, payload: dict[str, Any]) -> str:
    if status == 200:
        return _CLAIMED if payload.get("status") == "claimed" else _RETRY
    return _REFUSALS.get(str(payload.get("error") or ""), _RETRY)


def _apply(record: dict[str, Any], outcome: str, payload: dict[str, Any], retry_after: Optional[float],
           prior_attempts: int) -> dict[str, Any]:
    if record["state"] not in _ACTIVE:
        return record  # a finished move stays finished, whatever a late attempt heard
    record = {**record, "error": payload.get("error"), "next_attempt_at": None}
    if outcome == _CLAIMED:
        connectors = payload.get("connectors") if isinstance(payload.get("connectors"), dict) else None
        record.update(state=CLAIMED, error=None, result={
            "account_email": payload.get("account_email"),
            "destination_org_id": payload.get("destination_org_id"),
            # A service that does not report what moved (an older one) reads as unknown, never "none".
            "connectors_kind": (connectors or {}).get("kind") or "unknown",
            "moved_toolkits": list((connectors or {}).get("moved_toolkits") or []),
            "dropped_toolkits": list((connectors or {}).get("dropped_toolkits") or []),
            "cohort_carried": payload.get("cohort_carried") is True})
    elif outcome == _GONE:
        # Gone after an earlier attempt is most likely that attempt's lost success.
        record.update(state=GONE, probably_moved=payload.get("error") == "already_retired" or prior_attempts > 0)
    elif outcome == _FAIL or prior_attempts >= len(_RETRY_DELAYS_SECONDS):
        record.update(state=FAILED)
    else:
        record.update(state=RETRYING, next_attempt_at=_now() + (retry_after or _RETRY_DELAYS_SECONDS[prior_attempts]))
    if record["state"] in (CLAIMED, GONE):
        record.pop("anon_token", None)  # retired credential: nothing left to retry with
    return record


# --- The worker: retries a transient failure in a long-lived process ------------------------------

_WORKER_FIRST_TICK_SECONDS = 15.0
_WORKER_INTERVAL_SECONDS = 60.0
_worker_lock = threading.Lock()
_worker_stop = threading.Event()
_worker_thread: Optional[threading.Thread] = None


def start_worker() -> threading.Thread:
    """Start the process-wide retry worker (idempotent)."""
    global _worker_thread
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return _worker_thread
        _worker_stop.clear()
        # health: allow HX012 -- process-wide worker for the process's own profile; it must not inherit the starter's scope
        _worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="nous-guest-handoff")
        _worker_thread.start()
        return _worker_thread


def stop_worker(timeout: float = 5.0) -> None:
    global _worker_thread
    with _worker_lock:
        thread = _worker_thread
        _worker_stop.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=timeout)
    with _worker_lock:
        if _worker_thread is thread:
            _worker_thread = None


def _worker_loop() -> None:
    from tui_gateway.launch_profile_policy import launch_profile_scope_if_multiplexed
    wait = _WORKER_FIRST_TICK_SECONDS
    while not _worker_stop.wait(wait):
        try:
            with launch_profile_scope_if_multiplexed():
                drain()
        except Exception:
            logger.warning("Free-tier handoff retry tick failed; the records stay for the next one", exc_info=True)
        wait = _WORKER_INTERVAL_SECONDS


# --- Read side: status for the surfaces -----------------------------------------------------------

def public_view(record: dict[str, Any]) -> dict[str, Any]:
    """The wire shape the surfaces read; never the credential."""
    result = record.get("result") or {}
    next_at = record.get("next_attempt_at")
    return {
        "state": record.get("state", IDLE), "account_email": result.get("account_email"),
        "connectors_kind": result.get("connectors_kind"),
        "moved_toolkits": result.get("moved_toolkits", []), "dropped_toolkits": result.get("dropped_toolkits", []),
        "probably_moved": bool(record.get("probably_moved")), "error": record.get("error"),
        "retry_after": max(0, int(float(next_at) - _now())) if next_at else 0,
        "updated_at": record.get("updated_at")}


def handoff_status() -> dict[str, Any]:
    """This profile's latest handoff, or ``{"state": "idle"}``. A pure read (no lock, no network)."""
    home = _home_key()
    mine = [r for r in _records().values() if home in r.get("profiles", ())]
    if not mine:
        return {"state": IDLE}
    return public_view(max(mine, key=lambda r: str(r.get("updated_at") or "")))


def record_status(anon_token: str) -> Optional[dict[str, Any]]:
    """The handoff of one free-tier credential, or None when no record exists. A pure read: what a
    sign-in reports when its own run attempted nothing (another process held the record)."""
    record = _records().get(_token_key(anon_token))
    return public_view(record) if record else None


def wire_status(status: dict[str, Any]) -> dict[str, Any]:
    """A status as the desktop reads it: the fields, plus the user copy for them (:func:`result_lines`)."""
    return {**status, "lines": result_lines(status)}


def status_lines(status: dict[str, Any]) -> list[str]:
    """User copy for a non-idle handoff (``hermes auth status nous``)."""
    lines = result_lines(status)
    kind = status.get("connectors_kind")
    moved_some = kind in ("moved", "swapped") and status.get("moved_toolkits")
    if status.get("state") == CLAIMED and not moved_some and kind != "conflict":
        lines[0] = f"Signed in as {status.get('account_email') or 'your account'}. {lines[0]}"
    return lines


def result_lines(status: dict[str, Any]) -> list[str]:
    """What a handoff did, for the line after "Signed in as ..." (every sign-in surface) and for
    :func:`status_lines`."""
    from tools.tool_labels import app_title
    state = status.get("state")
    where = status.get("account_email") or "your account"
    moved = [app_title(slug) for slug in status.get("moved_toolkits") or []]
    dropped = [app_title(slug) for slug in status.get("dropped_toolkits") or []]
    if state in _ACTIVE:
        return ["Moving your free-tier apps to this account; Hermes keeps trying in the background."]
    if state == FAILED:
        reason = _FAILURE_COPY.get(str(status.get("error") or ""))
        return ["Your free-tier apps could not be moved." + (f" {reason}" if reason else "")]
    if state == GONE:
        return ["Your free-tier apps were probably moved already." if status.get("probably_moved")
                else "Nothing left to move from the free tier."]
    if state != CLAIMED:
        return []
    kind = status.get("connectors_kind")
    # An empty moved list is the truth: the service refuses a move it could not read, and gives a
    # free tier that only read its connectors a vendor row, so kind alone says nothing.
    if kind in ("moved", "swapped") and moved:
        lines = [f"Your apps moved to {where}: {', '.join(moved)}."]
    elif kind == "conflict":
        lines = [f"{where} already had its own apps, so the free-tier apps stayed behind."]
    elif kind in ("none", "moved", "swapped"):
        lines = ["There were no free-tier apps to move."]
    else:
        lines = ["Any free-tier apps are in this account now."]
    if dropped:
        lines.append(f"Reconnect in {where}: {', '.join(dropped)}.")
    return lines
