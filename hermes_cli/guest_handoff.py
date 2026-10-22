"""Guest handoff: after a Nous account replaces the free tier, move the free tier's apps into it.

The account service moves a free-tier identity's connected apps (and its preview cohort) into a
signed-in account with one call, ``POST /api/anonymous/claim {"token": "<anon_...>"}``, authorized by
the account's own ``hermes-cli`` Bearer token. Hermes never asks first: the move just happens after
the account is saved, and the result (the account email, what moved, what needs reconnecting) is
shown afterwards. Sign-in code knows nothing about this; two halves keep it that way:

- :func:`note_replacement` runs at the lowest write seams (``auth._store_provider_state`` and
  ``auth_nous._write_shared_nous_state``), under the caller's auth-store locks, whenever a non-guest
  Nous state replaces a guest. It writes a pending record and does no network. Records sit beside
  the shared Nous store, keyed by a hash of the ``anon_`` credential and bound to the destination
  account, so profiles that share one free-tier identity share one record and one claim.
- :func:`drain` runs the claim outside every lock: when a shared-store write finishes, when a
  :func:`deferred` block exits, after a free-tier reconcile, and on each Nous keepalive tick (a
  retry left by an earlier process). The synchronous run is bounded (``SYNC_BUDGET_SECONDS``) so a
  CLI or the desktop sees a result; what is still retrying resumes on a later run.

The apps only ever go to the record's destination: every attempt is reserved in one locked
read-modify-write that also proves the Bearer about to be sent belongs to that destination.

User copy never says guest / anonymous / claim.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger("hermes_cli.auth")

IDLE, PENDING, RETRYING, CLAIMED, GONE, FAILED = "idle", "pending", "retrying", "claimed", "gone", "failed"
# Read side only: an active record bound to an account this profile no longer holds.
PAUSED = "paused"
_ACTIVE = frozenset({PENDING, RETRYING})
HANDOFF_FILENAME = "guest_handoff.json"
SYNC_BUDGET_SECONDS = 10.0
CLAIM_TIMEOUT_SECONDS = 10.0
# The service reaps an idle free-tier identity after 14 days, apps included. Transient failures keep
# retrying (the slowest rung repeats) until this long after the record was made, safely before that.
RETRY_WINDOW_SECONDS = 13 * 24 * 3600
_BACKOFF_SECONDS = (2.0, 5.0, 15.0, 60.0, 300.0, 900.0, 3600.0)
# ``_is_expiring`` treats any token as expiring inside this skew, so the account token refreshes now.
_FORCE_REFRESH_SKEW = 10**9

# Outcome of one claim response.
_CLAIMED, _GONE, _RETRY, _FAIL, _REAUTH = "claimed", "gone", "retry", "fail", "reauth"
# Codes no later attempt can change. Every other refusal, unknown codes included, retries: a wrong
# "final" loses the apps for good, a wrong retry costs one request an hour.
_FINAL_CODES = frozenset({"client_not_allowed", "account_locked", "invalid_request", "retry_window_ended"})
# A plain reason per final code; any other code shows only the generic sentence, never the code.
_FAILURE_COPY = {
    "client_not_allowed": "This sign-in can't receive them. Sign in again with `hermes auth add nous`.",
    "account_locked": "This Nous account is locked.",
    "retry_window_ended": "Hermes kept trying for nearly two weeks, then stopped.",
}
# The service's refusal codes; anything unlisted retries (``_classify``).
_REFUSALS = {
    "unknown_token": _GONE, "already_retired": _GONE,
    **{code: _FAIL for code in _FINAL_CODES},
}

_records_lock_holder = threading.local()
_claim_lock_holder = threading.local()
_local = threading.local()
_sleep = time.sleep


@dataclass
class HandoffBatch:
    """What one :func:`drain` did: the public view of each record it ran."""

    results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def account_email(self) -> Optional[str]:
        return next((r["account_email"] for r in self.results if r.get("account_email")), None)


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


@contextmanager
def _lock(path: Path, holder: threading.local, timeout_seconds: float) -> Iterator[None]:
    from hermes_cli.auth import _file_lock
    with _file_lock(path, holder, timeout_seconds, f"Timed out waiting for {path.name}"):
        yield


def _load(path: Path, *, heal: bool = False) -> dict[str, Any]:
    """The records file. An unparseable one reads as empty; *heal* (writers, under the records lock)
    moves it aside to ``.corrupt`` first, so a bad file neither blocks a sign-in nor loses its data."""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        data = {}
    except ValueError:
        logger.warning("Free-tier handoff records at %s are unreadable%s", path,
                       "; moved aside to .corrupt" if heal else "")
        if heal:
            path.replace(path.with_suffix(".json.corrupt"))
        data = {}
    records = data.get("records") if isinstance(data, dict) else None
    return {"_schema": 1, "records": records if isinstance(records, dict) else {}}


def _save(path: Path, data: dict[str, Any]) -> None:
    from hermes_cli.auth import _save_private_json
    _save_private_json(path, data, sort_keys=True)


def _update(path: Path, key: str, change) -> Optional[dict[str, Any]]:
    """Apply *change(record) -> record | None* to one record under the records lock and return what
    was written (None: nothing written). The innermost lock: the hook takes it under the auth-store
    locks, and it never holds another lock itself."""
    with _lock(path.with_suffix(".lock"), _records_lock_holder, 15.0):
        data = _load(path, heal=True)
        record = change(data["records"].get(key))
        if record is not None:
            record["updated_at"] = _iso(_now())
            data["records"][key] = record
            _save(path, data)
        return record


def _subject(access_token: Any) -> Optional[str]:
    from hermes_cli.auth_constants import _decode_jwt_claims
    sub = _decode_jwt_claims(access_token).get("sub")
    return str(sub) if sub else None


# --- The hook: no network, runs under the writer's locks ------------------------------------------

def note_store_write(auth_store: dict[str, Any], provider_id: str, state: Any) -> None:
    """``auth._store_provider_state``'s hook: the Nous state the store holds now is the one replaced
    (the profile's own, else the root store's it falls back to)."""
    if provider_id == "nous":
        from hermes_cli.auth import _load_provider_state
        note_replacement(_load_provider_state(auth_store, "nous"), state)


def note_replacement(previous: Any, new: Any) -> None:
    """Record that a non-guest Nous state replaced the guest *previous*. Called from the lowest write
    seams; idempotent per ``anon_`` credential. A Claimed or Gone record is never reopened; a record
    not yet attempted follows the newest destination, one already attempted keeps its own (a lost
    answer may have moved the apps there)."""
    from hermes_cli.anon_auth import is_guest_state
    if not is_guest_state(previous) or is_guest_state(new) or not isinstance(new, dict):
        return
    anon_token, destination = previous.get("anon_token"), _subject(new.get("access_token"))
    if not isinstance(anon_token, str) or not anon_token or not destination:
        return  # no account subject: nothing a claim could ever be proven to land in
    try:
        path = _records_path()
    except RuntimeError:
        return  # no Hermes home yet (pre-setup): there is no shared store either
    home = _home_key()

    def change(record: Optional[dict[str, Any]]) -> dict[str, Any]:
        if record is None:
            record = {"state": PENDING, "anon_token": anon_token, "destination": destination,
                      "attempts": 0, "profiles": [], "created_at": _iso(_now())}
        elif record["state"] in _ACTIVE and not record.get("attempts"):
            record["destination"] = destination
        if home not in record["profiles"]:
            record["profiles"].append(home)
        return record

    record = _update(path, _token_key(anon_token), change)
    logger.info("Nous sign-in replaced the free tier; handoff %s (%s)", record["state"],
                _token_key(anon_token)[:8])


# --- The runner: outside every lock ---------------------------------------------------------------

def _holds_auth_locks() -> bool:
    from hermes_cli import auth, auth_nous
    if getattr(auth_nous._nous_shared_lock_holder, "depth", 0):
        return True
    with auth._auth_target_lock_holders_guard:
        holders = list(auth._auth_target_lock_holders.values())
    return any(getattr(holder, "depth", 0) for holder in holders)


@contextmanager
def deferred() -> Iterator[HandoffBatch]:
    """Hold every drain on this thread until the block exits, then drain once (outside the caller's
    lock): a surface that saves a sign-in under its own lock wraps the save in this."""
    batch = HandoffBatch()
    _local.holds = getattr(_local, "holds", 0) + 1
    try:
        yield batch
    finally:
        _local.holds -= 1
    ran = drain()
    if ran is not None:
        batch.results = ran.results


def drain() -> Optional[HandoffBatch]:
    """Run this profile's due records now. None when a lock or a :func:`deferred` block on this
    thread holds it back."""
    if getattr(_local, "holds", 0) or getattr(_local, "running", False) or _holds_auth_locks():
        return None
    _local.running = True
    try:
        return HandoffBatch(results=run_due(budget_seconds=SYNC_BUDGET_SECONDS))
    finally:
        _local.running = False


def _remaining(deadline: float) -> float:
    return deadline - time.monotonic()


def _retryable(record: dict[str, Any]) -> bool:
    """A record a run may still attempt: active, or Failed on a code that is not final (an older
    build closed transient failures after 8 attempts; those re-arm here, on any run: a sign-in,
    ``hermes auth status``, ``hermes auth upgrade``, the retry worker). The retry window is checked
    when the attempt is reserved, so a record past it is closed, not silently skipped."""
    state = record.get("state")
    return state in _ACTIVE or (state == FAILED and record.get("error") not in _FINAL_CODES
                                and bool(record.get("anon_token")))


def _due(record: dict[str, Any], now: float) -> bool:
    return _retryable(record) and float(record.get("next_attempt_at") or 0) <= now


def _window_ended(record: dict[str, Any], now: float) -> bool:
    try:
        created = datetime.fromisoformat(str(record.get("created_at"))).timestamp()
    except ValueError:
        return True
    return now >= created + RETRY_WINDOW_SECONDS


# --- The worker: retries in a long-lived process, for every profile -------------------------------
#
# A move queued by a short process (a CLI sign-in during an outage) is finished by whichever long-lived
# Nous process runs next (Desktop backend, gateway, CLI agent), whatever its own profile or identity.
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
        # health: allow HX012 -- process-wide worker that binds each record's own profile scope per run; it must not inherit the starter's
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
    wait = _WORKER_FIRST_TICK_SECONDS
    while not _worker_stop.wait(wait):
        try:
            retry_all_profiles()
        except Exception:
            logger.warning("Free-tier handoff retry tick failed; the records stay for the next one", exc_info=True)
        wait = _WORKER_INTERVAL_SECONDS


def retry_all_profiles() -> list[dict[str, Any]]:
    """Run every due record under its owning profile's scope (home + secrets), so a named profile's
    move retries from a process launched as another profile. The bound-account check and the Bearer
    proof in ``_reserve`` still decide whether anything is sent."""
    try:
        records = _load(_records_path())["records"].values()
    except RuntimeError:
        return []
    now = _now()
    homes = sorted({home for record in records if _due(record, now) for home in record.get("profiles", ())})
    results: list[dict[str, Any]] = []
    for home in homes:
        if not Path(home).is_dir():
            continue  # a deleted profile: nothing there holds the destination account
        with _profile_scope(home):
            batch = drain()
        results.extend(batch.results if batch else [])
    return results


@contextmanager
def _profile_scope(home: str) -> Iterator[None]:
    """The owning profile's home and secrets for one drain (the cron ticker binds a profile the same
    way); no terminal scope: nothing here runs a command."""
    from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
    from hermes_constants import get_routing_process_hermes_home, hermes_home_key
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    foreign = hermes_home_key(get_routing_process_hermes_home()) != home
    home_token = set_hermes_home_override(home)
    secret_token = set_secret_scope(build_profile_secret_scope(Path(home)), profile_home=home if foreign else None)
    try:
        yield
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def run_due(*, budget_seconds: float = SYNC_BUDGET_SECONDS) -> list[dict[str, Any]]:
    """Attempt every active, due record of this profile within *budget_seconds*. Returns their public
    views; a record the budget did not reach stays as it is for a later run."""
    try:
        path = _records_path()
    except RuntimeError:
        return []
    home, now = _home_key(), _now()
    due = [key for key, record in _load(path)["records"].items()
           if home in record.get("profiles", ()) and _due(record, now)]
    deadline = time.monotonic() + budget_seconds
    results = []
    for key in due:
        if _remaining(deadline) <= 0:
            break
        if (view := _run_one(path, key, deadline)) is not None:
            results.append(view)
    return results


def _account_bound_to(record: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The profile's account when it is the record's destination; None while the profile still holds
    a guest (the save never landed) or holds another account (retries bind to the destination)."""
    from hermes_cli.anon_auth import is_guest_state
    from hermes_cli.auth import get_provider_auth_state
    account = get_provider_auth_state("nous")
    if not isinstance(account, dict) or is_guest_state(account) or not account.get("access_token"):
        return None
    destination = record.get("destination")
    return account if destination and _subject(account["access_token"]) == destination else None


def _run_one(path: Path, key: str, deadline: float) -> Optional[dict[str, Any]]:
    record = _load(path)["records"].get(key)
    if record is None or _account_bound_to(record) is None:
        return None  # a cheap skip; the binding that counts is proven in ``_reserve``
    # The claim lock spans read -> claim -> write so two processes never claim one record twice.
    # Order: claim lock, then the auth-store locks (token refresh), then the records lock. A lock
    # timeout anywhere in here is contention, not a verdict: the record stays as it is for a later run.
    try:
        with _lock(path.with_name(HANDOFF_FILENAME + ".claim.lock"), _claim_lock_holder,
                   max(0.1, _remaining(deadline))):
            return _claim_locked(path, key, deadline)
    except TimeoutError:
        return None


def _reserve(path: Path, key: str, bearer_subject: Optional[str]) -> Optional[tuple[dict[str, Any], int]]:
    """Reserve one attempt in a single locked read-modify-write: the record is still retryable and
    due, inside its retry window, and bound to the very account whose Bearer is about to be sent.
    Returns ``(record, attempts before this one)``, or None when nothing may be sent now. A record
    past its window is closed as Failed instead; one an older build failed is re-armed."""
    reserved: list[int] = []

    def change(record: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        now = _now()
        if (record is None or not _due(record, now)
                or not bearer_subject or record.get("destination") != bearer_subject):
            return None
        if _window_ended(record, now):
            return {**record, "state": FAILED, "error": "retry_window_ended"}
        prior = int(record.get("attempts") or 0)
        reserved.append(prior)
        # Counted before the request leaves: an answer lost to a crash still counts as an attempt.
        return {**record, "state": RETRYING if record["state"] == FAILED else record["state"],
                "attempts": prior + 1, "last_attempt_at": _iso(now)}

    record = _update(path, key, change)
    return (record, reserved[0]) if reserved else None


def _claim_locked(path: Path, key: str, deadline: float) -> Optional[dict[str, Any]]:
    from hermes_cli.auth import get_provider_auth_state
    from hermes_cli.auth_constants import AuthError
    force = False
    while True:
        current = _load(path)["records"].get(key)
        if current is None or _remaining(deadline) <= 0:
            return public_view(current) if current else None
        try:
            bearer = _bearer(force=force, timeout=_remaining(deadline))
        except AuthError as exc:
            logger.info("Free-tier handoff (%s): account token unavailable (%s)", key[:8], exc.code)
            return public_view(current)
        if _remaining(deadline) <= 0:
            return public_view(current)
        reservation = _reserve(path, key, _subject(bearer))
        if reservation is None:
            return public_view(_load(path)["records"].get(key) or current)
        record, prior = reservation
        outcome, payload, retry_after = _post(
            record["anon_token"], bearer, get_provider_auth_state("nous") or {},
            timeout=min(CLAIM_TIMEOUT_SECONDS, max(0.1, _remaining(deadline))))
        logger.info("Free-tier handoff attempt %d (%s): %s", prior + 1, key[:8], payload.get("error") or outcome)
        if outcome == _REAUTH and not force:
            # A refused bearer never reached the claim, so it is not an attempt.
            _update(path, key, lambda r: {**r, "attempts": prior} if r.get("attempts") == prior + 1 else None)
            force = True
            continue
        record = _update(path, key, lambda r: _apply(r, outcome, payload, retry_after, prior))
        if outcome == _CLAIMED and payload.get("cohort_carried") is True:
            _refresh_account_token(timeout=max(1.0, _remaining(deadline)))
        wait = max(0.0, float(record.get("next_attempt_at") or 0) - _now())
        if record["state"] != RETRYING or wait >= _remaining(deadline):
            return public_view(record)
        _sleep(wait)


def _bearer(*, force: bool, timeout: float) -> str:
    from hermes_cli import auth
    from hermes_cli.auth_nous import _nous_shared_store_lock
    # The resolver's 5 s memo can still hold the guest's token from just before the save.
    with auth._RESOLVE_TOKEN_CACHE_LOCK:
        auth._RESOLVE_TOKEN_CACHE.pop(_home_key(), None)
    # The resolver waits its own, longer floors for these locks. Taking them first within the
    # budget (same order: auth, then shared) makes its acquisitions reentrant, so a held lock costs
    # the budget, not 15 s; a timeout leaves the record for a later run.
    with auth._auth_store_lock(timeout), _nous_shared_store_lock(timeout_seconds=timeout):
        if force:
            return auth.resolve_nous_access_token(timeout_seconds=timeout, refresh_skew_seconds=_FORCE_REFRESH_SKEW)
        return auth.resolve_nous_access_token(timeout_seconds=timeout)


def _refresh_account_token(*, timeout: float) -> None:
    """The account's token picks up the carried cohort only on its next mint, so mint it now."""
    from hermes_cli.auth_constants import AuthError
    try:
        _bearer(force=True, timeout=timeout)
    except AuthError as exc:
        logger.info("Free-tier handoff: account token refresh after the move failed: %s", exc)


def _post(anon_token: str, bearer: str, account: dict[str, Any], *,
          timeout: float) -> tuple[str, dict[str, Any], Optional[float]]:
    from agent.retry_utils import parse_retry_after_seconds
    from hermes_cli.anon_auth import _anon_headers
    from hermes_cli.auth import _nous_portal_base_url, _resolve_verify
    from hermes_cli.auth_constants import httpx
    from hermes_cli.auth_nous import _nous_http_client
    verify = _resolve_verify(insecure=None, ca_bundle=None, auth_state=account)
    try:
        with _nous_http_client(timeout, verify) as client:
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
    if status == 401:
        return _REAUTH
    return _REFUSALS.get(str(payload.get("error") or ""), _RETRY)


def _apply(record: dict[str, Any], outcome: str, payload: dict[str, Any], retry_after: Optional[float],
           prior_attempts: int) -> dict[str, Any]:
    if record["state"] == CLAIMED:
        return record  # never overwritten, whatever a racing attempt heard
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
    elif outcome == _FAIL:
        record.update(state=FAILED)
    else:
        ladder = _BACKOFF_SECONDS[min(prior_attempts, len(_BACKOFF_SECONDS) - 1)]
        record.update(state=RETRYING, next_attempt_at=_now() + (retry_after if retry_after else ladder))
    if record["state"] in (CLAIMED, GONE):
        record.pop("anon_token", None)  # retired credential: nothing left to retry with
    return record


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


def _status_of(record: dict[str, Any]) -> dict[str, Any]:
    view = public_view(record)
    if _retryable(record):
        view["state"] = RETRYING if _account_bound_to(record) else PAUSED
    return view


def handoff_status() -> dict[str, Any]:
    """This profile's latest handoff, or ``{"state": "idle"}``. A pure read (no lock, no network). An
    active move bound to an account this profile no longer holds reads as ``paused``."""
    try:
        records = _load(_records_path())["records"].values()
    except RuntimeError:
        return {"state": IDLE}
    home = _home_key()
    mine = [r for r in records if home in r.get("profiles", ())]
    if not mine:
        return {"state": IDLE}
    return _status_of(max(mine, key=lambda r: str(r.get("updated_at") or "")))


def record_status(anon_token: str) -> Optional[dict[str, Any]]:
    """The handoff of one free-tier credential (as :func:`handoff_status` reads it), or None when no
    record exists. A pure read: what a sign-in reports when its own run attempted nothing (another
    process held the claim, or the budget ran out first)."""
    try:
        record = _load(_records_path())["records"].get(_token_key(anon_token))
    except RuntimeError:
        return None
    return _status_of(record) if record else None


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
    state = status.get("state")
    where = status.get("account_email") or "your account"
    moved, dropped = status.get("moved_toolkits") or [], status.get("dropped_toolkits") or []
    if state in _ACTIVE:
        return ["Moving your free-tier apps to this account; Hermes keeps trying in the background."]
    if state == PAUSED:
        return ["Your free-tier apps are set to move to another Nous account. Sign in to that account "
                "to finish moving them."]
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
