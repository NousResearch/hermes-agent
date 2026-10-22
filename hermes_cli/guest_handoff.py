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
_ACTIVE = frozenset({PENDING, RETRYING})
HANDOFF_FILENAME = "guest_handoff.json"
SYNC_BUDGET_SECONDS = 10.0
CLAIM_TIMEOUT_SECONDS = 10.0
MAX_ATTEMPTS = 8
_BACKOFF_SECONDS = (2.0, 5.0, 15.0, 60.0, 300.0, 900.0, 3600.0)
# ``_is_expiring`` treats any token as expiring inside this skew, so the account token refreshes now.
_FORCE_REFRESH_SKEW = 10**9

# Outcome of one claim response.
_CLAIMED, _GONE, _RETRY, _FAIL, _REAUTH = "claimed", "gone", "retry", "fail", "reauth"
# The service's refusal codes; anything unlisted falls back on the HTTP status (``_classify``).
_REFUSALS = {
    "unknown_token": _GONE, "already_retired": _GONE,
    "account_busy": _RETRY, "rate_limited": _RETRY,
    # The anonymous gate: switched off here (404), its breaker tripped or unreadable (503).
    "not_found": _RETRY, "temporarily_disabled": _RETRY, "temporarily_unavailable": _RETRY,
    "no_destination": _FAIL, "unexpected_account_state": _FAIL, "client_not_allowed": _FAIL,
    "account_locked": _FAIL, "invalid_request": _FAIL,
}

_records_lock_holder = threading.local()
_claim_lock_holder = threading.local()
_local = threading.local()
_sleep = time.sleep


@dataclass
class HandoffBatch:
    """What one :func:`drain` did: each record it ran, and the config settle when a replacement on
    this thread asked for one (``anon_auth.settle_after_upgrade``)."""

    results: list[dict[str, Any]] = field(default_factory=list)
    settled: Optional[dict[str, Any]] = None

    def settle(self, account_state: dict[str, Any]) -> dict[str, Any]:
        """The settle this drain ran, else run it now (a sign-in that replaced no guest still moves
        a config the free tier's route left behind)."""
        if self.settled is not None:
            return self.settled
        from hermes_cli.anon_auth import settle_after_upgrade
        return settle_after_upgrade(account_state)

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


def _load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        data = {}
    records = data.get("records") if isinstance(data, dict) else None
    return {"_schema": 1, "records": records if isinstance(records, dict) else {}}


def _save(path: Path, data: dict[str, Any]) -> None:
    from hermes_cli.auth import _save_private_json
    _save_private_json(path, data, sort_keys=True)


def _update(path: Path, key: str, change) -> Optional[dict[str, Any]]:
    """Apply *change(record) -> record* to one record under the records lock (innermost lock: it is
    taken under the auth-store locks by the hook, and never holds another lock itself)."""
    with _lock(path.with_suffix(".lock"), _records_lock_holder, 15.0):
        data = _load(path)
        record = change(data["records"].get(key))
        if record is not None:
            record["updated_at"] = _iso(_now())
            data["records"][key] = record
            _save(path, data)
        return record


def _destination(state: dict[str, Any]) -> Optional[str]:
    from hermes_cli.auth_constants import _decode_jwt_claims
    sub = _decode_jwt_claims(state.get("access_token")).get("sub")
    return str(sub) if sub else None


def _queued() -> set[str]:
    if not hasattr(_local, "queued"):
        _local.queued = set()
    return _local.queued


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
    anon_token = previous.get("anon_token")
    if not isinstance(anon_token, str) or not anon_token or not new.get("access_token"):
        return
    try:
        path = _records_path()
    except RuntimeError:
        return  # no Hermes home yet (pre-setup): there is no shared store either
    home, destination = _home_key(), _destination(new)

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
    _queued().add(home)
    logger.info("Nous sign-in replaced the free tier; handoff %s (%s)", record["state"],
                _token_key(anon_token)[:8])


def note_promoted(anon_token: str, *, account_email: Optional[str]) -> None:
    """The promotion sign-in (``anon_sign_in.run_sign_in``) already moved this identity's apps on the
    service: record the handoff as done so the save that follows sends no claim."""
    if not anon_token:
        return
    home = _home_key()

    def change(record: Optional[dict[str, Any]]) -> dict[str, Any]:
        record = record or {"attempts": 0, "profiles": [], "created_at": _iso(_now())}
        record.pop("anon_token", None)
        record.update(state=CLAIMED, error=None, result={
            "account_email": account_email, "connectors_kind": "moved", "moved_toolkits": [],
            "dropped_toolkits": []})
        if home not in record["profiles"]:
            record["profiles"].append(home)
        return record

    _update(_records_path(), _token_key(anon_token), change)


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
        batch.results, batch.settled = ran.results, ran.settled


def drain() -> Optional[HandoffBatch]:
    """Run this profile's due records now, then settle the config once when a replacement on this
    thread asked for it. None when a lock or a :func:`deferred` block on this thread holds it back."""
    if getattr(_local, "holds", 0) or getattr(_local, "running", False) or _holds_auth_locks():
        return None
    home = _home_key()
    settle = home in _queued()
    _queued().discard(home)
    _local.running = True
    try:
        batch = HandoffBatch(results=run_due(budget_seconds=SYNC_BUDGET_SECONDS))
    finally:
        _local.running = False
    if settle:
        from hermes_cli.anon_auth import is_guest_state, settle_after_upgrade
        from hermes_cli.auth import get_provider_auth_state
        account = get_provider_auth_state("nous")
        if isinstance(account, dict) and not is_guest_state(account):
            batch.settled = settle_after_upgrade(account)
    return batch


def run_due(*, budget_seconds: float = SYNC_BUDGET_SECONDS) -> list[dict[str, Any]]:
    """Attempt every active, due record of this profile. Returns their public views."""
    try:
        path = _records_path()
    except RuntimeError:
        return []
    home, now = _home_key(), _now()
    due = [key for key, record in _load(path)["records"].items()
           if record.get("state") in _ACTIVE and home in record.get("profiles", ())
           and float(record.get("next_attempt_at") or 0) <= now]
    deadline = time.monotonic() + budget_seconds
    return [view for key in due if (view := _run_one(path, key, deadline)) is not None]


def _account_bound_to(record: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The profile's account when it is the record's destination; None while the profile still holds
    a guest (the save never landed) or holds another account (retries bind to the destination)."""
    from hermes_cli.anon_auth import is_guest_state
    from hermes_cli.auth import get_provider_auth_state
    account = get_provider_auth_state("nous")
    if not isinstance(account, dict) or is_guest_state(account) or not account.get("access_token"):
        return None
    destination = record.get("destination")
    return account if not destination or _destination(account) == destination else None


def _run_one(path: Path, key: str, deadline: float) -> Optional[dict[str, Any]]:
    record = _load(path)["records"].get(key)
    if record is None or _account_bound_to(record) is None:
        return None
    # The claim lock spans read -> claim -> write so two processes never claim one record twice.
    # Order: claim lock, then the auth-store locks (token refresh), then the records lock. A lock
    # timeout anywhere in here is contention, not a verdict: the record stays as it is for a later run.
    try:
        with _lock(path.with_name(HANDOFF_FILENAME + ".claim.lock"), _claim_lock_holder, CLAIM_TIMEOUT_SECONDS + 5):
            return _claim_locked(path, key, deadline)
    except TimeoutError:
        return None


def _claim_locked(path: Path, key: str, deadline: float) -> Optional[dict[str, Any]]:
    reauthed = False
    while True:
        record = _load(path)["records"].get(key)
        account = _account_bound_to(record) if record else None
        if record is None or record["state"] not in _ACTIVE or account is None:
            return public_view(record) if record else None
        prior_attempts = int(record.get("attempts") or 0)
        # Counted before the request leaves: an answer lost to a crash still counts as an attempt.
        _update(path, key, lambda r: {**r, "attempts": prior_attempts + 1, "last_attempt_at": _iso(_now())})
        outcome, payload, retry_after = _attempt(record["anon_token"], account, force=reauthed)
        logger.info("Free-tier handoff attempt %d (%s): %s", prior_attempts + 1, key[:8],
                    payload.get("error") or outcome)
        if outcome == _REAUTH and not reauthed:
            # A refused bearer never reached the claim, so it is not an attempt.
            _update(path, key, lambda r: {**r, "attempts": prior_attempts})
            reauthed = True
            continue
        record = _update(path, key, lambda r: _apply(r, outcome, payload, retry_after, prior_attempts))
        if outcome == _CLAIMED and payload.get("cohort_carried") is True:
            _refresh_account_token()
        wait = max(0.0, float(record.get("next_attempt_at") or 0) - _now())
        if record["state"] != RETRYING or time.monotonic() + wait > deadline:
            return public_view(record)
        _sleep(wait)


def _bearer(*, force: bool) -> str:
    from hermes_cli import auth
    if force:
        with auth._RESOLVE_TOKEN_CACHE_LOCK:
            auth._RESOLVE_TOKEN_CACHE.pop(_home_key(), None)
        return auth.resolve_nous_access_token(refresh_skew_seconds=_FORCE_REFRESH_SKEW)
    return auth.resolve_nous_access_token()


def _refresh_account_token() -> None:
    """The account's token picks up the carried cohort only on its next mint, so mint it now."""
    from hermes_cli.auth_constants import AuthError
    try:
        _bearer(force=True)
    except AuthError as exc:
        logger.info("Free-tier handoff: account token refresh after the move failed: %s", exc)


def _attempt(anon_token: str, account: dict[str, Any], *, force: bool) -> tuple[str, dict[str, Any], Optional[float]]:
    from agent.retry_utils import parse_retry_after_seconds
    from hermes_cli.anon_auth import _anon_headers
    from hermes_cli.auth import _nous_portal_base_url, _resolve_verify
    from hermes_cli.auth_constants import AuthError, httpx
    from hermes_cli.auth_nous import _nous_http_client
    try:
        bearer = _bearer(force=force)
    except AuthError as exc:
        return (_RETRY, {"error": str(exc.code or "account_token_unavailable")},
                exc.retry_after if exc.retryable else None)
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
    if status == 401:
        return _REAUTH
    code = str(payload.get("error") or "")
    if code in _REFUSALS:
        return _REFUSALS[code]
    return _RETRY if status in (404, 429) or status >= 500 else _FAIL


def _apply(record: dict[str, Any], outcome: str, payload: dict[str, Any], retry_after: Optional[float],
           prior_attempts: int) -> dict[str, Any]:
    if record["state"] == CLAIMED:
        return record  # never overwritten, whatever a racing attempt heard
    record = {**record, "error": payload.get("error"), "next_attempt_at": None}
    if outcome == _CLAIMED:
        connectors = payload.get("connectors") if isinstance(payload.get("connectors"), dict) else {}
        record.update(state=CLAIMED, error=None, result={
            "account_email": payload.get("account_email"),
            "destination_org_id": payload.get("destination_org_id"),
            "connectors_kind": connectors.get("kind"),
            "moved_toolkits": list(connectors.get("moved_toolkits") or []),
            "dropped_toolkits": list(connectors.get("dropped_toolkits") or []),
            "cohort_carried": payload.get("cohort_carried") is True})
    elif outcome == _GONE:
        # Gone after an earlier attempt is most likely that attempt's lost success.
        record.update(state=GONE, probably_moved=payload.get("error") == "already_retired" or prior_attempts > 0)
    elif outcome == _FAIL or int(record.get("attempts") or 0) >= MAX_ATTEMPTS:
        record.update(state=FAILED, error=payload.get("error") or "retries_exhausted")
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


def handoff_status() -> dict[str, Any]:
    """This profile's latest handoff, or ``{"state": "idle"}``. A pure read (no lock, no network)."""
    try:
        records = _load(_records_path())["records"].values()
    except RuntimeError:
        return {"state": IDLE}
    home = _home_key()
    mine = [r for r in records if home in r.get("profiles", ())]
    if not mine:
        return {"state": IDLE}
    return public_view(max(mine, key=lambda r: str(r.get("updated_at") or "")))


def status_lines(status: dict[str, Any]) -> list[str]:
    """User copy for a non-idle handoff (``hermes auth status nous``)."""
    state = status.get("state")
    where = status.get("account_email") or "your account"
    moved, dropped = status.get("moved_toolkits") or [], status.get("dropped_toolkits") or []
    if state in _ACTIVE:
        return ["Moving your free-tier apps to this account; Hermes keeps trying in the background."]
    if state == FAILED:
        return [f"Your free-tier apps could not be moved ({status.get('error')})."]
    if state == GONE:
        return ["Your free-tier apps were probably moved already." if status.get("probably_moved")
                else "There were no free-tier apps left to move."]
    if state != CLAIMED:
        return []
    kind = status.get("connectors_kind")
    if kind in ("moved", "swapped"):
        lines = [f"Your apps moved to {where}" + (f": {', '.join(moved)}." if moved else ".")]
    elif kind == "conflict":
        lines = [f"{where} already had its own apps, so the free-tier apps stayed behind."]
    else:
        lines = [f"Signed in as {where}. There were no free-tier apps to move."]
    if dropped:
        lines.append(f"Reconnect in {where}: {', '.join(dropped)}.")
    return lines
