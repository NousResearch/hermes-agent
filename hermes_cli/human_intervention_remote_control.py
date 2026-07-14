"""Local pending human-intervention store for CLI remote control.

Tracks pending CLI human-intervention prompts (approval / sudo / clarify) in a
small JSON file so a mobile gateway command can later deny or extend them.

THIS MODULE IS PURE LOGIC — no CLI or gateway wiring lives here.

Store layout
------------
``$HERMES_HOME/runtime/human-interventions.json`` shaped as::

    {"records": {code: {...record...}}}

Concurrency
-----------
Cross-process safety is enforced via exclusive file locking (fcntl.flock on
POSIX, msvcrt.locking on Windows) on a dedicated .lock file. Every
read-modify-write acquires the lock, loads, mutates, and saves atomically.
This handles concurrent CLI processes and gateway commands racing to update
the same intervention record.

Time
----
All time reads go through the module-level :func:`_now` indirection so tests
can monkeypatch it for deterministic expiry / cleanup behaviour.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, field

from hermes_constants import get_hermes_home

# Expiry slack applied to deadlines before a token is considered too late to
# act on (covers clock skew + in-flight gateway round-trips).
GRACE_SECONDS = 30

# Resolved records linger this long before cleanup reaps them.
RESOLVED_TTL_SECONDS = 5 * 60

# Default extension applied when a caller asks to extend without a duration.
DEFAULT_EXTEND_MINUTES = 15

# Length (digits) of an auto-generated typed-confirm approval token.
DEFAULT_APPROVE_TOKEN_LEN = 4

# "deny"/"extend" are the always-available remote actions. "approve" is gated
# on the per-record approve_tier rather than this tuple (it was never a member
# of _REMOTE_ACTIONS and stays opt-in per intervention).
_REMOTE_ACTIONS = ("deny", "extend")


# Cross-process locking via a dedicated .lock file
@contextlib.contextmanager
def _store_lock():
    """Acquire exclusive cross-process lock on the intervention store.

    Uses fcntl.flock on POSIX and msvcrt.locking on Windows. Blocks until
    the lock is acquired. Ensures atomic read-modify-write across concurrent
    CLI and gateway processes.
    """
    lock_path = _store_path().with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # Open lock file in append mode so it's created if missing
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_APPEND | os.O_RDWR, 0o600)
    try:
        if platform.system() == "Windows":
            import msvcrt
            # Lock first byte of the file
            msvcrt.locking(lock_fd, msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

        yield

    finally:
        # Release lock (closing the fd releases flock automatically on POSIX,
        # but explicitly unlock on Windows)
        if platform.system() == "Windows":
            try:
                import msvcrt
                msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
        os.close(lock_fd)


def _now() -> float:
    """Return the current wall-clock time.

    Indirection point so tests can monkeypatch time deterministically.
    """
    return time.time()


@dataclass
class PendingIntervention:
    code: str
    kind: str            # "approval" | "sudo" | "clarify"
    title: str
    preview: str
    session_key: str
    state: str           # "pending" | "denied" | "extended" | "approved" | "expired" | "resolved"
    created_ts: float
    deadline_ts: float
    max_deadline_ts: float
    decision: str | None = None        # "deny" | "extend" | "approve" | None
    decision_ts: float | None = None
    decision_source: str = ""
    # None => both deny and extend allowed. Otherwise only listed actions are.
    allowed_actions: list[str] | None = None
    # Tiered remote approval. risk_level is advisory metadata; approve_tier
    # gates whether approve can be driven remotely and how.
    risk_level: str = ""
    approve_tier: str = "none"          # "none" | "one_tap" | "typed_confirm"
    approve_token: str = ""             # set only for the typed_confirm tier

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PendingIntervention":
        # Tolerate extra/legacy keys by filtering to known fields.
        known = cls.__dataclass_fields__  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _store_path():
    runtime_dir = get_hermes_home() / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir / "human-interventions.json"


def _load_records() -> dict:
    """Return the raw ``{code: record_dict}`` mapping.

    A missing or corrupt file is treated as an empty store.
    """
    path = _store_path()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    records = data.get("records")
    if not isinstance(records, dict):
        return {}
    return records


def _save_records(records: dict) -> None:
    """Atomically persist ``records`` via write-temp-then-replace."""
    path = _store_path()
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{random.randint(0, 1_000_000)}")
    payload = {"records": records}
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _generate_code(existing: dict) -> str:
    """Return a 4-digit code not colliding with existing pending records."""
    for _ in range(10_000):
        code = f"{random.randint(0, 9999):04d}"
        if code not in existing:
            return code
    # Astronomically unlikely; fall back to a unique-ish value.
    return f"{random.randint(0, 9999):04d}"


def _generate_token(n: int = DEFAULT_APPROVE_TOKEN_LEN, *, avoid: str = "") -> str:
    """Return a zero-padded numeric token of ``n`` digits.

    Distinct from ``avoid`` (typically the record's code) so a single number
    can't satisfy both the addressing code and the typed-confirm gate.
    """
    n = max(1, int(n))
    high = 10**n - 1
    for _ in range(10_000):
        token = f"{random.randint(0, high):0{n}d}"
        if token != avoid:
            return token
    # Astronomically unlikely; return whatever we have.
    return f"{random.randint(0, high):0{n}d}"


def _finite_number(value) -> float | None:
    """Return ``value`` as a finite float, rejecting bools/strings/NaN/inf."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _record_from_raw(raw: dict) -> PendingIntervention | None:
    """Deserialize and validate one record defensively.

    Gateway commands and CLI poll loops must never crash just because the JSON
    store was truncated, hand-edited, or written by an older schema. Treat a
    malformed per-code record as absent; cleanup will later reap it if it is
    still present.
    """
    try:
        if not isinstance(raw, dict):
            return None

        required = {
            "code", "kind", "title", "preview", "session_key", "state",
            "created_ts", "deadline_ts", "max_deadline_ts",
        }
        if not required.issubset(raw):
            return None

        normalized = dict(raw)
        for key in ("created_ts", "deadline_ts", "max_deadline_ts"):
            value = _finite_number(normalized.get(key))
            if value is None:
                return None
            normalized[key] = value

        if normalized.get("decision_ts") is not None:
            decision_ts = _finite_number(normalized.get("decision_ts"))
            if decision_ts is None:
                return None
            normalized["decision_ts"] = decision_ts

        for key in ("code", "kind", "title", "preview", "session_key", "state"):
            if not isinstance(normalized.get(key), str):
                return None

        if normalized["state"] not in {
            "pending", "denied", "extended", "approved", "expired", "resolved",
        }:
            return None

        decision = normalized.get("decision")
        if decision is not None and decision not in {"deny", "extend", "approve"}:
            return None

        allowed_actions = normalized.get("allowed_actions")
        if allowed_actions is not None:
            if not isinstance(allowed_actions, list) or not all(
                isinstance(item, str) for item in allowed_actions
            ):
                return None

        for key in ("decision_source", "risk_level", "approve_tier", "approve_token"):
            if key in normalized and not isinstance(normalized.get(key), str):
                return None
        if normalized.get("approve_tier", "none") not in {"", "none", "one_tap", "typed_confirm"}:
            return None

        return PendingIntervention.from_dict(normalized)
    except (AttributeError, TypeError, ValueError):
        return None


def _is_past_action_deadline(rec: PendingIntervention, now: float) -> bool:
    """Return True when no remote action may be applied anymore.

    Both deadlines are enforced at the store boundary: the current prompt
    deadline and the absolute max-total-wait guard. This keeps the cap
    server-side even if a stale/tampered record carries an overlong
    ``deadline_ts``.
    """
    return (
        now > rec.deadline_ts + GRACE_SECONDS
        or now > rec.max_deadline_ts + GRACE_SECONDS
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_pending_intervention(
    *,
    kind: str,
    title: str,
    preview: str,
    session_key: str,
    timeout_seconds: int,
    max_total_wait_minutes: int = 15,
    code: str | None = None,
    allowed_actions: list[str] | None = None,
    risk_level: str = "",
    approve_tier: str = "none",
    approve_token: str | None = None,
    approve_token_len: int = DEFAULT_APPROVE_TOKEN_LEN,
) -> PendingIntervention:
    """Create and persist a new pending intervention, returning the record.

    Tiered remote approval is opt-in per record via ``approve_tier``:
      * ``"none"``         — approve cannot be driven remotely (default).
      * ``"one_tap"``      — a remote approve needs no token.
      * ``"typed_confirm"``— a remote approve must echo ``approve_token``.
                             When no token is supplied one is generated
                             (``approve_token_len`` digits, distinct from the
                             addressing ``code``).

    Args:
        approve_token_len: Number of digits for auto-generated typed-confirm
            tokens (default: DEFAULT_APPROVE_TOKEN_LEN). Wire from config's
            remote_control.approve_token_len.
    """
    with _store_lock():
        records = _load_records()
        if code is None:
            code = _generate_code(records)
        elif code in records:
            # Explicit code collision: overwrite is the caller's intent in
            # tests, but regenerate uniqueness only applies to auto codes.
            pass

        # Resolve the approve token according to the tier.
        if approve_tier == "typed_confirm":
            token = approve_token if approve_token is not None else _generate_token(
                approve_token_len, avoid=code
            )
        else:
            token = ""

        now = _now()
        timeout_seconds = max(0, int(timeout_seconds))
        max_wait_seconds = max(0, int(max_total_wait_minutes) * 60)
        max_deadline_ts = now + max_wait_seconds
        deadline_ts = now + min(timeout_seconds, max_wait_seconds)
        rec = PendingIntervention(
            code=code,
            kind=kind,
            title=title,
            preview=preview,
            session_key=session_key,
            state="pending",
            created_ts=now,
            deadline_ts=deadline_ts,
            max_deadline_ts=max_deadline_ts,
            decision=None,
            decision_ts=None,
            decision_source="",
            allowed_actions=list(allowed_actions) if allowed_actions is not None else None,
            risk_level=risk_level,
            approve_tier=approve_tier,
            approve_token=token,
        )
        records[code] = rec.to_dict()
        _save_records(records)
        return rec


def get_pending_intervention(code: str) -> PendingIntervention | None:
    """Return the record for ``code`` or ``None``. Does not mutate state."""
    with _store_lock():
        records = _load_records()
        raw = records.get(code)
        if raw is None:
            return None
        return _record_from_raw(raw)


def _action_allowed(rec: PendingIntervention, action: str) -> bool:
    if action == "approve":
        return False
    if action not in _REMOTE_ACTIONS:
        return False
    if rec.allowed_actions is None:
        return True
    return action in rec.allowed_actions


def set_remote_decision(
    code: str,
    action: str,
    *,
    minutes: int | None = None,
    token: str | None = None,
    source: str = "",
    max_extend_minutes: int | None = None,
) -> tuple[bool, str, PendingIntervention | None]:
    """Apply a remote ``deny``/``extend``/``approve`` decision.

    Returns ``(ok, reason, record_or_None)``. On failure ``reason`` is one of
    ``not_found`` / ``expired`` / ``action_not_allowed`` / ``already_resolved``
    / ``approve_not_allowed`` / ``bad_token`` / ``invalid_minutes`` /
    / ``exceeds_max_extend``.

    ``deny`` and ``extend`` keep the phase-1 ``_action_allowed`` gate.
    ``approve`` is gated separately on the record's ``approve_tier`` (it is
    never a member of ``_REMOTE_ACTIONS``): the ``one_tap`` tier needs no
    token, while ``typed_confirm`` requires ``token`` to match the stored
    ``approve_token``.

    Args:
        max_extend_minutes: Maximum allowed extension in minutes. Wire from
            config's remote_control.max_extend_minutes. When None, no cap is
            enforced beyond the record's max_deadline_ts.
    """
    with _store_lock():
        records = _load_records()
        raw = records.get(code)
        if raw is None:
            return (False, "not_found", None)

        rec = _record_from_raw(raw)
        if rec is None:
            records.pop(code, None)
            _save_records(records)
            return (False, "not_found", None)

        if action == "approve":
            # Tier gate — approve is opt-in per record, not via _action_allowed.
            tier = getattr(rec, "approve_tier", "none") or "none"
            if tier == "none":
                return (False, "approve_not_allowed", rec)

            if rec.state == "resolved":
                return (False, "already_resolved", rec)

            now = _now()
            if _is_past_action_deadline(rec, now):
                return (False, "expired", rec)

            if tier == "typed_confirm":
                expected = rec.approve_token or ""
                if not token or token != expected:
                    return (False, "bad_token", rec)

            rec.state = "approved"
            rec.decision = "approve"
            rec.decision_ts = now
            rec.decision_source = source
            records[code] = rec.to_dict()
            _save_records(records)
            return (True, "ok", rec)

        if not _action_allowed(rec, action):
            return (False, "action_not_allowed", rec)

        if rec.state == "resolved":
            return (False, "already_resolved", rec)

        now = _now()
        if _is_past_action_deadline(rec, now):
            return (False, "expired", rec)

        reason = "ok"
        if action == "deny":
            rec.state = "denied"
            rec.decision = "deny"
            rec.decision_ts = now
            rec.decision_source = source
        elif action == "extend":
            mins = DEFAULT_EXTEND_MINUTES if minutes is None else minutes
            try:
                mins = int(mins)
            except (TypeError, ValueError):
                return (False, "invalid_minutes", rec)
            if mins <= 0:
                return (False, "invalid_minutes", rec)
            # Enforce max_extend_minutes cap if provided (wired from config)
            if max_extend_minutes is not None and mins > max_extend_minutes:
                return (False, "exceeds_max_extend", rec)
            new_deadline = rec.deadline_ts + mins * 60
            if new_deadline >= rec.max_deadline_ts:
                new_deadline = rec.max_deadline_ts
                reason = "clamped"
            rec.deadline_ts = new_deadline
            rec.state = "extended"
            rec.decision = "extend"
            rec.decision_ts = now
            rec.decision_source = source
        else:  # pragma: no cover - guarded by _action_allowed
            return (False, "action_not_allowed", rec)

        records[code] = rec.to_dict()
        _save_records(records)
        return (True, reason, rec)


def consume_remote_decision(code: str) -> PendingIntervention | None:
    """Consume a pending decision once, for the CLI poll loop.

    Returns a snapshot reflecting the decision that was just consumed, or
    ``None`` when there is nothing actionable (no record, or still pending).

    Consumption semantics:
      * ``denied``   -> snapshot returned, stored record becomes ``resolved``
                        (a deny is terminal; further consumes return ``None``).
      * ``approved`` -> snapshot returned (decision ``approve``), stored record
                        becomes ``resolved`` (approve is one-shot/terminal;
                        further consumes return ``None``).
      * ``extended`` -> snapshot returned (with the already-extended deadline),
                        stored record reset to ``pending`` with ``decision``
                        cleared so the CLI can apply the new deadline and keep
                        waiting. Repeated extends are therefore each consumed
                        exactly once.
    """
    with _store_lock():
        records = _load_records()
        raw = records.get(code)
        if raw is None:
            return None

        rec = _record_from_raw(raw)
        if rec is None:
            records.pop(code, None)
            _save_records(records)
            return None

        if _is_past_action_deadline(rec, _now()):
            rec.state = "expired"
            records[code] = rec.to_dict()
            _save_records(records)
            return None

        if rec.state in ("denied", "approved"):
            snapshot = PendingIntervention.from_dict(rec.to_dict())
            rec.state = "resolved"
            records[code] = rec.to_dict()
            _save_records(records)
            return snapshot

        if rec.state == "extended":
            # Treat the absolute max deadline as a store-side invariant even
            # for stale/tampered records. The CLI consumes this wall-clock
            # value directly, so never hand back an overlong deadline.
            if rec.deadline_ts > rec.max_deadline_ts:
                rec.deadline_ts = rec.max_deadline_ts
            snapshot = PendingIntervention.from_dict(rec.to_dict())
            rec.state = "pending"
            rec.decision = None
            rec.decision_ts = None
            # decision_source retained for audit; deadline already extended.
            records[code] = rec.to_dict()
            _save_records(records)
            return snapshot

        # pending / resolved / expired => nothing to hand back.
        return None


def cleanup_expired() -> int:
    """Drop records past their max deadline (plus grace) or stale-resolved.

    Returns the number of records removed.
    """
    with _store_lock():
        records = _load_records()
        now = _now()
        to_remove = []
        for code, raw in records.items():
            rec = _record_from_raw(raw)
            if rec is None:
                to_remove.append(code)
                continue
            if now > rec.max_deadline_ts + GRACE_SECONDS:
                to_remove.append(code)
            elif rec.state == "resolved" and rec.decision_ts is not None \
                    and now > rec.decision_ts + RESOLVED_TTL_SECONDS:
                to_remove.append(code)
        for code in to_remove:
            records.pop(code, None)
        if to_remove:
            _save_records(records)
        return len(to_remove)
