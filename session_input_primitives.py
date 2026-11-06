"""Admission-time batching primitives for session input.

A plugin (a *provider*) decides **whether** input may be batched, the waiting
windows, the text a batch renders as, and whether a running turn should take the
input early. This module owns the parts that must outlive the plugin:

* the versioned batch record inside the admission payload, committed in the same
  transaction as the retry alias (never a half-visible fold),
* the deadline the claim path honors, so waiting happens outside the write lock,
* fail-open provider calls: a broken or unloaded provider only means "no batch".

With no provider registered every function here is a no-op, so admission,
claiming and settlement behave exactly as they do upstream.

Provider protocol (all hooks are optional; an exception disables that hook):

``policy(profile_id, payload) -> mapping | None``
    ``{"eligible": bool, "first_window_seconds": float, "quiet_seconds": float,
    "hard_cap_seconds": float, "signature": str}``. ``signature`` is any stable
    token: only payloads whose signature matches may share one batch. Unknown
    keys, out-of-range numbers and non-``eligible`` snapshots mean "do not batch".
``render(parts) -> str``
    Text a batch is committed with. ``parts`` are ``{"at": epoch, "text": str}``.
``format_part(text, *, count, arrived) -> str``
    Presentation of one ticket appended to a live turn.
``admitted(snapshot) -> bool``
    Called after the admission commit: should this admitted batch be offered to
    the running generation now? Never called with a transaction open.
``transfer_mode(phase) -> "steer" | "redirect" | "queue"``
    How to hand it over; ``queue`` leaves it in the FIFO.

The provider never receives a SQLite connection, a live agent, or a lock.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Mapping

log = logging.getLogger("session_input_primitives")

# Payload key new batches are written under.
META = "session_input_batch_v1"
# Retry alias key prefix new aliases are written under.
ALIAS = "session.input.alias.v1."

_MAX_WINDOW = 60.0
_MAX_SIGNATURE = 2048

_PROVIDER: Any = None
_LOCK = threading.Lock()

# Reading records written before this module existed is a deployment concern, so
# core names no legacy key itself: a deployment that migrated older rows
# registers the names it wrote, and new writes never use them.
_LEGACY_BATCH_KEYS: tuple[str, ...] = ()
_LEGACY_ALIAS_PREFIXES: tuple[str, ...] = ()


def register_legacy_names(*, batch_keys=(), alias_prefixes=()) -> None:
    """Opt in to reading pre-existing batch keys / retry alias prefixes.

    Both accept an iterable of names, appended in the order given (oldest first)
    and de-duplicated. Unregistered names are simply not read; nothing else
    changes. Call :func:`clear_legacy_names` to go back to the core defaults.
    """
    global _LEGACY_BATCH_KEYS, _LEGACY_ALIAS_PREFIXES
    with _LOCK:
        _LEGACY_BATCH_KEYS = tuple(dict.fromkeys(
            (*_LEGACY_BATCH_KEYS, *[k for k in batch_keys if isinstance(k, str) and k])))
        _LEGACY_ALIAS_PREFIXES = tuple(dict.fromkeys(
            (*_LEGACY_ALIAS_PREFIXES, *[p for p in alias_prefixes if isinstance(p, str) and p])))


def clear_legacy_names() -> None:
    """Drop every registered legacy name (the default, and what tests start from)."""
    global _LEGACY_BATCH_KEYS, _LEGACY_ALIAS_PREFIXES
    with _LOCK:
        _LEGACY_BATCH_KEYS = ()
        _LEGACY_ALIAS_PREFIXES = ()


def legacy_batch_keys() -> tuple[str, ...]:
    return _LEGACY_BATCH_KEYS


def legacy_alias_prefixes() -> tuple[str, ...]:
    return _LEGACY_ALIAS_PREFIXES


def register_provider(provider) -> None:
    """Install the batch provider (a plugin). Replacing one is allowed."""
    global _PROVIDER
    with _LOCK:
        _PROVIDER = provider
    log.info("session input provider registered: %s", type(provider).__name__)


def unregister_provider(provider=None) -> None:
    """Remove *provider*, or whatever is registered when *provider* is None."""
    global _PROVIDER
    with _LOCK:
        if provider is None or _PROVIDER is provider:
            _PROVIDER = None


def current_provider():
    return _PROVIDER


def has_provider() -> bool:
    return _PROVIDER is not None


def _call(name: str, *args, **kwargs):
    provider = _PROVIDER
    if provider is None:
        return None
    callback = getattr(provider, name, None)
    if callback is None:
        return None
    try:
        return callback(*args, **kwargs)
    except Exception:  # a plugin must never break admission or a live turn
        log.warning("session input provider hook %s failed", name, exc_info=True)
        return None


@dataclass(frozen=True)
class BatchPolicy:
    """Validated windows a provider asked for; ``None`` means batching is off."""

    first_window_seconds: float = 2.0
    quiet_seconds: float = 2.0
    hard_cap_seconds: float = 8.0

    def __post_init__(self):
        values = (self.first_window_seconds, self.quiet_seconds, self.hard_cap_seconds)
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not 0 < v <= _MAX_WINDOW for v in values):
            raise ValueError("input windows must be numbers in (0, 60]")
        if self.hard_cap_seconds < max(self.first_window_seconds, self.quiet_seconds):
            raise ValueError("input hard cap must cover the first and quiet windows")


@dataclass(frozen=True)
class AdmissionSnapshot:
    """Immutable view of one admitted batch handed to a provider."""

    profile_id: str
    session_id: str
    admission_id: str
    generation: int | None
    principal_id: str
    parts: tuple
    text: str


@dataclass(frozen=True)
class TurnPhase:
    """What the running generation is doing, for the provider's hand-over choice."""

    executing_tools: bool
    model_request_active: bool
    redirect_pending: bool
    steer_pending: bool


def profile_id(db) -> str:
    path = getattr(db, "db_path", None)
    return str(Path(path).resolve().parent) if path else ""


def input_payload_fields() -> frozenset:
    """Payload fields a batch record may add; per-surface gates union this in."""
    return frozenset({META, *legacy_batch_keys()})


def batch_record(payload: Mapping) -> tuple[str, dict] | tuple[None, None]:
    """``(key, record)`` for a batched payload, reading legacy keys too."""
    if not isinstance(payload, Mapping):
        return None, None
    for key in (META, *legacy_batch_keys()):
        record = payload.get(key)
        if isinstance(record, Mapping) and isinstance(record.get("parts"), list) and record["parts"]:
            return key, dict(record)
    return None, None


def original_payload(payload: Mapping) -> Mapping:
    """The payload an admission was created from, unwrapping a batch record."""
    _, record = batch_record(payload)
    original = record.get("original") if record else None
    return original if isinstance(original, Mapping) else payload


def _validated(raw) -> tuple[BatchPolicy, str] | None:
    if not isinstance(raw, Mapping):
        return None
    allowed = {"eligible", "first_window_seconds", "quiet_seconds", "hard_cap_seconds", "signature"}
    extra = set(raw) - allowed
    if extra:
        log.warning("session input policy ignored: unknown fields %s", sorted(extra))
        return None
    if raw.get("eligible") is not True:
        return None
    try:
        policy = BatchPolicy(raw.get("first_window_seconds"), raw.get("quiet_seconds"),
                             raw.get("hard_cap_seconds"))
    except ValueError as exc:
        log.warning("session input policy ignored: %s", exc)
        return None
    signature = raw.get("signature")
    if not isinstance(signature, str) or not signature or len(signature) > _MAX_SIGNATURE:
        log.warning("session input policy ignored: signature must be a non-empty string")
        return None
    return policy, signature


def policy_for(db, payload=None) -> BatchPolicy | None:
    """The profile-scoped policy snapshot, or None when nothing may be batched."""
    snapshot = snapshot_for(db, payload)
    return snapshot[0] if snapshot else None


def snapshot_for(db, payload=None):
    """Freeze ``(BatchPolicy, signature)`` before any write transaction opens."""
    if _PROVIDER is None:
        return None
    return _validated(_call("policy", profile_id(db), payload))


def policy_setting(db, name: str, default=None):
    policy = policy_for(db)
    return getattr(policy, name, default) if policy is not None else default


def _snapshot(db, payload) -> tuple[BatchPolicy, str] | None:
    return snapshot_for(db, payload)


def render_text(parts) -> str:
    rendered = _call("render", parts)
    if isinstance(rendered, str) and rendered:
        return rendered
    return "\n\n".join(str(part.get("text", "")) for part in parts)


def format_part(text: str, *, count: int, arrived=None) -> str:
    rendered = _call("format_part", text, count=count, arrived=arrived)
    return rendered if isinstance(rendered, str) and rendered else text


def wants_transfer(snapshot: AdmissionSnapshot) -> bool:
    return _call("admitted", snapshot) is True


def transfer_mode(phase: TurnPhase) -> str:
    mode = _call("transfer_mode", phase)
    return mode if mode in {"steer", "redirect", "queue"} else "queue"


def _alias_key(prefix: str, principal, session, request) -> str:
    encoded = json.dumps([principal, session, request], separators=(",", ":"))
    return prefix + hashlib.sha256(encoded.encode()).hexdigest()


def retry_alias(conn, *, principal, session, request, digest=None):
    """An accepted request must stay retryable even with the plugin unloaded.

    Prefixes a deployment registered with :func:`register_legacy_names` resolve
    too, so a follower's retry after rollback still lands on the batch it was
    folded into while that deployment migrates.
    """
    from hermes_state_runtime import ADMITTED_REQUEST_CONFLICT, RuntimeStoreError, _admission, _row
    for prefix in (ALIAS, *legacy_alias_prefixes()):
        saved = conn.execute("SELECT value FROM state_meta WHERE key=?",
                             (_alias_key(prefix, principal, session, request),)).fetchone()
        if saved is None:
            continue
        alias = json.loads(saved[0])
        if digest is not None and alias["digest"] != digest:
            raise RuntimeStoreError("admission_conflict", **ADMITTED_REQUEST_CONFLICT)
        return _row(_admission(conn, alias["admission_id"]))
    return None


def prepare_input(db, conn, *, principal, session, request, encoded, digest, arrived, snapshot=None):
    """Fold into an unclaimed tail in the SAME transaction as its retry alias.

    The head's original payload_digest stays immutable; followers' exact digests
    point at that receipt. No ACK or execution can observe half a fold.
    """
    from hermes_state_runtime import _row
    payload = json.loads(encoded)
    snapshot = snapshot or _snapshot(db, payload)
    if snapshot is None:
        return encoded, None
    policy, signature = snapshot
    part = {"at": arrived, "text": payload["text"]}
    tail = conn.execute("SELECT * FROM session_admissions WHERE target_session_id=? "
                        "AND status IN ('queued','started','unknown') ORDER BY seq DESC LIMIT 1", (session,)).fetchone()
    if tail is not None and tail["status"] == "queued" and tail["principal_id"] == principal:
        previous = json.loads(tail["payload_json"])
        key, record = batch_record(previous)
        if (record and arrived < record["deadline"] and record.get("signature") == signature
                and record.get("principal") in (None, principal)):
            record["parts"] = [*record["parts"], part]
            record["last"] = arrived
            record["deadline"] = min(arrived + policy.quiet_seconds,
                                     record["first"] + policy.hard_cap_seconds)
            previous[key] = record
            previous["text"] = render_text(record["parts"])
            conn.execute("UPDATE session_admissions SET payload_json=? WHERE admission_id=? AND status='queued'",
                         (json.dumps(previous, ensure_ascii=False), tail["admission_id"]))
            conn.execute("INSERT INTO state_meta(key,value) VALUES(?,?)", (
                _alias_key(ALIAS, principal, session, request),
                json.dumps({"admission_id": tail["admission_id"], "digest": digest})))
            log.info("input batch path=fold count=%d", len(record["parts"]))
            return encoded, _row(conn.execute("SELECT * FROM session_admissions WHERE admission_id=?",
                                             (tail["admission_id"],)).fetchone())
    original = deepcopy(payload)
    payload[META] = {"first": arrived, "last": arrived, "principal": principal,
                     "deadline": arrived + policy.first_window_seconds,
                     "signature": signature, "parts": [part], "original": original}
    payload["text"] = render_text([part])
    log.info("input batch path=open count=1 first_window=%.3f", policy.first_window_seconds)
    return json.dumps(payload, ensure_ascii=False), None


def remaining(row, now=None) -> float:
    """Seconds until a queued batch's deadline; 0 for anything not waiting."""
    if row is None or row.get("status") != "queued":
        return 0.0
    _, record = batch_record(row.get("payload", {}))
    return max(0.0, record["deadline"] - (time.time() if now is None else now)) if record else 0.0


async def wait_ready(authority, ref) -> None:
    """Wait outside the authority's mutation lock so later input can extend the window.

    A final transaction-side check closes the timer/claim race. Deadlines are
    durable, so a process restart never loses accepted text to a timer.
    """
    import asyncio

    from hermes_state_runtime import list_session_admissions
    while _PROVIDER is not None:
        rows = await asyncio.to_thread(list_session_admissions, authority.db, session_id=ref.session_id)
        if any(row["status"] in {"started", "unknown"} for row in rows):
            return
        head = next((row for row in rows if row["status"] == "queued"), None)
        delay = remaining(head)
        if not delay or authority.runner._draining or authority.retiring:
            return
        await asyncio.sleep(min(delay, 0.25))  # controls/cancel/restart stay responsive


def _phase(agent) -> TurnPhase:
    model = getattr(agent, "_model_request_active", None)
    return TurnPhase(
        executing_tools=bool(getattr(agent, "_executing_tools", False)),
        model_request_active=bool(model is not None and model.is_set()),
        redirect_pending=bool(getattr(agent, "_pending_redirect", None)),
        steer_pending=bool(getattr(agent, "_pending_steer", None)),
    )


def agent_phase(agent) -> TurnPhase:
    return _phase(agent)
