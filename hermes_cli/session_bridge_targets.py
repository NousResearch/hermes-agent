"""Ticket-bound eligibility records for native session transports.

The registry is an adapter boundary, not a discovery shortcut.  A provider
adapter records facts it authenticated itself; user wording, a transcript file,
or a same-UID caller cannot establish session identity or transport authority.
Resolution starts with the ticket and returns one exact active attempt, so an
operator is never asked to copy a native session or mailbox handle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
import sqlite3
import time
from typing import Any, Callable
import uuid


_TICKET_REF = re.compile(r"^[A-Z][A-Z0-9]{1,15}-[0-9]{3,}$")
_MAX_OBSERVATION_AGE = 60.0


class TargetUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SessionTargetObservation:
    observation_id: str
    ticket_ref: str
    assignment_id: str
    attempt_id: str
    assignment_active: bool
    attempt_active: bool
    backend: str
    native_session_id: str
    delivery_handle: str
    agent_id: str
    team_name: str | None
    transport: str
    owner_pid: int | None
    owner_started_at: float | None
    ownership_kind: str
    identity_verified: bool
    transport_authenticated: bool
    hermes_callable: bool
    reply_capable: bool
    cancel_capable: bool
    non_interrupting: bool
    answer_only: bool
    authority: str
    provenance: dict[str, Any]
    observed_at: float
    expires_at: float


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS session_bridge_targets (
            observation_id TEXT PRIMARY KEY,
            ticket_ref TEXT NOT NULL,
            assignment_id TEXT NOT NULL,
            attempt_id TEXT NOT NULL,
            assignment_active INTEGER NOT NULL,
            attempt_active INTEGER NOT NULL,
            backend TEXT NOT NULL,
            native_session_id TEXT NOT NULL,
            delivery_handle TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            team_name TEXT,
            transport TEXT NOT NULL,
            owner_pid INTEGER,
            owner_started_at REAL,
            ownership_kind TEXT NOT NULL,
            identity_verified INTEGER NOT NULL,
            transport_authenticated INTEGER NOT NULL,
            hermes_callable INTEGER NOT NULL,
            reply_capable INTEGER NOT NULL,
            cancel_capable INTEGER NOT NULL,
            non_interrupting INTEGER NOT NULL,
            answer_only INTEGER NOT NULL,
            authority TEXT NOT NULL,
            provenance_json TEXT NOT NULL,
            observation_sha256 TEXT NOT NULL,
            observed_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            revoked_at REAL
        );
        CREATE INDEX IF NOT EXISTS session_bridge_targets_ticket
            ON session_bridge_targets(ticket_ref, observed_at);
        """
    )


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _payload(observation: SessionTargetObservation) -> dict[str, Any]:
    value = asdict(observation)
    value["ticket_ref"] = _required_text(value["ticket_ref"], "ticket_ref").upper()
    if not _TICKET_REF.fullmatch(value["ticket_ref"]):
        raise ValueError("ticket_ref has an unsupported format")
    for key in (
        "assignment_id", "attempt_id", "backend", "native_session_id",
        "delivery_handle", "agent_id", "transport", "ownership_kind", "authority",
    ):
        value[key] = _required_text(value[key], key)
    try:
        value["observation_id"] = str(uuid.UUID(value["observation_id"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("observation_id must be a UUID") from exc
    if value["ownership_kind"] not in {"local-process", "provider-attested"}:
        raise ValueError("ownership_kind must be local-process or provider-attested")
    if value["ownership_kind"] == "local-process" and (
        not isinstance(value["owner_pid"], int) or value["owner_pid"] <= 0
        or not isinstance(value["owner_started_at"], (int, float))
    ):
        raise ValueError("local-process ownership requires pid and start time")
    if not isinstance(value["provenance"], dict) or not value["provenance"]:
        raise ValueError("authenticated observation provenance is required")
    if value["expires_at"] <= value["observed_at"]:
        raise ValueError("target observation must expire after it was observed")
    return value


def record_provider_observation(
    conn: sqlite3.Connection, observation: SessionTargetObservation,
) -> SessionTargetObservation:
    """Persist facts already authenticated by a provider-native adapter.

    This function does not authenticate its input.  It is intentionally absent
    from the public CLI and local file ingress; only the provider adapter that
    obtained the native identity and delivery handle may call it.
    """
    value = _payload(observation)
    provenance_json = json.dumps(value.pop("provenance"), sort_keys=True, separators=(",", ":"))
    canonical = json.dumps(
        {**value, "provenance": json.loads(provenance_json)},
        sort_keys=True, separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    columns = tuple(value) + ("provenance_json", "observation_sha256")
    params = tuple(value.values()) + (provenance_json, digest)
    conn.execute(
        f"INSERT INTO session_bridge_targets ({','.join(columns)}) "
        f"VALUES ({','.join('?' for _ in columns)})",
        params,
    )
    return observation


def _default_liveness(pid: int, started_at: float) -> bool:
    try:
        import psutil  # type: ignore
        process = psutil.Process(pid)
        return process.is_running() and abs(float(process.create_time()) - started_at) < 0.001
    except Exception:
        return False


def _from_row(row: sqlite3.Row) -> SessionTargetObservation:
    return SessionTargetObservation(
        observation_id=row["observation_id"], ticket_ref=row["ticket_ref"],
        assignment_id=row["assignment_id"], attempt_id=row["attempt_id"],
        assignment_active=bool(row["assignment_active"]),
        attempt_active=bool(row["attempt_active"]),
        backend=row["backend"], native_session_id=row["native_session_id"],
        delivery_handle=row["delivery_handle"], agent_id=row["agent_id"],
        team_name=row["team_name"], transport=row["transport"],
        owner_pid=row["owner_pid"], owner_started_at=row["owner_started_at"],
        ownership_kind=row["ownership_kind"], identity_verified=bool(row["identity_verified"]),
        transport_authenticated=bool(row["transport_authenticated"]),
        hermes_callable=bool(row["hermes_callable"]),
        reply_capable=bool(row["reply_capable"]), cancel_capable=bool(row["cancel_capable"]),
        non_interrupting=bool(row["non_interrupting"]), answer_only=bool(row["answer_only"]),
        authority=row["authority"], provenance=json.loads(row["provenance_json"]),
        observed_at=float(row["observed_at"]), expires_at=float(row["expires_at"]),
    )


def resolve_ticket_target(
    conn: sqlite3.Connection, ticket_ref: str, *, now: float | None = None,
    liveness: Callable[[int, float], bool] = _default_liveness,
) -> SessionTargetObservation:
    """Resolve one eligible active assignment/attempt from a ticket reference."""
    ref = _required_text(ticket_ref, "ticket_ref").upper()
    if not _TICKET_REF.fullmatch(ref):
        raise ValueError("ticket_ref has an unsupported format")
    when = time.time() if now is None else now
    rows = conn.execute(
        "SELECT * FROM session_bridge_targets WHERE ticket_ref=? AND revoked_at IS NULL "
        "ORDER BY observed_at DESC", (ref,),
    ).fetchall()
    eligible_by_attempt: dict[tuple[str, str], SessionTargetObservation] = {}
    reasons: set[str] = set()
    for row in rows:
        item = _from_row(row)
        if item.observed_at > when or when - item.observed_at > _MAX_OBSERVATION_AGE or item.expires_at <= when:
            reasons.add("stale or expired")
            continue
        missing = [name for name in (
            "assignment_active", "attempt_active",
            "identity_verified", "transport_authenticated", "hermes_callable", "reply_capable",
            "cancel_capable", "non_interrupting", "answer_only",
        ) if not getattr(item, name)]
        if missing:
            reasons.add("missing " + ", ".join(missing))
            continue
        if item.ownership_kind == "local-process" and not liveness(
            int(item.owner_pid or 0), float(item.owner_started_at or 0),
        ):
            reasons.add("owner is not provably live")
            continue
        key = (item.assignment_id, item.attempt_id)
        previous = eligible_by_attempt.get(key)
        if previous is None or item.observed_at > previous.observed_at:
            eligible_by_attempt[key] = item
    eligible = list(eligible_by_attempt.values())
    if len(eligible) == 1:
        return eligible[0]
    if len(eligible) > 1:
        raise TargetUnavailable(
            f"ticket {ref} has multiple eligible active session attempts; routing is ambiguous"
        )
    detail = "; ".join(sorted(reasons)) if reasons else "no provider-native observation"
    raise TargetUnavailable(f"ticket {ref} has no eligible active session target: {detail}")

