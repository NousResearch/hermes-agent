"""In-memory reference implementation of ``SessionBindingRepository`` (test fixture).

Registered as the ``bindings`` attribute; earns ``receipted_writes``. Every write commits through the shared authority transaction (``_authority.py``): the session row, the routing entry and the receipt stage on the copy-on-write view and publish together, with the failpoints below in the crash windows between the staged writes. Receipts, legacy bindings and history-control records live in the authority ``meta`` space (``bindings_lineage``); API binding receipts and declared keys in the scope's ``api_bindings`` table (``bindings_api``). Routing entries are stored as the raw entry JSON text (``StoredRoute``).

Reference limits: the API binding's route heal logs nothing; ``epoch`` is the runtime epoch ``AuthorityState`` keeps.
"""

from __future__ import annotations

import copy
import json
import sqlite3
import time
from collections.abc import Mapping
from contextlib import closing
from datetime import UTC, datetime
from typing import Any, Optional

from persistence import AuthorityRefusal, PermanentStoreError, ProfileScope, ProofRejected, SessionId
from persistence.authority.bindings import (
    BindingMarks,
    LegacyAdoptionScan,
    LocalLineage,
    NewSessionBinding,
    ReservedBinding,
    SessionBinding,
)
from persistence.authority.vocabulary import LEGACY_LOCAL_SOURCES, OwnerRule
from persistence.conversation.sessions import SessionLineageKind
from persistence.core.canonical import canonical_json
from persistence.core.identity import JsonValue
from session_rules.legacy_local import binding_names, client_history_row, legacy_history_row, not_client_labels, same_owner, validate_legacy_row

from .._authority import AuthorityState, authority_state
from .._lease_fence import refuse_live_leases
from ..backend import InMemoryBackend, register_repository
from ..conversation.sessions import blank_session_row
from .bindings_api import ApiBindingVerbs
from .bindings_lineage import (
    HISTORY_PREFIX,
    LEGACY_PREFIX,
    RECEIPT_PREFIX,
    decode,
    local_conversation_ids,
    local_lineage_owner,
    local_physical_target,
    store_route,
    validate_local_lineage,
)
from .bindings_resets import LineageWriteVerbs
from .bindings_retry import same_create_retry
from .coordination import conversation_key

__all__ = [
    "BETWEEN_ROUTE_AND_RECEIPT",
    "BETWEEN_SESSION_AND_ROUTE",
    "InMemorySessionBindingRepository",
    "resolve_physical_target",
]

# Crash windows inside reserve, in staging order: after the session row, after the routing entry.
BETWEEN_SESSION_AND_ROUTE = "bindings-session-row-staged"
BETWEEN_ROUTE_AND_RECEIPT = "bindings-route-entry-staged"

_RECEIPT_PREFIX = RECEIPT_PREFIX
_IDENTITY = ("profile_id", "principal_id", "request_id", "session_id", "route")


def _bound_id(value: Any, parameter: int = 1) -> Any:
    """``value`` as SQLite binds it into a TEXT column (text affinity), with sqlite3's binding refusals."""
    if value is None or isinstance(value, bytes):
        return value
    if isinstance(value, str):
        value.encode("utf-8")
        return value
    if isinstance(value, int):
        value = int.__int__(value)
        if not -(2**63) <= value < 2**63:
            raise OverflowError("Python int too large to convert to SQLite INTEGER")
        return str(value)
    if isinstance(value, float):
        with closing(sqlite3.connect(":memory:")) as conn:
            return conn.execute("SELECT CAST(? AS TEXT)", (value,)).fetchone()[0]
    raise sqlite3.ProgrammingError(f"Error binding parameter {parameter}: type '{type(value).__name__}' is not supported")


def resolve_physical_target(backend: InMemoryBackend, scope: ProfileScope, authority: dict[str, Any], tables: dict[str, Any], session_id: str) -> str:
    """The local physical target over one (staged or committed) view; the worker reservation reads the lineage in its own transaction through it."""
    return local_physical_target(backend, scope, authority["meta"], tables.get("sessions", {}), session_id)


def _to_binding(stored: Any) -> SessionBinding:
    """The stored receipt as a detached record; an undecodable receipt or one that is not an object is ``storage_unavailable``."""
    try:
        receipt = decode(stored)
        if not isinstance(receipt, dict):
            raise ValueError("invalid receipt")
        receipt = copy.deepcopy(receipt)
        lineage = receipt.get("lineage", [receipt["session_id"]])
        return SessionBinding(
            profile_id=receipt["profile_id"], principal_id=receipt["principal_id"], request_id=receipt["request_id"],
            session_id=SessionId(receipt["session_id"]), route=receipt["route"], entry=receipt["entry"], policy=receipt["policy"],
            lineage=tuple(SessionId(sid) for sid in lineage), legacy_session_id=receipt.get("legacy_session_id"))
    except (TypeError, ValueError, KeyError) as exc:
        raise AuthorityRefusal("storage_unavailable") from exc


def _receipt_of(binding: SessionBinding) -> dict[str, Any]:
    """The receipt dict a ``SessionBinding`` was read from (``lineage`` present unless it is the default)."""
    receipt: dict[str, Any] = {
        "profile_id": binding.profile_id, "principal_id": binding.principal_id, "request_id": binding.request_id,
        "session_id": binding.session_id, "route": binding.route, "entry": dict(binding.entry), "policy": dict(binding.policy)}
    if binding.legacy_session_id is not None:
        receipt["legacy_session_id"] = binding.legacy_session_id
    if binding.lineage != (binding.session_id,):
        receipt["lineage"] = list(binding.lineage)
    return json.loads(json.dumps(receipt))


def _new_receipt(binding: NewSessionBinding) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "profile_id": binding.profile_id, "principal_id": binding.principal_id, "request_id": binding.request_id,
        "session_id": binding.session_id, "route": binding.route, "entry": copy.deepcopy(dict(binding.entry)),
        "policy": copy.deepcopy(dict(binding.policy))}
    if binding.legacy_session_id is not None:
        receipt["legacy_session_id"] = binding.legacy_session_id
    if binding.lineage is not None:
        receipt["lineage"] = list(binding.lineage)
    return receipt


def _row_cwd(policy: Mapping[str, Any]) -> Any:
    """The create request's ``row_cwd`` (``""`` = none) when it sent one, else the policy cwd."""
    try:
        request = json.loads(policy.get("request_json") or "{}")
    except (TypeError, ValueError):
        request = {}
    value = request.get("row_cwd") if isinstance(request, dict) else None
    return policy["cwd"] if not isinstance(value, str) else (value or None)


def legacy_account_label(row: Mapping[str, Any]) -> bool:
    """Whether the row's user is the Desktop Nous-account stamp, a label and not an owner."""
    user_id = row.get("user_id")
    return row["source"] == "desktop" and isinstance(user_id, str) and user_id.startswith("nous:")


def legacy_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """The columns the legacy-history rules read, as the stored row holds them (``origin`` as its JSON text)."""
    origin = row.get("origin")
    return {"source": row.get("source"), "user_id": row.get("user_id"), "session_key": row.get("session_key"), "chat_id": row.get("chat_id"),
            "origin_json": None if origin is None else json.dumps(origin), "profile_name": row.get("profile_name")}


def binding_text(saved: Any) -> Optional[str]:
    """A stored history binding as the compact JSON text ``history_binding`` returns, None when absent."""
    return None if saved is None else json.dumps(saved, separators=(",", ":"))


def _require_idle(handle: Any, scope: ProfileScope, authority: dict[str, Any], sessions: Mapping[str, Any], session_ids: list[str]) -> None:
    """Queued, started or unknown admissions and live workers refuse, then the transcript guards of a row not ended by compression."""
    for sid in session_ids:
        states = {row["status"] for row in authority["admissions"].values()
                  if row.get("target_session_id") == sid and row["status"] in ("queued", "started", "unknown")}
        states |= {row["status"] for row in authority["workers"].values()
                   if row.get("session_id") == sid and row["status"] in ("registered", "running", "unknown")}
        if "unknown" in states:
            raise AuthorityRefusal("unknown_execution")
        if states:
            raise AuthorityRefusal("session_busy")
        if (sessions.get(sid) or {}).get("end_reason") != "compression":
            refuse_live_leases(handle, scope, SessionId(sid))


def _bind_legacy_target(handle: Any, scope: ProfileScope, authority: dict[str, Any], tables: dict[str, Any], receipt: dict[str, Any], owner: Optional[OwnerRule]) -> None:
    """Adopt a legacy local session as the receipt's target after the ownership, idle, lease and not-yet-bound checks under ``owner``: the row takes the receipt's route and origin (a client app's label becomes ``desktop``) and the legacy binding record is written."""
    meta, sessions = authority["meta"], tables.setdefault("sessions", {})
    target = receipt["legacy_session_id"]
    if target != receipt["session_id"]:
        raise AuthorityRefusal("storage_unavailable")
    row = sessions.get(target)
    if row is None:
        raise AuthorityRefusal("not_found")
    validate_legacy_row(handle.owner_profile, legacy_row(row), subject=receipt["principal_id"], rule=owner)
    imported = meta.get(HISTORY_PREFIX + target)
    if imported is not None and not binding_names(binding_text(imported), receipt["profile_id"], receipt["principal_id"], owner):
        raise ProofRejected("session", target)
    if handle.lineage._chain_step(scope, sessions, target) is not None:
        raise AuthorityRefusal("admission_conflict")
    _require_idle(handle, scope, authority, sessions, [target])
    lease = tables.get("coord_turn_leases", {}).get(conversation_key(handle, scope, target))
    if lease is not None and lease[1] > time.time():
        raise AuthorityRefusal("runtime_coordination_required")
    if LEGACY_PREFIX + target in meta:
        raise AuthorityRefusal("storage_unavailable")
    label = "desktop" if client_history_row(legacy_row(row), owner) else row["source"]
    row.update(source=label, session_key=receipt["route"], chat_id=receipt["session_id"], user_id=receipt["principal_id"], chat_type="dm",
               origin=copy.deepcopy(receipt["entry"]["origin"]))
    meta[LEGACY_PREFIX + target] = {key: receipt[key] for key in ("session_id", "profile_id", "principal_id")} | {"legacy_session_id": target}


def _reserved_row(handle: Any, binding: NewSessionBinding, receipt: dict[str, Any]) -> dict[str, Any]:
    """The row a reserve inserts, in the fresh-row shape of the sessions reference."""
    policy, entry = receipt["policy"], receipt["entry"]
    try:
        chat_id = _bound_id(entry["origin"]["chat_id"], 6)
    except sqlite3.ProgrammingError as exc:
        raise PermanentStoreError(str(exc)) from exc
    row = blank_session_row(receipt["session_id"], lineage_kind=SessionLineageKind.NONE, model_config={})
    row.update(source=policy["source"], created_source=binding.created_source or None, user_id=receipt["principal_id"],
               session_key=receipt["route"], chat_id=chat_id, chat_type="dm", model=policy["model"], cwd=_row_cwd(policy),
               profile_name=handle.owner_profile, origin=copy.deepcopy(entry["origin"]),
               started_at=datetime.fromtimestamp(datetime.fromisoformat(entry["created_at"]).timestamp(), UTC))
    return row


def _owner_user(row: Mapping[str, Any]) -> Any:
    """The row's owner identity; the Desktop Nous-account label is none."""
    return None if legacy_account_label(row) else row.get("user_id")


def _routed(row: Mapping[str, Any]) -> bool:
    return bool(row.get("session_key") or row.get("chat_id") or row.get("origin"))


def history_row(row: Mapping[str, Any], owner: Optional[OwnerRule]) -> bool:
    """Owner history a sidebar edit may claim: legacy history (a local surface's, or a client app's under ``owner``), or an import."""
    return row.get("source") == "import" or legacy_history_row(legacy_row(row), owner)


def _policy_target(stored: Any) -> Any:
    """The stored receipt's ``entry['session_id']``, None when it does not decode to an object holding an object ``entry``."""
    try:
        value = decode(stored)
    except (TypeError, ValueError):
        return None
    entry = value.get("entry") if isinstance(value, dict) else None
    return entry.get("session_id") if isinstance(entry, dict) else None


class InMemorySessionBindingRepository(ApiBindingVerbs, LineageWriteVerbs):
    """Reference implementation; see ``persistence/authority/bindings.py`` for the contract text."""

    def __init__(self, backend: InMemoryBackend) -> None:
        self._backend = backend

    def _state(self) -> AuthorityState:
        return authority_state(self._backend, self._backend.scope)

    def _committed(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """``(meta, sessions)`` of the committed state, for the reads."""
        sessions: dict[str, Any] = self._backend.scope_store(self._backend.scope).setdefault("sessions", {})
        return self._state().snapshot["meta"], sessions

    def reserve(self, *, epoch: int, binding: NewSessionBinding, client_request: Mapping[str, JsonValue] | None = None, owner: OwnerRule | None) -> ReservedBinding:
        request: Mapping[str, Any] = client_request if client_request is not None else {}
        if client_request is None and binding.policy.get("client_env_json") is not None:
            raise AuthorityRefusal("invalid_params")
        receipt = json.loads(canonical_json(_new_receipt(binding)))
        sid, route = receipt["session_id"], receipt["route"]
        state = self._state()
        with state.transaction() as staged:
            state.require_epoch(staged.authority, epoch)
            meta = staged.authority["meta"]
            old = meta.get(RECEIPT_PREFIX + sid)
            if old is not None:
                try:
                    saved = decode(old)
                    same = all(saved[k] == receipt[k] for k in _IDENTITY)
                    same = same and same_create_retry(saved["policy"], receipt["policy"]["request_json"], request)
                except (ValueError, TypeError, KeyError) as exc:
                    raise AuthorityRefusal("storage_unavailable") from exc
                if not same:
                    raise AuthorityRefusal("invalid_params")
                return ReservedBinding(binding=_to_binding(saved), created=False)
            if sid in staged.authority["retired"]:
                raise AuthorityRefusal("not_found")
            sessions = staged.scope.setdefault("sessions", {})
            if "legacy_session_id" not in receipt and sid in sessions:
                raise AuthorityRefusal("storage_unavailable")
            if route in staged.scope.setdefault("routing", {}).setdefault("", {}):
                raise AuthorityRefusal("admission_conflict")
            if "legacy_session_id" in receipt:
                _bind_legacy_target(self._backend, self._backend.scope, staged.authority, staged.scope, receipt, owner)
            if sid not in sessions:
                sessions[sid] = _reserved_row(self._backend, binding, receipt)
            state.failpoints.hit(BETWEEN_SESSION_AND_ROUTE)
            store_route(staged.scope, route, receipt["entry"])
            state.failpoints.hit(BETWEEN_ROUTE_AND_RECEIPT)
            meta[RECEIPT_PREFIX + sid] = receipt
        return ReservedBinding(binding=_to_binding(receipt), created=True)

    def end_idle(self, *, epoch: int, session_id: SessionId, target_id: SessionId, reason: str) -> int:
        from ..conversation.transcript_store import advance_conversation_generation

        state = self._state()
        authority = state.snapshot
        state.require_epoch(authority, epoch)
        if any(row.get("target_session_id") == session_id and row["status"] in ("queued", "started", "unknown") for row in authority["admissions"].values()):
            return 0
        if any(row.get("session_id") in (session_id, target_id) and row["status"] in ("registered", "running", "unknown") for row in authority["workers"].values()):
            return 0
        row = self._committed()[1].get(str(target_id))
        if row is None or row["ended_at"] is not None:
            return 0
        row.update(ended_at=datetime.now(UTC), end_reason=reason)
        advance_conversation_generation(self._backend, self._backend.scope, str(target_id), reason)
        return 1

    # ── reads ────────────────────────────────────────────────────────────

    def target_has_history(self, session_id: str) -> bool:
        from ..conversation.transcript_store import transcripts_of

        authority, (_meta, sessions) = self._state().snapshot, self._committed()
        transcript = transcripts_of(self._backend, self._backend.scope).get(session_id)
        return bool(
            (transcript is not None and transcript.rows)
            or any(row.get("target_session_id") == session_id for row in authority["admissions"].values())
            or any(row.get("session_id") == session_id for row in authority["workers"].values())
            or any(row.get("parent_session_id") == session_id for row in sessions.values()))

    def receipt(self, session_id: SessionId) -> SessionBinding:
        stored = self._committed()[0].get(RECEIPT_PREFIX + str(session_id))
        if stored is None:
            raise AuthorityRefusal("storage_unavailable")
        return _to_binding(stored)

    def local_session_ids(self) -> list[SessionId]:
        return [SessionId(key[len(RECEIPT_PREFIX):]) for key in self._committed()[0] if isinstance(key, str) and key.startswith(RECEIPT_PREFIX)]

    def list_lineages(self) -> list[LocalLineage]:
        found: list[LocalLineage] = []
        for key, raw in self._committed()[0].items():
            if not (isinstance(key, str) and key.startswith(RECEIPT_PREFIX)):
                continue
            try:
                receipt = decode(raw)
            except (TypeError, ValueError):
                continue
            owner = receipt.get("session_id") if isinstance(receipt, dict) else None
            lineage = receipt.get("lineage") if isinstance(owner, str) else None
            if isinstance(lineage, list):
                found.append(LocalLineage(owner=SessionId(owner), lineage=tuple(SessionId(sid) for sid in lineage if isinstance(sid, str))))
        return found

    def binding_marks(self, session_id: SessionId, *, lineage_of: SessionId | None = None) -> BindingMarks:
        meta, sessions = self._committed()
        policy, legacy = RECEIPT_PREFIX + str(session_id) in meta, LEGACY_PREFIX + str(session_id) in meta
        owner = local_lineage_owner(meta, sessions, str(lineage_of)) if lineage_of and (policy or legacy) else None
        return BindingMarks(policy=policy, legacy=legacy, lineage_owner=None if owner is None else SessionId(owner))

    def lineage_owner(self, session_id: SessionId) -> SessionId:
        return SessionId(local_lineage_owner(*self._committed(), str(session_id)))

    def conversation_ids(self, session_id: SessionId) -> list[SessionId]:
        return [SessionId(sid) for sid in local_conversation_ids(*self._committed(), str(session_id))]

    def validate_lineage(self, receipt: SessionBinding) -> SessionId:
        meta, sessions = self._committed()
        return SessionId(validate_local_lineage(self._backend, self._backend.scope, meta, sessions, _receipt_of(receipt)))

    def legacy_adoption_scan(self, owner: OwnerRule) -> LegacyAdoptionScan:
        meta, sessions = self._committed()
        refused = not_client_labels(owner)
        legacy = sorted((row["started_at"], sid) for sid, row in sessions.items()
                        if row["source"] in LEGACY_LOCAL_SOURCES or (row["source"] not in refused and not _routed(row)))
        marked = frozenset(key[len(prefix):] for key in meta if isinstance(key, str)
                           for prefix in (RECEIPT_PREFIX, LEGACY_PREFIX) if key.startswith(prefix))
        return LegacyAdoptionScan(session_ids=tuple(SessionId(sid) for _started, sid in legacy), marked=marked)

    def history_binding(self, session_id: SessionId) -> Optional[str]:
        saved = self._committed()[0].get(HISTORY_PREFIX + str(session_id))
        return None if saved is None else json.dumps(saved, separators=(",", ":"))

    def handoff_head(self, session_id: SessionId) -> tuple[Optional[SessionId], SessionId]:
        meta, sessions = self._committed()
        owner = local_lineage_owner(meta, sessions, str(session_id))
        saved = meta.get(RECEIPT_PREFIX + owner)
        target = _policy_target(saved) if saved is not None else None
        start = SessionId(target or session_id)
        return (SessionId(owner) if target else None), self._backend.lineage.tip(start)

    def boot_session_ids(self) -> list[SessionId]:
        from .mutation_commands import VIEWER_CLOSE_PREFIX

        meta, sessions = self._committed()
        admissions = self._state().snapshot["admissions"].values()
        kept: list[SessionId] = []
        for key, raw in meta.items():
            if not (isinstance(key, str) and key.startswith(RECEIPT_PREFIX)):
                continue
            owner, target = key[len(RECEIPT_PREFIX):], _policy_target(raw)
            row = sessions.get(target) if isinstance(target, str) else None
            if (row is None or row["ended_at"] is None or row["end_reason"] == "compression"
                    or any(a["target_session_id"] == owner and a["status"] in ("queued", "started", "unknown") for a in admissions)
                    or VIEWER_CLOSE_PREFIX + owner in meta):
                kept.append(SessionId(owner))
        return kept

    def compression_adoptable(self, receipt: SessionBinding) -> bool:
        from .bindings_lineage import adoptable_compression

        tables = self._backend.scope_store(self._backend.scope)
        return adoptable_compression(self._backend, self._backend.scope, tables, _receipt_of(receipt)) is not None

    def authorize_history(self, *, session_id: SessionId, profile_id: str, subject: str, may_create: bool, native_transport: bool | None = None,
                          owner: OwnerRule | None) -> bool:
        meta, sessions = self._committed()
        sid = str(session_id)
        row = sessions.get(sid)
        if native_transport is not None:
            self._require_history_claim(meta, row, sid, subject, native_transport, owner)
        if row is None or _routed(row):
            return False
        saved = meta.get(HISTORY_PREFIX + sid)
        if saved is not None:
            if not binding_names(binding_text(saved), profile_id, subject, owner):
                raise ProofRejected("session", sid)
            return True
        if not history_row(row, owner):
            return False
        if _owner_user(row) and not same_owner(row["user_id"], subject, owner):
            raise ProofRejected("session", sid)
        if not may_create:
            raise ProofRejected("session", sid)
        return True

    def _require_history_claim(self, meta: Mapping[str, Any], row: Optional[Mapping[str, Any]], sid: str, subject: str, native_transport: bool,
                               owner: Optional[OwnerRule]) -> None:
        """The history-claim check: an unrouted, unclaimed row outside this subject's own history refuses ``ProofRejected`` unless ``native_transport`` (the registry's answer) holds."""
        if row is None or _routed(row) or HISTORY_PREFIX + sid in meta:
            return
        own_profile = row.get("profile_name") in (None, "", self._backend.owner_profile or "default")
        if own_profile and _owner_user(row) and same_owner(row["user_id"], subject, owner):
            return
        if not native_transport:
            raise ProofRejected("session", sid)


register_repository("bindings", InMemorySessionBindingRepository, {"receipted_writes": True})
