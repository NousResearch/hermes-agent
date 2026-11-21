"""``prepare_mutation`` and ``mutate`` of the runtime reference, over the closed ``MutationCommand`` union.

Receipts live in the authority ``meta`` table under the ``gateway.mutation.v1.*`` keys with the canonical JSON value ``{"digest", "result"}``, which ``policy_picks`` reads. The write runs under ``AuthorityState.guarded``: the command handlers compose the session, message and maintenance references' own steps, and any refusal restores every table. :data:`RECEIPT_FAILPOINT` sits between the command's writes and the receipt.
"""

from __future__ import annotations

import json
from typing import Any

from persistence.authority.admissions import GateRefusal
from persistence.authority.runtime import (
    _GENERATION_REQUIRED,
    ClaimHistory,
    HistoryWrite,
    ImportControl,
    LocalSnapshot,
    MutationCommand,
    MutationOutcome,
    PreparedMutation,
    SessionGone,
    mutation_receipt_key,
    parse_mutation,
)
from persistence.authority.vocabulary import OwnerRule
from persistence.core.canonical import canonical_json
from persistence.core.errors import AuthorityRefusal, ProofRejected
from session_rules.legacy_local import binding_names, same_owner

from .._authority import session_runtime
from . import admissions_rows as rows
from .bindings import _owner_user, binding_text, history_row
from . import mutation_guards as guards
from .mutation_commands import apply_command

#: Failpoint between a command's writes and its receipt.
RECEIPT_FAILPOINT = "runtime.between-command-and-receipt"


def _checked(principal_id: Any, session_id: Any, request_id: Any, expected_revision: Any, command: Any, expected_generation: Any) -> tuple[MutationCommand, dict[str, Any], str, str]:
    for value in (principal_id, session_id, request_id):
        rows.text(value)
    if type(expected_revision) is not int or expected_revision < 0:
        raise AuthorityRefusal("invalid_params")
    operation = getattr(command, "operation", None)
    payload_of = getattr(command, "payload", None)
    parsed = parse_mutation(operation, payload_of() if callable(payload_of) else None)
    if expected_generation is not None and (type(expected_generation) is not int or expected_generation < 0):
        raise AuthorityRefusal("invalid_params")
    if parsed.operation in _GENERATION_REQUIRED and expected_generation is None:
        raise AuthorityRefusal("invalid_params")
    payload = rows.encode(parsed.payload())
    key, digest = mutation_receipt_key(principal_id, session_id, request_id, operation=parsed.operation, payload=payload,
                                       expected_revision=expected_revision, expected_generation=expected_generation)
    return parsed, payload, key, digest


def _history_key(session_id: str) -> str:
    from .bindings import HISTORY_PREFIX

    return HISTORY_PREFIX + session_id


def _authorize_history(authority: dict[str, Any], tables: dict[str, Any], actor: Any, session_id: str, owner: OwnerRule | None) -> None:
    """Authorize and claim the row's history, refusing a row that holds no claimable history; the binding is stored where the binding reference keeps it."""
    row = tables.get("sessions", {}).get(session_id)
    if row is None or row.get("session_key") or row.get("chat_id") or row.get("origin") is not None:
        raise ProofRejected("session", session_id)
    key, binding = _history_key(session_id), [actor.profile_id, actor.subject]
    saved = authority["meta"].get(key)
    if saved is not None:
        if not binding_names(binding_text(saved), actor.subject, owner):
            raise ProofRejected("session", session_id)
        return
    if not history_row(row, owner):
        raise ProofRejected("session", session_id)
    if _owner_user(row) and not same_owner(row["user_id"], actor.subject, owner):
        raise ProofRejected("session", session_id)
    if not actor.may_create:
        raise ProofRejected("session", session_id)
    authority["meta"][key] = binding


class MutationMixin:
    """The receipted mutation verbs; mixed into ``InMemorySessionRuntimeRepository``."""

    def _fenced(self, epoch: int, session_id: str, parsed: MutationCommand, key: str, digest: str, expected_revision: int, expected_generation: int | None,
                live_guard: GateRefusal | None, owner: OwnerRule | None = None) -> MutationOutcome | dict[str, Any] | None:
        """Epoch, receipt replay, existence, revision, generation and the live guard, in that order; the replayed outcome, or the session row."""
        state = self._state()
        authority, tables = state.snapshot, self._tables()
        state.require_epoch(authority, epoch)
        stored = authority["meta"].get(key)
        if stored is not None:
            receipt = json.loads(stored)
            if receipt["digest"] != digest:
                raise AuthorityRefusal("admission_conflict")
            return MutationOutcome(result=receipt["result"], replayed=True, gone=None)
        session = tables.get("sessions", {}).get(session_id)
        if session is None and parsed.operation != "import":
            raise AuthorityRefusal("not_found")
        runtime = authority["sessions"].get(session_id, {})
        if (runtime.get("revision", 0) if session is not None else 0) != expected_revision:
            raise rows.revision_conflict(session_id)
        if expected_generation is not None and (session is None or runtime.get("generation", 0) != expected_generation):
            raise rows.stale("session", session_id)
        if live_guard is not None:
            if parsed.operation == "delete":
                guards.delete_targets(self._backend, authority, tables, session_id, owner)
            raise live_guard.refusal()
        return session

    def prepare_mutation(self, epoch: int, *, principal_id: str, session_id: str, request_id: str, expected_revision: int, command: MutationCommand,
                         expected_generation: int | None = None, live_guard: GateRefusal | None = None) -> LocalSnapshot | MutationOutcome:
        parsed, _payload, key, digest = _checked(principal_id, session_id, request_id, expected_revision, command, expected_generation)
        fenced = self._fenced(epoch, session_id, parsed, key, digest, expected_revision, expected_generation, live_guard)
        if isinstance(fenced, MutationOutcome):
            return fenced
        authority, tables = self._state().snapshot, self._tables()
        snapshot = guards.local_snapshot(self._backend, authority, tables, session_id)
        if parsed.operation == "branch":
            guards.require_idle(self._backend, authority, tables, list(dict.fromkeys([session_id, snapshot.target])))
        return snapshot

    def mutate(self, epoch: int, *, principal_id: str, session_id: str, request_id: str, expected_revision: int, command: MutationCommand,
               expected_generation: int | None = None, live_guard: GateRefusal | None = None, history: HistoryWrite | None = None,
               prepared: PreparedMutation | None = None, media_root: str | None = None, owner: OwnerRule | None) -> MutationOutcome:
        parsed, payload, key, digest = _checked(principal_id, session_id, request_id, expected_revision, command, expected_generation)
        state = self._state()
        with state.guarded():
            fenced = self._fenced(epoch, session_id, parsed, key, digest, expected_revision, expected_generation, live_guard, owner)
            if isinstance(fenced, MutationOutcome):
                return fenced
            session = fenced
            was_archived = bool(session and session.get("archived"))
            if isinstance(history, ClaimHistory):
                _authorize_history(state.snapshot, self._tables(), history.actor, session_id, owner)
            elif isinstance(history, ImportControl):
                for sid in history.session_ids:
                    if sid not in self._tables().get("sessions", {}):
                        state.snapshot["meta"][_history_key(sid)] = [history.actor.profile_id, history.actor.subject]
            affected, projection = apply_command(self, session_id, parsed, prepared, media_root, owner)
            authority = state.snapshot
            gone = None
            if parsed.operation == "delete":
                gone = SessionGone(tuple(projection.get("deleted_ids") or (session_id,)), "delete")
            elif payload.get("archived") is True and session is not None and not was_archived:
                gone = SessionGone(tuple(affected) or (session_id,), "archive")
            for target in affected:
                session_runtime(authority, target)["revision"] += 1
            updated = self._tables().get("sessions", {}).get(session_id)
            runtime = authority["sessions"].get(session_id, {})
            result = {"session_id": session_id, "revision": runtime.get("revision", 0) if updated else expected_revision + 1,
                      "execution_generation": runtime.get("generation", 0) if updated else expected_generation, "operation": parsed.operation, **projection}
            state.failpoints.hit(RECEIPT_FAILPOINT)
            authority["meta"][key] = canonical_json({"digest": digest, "result": result})
            return MutationOutcome(result=json.loads(canonical_json(result)), replayed=False, gone=gone)
