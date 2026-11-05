"""The runtime verbs beside ``mutate``: epoch, deferred and chat picks, policy picks and refresh, viewer closes, retirement reads, the ledger-gated ends and the delete footprint.

Durable values live in the authority ``meta`` table under string keys with string values: mutation receipts, the deferred-pick slot, viewer closes.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from persistence.authority.bindings import api_generation_id, generation_request_id, local_session_id, room_session_id
from persistence.authority.runtime import (
    ApiGenerations,
    ChatPickParams,
    DeferredPick,
    DeleteFootprint,
    GenerationChain,
    GenerationPick,
    GenerationRow,
    PinnedIdentity,
    RoomGenerations,
    SessionRuntimeState,
    StoredDeferredPick,
    mutation_receipt_key,
    parse_mutation,
)
from persistence.core.canonical import canonical_json
from persistence.core.errors import AuthorityRefusal
from persistence.core.identity import SessionId

from .._authority import session_runtime
from ..conversation.sessions import _now
from ..conversation.transcript_store import advance_conversation_generation
from . import admissions_rows as rows
from . import mutation_guards as guards
from .mutation_commands import DEFERRED_PICK_PREFIX, VIEWER_CLOSE_PREFIX

_RECEIPT_PREFIX = "gateway.mutation.v1."


def _derive(chain: GenerationChain, generation: int) -> str:
    if isinstance(chain, ApiGenerations):
        return str(api_generation_id(chain.session_id, generation))
    if isinstance(chain, RoomGenerations):
        return str(room_session_id(chain.identity, generation))
    return str(local_session_id(chain.profile_id, chain.principal_id, generation_request_id(chain.base_request_id, generation)))


def _receipts(meta: Mapping[Any, Any]) -> list[dict[str, Any]]:
    out = []
    for key, value in meta.items():
        if isinstance(key, str) and key.startswith(_RECEIPT_PREFIX):
            try:
                out.append(json.loads(value))
            except (TypeError, ValueError):
                continue
    return out


def _ledger_status(authority: Mapping[str, Any], session_id: str) -> str | None:
    """The status of the first non-terminal admission targeted at the id, else of the first non-terminal worker on it."""
    for table, column in ((authority["admissions"], "target_session_id"), (authority["workers"], "session_id")):
        for row in table.values():
            if row[column] == session_id and row["status"] != "terminal":
                return str(row["status"])
    return None


class RuntimeMetaMixin:
    """The runtime verbs other than the receipted mutation; mixed into ``InMemorySessionRuntimeRepository``."""

    def begin_epoch(self, *, instance_id: str) -> int:
        rows.text(instance_id)
        return int(self._state().begin_epoch(instance_id))

    def current_epoch(self) -> int | None:
        epoch = self._state().snapshot["epoch"]
        return epoch or None

    def has_mutation_receipt(self, principal_id: str, session_id: str, request_id: str) -> bool:
        for value in (principal_id, session_id, request_id):
            rows.text(value)
        key, _digest = mutation_receipt_key(principal_id, session_id, request_id, operation="model", payload={}, expected_revision=0)
        return key in self._state().snapshot["meta"]

    def _replay(self, meta: Mapping[Any, Any], key: str, digest: str) -> Any:
        """The stored receipt's result for ``key``, None when absent; another digest is ``admission_conflict``."""
        stored = meta.get(key)
        if stored is None:
            return None
        receipt = json.loads(stored)
        if receipt["digest"] != digest:
            raise AuthorityRefusal("admission_conflict")
        return receipt["result"]

    def defer_model_pick(self, epoch: int, *, key: str, digest: str, session_id: str, pick: DeferredPick) -> dict[str, Any]:
        parse_mutation(pick.operation, dict(pick.payload))
        entry = {"principal_id": pick.principal_id, "request_id": pick.request_id, "payload": rows.encode(dict(pick.payload)),
                 "application_id": "deferred-pick:" + pick.request_id}
        state = self._state()
        with state.transaction() as staged:
            meta = staged.authority["meta"]
            state.require_epoch(staged.authority, epoch)
            replay = self._replay(meta, key, digest)
            if replay is not None:
                return replay
            rows.require_session(staged.scope, session_id)
            runtime = session_runtime(staged.authority, session_id)
            slot = DEFERRED_PICK_PREFIX + session_id
            current = json.loads(meta[slot]) if slot in meta else {}
            if pick.operation == "reasoning":
                current.get("payload", {}).pop("reasoning", None)
                value, picked = {**current, "reasoning": entry}, {"effort": pick.payload["effort"]}
            else:
                older = "reasoning" in current and "reasoning" not in pick.payload
                value = {**entry, **({"reasoning": current["reasoning"]} if older else {})}
                picked = {"model": pick.shown or pick.payload["model"]}
            result = {"session_id": session_id, "operation": pick.operation, "status": "deferred", **picked,
                      "revision": runtime["revision"], "execution_generation": runtime["generation"]}
            meta[slot] = canonical_json(value)
            meta[key] = canonical_json({"digest": digest, "result": result})
            return json.loads(canonical_json(result))

    def deferred_pick(self, session_id: str) -> StoredDeferredPick | None:
        stored = self._state().snapshot["meta"].get(DEFERRED_PICK_PREFIX + session_id)
        return None if stored is None else StoredDeferredPick(pick=json.loads(stored), stored=stored)

    def forget_deferred_pick(self, session_id: str, *, stored: str) -> None:
        with self._state().transaction() as staged:
            if staged.authority["meta"].get(DEFERRED_PICK_PREFIX + session_id) == stored:
                del staged.authority["meta"][DEFERRED_PICK_PREFIX + session_id]

    def mark_deferred_pick_asked(self, session_id: str, *, stored: str, confirm: str) -> None:
        try:
            value = canonical_json({**json.loads(stored), "awaiting_confirm": confirm})
        except (TypeError, ValueError) as exc:
            raise AuthorityRefusal("invalid_params") from exc
        with self._state().transaction() as staged:
            if staged.authority["meta"].get(DEFERRED_PICK_PREFIX + session_id) == stored:
                staged.authority["meta"][DEFERRED_PICK_PREFIX + session_id] = value

    def replace_deferred_pick(self, session_id: str, *, stored: str, pick: Mapping[str, Any]) -> str:
        value = canonical_json(dict(pick))
        with self._state().transaction() as staged:
            if staged.authority["meta"].get(DEFERRED_PICK_PREFIX + session_id) == stored:
                staged.authority["meta"][DEFERRED_PICK_PREFIX + session_id] = value
        return value

    def _fenced_mount(self, authority: dict[str, Any], tables: Mapping[str, Any], session_id: str, params: ChatPickParams) -> dict[str, int]:
        """The session's runtime entry when its revision and generation match the pick's expectations; ``revision_conflict`` or a stale generation otherwise."""
        rows.require_session(tables, session_id)
        runtime = authority["sessions"].get(session_id, {"revision": 0, "generation": 0})
        if runtime["revision"] != params.expected_revision:
            raise rows.revision_conflict(session_id)
        if runtime["generation"] != params.expected_generation:
            raise rows.stale("session", session_id)
        return runtime

    def prepare_chat_pick(self, epoch: int, *, key: str, digest: str, session_id: str, params: ChatPickParams) -> dict[str, Any] | SessionRuntimeState:
        state = self._state()
        authority = state.snapshot
        state.require_epoch(authority, epoch)
        replay = self._replay(authority["meta"], key, digest)
        if replay is not None:
            return replay
        runtime = self._fenced_mount(authority, self._tables(), session_id, params)
        return SessionRuntimeState(revision=runtime["revision"], generation=runtime["generation"])

    def commit_chat_pick(self, epoch: int, *, key: str, digest: str, session_id: str, params: ChatPickParams, result: Mapping[str, Any]) -> dict[str, Any]:
        committed = rows.encode(dict(result))
        state = self._state()
        with state.transaction() as staged:
            state.require_epoch(staged.authority, epoch)
            replay = self._replay(staged.authority["meta"], key, digest)
            if replay is not None:
                return replay
            self._fenced_mount(staged.authority, staged.scope, session_id, params)
            runtime = session_runtime(staged.authority, session_id)
            runtime["revision"] += 1
            committed["revision"] = runtime["revision"]
            staged.authority["meta"][key] = canonical_json({"digest": digest, "result": committed})
            return committed

    def _picks_on(self, authority: Mapping[str, Any], session_id: str, before: Any = None) -> set[str]:
        """The policy operations picked on the session (before revision ``before`` when given): committed receipts plus the local receipt's ``picks``, else the picks inherited from the branch parent."""
        results = [r.get("result") for r in _receipts(authority["meta"]) if isinstance(r.get("result"), dict)]
        picked = {r["operation"] for r in results if r.get("operation") in ("model", "reasoning", "workspace") and (r.get("status") or "") != "deferred"
                  and r.get("session_id") == session_id and (before is None or (r.get("revision") is not None and r["revision"] < before))}
        receipt = rows.local_receipt(authority, session_id)
        if receipt is None:
            return picked
        if "picks" in receipt:
            return picked | set(receipt["picks"])
        branched = next((r for r in results if r.get("operation") == "branch" and r.get("branched_session_id") == session_id), None)
        return picked | self._picks_on(authority, branched.get("session_id"), branched.get("revision")) if branched else picked

    def policy_picks(self, session_id: str) -> set[str]:
        return self._picks_on(self._state().snapshot, session_id)

    def commit_refreshed_policy(self, epoch: int, *, session_id: str, old: Mapping[str, Any], new: Mapping[str, Any], generation: int) -> int:
        state = self._state()
        with state.transaction() as staged:
            authority, tables = staged.authority, staged.scope
            state.require_epoch(authority, epoch)
            snapshot = guards.local_snapshot(self._backend, authority, tables, session_id)
            guards.require_idle(self._backend, authority, tables, list(dict.fromkeys([session_id, snapshot.target])))
            row = tables.get("sessions", {}).get(session_id)
            if snapshot.receipt["policy"] != old or row is None or authority["sessions"].get(session_id, {}).get("generation", 0) != generation:
                raise rows.revision_conflict(session_id)
            rows.local_receipt(authority, session_id)["policy"] = rows.encode(dict(new))
            runtime = session_runtime(authority, session_id)
            runtime["generation"] += 1
            return runtime["generation"]

    def store_viewer_close(self, epoch: int, *, session_id: str, record: str) -> None:
        state = self._state()
        with state.transaction() as staged:
            state.require_epoch(staged.authority, epoch)
            staged.authority["meta"][VIEWER_CLOSE_PREFIX + session_id] = record

    def forget_viewer_close(self, epoch: int, *, session_id: str, record: str) -> bool:
        state = self._state()
        with state.transaction() as staged:
            state.require_epoch(staged.authority, epoch)
            if staged.authority["meta"].get(VIEWER_CLOSE_PREFIX + session_id) != record:
                return False
            del staged.authority["meta"][VIEWER_CLOSE_PREFIX + session_id]
            return True

    def viewer_close(self, session_id: str) -> str | None:
        return self._state().snapshot["meta"].get(VIEWER_CLOSE_PREFIX + session_id)

    def viewer_closes(self) -> list[tuple[str, str]]:
        meta = self._state().snapshot["meta"]
        return [(key[len(VIEWER_CLOSE_PREFIX):], meta[key]) for key in sorted(k for k in meta if isinstance(k, str) and k.startswith(VIEWER_CLOSE_PREFIX))]

    def retired_session(self, session_id: str) -> bool:
        return session_id in self._state().snapshot["retired"]

    def first_generation(self, chain: GenerationChain, *, pinned: PinnedIdentity | None = None, start: int = 0) -> GenerationPick:
        authority = self._state().snapshot
        sessions = self._tables().get("sessions", {})
        generation = start
        while True:
            sid = _derive(chain, generation)
            if sid not in authority["retired"]:
                row = sessions.get(sid)
                return GenerationPick(generation=generation, session_id=sid, row=None if row is None else GenerationRow(title=row.get("title"), source=row.get("source"), archived=bool(row.get("archived"))))
            if pinned is not None and rows._identity_key(pinned.principal_id, sid, pinned.request_id) in authority["meta"]:
                return GenerationPick(generation=generation, session_id=sid, row=None)
            generation += 1

    def expiring_retirements(self, session_ids: Sequence[str]) -> set[str]:
        retired = self._state().snapshot["retired"]
        return {sid for sid in session_ids if (retired.get(sid) or {}).get("expires")}

    def _end(self, row: dict[str, Any], reason: str) -> None:
        row["ended_at"] = _now()
        row["end_reason"] = reason
        advance_conversation_generation(self._backend, self._backend.scope, row["session_id"], reason)

    def end_unless_ledger_busy(self, session_id: str, *, session_key: str | None, end_reason: str) -> bool:
        authority, tables = self._state().snapshot, self._tables()
        row = tables.get("sessions", {}).get(session_id)
        ids = {session_id, str(session_key or "")}
        if row is not None and str(row.get("chat_id") or "").startswith("local-"):
            ids.add(str(row["chat_id"]))
        ids.add(rows.local_lineage_owner(authority, tables, session_id))
        if any(_ledger_status(authority, sid) is not None for sid in filter(None, ids)):
            return False
        if row is not None and row.get("ended_at") is None:
            self._end(row, end_reason)
        return True

    def pending_ledger_status(self, session_key: str) -> str | None:
        return _ledger_status(self._state().snapshot, session_key)

    def delete_footprint(self, session_id: SessionId, owner: Any) -> DeleteFootprint:
        tables = self._tables()
        targets = guards.delete_targets(self._backend, self._state().snapshot, tables, str(session_id), owner)
        doomed = set(targets)
        children = [sid for sid, row in tables.get("sessions", {}).items() if row["parent_session_id"] in doomed]
        return DeleteFootprint(targets=tuple(SessionId(sid) for sid in targets), children=tuple(SessionId(sid) for sid in children))

    def end_after_settlement(self, epoch: int, *, session_id: str, admission_id: str, generation: int, admission_epoch: int, reason: str) -> bool:
        authority, tables = self._state().snapshot, self._tables()
        row = tables.get("sessions", {}).get(session_id)
        if row is None or row.get("ended_at") is not None or authority["sessions"].get(session_id, {}).get("generation", 0) != generation:
            return False
        if authority["epoch"] == 0 or authority["epoch"] != epoch:
            return False
        settled = authority["admissions"].get(admission_id)
        if settled is None or settled["target_session_id"] != session_id or settled["status"] != "terminal" or settled["generation"] != generation or settled["owner_epoch"] != admission_epoch:
            return False
        if any(r["target_session_id"] == session_id and r["status"] != "terminal" for r in authority["admissions"].values()):
            return False
        self._end(row, reason)
        return True
