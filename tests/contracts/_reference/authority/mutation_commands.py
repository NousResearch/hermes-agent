"""The command handlers ``mutate`` runs inside its write.

Each handler returns ``(affected ids, projection)``. Reference limits: ``reset``, ``branch`` and ``compress`` run their fences (the local receipt, its lineage, the idle and advanceable guards, the prepared snapshot and the branch's prepared child) and then refuse ``CapabilityUnsupported``: their publication builds gateway routing entries and compression bodies the in-memory reference does not model.
"""

from __future__ import annotations

import copy
import json
import time
from typing import Any, Mapping

from persistence.authority.bindings import local_session_id
from persistence.authority.runtime import COMPACTED_TARGET_DETAIL, MutationCommand, PreparedMutation
from persistence.core.canonical import canonical_json
from persistence.core.conventions import RowVisibility
from persistence.core.errors import AuthorityRefusal, CapabilityUnsupported
from persistence.core.identity import SessionId
from persistence.housekeeping.maintenance import IMPORT_MAX_SESSIONS
from session_rules.imports import validate_import_payload
from session_rules.legacy_local import bindings_agree
from session_rules.turns import typed_prompt_content

from .._authority import session_runtime
from ..conversation import _turn_policy as policy
from ..conversation.transcript_store import _messages_namespace, _replay_dict, advance_conversation_generation, insert_rows
from . import admissions_rows as rows
from . import mutation_guards as guards
from .bindings import binding_text

#: Meta key prefixes of the deferred-pick slot and of a viewer close.
DEFERRED_PICK_PREFIX = "gateway.deferred_pick.v1."
VIEWER_CLOSE_PREFIX = "gateway.viewer_close.v1:"


def _ctx(repo: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    return repo._backend, repo._state().snapshot, repo._tables()


def _metadata(repo: Any, session_id: str, payload: Mapping[str, Any]) -> tuple[set[str], dict[str, Any]]:
    store = repo._backend
    affected: set[str] = set()
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "title":
            if store.sessions.set_title(SessionId(session_id), value):
                affected.add(session_id)
            result[key] = repo._tables()["sessions"][session_id]["title"]
            continue
        column = "last_read_at" if key == "unread" else key
        values: dict[str, Any] = {column: (0.0 if value else time.time()) if key == "unread" else bool(value)}
        if key == "archived":
            values["auto_archived"] = False
        affected.update(store.sessions._set_lineage_flag(store.scope, SessionId(session_id), values))
        if key == "archived" and value:
            affected.update(store.sessions._retire_canonical_bot_chat(store.scope, session_id))
        result[key] = value
    return affected, result


def _delete(repo: Any, session_id: str, media_root: str | None, owner: Any) -> tuple[set[str], dict[str, Any]]:
    """The delete handler; the retired admissions' hosted documents are those under ``media_root``, the walk reads stored owners under ``owner``."""
    store, authority, tables = _ctx(repo)
    targets = guards.delete_targets(store, authority, tables, session_id, owner)
    guards.require_idle(store, authority, tables, targets)
    for sid in targets:
        advance_conversation_generation(store, store.scope, sid, "session_reset")
    tables = repo._tables()
    doomed = set(targets)
    for row in tables.get("sessions", {}).values():
        if row.get("parent_session_id") in doomed and row["session_id"] not in doomed:
            row["parent_session_id"] = None
            session_runtime(authority, row["session_id"])["revision"] += 1
    type(store.sessions)._retire_staged(tables, authority, targets, media_root)
    for sid in targets:
        authority["meta"].pop(VIEWER_CLOSE_PREFIX + sid, None)
        tables.get("messages", {}).get("transcripts", {}).pop(sid, None)
    return set(), {"deleted_ids": targets}


def _summarized_by_compaction(repo: Any, physical: str, message_id: int) -> bool:
    """Whether ``message_id`` is a user row compaction summarized: archived on the physical target, or active or archived on another session of its lineage."""
    store, _authority, tables = _ctx(repo)
    for sid, transcript in tables.get("messages", {}).get("transcripts", {}).items():
        row = transcript.find(message_id)
        if row is None:
            continue
        if row.message.role != "user":
            return False
        if sid == physical:
            return row.visibility is RowVisibility.COMPACTION_ARCHIVED
        if row.visibility not in (RowVisibility.ACTIVE, RowVisibility.COMPACTION_ARCHIVED):
            return False
        sessions = tables.get("sessions", {})
        return physical in sessions and sid in store.sessions._lineage_ids(sessions, physical)
    return False


def _rewind(repo: Any, session_id: str, payload: Mapping[str, Any]) -> tuple[set[str], dict[str, Any]]:
    """The rewind handler and its transcript rewind."""
    from ..conversation.messages_turns import _scaffold_message

    store, authority, tables = _ctx(repo)
    physical = rows.physical_target(store, authority, tables, session_id)
    guards.require_idle(store, authority, tables, list(dict.fromkeys([session_id, physical])))
    target_id = payload["target_message_id"]
    transcript = tables.get("messages", {}).get("transcripts", {}).get(physical)
    row = transcript.find(target_id) if transcript is not None else None
    if row is None or row.visibility is not RowVisibility.ACTIVE or row.message.role != "user":
        if _summarized_by_compaction(repo, physical, target_id):
            raise AuthorityRefusal("invalid_params", field="target_message_id", detail=COMPACTED_TARGET_DETAIL)
        raise AuthorityRefusal("invalid_params", field="target_message_id", detail="target user message is no longer in session history")
    message = row.message
    handoff, live = policy.split_user_originated_turn({"role": message.role, "content": message.content, "display_kind": message.display_kind,
                                                       "display_metadata": message.display_metadata, "_compressed_summary": message.compressed_summary})
    carrier = live if handoff is not None else None
    preserve = carrier is not None or payload.get("preserve_compaction_handoff", False)
    if preserve and handoff is None:
        from persistence.conversation.messages import RewindTargetUnavailable

        raise RewindTargetUnavailable("preserve_compaction_handoff requires an active composite carrier")
    target = _replay_dict(row, include_row_ids=True, include_summary_markers=True, stamp_snapshot=False)
    rewound = [r for r in transcript.active() if r.id >= row.id]
    for r in rewound:
        r.visibility = RowVisibility.REWOUND
    scaffold = insert_rows(tables["messages"], transcript, [_scaffold_message(handoff)])[0] if preserve else None
    transcript.rewind_count += 1
    transcript.recount()
    if carrier is not None:
        target["content"] = carrier.get("content")
    target["content"] = typed_prompt_content(target.get("content"), target)
    session_runtime(authority, session_id)["generation"] += 1
    head = max((r.id for r in transcript.active()), default=None)
    return {session_id, physical}, {"rewound_count": len(rewound), "target_message": target, "new_head_id": head,
                                    "replacement_message_id": scaffold.id if scaffold is not None else None}


def _import(repo: Any, session_id: str, payload: Mapping[str, Any], owner: Any) -> tuple[set[str], dict[str, Any]]:
    """The import handler over the maintenance reference's import steps; a parent outside the payload links when its binding names the child's owner under ``owner``."""
    from .mutation import _history_key

    store, authority, tables = _ctx(repo)
    items = payload["sessions"]
    if len(items) > IMPORT_MAX_SESSIONS:
        raise AuthorityRefusal("invalid_params")
    normalized, errors = validate_import_payload(items)
    if errors or not normalized or session_id != normalized[0]["session"]["id"]:
        raise AuthorityRefusal("invalid_params")
    imported: list[str] = []
    skipped: list[str] = []
    parents: list[tuple[str, str]] = []
    _messages_namespace(store, store.scope)
    sessions = tables.setdefault("sessions", {})
    for item in normalized:
        raw = item["session"]
        sid = raw["id"]
        if sid in authority["retired"]:
            raise AuthorityRefusal("admission_conflict", field="sessions", detail=f"session {sid} was deleted here; a deleted session is not restored")
        if sid in sessions:
            skipped.append(sid)
            continue
        store.maintenance._insert_imported(tables, raw, item["messages"], sid)
        if raw.get("parent_session_id"):
            parents.append((sid, raw["parent_session_id"]))
        imported.append(sid)
    meta = authority["meta"]
    fresh = set(imported)
    foreign = [(sid, parent) for sid, parent in parents if parent not in fresh and (
        meta.get(_history_key(parent)) is None
        or not bindings_agree(binding_text(meta.get(_history_key(parent))), binding_text(meta.get(_history_key(sid))), owner))]
    detached = store.maintenance._attach_parents(sessions, [link for link in parents if link not in foreign]) + len(foreign)
    return set(imported), {"ok": True, "imported": len(imported), "skipped": len(skipped), "imported_ids": imported, "skipped_ids": skipped,
                           "detached": detached, "errors": []}


def _commit_policy(repo: Any, session_id: str, prepared: PreparedMutation | None) -> tuple[Any, int]:
    """Validate the prepared snapshot, store the prepared policy on the local receipt and bump the execution generation; ``(snapshot, generation)``."""
    store, authority, tables = _ctx(repo)
    snapshot = guards.validate_prepared(store, authority, tables, session_id, prepared)
    rows.local_receipt(authority, session_id)["policy"] = copy.deepcopy(dict(prepared.values["policy"]))
    entry = session_runtime(authority, session_id)
    entry["generation"] += 1
    return snapshot, entry["generation"]


def _model(repo: Any, session_id: str, payload: Mapping[str, Any], prepared: PreparedMutation | None) -> tuple[set[str], dict[str, Any]]:
    """The model handler: the target row takes the chosen model and provider; a deferred pick keeps only its reasoning level, dropped when the payload sets one."""
    snapshot, generation = _commit_policy(repo, session_id, prepared)
    chosen = prepared.values["policy"]
    model, config = chosen["model"], json.loads(chosen["config_json"])["model"]
    row = repo._tables()["sessions"][snapshot.target]
    row["model"] = model
    row["model_config"] = {**(row.get("model_config") or {}), "provider": config["provider"]}
    meta = repo._state().snapshot["meta"]
    key = DEFERRED_PICK_PREFIX + session_id
    level = json.loads(meta[key]).get("reasoning") if key in meta and "reasoning" not in payload else None
    if level is None:
        meta.pop(key, None)
    else:
        meta[key] = canonical_json({"reasoning": level})
    return {session_id, snapshot.target}, {"model": model, "provider": config["provider"], "execution_generation": generation,
                                           **({"effort": payload["reasoning"]} if "reasoning" in payload else {})}


def _reasoning(repo: Any, session_id: str, payload: Mapping[str, Any], prepared: PreparedMutation | None) -> tuple[set[str], dict[str, Any]]:
    """The reasoning handler: the deferred pick loses its reasoning level, and the slot goes when it holds no application."""
    snapshot, generation = _commit_policy(repo, session_id, prepared)
    meta = repo._state().snapshot["meta"]
    key = DEFERRED_PICK_PREFIX + session_id
    if key in meta:
        pick = json.loads(meta[key])
        pick.pop("reasoning", None)
        pick.get("payload", {}).pop("reasoning", None)
        if "application_id" in pick:
            meta[key] = canonical_json(pick)
        else:
            del meta[key]
    return {session_id, snapshot.target}, {"effort": payload["effort"], "execution_generation": generation}


def _workspace(repo: Any, session_id: str, payload: Mapping[str, Any], prepared: PreparedMutation | None) -> tuple[set[str], dict[str, Any]]:
    """The workspace handler: every row of the local conversation and the target take the cwd and git metadata, the git generation bumped."""
    snapshot, generation = _commit_policy(repo, session_id, prepared)
    cwd, git = prepared.values["policy"]["cwd"], prepared.values["git"]
    authority, tables = repo._state().snapshot, repo._tables()
    owner = rows.local_lineage_owner(authority, tables, session_id)
    receipt = rows.local_receipt(authority, owner)
    lineage = receipt.get("lineage") if isinstance(receipt, dict) else None
    ids = [owner, *(sid for sid in lineage if isinstance(sid, str))] if isinstance(lineage, list) else [session_id]
    for sid in dict.fromkeys([*ids, snapshot.target]):
        row = tables.get("sessions", {}).get(sid)
        if row is not None:
            row.update(cwd=cwd, git_branch=git["branch"] or None, git_repo_root=git["root"] or None, git_metadata_generation=(row.get("git_metadata_generation") or 0) + 1)
    return {session_id, snapshot.target}, {"cwd": cwd, "branch": git["branch"] or None, "git_repo_root": git["root"] or None, "execution_generation": generation}


def _editor_mcp(repo: Any, session_id: str, payload: Mapping[str, Any], prepared: PreparedMutation | None) -> tuple[set[str], dict[str, Any]]:
    snapshot, generation = _commit_policy(repo, session_id, prepared)
    return {session_id, snapshot.target}, {"tools_changed": prepared.values["tools_changed"], "mcp_servers": prepared.values["mcp_servers"], "execution_generation": generation}


def _edit_policy(repo: Any, session_id: str, payload: Mapping[str, Any], prepared: PreparedMutation | None) -> tuple[set[str], dict[str, Any]]:
    snapshot, generation = _commit_policy(repo, session_id, prepared)
    return {session_id, snapshot.target}, {"edit_approval_policy": payload["policy"], "execution_generation": generation}


def _local_receipt_or_refuse(repo: Any, session_id: str) -> tuple[dict[str, Any], str]:
    store, authority, tables = _ctx(repo)
    receipt = rows.local_receipt(authority, session_id)
    if receipt is None:
        raise AuthorityRefusal("runtime_coordination_required")
    return receipt, rows.physical_target(store, authority, tables, session_id)


def _reset(repo: Any, session_id: str, payload: Mapping[str, Any], prepared: PreparedMutation | None) -> tuple[set[str], dict[str, Any]]:
    """The reset handler's fences; the publication is a reference limit."""
    _receipt, target = _local_receipt_or_refuse(repo, session_id)
    store, authority, tables = _ctx(repo)
    lineage = rows.runtime_lineage(store, authority, tables, session_id)
    for worker in authority["workers"].values():
        if worker["session_id"] in lineage and worker["status"] == "unknown" and not str(worker.get("execution_id", "")).startswith("admission-worker:"):
            worker["status"] = "terminal"
    guards.require_target_advanceable(store, authority, tables, list(dict.fromkeys([session_id, target])))
    raise CapabilityUnsupported("mutation.reset")


def _branch(repo: Any, session_id: str, payload: Mapping[str, Any], prepared: PreparedMutation | None) -> tuple[set[str], dict[str, Any]]:
    """The branch handler's fences; the publication is a reference limit."""
    receipt, target = _local_receipt_or_refuse(repo, session_id)
    store, authority, tables = _ctx(repo)
    guards.require_idle(store, authority, tables, list(dict.fromkeys([session_id, target])))
    parent_policy = receipt["policy"]
    if prepared is None:
        if parent_policy.get("editor_mcp_json"):
            raise AuthorityRefusal("runtime_coordination_required")
    else:
        child = local_session_id(receipt["profile_id"] if "profile_id" in receipt else store.owner_profile, receipt["principal_id"], prepared.values["request_id"])
        if child != prepared.values["child"] or parent_policy.get("editor_mcp_json") != prepared.values["parent_editor_mcp_json"]:
            raise rows.revision_conflict(session_id)
    through = payload.get("through_message_id")
    if through is not None:
        transcript = tables.get("messages", {}).get("transcripts", {}).get(target)
        row = transcript.find(through) if transcript is not None else None
        if row is None or row.visibility is not RowVisibility.ACTIVE:
            raise AuthorityRefusal("invalid_params")
    raise CapabilityUnsupported("mutation.branch")


def _compress(repo: Any, session_id: str, payload: Mapping[str, Any], prepared: PreparedMutation | None) -> tuple[set[str], dict[str, Any]]:
    """The compress handler's fence; the publication is a reference limit."""
    store, authority, tables = _ctx(repo)
    guards.validate_prepared(store, authority, tables, session_id, prepared)
    raise CapabilityUnsupported("mutation.compress")


_PREPARED = {"model": _model, "reasoning": _reasoning, "workspace": _workspace, "editor_mcp": _editor_mcp, "edit_policy": _edit_policy,
             "reset": _reset, "branch": _branch, "compress": _compress}


def apply_command(repo: Any, session_id: str, command: MutationCommand, prepared: PreparedMutation | None, media_root: str | None, owner: Any = None) -> tuple[set[str], dict[str, Any]]:
    """The handler of ``command``'s operation; only a delete reads ``media_root``, a delete and an import ``owner``."""
    payload = rows.encode(command.payload())
    if command.operation in _PREPARED:
        return _PREPARED[command.operation](repo, session_id, payload, prepared)
    if command.operation == "delete":
        return _delete(repo, session_id, media_root, owner)
    if command.operation == "import":
        return _import(repo, session_id, payload, owner)
    if command.operation == "rewind":
        return _rewind(repo, session_id, payload)
    return _metadata(repo, session_id, payload)
