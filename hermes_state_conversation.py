"""Replay projection from durable transcript rows."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Tuple
from agent.context_compressor import _DB_PERSISTED_MARKER as _DB_PERSISTED_MARKER_KEY
from agent.message_metadata import DB_ROW_SNAPSHOT, TOOL_CALL_UID, index_tool_call_uids, resolve_tool_call_uid
from hermes_state_common import _json_or
from hermes_state_identity import _restore_identity_columns
logger = logging.getLogger("hermes_state")


def _restore_assistant_fields(row, msg, unindexed_tool_owners):
    msg.update((col, row[col]) for col in ("finish_reason", "reasoning") if row[col])
    if row["reasoning_content"] is not None:
        msg["reasoning_content"] = row["reasoning_content"]
    msg.update(
        (col, _json_or(row[col], None, f"Failed to deserialize {col}, falling back to None"))
        for col in ("reasoning_details", "codex_reasoning_items", "codex_message_items") if row[col])
    if msg.get("tool_calls"):
        unindexed_tool_owners.append(msg)


def rows_to_conversation(self, rows, *, session_id: str, include_ancestors: bool, repair_alternation: bool,
                          include_row_ids: bool = False,
                          include_summary_markers: bool = False) -> List[Dict[str, Any]]:
    """Decode fetched rows (ordered by id, pre-filtered) into OpenAI format, stable key order. Every dict is
    stamped ``_DB_PERSISTED_MARKER_KEY`` (born durable) so an identity-losing handoff never re-appends the
    transcript on flush. Unaddressed live-replay projections also carry the stored-row CAS digest: if a later rewrite
    loses its physical ``_row_id``, logical ``message_uid`` can recover the row without guessing by
    mutable payload while the digest still fences a concurrent winner. ``_row_id`` is opt-in (gateway
    reactions); reasoning restored on assistant rows only; ``api_content`` VERBATIM (no sanitize/strip)
    so replay keeps the provider prompt cache byte-stable."""
    from hermes_state import _strip_background_review_harness, _strip_stale_tool_call_markers
    # Runtime import avoids the transcript_repair -> hermes_state_messages module cycle.
    from agent.transcript_repair import transcript_row_snapshot
    # Only the unaddressed live replay gets the digest: row-addressed loaders (include_row_ids) keep the
    # legacy resumed-dict path, whose rewrite never re-writes columns the projection does not decode
    # (a CAS-match rewrite of a resumed row would otherwise null token_count).
    stamp_snapshot = repair_alternation and not include_row_ids
    messages = []
    exact_user_clones: Dict[Tuple[Any, str], Dict[str, Any]] = {}
    tool_uid_index: Dict[str, str] = {}  # pairing-id variant -> uid, from the assistant rows indexed so far
    # Assistant rows since the last user row not yet indexed: only a result without a stored uid (an older
    # build's) needs the index, so rows this build wrote never pay for it. Indexed in order: same shadowing.
    unindexed_tool_owners: List[Dict[str, Any]] = []
    for row in rows:
        content = self._loaded_view_content(row["role"], self._decode_content(row["content"]))
        # Underscore-prefixed like ``_row_id``: transports strip it before the wire; compression's
        # assembly copies strip it so rotated child handoffs still flush (_fresh_compaction_message_copy).
        msg = {"role": row["role"], "content": content, _DB_PERSISTED_MARKER_KEY: True}
        if stamp_snapshot:
            msg[DB_ROW_SNAPSHOT] = transcript_row_snapshot(row)
        # Born durable (#92231): this dict is materialized FROM a durable row, so stamp the persistence
        # marker at the source instead of relying on every restore caller to thread the loaded list back
        # through a flush as ``conversation_history=`` — any identity-losing handoff (compression's
        # durable-snapshot adoption, incremental persists with no history arg) would otherwise re-append
        # the ENTIRE transcript on flush.
        if include_row_ids and row["id"] is not None:
            msg["_row_id"] = row["id"]
        # Durable identity and topic label are internal replay metadata, never provider payload.
        _restore_identity_columns(row, msg)
        if row["topic_id"] is not None:
            msg["_topic_id"] = row["topic_id"]
        msg.update((col, row[col]) for col in ("api_content", "display_kind") if row[col])
        if row["display_metadata"] and (decoded := self._decode_display_metadata(row["display_metadata"])) is not None:
            msg["display_metadata"] = decoded
        if include_summary_markers and row["_compressed_summary"]:
            msg["_compressed_summary"] = True
        msg.update(
            (col, row[col]) for col in ("timestamp", "tool_call_id", "tool_name", "effect_disposition") if row[col])
        if row["tool_calls"]:
            msg["tool_calls"] = _json_or(
                row["tool_calls"], [], "Failed to deserialize tool_calls in conversation replay, falling back to []")
        if row["platform_message_id"]:  # platform-side id exposed as ``message_id`` (JSONL transcript compat)
            msg["message_id"] = row["platform_message_id"]
        if row["observed"]:
            msg["observed"] = True
        if row["role"] == "assistant":
            _restore_assistant_fields(row, msg, unindexed_tool_owners)
        elif row["role"] == "user":
            tool_uid_index.clear()  # a result never pairs across a user turn
            unindexed_tool_owners.clear()
        elif row["role"] == "tool" and not row["tool_call_uid"] and row["tool_call_id"]:
            # No stored uid (a result appended by an older build or a lone append): the one its assistant
            # row named. Rows are read in id order, so the call always precedes its result. Provider ids
            # repeat: a later row's calls shadow an earlier occurrence's, and a row without a map (an older
            # writer's) leaves its results unpaired rather than mispaired.
            for owner in unindexed_tool_owners:
                index_tool_call_uids(tool_uid_index, owner)
            unindexed_tool_owners.clear()
            if tool_uid := resolve_tool_call_uid(tool_uid_index, row["tool_call_id"]):
                msg[TOOL_CALL_UID] = tool_uid
        if include_ancestors:
            skip, exact_clone_key = self._dedupe_replayed_user(messages, msg, exact_user_clones)
            if skip:
                continue
            if exact_clone_key is not None:
                exact_user_clones[exact_clone_key] = msg
        messages.append(msg)
    # Defense-in-depth: strip a background-review harness turn (older builds shared the parent's
    # session_id) plus its curator reply, and bare tool-call marker content ("[memory]") persisted as an answer.
    messages = _strip_stale_tool_call_markers(_strip_background_review_harness(messages))
    if repair_alternation and messages:
        from agent.agent_runtime_helpers import repair_message_sequence
        repaired = repair_message_sequence(None, messages)
        if repaired:
            logger.info("Repaired %d message-alternation violation(s) while "
                "restoring session %s — durable transcript kept them, "
                "see repair_message_sequence", repaired, session_id)
    return messages
