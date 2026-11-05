"""The worker family's shared vocabulary: the execution-id separator, the message fields the worker wire carries, the row annotation keys, the sidecar and constructor key sets and the receipt bounds.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

__all__ = [
    "CHILD_SEPARATOR",
    "HANDOFF_MESSAGE_FIELDS",
    "RECEIPT_PAYLOAD_MAX_BYTES",
    "ROW_ANNOTATION_KEYS",
    "SIDECAR_KEYS",
    "TRANSCRIPT_BATCH_MAX",
    "WORKER_CONSTRUCTOR_CONFIG_KEYS",
    "WORKER_MESSAGE_FIELDS",
    "row_annotations",
]

#: A delegated child's execution id is its parent's id, this separator and a random suffix, so every descendant of an execution shares one id prefix.
CHILD_SEPARATOR = "/child:"

#: The persistence-only keys every persisted message dict may carry (agent/message_metadata.py:42-52): row identity, repair bookkeeping and the alternation repair's row counts.
_PERSISTENCE_ONLY_FIELDS = frozenset({
    "timestamp", "display_kind", "display_metadata", "_row_id", "_submit_row_session_id", "_merged_turn_prefix",
    "message_uid", "_absorbed_message_uids", "_tool_call_uids", "_tool_call_uid",
    "_merged_durable_rows", "_unnamed_durable_rows", "_retired_durable_rows",
    "_db_row_snapshot", "_canonical_row", "_restored_from_store",
})

#: The keys a transcript row on the worker wire may carry; the owner refuses any other key.
WORKER_MESSAGE_FIELDS = frozenset({
    "role", "content", "tool_name", "tool_calls", "tool_call_id", "token_count",
    "finish_reason", "reasoning", "reasoning_content", "reasoning_details",
    "codex_reasoning_items", "codex_message_items", "platform_message_id", "message_id",
    "observed", "effect_disposition", "_compressed_summary", "timestamp", "api_content",
    "display_kind", "display_metadata", "_row_id", "_canonical_content", "_db_row_snapshot", "_canonical_row", "data_use",
}) | _PERSISTENCE_ONLY_FIELDS | {"_absorbed_row_ids"}

#: Compaction and publication handoff rows also carry the summary's user-turn flag.
HANDOFF_MESSAGE_FIELDS = WORKER_MESSAGE_FIELDS | {"_compressed_summary_has_user_turn"}

#: The row state the owner stamps on each written message and returns to the worker, ``None`` meaning absent.
ROW_ANNOTATION_KEYS = ("_row_id", "timestamp", "_db_row_snapshot", "_canonical_row", "message_uid", "_tool_call_uids", "_tool_call_uid")

#: The ``model_config`` keys a worker may patch (``patch_session_sidecars`` and a compaction's ``model_config_patch``).
SIDECAR_KEYS = frozenset({"_usage_anchor", "_proactive_prune_rearm_tokens", "_anthropic_rejected_thinking"})

#: The ``model_config`` keys the worker's constructor upsert accepts.
WORKER_CONSTRUCTOR_CONFIG_KEYS = frozenset({"max_iterations", "max_tokens", "reasoning_config", "yolo_mode"})

#: The largest encoded payload one worker receipt accepts.
RECEIPT_PAYLOAD_MAX_BYTES = 4 * 1024 * 1024

#: The most rows one transcript batch carries.
TRANSCRIPT_BATCH_MAX = 1000


def row_annotations(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The annotation of each written message dict: its :data:`ROW_ANNOTATION_KEYS` values, ``None`` where absent."""
    return [{key: message.get(key) for key in ROW_ANNOTATION_KEYS} for message in messages]
