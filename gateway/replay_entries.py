"""Rebuild durable plain-text messages without losing replay or display metadata."""

from typing import Any

from agent.conversation_compression_archive import (
    MERGED_DURABLE_ROWS, RETIRED_DURABLE_ROWS, UNNAMED_DURABLE_ROWS,
)
from agent.message_metadata import (
    ABSORBED_MESSAGE_UIDS, MESSAGE_UID, copy_identity_fields, scaffolding_replay_fields,
)

# Keep provider echo/signature fields verbatim, including the empty thinking sentinel.
_ASSISTANT_REPLAY_FIELDS: tuple[str, ...] = (
    "reasoning", "reasoning_content", "reasoning_details", "codex_reasoning_items", "codex_message_items",
    "finish_reason")


def _build_replay_entry(
    role: str, content: Any, msg: dict[str, Any], preserve_timestamp: bool = False
) -> dict[str, Any]:
    """Build a replay entry for a non-tool-calling message, preserving ``_ASSISTANT_REPLAY_FIELDS``.

    ``preserve_timestamp``: only user rows need it (stale-dangerous-confirmation stripper). Falsy fields are
    dropped EXCEPT ``reasoning_content``: DeepSeek/Kimi treat "" as a sentinel; dropping it can 400.

    Empty values: most fields are dropped when falsy (matching the original PR #2974 behaviour) since an
    empty list/string for those carries no information. The exception is ``reasoning_content``:
    DeepSeek/Kimi thinking-mode replay treats an empty string as a meaningful sentinel that
    ``_copy_reasoning_content_for_api`` upgrades to a single space. Dropping it here would make the gateway
    send no ``reasoning_content`` at all on the next turn, which can cause HTTP 400 from strict thinking
    providers.
    """
    entry: dict[str, Any] = {"role": role, "content": content}
    # Persisted context-engine provenance belongs to replay, not provider input.
    # The shared API projection strips display metadata from each wire copy.
    if isinstance(msg.get("display_metadata"), dict):
        entry["display_metadata"] = msg["display_metadata"]
    if isinstance(msg.get("display_kind"), str) and msg["display_kind"]:
        entry["display_kind"] = msg["display_kind"]
    # api_content sidecar keeps the request prefix byte-stable — ONLY if this pipeline did not rewrite
    # content. The caller renders timestamps AFTER this check so a stamp alone never drops the sidecar.
    _sidecar = msg.get("api_content")
    if (
        role in ("user", "assistant")
        and isinstance(_sidecar, str)
        and _sidecar
        and content == msg.get("content")):
        entry["api_content"] = _sidecar
    if role == "assistant":
        for _rkey in _ASSISTANT_REPLAY_FIELDS:
            if _rkey not in msg:
                continue
            _rval = msg.get(_rkey)
            if (_rval is None) if _rkey == "reasoning_content" else (not _rval):
                continue
            entry[_rkey] = _rval
    if preserve_timestamp and msg.get("timestamp"):
        entry["timestamp"] = msg["timestamp"]
    entry.update(scaffolding_replay_fields(msg))  # a proactive prune re-persists this view: keep the flags
    # Replay rebuilds the SAME conversation: every role keeps its uid and merge witness, so a context engine sees the
    # uids the store holds. Tool-call uid maps stay on rows that still carry their calls (those pass through whole).
    copy_identity_fields({key: msg[key] for key in (MESSAGE_UID, ABSORBED_MESSAGE_UIDS) if key in msg}, entry)
    # Replay rewrites are view-only: keep the durable-row stamp so marker-only
    # flushes skip rows already in state.db (#121462/#123462).
    if msg.get("_db_persisted"):
        entry["_db_persisted"] = True
    # A merged user dict lost that stamp; this one is what ties it to its durable rows at compaction.
    # The other counts the rows the repair dropped behind a dict or folded into an assistant turn,
    # which this view cannot name either; the last names them for the commit.
    for stamp in (MERGED_DURABLE_ROWS, UNNAMED_DURABLE_ROWS, RETIRED_DURABLE_ROWS):
        if msg.get(stamp):
            entry[stamp] = msg[stamp]
    return entry
