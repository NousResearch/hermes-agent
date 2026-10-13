"""Identity carry for re-inserted transcript rows: a compaction or rotation hands its carried tail back as
message dicts, and these keep each copy the same logical message as the durable original it repeats (same
display identity, same text), so display reads fold the generations and nothing the user wrote is edited."""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.message_metadata import MESSAGE_UID
from hermes_cli.timefmt import coerce_epoch
from hermes_state_common import _placeholders
from hermes_state_messages import _parse_tool_calls, _scrub_surrogates

logger = logging.getLogger("hermes_state")  # caplog tests pin the origin module's name

_UID_CHUNK = 500


class SessionCarryMixin:
    """Carry the durable original's identity onto re-inserted carried rows (mixed into ``SessionDB``)."""

    def _carry_parent_timestamps(self, conn, parent_session_id: str, messages: list[dict[str, Any]]) -> None:
        """Adopt the durable parent row's timestamp onto carried handoff rows so the re-inserted child row keeps
        the original's display identity (see _display_dedupe_key). Idempotent; never overwrites a timestamp the
        dict already carries; rows with no match keep their caller-supplied/now_ts value.

        Content identity = _display_dedupe_key's content key minus the timestamp (role, encoded content,
        tool_call_id, encoded tool_calls, tool_name). Donors are the parent's ACTIVE rows, consumed first-match-wins in id
        order so two identical-content turns cannot both grab the same parent row. Best-effort: on any error
        the insert falls back to today's behavior (publish-time stamp) and never breaks publish.
        """
        try:
            candidates = [i for i, msg in enumerate(messages)
                if isinstance(msg, dict) and coerce_epoch(msg.get("timestamp"), field="message timestamp") is None]
            if not candidates:
                return
            donors: dict[tuple[Any, ...], list[Any]] = {}
            for row in conn.execute(
                    "SELECT id, role, content, timestamp, tool_call_id, tool_calls, tool_name FROM messages "
                    "WHERE session_id = ? AND active = 1 ORDER BY id", (parent_session_id,)).fetchall():
                donors.setdefault((row["role"], row["content"], row["tool_call_id"], row["tool_calls"],
                    row["tool_name"]), []).append(row["timestamp"])
            for i in candidates:
                msg = messages[i]
                tool_calls = _parse_tool_calls(msg.get("tool_calls"))
                key = (msg.get("role", "unknown"), self._encode_content(msg.get("content")),
                    msg.get("tool_call_id"), json.dumps(tool_calls) if tool_calls else None,
                    _scrub_surrogates(msg.get("tool_name")))
                queue = donors.get(key)
                if queue:
                    msg["timestamp"] = queue.pop(0)
        except Exception:
            return  # Best-effort: a failed carry falls back to the publish-time stamp, never breaks publish.

    def _carry_durable_content(self, conn, messages: list[dict[str, Any]]) -> None:
        """Give a carried user/assistant dict its durable original's stored text back when the two differ
        only by the load lens.

        A resumed agent holds those strings as ``_rows_to_messages`` decoded them: sanitized and stripped
        (:meth:`_loaded_view_content`). Re-inserting that text as the carried copy silently edits the
        logical message (a trailing newline the user sent is gone), and because ``_display_dedupe_key``
        hashes content, the copy no longer folds into its archived original: every display read lists the
        message twice and content-keyed consumers see it as new. The original is named by the dict's
        ``message_uid``; any difference beyond that lens is a real edit and is kept as written.
        Best-effort: on any error the dicts are inserted as they are, and the failure is logged."""
        try:
            carried: dict[str, list[dict[str, Any]]] = {}
            for msg in messages:
                if not isinstance(msg, dict) or not isinstance(msg.get("content"), str):
                    continue
                uid = msg.get(MESSAGE_UID)
                if uid and msg.get("role") in ("user", "assistant"):
                    carried.setdefault(uid, []).append(msg)
            if not carried:
                return
            uids = list(carried)
            originals: dict[str, tuple[str, Any]] = {}
            for start in range(0, len(uids), _UID_CHUNK):
                chunk = uids[start:start + _UID_CHUNK]
                for row in conn.execute(
                        f"SELECT message_uid, role, content FROM messages WHERE message_uid IN ({_placeholders(chunk)}) "
                        "ORDER BY id", chunk).fetchall():
                    originals.setdefault(row["message_uid"], (row["role"], row["content"]))
            for uid, msgs in carried.items():
                if uid not in originals:
                    continue
                role, raw = originals[uid]
                stored = self._decode_content(raw)
                if not isinstance(stored, str):
                    continue
                for msg in msgs:
                    if (msg.get("role") == role and msg["content"] != stored
                            and self._loaded_view_content(role, stored) == msg["content"]):
                        msg["content"] = stored
        except Exception:
            # Best-effort boundary: the copy keeps the caller's text, exactly as before this carry existed.
            logger.warning("carry of durable message text failed; inserting the carried rows as given", exc_info=True)
