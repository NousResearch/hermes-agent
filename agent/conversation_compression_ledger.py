"""Decision ledger handoff at the compaction boundary."""

import logging
import re

from hermes_state_ledger import (
    LEDGER_PRIORITY, MAX_LEDGER_ENTRIES,
    bounded_decision_text, bounded_ledger_turn_id, omitted_decision,
)

logger = logging.getLogger(__name__)
_DECISION_LEDGER_HEADER = "[DECISION LEDGER — VERBATIM]"
_DECISION_LEDGER_FOOTER = "[END DECISION LEDGER]"
# UTF-8 bytes bound characters AND byte-tokenizer tokens conservatively, without
# assuming English's usual chars/token ratio. Includes labels, markers and newlines.
MAX_LEDGER_HANDOFF_BYTES = 8192
_LEDGER_BLOCK = re.compile(re.escape(_DECISION_LEDGER_HEADER) + r".*?" + re.escape(_DECISION_LEDGER_FOOTER), re.DOTALL)


def ledger_message_kind(message: dict) -> str:
    metadata = message.get("display_metadata") or {}
    kind = metadata.get("decision_ledger") or (
        "synthetic" if message.get("_decision_ledger_synthetic") else
        "embedded" if message.get("_decision_ledger_embedded") else ""
    )
    content = message.get("content")
    if isinstance(content, str):
        if not kind and content.startswith(_DECISION_LEDGER_HEADER) and content.endswith(_DECISION_LEDGER_FOOTER):
            kind = "synthetic"  # Upgrade original PR carriers without metadata.
        if kind == "synthetic" and _LEDGER_BLOCK.sub("", content).strip():
            kind = "embedded"  # Ordinary sequence repair absorbed a later human turn.
    elif isinstance(content, list) and kind == "synthetic":
        kind = "embedded"
    return kind


def strip_ledger_from_user_anchor(message: dict) -> dict:
    if not ledger_message_kind(message):
        return message
    copy = dict(message)
    copy["display_metadata"] = dict(message.get("display_metadata") or {})
    _strip_old_handoff([copy])
    copy.pop("_decision_ledger_synthetic", None)
    return copy


def _strip_old_handoff(compressed: list) -> None:
    for index in range(len(compressed) - 1, -1, -1):
        message = compressed[index]
        if not isinstance(message, dict):
            continue
        kind = ledger_message_kind(message)
        if kind == "synthetic":
            compressed.pop(index)
        elif kind == "embedded":
            message.pop("_decision_ledger_embedded", None)
            message.pop("_decision_ledger_synthetic", None)
            message.get("display_metadata", {}).pop("decision_ledger", None)
            from agent.conversation_compression import _replace_message_content
            content = message.get("content")
            if isinstance(content, str):
                content = _LEDGER_BLOCK.sub("", content).strip()
            elif isinstance(content, list):
                content = [dict(part, text=_LEDGER_BLOCK.sub("", part["text"]).strip())
                           if isinstance(part, dict) and isinstance(part.get("text"), str) else part for part in content]
            _replace_message_content(message, content)
            from agent.message_metadata import MERGED_TURN_PREFIX
            prefix = message.get(MERGED_TURN_PREFIX)
            if isinstance(prefix, str):
                prefix = _LEDGER_BLOCK.sub("", prefix).strip()
                if prefix:
                    message[MERGED_TURN_PREFIX] = prefix
                else:
                    message.pop(MERGED_TURN_PREFIX, None)


def _render_ledger(entries: list) -> str:
    entries = entries[-MAX_LEDGER_ENTRIES:]
    # Reserve one explicit omission marker per event, then spend the remaining
    # budget on full text by safety priority and recency. Never emit partial text.
    lines = [f"- {entry['kind']}: {omitted_decision(entry['kind'], 'compaction budget')}" for entry in entries]
    used = len(("\n".join([_DECISION_LEDGER_HEADER, *lines, _DECISION_LEDGER_FOOTER])).encode("utf-8"))
    order = sorted(range(len(entries)), key=lambda i: (LEDGER_PRIORITY[entries[i]["kind"]], -i))
    for index in order:
        entry = entries[index]
        turn_id = bounded_ledger_turn_id(entry.get("turn_id") or "")
        label = f"{entry['kind']} ({turn_id})" if turn_id else entry["kind"]
        text = bounded_decision_text(entry["kind"], entry.get("text") or "")
        # Escape ledger delimiters/newlines so event text cannot forge handoff structure.
        candidate = f"- {label}: {text}".replace(_DECISION_LEDGER_HEADER, "[quoted ledger header]").replace(_DECISION_LEDGER_FOOTER, "[quoted ledger footer]")
        candidate = candidate.replace("\n", "\\n").replace("\r", "\\r")
        delta = len(candidate.encode("utf-8")) - len(lines[index].encode("utf-8"))
        if used + delta <= MAX_LEDGER_HANDOFF_BYTES:
            lines[index] = candidate
            used += delta
    return "\n".join([_DECISION_LEDGER_HEADER, *lines, _DECISION_LEDGER_FOOTER])


def fold_decision_ledger(agent, compressed: list) -> None:
    """Replace stale evidence without removing a human request merged into it."""
    _strip_old_handoff(compressed)
    reader = getattr(getattr(agent, "_session_db", None), "get_decision_ledger_entries", None)
    session_id = getattr(agent, "session_id", "") or ""
    if not session_id or not callable(reader):
        return
    try:
        entries = reader(session_id)
    except Exception:
        logger.debug("Could not load decision ledger for compaction", exc_info=True)
        return
    if not entries:
        return
    content = _render_ledger(entries)
    tail = compressed[-1] if compressed and isinstance(compressed[-1], dict) else None
    if tail is not None and tail.get("role") == "user":
        from agent.context_compressor import _append_text_to_content
        from agent.conversation_compression import _replace_message_content
        _replace_message_content(tail, _append_text_to_content(tail.get("content"), "\n\n" + content))
        tail.setdefault("display_metadata", {})["decision_ledger"] = "embedded"
        tail["_decision_ledger_embedded"] = True
    else:
        compressed.append({"role": "user", "content": content, "_decision_ledger_synthetic": True, "display_metadata": {"decision_ledger": "synthetic"}})
