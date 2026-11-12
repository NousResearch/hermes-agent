"""``session.transcript`` row kinds: every row carries ``row_kind``.

- ``'message'``: a display row, exactly as ``session.resume``'s ``display_messages`` projects it.
- ``'compaction'``: a compaction handoff at its position in the transcript: ``role: 'system'``, ``text``
  the summary a reader sees (``hermes_agent.agent.compaction_display.compaction_summary``: without the preamble, end
  marker, re-quoted user messages and recovery footer written for the model), ``compacted_messages`` (how
  many messages it summarized, when the handoff names it), ``timestamp`` and the stored ``row_id``.

The display projection drops a handoff (``project_compaction_message_for_display``); here each one is put
back as its own row just before the row it was stored on. A later compaction can carry, re-wrap or unwrap
a summary carrier, and the display dedupe then shows one copy of that message; the older handoffs on its
earlier copies (``transcript_handoff_rows``) are still shown, each once.
"""
from __future__ import annotations

import json

ROW_KIND_MESSAGE = 'message'
ROW_KIND_COMPACTION = 'compaction'
# The placeholder a handoff holds in the projection: a system row the projection passes through unchanged.
_MARK_KEY = '__session_transcript_handoff__'
_MARK_TEXT = 'compaction handoff'


def handoff_row(message, summary):
    row = {'row_kind': ROW_KIND_COMPACTION, 'role': 'system', 'text': summary['summary'],
           'compacted_messages': summary['compacted_messages'], 'row_id': message['id']}
    timestamp = message.get('timestamp')
    if isinstance(timestamp, (int, float)) and timestamp > 0:
        row['timestamp'] = float(timestamp)
    return row


def _mark_index(row):
    metadata = row.get('display_metadata')
    index = metadata.get(_MARK_KEY) if isinstance(metadata, dict) else None
    return index if row.get('role') == 'system' and row.get('text') == _MARK_TEXT and type(index) is int else None


def handoff_occurrence(message):
    # A handoff's copies (a later generation or a rotated segment carrying it) keep its uid and content.
    return message.get('message_uid') or None, json.dumps(message.get('content'), default=str)


def shown_occurrences(messages):
    """The handoff occurrences among stored *messages* (``transcript_handoff_rows`` candidates)."""
    from hermes_agent.agent.compaction_display import compaction_summary
    return {handoff_occurrence(message) for message in messages if compaction_summary(message) is not None}


def page_handoffs(db, rows, segments, earlier=None):
    """Per row of the stored *rows* (``_rows_to_conversation`` dicts of *segments*, display order): the
    ``compaction`` rows shown just before it, one for each handoff stored on the row or on an earlier copy
    of its message. Each handoff is shown once in the whole transcript, before the first row that holds
    it, wherever the page starts: ``earlier(candidates)`` (stored handoff messages) is the occurrences
    among them a row before *rows* holds."""
    from hermes_agent.agent.compaction_display import compaction_summary
    found = db.transcript_handoff_rows([row['_row_id'] for row in rows if type(row.get('_row_id')) is int], segments)
    candidates = [message for messages in found.values() for message in messages]
    seen, out = set(earlier(candidates)) if earlier is not None and candidates else set(), []
    for row in rows:
        mine = []
        for message in found.get(row.get('_row_id'), ()):
            occurrence = handoff_occurrence(message)
            summary = compaction_summary(message) if occurrence not in seen else None
            if summary is None:
                continue
            seen.add(occurrence)
            mine.append(handoff_row(message, summary))
        out.append(mine)
    return out


def project_entries(entries, project):
    """``project(rows)`` of *entries* (display order): ``(ROW_KIND_COMPACTION, handoff row)`` is put back in
    place as is, ``(ROW_KIND_MESSAGE, stored row)`` is projected and tagged ``row_kind: 'message'``."""
    handoffs, marked = [], []
    for kind, value in entries:
        if kind == ROW_KIND_COMPACTION:
            marked.append({'role': 'system', 'content': _MARK_TEXT, 'display_metadata': {_MARK_KEY: len(handoffs)}})
            handoffs.append(value)
        else:
            marked.append(value)
    rows = []
    for row in project(marked):
        index = _mark_index(row)
        rows.append(handoffs[index] if index is not None and index < len(handoffs)
                    else {**row, 'row_kind': ROW_KIND_MESSAGE})
    return rows
