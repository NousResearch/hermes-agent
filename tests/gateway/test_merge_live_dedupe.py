"""RED→GREEN: the merge-side clone guard (session mwlf5loyekujbd signature).

Production signature: after a rotated compression, the live set carries every row of each
compaction generation, and the display lineage fold (``_dedupe_display_generations``) shows a
carried tail once — at the first copy's position. The recovery merge (display-only) then painted
the carried row AND its clone, doubling every user/assistant reply in the restored transcript.

The guard: a live row whose ``(message_uid, timestamp)`` identity matches a row the display set
already shows (i.e. a row that is itself not live) is dropped from the merge. Live rows shown by
the display set keep painting — exactly as before the guard.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gateway.session_local_recovery import _merge_live_in_stored_order


def _row(row_id, role, content, ts, uid=None, tool_call_id=None, tool_name=None):
    """A replay dict shaped like ``as_openai_message`` + ``_row_id``: absent when None."""
    out = {'_row_id': row_id, 'role': role, 'content': content}
    if ts is not None:
        out['timestamp'] = ts
    if uid is not None:
        out['message_uid'] = uid
    if tool_call_id is not None:
        out['tool_call_id'] = tool_call_id
    if tool_name is not None:
        out['tool_name'] = tool_name
    return out


def test_clone_shown_by_archived_row_drops():
    """A live row sharing (uid, ts) with a row the display set shows but that is itself not live
    (an earlier-generation copy the fold kept) is a clone — dropped from the merge. The clone's
    position is preserved (the row the display set shows takes it)."""
    display = [
        _row(1, 'user', '升级下ekko studio', 100.0, uid='u-1'),       # shown, not live (fold kept it)
        _row(2, 'assistant', '好的，我来执行升级', 101.0, uid='a-1'),  # shown, not live
    ]
    live = [
        _row(3, 'user', '升级下ekko studio', 100.0, uid='u-1'),       # clone of row 1 (display shows row 1)
        _row(4, 'assistant', '好的，我来执行升级', 101.0, uid='a-1'),  # clone of row 2 (display shows row 2)
    ]
    merged = _merge_live_in_stored_order(display, live)
    # the clones are dropped; the rows the display set already shows take their position (archived)
    assert [r['_row_id'] for r in merged] == [1, 2], [r['_row_id'] for r in merged]
    assert merged[0].get('archived') is True
    assert merged[1].get('archived') is True


def test_clone_shown_by_live_row_keeps_painting():
    """A live row shown by the display set (its own row_id in the display set) keeps painting —
    the guard must not drop it (that was the regression in the first draft of the patch)."""
    display = [
        _row(1, 'user', '升级下ekko studio', 100.0, uid='u-1'),
        _row(2, 'assistant', '好的', 101.0, uid='a-1'),
    ]
    live = list(display)
    merged = _merge_live_in_stored_order(display, live)
    assert [r['_row_id'] for r in merged] == [1, 2], [r['_row_id'] for r in merged]


def test_distinct_repeats_stay():
    """The user genuinely sent '继续' twice (different uids, different timestamps): both paint."""
    display = [
        _row(1, 'user', '继续', 100.0, uid='u-1'),
        _row(2, 'assistant', 'ok', 101.0, uid='a-1'),
        _row(3, 'user', '继续', 200.0, uid='u-2'),
        _row(4, 'assistant', 'ok2', 201.0, uid='a-2'),
    ]
    live = list(display)
    merged = _merge_live_in_stored_order(display, live)
    assert [r['_row_id'] for r in merged] == [1, 2, 3, 4], [r['_row_id'] for r in merged]


def test_archived_rows_anchor_between_live():
    """Archived display rows still land before the live row they preceded (pre-guard behavior)."""
    display = [
        _row(1, 'user', 'old ask', 100.0, uid='u-0'),   # not live → archived anchor
        _row(2, 'user', 'live ask', 200.0, uid='u-1'),  # live
    ]
    live = [_row(2, 'user', 'live ask', 200.0, uid='u-1')]
    merged = _merge_live_in_stored_order(display, live)
    assert [r['_row_id'] for r in merged] == [1, 2], [r['_row_id'] for r in merged]
    assert merged[0].get('archived') is True
    assert 'archived' not in merged[1]
