"""Contract suite for the alternation repair's markers on held messages, and for the prune rewrite's diff.

A reload without row ids hands a compactor messages the repair merged, dropped or folded; the markers on a ``HeldMessage`` name the durable rows each stands for, so ``compact`` archives exactly those and clones every other active row after the replacement. Without a holding proof, the same markers on a carried ``NewMessage`` widen the watermark path's tail rewind by the rows it stands for. ``rewrite_pruned_rows`` writes only the columns the replacement changed against the row as written. Every case runs on every registered backend and reads the outcome back through the public transcript views.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from persistence import RowVisibility
from persistence.core.identity import SessionId
from persistence.conversation.messages import HeldCoverage, HeldMessage, NewMessage, PrunedRowChange, TranscriptRead, TranscriptView
from persistence.conversation.sessions import NewSession

SESSION = SessionId("held-markers-session")
CALL = {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}


@pytest.fixture
def repo(backend, scope_a):
    store = backend.open(scope_a)
    store.sessions.create(NewSession(session_id=SESSION, source="cli"))
    return store.messages


def _msg(content: str, i: int, role: str = "user", **kwargs: object) -> NewMessage:
    return NewMessage(role=role, content=content, timestamp=datetime(2026, 10, 29, 9, 0, i, tzinfo=UTC).timestamp(), **kwargs)  # type: ignore[arg-type]


def _seed(repo, *rows: NewMessage) -> list:
    for row in rows:
        repo.append(SESSION, row)
    return list(repo.read_transcript(SESSION))


def _states(repo) -> dict:
    return {r.id: r.visibility for r in repo.read_transcript(SESSION, TranscriptRead(view=TranscriptView.ALL))}


def _live(repo) -> list:
    return [(r.role, r.content) for r in repo.read_transcript(SESSION)]


def test_marker_fields_validate():
    with pytest.raises(ValueError):
        HeldMessage(role="user", content="x", merged_durable_rows=-1)
    with pytest.raises(ValueError):
        HeldMessage(role="user", content="x", retired_durable_rows=({"role": "tool"},))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        HeldMessage(role="user", content="x", tool_call_ids=["c1"])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        HeldMessage(role="user", content="x", retired_row=1)  # type: ignore[arg-type]


def test_a_merged_user_run_is_named_and_an_unheld_row_survives(repo):
    alpha, beta, other, reply = _seed(repo, _msg("alpha", 0), _msg("beta", 1), _msg("other surface", 2, "assistant"), _msg("reply", 3, "assistant"))
    watermark = repo.active_watermark(SESSION)
    # The repair joined the user;user pair into one held message equal to neither row.
    merged = HeldMessage(role="user", content="alpha\n\nbeta", persisted=True, merged_durable_rows=2)
    repo.compact(SESSION, [_msg("SUMMARY", 9, "assistant")], watermark=watermark,
                 coverage=HeldCoverage(covered_ids=(reply.id,), unresolved=(merged,)))
    assert _live(repo) == [("assistant", "SUMMARY"), ("assistant", "other surface")]
    states = _states(repo)
    assert [states[r.id] for r in (alpha, beta, reply)] == [RowVisibility.COMPACTION_ARCHIVED] * 3
    assert states[other.id] is RowVisibility.REWOUND


def test_a_merged_run_that_names_no_rows_abandons_the_proof(repo):
    _seed(repo, _msg("alpha", 0), _msg("beta", 1), _msg("other surface", 2, "assistant"))
    watermark = repo.active_watermark(SESSION)
    merged = HeldMessage(role="user", content="alpha\n\ngamma", merged_durable_rows=2)
    repo.compact(SESSION, [_msg("SUMMARY", 9, "assistant")], watermark=watermark, coverage=HeldCoverage(unresolved=(merged,)))
    assert _live(repo) == [("assistant", "SUMMARY")]


def test_a_retired_row_is_named_by_its_tool_call_id(repo):
    ask, orphan, _other, kept = _seed(repo, _msg("ask", 0), _msg("same", 1, "tool", tool_call_id="x"), _msg("other surface", 2, "assistant"),
                                     _msg("same", 3, "tool", tool_call_id="y"))
    watermark = repo.active_watermark(SESSION)
    record = HeldMessage(role="tool", content="same", tool_call_id="x")
    # The repair dropped the orphan result behind the prompt; its record names it where its text names two rows.
    survivor = HeldMessage(role="user", content="ask", persisted=True, retired_durable_rows=(record,))
    retired = replace(record, persisted=True, retired_row=True)
    repo.compact(SESSION, [_msg("SUMMARY", 9, "assistant")], watermark=watermark,
                 coverage=HeldCoverage(covered_ids=(kept.id,), unresolved=(survivor, retired)))
    assert _live(repo) == [("assistant", "SUMMARY"), ("assistant", "other surface")]
    states = _states(repo)
    assert [states[r.id] for r in (ask, orphan, kept)] == [RowVisibility.COMPACTION_ARCHIVED] * 3


def test_a_retired_row_matching_no_single_row_abandons_the_proof(repo):
    _seed(repo, _msg("ask", 0), _msg("same", 1, "tool", tool_call_id="x"), _msg("other surface", 2, "assistant"))
    watermark = repo.active_watermark(SESSION)
    retired = HeldMessage(role="tool", content="same", tool_call_id="z", persisted=True, retired_row=True)
    repo.compact(SESSION, [_msg("SUMMARY", 9, "assistant")], watermark=watermark, coverage=HeldCoverage(unresolved=(retired,)))
    assert _live(repo) == [("assistant", "SUMMARY")]


def test_a_folded_rows_own_record_names_it_in_the_carried_tail(repo):
    q, one, two, nxt = _seed(repo, _msg("q", 0), _msg("part one", 1, "assistant"), _msg("part two", 2, "assistant"), _msg("next", 3))
    watermark = repo.active_watermark(SESSION)
    own, folded = HeldMessage(role="assistant", content="part one", own_row=True), HeldMessage(role="assistant", content="part two")
    # A fold rewrote the survivor's text: its own record names its row, and the folded row stands behind it.
    survivor = HeldMessage(role="assistant", content="part one\n\npart two", persisted=True, retired_durable_rows=(own, folded))
    unresolved = (survivor, replace(own, persisted=True, retired_row=True), replace(folded, persisted=True, retired_row=True))
    replacement = [_msg("SUMMARY", 9, "assistant"), _msg("part one\n\npart two", 1, "assistant"), _msg("next", 3)]
    repo.compact(SESSION, replacement, watermark=watermark, carried_tail_count=2,
                 coverage=HeldCoverage(covered_ids=(q.id, nxt.id), unresolved=unresolved))
    assert _live(repo) == [("assistant", "SUMMARY"), ("assistant", "part one\n\npart two"), ("user", "next")]
    states = _states(repo)
    # The tail's originals are the survivor's own row, the row folded into it and the carried next row.
    assert states[q.id] is RowVisibility.COMPACTION_ARCHIVED
    assert [states[r.id] for r in (one, two, nxt)] == [RowVisibility.REWOUND] * 3


def test_new_message_marker_fields_validate_and_read_off_a_live_dict():
    with pytest.raises(ValueError):
        NewMessage(role="user", content="x", merged_durable_rows=-1)
    with pytest.raises(ValueError):
        NewMessage(role="user", content="x", unnamed_durable_rows=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        NewMessage(role="user", content="x", absorbed_row_ids=[1])  # type: ignore[arg-type]
    live = {"role": "user", "content": "a", "_merged_durable_rows": 2, "_unnamed_durable_rows": 1, "_absorbed_row_ids": [7, True, -1]}
    message = NewMessage.from_live(live)
    assert (message.merged_durable_rows, message.unnamed_durable_rows, message.absorbed_row_ids) == (2, 1, (7,))
    assert NewMessage.from_live({"role": "user", "content": "a", "_merged_durable_rows": True}).merged_durable_rows == 0


@pytest.mark.parametrize(("markers", "rewound_first"), [
    (lambda first: {"merged_durable_rows": 2}, True),
    (lambda first: {"unnamed_durable_rows": 1}, True),
    (lambda first: {"merged_durable_rows": 2, "absorbed_row_ids": (first,)}, False),
    (lambda first: {}, False),
], ids=["merged", "unnamed", "merged_listed", "none"])
def test_the_watermark_path_rewinds_every_row_a_carried_message_stands_for(repo, markers, rewound_first):
    a, b, r = _seed(repo, _msg("a", 0), _msg("b", 1), _msg("r", 2, "assistant"))
    watermark = repo.active_watermark(SESSION)
    # No coverage proof: the tail's originals are the newest rows at or below the watermark, one per carried
    # message plus the rows a merged or repaired message stands for that it does not list.
    replacement = [_msg("SUMMARY", 9, "assistant"), _msg("a\n\nb", 0, **markers(a.id)), _msg("r", 2, "assistant")]
    repo.compact(SESSION, replacement, watermark=watermark, carried_tail_count=2)
    assert _live(repo) == [("assistant", "SUMMARY"), ("user", "a\n\nb"), ("assistant", "r")]
    states = _states(repo)
    first = RowVisibility.REWOUND if rewound_first else RowVisibility.COMPACTION_ARCHIVED
    assert [states[row.id] for row in (a, b, r)] == [first, RowVisibility.REWOUND, RowVisibility.REWOUND]


def _tool_turn(repo):
    _seed(repo, _msg("go", 0), _msg("", 1, "assistant", tool_calls=(CALL,)))
    written = _msg("X" * 50, 2, "tool", tool_call_id="c1", tool_name="t")
    result = repo.append(SESSION, written)
    return HeldMessage(role="tool", content="X" * 50, tool_call_id="c1", row_id=result, persisted=True), written


def _tool_rows(repo) -> list:
    return [(r.visibility, r.tool_name, r.content) for r in repo.read_transcript(SESSION, TranscriptRead(view=TranscriptView.ALL)) if r.role == "tool"]


def test_a_content_only_prune_rewrites_the_content(repo):
    original, written = _tool_turn(repo)
    done = repo.rewrite_pruned_rows(SESSION, [PrunedRowChange(original, replace(written, content="[pruned]"), written)])
    assert done.rewritten == 1
    assert sorted(_tool_rows(repo), key=str) == sorted([(RowVisibility.ACTIVE, "t", "[pruned]"), (RowVisibility.COMPACTION_ARCHIVED, "t", "X" * 50)], key=str)


def test_a_column_the_replacement_drops_is_written_empty(repo):
    original, written = _tool_turn(repo)
    done = repo.rewrite_pruned_rows(SESSION, [PrunedRowChange(original, replace(written, content="[pruned]", tool_name=None), written)])
    assert done.rewritten == 1
    assert [row for row in _tool_rows(repo) if row[0] is RowVisibility.ACTIVE] == [(RowVisibility.ACTIVE, None, "[pruned]")]


def test_a_change_with_no_column_changed_rewrites_nothing(repo):
    original, written = _tool_turn(repo)
    done = repo.rewrite_pruned_rows(SESSION, [PrunedRowChange(original, written, written)])
    assert (done.rewritten, done.snapshots) == (0, (None,))
    assert _tool_rows(repo) == [(RowVisibility.ACTIVE, "t", "X" * 50)]
