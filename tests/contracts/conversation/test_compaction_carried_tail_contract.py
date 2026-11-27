"""Contract cases for a compaction whose replacement carries a row above the watermark: a replacement row that names the tail row (``repair_target``, else ``message_uid``) and replays as it stands for that row, which is then not cloned; a still-unclaimed replacement naming the row by uid claims it weakly only when the compressor's own rewrite explains the difference, and a weak claim is archived, not rewound. Anything unproven is cloned after the replacement, so a message may be stored twice but the model never loses an input or keeps a stale one."""

from __future__ import annotations

import pytest

from persistence import RowVisibility
from persistence.conversation.messages import ConversationRequest, NewMessage, TranscriptRead, TranscriptView
from persistence.conversation.sessions import NewSession

SID = "carried-tail-contract"
SUMMARY_TEXT = "[CONTEXT COMPACTION] summary of the earlier turns"
SUMMARY = NewMessage("user", SUMMARY_TEXT)
ALL = TranscriptRead(view=TranscriptView.ALL)


@pytest.fixture
def store(backend, scope_a):
    store = backend.open(scope_a)
    store.sessions.create(NewSession(SID, "cli"))
    for i in range(4):
        store.messages.append(SID, NewMessage("user" if i % 2 == 0 else "assistant", f"old turn {i}"))
    return store


def _copies(store, content: str) -> list:
    return [row for row in store.messages.read_transcript(SID, ALL) if row.content == content]


def _live(store) -> list:
    return [m.content for m in store.messages.read_conversation(SID)]


def _held(store, count: int) -> list[NewMessage]:
    """The compressor's row-addressed copies of the last ``count`` live rows."""
    loaded = store.messages.read_conversation(SID, ConversationRequest(include_row_ids=True))[-count:]
    return [NewMessage.from_live(m.as_openai_message()) for m in loaded]


def _display(store) -> str:
    return "\n".join(str(r.content) for r in store.messages.read_transcript(SID, TranscriptRead(view=TranscriptView.DISPLAY)))


def test_a_held_row_above_the_watermark_is_carried_once(store):
    watermark = store.messages.active_watermark(SID)
    store.messages.append(SID, NewMessage("user", "steer during the summary call"))
    store.messages.append(SID, NewMessage("assistant", "reply during the summary call"))
    held = _held(store, 2)
    assert all(m.repair_target and m.message_uid for m in held)

    result = store.messages.compact(SID, [SUMMARY, *held], watermark=watermark, carried_tail_count=2)

    assert result.active_count == 3
    assert _live(store) == [SUMMARY_TEXT, "steer during the summary call", "reply during the summary call"]
    for content in ("steer during the summary call", "reply during the summary call"):
        copies = _copies(store, content)
        assert sorted(r.visibility.value for r in copies) == ["active", "rewound"], f"{content!r}: original and carried copy, no clone"
        assert len({r.message_uid for r in copies}) == 1
    # The carried copies named rows above the watermark, so the tail rewind does not reach below it.
    for i in range(4):
        assert [r.visibility for r in _copies(store, f"old turn {i}")] == [RowVisibility.COMPACTION_ARCHIVED]


def test_a_foreign_identical_row_above_the_watermark_is_still_cloned(store):
    store.messages.append(SID, NewMessage("assistant", "same text twice"))
    (held,) = _held(store, 1)
    watermark = store.messages.active_watermark(SID)
    store.messages.append(SID, NewMessage("assistant", "same text twice"))

    store.messages.compact(SID, [SUMMARY, held], watermark=watermark, carried_tail_count=1)

    copies = _copies(store, "same text twice")
    assert len(copies) == 4, "held original and its carried copy, foreign original and its clone"
    assert sorted(r.visibility.value for r in copies) == ["active", "active", "rewound", "rewound"]
    assert len({r.message_uid for r in copies if r.visibility is RowVisibility.ACTIVE}) == 2


def test_matching_text_never_claims_an_independent_append(store):
    store.messages.append(SID, NewMessage("assistant", "continue"))
    watermark = store.messages.active_watermark(SID)
    store.messages.append(SID, NewMessage("assistant", "continue"))

    store.messages.compact(SID, [SUMMARY, NewMessage("assistant", "continue")], watermark=watermark, carried_tail_count=1)

    assert _live(store) == [SUMMARY_TEXT, "continue", "continue"]


def test_a_claim_keeps_the_rows_current_replay_sidecar(store):
    watermark = store.messages.active_watermark(SID)
    stamp = store.messages.append_batch(SID, [NewMessage("user", "hello", api_content="first exact provider input")]).outcomes[0]
    held = NewMessage("user", "hello", api_content="first exact provider input", repair_target=stamp.id,
                      db_row_snapshot=stamp.db_row_snapshot, message_uid=stamp.message_uid)
    store.messages.set_message_api_content(SID, stamp.id, "hello", "new exact provider input")

    store.messages.compact(SID, [SUMMARY, held], watermark=watermark, carried_tail_count=1)

    assert "new exact provider input" in [m.api_content for m in store.messages.read_conversation(SID)]


def test_a_rewritten_carry_of_the_current_version_claims_weakly_and_is_archived(store):
    """The compressor kept the row's current text as whole lines inside a longer text: the copy stands for the row, which is archived as compacted history, not rewound and not cloned."""
    watermark = store.messages.active_watermark(SID)
    stamp = store.messages.append_batch(SID, [NewMessage("user", "keep this line")]).outcomes[0]
    merged = NewMessage("user", f"{SUMMARY_TEXT}\n\nkeep this line", repair_target=stamp.id,
                        db_row_snapshot=stamp.db_row_snapshot, message_uid=stamp.message_uid)

    store.messages.compact(SID, [merged], watermark=watermark, carried_tail_count=1)

    assert _live(store) == [f"{SUMMARY_TEXT}\n\nkeep this line"]
    assert [r.visibility for r in _copies(store, "keep this line")] == [RowVisibility.COMPACTION_ARCHIVED]


def _row_rewritten_after_hold(store):
    """The compressor took a uid-only copy, then the row was rewritten in place (an @file expansion)."""
    store.messages.append(SID, NewMessage("user", "raw @file"))
    (row,) = store.messages.read_conversation(SID, ConversationRequest(include_row_ids=True))[-1:]
    store.messages.set_user_message_content(SID, row.row_id, "expanded file body")
    return [SUMMARY, NewMessage("user", "raw @file", message_uid=row.message_uid)], [SUMMARY_TEXT, "raw @file", "expanded file body"]


def _edit_reinserted_with_its_uid(store):
    """Two tail rows share a uid (an edit re-inserted with it); the held copy is the edit."""
    store.messages.append(SID, NewMessage("assistant", "v1 draft"))
    uid = store.messages.read_conversation(SID)[-1].message_uid
    store.messages.append_batch(SID, [NewMessage("assistant", "v2 edited", message_uid=uid)])
    return [SUMMARY, NewMessage("assistant", "v2 edited", message_uid=uid)], [SUMMARY_TEXT, "v2 edited", "v1 draft"]


def _row_shortened_after_row_addressed_load(store):
    """A row-addressed load without a version digest, then an in-place edit shortened the row to a whole line of the older copy: the copy must not stand for the row."""
    store.messages.append(SID, NewMessage("user", "fix bug\nthen delete database"))
    loaded = store.messages.read_conversation(SID, ConversationRequest(include_row_ids=True))[-1]
    store.messages.set_user_message_content(SID, loaded.row_id, "fix bug")
    held = NewMessage("user", "fix bug\nthen delete database", repair_target=loaded.row_id, message_uid=loaded.message_uid)
    return [SUMMARY, held], [SUMMARY_TEXT, "fix bug\nthen delete database", "fix bug"]


def _row_shortened_after_flushed_hold(label: str, before: str, after: str, *, digest: bool = True):
    """The compressor copied the flushed row (row id, uid and its version digest, or the uid alone), then an in-place edit shortened the row: its new text sits inside the older copy, which must not stand for it."""
    def build(store):
        stamp = store.messages.append_batch(SID, [NewMessage("user", before)]).outcomes[0]
        held = (NewMessage("user", before, repair_target=stamp.id, db_row_snapshot=stamp.db_row_snapshot,
                           message_uid=stamp.message_uid) if digest else NewMessage("user", before, message_uid=stamp.message_uid))
        store.messages.set_user_message_content(SID, stamp.id, after)
        return [SUMMARY, held], [SUMMARY_TEXT, before, after]
    build.__name__ = f"shortened_{label}"
    return build


@pytest.mark.parametrize("build", [
    _row_rewritten_after_hold, _row_shortened_after_row_addressed_load, _edit_reinserted_with_its_uid,
    _row_shortened_after_flushed_hold("suffix", "fix bug in parser", "fix bug"),
    _row_shortened_after_flushed_hold("instruction", "yes, but first delete the prod database", "yes"),
    _row_shortened_after_flushed_hold("line_digest", "fix bug\nthen drop the prod table", "fix bug"),
    _row_shortened_after_flushed_hold("line_uid_only", "fix bug\nthen drop the prod table", "fix bug", digest=False),
], ids=lambda build: build.__name__)
def test_a_rewritten_carry_claims_only_what_the_compressor_rewrote(store, build):
    watermark = store.messages.active_watermark(SID)
    compacted, expected_live = build(store)
    floor = int(watermark.last_active or 0)
    versions = [r.content for r in store.messages.read_transcript(SID, ALL) if r.visibility is RowVisibility.ACTIVE and r.id > floor]

    store.messages.compact(SID, compacted, watermark=watermark, carried_tail_count=1)

    assert _live(store) == expected_live
    display = _display(store)
    assert versions and all(text in display for text in versions)
