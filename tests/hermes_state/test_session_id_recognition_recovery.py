"""Session recovery: salvage must recognise every session-id shape this repo really mints.

Follow-up to the physical-layout mapping (7704712168, fixes #101409), which taught
``hermes sessions recover`` to map cells by NAME. It still classified a row as a
session only when the id matched ``SESSION_ID_PATTERN``
(``^\\d{8}_\\d{6}_``) -- but several first-class surfaces DERIVE an id instead of
minting through ``new_session_id``:

* ``cron/scheduler.py``     f"cron_{job_id}_{now:%Y%m%d_%H%M%S}"
* ``cli_commands_mixin.py`` f"bg_{now:%H%M%S}_{uuid4().hex[:6]}"   (``/bg``)
* ``api_server_room_dispatch.py`` f"room_{sha256(seed)[:32]}"

Measured on a real 3,373-session store: 379 rows (11.2%) -- 375 cron, 1 bg, plus
junk -- failed the pattern. Two distinct consequences, both covered here:

1. DROPPED ROWS. ``classify_lost_and_found_row`` returned None for those rows, so
   ``hermes sessions recover`` silently discarded every cron/bg/room session.
2. POISONED LAYOUT INFERENCE. ``parent_session_id`` is a text-shape rule checked
   with the same predicate, so an ordinary timestamp-id CHILD of a cron job (which
   classifies fine) carried an unrecognised parent id -- and because one bad
   sampled value vetoes a candidate layout, a single such row collapsed inference
   for the WHOLE table to empty, i.e. the positional-guessing fallback #101409
   exists to avoid.

The widened recognizers are layout SENTINELS, so they stay anchored: the
adversarial test below pins that near-misses and arbitrary text are still
rejected and that a wrong layout is still vetoed.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from typing import Optional

import pytest

from hermes_state import SessionDB
from hermes_state_ids import (
    SESSION_ID_PATTERN,
    SESSION_ID_RECOGNIZERS,
    is_known_session_id,
    new_session_id,
)
from hermes_cli.session_lost_and_found import (
    LayoutEvidence,
    _cell_fits,
    _declared_types,
    _is_session_id,
    _row_invariants_hold,
    classify_lost_and_found_row,
    infer_physical_layouts,
)

# Exactly as the minting sites build them (see module docstring for file/line).
CRON_ID = "cron_0b36145ae5c7_20260626_100059"
CRON_SLUG_ID = "cron_job-1_20260707_195958"      # legacy/user job id, real in live stores
BG_ID = "bg_124526_be5bed"
ROOM_ID = "room_" + "0123456789abcdef" * 2

DERIVED_IDS = [CRON_ID, CRON_SLUG_ID, BG_ID, ROOM_ID]

# Near-misses and arbitrary cell values. These sit at the id position of a candidate
# layout, so accepting any of them costs real wrong-column-mapping safety.
NOT_SESSION_IDS = [
    "", "s", "session-key", "pre-compress-key",
    "cron", "cron_", "bg_", "room_",
    "cron_job1", "cron_abc", "cron_abc_2026", "cron_abc_20260101",
    "cron_abc_20260101_1005", "cron__20260101_100500",
    "cron_abc_20260101_100500_extra",
    "bg_12345_abcdef", "bg_1234567_abcdef", "bg_123456_abcdeg", "bg_123456_ABCDEF",
    "room_" + "a" * 31, "room_" + "a" * 33, "room_" + "g" * 32,
    "/Users/adam/some/path", "https://example.com", "hello world",
    '{"json": true}', "claude-opus-5", "assistant",
    "20260101_100500",
]


def _timestamp_id() -> str:
    return new_session_id(datetime(2026, 1, 2, 3, 4, 5))


def _physical_columns(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")'))


def _sessions_rows(conn: sqlite3.Connection) -> tuple[tuple[str, ...], list[tuple]]:
    columns = _physical_columns(conn, "sessions")
    quoted = ", ".join(f'"{c}"' for c in columns)
    return columns, [tuple(r) for r in conn.execute(f"SELECT {quoted} FROM sessions")]


def _infer_like_recovery(conn: sqlite3.Connection, columns: tuple[str, ...],
                         rows: list[tuple]) -> tuple[Optional[list], int]:
    """Run inference the way the salvage lane does: only rows that CLASSIFY as
    ``sessions`` become layout evidence (session_lost_and_found.py, pass 1)."""
    evidence = LayoutEvidence("sessions")
    kept = 0
    for cells in rows:
        if classify_lost_and_found_row(len(cells), cells) == "sessions":
            evidence.add(cells)
            kept += 1
    mapping = infer_physical_layouts(
        evidence, _declared_types(conn, "sessions")
    ).get(len(columns))
    return mapping, kept


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A real post-upgrade store; the factory seeds sessions through the real SessionDB."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")

    def add(session_id: str, **kwargs) -> str:
        db.create_session(session_id, source=kwargs.pop("source", "cli"),
                          model="claude-opus-5", cwd=str(tmp_path), **kwargs)
        return session_id

    def bulk(count: int) -> None:
        for _ in range(count):
            add(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}")

    return db, add, bulk, lambda: sqlite3.connect(str(tmp_path / "state.db"))


# ── 1. recognition contract ────────────────────────────────────────────


@pytest.mark.parametrize("session_id", DERIVED_IDS)
def test_derived_session_ids_are_recognized(session_id: str) -> None:
    """A cron/bg/room id is a real session id even though it is not minted by
    ``new_session_id``; salvage must not treat it as an arbitrary cell."""
    assert is_known_session_id(session_id), (
        f"{session_id!r} is minted by a real surface but salvage does not recognise it: "
        f"hermes sessions recover will DROP those sessions and, via parent_session_id, "
        f"veto every candidate layout for the whole sessions table"
    )
    assert _is_session_id(session_id)


@pytest.mark.parametrize("value", NOT_SESSION_IDS)
def test_near_misses_are_still_rejected(value: str) -> None:
    """The recognizers are layout sentinels. Widening them must not make arbitrary
    text (or a near-miss id) admissible at the id position."""
    assert not is_known_session_id(value), (
        f"{value!r} is accepted as a session id -- the id sentinel no longer "
        f"discriminates between candidate layouts"
    )


def test_minted_timestamp_ids_still_recognized() -> None:
    """The original shape keeps working, and the narrow pattern is still exported
    for callers that specifically want the timestamp form."""
    sid = _timestamp_id()
    assert SESSION_ID_PATTERN.match(sid) and is_known_session_id(sid)
    assert SESSION_ID_PATTERN in SESSION_ID_RECOGNIZERS


# ── 2. dropped rows (consequence 1) ────────────────────────────────────


@pytest.mark.parametrize("session_id", DERIVED_IDS)
def test_derived_id_sessions_are_not_dropped_by_salvage(store, session_id: str) -> None:
    """Before the fix ``classify_lost_and_found_row`` returned None for these rows,
    so recovery discarded the session entirely."""
    _db, add, _bulk, connect = store
    add(session_id, source="cron" if session_id.startswith("cron_") else "cli")
    conn = connect()
    try:
        columns, rows = _sessions_rows(conn)
        row = next(r for r in rows if r[0] == session_id)
        assert classify_lost_and_found_row(len(columns), row) == "sessions"
    finally:
        conn.close()


def test_real_world_id_mix_is_fully_classified(store) -> None:
    """The measured real-store mix (~11% derived ids) must survive salvage intact."""
    _db, add, bulk, connect = store
    bulk(20)
    for sid in DERIVED_IDS:
        add(sid, source="cron" if sid.startswith("cron_") else "cli")
    conn = connect()
    try:
        columns, rows = _sessions_rows(conn)
        dropped = [r[0] for r in rows
                   if classify_lost_and_found_row(len(columns), r) != "sessions"]
        assert not dropped, f"salvage would silently discard these sessions: {dropped}"
    finally:
        conn.close()


# ── 3. poisoned layout inference (consequence 2) ───────────────────────


def test_child_of_a_cron_job_does_not_poison_layout_inference(store) -> None:
    """The regression that made the 21c5547faa fix inert on any install that ran cron.

    The cron session's own row never reaches inference (it failed classification),
    but its CHILD has an ordinary timestamp id and classifies fine -- carrying the
    cron id in ``parent_session_id``. One unrecognised sampled value vetoes every
    candidate, so inference returned {} and recovery fell back to positional
    guessing for the entire table.
    """
    _db, add, bulk, connect = store
    bulk(20)
    add(CRON_ID, source="cron")
    add(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_child1", parent_session_id=CRON_ID)

    conn = connect()
    try:
        columns, rows = _sessions_rows(conn)
        mapping, kept = _infer_like_recovery(conn, columns, rows)
        assert mapping is not None, (
            "layout inference collapsed to empty because one row's parent_session_id "
            "is a cron id -- recovery now maps this store's cells POSITIONALLY, the "
            "exact failure #101409/21c5547faa exists to prevent"
        )
        assert kept == len(rows)
        misnamed = [(i, columns[i], mapping[i]) for i in range(len(columns))
                    if mapping[i] is not None and mapping[i] != columns[i]]
        assert not misnamed, f"cells mapped to the wrong columns: {misnamed}"
        # The fix must actually resolve the layout, not merely avoid crashing.
        assert sum(1 for m in mapping if m is not None) == len(columns)
    finally:
        conn.close()


def test_layout_inference_survives_every_derived_id_shape(store) -> None:
    """Each derived shape, present as a parent id, must leave inference intact."""
    _db, add, bulk, connect = store
    bulk(20)
    for index, parent in enumerate(DERIVED_IDS):
        add(parent, source="cron" if parent.startswith("cron_") else "cli")
        add(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_kid{index:03d}",
            parent_session_id=parent)
    conn = connect()
    try:
        columns, rows = _sessions_rows(conn)
        mapping, kept = _infer_like_recovery(conn, columns, rows)
        assert mapping is not None and kept == len(rows)
        assert all(mapping[i] == columns[i] for i in range(len(columns)))
    finally:
        conn.close()


# ── 4. the safety property the fix must not weaken ─────────────────────


def test_a_wrong_layout_is_still_rejected(store) -> None:
    """Adversarial: the TRUE layout is accepted and rotated (wrong) layouts are
    still vetoed, with derived-id rows present. Widening the recognizers must buy
    recall, never wrong-column mappings."""
    _db, add, bulk, connect = store
    bulk(10)
    add(CRON_ID, source="cron")
    add(BG_ID)
    conn = connect()
    try:
        columns, rows = _sessions_rows(conn)
        types = _declared_types(conn, "sessions")

        def layout_accepts(layout: tuple[str, ...]) -> bool:
            fits = all(_cell_fits("sessions", layout[i], row[i], types)
                       for row in rows for i in range(len(layout)))
            return fits and _row_invariants_hold("sessions", layout, rows)

        assert layout_accepts(columns), "the real layout must still be accepted"
        for shift in range(1, 6):
            rotated = columns[shift:] + columns[:shift]
            assert not layout_accepts(rotated), (
                f"a layout rotated by {shift} was accepted -- the id/text-shape "
                f"sentinels no longer reject a wrong column mapping"
            )
    finally:
        conn.close()


def test_a_genuinely_corrupt_id_still_fails_classification(store) -> None:
    """Torn/garbled cells must NOT be waved through. Recall for real shapes is not
    tolerance for junk -- the predicate stays an exact-shape check."""
    _db, _add, bulk, connect = store
    bulk(5)
    conn = connect()
    try:
        columns, rows = _sessions_rows(conn)
        for junk in ("s", "session-key", "\x00\x01garbled", "cron_abc_2026"):
            torn = (junk,) + rows[0][1:]
            assert classify_lost_and_found_row(len(columns), torn) != "sessions", (
                f"{junk!r} was accepted as a session row"
            )
    finally:
        conn.close()
