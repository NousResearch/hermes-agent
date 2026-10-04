"""Session recovery must recognise every session-id shape this repo really mints.

``hermes sessions recover`` classifies schema-less lost_and_found rows by the id cell. Ids that
surfaces DERIVE (cron, /bg, rooms, API server, ACP) instead of minting via ``new_session_id`` were
dropped, and as a ``parent_session_id`` one of them vetoed layout inference for the whole table.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from typing import Optional

import pytest

from hermes_state import SessionDB
from hermes_cli.session_recovery import _quoted_columns, _table_columns
from hermes_cli.session_lost_and_found import (
    LayoutEvidence,
    _declared_types,
    _is_session_id,
    classify_lost_and_found_row,
    infer_physical_layouts,
)

# Exactly as the minting sites build them (see module docstring for file/line).
CRON_ID = "cron_0b36145ae5c7_20260626_100059"
CRON_SLUG_ID = "cron_job-1_20260707_195958"      # legacy/user job id, real in live stores
BG_ID = "bg_124526_be5bed"
ROOM_ID = "room_" + "0123456789abcdef" * 2

TUI_BG_ID = "bg_be5bed"
API_ID = "api_1790000000_0123abcd"
API_CHAT_ID = "api-0123456789abcdef"
RUN_ID = "run_" + "0123456789abcdef" * 2
UUID_ID = "0b36145a-e5c7-4d2a-9f00-0123456789ab"

DERIVED_IDS = [CRON_ID, CRON_SLUG_ID, BG_ID, ROOM_ID, TUI_BG_ID, API_ID, API_CHAT_ID, RUN_ID, UUID_ID]

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
    "bg_abcde", "api_123_0123abcd", "api-0123", "run_" + "a" * 31,
    "0b36145a-e5c7-1d2a-9f00-0123456789ab",
]


@pytest.mark.parametrize("value,expected", [(v, True) for v in DERIVED_IDS] + [(v, False) for v in NOT_SESSION_IDS])
def test_salvage_recognizes_minted_ids_and_rejects_arbitrary_cells(value: str, expected: bool) -> None:
    assert _is_session_id(value) is expected


def _sessions_rows(conn: sqlite3.Connection) -> tuple[list[str], list[tuple]]:
    columns = _table_columns(conn, "sessions")
    quoted = _quoted_columns(columns)[0]
    return columns, [tuple(r) for r in conn.execute(f"SELECT {quoted} FROM sessions")]


def _infer_like_recovery(conn: sqlite3.Connection, columns: list[str],
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
def store(tmp_path):
    """A real post-upgrade store; the factory seeds sessions through the real SessionDB."""
    with SessionDB(db_path=tmp_path / "state.db") as db:

        def add(session_id: str, **kwargs) -> str:
            db.create_session(session_id, source=kwargs.pop("source", "cli"),
                              model="claude-opus-5", cwd=str(tmp_path), **kwargs)
            return session_id

        def bulk(count: int) -> None:
            for _ in range(count):
                add(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}")

        yield add, bulk, lambda: sqlite3.connect(str(tmp_path / "state.db"))


def test_layout_inference_survives_every_derived_id_shape(store) -> None:
    """Each derived shape, present as a parent id, must leave inference intact."""
    add, bulk, connect = store
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
