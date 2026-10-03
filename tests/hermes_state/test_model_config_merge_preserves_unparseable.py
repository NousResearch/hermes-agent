"""Regression tests: a failed ``model_config`` decode must never be written back.

``_merge_model_config_json`` is the one place that keeps ``_branched_from`` /
``_delegate_from`` / ``_reset_from`` alive. ``_parse_model_config`` collapsed
malformed JSON and JSON scalars to ``{}`` (indistinguishable from a legal empty
config), so the merge then wrote the patched empty dict back as the WHOLE
``model_config`` — one bad row plus one routine ``/yolo`` toggle permanently
destroyed every lineage marker in it.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def _raw_config(db: SessionDB, session_id: str) -> Optional[str]:
    row = db._execute_write(
        lambda conn: conn.execute(
            "SELECT model_config FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    )
    return row[0] if row else None


def _session_with_raw_config(db: SessionDB, session_id: str, raw: Optional[str]) -> None:
    db.create_session(session_id, source="webui")
    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE sessions SET model_config = ? WHERE id = ?", (raw, session_id)
        )
    )


def test_merge_refuses_unparseable_config_and_preserves_row(
    db: SessionDB, caplog
) -> None:
    """Truncated JSON + one /yolo toggle must not rewrite the row's config."""
    raw = '{"yolo_mode": false, "_delegate_from": "abc"'  # truncated write
    _session_with_raw_config(db, "s1", raw)

    with caplog.at_level(logging.ERROR, logger="hermes_state"):
        db.set_session_yolo("s1", True)

    assert _raw_config(db, "s1") == raw  # byte-identical: the failed decode never writes
    assert "failed to decode" in caplog.text
    assert "refusing" in caplog.text


def test_merge_refuses_non_dict_json_and_preserves_row(db: SessionDB) -> None:
    """A JSON scalar/array payload is also a failed decode — never written back."""
    _session_with_raw_config(db, "s2", '"5"')

    db.set_session_yolo("s2", True)

    assert _raw_config(db, "s2") == '"5"'


@pytest.mark.parametrize("raw", ['{"_delegate_from": "abc"', '"5"', '[1, 2]'])
@pytest.mark.parametrize("coverage", ["covered_ids", "watermark"])
def test_compaction_commits_without_rewriting_invalid_config(db, raw, coverage):
    _session_with_raw_config(db, "compact", raw)
    db.append_message("compact", "user", "old")
    watermark = db.get_active_message_watermark("compact")
    kwargs = {coverage: [watermark] if coverage == "covered_ids" else watermark}

    assert db.archive_and_compact(
        "compact", [{"role": "assistant", "content": "summary"}],
        model_config_patch={"_proactive_pruned": True}, **kwargs,
    ) == 1
    assert _raw_config(db, "compact") == raw
    assert [m["content"] for m in db.get_messages("compact")] == ["summary"]
    assert db.get_session("compact")["message_count"] == 1
    rows = db._execute_write(lambda conn: conn.execute(
        "SELECT active, compacted FROM messages WHERE session_id = ? AND content = ?",
        ("compact", "old"),
    ).fetchall())
    assert [(r["active"], r["compacted"]) for r in rows] == [(0, 1)]


@pytest.mark.parametrize("raw", ["", None, "{}"])
def test_empty_config_still_merges_normally(db: SessionDB, raw: Optional[str]) -> None:
    """'' / NULL / '{}' remain the legal empty-config shapes: patch applies."""
    sid = f"s_empty_{raw!r}"
    _session_with_raw_config(db, sid, raw)

    db.set_session_yolo(sid, True)

    stored = _raw_config(db, sid)
    assert stored is not None
    assert json.loads(stored) == {"yolo_mode": True}
