from __future__ import annotations

from pathlib import Path

from hermes_state import SessionDB


def test_get_session_lineage_rich_returns_compression_segments(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    root_id = db.create_session("root-session", "tui")
    db.set_session_title(root_id, "Before compression")
    db.append_message(root_id, "user", "first debugging prompt")
    db.end_session(root_id, "compression")

    tip_id = db.create_session("tip-session", "tui", parent_session_id=root_id)
    db.set_session_title(tip_id, "After compression")
    db.append_message(tip_id, "user", "continued debugging prompt")

    lineage = db.get_session_lineage_rich(tip_id)

    assert [row["id"] for row in lineage] == [root_id, tip_id]
    assert [row["title"] for row in lineage] == ["Before compression", "After compression"]
    assert lineage[0]["preview"] == "first debugging prompt"
    assert lineage[1]["preview"] == "continued debugging prompt"
