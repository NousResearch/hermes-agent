"""Session portability must preserve title ownership and upgrade behavior."""

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("source", ["derived", "llm", "user", None])
def test_export_import_preserves_title_authority(tmp_path, source):
    donor = SessionDB(tmp_path / "donor.db")
    restored = SessionDB(tmp_path / "restored.db")
    try:
        session_id = "portable-title"
        donor.create_session(session_id, "cli")
        title = "Investigate the failing build"
        if source in ("derived", "llm"):
            donor.set_auto_title(session_id, title, source=source)
        else:
            donor.set_session_title(session_id, title)
        payload = donor.export_session(session_id)
        assert payload is not None
        if source is None:
            # Older exports predate provenance; keep their user-title protection.
            payload.pop("title_source")
        assert restored.import_sessions([payload])["ok"]
        assert restored.get_session_title(session_id) == title
        restored_source = restored.get_session_title_source(session_id)
        upgraded = restored.set_auto_title(
            session_id, "Diagnose build failure", source="llm"
        )
        assert upgraded == (source == "derived")
        assert restored_source == source
        assert restored.get_session_title(session_id) == (
            "Diagnose build failure" if upgraded else title
        )
    finally:
        restored.close()
        donor.close()


@pytest.mark.parametrize("source", ["unknown", {"source": "derived"}])
def test_invalid_title_provenance_rejects_whole_import(tmp_path, source):
    db = SessionDB(tmp_path / "target.db")
    try:
        result = db.import_sessions([
            {"id": "valid", "messages": []},
            {"id": "invalid", "title": "Imported title", "title_source": source, "messages": []},
        ])
        assert not result["ok"]
        assert result["imported"] == 0
        assert db.get_session("valid") is None
        assert db.get_session("invalid") is None
    finally:
        db.close()
