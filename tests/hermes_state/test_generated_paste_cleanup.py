from hermes_state import SessionDB


def test_delete_session_reclaims_only_unreferenced_generated_pastes(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    sessions_dir = tmp_path / "sessions"
    paste_dir = tmp_path / "composer-pastes"
    paste_dir.mkdir()

    shared = paste_dir / "pasted_content_shared.txt"
    doomed = paste_dir / "pasted_content_doomed.txt"
    ordinary = paste_dir / "manual-notes.txt"
    for path in (shared, doomed, ordinary):
        path.write_text(path.name, encoding="utf-8")

    db.create_session("deleted", source="desktop")
    db.create_session("survivor", source="desktop")
    db.append_message("deleted", "user", f"@file:{shared} @file:{doomed}")
    db.append_message("survivor", "user", f"@file:{shared}")

    assert db.delete_session("deleted", sessions_dir=sessions_dir) is True

    assert shared.exists()
    assert not doomed.exists()
    assert ordinary.exists()
    db.close()
