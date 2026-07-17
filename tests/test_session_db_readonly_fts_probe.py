from hermes_state import SessionDB


def test_read_only_session_db_probes_existing_fts_tables(tmp_path):
    db_path = tmp_path / "state.db"
    writer = SessionDB(db_path=db_path)
    try:
        assert writer._fts_enabled is True
        expected_trigram = writer._trigram_available
    finally:
        writer.close()

    reader = SessionDB(db_path=db_path, read_only=True)
    try:
        assert reader._fts_enabled is True
        assert reader._trigram_available is expected_trigram
    finally:
        reader.close()
