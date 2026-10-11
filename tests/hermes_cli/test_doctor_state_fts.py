from hermes_cli import doctor_state


def test_large_db_with_structurally_stale_trigram_gets_optimize_notice():
    rows = doctor_state._render_state_db_stats({
        "logical_size_bytes": doctor_state.STATE_DB_SIZE_WARN_BYTES + 1,
        "fts_tables": {"messages_fts_trigram": True},
        "fts_storage_version": 3,
        "fts_storage_upgrade_pending": True,
    })

    assert any("hermes sessions optimize-storage" in detail for _kind, _text, detail in rows)