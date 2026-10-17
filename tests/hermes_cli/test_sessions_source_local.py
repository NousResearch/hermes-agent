"""``hermes sessions list|browse|stats --source cli`` must see gateway rows stored as ``local``."""
from argparse import Namespace

from hermes_state import SessionDB


def test_list_and_browse_source_cli_include_the_gateway_row(tmp_path, capsys, monkeypatch):
    from hermes_cli.sessions_cmd import _cmd_browse, _cmd_list

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("cli-session", source="local")
    db.create_session("desk-session", source="local")
    with db._lock:
        db._conn.executemany(
            "UPDATE sessions SET created_source=?, started_at=? WHERE id=?",
            [("cli", 1, "cli-session"), ("desktop", 2, "desk-session")],
        )
        db._conn.commit()

    _cmd_list(db, Namespace(limit=20, source="cli", workspace=None))
    listed = capsys.readouterr().out
    assert "cli-session" in listed
    assert "desk-session" not in listed
    assert "No sessions found." not in listed

    offered = {}

    def picker(sessions, session_db=None):
        offered["ids"] = [row["id"] for row in sessions]

    monkeypatch.setattr("hermes_cli.sessions_cmd._session_browse_picker", picker)
    _cmd_browse(db, Namespace(limit=20, source="cli"))
    assert offered["ids"] == ["cli-session"]
    assert "No sessions found." not in capsys.readouterr().out


def test_stats_counts_cli_segments_while_list_shows_one_conversation(tmp_path, capsys):
    """Stats counts every cli segment. List shows the conversation once, as its tip."""
    from hermes_cli.sessions_cmd import _cmd_list, _cmd_stats

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("segment-root", source="local")
    with db._lock:
        db._conn.execute("UPDATE sessions SET created_source=? WHERE id=?", ("cli", "segment-root"))
        db._conn.commit()
    db.append_message("segment-root", role="user", content="hi")
    db.publish_compression_child(
        parent_session_id="segment-root", child_session_id="segment-tip", source="local",
        messages=[{"role": "user", "content": "hi"}], require_compression_lease=False)
    _cmd_list(db, Namespace(limit=20, source="cli", workspace=None))
    listed = capsys.readouterr().out
    assert "segment-tip" in listed
    assert "segment-root" not in listed
    _cmd_stats(db, Namespace())
    assert "cli: 2 sessions" in capsys.readouterr().out


def test_list_includes_a_compression_tip_whose_root_has_no_created_source(tmp_path):
    """Search and stats see a tip healed to cli. List must show that conversation too.

    The page hides compression children and paints the root as the tip, so the usage
    write that heals the tip also stamps the NULL ancestors of that chain.
    """
    from hermes_cli.gateway_session_store import SurfaceLabelReader

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("chain-root", source="local")
    with db._lock:
        db._conn.execute("UPDATE sessions SET created_source=NULL WHERE id=?", ("chain-root",))
        db._conn.commit()
    db.append_message("chain-root", role="user", content="hi")
    db.publish_compression_child(
        parent_session_id="chain-root", child_session_id="chain-mid", source="local",
        messages=[{"role": "user", "content": "hi"}], require_compression_lease=False)
    db.publish_compression_child(
        parent_session_id="chain-mid", child_session_id="chain-tip", source="local",
        messages=[{"role": "user", "content": "hi"}], require_compression_lease=False)
    db.update_token_counts(
        "chain-tip", input_tokens=2, output_tokens=1, model="m", api_call_count=1, source="cli")
    assert [db.get_session(sid)["created_source"] for sid in ("chain-root", "chain-mid", "chain-tip")] == [
        "cli", "cli", "cli"]
    reader = SurfaceLabelReader(db)
    listed = [row["id"] for row in reader.list_sessions_rich(source="cli", limit=10)]
    assert listed == ["chain-tip"]
    assert "chain-tip" in [row["id"] for row in reader.search_sessions(source="cli", limit=10)]
    assert reader.session_count(source="cli") == 3  # stats counts every segment
    db.close()


def test_reader_session_count_includes_gateway_rows_and_a_pre_column_store(tmp_path):
    """session_count through the reader matches a local/cli row, and a read-only
    store from before the column still counts its source=cli rows without migrating."""
    from hermes_cli.gateway_session_store import SurfaceLabelReader

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("gateway-cli", source="local")
    with db._lock:
        db._conn.execute("UPDATE sessions SET created_source=? WHERE id=?", ("cli", "gateway-cli"))
        db._conn.commit()
    assert SurfaceLabelReader(db).session_count(source="cli") == 1
    db.close()

    path = tmp_path / "old.db"
    writer = SessionDB(db_path=path)
    writer.create_session("legacy-cli", source="cli")
    with writer._lock:
        writer._conn.execute("DROP INDEX IF EXISTS idx_sessions_local_created_started")
        writer._conn.execute("ALTER TABLE sessions DROP COLUMN created_source")
        writer._conn.commit()
    writer.close()
    ro = SessionDB(db_path=path, read_only=True)
    try:
        assert SurfaceLabelReader(ro).session_count(source="cli") == 1
        columns = {row[1] for row in ro._conn.execute("PRAGMA table_info(sessions)")}
        assert "created_source" not in columns
    finally:
        ro.close()
