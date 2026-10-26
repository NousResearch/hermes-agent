"""``hermes sessions list|browse|stats --source cli`` must see gateway rows stored as ``local``."""
from argparse import Namespace

from hermes_state import SessionDB


def test_list_includes_a_compression_tip_whose_root_has_no_created_source(tmp_path, capsys):
    """Search and stats see a tip healed to cli. List must show that conversation too.

    The page hides compression children and paints the root as the tip, so the usage
    write that heals the tip also stamps the NULL ancestors of that chain.
    """
    from hermes_cli.gateway_session_store import SurfaceLabelReader
    from hermes_cli.sessions_cmd import _cmd_list, _cmd_stats

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("chain-root", source="local")
    with db._lock:
        db._conn.execute("UPDATE sessions SET created_source=NULL WHERE id=?", ("chain-root",))
        db._conn.commit()
    db.append_message("chain-root", role="user", content="hi")
    # A subagent child is not a continuation: its own surface must not stamp the root.
    db.create_session("delegate-run", source="subagent", parent_session_id="chain-root")
    assert db.get_session("chain-root")["created_source"] is None
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
    capsys.readouterr()
    _cmd_list(db, Namespace(limit=20, source="cli", workspace=None))
    listed = capsys.readouterr().out
    assert "chain-tip" in listed and "chain-root" not in listed
    assert "chain-tip" in [row["id"] for row in SurfaceLabelReader(db).search_sessions(source="cli", limit=10)]
    _cmd_stats(db, Namespace())
    assert "cli: 3 sessions" in capsys.readouterr().out  # stats counts every segment
    db.close()


def test_reader_on_a_read_only_store_from_before_the_column_and_the_runtime_tables(tmp_path):
    """A read-only open does not migrate. A store without ``created_source`` still counts its
    source=cli rows, and one without ``session_admissions`` still resolves ``-c`` by messages."""
    from hermes_cli.gateway_session_store import SurfaceLabelReader

    path = tmp_path / "old.db"
    writer = SessionDB(db_path=path)
    writer.create_session("legacy-cli", source="cli")
    writer.append_message("legacy-cli", role="user", content="hi")
    writer.create_session("legacy-stub", source="cli")
    with writer._lock:
        writer._conn.execute("UPDATE sessions SET started_at = started_at + 60 WHERE id = 'legacy-stub'")
        writer._conn.execute("DROP INDEX IF EXISTS idx_sessions_local_created_started")
        writer._conn.execute("ALTER TABLE sessions DROP COLUMN created_source")
        writer._conn.execute("DROP TABLE session_admissions")
        writer._conn.commit()
    writer.close()
    ro = SessionDB(db_path=path, read_only=True)
    try:
        reader = SurfaceLabelReader(ro)
        assert reader.session_count(source="cli") == 2
        assert [row["id"] for row in reader.search_sessions(source="cli", limit=1, skip_launch_stubs=True)] == [
            "legacy-cli"]
        tables = {row[0] for row in ro._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        columns = {row[1] for row in ro._conn.execute("PRAGMA table_info(sessions)")}
        assert "created_source" not in columns and "session_admissions" not in tables
    finally:
        ro.close()
