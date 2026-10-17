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
