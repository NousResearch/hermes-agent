"""``hermes sessions list|browse|stats --source cli`` must see gateway rows stored as ``local``."""
from argparse import Namespace

from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage
from persistence.conversation.usage import BillingRoute, TokenCounts, UsageDelta
from persistence_assembly import open_persistence, open_persistence_reader, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_messages, seed_session


def test_list_includes_a_compression_tip_whose_root_has_no_created_source(tmp_path, capsys):
    """Search and stats see a tip healed to cli. List must show that conversation too.

    The page hides compression children and paints the root as the tip, so the usage
    write that heals the tip also stamps the NULL ancestors of that chain.
    """
    from hermes_cli.gateway_session_store import SurfaceLabelReader
    from hermes_cli.sessions_cmd import _cmd_list, _cmd_stats

    db = open_persistence(profile_scope(home=tmp_path))
    seed_session(db, "chain-root", "local")
    raw_holder(db)._execute_write(
        lambda conn: conn.execute("UPDATE sessions SET created_source=NULL WHERE id=?", ("chain-root",)))
    seed_messages(db, "chain-root", {"role": "user", "content": "hi"})
    # A subagent child is not a continuation: its own surface must not stamp the root.
    seed_session(db, "delegate-run", "subagent", parent_session_id="chain-root")
    assert session_row(db, "chain-root")["created_source"] is None
    db.lineage.publish_compression_child(
        "chain-root", CompressionChild(session_id="chain-mid", source="local"),
        [NewMessage.from_live({"role": "user", "content": "hi"})], require_compression_lease=False)
    db.lineage.publish_compression_child(
        "chain-mid", CompressionChild(session_id="chain-tip", source="local"),
        [NewMessage.from_live({"role": "user", "content": "hi"})], require_compression_lease=False)
    db.usage.queue_usage("chain-tip", UsageDelta(
        tokens=TokenCounts(input_tokens=2, output_tokens=1), api_call_count=1, route=BillingRoute(model="m"),
        source="cli"))
    db.usage.flush_usage()
    assert [session_row(db, sid)["created_source"] for sid in ("chain-root", "chain-mid", "chain-tip")] == [
        "cli", "cli", "cli"]
    capsys.readouterr()
    with open_persistence_reader(tmp_path / "state.db") as store:
        _cmd_list(Namespace(limit=20, source="cli", workspace=None), persistence=store)
        listed = capsys.readouterr().out
        assert "chain-tip" in listed and "chain-root" not in listed
        assert "chain-tip" in [row["id"] for row in SurfaceLabelReader(store).search_sessions(source="cli", limit=10)]
        _cmd_stats(Namespace(), persistence=store)
    assert "cli: 3 sessions" in capsys.readouterr().out  # stats counts every segment
    db.close()


def test_reader_on_a_read_only_store_from_before_the_column_and_the_runtime_tables(tmp_path):
    """A read-only open does not migrate. A store without ``created_source`` still counts its
    source=cli rows, and one without ``session_admissions`` still resolves ``-c`` by messages."""
    from hermes_cli.gateway_session_store import SurfaceLabelReader

    path = tmp_path / "state.db"

    def downgrade(conn):
        conn.execute("UPDATE sessions SET started_at = started_at + 60 WHERE id = 'legacy-stub'")
        conn.execute("DROP INDEX IF EXISTS idx_sessions_local_created_started")
        conn.execute("ALTER TABLE sessions DROP COLUMN created_source")
        conn.execute("DROP TABLE session_admissions")

    with open_persistence(profile_scope(home=tmp_path)) as writer:
        seed_session(writer, "legacy-cli", "cli")
        seed_messages(writer, "legacy-cli", {"role": "user", "content": "hi"})
        seed_session(writer, "legacy-stub", "cli")
        raw_holder(writer)._execute_write(downgrade)
    with open_persistence_reader(path) as store:
        reader = SurfaceLabelReader(store)
        assert reader.session_count(source="cli") == 2
        assert [row["id"] for row in reader.search_sessions(source="cli", limit=1, skip_launch_stubs=True)] == [
            "legacy-cli"]
    with open_persistence_reader(path) as ro, raw_holder(ro)._read_ctx() as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        assert "created_source" not in columns and "session_admissions" not in tables
