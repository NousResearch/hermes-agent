"""Regression coverage for FTS storage upgrade discoverability."""

import sqlite3
from types import SimpleNamespace

import pytest


def test_update_notice_offers_v1_trigram_tool_calls_rebuild(tmp_path, monkeypatch, capsys):
    """A deployed v1 trigram projection still receives the opt-in notice."""
    from hermes_cli import update_cmd
    import hermes_constants
    import hermes_state

    db_path = tmp_path / "state.db"
    db_path.touch()
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE messages_fts (content TEXT, tool_name TEXT, tool_calls TEXT);
        CREATE TABLE messages_fts_trigram (content TEXT, tool_name TEXT, tool_calls TEXT);
        """
    )

    class FakeSessionDB:
        def __init__(self, **_kwargs):
            self._conn = conn

        def close(self):
            pass

        _db_needs_fts_storage_upgrade = staticmethod(
            hermes_state.SessionDB._db_needs_fts_storage_upgrade
        )

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(hermes_state, "SessionDB", FakeSessionDB)
    # Report a large state.db without patching Path.stat globally: a
    # 1-arg lambda on the class breaks pathlib.exists(follow_symlinks=...)
    # for every caller in the process (pytest's own teardown included).
    real_stat = update_cmd.Path.stat

    def _stat(path, *args, **kwargs):
        if path.name == "state.db":
            return SimpleNamespace(st_size=512 * 1024 ** 2)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(update_cmd.Path, "stat", _stat)

    update_cmd._print_fts_optimize_available_notice()

    assert "hermes sessions optimize-storage" in capsys.readouterr().out
    conn.close()


@pytest.mark.parametrize("base_layout", ["absent", "legacy", "current"])
@pytest.mark.parametrize("trigram_layout", ["absent", "legacy", "current"])
@pytest.mark.parametrize("marker", [None, 1, 3])
@pytest.mark.parametrize("large", [False, True])
def test_doctor_collects_structural_fts_upgrade_status(
    tmp_path, monkeypatch, base_layout, trigram_layout, marker, large,
):
    """Real read-only collection must ignore forged or lagging version markers."""
    from hermes_cli import doctor_state
    from hermes_state_dbfile import collect_state_db_stats

    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            "CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT);"
            "CREATE TABLE messages (content TEXT, tool_name TEXT, tool_calls TEXT);"
            "CREATE TABLE sessions (id TEXT);"
        )
        if base_layout == "legacy":
            conn.execute("CREATE VIRTUAL TABLE messages_fts USING fts5(content)")
        elif base_layout == "current":
            conn.execute(
                "CREATE VIRTUAL TABLE messages_fts USING "
                "fts5(content, tool_name, tool_calls, content='messages')"
            )
        if trigram_layout != "absent":
            columns = "content, tool_name" + (", tool_calls" if trigram_layout == "legacy" else "")
            conn.execute(f"CREATE VIRTUAL TABLE messages_fts_trigram USING fts5({columns}, tokenize='trigram')")
        if marker is not None:
            conn.execute("INSERT INTO state_meta VALUES ('fts_storage_version', ?)", (str(marker),))

    before = db_path.read_bytes()
    stats = collect_state_db_stats(db_path)
    needed = base_layout == "legacy" or trigram_layout == "legacy"
    assert stats["fts_storage_upgrade_needed"] is needed
    assert stats["fts_storage_version"] == marker
    assert stats["fts_rebuild_pending"] is False
    assert stats["logical_size_bytes"] > 0
    # Exercise the size gate without allocating a GiB database or forging collected stats.
    monkeypatch.setattr(doctor_state, "STATE_DB_SIZE_WARN_BYTES", 0 if large else stats["logical_size_bytes"])
    rows = doctor_state._render_state_db_stats(stats)
    offered = any("hermes sessions optimize-storage" in detail for _, _, detail in rows)
    assert offered is (large and needed)
    assert db_path.read_bytes() == before


@pytest.mark.parametrize("progress, pending", [(0, True), (10, False)])
def test_doctor_collects_pending_rebuild_with_settled_layout(tmp_path, monkeypatch, progress, pending):
    from hermes_cli import doctor_state
    from hermes_state_dbfile import collect_state_db_stats

    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            "CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT);"
            "CREATE VIRTUAL TABLE messages_fts USING fts5(content, tool_name, tool_calls);"
            "CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(content, tool_name, tokenize='trigram');"
            "INSERT INTO state_meta VALUES ('fts_rebuild_high_water', '10');"
        )
        conn.execute("INSERT INTO state_meta VALUES ('fts_rebuild_progress', ?)", (str(progress),))
    stats = collect_state_db_stats(db_path)
    assert stats["fts_storage_upgrade_needed"] is False
    assert stats["fts_rebuild_pending"] is pending
    monkeypatch.setattr(doctor_state, "STATE_DB_SIZE_WARN_BYTES", 0)
    rows = doctor_state._render_state_db_stats(stats)
    assert any("hermes sessions optimize-storage" in detail for _, _, detail in rows) is pending
