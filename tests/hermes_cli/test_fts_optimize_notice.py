"""Regression coverage for FTS storage upgrade discoverability."""

import sqlite3
from types import SimpleNamespace


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
