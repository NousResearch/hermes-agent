"""Tests for the per-plugin durable storage convention (plugins/plugin_storage).

The contract under test: data lives under ``<rabbit home>/plugin-data/<name>/``
(NOT the ``plugins/<name>/`` install tree), names that could escape the root
are rejected, and the sqlite helper opens a WAL-mode connection inside the
data dir.
"""

from __future__ import annotations

import pytest

from rabbit_constants import reset_rabbit_home_override, set_rabbit_home_override
from plugins.plugin_storage import plugin_data_dir, plugin_db


@pytest.fixture
def rabbit_home(tmp_path):
    token = set_rabbit_home_override(str(tmp_path))
    try:
        yield tmp_path
    finally:
        reset_rabbit_home_override(token)


def test_data_dir_lives_outside_the_install_tree(rabbit_home):
    root = plugin_data_dir("my-plugin")

    assert root == rabbit_home / "plugin-data" / "my-plugin"
    assert root.is_dir()
    # The invariant that motivated the module: data must not live under the
    # install tree that `rabbit plugins remove` deletes.
    assert (rabbit_home / "plugins") not in root.parents


@pytest.mark.parametrize("bad", ["", ".", "..", "../escape", "a/b", "a\\b", "x" * 65])
def test_hostile_names_are_rejected(rabbit_home, bad):
    with pytest.raises(ValueError):
        plugin_data_dir(bad)


def test_plugin_db_journal_mode_is_the_shared_fallback_verdict(rabbit_home):
    """Plugin DBs take the journal mode the core WAL helper decides for this SQLite build and
    filesystem (WAL normally; DELETE on WAL-reset-bug builds or network FS) — never a raw PRAGMA."""
    from rabbit_state_wal import is_sqlite_wal_reset_vulnerable

    conn = plugin_db("board")
    try:
        conn.execute("CREATE TABLE t (x)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()

        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == ("delete" if is_sqlite_wal_reset_vulnerable() else "wal")
    finally:
        conn.close()

    assert (rabbit_home / "plugin-data" / "board" / "data.db").exists()


def test_plugin_db_rejects_path_shaped_filenames(rabbit_home):
    with pytest.raises(ValueError):
        plugin_db("board", filename="../outside.db")
    with pytest.raises(ValueError):
        plugin_db("board", filename="")
