"""Every ``database.*`` key the state.db connection loader reads is a key ``hermes config`` recognises.

``apply_database_pragmas`` reads ``cache_size``, ``mmap_size``, ``temp_store`` and ``synchronous`` (the
durability switch ``cli-config.yaml.example`` documents) besides the two keys DEFAULT_CONFIG seeds, so
``hermes config set database.synchronous NORMAL`` worked but printed "not a recognized config key", and
anything that validates keys before writing them refused it.

The first two tests record every key the loader asks for and hold each one to the validator, so a
pragma added to the loader without a line in ``config_runtime_keys.CORE_RUNTIME_KEYS`` fails here.
"""

import sqlite3

import pytest

NOTICE = "not a recognized config key"
PRAGMA_KEYS = {
    "database.synchronous",
    "database.cache_size",
    "database.mmap_size",
    "database.temp_store",
    "database.wal_autocheckpoint",
    "database.journal_size_limit",
}


def _keys_the_loader_reads(monkeypatch, tmp_path):
    """Run the real loader on a real connection, with ``cfg_get`` recording every path it is asked for."""
    import hermes_cli.config as config_module
    from hermes_state_wal import apply_database_pragmas

    asked = []

    def spy(cfg, *path, default=None):
        asked.append(".".join(path))
        return default

    monkeypatch.setattr(config_module, "cfg_get", spy)
    monkeypatch.setattr(config_module, "load_config_readonly", dict)
    conn = sqlite3.connect(str(tmp_path / "pragmas.db"))
    try:
        apply_database_pragmas(conn, db_label="test.db")
    finally:
        conn.close()
    return asked


def test_the_loader_reads_the_six_database_pragma_keys(monkeypatch, tmp_path):
    assert PRAGMA_KEYS <= set(_keys_the_loader_reads(monkeypatch, tmp_path))


def test_every_key_the_loader_reads_is_recognised(monkeypatch, tmp_path):
    from hermes_cli.config import _validate_config_key

    asked = {key for key in _keys_the_loader_reads(monkeypatch, tmp_path) if key.startswith("database.")}
    assert asked, "the loader asked for no database.* key at all"
    unknown = sorted(key for key in asked if not _validate_config_key(key)[0])
    assert not unknown, f"read by apply_database_pragmas but unknown to `hermes config`: {unknown}"


@pytest.mark.parametrize("key,value", [
    ("database.synchronous", "NORMAL"),
    ("database.cache_size", "-65536"),
    ("database.mmap_size", "268435456"),
    ("database.temp_store", "2"),
])
def test_config_set_of_a_pragma_key_prints_no_notice(key, value, capsys):
    from hermes_cli.config import set_config_value

    set_config_value(key, value)
    out = capsys.readouterr().out
    assert f"Set {key}" in out
    assert NOTICE not in out


def test_a_typo_under_database_still_gets_the_notice(capsys):
    from hermes_cli.config import set_config_value

    set_config_value("database.synchronus", "NORMAL")
    assert NOTICE in capsys.readouterr().out


@pytest.mark.parametrize("word,level", [("NORMAL", 1), ("FULL", 2), ("normal", 1)])
def test_the_value_config_set_writes_is_the_level_the_connection_runs_at(word, level, tmp_path, capsys):
    """Read the effect back through the loader every connection runs, not the config text."""
    from hermes_cli.config import set_config_value
    from hermes_state_wal import apply_database_pragmas

    set_config_value("database.synchronous", word)
    capsys.readouterr()
    conn = sqlite3.connect(str(tmp_path / "level.db"))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        apply_database_pragmas(conn, db_label="test.db")
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == level
    finally:
        conn.close()
