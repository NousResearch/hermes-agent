"""``DEFAULT_CONFIG["database"]`` for ``hermes_cli.config_defaults`` (pure-data leaf, no imports).

Split out of the ``config_defaults`` facade so the SQLite/WAL knobs and the retired-WAL capture
sizing can grow without growing the facade (code health: FILE_LINES).
"""


def database_defaults():
    """The ``database`` section: journal mode, WAL sizing pragmas, retired-WAL capture cap."""
    return {
        # journal_mode: SQLite journal mode for every Hermes DB. "wal" default; use "delete" on
        # weak-fsync/shared filesystems where WAL is not crash-safe (macOS virtiofs, NFS, SMB).
        "journal_mode": "wal",
        # WAL sizing pragmas (ints). None = SQLite defaults (autocheckpoint 1000 pages, no limit).
        "wal_autocheckpoint": None,
        "journal_size_limit": None,
        # Byte cap on the main-image copy inside a retired-WAL capture artifact (hermes_state_dbfile):
        # above it the artifact keeps only the 100-byte SQLite header, and says so in the log and in
        # manifest.json's main.mode. None = the shipped 512 MiB, which a long-lived store outgrows.
        "retired_generation_max_bytes": None,
    }
