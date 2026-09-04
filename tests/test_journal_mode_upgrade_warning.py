"""An existing database's journal mode must not be rewritten silently (#89293).

``PRAGMA journal_mode`` is a property of the FILE. Switching an existing
database to WAL rewrites its header and outlives the process that did it.

``apply_wal_with_fallback`` already treats on-disk WAL as authoritative and
refuses to live-downgrade it. The mirror case had no protection at all: an
on-disk DELETE database was flipped to WAL by the *default* configured value,
with no log line -- so an operator who set DELETE on the file (the documented
mitigation for the SQLite 3.50.4 WAL-reset bug) had no way to learn their
choice had been overwritten, or that ``database.journal_mode`` is the lever
that makes it stick.

#89293 is the field report: after upgrading past the vulnerable SQLite,
``is_sqlite_wal_reset_vulnerable()`` stopped short-circuiting to DELETE and
4 of 5 databases silently returned to WAL.

This is a LOG-ONLY change. Half of these tests exist to prove that: the return
value, the resulting header, and the never-live-downgrade rule are all pinned
unchanged, and the brand-new-database case must stay silent or the warning
would fire on every fresh install.
"""

from __future__ import annotations

import sqlite3

import pytest
import yaml


def _write_config(monkeypatch: pytest.MonkeyPatch, tmp_path, config: object) -> None:
    home = tmp_path / "hermes-home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _configure_mode(monkeypatch: pytest.MonkeyPatch, tmp_path, mode: object) -> None:
    _write_config(monkeypatch, tmp_path, {"database": {"journal_mode": mode}})


def _disable_vulnerable_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "hermes_state.is_sqlite_wal_reset_vulnerable",
        lambda **kwargs: False,
    )


def _make_delete_db_with_content(path) -> None:
    """Create a real, non-empty database left in journal_mode=DELETE."""
    conn = sqlite3.connect(str(path))
    try:
        assert conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0].lower() == "delete"
        conn.execute("CREATE TABLE t (x)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _reset_dedup():
    """Order-independence: the warning is deduped per process per db_label."""
    import hermes_state

    hermes_state._journal_upgrade_warned_paths.clear()
    yield
    hermes_state._journal_upgrade_warned_paths.clear()


class TestTheContentProbe:
    """``_database_has_content`` is what keeps fresh installs quiet."""

    def test_a_brand_new_database_has_no_content(self, tmp_path):
        from hermes_state import _database_has_content

        conn = sqlite3.connect(str(tmp_path / "new.db"))
        try:
            assert _database_has_content(conn) is False
        finally:
            conn.close()

    def test_a_database_with_a_table_has_content(self, tmp_path):
        from hermes_state import _database_has_content

        path = tmp_path / "used.db"
        _make_delete_db_with_content(path)
        conn = sqlite3.connect(str(path))
        try:
            assert _database_has_content(conn) is True
        finally:
            conn.close()

    def test_an_unreadable_probe_answers_no_content(self, tmp_path):
        """Fail-quiet.

        Answering True on an error would emit the warning for a database we
        could not measure, which includes every fresh one.
        """
        from hermes_state import _database_has_content

        conn = sqlite3.connect(":memory:")
        try:
            conn.close()
            assert _database_has_content(conn) is False
        finally:
            pass


class TestTheWarningFires:

    def test_an_existing_delete_database_warns_when_flipped(
        self, monkeypatch, tmp_path, caplog
    ):
        from hermes_state import apply_wal_with_fallback

        _configure_mode(monkeypatch, tmp_path, "wal")
        _disable_vulnerable_gate(monkeypatch)
        path = tmp_path / "existing-delete.db"
        _make_delete_db_with_content(path)

        conn = sqlite3.connect(str(path))
        try:
            with caplog.at_level("WARNING", logger="hermes_state"):
                assert apply_wal_with_fallback(conn, db_label="state.db") == "wal"
        finally:
            conn.close()

        blob = "\n".join(r.getMessage() for r in caplog.records)
        assert "state.db" in blob
        assert "delete" in blob.lower()

    def test_the_warning_names_the_setting_that_makes_it_stick(
        self, monkeypatch, tmp_path, caplog
    ):
        """The whole point.

        Telling an operator their mode changed, without telling them which
        lever survives an open, leaves them doing the same PRAGMA again.
        """
        from hermes_state import apply_wal_with_fallback

        _configure_mode(monkeypatch, tmp_path, "wal")
        _disable_vulnerable_gate(monkeypatch)
        path = tmp_path / "existing-delete.db"
        _make_delete_db_with_content(path)

        conn = sqlite3.connect(str(path))
        try:
            with caplog.at_level("WARNING", logger="hermes_state"):
                apply_wal_with_fallback(conn, db_label="state.db")
        finally:
            conn.close()

        blob = "\n".join(r.getMessage() for r in caplog.records)
        assert "database.journal_mode" in blob, (
            "the warning must name the config key, not just report the change"
        )

    def test_the_flip_still_happens(self, monkeypatch, tmp_path):
        """Log-only: WAL is still applied, and it still persists.

        Staying on DELETE is itself treated as a defect elsewhere in the tree
        (managed_uv repairs it on update, citing ~2600x slower appends), so
        this must warn about the change without preventing it.
        """
        from hermes_state import apply_wal_with_fallback

        _configure_mode(monkeypatch, tmp_path, "wal")
        _disable_vulnerable_gate(monkeypatch)
        path = tmp_path / "existing-delete.db"
        _make_delete_db_with_content(path)

        conn = sqlite3.connect(str(path))
        try:
            assert apply_wal_with_fallback(conn, db_label="state.db") == "wal"
            assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        finally:
            conn.close()

    def test_it_fires_once_per_process_per_database(
        self, monkeypatch, tmp_path, caplog
    ):
        """kanban opens a connection per operation; undeduped this is a flood."""
        from hermes_state import apply_wal_with_fallback

        _configure_mode(monkeypatch, tmp_path, "wal")
        _disable_vulnerable_gate(monkeypatch)

        with caplog.at_level("WARNING", logger="hermes_state"):
            for name in ("a", "b"):
                path = tmp_path / f"{name}.db"
                _make_delete_db_with_content(path)
                conn = sqlite3.connect(str(path))
                try:
                    apply_wal_with_fallback(conn, db_label="same-label.db")
                finally:
                    conn.close()

        hits = [r for r in caplog.records if "same-label.db" in r.getMessage()]
        assert len(hits) == 1, f"expected one deduped warning, got {len(hits)}"

    def test_a_second_database_gets_its_own_warning(
        self, monkeypatch, tmp_path, caplog
    ):
        """#89293 saw four databases flip. Dedup is per label, not global."""
        from hermes_state import apply_wal_with_fallback

        _configure_mode(monkeypatch, tmp_path, "wal")
        _disable_vulnerable_gate(monkeypatch)

        with caplog.at_level("WARNING", logger="hermes_state"):
            for label in ("state.db", "kanban.db"):
                path = tmp_path / f"{label}"
                _make_delete_db_with_content(path)
                conn = sqlite3.connect(str(path))
                try:
                    apply_wal_with_fallback(conn, db_label=label)
                finally:
                    conn.close()

        blob = "\n".join(r.getMessage() for r in caplog.records)
        assert "state.db" in blob and "kanban.db" in blob


class TestTheWarningStaysQuiet:
    """Every one of these would be a false positive shipped to every user."""

    def test_a_brand_new_database_is_silent(self, monkeypatch, tmp_path, caplog):
        """The load-bearing guard.

        A fresh file reports journal_mode=delete (SQLite's default) and is
        about to be switched to WAL, which looks identical to the reported
        bug from `current_mode` alone. Only page_count tells them apart, and
        every opener applies WAL before creating any schema -- so without
        this guard the warning fires on every first run of every install.
        """
        from hermes_state import apply_wal_with_fallback

        _configure_mode(monkeypatch, tmp_path, "wal")
        _disable_vulnerable_gate(monkeypatch)

        conn = sqlite3.connect(str(tmp_path / "fresh.db"))
        try:
            with caplog.at_level("WARNING", logger="hermes_state"):
                assert apply_wal_with_fallback(conn, db_label="fresh.db") == "wal"
        finally:
            conn.close()

        assert not [r for r in caplog.records if "fresh.db" in r.getMessage()]

    def test_an_existing_wal_database_is_silent(self, monkeypatch, tmp_path, caplog):
        """No flip happens: the probe returns early. Nothing to report."""
        from hermes_state import apply_wal_with_fallback

        _configure_mode(monkeypatch, tmp_path, "wal")
        _disable_vulnerable_gate(monkeypatch)
        path = tmp_path / "already-wal.db"
        conn = sqlite3.connect(str(path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE t (x)")
            conn.commit()
            with caplog.at_level("WARNING", logger="hermes_state"):
                assert apply_wal_with_fallback(conn, db_label="already-wal.db") == "wal"
        finally:
            conn.close()

        assert not [r for r in caplog.records if "already-wal.db" in r.getMessage()]

    def test_configured_delete_is_silent(self, monkeypatch, tmp_path, caplog):
        """The operator used the durable lever. There is nothing to tell them."""
        from hermes_state import apply_wal_with_fallback

        _configure_mode(monkeypatch, tmp_path, "delete")
        _disable_vulnerable_gate(monkeypatch)
        path = tmp_path / "configured-delete.db"
        _make_delete_db_with_content(path)

        conn = sqlite3.connect(str(path))
        try:
            with caplog.at_level("WARNING", logger="hermes_state"):
                assert apply_wal_with_fallback(conn, db_label="configured-delete.db") == "delete"
            assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
        finally:
            conn.close()

        assert not [
            r for r in caplog.records if "configured-delete.db" in r.getMessage()
        ]

    def test_the_wal_reset_vulnerable_path_is_silent(
        self, monkeypatch, tmp_path, caplog
    ):
        """That branch returns before any flip, so it must not warn.

        It is also the branch that KEPT #89293's databases on DELETE before
        the SQLite upgrade -- warning here would blame the guard that was
        doing its job.
        """
        from hermes_state import apply_wal_with_fallback

        _configure_mode(monkeypatch, tmp_path, "wal")
        monkeypatch.setattr(
            "hermes_state.is_sqlite_wal_reset_vulnerable",
            lambda **kwargs: True,
        )
        path = tmp_path / "vulnerable.db"
        _make_delete_db_with_content(path)

        conn = sqlite3.connect(str(path))
        try:
            with caplog.at_level("WARNING", logger="hermes_state"):
                apply_wal_with_fallback(conn, db_label="vulnerable.db")
        finally:
            conn.close()

        assert not [
            r
            for r in caplog.records
            if "database.journal_mode" in r.getMessage()
        ]


class TestTheExistingContractIsUnchanged:
    """Behaviour preservation for the rules this change sits next to."""

    def test_on_disk_wal_is_still_never_live_downgraded(self, monkeypatch, tmp_path):
        from hermes_state import apply_wal_with_fallback

        _configure_mode(monkeypatch, tmp_path, "delete")
        path = tmp_path / "existing-wal.db"
        conn = sqlite3.connect(str(path))
        try:
            assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
            monkeypatch.setattr(
                "hermes_state.is_sqlite_wal_reset_vulnerable",
                lambda **kwargs: True,
            )
            assert apply_wal_with_fallback(conn, db_label="existing-wal.db") == "wal"
            assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        finally:
            conn.close()

    def test_default_config_still_yields_wal_on_a_fresh_database(
        self, monkeypatch, tmp_path
    ):
        from hermes_state import apply_wal_with_fallback

        _configure_mode(monkeypatch, tmp_path, "wal")
        _disable_vulnerable_gate(monkeypatch)
        conn = sqlite3.connect(str(tmp_path / "default.db"))
        try:
            assert apply_wal_with_fallback(conn, db_label="default.db") == "wal"
            assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        finally:
            conn.close()
