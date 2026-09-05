"""Bounded FTS-corruption recovery on live SessionDB paths.

A corrupted FTS5 shadow table (``messages_fts_data``) makes every message
write raise ``sqlite3.DatabaseError: database disk image is malformed``
through the FTS sync triggers, while the canonical ``messages`` rows stay
intact. Before this fix the gateway swallowed the failure at debug level and
the in-memory session advanced while disk silently fell behind — surfacing
later as "Persisted transcript lagged live cached history" amnesia.

The fix records a durable stale marker, detaches the FTS sync triggers, and
retries the canonical write immediately. Live search degrades to canonical
``LIKE`` queries. The existing guarded stale-open or explicit repair path may
rebuild later, outside the failed live write/search operation.
"""

import json
import os
import sqlite3

import pytest

import hermes_state
import hermes_state_holders
import hermes_state_schema
from hermes_state import SessionDB
from hermes_state_common import FTS_REBUILD_DEFERRAL_KEY, FTS_STALE_KEY, LEGACY_FTS_SQL, LEGACY_FTS_TRIGRAM_SQL, SCHEMA_SQL, _FTS_TRIGGERS
from hermes_state_dbfile import _concrete_state_db_holder_pids, _is_inactive_orphan_desktop_holder


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    try:
        d.close()
    except Exception:
        pass


def _corrupt_fts(db_path):
    raw = sqlite3.connect(str(db_path))
    raw.execute(
        "UPDATE messages_fts_data SET block = X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'"
    )
    raw.commit()
    raw.close()


def _corrupt_trigram_fts(db_path):
    raw = sqlite3.connect(str(db_path))
    raw.execute(
        "UPDATE messages_fts_trigram_data "
        "SET block = X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'"
    )
    raw.commit()
    raw.close()


def _message_contents(db_path):
    raw = sqlite3.connect(str(db_path))
    rows = raw.execute("SELECT content FROM messages ORDER BY id").fetchall()
    raw.close()
    return [r[0] for r in rows]


def _meta_value(db_path, key):
    raw = sqlite3.connect(str(db_path))
    row = raw.execute(
        "SELECT value FROM state_meta WHERE key = ?", (key,)
    ).fetchone()
    raw.close()
    return None if row is None else row[0]


def _base_fts_triggers(db_path):
    raw = sqlite3.connect(str(db_path))
    rows = raw.execute(
        "SELECT name FROM sqlite_master WHERE type = 'trigger' "
        f"AND name IN ({','.join('?' for _ in _FTS_TRIGGERS)})",
        _FTS_TRIGGERS,
    ).fetchall()
    raw.close()
    return {row[0] for row in rows}


class TestRuntimeFtsRebuild:
    def test_reap_candidates_exclude_uninspectable_holder_suspicions(
        self, tmp_path
    ):
        db_path = tmp_path / "state.db"

        assert _concrete_state_db_holder_pids(
            db_path,
            [
                (222, "uninspectable holder: python -m hermes_cli.main serve --port 0"),
                (-1, "open-file scan failed"),
            ],
        ) == []

    def test_reap_candidates_deduplicate_multiple_proven_watched_fds(self, tmp_path):
        db_path = tmp_path / "state.db"

        assert _concrete_state_db_holder_pids(
            db_path,
            [
                (222, str(db_path)),
                (222, f"{db_path}-wal"),
                (222, f"{db_path}-shm (deleted)"),
            ],
        ) == [222]

    def test_inactive_orphan_reap_predicate_preserves_live_or_ambiguous_holders(self):
        common = {
            "ppid": 1,
            "age_seconds": 120.0,
            "min_age_seconds": 60.0,
            "ephemeral_backend": True,
            "connection_statuses": [],
        }
        assert _is_inactive_orphan_desktop_holder(**common)
        assert not _is_inactive_orphan_desktop_holder(**{**common, "ppid": 42})
        assert not _is_inactive_orphan_desktop_holder(
            **{**common, "age_seconds": 10.0}
        )
        assert not _is_inactive_orphan_desktop_holder(
            **{**common, "ephemeral_backend": False}
        )
        assert not _is_inactive_orphan_desktop_holder(
            **{
                **common,
                "connection_statuses": ["ESTABLISHED"],
            }
        )

    @pytest.mark.parametrize(
        "argv",
        (
            ("journalctl", "-u", "hermes-agent.service"),
            ("grep", "hermes-agent", "/var/log/syslog"),
            (
                "/usr/sbin/tailscaled",
                "be-child",
                "ssh",
                "--cmd=python -m hermes_cli.main gateway",
            ),
            ("tmux", "new-session", "/opt/hermes-agent/.venv/bin/hermes gateway"),
            ("python3", "/opt/hermes-agent/tools/check_state.py"),
            ("hermes-monitor", "gateway"),
            ("hermesctl", "serve"),
            ("python3", "worker.py", "hermes_cli.main"),
            ("python3", "-m", "other.module", "hermes_cli.main"),
            ("python3", "-c", "hermes_cli.main"),
            ("python3", "-Icprint('hermes_cli.main')", "hermes_cli/main.py"),
        ),
    )
    def test_uninspectable_non_hermes_process_is_not_a_holder(self, argv):
        assert not hermes_state_holders._looks_like_hermes(argv)

    @pytest.mark.parametrize(
        "argv",
        (
            ("/usr/local/bin/hermes", "gateway"),
            ("/usr/local/bin/hermes-agent", "serve"),
            ("/usr/local/bin/hermes-acp", "--stdio"),
            ("/usr/bin/python3", "-m", "hermes_cli.main", "gateway"),
            ("/usr/bin/python3", "-m", "acp_adapter"),
            ("/usr/bin/python3", "-Im", "hermes_cli.main", "gateway"),
            ("/usr/bin/python3", "-mhermes_cli.main", "gateway"),
            ("/usr/bin/python3", "-W", "ignore", "-m", "hermes_cli.main"),
            ("/usr/bin/python3", "-Xdev", "-m", "hermes_cli.main"),
            (
                "/opt/hermes-agent/.venv/bin/python",
                "/opt/hermes-agent/hermes_cli/main.py",
                "gateway",
            ),
            ("python.exe", "--", "hermes_cli/main.py", "gateway"),
            ("python3", "/opt/hermes-agent/run_agent.py", "--query", "hello"),
        ),
    )
    def test_uninspectable_hermes_process_remains_a_holder(self, argv):
        assert hermes_state_holders._looks_like_hermes(argv)

    @pytest.mark.linux_only
    def test_foreign_holder_detection_proc_readlink_deleted_wal(
        self, db, tmp_path, monkeypatch
    ):
        """Linux /proc/<pid>/fd readlinks preserve '(deleted)' suffix.

        psutil.open_files() drops these entries (isfile_strict stats the
        literal path and fails).  The /proc path catches the split-brain
        holder that psutil silently misses.
        """
        db_path = tmp_path / "state.db"
        db_path_wal = str(db_path) + "-wal"

        # Build a fake /proc with two PIDs: self (111) and foreign (222).
        proc_root = tmp_path / "proc"
        for pid in (111, 222, 333):
            fd_dir = proc_root / str(pid) / "fd"
            fd_dir.mkdir(parents=True)
        # PID 222 holds the deleted WAL sidecar
        os.symlink(db_path_wal + " (deleted)", str(proc_root / "222" / "fd" / "3"))
        # PID 111 (self) holds the db — should be excluded
        os.symlink(str(db_path), str(proc_root / "111" / "fd" / "3"))
        # PID 333 holds an unrelated file
        other = tmp_path / "other.db"
        other.touch()
        os.symlink(str(other), str(proc_root / "333" / "fd" / "3"))

        monkeypatch.setattr(hermes_state_holders.os, "getpid", lambda: 111)
        real_listdir = os.listdir
        def _listdir(path):
            if isinstance(path, str):
                path = path.replace("/proc", str(proc_root))
            return real_listdir(path)
        monkeypatch.setattr(hermes_state_holders.os, "listdir", _listdir)
        real_readlink = os.readlink
        def _readlink(path):
            path = path.replace("/proc", str(proc_root))
            return real_readlink(path)
        monkeypatch.setattr(hermes_state_holders.os, "readlink", _readlink)
        real_stat = os.stat
        def _stat(path, *args, **kwargs):
            path_s = str(path).replace("/proc", str(proc_root))
            if path_s.endswith("/222/fd/3"):
                # A real /proc fd remains statable after unlink and retains
                # the deleted sidecar's filesystem identity: same device as
                # state.db, but an inode no live watched path can reach.
                fields = list(real_stat(db_path))
                fields[1] += 1000
                return os.stat_result(fields)
            return real_stat(path_s, *args, **kwargs)
        monkeypatch.setattr(hermes_state_holders.os, "stat", _stat)

        holders = hermes_state_holders.foreign_state_db_holders(db_path)
        assert holders == [(222, db_path_wal + " (deleted)")]

    @pytest.mark.linux_only
    @pytest.mark.parametrize("different_device", (False, True))
    def test_foreign_holder_ignores_same_path_with_different_file_identity(
        self, db, tmp_path, monkeypatch, different_device
    ):
        """A namespace peer's different state.db is not a holder of the host's.

        A peer process can appear in /proc with a string-identical path for a
        different inode, either on the same filesystem or a different one.
        Matching on path alone -- or on device alone -- would defer automatic
        FTS maintenance forever while corruption compounds.

        Identity must come from (st_dev, st_ino), not the path text.
        """
        db_path = tmp_path / "state.db"

        proc_root = tmp_path / "proc"
        for pid in (111, 222):
            (proc_root / str(pid) / "fd").mkdir(parents=True)
        # PID 222 = container process holding ITS OWN state.db, which happens
        # to have the identical absolute path inside its mount namespace.
        guest_db = tmp_path / "guest_state.db"
        guest_db.touch()
        os.symlink(str(guest_db), str(proc_root / "222" / "fd" / "3"))

        monkeypatch.setattr(hermes_state_holders.os, "getpid", lambda: 111)

        real_listdir = os.listdir
        def _listdir(path):
            if isinstance(path, str):
                path = path.replace("/proc", str(proc_root))
            return real_listdir(path)
        monkeypatch.setattr(hermes_state_holders.os, "listdir", _listdir)

        # The guest fd reports the host's path (identical string), which is
        # exactly what the kernel shows across mount namespaces.
        def _readlink(path):
            path = path.replace("/proc", str(proc_root))
            if path.endswith(f"{proc_root}/222/fd/3") or "222" in path:
                return str(db_path)
            return os.readlink(path)
        monkeypatch.setattr(hermes_state_holders.os, "readlink", _readlink)

        # ...but stat()ing the descriptor resolves to the peer's own inode.
        real_stat = os.stat
        def _stat(path, *a, **kw):
            path_s = str(path).replace("/proc", str(proc_root))
            st = real_stat(path_s, *a, **kw)
            if different_device and path_s.endswith("/222/fd/3"):
                fields = list(st)
                # os.stat_result positional layout: st_dev is index 2.
                fields[2] = st.st_dev + 1000
                return os.stat_result(fields)
            return st
        monkeypatch.setattr(hermes_state_holders.os, "stat", _stat)

        assert hermes_state_holders.foreign_state_db_holders(db_path) == []

    @pytest.mark.linux_only
    def test_foreign_holder_uninspectable_process_cmdline_fallback(
        self, db, tmp_path, monkeypatch
    ):
        """A process whose fd table is unreadable (different user) is still
        flagged when /proc/<pid>/cmdline identifies it as a Hermes process."""
        db_path = tmp_path / "state.db"

        proc_root = tmp_path / "proc"
        for pid in (111, 222):
            (proc_root / str(pid) / "fd").mkdir(parents=True)
        # PID 222's fd dir is unreadable (PermissionError)
        os.chmod(proc_root / "222" / "fd", 0o000)
        # PID 222's cmdline is world-readable and looks like Hermes
        cmdline_path = proc_root / "222" / "cmdline"
        cmdline_path.write_bytes(
            b"python3\x00-m\x00hermes_cli.main\x00chat\x00"
        )

        monkeypatch.setattr(hermes_state_holders.os, "getpid", lambda: 111)
        real_listdir = os.listdir
        def _listdir(path):
            if isinstance(path, str):
                if path == "/proc/222/fd":
                    raise PermissionError(path)
                path = path.replace("/proc", str(proc_root))
            return real_listdir(path)
        monkeypatch.setattr(hermes_state_holders.os, "listdir", _listdir)
        # _read_proc_argv opens /proc/<pid>/cmdline directly; redirect
        # it to our fake proc tree.
        def _fake_argv(pid):
            fake_path = str(proc_root / str(pid) / "cmdline")
            try:
                with open(fake_path, "rb") as f:
                    raw = f.read()
                if not raw:
                    return None
                return raw.decode("utf-8", "replace").rstrip("\x00").split("\x00")
            except OSError:
                return None
        monkeypatch.setattr(hermes_state_holders, "_read_proc_argv", _fake_argv)

        holders = hermes_state_holders.foreign_state_db_holders(db_path)
        # Should include PID 222 with the cmdline info
        assert len(holders) == 1
        assert holders[0][0] == 222
        assert "hermes_cli.main" in holders[0][1]

        # Cleanup
        os.chmod(proc_root / "222" / "fd", 0o755)

    def test_corruption_error_classification_requires_fts_evidence(self):
        """Generic structural corruption must not enter live FTS repair.

        Older SQLite builds may use the generic malformed-image text for an FTS
        virtual-table failure, but still expose SQLITE_CORRUPT_VTAB.  Preserve
        that route while failing closed for unscoped SQLITE_CORRUPT errors.
        """
        generic = sqlite3.DatabaseError("database disk image is malformed")
        assert not SessionDB._is_fts_write_corruption_error(generic)

        structural = sqlite3.DatabaseError("database disk image is malformed")
        structural.sqlite_errorcode = sqlite3.SQLITE_CORRUPT
        structural.sqlite_errorname = "SQLITE_CORRUPT"
        assert not SessionDB._is_fts_write_corruption_error(structural)

        fts_virtual_table = sqlite3.DatabaseError("database disk image is malformed")
        fts_virtual_table.sqlite_errorcode = sqlite3.SQLITE_CORRUPT_VTAB
        fts_virtual_table.sqlite_errorname = "SQLITE_CORRUPT_VTAB"
        assert SessionDB._is_fts_write_corruption_error(fts_virtual_table)

        contradictory = sqlite3.IntegrityError(
            'fts5: corrupt structure record for table "messages_fts"'
        )
        contradictory.sqlite_errorcode = sqlite3.SQLITE_CONSTRAINT_TRIGGER
        contradictory.sqlite_errorname = "SQLITE_CONSTRAINT_TRIGGER"
        assert not SessionDB._is_fts_write_corruption_error(contradictory)

        assert SessionDB._is_fts_write_corruption_error(
            sqlite3.DatabaseError(
                'fts5: corrupt structure record for table "messages_fts"'
            )
        )
        assert not SessionDB._is_fts_write_corruption_error(
            sqlite3.DatabaseError("no such table: nothing_fts_related")
        )

    def test_structural_corruption_propagates_without_live_fts_mutation(
        self, db, tmp_path, monkeypatch
    ):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")

        rebuild_called = False

        def _unexpected_rebuild():
            nonlocal rebuild_called
            rebuild_called = True
            raise AssertionError("structural corruption must not rebuild FTS")

        monkeypatch.setattr(db, "rebuild_fts", _unexpected_rebuild)
        structural = sqlite3.DatabaseError("database disk image is malformed")
        structural.sqlite_errorcode = sqlite3.SQLITE_CORRUPT
        structural.sqlite_errorname = "SQLITE_CORRUPT"

        with pytest.raises(sqlite3.DatabaseError) as caught:
            db._execute_write(lambda _conn: (_ for _ in ()).throw(structural))

        # Structural corruption quarantines the handle: the typed error wraps
        # the original (cause preserved, SQLite result code copied) and the
        # sticky flag is set, so later writes fail fast.
        from hermes_state import StateDbCorruptError

        assert isinstance(caught.value, StateDbCorruptError)
        assert caught.value.__cause__ is structural
        assert caught.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT
        assert db._db_corrupt is True
        assert rebuild_called is False
        assert db._fts_stale is False
        assert _meta_value(tmp_path / "state.db", FTS_STALE_KEY) is None
        assert _base_fts_triggers(tmp_path / "state.db") == set(_FTS_TRIGGERS)

    def test_fts_looking_constraint_error_does_not_mutate_fts(
        self, db, tmp_path, monkeypatch
    ):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")

        rebuild_called = False

        def _unexpected_rebuild():
            nonlocal rebuild_called
            rebuild_called = True
            raise AssertionError("contradictory error code must fail closed")

        monkeypatch.setattr(db, "rebuild_fts", _unexpected_rebuild)
        contradictory = sqlite3.IntegrityError(
            'fts5: corrupt structure record for table "messages_fts"'
        )
        contradictory.sqlite_errorcode = sqlite3.SQLITE_CONSTRAINT_TRIGGER
        contradictory.sqlite_errorname = "SQLITE_CONSTRAINT_TRIGGER"

        with pytest.raises(sqlite3.IntegrityError) as caught:
            db._execute_write(lambda _conn: (_ for _ in ()).throw(contradictory))

        assert caught.value is contradictory
        assert rebuild_called is False
        assert db._fts_stale is False
        assert _meta_value(tmp_path / "state.db", FTS_STALE_KEY) is None
        assert _base_fts_triggers(tmp_path / "state.db") == set(_FTS_TRIGGERS)

    def test_append_defers_rebuild_after_fts_corruption(
        self, db, tmp_path, monkeypatch
    ):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "hello world")

        _corrupt_fts(db_path)
        monkeypatch.setattr(
            db,
            "rebuild_fts",
            lambda: pytest.fail("live write must not rebuild the full FTS index"),
        )

        # The canonical write survives without waiting for a full index scan.
        msg_id = db.append_message("s1", "user", "healed append")
        assert msg_id is not None
        assert _message_contents(db_path) == [
            "hello world",
            "healed append",
        ]
        assert db._fts_stale is True
        assert _meta_value(db_path, FTS_STALE_KEY) == "1"
        assert _base_fts_triggers(db_path) == set()

    def test_search_works_from_canonical_rows_after_fail_open(self, db, tmp_path):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "before corruption")
        _corrupt_fts(db_path)
        db.append_message("s1", "user", "searchable needle text")

        results = db.search_messages("needle")
        assert results
        assert any("needle" in (row.get("snippet") or "") for row in results)
        assert db._fts_stale is True

    def test_search_messages_defers_rebuild_after_fts_corruption(
        self, db, tmp_path, monkeypatch
    ):
        """A read-only session that only SEARCHES (no write after corruption)
        must stay available without starting an unbounded index scan.
        """
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "a searchable needle here")

        _corrupt_fts(db_path)
        monkeypatch.setattr(
            db,
            "rebuild_fts",
            lambda: pytest.fail("live search must not rebuild the full FTS index"),
        )

        results = db.search_messages("needle")

        assert db._fts_stale is True
        assert _meta_value(db_path, FTS_STALE_KEY) == "1"
        assert _base_fts_triggers(db_path) == set()
        assert results
        assert any("needle" in (r.get("snippet") or "") for r in results)

    def test_trigram_search_defers_rebuild_after_fts_corruption(
        self, db, tmp_path, monkeypatch
    ):
        """The CJK/trigram MATCH branch has the same read-corruption exposure
        as the main FTS5 branch and must fall back to canonical rows.
        """
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        if not db._trigram_available:
            pytest.skip("trigram tokenizer unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "关于大别山项目的进展报告")

        _corrupt_trigram_fts(db_path)
        monkeypatch.setattr(
            db,
            "rebuild_fts",
            lambda: pytest.fail("live search must not rebuild the full FTS index"),
        )

        # >=3 CJK chars per token → routed to the trigram branch.
        results = db.search_messages("大别山项目")

        assert db._fts_stale is True
        assert _meta_value(db_path, FTS_STALE_KEY) == "1"
        assert _base_fts_triggers(db_path) == set()
        assert results
        assert any("大别山项目" in (r.get("snippet") or "") for r in results)

    def test_corruption_fails_open_and_rebuilds_on_reopen(self, db, tmp_path):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "seed")
        _corrupt_fts(db_path)
        db.append_message("s1", "user", "corruption survives")
        assert _message_contents(db_path) == [
            "seed",
            "corruption survives",
        ]
        assert db._fts_stale is True
        assert _meta_value(db_path, FTS_STALE_KEY) == "1"
        assert _base_fts_triggers(db_path) == set()

        # Search remains available from canonical rows while FTS is stale.
        results = db.search_messages("corruption survives")
        assert results
        assert any("corruption survives" in row["snippet"] for row in results)

        # A later open atomically rebuilds all canonical rows before triggers
        # return, then clears the durable breadcrumb.
        db.close()
        reopened = SessionDB(db_path=db_path)
        try:
            assert reopened._fts_stale is False
            assert _meta_value(db_path, FTS_STALE_KEY) is None
            assert _base_fts_triggers(db_path) == set(_FTS_TRIGGERS)
            results = reopened.search_messages("corruption survives")
            assert results
        finally:
            reopened.close()

    def test_non_fts_write_error_after_fail_open_raises_not_hangs(
        self, db, tmp_path
    ):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "seed")
        _corrupt_fts(db_path)
        db.append_message("s1", "user", "canonical survives")

        def _persistent_non_fts_error(conn):
            raise sqlite3.DatabaseError("routine integrity check failed")

        with pytest.raises(sqlite3.DatabaseError, match="routine integrity"):
            db._execute_write(_persistent_non_fts_error)

    def test_live_write_does_not_scan_foreign_holders(
        self, db, tmp_path, monkeypatch
    ):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "seed")
        _corrupt_fts(db_path)

        monkeypatch.setattr(
            db,
            "_foreign_state_db_holders",
            lambda: pytest.fail("live fail-open must not enter rebuild admission"),
            raising=False,
        )

        db.append_message("s1", "user", "canonical survives foreign holder")

        assert _message_contents(db_path)[-1] == "canonical survives foreign holder"
        assert db._fts_stale is True
        assert _meta_value(db_path, FTS_STALE_KEY) == "1"
        assert _base_fts_triggers(db_path) == set()

    def test_stale_search_preserves_not_semantics(self, db, tmp_path, monkeypatch):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "python language guide")
        db.append_message("s1", "user", "python java interoperability")
        _corrupt_fts(db_path)

        monkeypatch.setattr(
            db,
            "rebuild_fts",
            lambda: (_ for _ in ()).throw(
                sqlite3.DatabaseError("rebuild could not read corrupt FTS")
            ),
        )
        db.append_message("s1", "user", "canonical write survives")
        assert db._fts_stale is True

        results = db.search_messages("python NOT java")
        snippets = [row["snippet"] for row in results]
        assert any("python language guide" in snippet for snippet in snippets)
        assert all("java" not in snippet for snippet in snippets)

    def test_existing_peer_observes_fail_open_marker(
        self, db, tmp_path, monkeypatch
    ):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "seed")
        peer = SessionDB(db_path=db_path)
        try:
            _corrupt_fts(db_path)

            def _failed_rebuild():
                raise sqlite3.DatabaseError("rebuild failed")

            monkeypatch.setattr(db, "rebuild_fts", _failed_rebuild)
            db.append_message("s1", "user", "visible through canonical search")

            assert peer._fts_stale is False
            results = peer.search_messages("canonical search")
            assert peer._fts_stale is True
            assert results
        finally:
            peer.close()

    def test_failed_startup_rebuild_keeps_fts_detached(
        self, db, tmp_path, monkeypatch
    ):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "seed")
        _corrupt_fts(db_path)
        monkeypatch.setattr(
            db,
            "rebuild_fts",
            lambda: (_ for _ in ()).throw(sqlite3.DatabaseError("still corrupt")),
        )
        db.append_message("s1", "user", "before restart")
        db.close()

        monkeypatch.setattr(
            SessionDB,
            "_recover_stale_fts",
            lambda self, cursor, legacy: False,
        )
        reopened = SessionDB(db_path=db_path)
        try:
            assert reopened._fts_stale is True
            assert _meta_value(db_path, FTS_STALE_KEY) == "1"
            assert _base_fts_triggers(db_path) == set()
            reopened.append_message("s1", "user", "after failed recovery")
            assert _message_contents(db_path)[-1] == "after failed recovery"
            assert reopened.search_messages("failed recovery")
        finally:
            reopened.close()

    def test_foreign_holder_defers_startup_stale_rebuild(
        self, db, tmp_path, monkeypatch
    ):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "seed")
        _corrupt_fts(db_path)
        monkeypatch.setattr(
            db,
            "rebuild_fts",
            lambda: (_ for _ in ()).throw(sqlite3.DatabaseError("still corrupt")),
        )
        db.append_message("s1", "user", "before restart")
        db.close()

        monkeypatch.setattr(
            SessionDB,
            "_foreign_state_db_holders",
            lambda self: [(4242, str(db_path) + "-wal")],
            raising=False,
        )
        reopened = SessionDB(db_path=db_path)
        try:
            assert reopened._fts_stale is True
            assert _meta_value(db_path, FTS_STALE_KEY) == "1"
            assert _base_fts_triggers(db_path) == set()
            reopened.append_message("s1", "user", "after deferred recovery")
            assert _message_contents(db_path)[-1] == "after deferred recovery"
        finally:
            reopened.close()

    def test_repeated_deferrals_reap_inactive_orphan_then_rebuild(
        self, db, tmp_path, monkeypatch
    ):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "seed")
        _corrupt_fts(db_path)
        monkeypatch.setattr(
            db,
            "rebuild_fts",
            lambda: (_ for _ in ()).throw(sqlite3.DatabaseError("still corrupt")),
        )
        db.append_message("s1", "user", "before restart")
        db.close()

        raw = sqlite3.connect(str(db_path))
        raw.execute(
            "INSERT INTO state_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (
                FTS_REBUILD_DEFERRAL_KEY,
                json.dumps({"first_seen": 1.0, "last_seen": 30.0, "attempts": 2}),
            ),
        )
        raw.commit()
        raw.close()

        holder_scans = iter(([(4242, str(db_path) + "-wal")], []))
        reaped = []
        monkeypatch.setattr(
            SessionDB,
            "_foreign_state_db_holders",
            lambda self: next(holder_scans),
        )
        monkeypatch.setattr(
            SessionDB,
            "_reap_inactive_orphan_desktop_holders",
            lambda self, holders, *, min_age_seconds: reaped.extend(holders) or [4242],
        )
        monkeypatch.setattr(hermes_state_schema.time, "time", lambda: 120.0)

        reopened = SessionDB(db_path=db_path)
        try:
            assert reaped == [(4242, str(db_path) + "-wal")]
            assert reopened._fts_stale is False
            assert _meta_value(db_path, FTS_STALE_KEY) is None
            assert _meta_value(db_path, FTS_REBUILD_DEFERRAL_KEY) is None
            assert reopened.search_messages("before restart")
        finally:
            reopened.close()

    def test_legacy_inline_fts_fails_open_and_recovers(self, tmp_path, monkeypatch):
        db_path = tmp_path / "legacy-state.db"
        raw = sqlite3.connect(str(db_path))
        raw.executescript(SCHEMA_SQL)
        try:
            raw.executescript(LEGACY_FTS_SQL + LEGACY_FTS_TRIGRAM_SQL)
        except sqlite3.OperationalError as exc:
            raw.close()
            pytest.skip(f"required FTS tokenizer unavailable: {exc}")
        raw.commit()
        raw.close()

        legacy = SessionDB(db_path=db_path)
        try:
            assert legacy._db_has_legacy_inline_fts(legacy._conn.cursor())
            legacy.create_session("s1", source="test")
            legacy.append_message("s1", "user", "legacy seed")
            _corrupt_fts(db_path)
            monkeypatch.setattr(
                legacy,
                "rebuild_fts",
                lambda: (_ for _ in ()).throw(
                    sqlite3.DatabaseError("legacy rebuild failed")
                ),
            )
            legacy.append_message("s1", "user", "legacy canonical survives")
            assert _message_contents(db_path)[-1] == "legacy canonical survives"
            assert _meta_value(db_path, FTS_STALE_KEY) == "1"
        finally:
            legacy.close()

        recovered = SessionDB(db_path=db_path)
        try:
            assert recovered._fts_stale is False
            assert _meta_value(db_path, FTS_STALE_KEY) is None
            assert recovered.search_messages("canonical survives")
        finally:
            recovered.close()


def _corrupt_canonical_btree(db_path):
    """Physically damage every ``messages`` table B-tree leaf page.

    Real byte-flip corruption (no mocks): checkpoint the WAL so all pages are
    in the main file, locate the leaves via ``dbstat``, and clobber each leaf's
    page-header cell-count bytes. Any subsequent write or read touching the
    ``messages`` tree raises a genuine bare ``SQLITE_CORRUPT`` from SQLite.
    """
    raw = sqlite3.connect(str(db_path))
    try:
        raw.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        page_size = raw.execute("PRAGMA page_size").fetchone()[0]
        try:
            leaves = [
                row[0]
                for row in raw.execute(
                    "SELECT pageno FROM dbstat "
                    "WHERE name='messages' AND pagetype='leaf'"
                ).fetchall()
            ]
        except sqlite3.Error:
            pytest.skip("dbstat virtual table unavailable in this build")
    finally:
        raw.close()
    assert leaves, "expected at least one messages leaf page"
    with open(db_path, "r+b") as f:
        for leaf in leaves:
            f.seek((leaf - 1) * page_size + 3)
            f.write(b"\xff\xff\xff\xff")


class TestPhysicalCorruptionAcceptance:
    """Real-fixture acceptance tests for the fail-closed classifier (#97940).

    The field incident behind issue #97940: canonical B-trees were physically
    damaged, SQLite raised the generic ``database disk image is malformed``,
    and the over-broad classifier routed it into the FTS-only self-heal —
    logging "canonical message rows are preserved" while transcript writes
    silently failed for ~10 hours. These tests damage a real database with
    byte flips (canonical tree) and shadow-table stomps (FTS-only) and assert
    the two corruption classes are handled differently end to end.
    """

    def test_canonical_btree_corruption_fails_closed(
        self, tmp_path, caplog
    ):
        """Bare SQLITE_CORRUPT from real canonical damage must propagate.

        No FTS rebuild attempt, no trigger detach, no stale marker, and —
        critically — no log line claiming canonical rows are preserved.
        """
        db_path = tmp_path / "state.db"
        seed = SessionDB(db_path=db_path)
        try:
            if not seed._fts_enabled:
                pytest.skip("FTS5 unavailable in this build")
            seed.create_session("s1", source="test")
            for i in range(300):
                seed.append_message("s1", "user", f"canon row {i} " + "y" * 300)
        finally:
            seed.close()

        _corrupt_canonical_btree(db_path)

        db = SessionDB(db_path=db_path)
        try:
            caplog.clear()
            with caplog.at_level("WARNING", logger="hermes_state"):
                with pytest.raises(sqlite3.DatabaseError) as caught:
                    db.append_message("s1", "user", "post-corruption write")
            # The genuine structural error propagated, not an FTS retry result.
            assert getattr(caught.value, "sqlite_errorcode", None) in (
                sqlite3.SQLITE_CORRUPT,
                None,  # very old sqlite3 modules without errorcode attrs
            )
            assert "malformed" in str(caught.value).lower()
            # The classifier refused the FTS route entirely.
            assert not SessionDB._is_fts_write_corruption_error(caught.value)
            assert getattr(db, "_fts_runtime_rebuild_attempted", False) is False
            assert db._fts_stale is False
            # The misdiagnosis message from the field incident must be gone.
            assert "canonical message rows are preserved" not in caplog.text
            assert "attempting one-shot in-place FTS rebuild" not in caplog.text
            # Structural damage quarantines the handle: typed error, sticky
            # flag, later writes fail fast, and close() must not checkpoint
            # the WAL over a damaged page image (the #90950 page-1 clobber).
            from hermes_state import StateDbCorruptError

            assert isinstance(caught.value, StateDbCorruptError)
            assert db._db_corrupt is True
            with pytest.raises(StateDbCorruptError):
                db.append_message("s1", "user", "second write after corruption")
            caplog.clear()
            with caplog.at_level("WARNING", logger="hermes_state"):
                db.close()
            assert "Skipping the close-time WAL checkpoint" in caplog.text
        finally:
            db.close()

        # Fail-closed also means non-destructive: triggers untouched, no
        # stale-FTS marker persisted for a structural (non-FTS) failure.
        assert _base_fts_triggers(db_path) == set(_FTS_TRIGGERS)
        assert _meta_value(db_path, FTS_STALE_KEY) is None

    def test_fts_only_corruption_still_self_heals(self, db, tmp_path):
        """Contrast case: a real FTS shadow-table stomp raises
        SQLITE_CORRUPT_VTAB, is classified as FTS-scoped, and the write path
        self-heals with canonical rows intact — proving the narrowed
        classifier did not break the legitimate FTS repair route."""
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "before stomp")
        for i in range(50):
            db.append_message("s1", "user", f"seed row {i} " + "z" * 200)
        _corrupt_fts(tmp_path / "state.db")

        # Prove the fixture produces the FTS-scoped extended code for the
        # classifier (provenance check via a raw connection so no self-heal
        # runs). A MATCH read walks the stomped structure record on every
        # SQLite version; the insert trigger only trips on some versions.
        raw = sqlite3.connect(str(tmp_path / "state.db"))
        try:
            raw.execute(
                "SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'seed'"
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            assert getattr(exc, "sqlite_errorcode", None) == getattr(
                sqlite3, "SQLITE_CORRUPT_VTAB", 267
            )
            assert SessionDB._is_fts_write_corruption_error(exc)
        else:  # pragma: no cover - fixture must corrupt the index
            pytest.fail("FTS stomp fixture did not corrupt the index")
        finally:
            raw.close()

        # And the app-level write path still succeeds after real FTS-only
        # damage (self-heals on builds whose sync triggers surface the
        # corruption; passes through untouched on builds that defer it).
        msg_id = db.append_message("s1", "user", "healed after stomp")
        assert msg_id is not None
        contents = _message_contents(tmp_path / "state.db")
        assert contents[0] == "before stomp"
        assert contents[-1] == "healed after stomp"
        assert len(contents) == 52
