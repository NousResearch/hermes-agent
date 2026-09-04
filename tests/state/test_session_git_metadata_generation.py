"""Cross-process ordering for asynchronous session Git metadata probes."""

from __future__ import annotations

import sqlite3
import threading

from hermes_state import SCHEMA_VERSION, SessionDB


def _open_pair(tmp_path):
    path = tmp_path / "state.db"
    first = SessionDB(db_path=path)
    second = SessionDB(db_path=path)
    first.create_session("session", "desktop", cwd="/repo/A")
    return first, second


def _require_generation(value: int | None) -> int:
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def test_delayed_probe_cannot_overwrite_newer_a_b_a_claim(tmp_path):
    first, second = _open_pair(tmp_path)
    release_old = threading.Event()
    old_finished = threading.Event()
    old_result = []
    try:
        old_generation = _require_generation(
            first.update_session_cwd("session", "/repo/A")
        )

        def publish_old_probe():
            assert release_old.wait(5)
            old_result.append(
                first.publish_session_git_metadata(
                    "session",
                    "/repo/A",
                    old_generation,
                    "stale-branch",
                    "/repo/stale-root",
                )
            )
            old_finished.set()

        worker = threading.Thread(target=publish_old_probe)
        worker.start()

        second.update_session_cwd("session", "/repo/B")
        new_generation = _require_generation(
            second.update_session_cwd("session", "/repo/A")
        )
        assert new_generation > old_generation
        assert second.publish_session_git_metadata(
            "session",
            "/repo/A",
            new_generation,
            "new-branch",
            "/repo/new-root",
        )

        release_old.set()
        assert old_finished.wait(5)
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert old_result == [False]

        row = second.get_session("session")
        assert row is not None
        assert row["cwd"] == "/repo/A"
        assert row["git_branch"] == "new-branch"
        assert row["git_repo_root"] == "/repo/new-root"
    finally:
        release_old.set()
        first.close()
        second.close()


def test_repeated_same_cwd_claim_invalidates_older_probe(tmp_path):
    first, second = _open_pair(tmp_path)
    try:
        old_generation = _require_generation(
            first.update_session_cwd("session", "/repo/A")
        )
        new_generation = _require_generation(
            second.update_session_cwd("session", "/repo/A")
        )

        assert new_generation > old_generation
        assert second.publish_session_git_metadata(
            "session", "/repo/A", new_generation, "new", "/repo/A"
        )
        assert not first.publish_session_git_metadata(
            "session", "/repo/A", old_generation, "old", "/repo/old"
        )
        row = second.get_session("session")
        assert row is not None
        assert row["git_branch"] == "new"
    finally:
        first.close()
        second.close()


def test_cwd_move_clears_metadata_in_same_claim(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("session", "desktop", cwd="/repo/A")
        generation = _require_generation(
            db.update_session_cwd("session", "/repo/A")
        )
        assert db.publish_session_git_metadata(
            "session", "/repo/A", generation, "main", "/repo/A"
        )

        moved_generation = _require_generation(
            db.update_session_cwd("session", "/repo/B")
        )
        row = db.get_session("session")
        assert row is not None
        assert moved_generation > generation
        assert row["cwd"] == "/repo/B"
        assert row["git_branch"] is None
        assert row["git_repo_root"] is None
    finally:
        db.close()


def test_explicit_move_replaces_metadata_and_claims_generation(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("session", "desktop", cwd="/repo/A")
        initial_generation = _require_generation(
            db.update_session_cwd(
                "session",
                "/repo/A",
                git_branch="main",
                git_repo_root="/repo/A",
            )
        )

        moved_generation = _require_generation(
            db.update_session_cwd(
                "session",
                "/outside-git",
                replace_git_meta=True,
            )
        )

        row = db.get_session("session")
        assert row is not None
        assert moved_generation > initial_generation
        assert row["cwd"] == "/outside-git"
        assert row["git_branch"] is None
        assert row["git_repo_root"] is None
    finally:
        db.close()


def test_failed_new_probe_still_invalidates_older_worker(tmp_path):
    first, second = _open_pair(tmp_path)
    try:
        baseline = _require_generation(
            first.update_session_cwd("session", "/repo/A")
        )
        assert first.publish_session_git_metadata(
            "session", "/repo/A", baseline, "baseline", "/repo/A"
        )
        old_generation = _require_generation(
            first.update_session_cwd("session", "/repo/A")
        )
        second.update_session_cwd("session", "/repo/A")

        assert not first.publish_session_git_metadata(
            "session", "/repo/A", old_generation, "stale", "/repo/stale"
        )
        row = second.get_session("session")
        assert row is not None
        assert row["git_branch"] == "baseline"
        assert row["git_repo_root"] == "/repo/A"
    finally:
        first.close()
        second.close()


def test_generation_authority_is_scoped_to_each_profile_database(tmp_path):
    first = SessionDB(db_path=tmp_path / "profile-a.db")
    second = SessionDB(db_path=tmp_path / "profile-b.db")
    try:
        first.create_session("same-id", "desktop", cwd="/a")
        second.create_session("same-id", "desktop", cwd="/b")
        first_generation = _require_generation(
            first.update_session_cwd("same-id", "/a")
        )
        second_generation = _require_generation(
            second.update_session_cwd("same-id", "/b")
        )

        assert first.publish_session_git_metadata(
            "same-id", "/a", first_generation, "a", "/a"
        )
        assert second.publish_session_git_metadata(
            "same-id", "/b", second_generation, "b", "/b"
        )
        first_row = first.get_session("same-id")
        second_row = second.get_session("same-id")
        assert first_row is not None
        assert second_row is not None
        assert first_row["git_branch"] == "a"
        assert second_row["git_branch"] == "b"
    finally:
        first.close()
        second.close()


def test_legacy_sessions_table_reconciles_generation_column(tmp_path):
    path = tmp_path / "state.db"
    SessionDB(db_path=path).close()
    conn = sqlite3.connect(path)
    try:
        conn.execute("ALTER TABLE sessions DROP COLUMN git_metadata_generation")
        conn.execute("UPDATE schema_version SET version = 25")
        conn.commit()
    finally:
        conn.close()

    reopened = SessionDB(db_path=path)
    try:
        verify = sqlite3.connect(path)
        try:
            columns = {
                row[1]
                for row in verify.execute("PRAGMA table_info('sessions')")
            }
        finally:
            verify.close()
        assert "git_metadata_generation" in columns
        assert reopened._conn.execute(
            "SELECT version FROM schema_version"
        ).fetchone()[0] == SCHEMA_VERSION
        reopened.create_session("session", "desktop", cwd="/repo")
        assert reopened.update_session_cwd("session", "/repo") == 1
    finally:
        reopened.close()


def test_compact_session_rows_do_not_expose_internal_generation(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("session", "desktop", cwd="/repo")
        db.update_session_cwd("session", "/repo")

        rows = db.list_sessions_rich(compact_rows=True)
        assert len(rows) == 1
        assert "git_metadata_generation" not in rows[0]
    finally:
        db.close()
