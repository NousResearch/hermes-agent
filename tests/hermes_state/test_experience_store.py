"""Persistence tests for the Level 2 experience store (ExperienceStoreMixin).

Covers creation, dedup/merge, correction handling, relevance-independent
retrieval plumbing, pruning, stats, backward compatibility with a database
created before the ``experiences`` table existed, and read-only safety.
"""
import json
import sqlite3
import threading
import time

import pytest

from agent.experience import Experience, task_fingerprint
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _exp(task="fix the build", outcome="success", workspace="/proj", **kw):
    norm = kw.pop("task_norm", None) or task
    e = Experience(
        task=task,
        task_norm=norm,
        task_hash=task_fingerprint(norm),
        outcome=outcome,
        cwd=kw.pop("cwd", None) or workspace,
        workspace=workspace,
        **kw,
    )
    return e.to_row()


# ── Creation & persistence ──────────────────────────────────────────────


class TestCreation:
    def test_records_a_new_experience(self, db):
        rid = db.record_experience(_exp())
        assert rid
        row = db.get_experience(rid)
        assert row["task"] == "fix the build"
        assert row["outcome"] == "success"
        assert row["observations"] == 1
        assert row["success_count"] == 1
        assert row["failure_count"] == 0
        assert row["superseded"] == 0

    def test_rejects_incomplete_rows(self, db):
        assert db.record_experience({}) is None
        assert db.record_experience({"id": "x", "task": "", "task_hash": "h",
                                     "outcome": "success"}) is None
        assert db.experience_stats()["total"] == 0

    def test_survives_reopen(self, tmp_path):
        d1 = SessionDB(tmp_path / "state.db")
        rid = d1.record_experience(_exp())
        d1.close()
        d2 = SessionDB(tmp_path / "state.db")
        assert d2.get_experience(rid)["task"] == "fix the build"

    def test_confidence_is_laplace_smoothed(self, db):
        rid = db.record_experience(_exp())
        # one success -> (1+1)/(1+0+2) = 0.667, not 1.0
        assert db.get_experience(rid)["confidence"] == pytest.approx(0.6667, abs=1e-3)


# ── Dedup / merge ───────────────────────────────────────────────────────


class TestDeduplication:
    def test_same_task_same_cwd_merges(self, db):
        a = db.record_experience(_exp())
        b = db.record_experience(_exp())
        assert a == b
        row = db.get_experience(a)
        assert row["observations"] == 2
        assert row["success_count"] == 2
        assert db.experience_stats()["total"] == 1

    def test_same_task_different_workspace_is_a_separate_row(self, db):
        a = db.record_experience(_exp(workspace="/proj-a"))
        b = db.record_experience(_exp(workspace="/proj-b"))
        assert a != b
        assert db.experience_stats()["total"] == 2

    def test_subdirectories_of_one_project_share_a_row(self, db):
        """The whole point of scoping on workspace rather than raw cwd."""
        a = db.record_experience(_exp(workspace="/proj", cwd="/proj"))
        b = db.record_experience(_exp(workspace="/proj", cwd="/proj/src/api"))
        assert a == b
        assert db.get_experience(a)["observations"] == 2

    def test_workspace_falls_back_to_cwd_when_unset(self, db):
        """Outside a project, cwd is still the scoping key (pre-P2 behaviour)."""
        row = _exp(workspace="", cwd="/loose/dir")
        row["workspace"] = ""
        rid = db.record_experience(row)
        assert db.get_experience(rid)["workspace"] == "/loose/dir"

    def test_merge_tracks_mixed_outcomes(self, db):
        rid = db.record_experience(_exp(outcome="success"))
        db.record_experience(_exp(outcome="failure"))
        db.record_experience(_exp(outcome="failure"))
        row = db.get_experience(rid)
        assert (row["success_count"], row["failure_count"]) == (1, 2)
        assert row["observations"] == 3
        # 2 of 3 failed -> confidence below the 0.5 prior
        assert row["confidence"] < 0.5

    def test_recovered_partial_counts_as_success(self, db):
        rid = db.record_experience(
            _exp(outcome="partial", recovery="retried after failure and succeeded: patch")
        )
        row = db.get_experience(rid)
        assert row["success_count"] == 1 and row["failure_count"] == 0

    def test_unrecovered_partial_counts_as_failure(self, db):
        rid = db.record_experience(_exp(outcome="partial"))
        row = db.get_experience(rid)
        assert row["success_count"] == 0 and row["failure_count"] == 1


# ── Verification evidence (P1) ──────────────────────────────────────────


class TestVerificationAccounting:
    def test_passing_evidence_redeems_an_unrecovered_partial(self, db):
        # Tool errors happened, but the build/tests passed afterwards — the
        # agent reached a working state even without an explicit retry.
        rid = db.record_experience(_exp(outcome="partial", verification="passed"))
        row = db.get_experience(rid)
        assert (row["success_count"], row["failure_count"]) == (1, 0)
        assert row["verification"] == "passed"

    def test_passing_evidence_does_not_rescue_an_outright_failure(self, db):
        # The turn did not complete; stale evidence from earlier in the
        # session is not proof that this attempt worked.
        rid = db.record_experience(_exp(outcome="failure", verification="passed"))
        row = db.get_experience(rid)
        assert (row["success_count"], row["failure_count"]) == (0, 1)

    def test_stale_evidence_does_not_redeem_a_partial(self, db):
        rid = db.record_experience(_exp(outcome="partial", verification="stale"))
        row = db.get_experience(rid)
        assert (row["success_count"], row["failure_count"]) == (0, 1)

    def test_absent_evidence_preserves_pre_feature_accounting(self, db):
        rid = db.record_experience(_exp(outcome="success", verification=""))
        row = db.get_experience(rid)
        assert (row["success_count"], row["failure_count"]) == (1, 0)
        assert row["verification"] == ""

    def test_merge_refreshes_the_verification_verdict(self, db):
        rid = db.record_experience(_exp(outcome="success", verification="stale"))
        db.record_experience(_exp(outcome="success", verification="passed"))
        assert db.get_experience(rid)["verification"] == "passed"

    def test_stats_break_out_verification(self, db):
        db.record_experience(_exp(task="a task", verification="passed"))
        db.record_experience(_exp(task="b task", outcome="failure", verification="failed"))
        db.record_experience(_exp(task="c task", verification=""))
        s = db.experience_stats()
        assert (s["verified_pass"], s["verified_fail"], s["unverified"]) == (1, 1, 1)

    def test_re_attempt_revives_a_superseded_row(self, db):
        rid = db.record_experience(_exp(outcome="success"))
        db.record_experience_correction(rid, "wrong file")
        assert db.get_experience(rid)["superseded"] == 1
        db.record_experience(_exp(outcome="success"))
        assert db.get_experience(rid)["superseded"] == 0


# ── User correction ─────────────────────────────────────────────────────


class TestCorrection:
    def test_correction_lowers_confidence(self, db):
        rid = db.record_experience(_exp())
        before = db.get_experience(rid)["confidence"]
        assert db.record_experience_correction(rid, "that's wrong")
        after = db.get_experience(rid)
        assert after["confidence"] < before
        assert after["correction_count"] == 1
        assert after["user_correction"] == "that's wrong"

    def test_corrected_success_is_superseded(self, db):
        rid = db.record_experience(_exp(outcome="success"))
        db.record_experience_correction(rid, "no, wrong approach")
        assert db.get_experience(rid)["superseded"] == 1

    def test_corrected_failure_stays_live(self, db):
        # "this path fails" remains true even after the user corrects the fix.
        rid = db.record_experience(_exp(outcome="failure"))
        db.record_experience_correction(rid, "still broken")
        assert db.get_experience(rid)["superseded"] == 0

    def test_unknown_id_is_a_no_op(self, db):
        assert db.record_experience_correction("nope", "x") is False

    def test_correction_text_is_bounded(self, db):
        rid = db.record_experience(_exp())
        db.record_experience_correction(rid, "x" * 5000)
        assert len(db.get_experience(rid)["user_correction"]) == 240

    def test_latest_for_session_returns_newest(self, db):
        db.record_experience(_exp(task="first task", session_id="s1"))
        time.sleep(0.01)
        b = db.record_experience(_exp(task="second task", session_id="s1"))
        db.record_experience(_exp(task="other session", session_id="s2"))
        assert db.latest_experience_for_session("s1")["id"] == b
        assert db.latest_experience_for_session("") is None
        assert db.latest_experience_for_session("missing") is None


# ── Candidate retrieval ─────────────────────────────────────────────────


class TestCandidates:
    def test_returns_live_rows_only(self, db):
        keep = db.record_experience(_exp(task="alpha task"))
        drop = db.record_experience(_exp(task="beta task", outcome="success"))
        db.record_experience_correction(drop, "wrong")
        ids = {r["id"] for r in db.fetch_experience_candidates()}
        assert keep in ids and drop not in ids

    def test_respects_max_age(self, db):
        rid = db.record_experience(_exp())
        old = time.time() - 200 * 86400

        def _age(conn):
            conn.execute("UPDATE experiences SET updated_at = ?", (old,))

        db._execute_write(_age)
        assert db.fetch_experience_candidates(max_age_days=90) == []
        assert len(db.fetch_experience_candidates(max_age_days=365)) == 1
        assert db.get_experience(rid) is not None

    def test_same_workspace_rows_sort_first(self, db):
        db.record_experience(_exp(task="elsewhere", workspace="/other"))
        here = db.record_experience(_exp(task="right here", workspace="/proj"))
        rows = db.fetch_experience_candidates(workspace="/proj")
        assert rows[0]["id"] == here

    def test_other_workspaces_are_still_returned(self, db):
        """Cross-project knowledge about a tool or error class stays useful,
        so workspace is a sort key, never a WHERE clause."""
        db.record_experience(_exp(task="elsewhere", workspace="/other"))
        assert len(db.fetch_experience_candidates(workspace="/proj")) == 1

    def test_missing_experience_returns_none_not_raise(self, db):
        assert db.get_experience("does-not-exist") is None
        assert db.get_experience("") is None
        assert db.fetch_experience_candidates() == []


# ── Stats / export / prune ──────────────────────────────────────────────


class TestMaintenance:
    def test_stats_counts_outcomes(self, db):
        db.record_experience(_exp(task="a task", outcome="success"))
        db.record_experience(_exp(task="b task", outcome="failure"))
        db.record_experience(_exp(task="c task", outcome="interrupted"))
        db.record_experience(_exp(task="d task", outcome="partial", recovery="retried"))
        s = db.experience_stats()
        assert s["total"] == 4
        assert (s["success"], s["failure"], s["interrupted"], s["partial"]) == (1, 1, 1, 1)
        assert s["recovered"] == 1
        assert 0.0 < s["avg_confidence"] <= 1.0

    def test_stats_on_empty_store(self, db):
        assert db.experience_stats()["total"] == 0

    def test_prune_drops_expired_rows(self, db):
        db.record_experience(_exp())

        def _age(conn):
            conn.execute(
                "UPDATE experiences SET updated_at = ?", (time.time() - 400 * 86400,)
            )

        db._execute_write(_age)
        assert db.prune_experiences(max_age_days=365) == 1
        assert db.experience_stats()["total"] == 0

    def test_prune_enforces_row_cap_keeping_best_evidence(self, db):
        # One well-evidenced row plus filler; cap of 1 must keep the evidenced one.
        keep = db.record_experience(_exp(task="well known task"))
        db.record_experience(_exp(task="well known task"))
        db.record_experience(_exp(task="well known task"))
        for i in range(5):
            db.record_experience(_exp(task=f"one off task number {i}"))
        assert db.prune_experiences(max_rows=1) == 5
        rows = db.export_experiences()
        assert len(rows) == 1 and rows[0]["id"] == keep

    def test_export_decodes_json_columns(self, db):
        row = _exp()
        row["tools"] = json.dumps(["read_file", "patch"])
        row["metrics"] = json.dumps({"api_calls": 3})
        db.record_experience(row)
        out = db.export_experiences()[0]
        assert out["tools"] == ["read_file", "patch"]
        assert out["metrics"]["api_calls"] == 3

    def test_export_tolerates_corrupt_json(self, db):
        db.record_experience(_exp())

        def _corrupt(conn):
            conn.execute("UPDATE experiences SET tools = 'not json'")

        db._execute_write(_corrupt)
        assert db.export_experiences()[0]["tools"] == []

    def test_clear_removes_everything(self, db):
        db.record_experience(_exp(task="a task"))
        db.record_experience(_exp(task="b task"))
        assert db.clear_experiences() == 2
        assert db.experience_stats()["total"] == 0


# ── Concurrency ─────────────────────────────────────────────────────────


class TestConcurrentWriters:
    """Several Hermes processes share one state.db.

    ``record_experience`` reads then writes, so without a transaction that
    takes the write lock up front, two writers racing on the same task would
    both see "no existing row" and insert duplicates — permanently splitting a
    task's history. Each thread below gets its OWN ``SessionDB`` (its own
    connection), which is what makes this a stand-in for separate processes
    rather than a test of one shared lock.
    """

    def _race(self, tmp_path, rows, workers=8):
        SessionDB(tmp_path / "state.db").close()  # create the schema once
        errors = []
        barrier = threading.Barrier(workers)

        def _worker(i):
            db = SessionDB(tmp_path / "state.db")
            try:
                barrier.wait(timeout=30)
                db.record_experience(rows(i))
            except Exception as exc:  # noqa: BLE001 - surfaced by the assert
                errors.append(exc)
            finally:
                db.close()

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert not errors, f"concurrent writers raised: {errors[:3]}"
        return SessionDB(tmp_path / "state.db")

    def test_the_same_task_never_splits_into_duplicate_rows(self, tmp_path):
        workers = 8
        db = self._race(tmp_path, lambda i: _exp(outcome="success"), workers=workers)
        try:
            rows = db.export_experiences()
            assert len(rows) == 1, "the dedup read-then-write raced"
            assert rows[0]["observations"] == workers
            assert rows[0]["success_count"] == workers
        finally:
            db.close()

    def test_mixed_outcomes_are_all_counted(self, tmp_path):
        workers = 8
        db = self._race(
            tmp_path,
            lambda i: _exp(outcome="success" if i % 2 == 0 else "failure"),
            workers=workers,
        )
        try:
            row = db.export_experiences()[0]
            assert row["observations"] == workers
            # No lost update: every observation landed in exactly one counter.
            assert row["success_count"] + row["failure_count"] == workers
            assert row["success_count"] == workers // 2
        finally:
            db.close()

    def test_distinct_tasks_stay_distinct_under_load(self, tmp_path):
        workers = 8
        db = self._race(
            tmp_path, lambda i: _exp(task=f"distinct task number {i}"), workers=workers
        )
        try:
            assert len(db.export_experiences()) == workers
        finally:
            db.close()


# ── Backward compatibility & safety ─────────────────────────────────────


class TestBackwardCompatibility:
    def test_pre_existing_db_without_the_table_gets_it_on_open(self, tmp_path):
        """A state.db written before this feature must open and gain the table.

        The table is declared in SCHEMA_SQL, so the ordinary executescript on
        open creates it — no version-gated migration, no data loss.
        """
        path = tmp_path / "legacy.db"
        legacy = SessionDB(path)
        legacy.create_session("sess-legacy", source="cli")
        legacy.close()

        raw = sqlite3.connect(str(path))
        raw.execute("DROP TABLE experiences")
        raw.commit()
        assert raw.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name='experiences'"
        ).fetchone()[0] == 0
        raw.close()

        reopened = SessionDB(path)
        assert reopened.record_experience(_exp())
        assert reopened.experience_stats()["total"] == 1
        # Pre-existing session data survived the reopen.
        assert reopened.get_session("sess-legacy") is not None

    def test_read_only_db_never_writes(self, tmp_path):
        path = tmp_path / "state.db"
        SessionDB(path).close()
        ro = SessionDB(path, read_only=True)
        assert ro.record_experience(_exp()) is None
        assert ro.record_experience_correction("x", "y") is False
        assert ro.prune_experiences() == 0
        assert ro.clear_experiences() == 0

    def test_experience_writes_do_not_disturb_session_tables(self, db):
        db.create_session("s-1", source="cli")
        db.record_experience(_exp())
        assert db.get_session("s-1") is not None
        assert db.experience_stats()["total"] == 1
