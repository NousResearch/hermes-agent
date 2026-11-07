"""Upgrade and rollback against main's store as the oracle, both directions and writable.

Every fixture is built once by main (``_fixtures.py``) and copied per chain. A chain runs each store step in its
own process (``_oracle.run_step``) with a private HERMES_HOME. The subject is the backend holder, or main's store
itself as the control that proves the comparisons hold on the oracle alone. Chains run concurrently on a small pool
and each test waits for its own.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.persistence_sqlite.upgrade._oracle import (
    EMPTY_SHA256, FIXTURES, copy_store, file_digests, lost_fixture_rows, run_step, snapshot, snapshot_diff,
)

SUBJECTS = ("m1", "backend")
_WORKERS = 8

pytestmark = [pytest.mark.parametrize("subject", SUBJECTS), pytest.mark.parametrize("fixture", FIXTURES)]


def _assert_reads_hold(reads: dict, writes: list, fixture_version):
    """Both read-only reads answer identically, every write is there and the version never went down."""
    m1, subject = reads["m1"], reads["subject"]
    assert "error" not in m1 and "error" not in subject, reads
    assert m1 == subject
    sessions = {row[0]: row for row in m1["sessions"]}
    contents = {(row[0], row[2]) for row in m1["messages"]}
    meta = dict(map(tuple, m1["state_meta"]))
    for tag in writes:
        session_id = f"{tag}-session"
        assert sessions[session_id][2:] == [f"{tag}-model", 2], sessions[session_id]
        assert {(session_id, f"{tag} payment question"), (session_id, f"{tag} answer")} <= contents
        assert meta[f"{tag}-meta"] == tag and m1["meta_api"][f"{tag}-meta"] == tag
    (version,), = m1["schema_version"]
    if fixture_version is not None:
        assert version >= fixture_version


class _Chains:
    """Runs every (case, fixture, subject) chain this module collected, concurrently, once."""

    def __init__(self, root: Path):
        self.root = root
        self.pool = ThreadPoolExecutor(max_workers=_WORKERS)
        self.lock = threading.Lock()
        self.builds, self.chains = {}, {}

    def start(self, items, env: dict):
        with self.lock:
            if self.chains:
                return
            wanted = [(item.originalname, item.callspec.params["fixture"], item.callspec.params["subject"])
                      for item in items if item.originalname in CASES]
            for fixture in sorted({w[1] for w in wanted}):
                self.builds[fixture] = self.pool.submit(self._build, fixture, env)
            for key in wanted:
                self.chains[key] = self.pool.submit(self._run, *key, env)

    def future(self, case: str, fixture: str, subject: str):
        return self.chains[(case, fixture, subject)]

    def _build(self, fixture: str, env: dict) -> Path:
        out = self.root / "fixtures" / fixture
        out.mkdir(parents=True)
        run_step("m1", "build", {**env, "HERMES_HOME": str(out)}, db=str(out / "state.db"), name=fixture)
        return out

    def _run(self, case: str, fixture: str, subject: str, env: dict):
        built = self.builds[fixture].result()
        chain_dir = self.root / "chains" / f"{case}-{fixture}-{subject}"

        def copy(label: str):
            db = copy_store(built, chain_dir / label)
            return db, {**env, "HERMES_HOME": str(db.parent)}

        return CASES[case](fixture, subject, copy)


def _upgrade(fixture, subject, copy):
    db, env = copy("store")
    before = snapshot(db) if db.exists() else None
    version = before["tables"]["schema_version"][0][0] if before else None
    run_step(subject, "raw_write", env, db=str(db), tag="subject")
    run_step("m1", "api_write", env, db=str(db), tag="m1")
    reads = {"m1": run_step("m1", "read", env, db=str(db)), "subject": run_step(subject, "read", env, db=str(db))}
    return before, version, reads, snapshot(db), ["subject", "m1"]


def _rollback(fixture, subject, copy):
    db, env = copy("store")
    before = snapshot(db) if db.exists() else None
    version = before["tables"]["schema_version"][0][0] if before else None
    run_step("m1", "api_write", env, db=str(db), tag="m1-first")
    run_step(subject, "raw_write", env, db=str(db), tag="subject")
    run_step("m1", "api_write", env, db=str(db), tag="m1-again")
    reads = {"m1": run_step("m1", "read", env, db=str(db)), "subject": run_step(subject, "read", env, db=str(db))}
    return before, version, reads, snapshot(db), ["m1-first", "subject", "m1-again"]


def _equivalence(fixture, subject, copy):
    oracle_db, oracle_env = copy("oracle")
    subject_db, subject_env = copy("subject")
    run_step("m1", "open", oracle_env, db=str(oracle_db))
    run_step(subject, "open", subject_env, db=str(subject_db))
    return snapshot(oracle_db), snapshot(subject_db)


def _read_only(fixture, subject, copy):
    out = {}
    for label, side in (("m1", "m1"), ("subject", subject)):
        db, env = copy(label)
        before = file_digests(db)
        result = run_step(side, "read", env, db=str(db))
        out[label] = (before, file_digests(db), result)
    return out


CASES = {
    "test_upgrade_keeps_both_sides_writes": _upgrade,
    "test_rollback_keeps_both_sides_writes": _rollback,
    "test_first_open_migrates_like_m1": _equivalence,
    "test_read_only_open_never_migrates": _read_only,
}


@pytest.fixture(scope="module")
def _chains(tmp_path_factory):
    chains = _Chains(tmp_path_factory.mktemp("oracle"))
    yield chains
    chains.pool.shutdown(cancel_futures=True)


@pytest.fixture
def chain(_chains, request):
    """The calling test's chain as a future; the first test starts every collected chain with its env."""
    _chains.start([i for i in request.session.items if getattr(i, "module", None) is request.module], dict(os.environ))
    return _chains.future(request.node.originalname, request.node.callspec.params["fixture"],
                          request.node.callspec.params["subject"])


def test_upgrade_keeps_both_sides_writes(chain, fixture, subject):
    """The subject opens the fixture writable and writes, then main reopens and writes through its API; both
    read-only reads see every write and every fixture row."""
    before, version, reads, after, writes = chain.result()
    _assert_reads_hold(reads, writes, version)
    if before is not None:
        assert lost_fixture_rows(before, after) == []


def test_rollback_keeps_both_sides_writes(chain, fixture, subject):
    """Main writes, the subject reopens and writes, main reopens and writes again; all of it reads back."""
    before, version, reads, after, writes = chain.result()
    _assert_reads_hold(reads, writes, version)
    if before is not None:
        assert lost_fixture_rows(before, after) == []


def test_first_open_migrates_like_m1(chain, fixture, subject):
    """One writable open of identical copies: same schema, version row, state_meta, rows and FTS contents."""
    oracle, migrated = chain.result()
    assert oracle["application_id_set"] and migrated["application_id_set"]
    assert snapshot_diff(oracle, migrated) == []


def test_read_only_open_never_migrates(chain, fixture, subject):
    """A read-only open leaves the main file byte for byte and writes no WAL frame (main's open creates an absent
    -wal empty); both sides leave identical copies in the same state and read the same version."""
    runs = chain.result()
    for label, (before, after, result) in runs.items():
        assert after[""] == before[""], (label, result)
        no_frames = {before["-wal"], EMPTY_SHA256} if before["-wal"] is None else {before["-wal"]}
        assert after["-wal"] in no_frames, (label, result)
    assert runs["m1"][1] == runs["subject"][1]
    assert ("error" in runs["m1"][2]) == ("error" in runs["subject"][2]), runs
    if "error" not in runs["m1"][2]:
        assert runs["m1"][2]["schema_version"] == runs["subject"][2]["schema_version"]
