"""A repair plan cannot delete history when a copy appears after its scan."""
from pathlib import Path
import pytest
from hermes_state import SessionDB
from hermes_cli.sessions_repair_profiles import RepairPlan, Store, _Session

@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profile = tmp_path / "profiles" / "science"
    profile.mkdir(parents=True)
    with SessionDB(tmp_path / "state.db") as source, SessionDB(profile / "state.db") as target:
        work = _Session(read_only=False)
        try:
            yield tmp_path, profile, source, target, work
        finally:
            work.close()

def seed_source(db, sid, parent=None):
    db.create_session(sid, "cli", profile_name="science", session_key=f"agent:science:local:dm:{sid}",
                      parent_session_id=parent)
    db.append_message(sid, "user", f"source {sid}")

def rows(db, ids):
    return {table: [tuple(row) for row in db._conn.execute(
        f"SELECT * FROM {table} WHERE {col} IN ({','.join('?' for _ in ids)}) ORDER BY 1", ids)]
        for table, col in (("sessions", "id"), ("messages", "session_id"))}

@pytest.mark.parametrize("duplicate", ["parent", "child"])
@pytest.mark.parametrize("kind", ["matching", "divergent"])
@pytest.mark.parametrize("state", ["retired", "live"])
def test_late_retired_copy_holds_component_but_not_unrelated(stores, duplicate, kind, state):
    root, profile, source, target, work = stores
    seed_source(source, "parent")
    seed_source(source, "child", parent="parent")
    seed_source(source, "unrelated")
    plan = RepairPlan([Store("default", root, routing=True), Store("science", profile)], root=root).scan(work)
    assert {f.subject for f in plan.repairable} >= {"parent", "child", "unrelated"}
    target._conn.execute("INSERT INTO sessions(id,source,started_at,ended_at,end_reason,archived) "
                         "VALUES (?,'cli',1,?,?,?)",
                         (duplicate, 2 if state == "retired" else None,
                          "adopted_by_profile" if state == "retired" else None, int(state == "retired")))
    text = f"source {duplicate}" if kind == "matching" else "different history"
    target._conn.execute("INSERT INTO messages(session_id,role,content,timestamp) VALUES (?,'user',?,1)",
                         (duplicate, text))
    target._conn.commit()
    component = ("parent", "child")
    before = rows(source, component), rows(target, component)
    result = plan.apply(work, snapshot=lambda store: None)
    assert result["failures"], result
    assert (rows(source, component), rows(target, component)) == before
    assert source.get_session("unrelated") is None
    assert target.get_session("unrelated") is not None

def test_target_present_during_import_never_deletes_source(stores, monkeypatch):
    root, profile, source, _target, work = stores
    seed_source(source, "late")
    plan = RepairPlan([Store("default", root, routing=True), Store("science", profile)], root=root).scan(work)
    original = SessionDB.import_moved_session
    def races(self, payload, **kwargs):
        if payload["session"]["id"] == "late":
            self._conn.execute("INSERT INTO sessions(id,source,started_at,ended_at,end_reason) "
                               "VALUES ('late','cli',1,2,'adopted_by_profile')")
            self._conn.commit()
        return original(self, payload, **kwargs)
    monkeypatch.setattr(SessionDB, "import_moved_session", races)
    result = plan.apply(work, snapshot=lambda store: None)
    assert result["failures"], result
    assert source.get_messages("late")[0]["content"] == "source late"

def test_incomplete_copy_with_equal_count_never_deletes_source(stores, monkeypatch):
    root, profile, source, _target, work = stores
    seed_source(source, "late")
    plan = RepairPlan([Store("default", root, routing=True), Store("science", profile)], root=root).scan(work)
    original = SessionDB.import_moved_session
    def corrupts(self, payload, **kwargs):
        result = original(self, payload, **kwargs)
        self._conn.execute("UPDATE messages SET content='corrupt' WHERE session_id='late'")
        self._conn.commit()
        return result
    monkeypatch.setattr(SessionDB, "import_moved_session", corrupts)
    result = plan.apply(work, snapshot=lambda store: None)
    assert result["failures"] or result["totals"].get("copied_incomplete"), result
    assert source.get_messages("late")[0]["content"] == "source late"


def test_writable_scan_does_not_create_absent_default_store(tmp_path, monkeypatch):
    from hermes_cli.sessions_repair_profiles import scan_stores
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profile = tmp_path / "profiles" / "science"
    profile.mkdir(parents=True)
    with SessionDB(profile / "state.db"):
        pass
    default = tmp_path / "state.db"
    assert not default.exists()
    plan, work = scan_stores([Store("default", tmp_path, routing=True), Store("science", profile)], read_only=False)
    try:
        assert not default.exists()
        result = plan.apply(work, snapshot=lambda store: None)
        assert not result["failures"]
        assert not default.exists()
    finally:
        work.close()

def test_writable_scan_lazily_creates_default_for_real_move(tmp_path, monkeypatch):
    from hermes_cli.sessions_repair_profiles import scan_stores
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profile = tmp_path / "profiles" / "science"
    profile.mkdir(parents=True)
    with SessionDB(profile / "state.db") as source:
        source.create_session("move", "cli", profile_name="science", session_key="agent:main:local:dm:move")
        source.append_message("move", "user", "history")
    default = tmp_path / "state.db"
    plan, work = scan_stores([Store("default", tmp_path, routing=True), Store("science", profile)],
                             read_only=False, legacy_main="move")
    try:
        assert not default.exists()
        result = plan.apply(work, snapshot=lambda store: None)
        assert not result["failures"], result
        assert default.exists()
        assert work.db(Store("default", tmp_path, routing=True)).get_messages("move")[0]["content"] == "history"
    finally:
        work.close()


@pytest.mark.parametrize("blocked_owner", ["source", "target"])
def test_move_refuses_when_queued_accounting_cannot_flush(stores, monkeypatch, blocked_owner):
    root, profile, source, target, work = stores
    seed_source(source, "late")
    plan = RepairPlan([Store("default", root, routing=True), Store("science", profile)], root=root).scan(work)
    original = SessionDB.flush_token_counts
    blocked_path = source.db_path if blocked_owner == "source" else target.db_path
    def blocked(self, *args, **kwargs):
        return False if self.db_path == blocked_path else original(self, *args, **kwargs)
    monkeypatch.setattr(SessionDB, "flush_token_counts", blocked)
    result = plan.apply(work, snapshot=lambda store: None)
    assert result["failures"], result
    assert source.get_messages("late")[0]["content"] == "source late"
    assert target.get_session("late") is None

def test_source_usage_changed_after_copy_keeps_donor(stores, monkeypatch):
    root, profile, source, _target, work = stores
    seed_source(source, "late")
    plan = RepairPlan([Store("default", root, routing=True), Store("science", profile)], root=root).scan(work)
    original = SessionDB.import_moved_session
    def racing(self, payload, **kwargs):
        result = original(self, payload, **kwargs)
        source._conn.execute("UPDATE sessions SET input_tokens=99 WHERE id='late'")
        source._conn.commit()
        return result
    monkeypatch.setattr(SessionDB, "import_moved_session", racing)
    result = plan.apply(work, snapshot=lambda store: None)
    assert result["failures"], result
    assert source.get_session("late")["input_tokens"] == 99
    assert source.get_messages("late")[0]["content"] == "source late"


def test_late_third_copy_does_not_create_absent_target(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    science, other = tmp_path / "profiles" / "science", tmp_path / "profiles" / "other"
    science.mkdir(parents=True)
    other.mkdir()
    with SessionDB(tmp_path / "state.db") as source:
        seed_source(source, "late")
        work = _Session(read_only=False)
        try:
            plan = RepairPlan([Store("default", tmp_path, routing=True), Store("science", science)], root=tmp_path).scan(work)
            with SessionDB(other / "state.db") as foreign:
                foreign._conn.execute("INSERT INTO sessions(id,source,started_at,ended_at,end_reason) "
                                      "VALUES ('late','cli',1,2,'adopted_by_profile')")
                foreign._conn.commit()
                result = plan.apply(work, snapshot=lambda store: None)
                assert result["failures"], result
                assert not (science / "state.db").exists()
                assert source.get_messages("late")[0]["content"] == "source late"
        finally:
            work.close()


def test_move_refuses_a_writer_queued_between_flush_and_guard(stores, monkeypatch):
    import threading
    from contextlib import contextmanager
    import hermes_state_lineage as lineage
    root, profile, source, target, work = stores
    seed_source(source, "late")
    plan = RepairPlan([Store("default", root, routing=True), Store("science", profile)], root=root).scan(work)
    actual_source = work.db(Store("default", root, routing=True))
    started, release = threading.Event(), threading.Event()
    original_apply = actual_source._apply_token_batch
    original_guard = lineage.session_lineage_guard
    injected = False
    def delayed_batch(batch):
        started.set()
        assert release.wait(15)
        return original_apply(batch)
    @contextmanager
    def queue_before_guard(path, ids, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            actual_source.queue_token_counts("late", input_tokens=7, model="m")
            assert started.wait(5)
        with original_guard(path, ids, **kwargs) as observed:
            yield observed
    monkeypatch.setattr(actual_source, "_apply_token_batch", delayed_batch)
    monkeypatch.setattr(lineage, "session_lineage_guard", queue_before_guard)
    try:
        result = plan.apply(work, snapshot=lambda store: None)
        assert result["failures"], result
        assert source.get_messages("late")[0]["content"] == "source late"
        assert target.get_session("late") is None
    finally:
        release.set()
        assert actual_source.flush_token_counts(timeout=10)
    assert source.get_session("late")["input_tokens"] == 7


def test_repair_refuses_other_handle_pending_then_moves_after_drain(stores, monkeypatch):
    import threading
    import time
    root, profile, source, target, work = stores
    seed_source(source, "late")
    plan = RepairPlan([Store("default", root, routing=True), Store("science", profile)], root=root).scan(work)
    started, release = threading.Event(), threading.Event()
    with SessionDB(source.db_path) as producer:
        original = producer._apply_token_batch
        def delayed(batch):
            started.set()
            assert release.wait(15)
            return original(batch)
        monkeypatch.setattr(producer, "_apply_token_batch", delayed)
        producer.queue_token_counts("late", input_tokens=7, model="m", billing_provider="p", api_call_count=1)
        assert started.wait(5)
        before = rows(source, ("late",)), rows(target, ("late",))
        try:
            start = time.monotonic()
            refused = plan.apply(work, snapshot=lambda store: None)
            assert time.monotonic() - start < 3
            assert refused["failures"], refused
            assert (rows(source, ("late",)), rows(target, ("late",))) == before
        finally:
            release.set()
        assert producer.flush_token_counts(timeout=10)
    retry = RepairPlan([Store("default", root, routing=True), Store("science", profile)], root=root).scan(work)
    done = retry.apply(work, snapshot=lambda store: None)
    assert not done["failures"], done
    assert source.get_session("late") is None
    assert target.get_session("late")["input_tokens"] == 7

def test_repair_refuses_other_process_pending_then_moves_after_drain(stores, monkeypatch):
    import os
    from argparse import Namespace
    import hermes_cli.sessions_cmd_repair_profiles as cli_repair
    import select
    import subprocess
    import sys
    import time
    root, profile, source, target, _work = stores
    seed_source(source, "late")
    repair_stores = [Store("default", root, routing=True), Store("science", profile)]
    monkeypatch.setattr(cli_repair, "enumerate_stores", lambda: repair_stores)
    monkeypatch.setattr(cli_repair, "live_gateway_homes", lambda _stores: [])
    monkeypatch.setattr(cli_repair, "default_snapshot", lambda _store: None)
    args = Namespace(apply=True, json=False, yes=True, legacy_main="report")
    program = """
from pathlib import Path
import threading
import sys
from hermes_state import SessionDB
db = SessionDB(Path(sys.argv[1]))
started, release = threading.Event(), threading.Event()
original = db._apply_token_batch
def delayed(batch):
    started.set()
    assert release.wait(20)
    return original(batch)
db._apply_token_batch = delayed
db.queue_token_counts("late", input_tokens=7, model="m", billing_provider="p", api_call_count=1)
assert started.wait(5)
print("READY", flush=True)
sys.stdin.readline()
release.set()
assert db.flush_token_counts(timeout=10)
db.close()
print("DONE", flush=True)
"""
    child = subprocess.Popen([sys.executable, "-u", "-c", program, str(source.db_path)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, env=os.environ.copy())
    try:
        ready, _, _ = select.select([child.stdout], [], [], 10)
        assert ready, "child did not report readiness"
        line = child.stdout.readline().strip()
        if line != "READY":
            _out, err = child.communicate(timeout=5)
            pytest.fail(f"child setup failed: {err}")
        before = rows(source, ("late",)), rows(target, ("late",))
        start = time.monotonic()
        refused = cli_repair.cmd_repair_profiles(args)
        assert time.monotonic() - start < 3
        assert refused == 1
        assert (rows(source, ("late",)), rows(target, ("late",))) == before
        child.stdin.write("drain\n")
        child.stdin.flush()
        out, err = child.communicate(timeout=15)
        assert child.returncode == 0 and "DONE" in out, err
        done = cli_repair.cmd_repair_profiles(args)
        assert done == 0
        assert source.get_session("late") is None
        assert target.get_session("late")["input_tokens"] == 7
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=5)


def _repair_db_snapshot(db):
    tables = ("sessions", "messages", "session_model_usage", "state_meta", "gateway_routing")
    return {table: [tuple(row) for row in db._conn.execute(f"SELECT * FROM {table} ORDER BY 1")]
            for table in tables}

def test_repair_refuses_target_handle_pending_then_moves_after_drain(stores, monkeypatch):
    import threading
    import time
    root, profile, source, target, work = stores
    seed_source(source, "late")
    target.create_session("resident", "cli", profile_name="science", session_key="agent:science:local:dm:resident")
    plan = RepairPlan([Store("default", root, routing=True), Store("science", profile)], root=root).scan(work)
    started, release = threading.Event(), threading.Event()
    with SessionDB(target.db_path) as producer:
        original = producer._apply_token_batch
        def delayed(batch):
            started.set()
            assert release.wait(15)
            return original(batch)
        monkeypatch.setattr(producer, "_apply_token_batch", delayed)
        producer.queue_token_counts("resident", input_tokens=7, model="m", billing_provider="p", api_call_count=1)
        assert started.wait(5)
        before = _repair_db_snapshot(source), _repair_db_snapshot(target)
        try:
            start = time.monotonic()
            refused = plan.apply(work, snapshot=lambda store: None)
            assert time.monotonic() - start < 3
            assert refused["failures"], refused
            assert (_repair_db_snapshot(source), _repair_db_snapshot(target)) == before
        finally:
            release.set()
        assert producer.flush_token_counts(timeout=10)
    retry = RepairPlan([Store("default", root, routing=True), Store("science", profile)], root=root).scan(work)
    done = retry.apply(work, snapshot=lambda store: None)
    assert not done["failures"], done
    assert source.get_session("late") is None
    assert target.get_session("late") is not None
    assert target.get_session("resident")["input_tokens"] == 7

def test_repair_refuses_target_process_pending_then_moves_after_drain(stores):
    import os
    import select
    import subprocess
    import sys
    import time
    root, profile, source, target, work = stores
    seed_source(source, "late")
    target.create_session("resident", "cli", profile_name="science", session_key="agent:science:local:dm:resident")
    plan = RepairPlan([Store("default", root, routing=True), Store("science", profile)], root=root).scan(work)
    program = """
from pathlib import Path
import threading
import sys
from hermes_state import SessionDB
db = SessionDB(Path(sys.argv[1]))
started, release = threading.Event(), threading.Event()
original = db._apply_token_batch
def delayed(batch):
    started.set()
    assert release.wait(20)
    return original(batch)
db._apply_token_batch = delayed
db.queue_token_counts("resident", input_tokens=7, model="m", billing_provider="p", api_call_count=1)
assert started.wait(5)
print("READY", flush=True)
sys.stdin.readline()
release.set()
assert db.flush_token_counts(timeout=10)
db.close()
print("DONE", flush=True)
"""
    child = subprocess.Popen([sys.executable, "-u", "-c", program, str(target.db_path)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, env=os.environ.copy())
    try:
        ready, _, _ = select.select([child.stdout], [], [], 10)
        assert ready, "child did not report readiness"
        line = child.stdout.readline().strip()
        if line != "READY":
            _out, err = child.communicate(timeout=5)
            pytest.fail(f"child setup failed: {err}")
        before = _repair_db_snapshot(source), _repair_db_snapshot(target)
        start = time.monotonic()
        refused = plan.apply(work, snapshot=lambda store: None)
        assert time.monotonic() - start < 3
        assert refused["failures"], refused
        assert (_repair_db_snapshot(source), _repair_db_snapshot(target)) == before
        child.stdin.write("drain\n")
        child.stdin.flush()
        out, err = child.communicate(timeout=15)
        assert child.returncode == 0 and "DONE" in out, err
        retry = RepairPlan([Store("default", root, routing=True), Store("science", profile)], root=root).scan(work)
        done = retry.apply(work, snapshot=lambda store: None)
        assert not done["failures"], done
        assert source.get_session("late") is None
        assert target.get_session("late") is not None
        assert target.get_session("resident")["input_tokens"] == 7
    finally:
        if child.poll() is None:
            child.stdin.write("drain\n")
            child.stdin.flush()
            try:
                child.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate(timeout=5)
