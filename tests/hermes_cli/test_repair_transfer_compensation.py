"""Handled repair failures leave the source usable and permit a fresh retry."""
from pathlib import Path
import sqlite3

import pytest

import hermes_cli.sessions_repair_profiles as repair
from hermes_state import SessionDB


@pytest.fixture
def repair_stores(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profile = tmp_path / "profiles" / "science"
    profile.mkdir(parents=True)
    stores = [repair.Store("default", tmp_path, routing=True), repair.Store("science", profile)]
    with SessionDB(tmp_path / "state.db") as source, SessionDB(profile / "state.db") as target:
        for sid, parent in (("parent", None), ("child", "parent"), ("unrelated", None)):
            source.create_session(sid, "cli", profile_name="science",
                                  session_key=f"agent:science:local:dm:{sid}", parent_session_id=parent)
            source.append_message(sid, "user", f"source {sid}")
        target._conn.execute("INSERT INTO system_prompts(hash,prompt) VALUES ('resident-cache','preserve')")
        target._conn.commit()
        work = repair._Session(read_only=False)
        try:
            yield stores, source, target, work
        finally:
            work.close()


def plan_for(stores, work):
    return repair.RepairPlan(stores, root=stores[0].home).scan(work)


@pytest.mark.parametrize("failure", ["import", "verify"])
def test_handled_child_failure_compensates_new_component_and_retries(repair_stores, monkeypatch, failure):
    stores, source, target, work = repair_stores
    original = repair._verify_repair_copy
    verified = []
    def verify(src, dst, sid, payload):
        original(src, dst, sid, payload)
        verified.append(sid)
        if sid == "child" and failure == "verify":
            raise RuntimeError("injected child verification failure")
    monkeypatch.setattr(repair, "_verify_repair_copy", verify)
    if failure == "import":
        target._conn.execute("""CREATE TRIGGER reject_child BEFORE INSERT ON sessions
            WHEN new.id='child' BEGIN SELECT RAISE(ABORT, 'injected child import failure'); END""")
        target._conn.commit()
    result = plan_for(stores, work).apply(work, snapshot=lambda store: None)
    assert "parent" in verified
    assert ("child" in verified) == (failure == "verify")
    assert result["failures"]
    # Exercise the real ownership fence before merely checking target inventory.
    source.append_message("parent", "user", "still usable")
    source.append_message("child", "user", "also usable")
    assert source.get_session("child")["parent_session_id"] == "parent"
    assert target.get_session("parent") is None
    assert target.get_session("child") is None
    assert source.get_session("unrelated") is None
    assert target.get_session("unrelated") is not None
    assert target._conn.execute("SELECT prompt FROM system_prompts WHERE hash='resident-cache'").fetchone()[0] == "preserve"
    assert result["totals"].get("moved") == 1
    monkeypatch.setattr(repair, "_verify_repair_copy", original)
    target._conn.execute("DROP TRIGGER IF EXISTS reject_child")
    target._conn.commit()
    retry = plan_for(stores, work)
    assert not [finding for finding in retry.findings if finding.kind == "dual_home"]
    done = retry.apply(work, snapshot=lambda store: None)
    assert not done["failures"], done
    assert done["totals"]["moved"] == 2
    assert source.get_session("parent") is None
    assert source.get_session("child") is None
    assert target.get_session("child")["parent_session_id"] == "parent"
    assert target.get_messages("parent")[-1]["content"] == "still usable"
    assert target.get_messages("child")[-1]["content"] == "also usable"



def component_rows(db):
    return {table: [tuple(row) for row in db._conn.execute(
        f"SELECT * FROM {table} WHERE {column} IN ('parent','child') ORDER BY 1")]
        for table, column in (("sessions", "id"), ("messages", "session_id"),
                              ("session_model_usage", "session_id"))}


def test_compensation_failure_rolls_back_and_requires_adjudication(repair_stores, monkeypatch):
    from hermes_state_lineage import SessionLineageAmbiguousError
    stores, source, target, work = repair_stores
    original = repair._verify_repair_copy
    copied = {}
    def verify(src, dst, sid, payload):
        original(src, dst, sid, payload)
        if sid == "child":
            copied.update(component_rows(dst))
            raise RuntimeError("injected verification failure")
    monkeypatch.setattr(repair, "_verify_repair_copy", verify)
    target._conn.execute("""CREATE TRIGGER reject_compensation BEFORE DELETE ON sessions
        WHEN old.id='parent' BEGIN SELECT RAISE(ABORT, 'injected compensation failure'); END""")
    target._conn.commit()
    before = component_rows(source)
    result = plan_for(stores, work).apply(work, snapshot=lambda store: None)
    assert component_rows(source) == before
    assert component_rows(target) == copied
    assert result["totals"].get("moved") == 1
    assert result["totals"].get("imported") == 1
    assert all("compensation failed" in item["error"] for item in result["failures"])
    with pytest.raises(SessionLineageAmbiguousError):
        source.append_message("parent", "user", "must not land")
    retry = plan_for(stores, work)
    assert not [f for f in retry.repairable if f.subject in {"parent", "child"}]
    assert {f.subject for f in retry.findings if f.kind == "dual_home"} == {"parent", "child"}
    retry.apply(work, snapshot=lambda store: None)
    assert component_rows(source) == before
    assert component_rows(target) == copied


@pytest.mark.parametrize("outcome", ["uncertain", "present"])
def test_unconfirmed_import_is_not_compensated(repair_stores, monkeypatch, outcome):
    stores, source, target, work = repair_stores
    original = SessionDB.import_moved_session
    copied = {}
    def interrupted(self, payload, **kwargs):
        sid = payload["session"]["id"]
        if sid == "child" and outcome == "present":
            # Model an outside copy appearing after inventory, without claiming its creation.
            self._conn.execute("""INSERT INTO sessions(id,source,started_at,parent_session_id)
                VALUES ('child','cli',1,'parent')""")
            self._conn.commit()
            copied.update(component_rows(self))
            return "present"
        result = original(self, payload, **kwargs)
        if sid == "child":
            copied.update(component_rows(self))
            raise sqlite3.OperationalError("injected post-commit uncertainty")
        return result
    monkeypatch.setattr(SessionDB, "import_moved_session", interrupted)
    before = component_rows(source)
    result = plan_for(stores, work).apply(work, snapshot=lambda store: None)
    assert component_rows(source) == before
    assert component_rows(target) == copied
    errors = " ".join(item["error"] for item in result["failures"])
    assert "compensation failed" in errors
    assert ("import outcome uncertain" if outcome == "uncertain" else "target copy appeared") in errors
    assert result["totals"].get("moved") == 1
    retry = plan_for(stores, work)
    assert not [f for f in retry.repairable if f.subject in {"parent", "child"}]
    retry.apply(work, snapshot=lambda store: None)
    assert component_rows(target) == copied


def test_compensation_holds_complete_component_through_sqlite_commit(repair_stores, monkeypatch):
    import os
    import hermes_state_lineage as lineage
    stores, _source, target, work = repair_stores
    original_verify = repair._verify_repair_copy
    original_write = SessionDB._execute_write
    commits = []
    required = {f"{os.getpid()}:{path.resolve()}" for path in
                lineage._lineage_lock_paths(stores[0].home, {"parent", "child", "unrelated"})}
    def verify(src, dst, sid, payload):
        original_verify(src, dst, sid, payload)
        if sid == "child":
            raise RuntimeError("injected verification failure")
    def write(self, fn, *args, **kwargs):
        if fn.__name__ != "discard":
            return original_write(self, fn, *args, **kwargs)
        def trace(sql):
            if sql == "COMMIT":
                commits.append(required <= set(lineage._THREAD_LOCKS.active))
        self._conn.set_trace_callback(trace)
        try:
            result = original_write(self, fn, *args, **kwargs)
            assert required <= set(lineage._THREAD_LOCKS.active)
            assert not self._conn.in_transaction
            return result
        finally:
            self._conn.set_trace_callback(None)
    monkeypatch.setattr(repair, "_verify_repair_copy", verify)
    monkeypatch.setattr(SessionDB, "_execute_write", write)
    result = plan_for(stores, work).apply(work, snapshot=lambda store: None)
    assert result["failures"]
    assert commits == [True]
    assert target.get_session("parent") is None
    assert target.get_session("child") is None
