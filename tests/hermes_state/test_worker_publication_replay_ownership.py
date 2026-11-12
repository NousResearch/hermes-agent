import pytest
from hermes_state import SessionDB
from hermes_state_runtime import (begin_runtime_epoch, register_worker_execution, mutate_worker_execution,
    mutate_runtime_session, admission_fingerprint, RuntimeStoreError)
from hermes_state_terminal import terminal_worker_receipt
from hermes_state_lineage import SessionIdOwnedElsewhereError, SessionLineageAmbiguousError
from agent.runtime_session_store import RuntimeSessionStore

def snap(db):
    return {t:[tuple(x) for x in db._conn.execute("SELECT * FROM "+t+" ORDER BY 1")] for t in
            ("sessions","messages","state_meta","worker_executions","worker_receipts","compression_locks")}

@pytest.fixture
def pair(tmp_path,monkeypatch):
    monkeypatch.setenv("HERMES_HOME",str(tmp_path))
    (tmp_path/"profiles"/"science").mkdir(parents=True)
    with SessionDB(tmp_path/"state.db") as a, SessionDB(tmp_path/"profiles"/"science"/"state.db") as b:
        yield a,b

def test_publish_lost_ack_ambiguous_child(pair,tmp_path,monkeypatch):
    db,sibling=pair
    db.create_session("root","cli",system_prompt="PREFIX")
    epoch=begin_runtime_epoch(db,instance_id="critic")
    scope=dict(epoch=epoch,execution_id="w",session_id="root",generation=0)
    register_worker_execution(db,**scope,kind="compute",adoption_secret="fixture-proof")
    seen={}
    def rpc(method,**params):
        result=mutate_worker_execution(db,**params)
        if params["operation"]=="compression.publish":
            seen.update(params=params,result=result)
            raise TimeoutError("lost ack")
        return result
    store=RuntimeSessionStore(rpc,scope,tmp_path/"outbox")
    try:
        assert store.try_acquire_compression_lock("root","holder")
        with pytest.raises(TimeoutError):
            store.publish_compression_child(parent_session_id="root",child_session_id="child",source="cli",
                messages=[{"role":"assistant","content":"summary"}],system_prompt="PREFIX",compression_lock_holder="holder")
        p=seen["params"]
        assert db._read_one("SELECT session_id FROM worker_executions WHERE execution_id='w'")[0]=="child"
        assert mutate_worker_execution(db,**p)==seen["result"]
        # Deliberately introduce a legacy split-owner condition after the successful commit.
        sibling._conn.execute("INSERT INTO sessions(id,source,started_at,profile_name) VALUES ('child','cli',1,'science')")
        sibling._conn.commit()
        before=snap(db),snap(sibling)
        assert mutate_worker_execution(db,**p) == seen["result"]
        import hermes_state_runtime as runtime
        original = runtime._read_worker_receipt
        reads = []
        def first_miss(*args):
            reads.append(True)
            return (False, None) if len(reads) == 1 else original(*args)
        monkeypatch.setattr(runtime, "_read_worker_receipt", first_miss)
        assert mutate_worker_execution(db,**p) == seen["result"]
        assert len(reads) == 2
        monkeypatch.setattr(runtime, "_read_worker_receipt", original)
        with pytest.raises(RuntimeStoreError, match="stale_epoch"):
            mutate_worker_execution(db,**{**p, "epoch": p["epoch"] + 1})
        with pytest.raises(RuntimeStoreError, match="stale_generation"):
            mutate_worker_execution(db,**{**p, "generation": p["generation"] + 1})
        changed={**p,"sequence":p["sequence"]+1}
        with pytest.raises(SessionLineageAmbiguousError):
            mutate_worker_execution(db,**changed)
        assert (snap(db),snap(sibling))==before
    finally:
        store.failure=None
        store.journal["pending"]=[]
        store.close()


@pytest.mark.parametrize("null_result", [False, True])
def test_ordinary_worker_retry_survives_new_split_owner(pair, null_result):
    from hermes_state_runtime import apply_worker_receipt
    db, sibling = pair
    db.create_session("owned", "cli")
    epoch = begin_runtime_epoch(db, instance_id="critic")
    scope = dict(epoch=epoch, execution_id="ordinary", session_id="owned", generation=0)
    register_worker_execution(db, **scope, kind="compute", adoption_secret="fixture-proof")
    if null_result:
        def mutate(sequence):
            return apply_worker_receipt(db, **scope, sequence=sequence, operation="fixture.read", payload={},
                                        handlers={"fixture.read": lambda *_: None})
    else:
        def mutate(sequence):
            return mutate_worker_execution(db, **scope, sequence=sequence, operation="transcript.append",
                                           payload={"messages": [{"role": "user", "content": "hello"}]})
    first = mutate(1)
    assert first is None if null_result else first["count"] == 1
    sibling._conn.execute("INSERT INTO sessions(id,source,started_at,profile_name) VALUES ('owned','cli',1,'science')")
    sibling._conn.commit()
    before = snap(db), snap(sibling)
    assert mutate(1) == first
    with pytest.raises(SessionLineageAmbiguousError):
        mutate(2)
    assert (snap(db), snap(sibling)) == before


def test_null_receipt_commits_after_initial_cache_miss(pair, monkeypatch):
    import hermes_state_runtime as runtime
    db, sibling = pair
    db.create_session("owned", "cli")
    epoch = begin_runtime_epoch(db, instance_id="critic")
    scope = dict(epoch=epoch, execution_id="null-race", session_id="owned", generation=0)
    register_worker_execution(db, **scope, kind="compute", adoption_secret="fixture-proof")
    def mutate():
        return runtime.apply_worker_receipt(db, **scope, sequence=1, operation="fixture.read", payload={},
                                            handlers={"fixture.read": lambda *_: None})
    assert mutate() is None
    before = snap(db), snap(sibling)
    original = runtime._read_worker_receipt
    reads = []
    def initial_miss(*args):
        reads.append(True)
        return (False, None) if len(reads) == 1 else original(*args)
    monkeypatch.setattr(runtime, "_read_worker_receipt", initial_miss)
    assert mutate() is None
    assert (snap(db), snap(sibling)) == before
