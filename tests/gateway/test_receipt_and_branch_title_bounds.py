from types import SimpleNamespace
import pytest
from tests.gateway.test_messaging_session_picks import _compressed_chat, _mutate

@pytest.mark.asyncio
@pytest.mark.parametrize("title,expected", [("", None), ("  \u96ea \u200b\u2615  ", "\u96ea \u2615")])
async def test_titles(tmp_path, monkeypatch, title, expected):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner, _, _, viewer, root, fences = await _compressed_chat(tmp_path)
    try:
        reply = await viewer.dispatch(_mutate(root, fences, "rename", {"title": title}, "title"))
        assert reply["result"]["title"] == expected, reply
        db = runner.session_store._db
        before = [tuple(r) for r in db._conn.execute("SELECT * FROM state_meta ORDER BY key")]
        replay = await viewer.dispatch(_mutate(root, fences, "rename", {"title": title}, "title"))
        assert replay["result"] == reply["result"], replay
        assert [tuple(r) for r in db._conn.execute("SELECT * FROM state_meta ORDER BY key")] == before
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()

@pytest.mark.asyncio
@pytest.mark.parametrize("parent,payload,expected", [
    ("Parent", {}, "Parent #2"),
    ("Parent", {"title": ""}, "Parent #2"),
    ("Parent", {"title": "  \u96ea  "}, "\u96ea"),
    ("X"*98, {}, "X"*97 + " #2"),
    ("X"*100, {"title": ""}, "X"*97 + " #2"),
    ("X"*101, {}, "X"*97 + " #2"),
])
async def test_branch(tmp_path, monkeypatch, parent, payload, expected):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    import hermes_state_runtime as rt
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(run, "_load_gateway_config", dict)
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    db = store._db
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: None,
                             _adapter_for_source=lambda source: None)
    authority = SessionAuthority(runner, profile_id="owned", instance_id="owner", db=db,
                                 epoch=rt.begin_runtime_epoch(db, instance_id="owner"))
    owner = AuthorityConnection(authority, object(), {"user_id": "human"})
    ref = create_local_session(authority, owner.actor, dict(request_id="setup", source="cli",
                               cwd=str(tmp_path), model="m", toolsets=[]))
    sid = ref.session_id
    db._conn.execute("UPDATE sessions SET title=? WHERE id=?", (parent, sid))
    db._conn.commit()
    session = db.get_session(sid)
    try:
        async def dispatch():
            return await owner.dispatch({"id": 1, "method": "session.mutate", "params": {
                "session_id": sid, "request_id": "branch", "expected_revision": session["runtime_revision"],
                "expected_generation": session["runtime_generation"], "operation": "branch", "payload": payload}})
        reply = await dispatch()
        child = reply["result"]["branched_session_id"]
        assert db.get_session(child)["title"] == expected, reply
        assert db.get_session(sid)["title"] == parent
        if parent == "X"*100:
            for n in range(3, 12):
                current = db.get_session(sid)
                reply = await owner.dispatch({"id": n, "method": "session.mutate", "params": {
                    "session_id": sid, "request_id": f"branch-{n}",
                    "expected_revision": current["runtime_revision"],
                    "expected_generation": current["runtime_generation"], "operation": "branch", "payload": {}}})
                child = reply["result"]["branched_session_id"]
                suffix = f" #{n}"
                assert db.get_session(child)["title"] == "X" * (100 - len(suffix)) + suffix
            current = db.get_session(sid)
            before_bad = [tuple(r) for r in db._conn.execute("SELECT * FROM sessions ORDER BY id")]
            bad = await owner.dispatch({"id": 99, "method": "session.mutate", "params": {
                "session_id": sid, "request_id": "bad-title", "expected_revision": current["runtime_revision"],
                "expected_generation": current["runtime_generation"], "operation": "branch",
                "payload": {"title": "X"*101}}})
            assert bad["error"]["data"]["reason"] == "invalid_params", bad
            assert [tuple(r) for r in db._conn.execute("SELECT * FROM sessions ORDER BY id")] == before_bad
    finally:
        await owner.close()
        store.close_all_db_handles()

@pytest.mark.asyncio
async def test_exact_replay_with_foreign_ghost(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner, _, _, viewer, root, fences = await _compressed_chat(tmp_path)
    try:
        db = runner.session_store._db
        req = _mutate(root, fences, "rename", {"title": "Wanted"}, "first")
        first = await viewer.dispatch(req)
        assert first.get("result"), first
        second = await viewer.dispatch(_mutate(root, first["result"], "rename", {"title": "Other"}, "second"))
        assert second.get("result"), second
        named = tmp_path / "profiles" / "science"
        named.mkdir(parents=True)
        with SessionDB(named / "state.db") as sibling:
            db._conn.execute("INSERT INTO sessions(id,source,started_at,ended_at,end_reason,title) VALUES ('ghost','cli',1,2,'adopted_by_profile','Wanted')")
            db._conn.commit()
            sibling._conn.execute("INSERT INTO sessions(id,source,started_at,profile_name) VALUES ('ghost','cli',1,'science')")
            sibling._conn.commit()
            def snapshot():
                return [[tuple(r) for r in conn.execute(f"SELECT * FROM {t} ORDER BY 1")]
                        for conn in (db._conn, sibling._conn)
                        for t in ("sessions", "messages", "state_meta", "gateway_routing")]
            before = snapshot()
            retry = await viewer.dispatch(req)
            assert retry.get("result") == first["result"], retry
            assert snapshot() == before
            refused = await viewer.dispatch(_mutate(root, second["result"], "rename", {"title": "Wanted"}, "fresh"))
            assert refused.get("error", {}).get("data", {}).get("reason") == "admission_conflict", refused
            assert snapshot() == before
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()



def test_receipt_commits_between_first_read_and_lineage_refusal(tmp_path, monkeypatch):
    import hermes_state_lineage as lineage
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch, mutate_runtime_session
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("target", source="test")
        epoch = begin_runtime_epoch(db, instance_id="owner")
        kwargs = dict(epoch=epoch, principal_id="owner", session_id="target", request_id="race",
                      expected_revision=0, operation="rename", payload={"title": "Wanted"})
        original = lineage.guarded_discovered_session_write
        committed = []
        def concurrent_commit_then_refusal(db, ids, discover, write):
            monkeypatch.setattr(lineage, "guarded_discovered_session_write", original)
            committed.append(mutate_runtime_session(db, **kwargs))
            raise lineage.SessionIdOwnedElsewhereError("simulated ownership race")
        monkeypatch.setattr(lineage, "guarded_discovered_session_write", concurrent_commit_then_refusal)
        assert mutate_runtime_session(db, **kwargs) == committed[0]
        assert db.get_session("target")["runtime_revision"] == 1


def test_malformed_receipt_never_becomes_fresh_write(tmp_path):
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch, mutate_runtime_session, mutation_receipt_key
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("target", source="test")
        epoch = begin_runtime_epoch(db, instance_id="owner")
        kwargs = dict(epoch=epoch, principal_id="owner", session_id="target", request_id="bad",
                      expected_revision=0, operation="rename", payload={"title": "Wanted"})
        key, _ = mutation_receipt_key("owner", "target", "bad", operation="rename",
                                      payload={"title": "Wanted"}, expected_revision=0, expected_generation=None)
        db._conn.execute("INSERT INTO state_meta(key,value) VALUES (?,?)", (key, "{broken"))
        db._conn.commit()
        with pytest.raises(ValueError):
            mutate_runtime_session(db, **kwargs)
        assert db.get_session("target")["runtime_revision"] == 0
        assert db.get_session("target")["title"] is None
