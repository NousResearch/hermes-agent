"""Cooperative retirement is a reservation, never a sampled idle verdict."""

from fastapi.testclient import TestClient


def test_retirement_reserves_admission_until_cancel_or_permanent_commit(monkeypatch):
    from hermes_cli import web_server
    from tui_gateway import server
    from hermes_cli import backend_retirement

    monkeypatch.setattr(backend_retirement, "retirement", backend_retirement.RetirementFence())
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", "retirement-test-token")
    monkeypatch.setitem(server._methods, "test.retirement", lambda rid, params: server._ok(rid, {}))
    client = TestClient(web_server.app)
    headers = {"X-Hermes-Session-Token": "retirement-test-token"}
    rpc = {"id": "test", "method": "test.retirement", "params": {}}

    def post(action, token=None, authenticated=True):
        return client.post("/api/health/retirement", json={"action": action, **({"token": token} if token else {})},
                           headers=headers if authenticated else {})

    assert post("prepare", authenticated=False).status_code == 401
    for invalid in ({"action": []}, {"action": "commit"}, {"action": "cancel", "token": 123}, []):
        assert client.post("/api/health/retirement", json=invalid, headers=headers).status_code == 400
    assert client.post("/api/health/retirement", content="not json", headers=headers).status_code == 400
    assert "result" in server.handle_request(rpc)
    prepared = post("prepare")
    assert prepared.status_code == 200
    permit = prepared.json()
    assert permit == {"ok": True, "idle": True, "token": permit["token"]}
    assert isinstance(permit["token"], str) and permit["token"]
    # A diagnostic GET neither grants nor cancels the exclusive permit.
    assert client.get("/api/health/idle", headers=headers).json()["idle"] is True
    assert "error" in server.handle_request(rpc)
    assert post("prepare").json() == {"ok": False, "idle": False}
    assert post("cancel", "wrong-token").json() == {"ok": False}
    assert "error" in server.handle_request(rpc)
    assert post("cancel", permit["token"]).json() == {"ok": True}
    assert "result" in server.handle_request(rpc)
    next_permit = post("prepare").json()["token"]
    assert next_permit != permit["token"]
    assert post("commit", permit["token"]).json() == {"ok": False}
    assert post("commit", next_permit).json() == {"ok": True}
    assert post("commit", next_permit).json() == {"ok": True}
    assert post("cancel", next_permit).json() == {"ok": False}
    assert "error" in server.handle_request(rpc)
    # Recover a lost commit response without ever reopening the retired generation.
    assert post("prepare").json() == {"ok": True, "idle": True, "token": next_permit}


def test_only_uncommitted_permits_expire_and_unreadable_work_never_grants(monkeypatch):
    from hermes_cli.backend_retirement import RetirementFence
    from hermes_cli import web_server_idle_proof

    fence = RetirementFence()
    now = [100.0]
    monkeypatch.setattr(fence, "_now", lambda: now[0], raising=False)
    first = fence.prepare()["token"]
    now[0] += 30.0
    assert fence.commit(first) == {"ok": False}
    assert fence.cancel(first) == {"ok": False}
    with fence.work() as admitted:
        assert admitted
        assert fence.prepare() == {"ok": False, "idle": False}
    second = fence.prepare()["token"]
    assert first != second
    assert RetirementFence().commit(second) == {"ok": False}
    now[0] += 29.0
    assert fence.commit(second) == {"ok": True}
    now[0] += 3600.0
    assert fence.commit(second) == {"ok": True}
    assert fence.prepare() == {"ok": True, "idle": True, "token": second}
    assert fence.cancel(second) == {"ok": False}
    with fence.work() as admitted:
        assert not admitted
    monkeypatch.setattr(web_server_idle_proof, "idle_proof", lambda: {"idle": None})
    assert RetirementFence().prepare() == {"ok": False, "idle": None}


def test_prepare_fences_overlapping_new_admissions_while_allowing_reentrant_work(monkeypatch):
    import threading
    from hermes_cli.backend_retirement import RetirementFence
    from hermes_cli import web_server_idle_proof

    monkeypatch.setattr(web_server_idle_proof, "idle_proof", lambda: {"idle": True})
    fence = RetirementFence()
    entered_a = threading.Event()
    release_a = threading.Event()
    reentrant_results = []
    other_thread_results = []

    def worker_a():
        with fence.work() as admitted:
            assert admitted is True
            entered_a.set()
            assert release_a.wait(5.0)
            with fence.work() as nested:
                reentrant_results.append(nested)

    t_a = threading.Thread(target=worker_a)
    t_a.start()
    assert entered_a.wait(5.0)

    # First prepare while worker_a is active enters draining mode.
    assert fence.prepare() == {"ok": False, "idle": False}
    assert fence._draining is True

    # A new top-level turn on another thread is rejected while draining.
    def worker_b():
        with fence.work() as admitted:
            other_thread_results.append(admitted)

    t_b = threading.Thread(target=worker_b)
    t_b.start()
    t_b.join(5.0)
    assert other_thread_results == [False]

    # Finish worker_a (which also completes a nested re-entrant work() handoff on the same thread).
    release_a.set()
    t_a.join(5.0)
    assert reentrant_results == [True]
    assert fence.active_count() == 0
    assert fence._draining is False

    # Next prepare succeeds and issues a valid retirement token.
    permit = fence.prepare()
    assert permit["ok"] is True and permit["idle"] is True
    assert fence.cancel(permit["token"]) == {"ok": True}
    with fence.work() as admitted_after_cancel:
        assert admitted_after_cancel is True


def test_cross_thread_acquire_release_never_poisons_parent_thread_depth(monkeypatch):
    """Regression for #CF68 follow-up: _start_session_work and _pool.submit acquire on the caller
    thread and release on the spawned worker thread. With bind_thread=False + worker_thread_scope(),
    the parent thread never accumulates leaked _thread_depth, the worker thread gets re-entrant
    depth 1 while running, and _start_session_work(session={'running': True}) honors already-admitted
    turns even if prepare() set _draining in the millisecond gap between _lock_in_submit_turn and
    _start_session_work."""
    import threading
    from hermes_cli.backend_retirement import RetirementFence
    from hermes_cli import web_server_idle_proof

    monkeypatch.setattr(web_server_idle_proof, "idle_proof", lambda: {"idle": True})
    fence = RetirementFence()
    parent_tid = threading.get_ident()

    assert fence.acquire(bind_thread=False) is True
    assert fence._thread_depth.get(parent_tid, 0) == 0

    worker_nested = []

    def worker():
        try:
            with fence.worker_thread_scope():
                assert fence.prepare() == {"ok": False, "idle": False}
                assert fence._draining is True
                with fence.work() as nested:
                    worker_nested.append(nested)
        finally:
            fence.release(bind_thread=False)

    t = threading.Thread(target=worker)
    t.start()
    t.join(5.0)

    assert worker_nested == [True]
    assert fence.active_count() == 0
    assert fence._thread_depth == {}
    assert fence._draining is False

    # Verify force=True allows an already-admitted session turn (session["running"] == True)
    # to cross _start_session_work even while _draining is True from a concurrent turn.
    assert fence.acquire(bind_thread=False) is True
    assert fence.prepare() == {"ok": False, "idle": False}
    assert fence._draining is True
    assert fence.acquire(bind_thread=False, force=False) is False
    assert fence.acquire(bind_thread=False, force=True) is True
    fence.release(bind_thread=False)
    fence.release(bind_thread=False)
    assert fence.active_count() == 0
    assert fence._draining is False
