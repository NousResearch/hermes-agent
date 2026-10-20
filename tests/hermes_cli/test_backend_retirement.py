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
