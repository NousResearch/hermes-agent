"""Read-only idempotency-key run recovery API contracts."""

from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms import api_server_runs
from tests.gateway.test_api_server_runs import _make_adapter, _use_idempotency_db


def _create_runs_app(adapter):
    app = web.Application()
    for method, path, handler in api_server_runs._http_routes(adapter):
        if method == "GET" and path in (
            "/v1/runs/by-idempotency-key", "/v1/runs/{run_id}"
        ):
            app.router.add_get(path, handler)
    return app



@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


@pytest.mark.asyncio
async def test_lookup_by_key_is_authenticated_read_only_and_never_admits_a_run(
    auth_adapter, tmp_path
):
    adapter = auth_adapter
    _use_idempotency_db(adapter, tmp_path / "idem.db")
    scope_request = MagicMock()
    scope_request.headers = {}
    scope = adapter._run_idempotency_scope(scope_request)
    run_id = "run_" + "a" * 32
    store = adapter._run_idempotency_store
    store.reserve(scope, "known-key", "fingerprint", run_id,
                  {"object": "hermes.run", "run_id": run_id, "status": "completed"})
    app = _create_runs_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent") as create:
            unauthorized = await cli.get("/v1/runs/by-idempotency-key",
                                         headers={"Idempotency-Key": "known-key"})
            missing = await cli.get("/v1/runs/by-idempotency-key",
                                    headers={"Authorization": "Bearer sk-secret",
                                             "Idempotency-Key": "absent-key"})
            found = await cli.get("/v1/runs/by-idempotency-key",
                                  headers={"Authorization": "Bearer sk-secret",
                                           "Idempotency-Key": "known-key"})
            assert unauthorized.status == 401
            assert missing.status == 404
            assert missing.headers["Cache-Control"] == "no-store"
            assert found.status == 200
            assert await found.json() == {"run_id": run_id}
            create.assert_not_called()
    assert store.lookup(scope, "absent-key", "fingerprint") == ("missing", None)
    assert store.find_run_by_key(scope, "known-key") == run_id
    # A read-only lookup must not prune or renew terminal records.
    store._conn.execute(
        "UPDATE run_idempotency SET updated_at=? WHERE scope=? AND idempotency_key=?",
        (0, scope, "known-key"))
    store._conn.commit()
    assert store.find_run_by_key(scope, "known-key") is None
    assert store._conn.execute(
        "SELECT run_id FROM run_idempotency WHERE scope=? AND idempotency_key=?",
        (scope, "known-key")).fetchone() == (run_id,)

@pytest.mark.asyncio
async def test_lookup_survives_restart_but_is_scoped_and_requires_durable_store(
    auth_adapter, tmp_path
):
    from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
    db_path = tmp_path / "idem.db"
    _use_idempotency_db(auth_adapter, db_path)
    scope_request = MagicMock()
    scope_request.headers = {}
    scope = auth_adapter._run_idempotency_scope(scope_request)
    run_id = "run_" + "b" * 32
    auth_adapter._run_idempotency_store.reserve(
        scope, "recovery-key", "fingerprint", run_id,
        {"object": "hermes.run", "run_id": run_id, "status": "queued"})
    auth_adapter._run_idempotency_store.close()
    auth_adapter._run_idempotency_store = RunIdempotencyStore(str(db_path))
    app = _create_runs_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        headers = {"Authorization": "Bearer sk-secret", "Idempotency-Key": "recovery-key"}
        found = await cli.get("/v1/runs/by-idempotency-key", headers=headers)
        assert found.status == 200
        assert (await found.json())["run_id"] == run_id
        assert found.headers["Cache-Control"] == "no-store"
        no_key = await cli.get("/v1/runs/by-idempotency-key",
                               headers={"Authorization": "Bearer sk-secret"})
        assert no_key.status == 400
    foreign_adapter = _make_adapter(api_key="sk-other")
    _use_idempotency_db(foreign_adapter, db_path)
    foreign_app = _create_runs_app(foreign_adapter)
    async with TestClient(TestServer(foreign_app)) as cli:
        foreign = await cli.get("/v1/runs/by-idempotency-key",
                                headers={"Authorization": "Bearer sk-other",
                                         "Idempotency-Key": "recovery-key"})
        assert foreign.status == 404
    foreign_adapter._run_idempotency_store.close()
    auth_adapter._run_idempotency_store.close()
    auth_adapter._run_idempotency_store = RunIdempotencyStore(":memory:")
    memory_app = _create_runs_app(auth_adapter)
    async with TestClient(TestServer(memory_app)) as cli:
        unavailable = await cli.get("/v1/runs/by-idempotency-key", headers=headers)
        assert unavailable.status == 503

