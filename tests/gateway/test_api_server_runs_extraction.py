"""Compatibility seams for the extracted ``/v1/runs`` lifecycle."""

import inspect

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms import api_server
from gateway.platforms import api_server_room_grants
from gateway.platforms import api_server_runs


_HTTP_HANDLER_DELEGATES = (
    ("_handle_get_run", "_handle_get_run"),
    ("_handle_run_events", "_handle_run_events"),
    ("_handle_run_approval", "_handle_run_approval"),
    ("_handle_steer_run", "_handle_steer_run"),
    ("_handle_stop_run", "_handle_stop_run"),
)

_RUN_METHODS = {
    "_set_run_status",
    "_make_run_event_callback",
    "_run_idempotency_scope",
    "_check_run_auth",
    "_durable_run_status",
    "_handle_runs",
    "_request_owns_run",
    *(adapter_name for adapter_name, _ in _HTTP_HANDLER_DELEGATES),
    "_sweep_orphaned_runs",
    "_sweep_orphaned_runs_once",
}


def test_api_server_keeps_run_methods_on_the_adapter_class():
    assert _RUN_METHODS <= api_server.APIServerAdapter.__dict__.keys()


@pytest.mark.asyncio
@pytest.mark.parametrize(("adapter_name", "implementation_name"), _HTTP_HANDLER_DELEGATES)
async def test_run_http_handlers_delegate_without_changing_method_surface(
    monkeypatch, adapter_name, implementation_name
):
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    request = object()
    expected = object()
    implementation = AsyncMock(return_value=expected)
    monkeypatch.setattr(api_server_runs, implementation_name, implementation)

    assert await getattr(adapter, adapter_name)(request) is expected
    implementation.assert_awaited_once_with(
        adapter,
        request,
        _api_server=sys.modules[api_server.__name__],
    )


@pytest.mark.asyncio
async def test_decorated_run_admission_delegates_and_releases_slot(monkeypatch):
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    adapter._room_grant_token = MagicMock(return_value="room-grant")
    adapter._check_run_auth = MagicMock(return_value=None)
    adapter._draining_response = MagicMock(return_value=None)
    adapter._pending_agent_requests = 0
    request = SimpleNamespace(path="/v1/runs")
    expected = object()
    implementation = AsyncMock(return_value=expected)
    monkeypatch.setattr(api_server_runs, "_handle_runs", implementation)

    assert await adapter._handle_runs(request) is expected
    implementation.assert_awaited_once_with(
        adapter,
        request,
        _api_server=sys.modules[api_server.__name__],
    )
    adapter._check_run_auth.assert_called_once_with(request, permission="dispatch")
    assert adapter._pending_agent_requests == 0


def test_run_status_method_delegates(monkeypatch):
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    expected = {"run_id": "run-1", "status": "running"}
    implementation = MagicMock(return_value=expected)
    monkeypatch.setattr(api_server_runs, "_set_run_status", implementation)

    assert adapter._set_run_status("run-1", "running", last_event="run.started") is expected
    implementation.assert_called_once_with(
        adapter,
        "run-1",
        "running",
        last_event="run.started",
    )


def test_room_scoped_run_policy_delegates_with_legacy_api_bindings(monkeypatch):
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    request = object()
    scope = MagicMock(return_value="scope-hash")
    auth = MagicMock(return_value=None)
    monkeypatch.setattr(api_server_runs, "_run_idempotency_scope", scope)
    monkeypatch.setattr(api_server_runs, "_check_run_auth", auth)

    assert adapter._run_idempotency_scope(request) == "scope-hash"
    assert adapter._check_run_auth(request, permission="stop") is None
    scope.assert_called_once_with(
        adapter,
        request,
        _api_server=sys.modules[api_server.__name__],
    )
    auth.assert_called_once_with(
        adapter,
        request,
        permission="stop",
        _api_server=sys.modules[api_server.__name__],
    )


def test_run_state_initialization_and_teardown_are_shard_owned():
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    store = MagicMock()
    store_factory = MagicMock(return_value=store)

    api_server_runs._initialize_run_state(
        adapter,
        store_factory=store_factory,
    )

    store_factory.assert_called_once_with()
    assert adapter._run_idempotency_store is store
    assert adapter._run_idempotency_ids == set()
    assert adapter._run_owners == {}
    assert adapter._run_streams == {}
    assert adapter._run_streams_created == {}
    assert adapter._run_stream_subscribers == set()
    assert adapter._active_run_agents == {}
    assert adapter._active_run_tasks == {}
    assert adapter._stopping_run_ids == set()
    assert adapter._run_statuses == {}
    assert adapter._run_approval_sessions == {}

    api_server_runs._close_run_state(adapter)
    store.close.assert_called_once_with()


def test_run_capability_metadata_is_shard_owned():
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    adapter._run_idempotency_store = SimpleNamespace(durable=True)
    store_type = SimpleNamespace(RETENTION_SECONDS=123)

    assert api_server_runs._idempotency_capabilities(
        adapter,
        store_type=store_type,
    ) == {
        "supported": True,
        "durable": True,
        "retention_seconds": 123,
    }


def test_roomlink_and_run_route_tuples_are_shard_owned():
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)

    room_routes = api_server_room_grants._http_routes(adapter)
    run_routes = api_server_runs._http_routes(adapter)

    assert [(method, path) for method, path, _ in room_routes] == [
        ("POST", "/v1/room-members/invitations"),
        ("GET", "/v1/room-members/capabilities"),
        ("POST", "/v1/room-members/grants/refresh"),
        ("POST", "/v1/room-members/grants/revoke"),
        ("GET", "/v1/room-controls/{room_id}"),
        ("POST", "/v1/room-controls/{room_id}"),
        ("DELETE", "/v1/room-controls/{room_id}"),
    ]
    assert [(method, path) for method, path, _ in run_routes] == [
        ("POST", "/v1/runs"),
        ("GET", "/v1/runs/{run_id}"),
        ("GET", "/v1/runs/{run_id}/events"),
        ("POST", "/v1/runs/{run_id}/approval"),
        ("POST", "/v1/runs/{run_id}/steer"),
        ("POST", "/v1/runs/{run_id}/stop"),
    ]
    assert all(handler.__self__ is adapter for _, _, handler in room_routes[:4])
    assert all(
        inspect.getclosurevars(handler).nonlocals.get("self") is adapter
        for _, _, handler in room_routes[4:]
    )
    assert all(handler.__self__ is adapter for _, _, handler in run_routes)
