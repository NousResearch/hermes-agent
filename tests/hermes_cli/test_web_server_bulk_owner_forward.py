"""Bulk/housekeeping dashboard routes must reach the live gateway owner (#138915).

The bulk delete, "Delete empty" and prune endpoints only had an offline path that takes the
profile's exclusive-maintenance lock, so with any gateway running they died on a bare 409. These
tests pin the forwarding layer (``_bulk_owner_rpc`` and its three semantic wrappers): a discovered
owner gets the verb over the control socket, an owner-less home returns ``None`` so the caller
keeps its offline path, and a transport failure after discovery is a retryable 503 — never the
bare 409. The owner-side verb itself has its own authority/store tests.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from hermes_cli import web_server_sessions


def _actor():
    return SimpleNamespace(subject='auth:v1:["oidc","https://idp.example","user-a"]')


def _discover(monkeypatch, endpoint, seen=None):
    def fake_discover(home, timeout=2.0):
        if seen is not None:
            seen["home"] = home
        return SimpleNamespace(
            state="absent" if endpoint is None else "ready",
            endpoint=endpoint,
            control_home=True,
            reason_code=None,
            detail=None,
        )

    monkeypatch.setattr(
        "hermes_cli.gateway_runtime.discover_gateway_endpoint", fake_discover
    )


def _recording_connect(monkeypatch, calls, reply=None):
    class FakeClient:
        async def rpc(self, method, **params):
            calls["rpc"] = (method, params)
            return reply if reply is not None else {"ok": True}

    @asynccontextmanager
    async def fake_connect(endpoint=None, subject=None):
        calls["endpoint"] = endpoint
        calls["subject"] = subject
        yield FakeClient()

    monkeypatch.setattr("hermes_cli.gateway_client.connect_gateway", fake_connect)


@pytest.mark.asyncio
async def test_bulk_delete_forwards_selection_to_the_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    seen, calls = {}, {}
    _discover(monkeypatch, "owner-endpoint", seen)
    _recording_connect(
        monkeypatch, calls, reply={"deleted": 2, "skipped_active": ["busy-1"]}
    )

    result = await web_server_sessions._bulk_delete_via_gateway_owner(
        None, _actor(), ["sess-1", "sess-2", "sess-1"]
    )

    assert result == {"deleted": 2, "skipped_active": ["busy-1"]}
    assert seen["home"] == tmp_path.resolve()
    assert calls["endpoint"] == "owner-endpoint"
    # The ticket names the dashboard login, not the socket UID (owner-side authorization depends on it).
    assert calls["subject"] == 'auth:v1:["oidc","https://idp.example","user-a"]'
    method, params = calls["rpc"]
    assert method == "session.bulk_delete"
    # De-duplicated selection, passed straight through.
    assert params == {"ids": ["sess-1", "sess-2", "sess-1"]}


@pytest.mark.asyncio
async def test_empty_sweep_forwards_the_empty_sweep_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls = {}
    _discover(monkeypatch, "owner-endpoint")
    _recording_connect(monkeypatch, calls, reply={"deleted": 3, "skipped_active": []})

    result = await web_server_sessions._delete_empty_via_gateway_owner(None, _actor())

    assert result == {"deleted": 3, "skipped_active": []}
    method, params = calls["rpc"]
    assert method == "session.bulk_delete"
    assert params == {"empty_sweep": True}


@pytest.mark.asyncio
async def test_prune_forwards_scalar_filters_to_the_prune_verb(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls = {}
    _discover(monkeypatch, "owner-endpoint")
    _recording_connect(monkeypatch, calls, reply={"deleted": 7})

    filters = {"older_than_days": 90, "source": "cli", "archived": False}
    result = await web_server_sessions._prune_via_gateway_owner(None, _actor(), filters)

    assert result == {"deleted": 7}
    method, params = calls["rpc"]
    assert method == "session.prune"
    assert params == {"filters": filters}


@pytest.mark.asyncio
async def test_no_owner_returns_none_so_the_caller_keeps_the_offline_path(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _discover(monkeypatch, None)
    # If this is ever called, the forward wrongly tried to dial.
    monkeypatch.setattr(
        "hermes_cli.gateway_client.connect_gateway", pytest.fail, raising=False
    )

    result = await web_server_sessions._bulk_delete_via_gateway_owner(
        None, _actor(), ["sess-1"]
    )
    assert result is None


@pytest.mark.asyncio
async def test_a_transport_failure_after_discovery_is_a_retryable_503(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _discover(monkeypatch, "owner-endpoint")

    @asynccontextmanager
    async def dying_connect(endpoint=None, subject=None):
        raise OSError("socket closed by peer")
        yield  # pragma: no cover - the context manager raises on entry

    monkeypatch.setattr("hermes_cli.gateway_client.connect_gateway", dying_connect)

    with pytest.raises(HTTPException) as exc:
        await web_server_sessions._bulk_delete_via_gateway_owner(
            None, _actor(), ["sess-1"]
        )
    assert exc.value.status_code == 503
    assert exc.value.detail == "owner_unreachable"


@pytest.mark.asyncio
async def test_the_owners_refusal_frame_passes_through_as_a_verdict(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _discover(monkeypatch, "owner-endpoint")
    from hermes_cli.gateway_client import GatewayRPCError

    @asynccontextmanager
    async def refusing_connect(endpoint=None, subject=None):
        class Refusing:
            async def rpc(self, method, **params):
                raise GatewayRPCError("permission_denied")

        yield Refusing()

    monkeypatch.setattr("hermes_cli.gateway_client.connect_gateway", refusing_connect)

    with pytest.raises(GatewayRPCError):
        await web_server_sessions._bulk_delete_via_gateway_owner(
            None, _actor(), ["sess-1"]
        )
