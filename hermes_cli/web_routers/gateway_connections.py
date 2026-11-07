"""``/api/gateway-connections``: the Webapp's door to this machine's saved gateways list.

Thin HTTP over :mod:`hermes_cli.gateway_connections`, answering in the shapes native Desktop's
``hermesDesktop.connections`` IPC answers (``{ok, registry}``, plus ``connection`` on save), so the
browser bridge is a transport swap. Mounted only where the surface policy names
``gateway_connections`` (the Webapp) and, like every ``/api/`` route, only past the sign-in gate.
Token and header values never leave the server: the registry is :func:`public_registry`'s view.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Body, HTTPException

from hermes_cli import gateway_connections as store

router = APIRouter()

_STATUS = (
    (store.NoSuchConnection, 404),
    (store.ConnectionStoreError, 400),
    (store.ConnectionStoreBusy, 503),
)


def _call(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as exc:
        for kind, status in _STATUS:
            if isinstance(exc, kind):
                raise HTTPException(status_code=status, detail=str(exc)) from exc
        raise


def _answer(registry: dict, **extra: Any) -> dict:
    return {"ok": True, **extra, "registry": store.public_registry(registry)}


@router.get("/api/gateway-connections")
def list_connections() -> dict:
    return store.public_registry(_call(store.read_registry))


def _dial(registry: dict, connection_id: object) -> object:
    entry = next((c for c in registry["connections"] if c["id"] == connection_id), None)
    return entry and {k: entry.get(k) for k in ("kind", "url", "authMode", "token", "headers")}


@router.post("/api/gateway-connections")
def save_connection(payload: dict = Body(...)) -> dict:
    from hermes_cli.web_routers.webapp_relay_proxy import close_gateway_sockets

    before = _dial(_call(store.read_registry), payload.get("id")) if payload.get("id") else None
    connection, registry = _call(lambda: store.save_connection(payload))
    # An edit that moves the dial ends the Webapp's relayed sockets to the old target.
    if before is not None and before != _dial(_call(store.read_registry), connection["id"]):
        close_gateway_sockets(connection["id"])
    return _answer(registry, connection=connection)


@router.delete("/api/gateway-connections/{connection_id}")
def remove_connection(connection_id: str) -> dict:
    from hermes_cli.web_routers.webapp_relay_proxy import close_gateway_sockets

    from hermes_cli.gateway_connections_reach import forget

    registry = _call(lambda: store.remove_connection(connection_id))
    close_gateway_sockets(connection_id)
    forget(connection_id)
    return _answer(registry)


@router.post("/api/gateway-connections/launch-mode")
def set_launch_mode(body: dict = Body(...)) -> dict:
    return _answer(_call(lambda: store.set_launch_mode(str(body.get("mode") or ""))))


@router.post("/api/gateway-connections/last-used")
def set_last_used(body: dict = Body(...)) -> dict:
    return _answer(_call(lambda: store.set_last_used(str(body.get("id") or ""))))


@router.post("/api/gateway-connections/{connection_id}/primary")
def set_primary(connection_id: str) -> dict:
    return _answer(_call(lambda: store.set_primary(connection_id)))


@router.post("/api/gateway-connections/{connection_id}/test")
async def test_connection(connection_id: str) -> dict:
    from hermes_cli.gateway_connections_probe import probe_connection

    return await probe_connection(connection_id)
