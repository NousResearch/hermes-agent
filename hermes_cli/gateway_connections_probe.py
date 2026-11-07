"""The Webapp's Test button for a saved gateway: the reachability probe Desktop's
``hermes:connections:test`` runs, from the serving gateway's side.

The HTTP leg is the one Desktop runs first: ``GET <url>/api/status`` with the credential the entry
saves (``X-Hermes-Session-Token`` for a token-mode remote, ``Authorization: Bearer`` for an
oauth-mode sign-in in ``gateway-connection-tokens.json``) plus its extra headers. ``local`` is the
serving gateway itself. SSH entries are reached through a tunnel this module does not open; the
SSH reach module answers them once it is present.
"""

from __future__ import annotations

from typing import Any

from hermes_cli import gateway_connections as store

_TIMEOUT_S = 8.0


def _credential_headers(entry: dict) -> dict[str, str]:
    secret = store.connection_secret(entry["id"])
    headers = dict(secret["headers"])
    if entry.get("authMode") == "oauth":
        tokens = store.read_tokens(entry["id"])
        if not tokens:
            raise store.ConnectionStoreError("This connection is not signed in. Sign in to it first.")
        headers["Authorization"] = f"Bearer {tokens['accessToken']}"
    elif secret["token"]:
        headers["X-Hermes-Session-Token"] = secret["token"]
    else:
        raise store.ConnectionStoreError(
            "This connection has no saved session token. Edit the connection and paste one.")
    return headers


async def _probe(base_url: str, headers: dict[str, str]) -> dict[str, Any]:
    import httpx

    from hermes_cli.web_routers.config_env import _endpoint_probe_client

    async with _endpoint_probe_client(base_url, _TIMEOUT_S) as client:
        try:
            resp = await client.get(f"{base_url}/api/status", headers=headers)
        except httpx.HTTPError as exc:
            raise store.ConnectionStoreError(f"Could not reach the gateway: {exc}") from exc
    if resp.status_code in (401, 403):
        raise store.ConnectionStoreError(f"The gateway refused the saved credential (HTTP {resp.status_code}).")
    if resp.status_code >= 400:
        raise store.ConnectionStoreError(f"The gateway answered HTTP {resp.status_code}.")
    try:
        body = resp.json()
    except ValueError:
        body = {}
    return {"ok": True, "baseUrl": base_url, "version": (body or {}).get("version") or None}


async def probe_connection(connection_id: str) -> dict[str, Any]:
    from fastapi import HTTPException

    try:
        entry = next((c for c in store.read_registry()["connections"] if c["id"] == connection_id), None)
        if entry is None:
            raise store.NoSuchConnection(f'No connection with id "{connection_id}".')
        if entry["kind"] == "local":
            from hermes_cli import __version__

            return {"ok": True, "baseUrl": None, "version": __version__}
        if entry["kind"] == "ssh":
            raise store.ConnectionStoreError("Testing an SSH connection from the Webapp is not available yet.")
        return await _probe(entry["url"], _credential_headers(entry))
    except store.NoSuchConnection as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except store.ConnectionStoreError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except store.ConnectionStoreBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
