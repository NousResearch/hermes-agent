"""Read-only authenticated client for DanceOffice's beverage inventory REST endpoint."""
from __future__ import annotations

import json
from typing import Any

import httpx

ENDPOINT = "https://tanzschule-bietigheim.de/wp-json/tss/v1/beverage-inventory/latest"
DEFAULT_TIMEOUT_SECONDS = 15


class BeverageInventoryError(RuntimeError):
    """A safe, user-displayable error (never includes authorization material)."""


def _status_message(status: int) -> str:
    """Describe an HTTP failure without trusting or echoing the response body."""
    if status == 401:
        return "WordPress authentication failed (401); check the username and application password"
    if status == 403:
        return "WordPress authenticated the request but denied the inventory capability (403)"
    if status == 404:
        return "The WordPress inventory endpoint was not found (404)"
    if 500 <= status <= 599:
        return f"WordPress/server error ({status})"
    return f"WordPress request failed ({status})"


def fetch_latest_inventory(
    username: str,
    application_password: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    client: Any = httpx,
) -> dict[str, Any]:
    """Fetch inventory over HTTPS with WordPress Application Password Basic Auth.

    The endpoint is fixed rather than caller-controlled to keep credentials scoped to the
    DanceOffice origin and prevent redirect/SSRF credential leaks.
    """
    if not ENDPOINT.startswith("https://"):
        raise BeverageInventoryError("Refusing non-HTTPS WordPress endpoint")
    username = (username or "").strip()
    application_password = (application_password or "").strip()
    if not username or not application_password:
        raise BeverageInventoryError("Set HERMES_WP_API_USER and HERMES_WP_API_PASSWORD in Hermes secrets")

    try:
        response = client.get(
            ENDPOINT,
            auth=(username, application_password),
            headers={"Accept": "application/json", "User-Agent": "Hermes-Agent/wordpress-beverage-inventory"},
            timeout=timeout,
            follow_redirects=False,
        )
    except httpx.TimeoutException:
        raise BeverageInventoryError(f"WordPress request timed out after {timeout:g}s") from None
    except httpx.RequestError as exc:
        raise BeverageInventoryError(f"WordPress network error ({type(exc).__name__})") from None

    if response.status_code < 200 or response.status_code >= 300:
        raise BeverageInventoryError(_status_message(response.status_code))
    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        raise BeverageInventoryError("WordPress returned invalid JSON") from None
    if not isinstance(payload, dict):
        raise BeverageInventoryError("WordPress returned unexpected JSON (expected an object)")
    return payload


def register(ctx) -> None:
    """Register a profile-scoped, read-only tool using Hermes secret resolution."""
    from agent.secret_scope import get_secret

    def available() -> bool:
        return bool(
            (get_secret("HERMES_WP_API_USER", "") or "").strip()
            and (get_secret("HERMES_WP_API_PASSWORD", "") or "").strip()
        )

    def read_inventory(_params, **_kwargs) -> str:
        try:
            data = fetch_latest_inventory(
                get_secret("HERMES_WP_API_USER", "") or "",
                get_secret("HERMES_WP_API_PASSWORD", "") or "",
            )
            return json.dumps({"success": True, "data": data}, ensure_ascii=False)
        except BeverageInventoryError as exc:
            return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)

    ctx.register_tool(
        name="wordpress_beverage_inventory_latest",
        toolset="wordpress_beverage_inventory",
        schema={
            "name": "wordpress_beverage_inventory_latest",
            "description": "Read the latest DanceOffice beverage stocktake from WordPress using the configured read-only technical account. Does not modify inventory or place orders.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        handler=read_inventory,
        check_fn=available,
        requires_env=["HERMES_WP_API_USER", "HERMES_WP_API_PASSWORD"],
    )
