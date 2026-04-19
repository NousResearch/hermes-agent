"""Zectrix Note 4 MCP server.

This server wraps the Zectrix cloud API behind Model Context Protocol tools so
Hermes Agent (and any other MCP client) can manage Note 4 devices, todos, and
screen pushes.

Environment variables:
    ZECTRIX_API_KEY            Required. API key sent as X-API-Key.
    ZECTRIX_API_BASE_URL       Optional. Default: https://cloud.zectrix.com
    ZECTRIX_DEFAULT_DEVICE_ID   Optional. Default device to use when omitted.
    ZECTRIX_TIMEOUT_SECONDS     Optional. Default: 20
    ZECTRIX_RETRY_ATTEMPTS      Optional. Default: 3

Run directly:
    python -m integrations.zectrix_note_mcp
    # or, after adding the console script entry:
    zectrix-note-mcp

Hermes config example:
    mcp_servers:
      zectrix_note:
        command: "zectrix-note-mcp"
        env:
          ZECTRIX_API_KEY: "zt_xxx"
          ZECTRIX_DEFAULT_DEVICE_ID: "AA:BB:CC:DD:EE:FF"
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import httpx

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - allows import without optional extra
    FastMCP = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("ZECTRIX_API_BASE_URL", "https://cloud.zectrix.com")
API_KEY = os.getenv("ZECTRIX_API_KEY", "").strip()
DEFAULT_DEVICE_ID = os.getenv("ZECTRIX_DEFAULT_DEVICE_ID", "").strip() or None
TIMEOUT_SECONDS = float(os.getenv("ZECTRIX_TIMEOUT_SECONDS", "20"))
RETRY_ATTEMPTS = max(1, int(os.getenv("ZECTRIX_RETRY_ATTEMPTS", "3")))


class _FallbackFastMCP:
    """Import-safe fallback so helper functions can still be unit-tested."""

    def tool(self, func=None, *args, **kwargs):
        if callable(func):
            return func

        def decorator(inner):
            return inner

        return decorator

    def run(self, *args, **kwargs):
        raise RuntimeError(
            "mcp package is not installed. Install Hermes with the [mcp] extra "
            "or run: pip install mcp"
        )


mcp = FastMCP("zectrix-note") if FastMCP is not None else _FallbackFastMCP()


def _base_api_url() -> str:
    return API_BASE_URL.rstrip("/")


def _headers() -> dict[str, str]:
    if not API_KEY:
        raise RuntimeError(
            "ZECTRIX_API_KEY is not set. Export it in the environment before starting the MCP server."
        )
    return {
        "Accept": "application/json",
        "X-API-Key": API_KEY,
    }


def _join_url(path: str) -> str:
    return f"{_base_api_url()}/{path.lstrip('/')}"


def _unwrap_response(payload: Any) -> Any:
    if isinstance(payload, dict) and "code" in payload:
        code = payload.get("code")
        if code not in (0, "0", None, "success"):
            message = payload.get("msg") or payload.get("message") or "unknown error"
            raise RuntimeError(f"Zectrix API error {code}: {message}")
        return payload.get("data", payload)
    return payload


def _api_request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: Any | None = None,
    files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
    data: dict[str, Any] | None = None,
) -> Any:
    headers = _headers()
    url = _join_url(path)
    last_error: Exception | None = None

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            with httpx.Client(timeout=TIMEOUT_SECONDS, headers=headers) as client:
                response = client.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    files=files,
                    data=data,
                )
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    payload = response.json()
                else:
                    text = response.text.strip()
                    try:
                        payload = json.loads(text)
                    except Exception:
                        payload = {"text": text}
                return _unwrap_response(payload)
        except (httpx.RequestError, httpx.HTTPStatusError, RuntimeError) as exc:
            last_error = exc
            if attempt >= RETRY_ATTEMPTS:
                break
            sleep_seconds = min(2 ** (attempt - 1), 8)
            time.sleep(sleep_seconds)

    assert last_error is not None
    raise RuntimeError(f"Failed to call Zectrix API {method} {path}: {last_error}") from last_error


def _list_devices_raw() -> list[dict[str, Any]]:
    payload = _api_request("GET", "/open/v1/devices")
    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _resolve_device_id(device_id: str | None) -> str:
    if device_id and device_id.strip():
        return device_id.strip()
    if DEFAULT_DEVICE_ID:
        return DEFAULT_DEVICE_ID

    devices = _list_devices_raw()
    if len(devices) == 1:
        candidate = devices[0].get("deviceId") or devices[0].get("device_id") or devices[0].get("id")
        if candidate:
            return str(candidate)

    if not devices:
        raise RuntimeError(
            "No device_id was provided and the account has no discoverable devices. "
            "Set ZECTRIX_DEFAULT_DEVICE_ID or pass device_id explicitly."
        )

    raise RuntimeError(
        "No device_id was provided and multiple devices exist. "
        "Set ZECTRIX_DEFAULT_DEVICE_ID or pass device_id explicitly."
    )


def _coerce_optional_int(value: int | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _multipart_images(image_paths: list[str]) -> list[tuple[str, tuple[str, bytes, str]]]:
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    for raw_path in image_paths:
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        suffix = path.suffix.lower()
        mime = mimetypes.types_map.get(suffix, "application/octet-stream")
        files.append(("images", (path.name, path.read_bytes(), mime)))
    if len(files) > 5:
        raise ValueError("At most 5 images can be uploaded at once.")
    return files


@mcp.tool()
def health_check() -> dict[str, Any]:
    """Validate API connectivity and return a minimal account/device summary."""
    devices = _list_devices_raw()
    return {
        "ok": True,
        "api_base_url": _base_api_url(),
        "device_count": len(devices),
        "default_device_id": DEFAULT_DEVICE_ID,
    }


@mcp.tool()
def list_devices() -> dict[str, Any]:
    """List Note 4 devices that are available to the current API key."""
    devices = _list_devices_raw()
    return {
        "devices": devices,
        "count": len(devices),
        "default_device_id": DEFAULT_DEVICE_ID,
    }


@mcp.tool()
def list_todos(
    status: int | None = None,
    device_id: str | None = None,
) -> dict[str, Any]:
    """List todos. status: 0=pending, 1=completed."""
    params: dict[str, Any] = {}
    if status is not None:
        params["status"] = int(status)
    if device_id:
        params["deviceId"] = device_id
    payload = _api_request("GET", "/open/v1/todos", params=params or None)
    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]
    return {"todos": payload if isinstance(payload, list) else [], "raw": payload}


@mcp.tool()
def create_todo(
    title: str,
    description: str = "",
    due_date: str | None = None,
    due_time: str | None = None,
    repeat_type: str = "none",
    repeat_weekday: int | None = None,
    repeat_month: int | None = None,
    repeat_day: int | None = None,
    priority: int = 1,
    device_id: str | None = None,
) -> dict[str, Any]:
    """Create a todo on the cloud service."""
    payload: dict[str, Any] = {
        "title": title,
        "description": description,
        "repeatType": repeat_type,
        "priority": int(priority),
    }
    if due_date:
        payload["dueDate"] = due_date
    if due_time:
        payload["dueTime"] = due_time
    if repeat_weekday is not None:
        payload["repeatWeekday"] = int(repeat_weekday)
    if repeat_month is not None:
        payload["repeatMonth"] = int(repeat_month)
    if repeat_day is not None:
        payload["repeatDay"] = int(repeat_day)
    if device_id:
        payload["deviceId"] = device_id
    elif DEFAULT_DEVICE_ID:
        payload["deviceId"] = DEFAULT_DEVICE_ID

    data = _api_request("POST", "/open/v1/todos", json_body=payload)
    return {"todo": data}


@mcp.tool()
def update_todo(
    todo_id: int,
    title: str | None = None,
    description: str | None = None,
    due_date: str | None = None,
    due_time: str | None = None,
    priority: int | None = None,
) -> dict[str, Any]:
    """Update an existing todo."""
    payload: dict[str, Any] = {}
    if title is not None:
        payload["title"] = title
    if description is not None:
        payload["description"] = description
    if due_date is not None:
        payload["dueDate"] = due_date
    if due_time is not None:
        payload["dueTime"] = due_time
    if priority is not None:
        payload["priority"] = int(priority)

    data = _api_request("PUT", f"/open/v1/todos/{int(todo_id)}", json_body=payload)
    return {"todo": data}


@mcp.tool()
def toggle_todo_complete(todo_id: int) -> dict[str, Any]:
    """Toggle a todo between pending and completed state."""
    data = _api_request("PUT", f"/open/v1/todos/{int(todo_id)}/complete")
    return {"result": data}


@mcp.tool()
def delete_todo(todo_id: int) -> dict[str, Any]:
    """Delete a todo permanently."""
    data = _api_request("DELETE", f"/open/v1/todos/{int(todo_id)}")
    return {"result": data}


@mcp.tool()
def push_text(
    text: str,
    device_id: str | None = None,
    font_size: int = 20,
    page_id: str | None = None,
) -> dict[str, Any]:
    """Push plain text to a device screen."""
    resolved_device = _resolve_device_id(device_id)
    payload: dict[str, Any] = {"text": text, "fontSize": int(font_size)}
    if page_id:
        payload["pageId"] = page_id
    data = _api_request(
        "POST",
        f"/open/v1/devices/{resolved_device}/display/text",
        json_body=payload,
    )
    return {"device_id": resolved_device, "result": data}


@mcp.tool()
def push_structured_text(
    title: str | None = None,
    body: str | None = None,
    device_id: str | None = None,
    page_id: str | None = None,
) -> dict[str, Any]:
    """Push a title/body text card to a device screen."""
    if not title and not body:
        raise ValueError("At least one of title or body must be provided.")
    resolved_device = _resolve_device_id(device_id)
    payload: dict[str, Any] = {}
    if title:
        payload["title"] = title
    if body:
        payload["body"] = body
    if page_id:
        payload["pageId"] = page_id
    data = _api_request(
        "POST",
        f"/open/v1/devices/{resolved_device}/display/structured-text",
        json_body=payload,
    )
    return {"device_id": resolved_device, "result": data}


@mcp.tool()
def push_image(
    image_path: str | list[str],
    device_id: str | None = None,
    dither: bool = True,
    page_id: str | None = None,
) -> dict[str, Any]:
    """Push one or more local image files to the device screen."""
    resolved_device = _resolve_device_id(device_id)
    paths = [image_path] if isinstance(image_path, str) else list(image_path)
    files = _multipart_images(paths)
    data_fields: dict[str, Any] = {"dither": str(bool(dither)).lower()}
    if page_id:
        data_fields["pageId"] = page_id
    data = _api_request(
        "POST",
        f"/open/v1/devices/{resolved_device}/display/image",
        files=files,
        data=data_fields,
    )
    return {"device_id": resolved_device, "result": data, "images": paths}


@mcp.tool()
def clear_pages(device_id: str | None = None, page_id: str | None = None) -> dict[str, Any]:
    """Delete one page or clear all pages from a device."""
    resolved_device = _resolve_device_id(device_id)
    path = f"/open/v1/devices/{resolved_device}/display/pages"
    if page_id:
        path += f"/{page_id}"
    data = _api_request("DELETE", path)
    return {"device_id": resolved_device, "result": data, "page_id": page_id}


def main() -> None:
    """CLI entrypoint for stdio MCP mode."""
    import argparse

    parser = argparse.ArgumentParser(description="Zectrix Note 4 MCP server")
    parser.add_argument("--log-level", default=os.getenv("ZECTRIX_LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if FastMCP is None:
        raise RuntimeError(
            "The 'mcp' dependency is not installed. Install Hermes with the [mcp] extra or run 'pip install mcp'."
        )

    mcp.run()


if __name__ == "__main__":
    main()
