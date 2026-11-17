"""``GET /v1/schema``: a machine-readable description of the API server's surface.

Built at request time from the adapter's own route table and the closed error-code table, so
it cannot drift from what the listener actually serves.
"""

from __future__ import annotations

from typing import Any

from gateway.platforms.api_server_errors import ERROR_CODES, web

SCHEMA_VERSION = 1
_JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def error_object_schema() -> dict[str, Any]:
    """JSON Schema for the ``{"error": {...}}`` body every API server error carries."""
    nullable_str = {"type": ["string", "null"]}
    return {
        "$schema": _JSON_SCHEMA_DIALECT,
        "title": "Hermes API server error",
        "type": "object",
        "required": ["error"],
        "properties": {"error": {
            "type": "object",
            "required": ["message", "retryable", "retry_after_ms", "field", "request_id"],
            "properties": {
                "message": {"type": "string"},
                "type": {"type": "string", "description": "OpenAI-compatible error class."},
                "code": {**nullable_str, "description": "Stable machine code (see error_codes)."},
                "param": {**nullable_str, "description": "OpenAI-compatible offending input name."},
                "field": {**nullable_str, "description": "Offending input name (same as param)."},
                "retryable": {"type": "boolean",
                              "description": "Resending the same request can succeed."},
                "retry_after_ms": {"type": ["integer", "null"], "minimum": 0,
                                   "description": "Retry-After header in milliseconds."},
                "request_id": {"type": "string",
                               "description": "Echoes X-Request-Id (or a generated id)."}},
            "additionalProperties": True}},
    }


def schema_document(route_table: list[tuple]) -> dict[str, Any]:
    """The /v1/schema body: routes (method, path) plus the error schema and code table."""
    routes = sorted({(method, path) for method, path, _handler in route_table},
                    key=lambda row: (row[1], row[0]))
    return {
        "object": "hermes.api_server.schema",
        "version": SCHEMA_VERSION,
        "routes": [{"method": method, "path": path} for method, path in routes],
        "error": error_object_schema(),
        "error_codes": {code: {"status": entry.status, "retryable": entry.retryable}
                        for code, entry in sorted(ERROR_CODES.items())},
    }


def schema_route(adapter: Any) -> tuple[str, str, Any]:
    """The ``GET /v1/schema`` row for ``APIServerAdapter._http_route_table`` (Bearer auth)."""
    async def handler(request: web.Request) -> web.Response:
        auth_err = adapter._check_auth(request)
        if auth_err:
            return auth_err
        return web.json_response(schema_document(adapter._http_route_table()))
    return "GET", "/v1/schema", handler
