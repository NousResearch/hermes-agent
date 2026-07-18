"""A deliberately narrow HTTP adapter for kanban claim handoffs."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import json
import os
import secrets
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from hermes_cli import kanban_db


MAX_BODY_BYTES = 4_096
MAX_TASK_ID_CHARS = 128


@dataclass(frozen=True)
class ClaimShimConfig:
    """Fixed server-side identity and claim lifetime for one shim instance."""

    bearer_token: str
    claimer: str
    ttl_seconds: int


def load_server_config() -> tuple[ClaimShimConfig, str, int]:
    """Read the deliberate, private-only listener configuration from env."""
    bearer_token = os.environ.get("HERMES_CLAIM_SHIM_BEARER", "")
    if not bearer_token:
        raise ValueError("HERMES_CLAIM_SHIM_BEARER bearer secret is required")
    host = os.environ.get("HERMES_CLAIM_SHIM_BIND", "")
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("claim-shim host must be an explicit Tailnet IP") from exc
    tailnet_ranges = (
        ipaddress.ip_network("100.64.0.0/10"),
        ipaddress.ip_network("fd7a:115c:a1e0::/48"),
    )
    if not any(address in network for network in tailnet_ranges):
        raise ValueError("claim-shim host must be a non-wildcard Tailnet IP")
    claimer = os.environ.get("HERMES_CLAIM_SHIM_CLAIMER", "")
    if not claimer or len(claimer) > 128:
        raise ValueError("HERMES_CLAIM_SHIM_CLAIMER is required and bounded")
    try:
        ttl_seconds = int(os.environ.get("HERMES_CLAIM_SHIM_TTL_SECONDS", ""))
    except ValueError as exc:
        raise ValueError("HERMES_CLAIM_SHIM_TTL_SECONDS must be an integer") from exc
    if not 1 <= ttl_seconds <= 86_400:
        raise ValueError("HERMES_CLAIM_SHIM_TTL_SECONDS must be between 1 and 86400")
    try:
        port = int(os.environ.get("HERMES_CLAIM_SHIM_PORT", "8787"))
    except ValueError as exc:
        raise ValueError("HERMES_CLAIM_SHIM_PORT must be an integer") from exc
    if not 1 <= port <= 65_535:
        raise ValueError("HERMES_CLAIM_SHIM_PORT must be between 1 and 65535")
    return ClaimShimConfig(bearer_token, claimer, ttl_seconds), host, port


def create_app(config: ClaimShimConfig) -> FastAPI:
    """Create the non-discoverable, narrowly-routed claim-shim application."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def authorized(authorization: str | None) -> bool:
        expected = f"Bearer {config.bearer_token}"
        return authorization is not None and secrets.compare_digest(authorization, expected)

    async def read_json(request: Request, allowed_keys: frozenset[str]) -> tuple[dict[str, Any] | None, JSONResponse | None]:
        content_type = request.headers.get("content-type", "")
        if not content_type.lower().startswith("application/json"):
            return None, JSONResponse(status_code=415, content={"error": "unsupported_media_type"})
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > MAX_BODY_BYTES:
                    return None, JSONResponse(status_code=413, content={"error": "body_too_large"})
            except ValueError:
                return None, JSONResponse(status_code=400, content={"error": "invalid_json"})
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            return None, JSONResponse(status_code=413, content={"error": "body_too_large"})
        try:
            payload = json.loads(body)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
            return None, JSONResponse(status_code=400, content={"error": "invalid_json"})
        if not isinstance(payload, dict) or set(payload) != allowed_keys:
            return None, JSONResponse(status_code=400, content={"error": "invalid_request"})
        task_id = payload.get("task_id")
        if not isinstance(task_id, str) or not task_id or len(task_id) > MAX_TASK_ID_CHARS:
            return None, JSONResponse(status_code=400, content={"error": "invalid_request"})
        return payload, None

    def valid_github_pull_url(value: object) -> bool:
        if not isinstance(value, str) or not value or len(value) > 2_048:
            return False
        parsed = urlparse(value)
        parts = parsed.path.split("/")
        return (
            parsed.scheme == "https"
            and parsed.netloc == "github.com"
            and not parsed.params
            and not parsed.query
            and not parsed.fragment
            and len(parts) == 5
            and all(parts[index] for index in (1, 2, 4))
            and parts[3] == "pull"
            and parts[4].isdigit()
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/claim")
    async def claim(
        request: Request, authorization: str | None = Header(default=None)
    ) -> JSONResponse:
        if not authorized(authorization):
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
        payload, error = await read_json(request, frozenset({"task_id"}))
        if error is not None:
            return error
        assert payload is not None
        with kanban_db.connect_closing() as conn:
            task = kanban_db.claim_task(
                conn,
                payload["task_id"],
                claimer=config.claimer,
                ttl_seconds=config.ttl_seconds,
            )
            existing = None if task is not None else kanban_db.get_task(conn, payload["task_id"])
        if task is not None:
            return JSONResponse(
                status_code=200,
                content={"outcome": "claimed", "task_id": task.id},
            )
        if existing is None:
            return JSONResponse(status_code=404, content={"error": "not_found"})
        if existing.status == "running" and existing.claim_lock == config.claimer:
            return JSONResponse(
                status_code=200,
                content={"outcome": "already_owned", "task_id": existing.id},
            )
        return JSONResponse(status_code=409, content={"error": "claim_conflict"})

    @app.post("/to-review")
    async def to_review(
        request: Request, authorization: str | None = Header(default=None)
    ) -> JSONResponse:
        if not authorized(authorization):
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
        payload, error = await read_json(request, frozenset({"task_id", "pr_url"}))
        if error is not None:
            return error
        assert payload is not None
        pr_url = payload["pr_url"]
        if not valid_github_pull_url(pr_url):
            return JSONResponse(status_code=400, content={"error": "invalid_request"})
        with kanban_db.connect_closing() as conn:
            before = kanban_db.get_task(conn, payload["task_id"])
            task = kanban_db.to_review_task(
                conn,
                payload["task_id"],
                pr_url=pr_url,
                claimer=config.claimer,
            )
            existing = None if task is not None else kanban_db.get_task(conn, payload["task_id"])
        if task is not None:
            outcome = (
                "already_in_review"
                if before is not None and before.status == "review"
                else "review_started"
            )
            return JSONResponse(
                status_code=200,
                content={"outcome": outcome, "task_id": task.id},
            )
        if existing is None:
            return JSONResponse(status_code=404, content={"error": "not_found"})
        if existing.status == "review":
            if existing.result == pr_url:
                return JSONResponse(
                    status_code=200,
                    content={"outcome": "already_in_review", "task_id": existing.id},
                )
            return JSONResponse(status_code=409, content={"error": "review_conflict"})
        if existing.status == "running" and existing.claim_lock != config.claimer:
            return JSONResponse(status_code=409, content={"error": "ownership_conflict"})
        return JSONResponse(status_code=409, content={"error": "review_conflict"})

    return app


def main() -> None:
    """Run the shim only after private-listener configuration has validated."""
    config, host, port = load_server_config()
    import uvicorn

    uvicorn.run(create_app(config), host=host, port=port)


if __name__ == "__main__":
    main()
