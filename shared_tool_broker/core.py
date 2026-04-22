from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Iterable


logger = logging.getLogger("shared_tool_broker")


def redact_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    lowered = value.lower()
    if any(token in lowered for token in ("bearer ", "authorization", "api_key", "token", "secret", "refresh_token", "access_token")):
        return "<redacted>"
    if len(value) > 24 and any(ch.isdigit() for ch in value) and any(ch.isalpha() for ch in value):
        return value[:4] + "***" + value[-4:]
    return value


def sanitize_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: redact_value(value) for key, value in payload.items()}


def provider_error_category(exc: Exception) -> str:
    text = str(exc).lower()
    if any(token in text for token in ("401", "403", "unauthorized", "forbidden", "invalid_grant", "oauth")):
        return "auth"
    if any(token in text for token in ("404", "not found")):
        return "not_found"
    if any(token in text for token in ("429", "rate limit")):
        return "rate_limit"
    if any(token in text for token in ("timeout", "timed out")):
        return "timeout"
    if any(token in text for token in ("schema", "validation", "invalid")):
        return "validation"
    return "provider"


class ToolExecutionError(RuntimeError):
    def __init__(self, message: str, *, category: str = "provider") -> None:
        super().__init__(message)
        self.category = category


class IdempotencyStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def get(self, scope: str, key: str) -> Any | None:
        with self._lock:
            return self._load().get(scope, {}).get(key)

    def put(self, scope: str, key: str, value: Any) -> None:
        with self._lock:
            payload = self._load()
            payload.setdefault(scope, {})[key] = value
            self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def normalize_json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return normalize_json(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(k): normalize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_json(item) for item in value]
    return value


def result_payload(*, ok: bool = True, data: Any = None, debug: Any = None, warnings: Iterable[str] | None = None, request_id: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": ok, "data": normalize_json(data)}
    if warnings:
        payload["warnings"] = list(warnings)
    if debug is not None:
        payload["debug"] = normalize_json(debug)
    if request_id:
        payload["request_id"] = request_id
    return payload


async def maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


async def run_logged_tool(tool_name: str, provider: str, func, *, principal: str = "unknown", request_id: str | None = None):
    request_id = request_id or str(uuid.uuid4())
    started = time.perf_counter()
    logger.info(
        "tool.start provider=%s tool=%s request_id=%s principal=%s",
        provider,
        tool_name,
        request_id,
        principal,
    )
    try:
        result = await maybe_await(func())
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.info(
            "tool.finish provider=%s tool=%s request_id=%s principal=%s status=ok latency_ms=%s",
            provider,
            tool_name,
            request_id,
            principal,
            elapsed_ms,
        )
        return result
    except ToolExecutionError as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.warning(
            "tool.finish provider=%s tool=%s request_id=%s principal=%s status=error latency_ms=%s category=%s error=%s",
            provider,
            tool_name,
            request_id,
            principal,
            elapsed_ms,
            exc.category,
            exc,
        )
        raise
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        category = provider_error_category(exc)
        logger.exception(
            "tool.finish provider=%s tool=%s request_id=%s principal=%s status=error latency_ms=%s category=%s",
            provider,
            tool_name,
            request_id,
            principal,
            elapsed_ms,
            category,
        )
        raise ToolExecutionError(str(exc), category=category) from exc
