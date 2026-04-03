"""HTTP client + config resolution for the native RetainDB integration."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.retaindb.com"
DEFAULT_PREFETCH_TIMEOUT_MS = 1500
DEFAULT_CONTEXT_TOKENS = 1200
DEFAULT_FLUSH_BATCH_SIZE = 50
_VALID_MEMORY_MODES = {"hybrid", "retaindb", "local"}
_VALID_RECALL_MODES = {"hybrid", "context", "tools"}


def _get_hermes_home() -> Path:
    return Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))


def normalize_base_url(url: str | None) -> str:
    value = (url or DEFAULT_BASE_URL).strip()
    if not value:
        value = DEFAULT_BASE_URL
    value = re.sub(r"/+$", "", value)
    value = re.sub(r"/api/v1$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"/v1$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"/api$", "", value, flags=re.IGNORECASE)
    return value


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class RetainDBClientError(RuntimeError):
    """Raised when RetainDB returns a non-success response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        payload: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


@dataclass
class RetainDBClientConfig:
    """Resolved config for Hermes' native RetainDB integration."""

    enabled: bool = True
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    project: str | None = None
    memory_mode: str = "hybrid"
    recall_mode: str = "hybrid"
    write_frequency: str | int = "async"
    context_tokens: int = DEFAULT_CONTEXT_TOKENS
    prefetch_timeout_ms: int = DEFAULT_PREFETCH_TIMEOUT_MS
    flush_batch_size: int = DEFAULT_FLUSH_BATCH_SIZE
    disable_tool_exposure: bool = False
    debug_recall_trace: bool = False
    user_id_override: str | None = None
    agent_id: str = "hermes"

    @classmethod
    def from_global_config(cls) -> "RetainDBClientConfig":
        """Resolve RetainDB settings from Hermes config.yaml + ~/.hermes/.env."""
        try:
            from hermes_cli.config import get_env_value, load_config

            cfg = load_config() or {}
            raw = cfg.get("retaindb", {}) if isinstance(cfg, dict) else {}
            env_api_key = get_env_value("RETAINDB_API_KEY")
            env_base_url = get_env_value("RETAINDB_BASE_URL")
            env_project = get_env_value("RETAINDB_PROJECT")
        except Exception:
            raw = {}
            env_api_key = os.getenv("RETAINDB_API_KEY")
            env_base_url = os.getenv("RETAINDB_BASE_URL")
            env_project = os.getenv("RETAINDB_PROJECT")

        write_frequency = raw.get("write_frequency", "async")
        try:
            write_frequency = int(write_frequency)
        except (TypeError, ValueError):
            write_frequency = str(write_frequency or "async").strip().lower() or "async"

        memory_mode = str(raw.get("memory_mode", "hybrid") or "hybrid").strip().lower()
        if memory_mode not in _VALID_MEMORY_MODES:
            memory_mode = "hybrid"

        recall_mode = str(raw.get("recall_mode", "hybrid") or "hybrid").strip().lower()
        if recall_mode not in _VALID_RECALL_MODES:
            recall_mode = "hybrid"

        return cls(
            enabled=_coerce_bool(raw.get("enabled"), True),
            api_key=(env_api_key or "").strip() or None,
            base_url=normalize_base_url(env_base_url or raw.get("base_url") or DEFAULT_BASE_URL),
            project=(str(env_project or raw.get("project") or "").strip() or None),
            memory_mode=memory_mode,
            recall_mode=recall_mode,
            write_frequency=write_frequency,
            context_tokens=max(200, _coerce_int(raw.get("context_tokens"), DEFAULT_CONTEXT_TOKENS)),
            prefetch_timeout_ms=max(100, _coerce_int(raw.get("prefetch_timeout_ms"), DEFAULT_PREFETCH_TIMEOUT_MS)),
            flush_batch_size=max(1, _coerce_int(raw.get("flush_batch_size"), DEFAULT_FLUSH_BATCH_SIZE)),
            disable_tool_exposure=_coerce_bool(raw.get("disable_tool_exposure"), False),
            debug_recall_trace=_coerce_bool(raw.get("debug_recall_trace"), False),
            user_id_override=(str(raw.get("user_id_override") or "").strip() or None),
            agent_id=(str(raw.get("agent_id") or "hermes").strip() or "hermes"),
        )

    @property
    def queue_db_path(self) -> Path:
        return _get_hermes_home() / "retaindb_queue.db"

    @property
    def identity_cache_path(self) -> Path:
        return _get_hermes_home() / "retaindb_identity.json"

    def should_activate(self) -> bool:
        return bool(
            self.enabled
            and self.api_key
            and self.project
            and self.memory_mode != "local"
        )


class RetainDBClient:
    """Minimal requests-based client for Hermes' native RetainDB integration."""

    def __init__(self, config: RetainDBClientConfig):
        self.config = config
        self.base_url = normalize_base_url(config.base_url)
        self.api_key = (config.api_key or "").strip()

    def _headers(self, endpoint: str) -> dict[str, str]:
        attach_api_key_header = endpoint.startswith("/v1/memory") or endpoint.startswith("/v1/context/query")
        token = self.api_key.replace("Bearer ", "").strip()
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.api_key if self.api_key.startswith("Bearer ") else f"Bearer {self.api_key}",
            "x-sdk-runtime": "hermes-agent",
            "x-sdk-version": "native-retaindb-v1",
        }
        if attach_api_key_header and token:
            headers["X-API-Key"] = token
        return headers

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
    ) -> Any:
        if not self.api_key:
            raise RetainDBClientError("RETAINDB_API_KEY is not configured.")

        url = f"{self.base_url}{endpoint}"
        timeout = max(0.2, float(timeout_ms or self.config.prefetch_timeout_ms) / 1000.0)
        response = requests.request(
            method=method.upper(),
            url=url,
            params=params,
            json=json_body if method.upper() not in {"GET", "DELETE"} else None,
            headers=self._headers(endpoint),
            timeout=timeout,
        )

        payload: Any
        try:
            payload = response.json()
        except ValueError:
            payload = response.text

        if response.ok:
            return payload

        message = ""
        if isinstance(payload, dict):
            message = (
                str(payload.get("message") or "")
                or str(payload.get("error") or "")
                or str(payload.get("detail") or "")
            ).strip()
        elif isinstance(payload, str):
            message = payload.strip()

        if not message:
            message = f"RetainDB request failed with HTTP {response.status_code}"

        if response.status_code in {401, 403}:
            message = f"RetainDB authentication failed: {message}"
        elif response.status_code == 404 and "project" not in message.lower():
            message = f"RetainDB endpoint not found at {url}"

        raise RetainDBClientError(message, status_code=response.status_code, payload=payload)

    def validate_api_key(self) -> list[dict[str, Any]]:
        response = self._request("GET", "/v1/projects", timeout_ms=10000)
        return list((response or {}).get("projects") or [])

    def list_projects(self) -> list[dict[str, Any]]:
        response = self._request("GET", "/v1/projects", timeout_ms=10000)
        return list((response or {}).get("projects") or [])

    def create_project(self, name: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/projects",
            json_body={"name": name},
            timeout_ms=10000,
        )

    def query(
        self,
        *,
        query: str,
        project: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        include_memories: bool = True,
        max_tokens: int | None = None,
        top_k: int = 6,
        include_graph: bool = False,
        include_parent_content: bool = False,
        retrieval_profile: str = "precision_v1",
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        project_ref = project or self.config.project
        if not project_ref:
            raise RetainDBClientError("RetainDB project is not configured.")
        return self._request(
            "POST",
            "/v1/context/query",
            json_body={
                "project": project_ref,
                "query": query,
                "user_id": user_id,
                "session_id": session_id,
                "agent_id": agent_id,
                "include_memories": include_memories,
                "top_k": top_k,
                "include_graph": include_graph,
                "include_parent_content": include_parent_content,
                "max_tokens": max_tokens or self.config.context_tokens,
                "retrieval_profile": retrieval_profile,
            },
            timeout_ms=timeout_ms,
        )

    def search_memories(
        self,
        *,
        query: str,
        project: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        memory_type: str | None = None,
        top_k: int = 8,
        include_pending: bool = True,
        profile: str = "balanced",
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        project_ref = project or self.config.project
        if not project_ref:
            raise RetainDBClientError("RetainDB project is not configured.")

        body = {
            "project": project_ref,
            "query": query,
            "user_id": user_id,
            "session_id": session_id,
            "agent_id": agent_id,
            "top_k": top_k,
            "profile": profile,
            "include_pending": include_pending,
            "memory_types": [memory_type] if memory_type else None,
        }
        try:
            return self._request(
                "POST",
                "/v1/memory/search",
                json_body=body,
                timeout_ms=timeout_ms,
            )
        except RetainDBClientError as exc:
            if exc.status_code != 404:
                raise
            legacy_body = {
                "project": project_ref,
                "query": query,
                "user_id": user_id,
                "session_id": session_id,
                "agent_id": agent_id,
                "top_k": top_k,
                "memory_type": memory_type,
            }
            return self._request(
                "POST",
                "/v1/memories/search",
                json_body=legacy_body,
                timeout_ms=timeout_ms,
            )

    def get_user_profile(
        self,
        *,
        user_id: str,
        project: str | None = None,
        include_pending: bool = True,
        memory_types: str | None = None,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        project_ref = project or self.config.project
        if not project_ref:
            raise RetainDBClientError("RetainDB project is not configured.")
        params = {"project": project_ref, "include_pending": str(include_pending).lower()}
        if memory_types:
            params["memory_types"] = memory_types
        try:
            return self._request(
                "GET",
                f"/v1/memory/profile/{quote(user_id, safe='')}",
                params=params,
                timeout_ms=timeout_ms,
            )
        except RetainDBClientError as exc:
            if exc.status_code != 404:
                raise
            legacy = self._request(
                "GET",
                "/v1/memories",
                params={
                    "project": project_ref,
                    "user_id": user_id,
                    "limit": "200",
                },
                timeout_ms=timeout_ms,
            )
            memories = list((legacy or {}).get("memories") or [])
            return {"user_id": user_id, "memories": memories, "count": len(memories)}

    def add_memory(
        self,
        *,
        content: str,
        project: str | None = None,
        memory_type: str = "factual",
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
        write_mode: str = "sync",
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        project_ref = project or self.config.project
        if not project_ref:
            raise RetainDBClientError("RetainDB project is not configured.")
        try:
            return self._request(
                "POST",
                "/v1/memory",
                json_body={
                    "project": project_ref,
                    "content": content,
                    "memory_type": memory_type,
                    "user_id": user_id,
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "importance": importance,
                    "metadata": metadata or {},
                    "write_mode": write_mode,
                },
                timeout_ms=timeout_ms,
            )
        except RetainDBClientError as exc:
            if exc.status_code != 404:
                raise
            return self._request(
                "POST",
                "/v1/memories",
                json_body={
                    "project": project_ref,
                    "content": content,
                    "memory_type": memory_type,
                    "user_id": user_id,
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "importance": importance,
                    "metadata": metadata or {},
                },
                timeout_ms=timeout_ms,
            )

    def ingest_session(
        self,
        *,
        session_id: str,
        messages: list[dict[str, Any]],
        project: str | None = None,
        user_id: str | None = None,
        write_mode: str = "sync",
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        project_ref = project or self.config.project
        if not project_ref:
            raise RetainDBClientError("RetainDB project is not configured.")
        payload = {
            "project": project_ref,
            "session_id": session_id,
            "user_id": user_id,
            "messages": messages,
            "write_mode": write_mode,
        }
        return self._request(
            "POST",
            "/v1/memory/ingest/session",
            json_body=payload,
            timeout_ms=timeout_ms or 10000,
        )

    def delete_memory(self, memory_id: str, *, timeout_ms: int | None = None) -> dict[str, Any]:
        try:
            return self._request(
                "DELETE",
                f"/v1/memory/{quote(memory_id, safe='')}",
                timeout_ms=timeout_ms or 5000,
            )
        except RetainDBClientError as exc:
            if exc.status_code != 404:
                raise
            self._request(
                "DELETE",
                f"/v1/memories/{quote(memory_id, safe='')}",
                timeout_ms=timeout_ms or 5000,
            )
            return {"success": True, "deleted": memory_id}
