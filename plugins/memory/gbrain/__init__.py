"""GBrain memory provider for Hermes.

The provider talks to a GBrain MCP HTTP endpoint and defaults to read-only
recall.  Write-capable behavior must be explicitly enabled and is intended for
Athena/steward profiles under the user's single-writer governance pattern.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "http://127.0.0.1:3132/mcp"
_DEFAULT_MODE = "read-only"
_DEFAULT_SOURCE_ID = "__all__"
_DEFAULT_MAX_RESULTS = 6
_DEFAULT_TIMEOUT = 5.0
_DEFAULT_QUERY_TOOL = "query"
_DEFAULT_WRITE_TOOL = "create_page"
_VALID_MODES = {"read-only", "read-write"}
_MAX_CONTENT_CHARS = 4000

_CONTEXT_STRIP_RE = re.compile(
    r"<gbrain-memory-context>[\s\S]*?</gbrain-memory-context>\s*", re.IGNORECASE
)


def _default_config() -> dict:
    return {
        "endpoint": _DEFAULT_ENDPOINT,
        "mode": _DEFAULT_MODE,
        "source_id": _DEFAULT_SOURCE_ID,
        "max_results": _DEFAULT_MAX_RESULTS,
        "timeout": _DEFAULT_TIMEOUT,
        "query_tool": _DEFAULT_QUERY_TOOL,
        "query_tool_fallbacks": ["gbrain_query", "search"],
        "write_tool": _DEFAULT_WRITE_TOOL,
    }


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _coerce_config(raw: Optional[dict] = None) -> dict:
    config = _default_config()
    if isinstance(raw, dict):
        config.update({k: v for k, v in raw.items() if v is not None})

    endpoint = str(config.get("endpoint", "")).strip()
    config["endpoint"] = endpoint

    mode = str(config.get("mode", _DEFAULT_MODE)).strip().lower()
    config["mode"] = mode if mode in _VALID_MODES else _DEFAULT_MODE

    source_id = str(config.get("source_id", _DEFAULT_SOURCE_ID)).strip()
    config["source_id"] = source_id or _DEFAULT_SOURCE_ID

    for key, default, lower, upper in (
        ("max_results", _DEFAULT_MAX_RESULTS, 1, 20),
    ):
        try:
            config[key] = max(lower, min(upper, int(config.get(key, default))))
        except Exception:
            config[key] = default

    try:
        config["timeout"] = max(0.5, min(30.0, float(config.get("timeout", _DEFAULT_TIMEOUT))))
    except Exception:
        config["timeout"] = _DEFAULT_TIMEOUT

    config["query_tool"] = str(config.get("query_tool", _DEFAULT_QUERY_TOOL)).strip() or _DEFAULT_QUERY_TOOL
    config["write_tool"] = str(config.get("write_tool", _DEFAULT_WRITE_TOOL)).strip() or _DEFAULT_WRITE_TOOL

    fallbacks = config.get("query_tool_fallbacks", [])
    if isinstance(fallbacks, str):
        fallbacks = [item.strip() for item in fallbacks.split(",") if item.strip()]
    elif isinstance(fallbacks, list):
        fallbacks = [str(item).strip() for item in fallbacks if str(item).strip()]
    else:
        fallbacks = []
    config["query_tool_fallbacks"] = [name for name in fallbacks if name != config["query_tool"]]

    return config


def _load_gbrain_config(hermes_home: str) -> dict:
    config = _default_config()
    config_path = Path(hermes_home) / "gbrain-memory.json"
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                config.update(raw)
        except Exception:
            logger.debug("Failed to parse %s", config_path, exc_info=True)

    # Config.yaml wins over the profile-local provider file so
    # `hermes config set memory.gbrain.*` and `hermes memory setup gbrain` can
    # both drive the same provider.
    try:
        from hermes_cli.config import load_config

        mem_config = (load_config().get("memory") or {})
        yaml_config = mem_config.get("gbrain") if isinstance(mem_config, dict) else None
        if isinstance(yaml_config, dict):
            config.update(yaml_config)
    except Exception:
        logger.debug("Failed to read memory.gbrain config", exc_info=True)

    # Environment override is useful for tests and ephemeral deployments; it is
    # a URL, not a credential.
    env_endpoint = os.environ.get("GBRAIN_MCP_ENDPOINT")
    if env_endpoint is not None:
        config["endpoint"] = env_endpoint

    return _coerce_config(config)


def _save_gbrain_config(values: dict, hermes_home: str) -> None:
    config_path = Path(hermes_home) / "gbrain-memory.json"
    existing: dict[str, Any] = {}
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                existing = raw
        except Exception:
            existing = {}
    existing.update(values)
    from utils import atomic_json_write

    atomic_json_write(config_path, _coerce_config(existing), mode=0o600, sort_keys=True)


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = _CONTEXT_STRIP_RE.sub("", str(text))
    return text.strip()[:_MAX_CONTENT_CHARS]


def _extract_text_fragments(value: Any) -> list[str]:
    """Extract human-readable text snippets from common MCP result shapes."""
    fragments: list[str] = []
    if value is None:
        return fragments
    if isinstance(value, str):
        if value.strip():
            fragments.append(value.strip())
        return fragments
    if isinstance(value, list):
        for item in value:
            fragments.extend(_extract_text_fragments(item))
        return fragments
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            fragments.append(value["text"].strip())
            return fragments
        for key in ("content", "text", "markdown", "summary", "compiled", "answer"):
            inner = value.get(key)
            if inner:
                fragments.extend(_extract_text_fragments(inner))
        return fragments
    return fragments


def _format_search_results(payload: Any, max_results: int) -> str:
    """Format GBrain MCP payloads into compact memory context."""
    # Common native MCP shape: {content: [{type: text, text: "..."}], ...}
    fragments = _extract_text_fragments(payload)
    if fragments:
        text = "\n\n".join(fragments)
        return f"<gbrain-memory-context>\n## GBrain recalled context\n{text[:_MAX_CONTENT_CHARS]}\n</gbrain-memory-context>"

    # Defensive fallback for direct list/dict search payloads.
    items: list[Any] = []
    if isinstance(payload, dict):
        for key in ("results", "pages", "matches", "data"):
            if isinstance(payload.get(key), list):
                items = payload[key]
                break
    elif isinstance(payload, list):
        items = payload

    lines: list[str] = []
    for item in items[:max_results]:
        if isinstance(item, dict):
            slug = item.get("slug") or item.get("page_slug") or item.get("title") or "GBrain page"
            content = item.get("content") or item.get("text") or item.get("summary") or item.get("description") or ""
            content = _clean_text(content)
            if content:
                lines.append(f"- {slug}: {content}")
        elif item:
            lines.append(f"- {_clean_text(str(item))}")
    if not lines:
        return ""
    return "<gbrain-memory-context>\n## GBrain recalled context\n" + "\n".join(lines) + "\n</gbrain-memory-context>"


class _GBrainMCPClient:
    """Tiny stdlib JSON-RPC-over-HTTP client for GBrain's MCP endpoint."""

    def __init__(self, endpoint: str, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self._next_id = 1
        self._session_id = ""
        self._initialized = False
        self._lock = threading.Lock()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        try:
            return self._request("tools/call", {"name": name, "arguments": arguments})
        except RuntimeError as exc:
            # Some Streamable HTTP MCP servers require initialize first; many
            # tolerate direct tools/call. Retry once after initialization.
            if "initialized" not in str(exc).lower() and "initialize" not in str(exc).lower():
                raise
            self.initialize()
            return self._request("tools/call", {"name": name, "arguments": arguments})

    def initialize(self) -> None:
        if self._initialized:
            return
        result = self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "hermes-gbrain-memory", "version": "1.0.0"},
            },
            allow_uninitialized=True,
        )
        self._initialized = True
        # Best-effort initialized notification; failures are non-fatal because
        # some lightweight HTTP bridges ignore notifications.
        try:
            self._notify("notifications/initialized", {})
        except Exception:
            logger.debug("GBrain MCP initialized notification failed", exc_info=True)
        return result

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        self._post(payload)

    def _request(self, method: str, params: dict[str, Any], *, allow_uninitialized: bool = False) -> Any:
        with self._lock:
            req_id = self._next_id
            self._next_id += 1
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        response = self._post(payload)
        if "error" in response:
            err = response.get("error") or {}
            message = err.get("message") if isinstance(err, dict) else str(err)
            if allow_uninitialized:
                raise RuntimeError(message or "MCP request failed")
            raise RuntimeError(message or "MCP request failed")
        return response.get("result")

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "Hermes-GBrain-Memory/1.0",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        request = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                session_id = response.headers.get("Mcp-Session-Id") or response.headers.get("mcp-session-id")
                if session_id:
                    self._session_id = session_id
                body = response.read().decode("utf-8", errors="replace")
                content_type = response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"GBrain MCP HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"GBrain MCP unavailable: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("GBrain MCP request timed out") from exc

        return self._decode_response(body, content_type)

    @staticmethod
    def _decode_response(body: str, content_type: str = "") -> dict[str, Any]:
        body = body.strip()
        if not body:
            return {}
        if "text/event-stream" in content_type or body.startswith("event:") or body.startswith("data:"):
            for line in body.splitlines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    parsed = json.loads(data)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue
            raise RuntimeError("GBrain MCP returned no JSON event data")
        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise RuntimeError("GBrain MCP returned non-object JSON")
        return parsed


class GBrainMemoryProvider(MemoryProvider):
    """Read-mostly GBrain-backed memory provider."""

    def __init__(self) -> None:
        self._config = _coerce_config()
        self._client: Optional[_GBrainMCPClient] = None
        self._session_id = ""
        self._agent_identity = ""
        self._last_error = ""

    @property
    def name(self) -> str:
        return "gbrain"

    def is_available(self) -> bool:
        try:
            config = _load_gbrain_config(str(Path(os.environ.get("HERMES_HOME", "")).expanduser())) if os.environ.get("HERMES_HOME") else self._config
        except Exception:
            config = self._config
        return bool(str(config.get("endpoint", "")).strip())

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home") or os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
        self._config = _load_gbrain_config(str(hermes_home))
        self._session_id = session_id
        self._agent_identity = str(kwargs.get("agent_identity") or "").strip()
        self._last_error = ""
        if not self._config.get("endpoint"):
            raise RuntimeError("GBrain memory provider endpoint is not configured")
        self._client = _GBrainMCPClient(self._config["endpoint"], timeout=float(self._config["timeout"]))

    def system_prompt_block(self) -> str:
        mode = self._config.get("mode", _DEFAULT_MODE)
        if mode == "read-write":
            return (
                "GBrain memory provider is active in explicit read-write mode. "
                "Only write durable memories when the user or profile policy authorizes it; "
                "do not store secrets or raw credentials."
            )
        return (
            "GBrain memory provider is active in read-only mode. Use retrieved GBrain "
            "context as background memory. Do not write, ingest, delete, or modify GBrain; "
            "route candidate durable updates to the Athena steward unless read-write mode is explicitly configured."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        query = str(query or "").strip()
        if not query or not self._client:
            return ""
        result = self._search_payload(query, limit=int(self._config["max_results"]))
        if result.get("error"):
            self._last_error = result["error"]
            logger.debug("GBrain memory prefetch failed: %s", self._last_error)
            return ""
        return _format_search_results(result.get("result"), int(self._config["max_results"]))

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        # Deliberately no automatic conversation ingestion. The safe default is
        # read-only, and even read-write mode mirrors only explicit memory writes.
        return None

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._config.get("mode") != "read-write":
            return None
        if action != "add" or not content or not self._client:
            return None
        payload = {
            "title": f"Hermes memory candidate: {target}",
            "content": _clean_text(content),
            "tags": ["hermes-memory", "athena-reviewed"],
            "metadata": {
                "source": "hermes-memory-provider",
                "target": target,
                "session_id": self._session_id,
                "agent_identity": self._agent_identity,
                **(metadata or {}),
            },
        }
        try:
            self._client.call_tool(str(self._config["write_tool"]), payload)
        except Exception as exc:
            logger.warning("GBrain memory write mirror failed: %s", exc)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "gbrain_memory_search",
                "description": "Search GBrain persistent memory through the configured MCP endpoint.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Maximum results", "default": self._config.get("max_results", _DEFAULT_MAX_RESULTS)},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "gbrain_memory_store_candidate",
                "description": "Submit a durable memory candidate to GBrain only when explicit read-write mode is configured; otherwise returns a safe Athena-steward handoff.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Memory candidate content"},
                        "target": {"type": "string", "description": "memory or user", "default": "memory"},
                    },
                    "required": ["content"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "gbrain_memory_search":
            query = str(args.get("query") or "").strip()
            if not query:
                return tool_error("query is required")
            try:
                limit = max(1, min(20, int(args.get("limit") or self._config["max_results"])))
            except Exception:
                limit = int(self._config["max_results"])
            result = self._search_payload(query, limit=limit)
            if result.get("error"):
                return json.dumps({"ok": False, "error": result["error"]})
            formatted = _format_search_results(result.get("result"), limit)
            return json.dumps({"ok": True, "context": formatted, "raw": result.get("result")})

        if tool_name == "gbrain_memory_store_candidate":
            content = _clean_text(str(args.get("content") or ""))
            target = str(args.get("target") or "memory").strip() or "memory"
            if not content:
                return tool_error("content is required")
            if self._config.get("mode") != "read-write":
                return json.dumps({
                    "ok": False,
                    "blocked": True,
                    "mode": self._config.get("mode", _DEFAULT_MODE),
                    "message": "GBrain provider is read-only. Route this candidate to Athena/steward review instead of writing directly.",
                    "candidate": {"target": target, "content": content},
                })
            if not self._client:
                return json.dumps({"ok": False, "error": "GBrain MCP client is not initialized"})
            try:
                result = self._client.call_tool(
                    str(self._config["write_tool"]),
                    {
                        "title": f"Hermes memory candidate: {target}",
                        "content": content,
                        "tags": ["hermes-memory"],
                        "metadata": {"source": "hermes-memory-provider", "target": target},
                    },
                )
                return json.dumps({"ok": True, "result": result})
            except Exception as exc:
                return json.dumps({"ok": False, "error": str(exc)})

        raise NotImplementedError(f"GBrain provider does not handle {tool_name}")

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "endpoint",
                "description": "GBrain HTTP MCP endpoint",
                "default": _DEFAULT_ENDPOINT,
            },
            {
                "key": "mode",
                "description": "Safety mode",
                "default": _DEFAULT_MODE,
                "choices": ["read-only", "read-write"],
            },
            {
                "key": "source_id",
                "description": "GBrain source scope (__all__ or a source id)",
                "default": _DEFAULT_SOURCE_ID,
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        _save_gbrain_config(values, hermes_home)

    def post_setup(self, hermes_home: str, config: Dict[str, Any]) -> None:
        """Non-interactive safe setup used by `hermes memory setup gbrain`.

        The generic setup path uses curses for choice fields, which is awkward
        for headless profile provisioning. GBrain's safe default is read-only,
        so setup can activate that default and leave advanced edits to
        config.yaml or `$HERMES_HOME/gbrain-memory.json`.
        """
        from hermes_cli.config import save_config

        if not isinstance(config.get("memory"), dict):
            config["memory"] = {}
        provider_config = config["memory"].get("gbrain")
        if not isinstance(provider_config, dict):
            provider_config = {}
        merged = _coerce_config(provider_config)
        # Never let setup accidentally promote to write mode. Users can edit
        # mode explicitly after setup for Athena-only writer profiles.
        merged["mode"] = "read-only"
        config["memory"]["provider"] = "gbrain"
        config["memory"]["gbrain"] = merged
        save_config(config)
        _save_gbrain_config(merged, hermes_home)
        print("\n  Memory provider: gbrain")
        print("  Mode: read-only (safe default)")
        print(f"  Endpoint: {merged['endpoint']}")
        print("  Activation saved to config.yaml")
        print("  Provider config saved to gbrain-memory.json")
        print("\n  Start a new session to activate.\n")

    def _search_payload(self, query: str, *, limit: int) -> dict[str, Any]:
        if not self._client:
            return {"error": "GBrain MCP client is not initialized"}
        args = {
            "query": query,
            "limit": limit,
            "adaptive_return": True,
        }
        source_id = self._config.get("source_id")
        if source_id:
            args["source_id"] = source_id
        tool_names = [str(self._config["query_tool"])] + list(self._config.get("query_tool_fallbacks") or [])
        errors: list[str] = []
        for tool_name in tool_names:
            try:
                return {"result": self._client.call_tool(tool_name, args), "tool": tool_name}
            except Exception as exc:
                errors.append(f"{tool_name}: {exc}")
        return {"error": "; ".join(errors) or "GBrain MCP query failed"}


def register(ctx) -> None:
    ctx.register_memory_provider(GBrainMemoryProvider())
