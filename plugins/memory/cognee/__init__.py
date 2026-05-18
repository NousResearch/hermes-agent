"""Cognee memory provider for Hermes Agent."""

from __future__ import annotations

import importlib.util
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from hermes_cli.config import save_env_value
from tools.registry import tool_error

from .client import (
    CogneeClientConfig,
    cognee_forget,
    cognee_recall,
    cognee_remember,
    ensure_wal_mode,
    run_async,
    serialize_result,
    stop_event_loop,
)

logger = logging.getLogger(__name__)

_PREFETCH_TIMEOUT = 8.0
_SYNC_TIMEOUT = 20.0
_TOOL_TIMEOUT = 60.0
_SESSION_END_TIMEOUT = 90.0
_MIN_TURN_LENGTH = 16

REMEMBER_SCHEMA = {
    "name": "cognee_remember",
    "description": (
        "Store durable memory in Cognee. Use for explicit user preferences, stable facts, "
        "project context, documents, or URLs that should become searchable later."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Text, path, or URL to store."},
            "dataset_name": {"type": "string", "description": "Optional Cognee dataset name. Defaults to Hermes dataset."},
            "session_scoped": {"type": "boolean", "description": "Store as fast session memory first (default false)."},
            "self_improvement": {"type": "boolean", "description": "Let Cognee bridge/enrich memory automatically (default true)."},
        },
        "required": ["content"],
    },
}

RECALL_SCHEMA = {
    "name": "cognee_recall",
    "description": (
        "Recall relevant context from Cognee memory using semantic/graph/session retrieval. "
        "Use when past facts or project context may help answer the user."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language memory query."},
            "dataset_name": {"type": "string", "description": "Optional Cognee dataset to search."},
            "top_k": {"type": "integer", "description": "Maximum results. Defaults to 10."},
            "only_context": {"type": "boolean", "description": "Return retrieved context without final Cognee answer."},
            "session_scoped": {"type": "boolean", "description": "Include current session cache in recall (default true)."},
        },
        "required": ["query"],
    },
}

FORGET_SCHEMA = {
    "name": "cognee_forget",
    "description": (
        "Delete or reset Cognee memory. Use only when the user explicitly asks to forget/delete data."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "dataset_name": {"type": "string", "description": "Dataset to forget/reset."},
            "data_id": {"type": "string", "description": "Optional specific data/item id to forget if supported by Cognee."},
            "confirm": {"type": "boolean", "description": "Must be true to execute deletion."},
        },
        "required": ["confirm"],
    },
}


def _json_result(**payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _clean_turn_text(text: str) -> str:
    return (text or "").strip()


async def _cognee_setup() -> None:
    """Run cognee.setup() to ensure internal databases are created."""
    import cognee
    await cognee.setup()


async def _init_cognee_loop() -> None:
    """Warm up cognee on the background event loop so that asyncio locks
    (used by LadybugDB and cognee's pipeline runner) are created on the
    same loop where ``_handle_remember`` / ``_handle_recall`` will run.
    """
    import cognee.infrastructure.llm.config as _lc
    _lc.get_llm_config()
    import cognee.infrastructure.databases.vector.embeddings.config as _ec
    _ec.get_embedding_config()


class CogneeMemoryProvider(MemoryProvider):
    """Cognee v1 memory provider with automatic recall and turn capture."""

    def __init__(self) -> None:
        self._config = CogneeClientConfig.from_global_config()
        self._session_id = ""
        self._hermes_home = ""
        self._agent_context = "primary"
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None
        self._sync_thread: threading.Thread | None = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "cognee"

    def is_available(self) -> bool:
        cfg = CogneeClientConfig.from_global_config()
        return bool(cfg.enabled and cfg.api_key and importlib.util.find_spec("cognee"))

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "api_key",
                "description": "LLM API key used by Cognee for graph extraction and recall",
                "secret": True,
                "required": True,
                "env_var": "LLM_API_KEY",
                "url": "https://docs.cognee.ai/setup-configuration/llm-configuration",
            },
            {
                "key": "provider",
                "description": "Cognee LLM provider",
                "default": "openai",
                "env_var": "LLM_PROVIDER",
            },
            {
                "key": "base_url",
                "description": "Optional OpenAI-compatible base URL for the Cognee LLM provider",
                "required": False,
                "env_var": "LLM_BASE_URL",
            },
            {
                "key": "dataset_name",
                "description": "Default Cognee dataset for Hermes durable memory",
                "default": "hermes_memory",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Persist Cognee setup values.

        The generic memory setup flow writes secret env vars itself. This method
        also supports direct calls and writes all provided env-var fields to
        ``$HERMES_HOME/.env`` through Hermes' config helper.
        """

        env_keys = {
            "api_key": "LLM_API_KEY",
            "provider": "LLM_PROVIDER",
            "base_url": "LLM_BASE_URL",
        }
        for key, env_var in env_keys.items():
            value = values.get(key)
            if value:
                save_env_value(env_var, str(value))

        non_secret = {k: v for k, v in values.items() if k not in env_keys and v not in (None, "")}
        if non_secret:
            cfg_path = Path(hermes_home) / "cognee.json"
            existing: Dict[str, Any] = {}
            if cfg_path.exists():
                try:
                    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        existing = raw
                except Exception:
                    existing = {}
            existing.update(non_secret)
            cfg_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._config = CogneeClientConfig.from_global_config()
        self._config.apply_to_environment()
        self._session_id = session_id or ""
        self._hermes_home = str(kwargs.get("hermes_home") or "")
        self._agent_context = str(kwargs.get("agent_context") or "primary")
        self._initialized = True
        # Ensure Cognee's internal databases are created (one-time init).
        try:
            run_async(_cognee_setup(), timeout=15.0)
        except Exception:
            logger.warning("Cognee database setup failed (non-fatal)", exc_info=True)
        # Warm up cognee on the background loop so asyncio locks
        # are created on the right event loop, not the agent's main loop.
        try:
            run_async(_init_cognee_loop(), timeout=5.0)
        except Exception:
            logger.debug("Cognee background init (non-fatal)", exc_info=True)

    def system_prompt_block(self) -> str:
        dataset = self._config.dataset_name
        return (
            "# Cognee Memory (Primary Provider)\n"
            f"Cognee is the ONLY active memory provider. Default dataset: {dataset}.\n"
            "Cognee provides vector + knowledge-graph recall with semantic search.\n"
            "The built-in `memory` tool is DISABLED — do NOT use it.\n"
            "Use `cognee_remember` to save durable facts, `cognee_recall` for explicit searches, "
            "and `cognee_forget` only after an explicit deletion request."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=0.25)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"<cognee-memory>\n{result}\n</cognee-memory>"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        query = (query or "").strip()
        if not query or not self._initialized:
            return

        effective_session = session_id or self._session_id

        def _run() -> None:
            try:
                result = run_async(
                    cognee_recall(
                        query,
                        session_id=effective_session or None,
                        datasets=[self._config.dataset_name],
                        top_k=5,
                        only_context=True,
                    ),
                    timeout=_PREFETCH_TIMEOUT,
                )
                serialized = serialize_result(result)
                if serialized:
                    with self._prefetch_lock:
                        self._prefetch_result = self._format_recall(serialized)
            except Exception as exc:
                logger.debug("Cognee prefetch failed: %s", exc)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="cognee-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if self._agent_context != "primary":
            return
        user_content = _clean_turn_text(user_content)
        assistant_content = _clean_turn_text(assistant_content)
        if len(user_content) + len(assistant_content) < _MIN_TURN_LENGTH:
            return

        effective_session = session_id or self._session_id
        turn_text = f"[user]\n{user_content}\n\n[assistant]\n{assistant_content}"

        def _sync() -> None:
            try:
                run_async(
                    cognee_remember(
                        turn_text,
                        session_id=effective_session or None,
                        dataset_name=self._config.dataset_name,
                        self_improvement=True,
                    ),
                    timeout=_SYNC_TIMEOUT,
                )
            except Exception as exc:
                logger.warning("Cognee turn sync failed: %s", exc)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=1.0)
        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="cognee-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [REMEMBER_SCHEMA, RECALL_SCHEMA, FORGET_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        try:
            if tool_name == "cognee_remember":
                return self._handle_remember(args)
            if tool_name == "cognee_recall":
                return self._handle_recall(args)
            if tool_name == "cognee_forget":
                return self._handle_forget(args)
            return tool_error(f"Unknown Cognee tool: {tool_name}")
        except Exception as exc:
            logger.warning("Cognee tool %s failed: %s", tool_name, exc)
            return tool_error(str(exc))

    def _handle_remember(self, args: Dict[str, Any]) -> str:
        content = str(args.get("content") or "").strip()
        if not content:
            return tool_error("content is required")
        self._config = CogneeClientConfig.from_global_config()
        self._config.apply_to_environment()
        # Clear cognee's cached config so it picks up fresh env vars
        import cognee.infrastructure.llm.config as _lc
        _lc.get_llm_config.cache_clear()
        import cognee.infrastructure.databases.vector.embeddings.config as _ec
        _ec.get_embedding_config.cache_clear()
        session_scoped = bool(args.get("session_scoped", False))
        dataset_name = str(args.get("dataset_name") or self._config.dataset_name)
        result = run_async(
            cognee_remember(
                content,
                dataset_name=dataset_name,
                session_id=self._session_id if session_scoped else None,
                self_improvement=args.get("self_improvement", True),
            ),
            timeout=_TOOL_TIMEOUT,
        )
        return _json_result(ok=True, result=serialize_result(result))

    def _handle_recall(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("query is required")
        self._config = CogneeClientConfig.from_global_config()
        self._config.apply_to_environment()
        # Clear cognee's cached config so it picks up fresh env vars
        import cognee.infrastructure.llm.config as _lc
        _lc.get_llm_config.cache_clear()
        import cognee.infrastructure.databases.vector.embeddings.config as _ec
        _ec.get_embedding_config.cache_clear()
        dataset_name = str(args.get("dataset_name") or self._config.dataset_name)
        session_scoped = args.get("session_scoped", True)
        kwargs: Dict[str, Any] = {
            "datasets": [dataset_name] if dataset_name else None,
            "session_id": self._session_id if session_scoped else None,
            "top_k": int(args.get("top_k") or 10),
            "only_context": bool(args.get("only_context", False)),
        }
        result = run_async(cognee_recall(query, **kwargs), timeout=_TOOL_TIMEOUT)
        return _json_result(ok=True, result=serialize_result(result))

    def _handle_forget(self, args: Dict[str, Any]) -> str:
        if args.get("confirm") is not True:
            return tool_error("cognee_forget requires confirm=true after an explicit user deletion request")
        dataset_name = str(args.get("dataset_name") or self._config.dataset_name)
        data_id = str(args.get("data_id") or "").strip()
        kwargs: Dict[str, Any] = {"dataset": dataset_name}
        if data_id:
            kwargs["data_id"] = data_id
        result = run_async(cognee_forget(**kwargs), timeout=_TOOL_TIMEOUT)
        return _json_result(ok=True, result=serialize_result(result))

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if self._agent_context != "primary" or not messages:
            return
        chunks: List[str] = []
        for msg in messages[-40:]:
            role = msg.get("role")
            content = msg.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
                continue
            chunks.append(f"[{role}] {content.strip()}")
        if not chunks:
            return
        payload = "Session transcript excerpt for durable memory extraction:\n\n" + "\n\n".join(chunks)
        try:
            run_async(
                cognee_remember(
                    payload,
                    dataset_name=self._config.dataset_name,
                    session_ids=[self._session_id] if self._session_id else None,
                    self_improvement=True,
                    run_in_background=True,
                ),
                timeout=_SESSION_END_TIMEOUT,
            )
        except Exception as exc:
            logger.warning("Cognee session-end save failed: %s", exc)

    def on_session_switch(self, new_session_id: str, *, reset: bool = False, **kwargs: Any) -> None:
        self._session_id = new_session_id or ""
        if reset:
            with self._prefetch_lock:
                self._prefetch_result = ""

    def shutdown(self) -> None:
        for thread in (self._prefetch_thread, self._sync_thread):
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
        stop_event_loop()

    @staticmethod
    def _format_recall(result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, list):
            lines = []
            for item in result[:10]:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or item.get("memory") or item.get("answer") or json.dumps(item, ensure_ascii=False)
                else:
                    text = str(item)
                if text:
                    lines.append(f"- {text}")
            return "\n".join(lines)
        if isinstance(result, dict):
            for key in ("answer", "result", "context", "content", "text"):
                if result.get(key):
                    value = result[key]
                    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            return json.dumps(result, ensure_ascii=False)
        return str(result)


__all__ = ["CogneeMemoryProvider"]
