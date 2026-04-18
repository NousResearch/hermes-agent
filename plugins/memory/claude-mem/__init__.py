"""claude-mem memory provider for Hermes Agent."""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:37777"


def _load_config() -> dict:
    """Load config from $HERMES_HOME/claude-mem.json with env-var overrides."""
    from hermes_constants import get_hermes_home

    cfg: dict = {
        "base_url": os.environ.get("CLAUDE_MEM_WORKER_URL") or _DEFAULT_BASE_URL,
        "default_project": os.environ.get("CLAUDE_MEM_DEFAULT_PROJECT", ""),
    }
    try:
        cfg_path = Path(get_hermes_home()) / "claude-mem.json"
        if cfg_path.exists():
            file_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            cfg.update({k: v for k, v in file_cfg.items() if v})
    except Exception:
        pass
    return cfg


RECALL_SCHEMA = {
    "name": "claude_mem_recall",
    "description": (
        "Search claude-mem's cross-session memory for past observations, decisions, "
        "and session summaries relevant to the query. Returns a ranked list. "
        "Use when the user references something from a past session, or when you need "
        "to check whether a problem has been solved before."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language search query."},
            "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
            "obs_type": {
                "type": "string",
                "description": "Filter by observation type: bugfix, feature, decision, discovery, change, refactor.",
            },
        },
        "required": ["query"],
    },
}

SAVE_SCHEMA = {
    "name": "claude_mem_save",
    "description": (
        "Manually save a durable fact or decision to claude-mem. Use sparingly — "
        "claude-mem captures observations automatically from tool use. Only call this "
        "when the user explicitly asks you to remember something that wasn't captured."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The fact or decision to store."},
            "title": {"type": "string", "description": "Short title (optional)."},
        },
        "required": ["text"],
    },
}

TIMELINE_SCHEMA = {
    "name": "claude_mem_timeline",
    "description": (
        "Get context around a specific observation ID. Use after claude_mem_recall "
        "to pull surrounding work (what led up to a decision, what came after)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "anchor_id": {"type": "integer", "description": "Observation ID to anchor on."},
            "depth_before": {"type": "integer", "default": 3},
            "depth_after": {"type": "integer", "default": 3},
        },
        "required": ["anchor_id"],
    },
}


class ClaudeMemMemoryProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "claude-mem"

    def is_available(self) -> bool:
        # Dependency presence + config presence only; no network call.
        try:
            import requests  # noqa: F401
        except ImportError:
            return False
        _load_config()  # must not raise
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        from .client import ClaudeMemClient, ClaudeMemUnavailable

        self._session_id = session_id                   # IS the contentSessionId
        self._hermes_home = kwargs.get("hermes_home", "")
        self._platform = kwargs.get("platform", "cli")
        self._agent_context = kwargs.get("agent_context", "primary")
        self._project = kwargs.get("agent_workspace") or _load_config()["default_project"] or "hermes"

        self._cfg = _load_config()
        self._client = ClaudeMemClient(base_url=self._cfg["base_url"])
        self._prefetch_lock = threading.Lock()
        self._prefetch_result = ""
        self._prefetch_thread: threading.Thread | None = None
        self._sync_thread: threading.Thread | None = None

        # Skip session init for non-primary contexts (subagent/cron/flush).
        # Intent per agent/memory_provider.py:73-76.
        if self._agent_context != "primary" or self._platform == "cron":
            logger.debug("claude-mem skipped init: agent_context=%s platform=%s",
                         self._agent_context, self._platform)
            return

        def _init_bg():
            try:
                self._client.init_session(
                    content_session_id=session_id,
                    project=self._project,
                    platform_source=f"hermes-{self._platform}",
                )
            except ClaudeMemUnavailable as e:
                logger.warning("claude-mem worker unreachable at startup: %s", e)

        threading.Thread(target=_init_bg, daemon=True, name="claude-mem-init").start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if self._agent_context != "primary":
            return
        sid = session_id or self._session_id

        def _bg():
            try:
                self._client.post_observation(
                    content_session_id=sid,
                    tool_name="conversation",
                    tool_input={"user": user_content},
                    tool_response={"assistant": assistant_content},
                    cwd=os.getcwd(),
                    platform_source=f"hermes-{self._platform}",
                )
            except Exception as e:
                logger.debug("claude-mem sync_turn failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = threading.Thread(target=_bg, daemon=True, name="claude-mem-sync")
        self._sync_thread.start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        # MemoryManager wraps this in <memory-context> automatically
        # (agent/memory_manager.py:65-80), so return raw markdown.
        return result

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._agent_context != "primary":
            return

        def _bg():
            try:
                # Semantic endpoint requires >= 20 chars; fall back to /api/search.
                if query and len(query) >= 20:
                    res = self._client.context_semantic(
                        q=query, project=self._project, limit=5,
                    )
                    ctx = res.get("context", "")
                else:
                    res = self._client.search(
                        query=query, project=self._project,
                        limit=5, order_by="relevance",
                    )
                    ctx = res.get("text", "")

                if ctx:
                    with self._prefetch_lock:
                        self._prefetch_result = f"## Claude-Mem Recall\n{ctx}"
            except Exception as e:
                logger.debug("claude-mem prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(
            target=_bg, daemon=True, name="claude-mem-prefetch"
        )
        self._prefetch_thread.start()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if self._agent_context != "primary":
            return

        def _bg():
            try:
                self._client.complete_session(
                    content_session_id=self._session_id,
                    platform_source=f"hermes-{self._platform}",
                )
            except Exception as e:
                logger.debug("claude-mem complete_session failed: %s", e)

        threading.Thread(target=_bg, daemon=True, name="claude-mem-end").start()

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if self._agent_context != "primary":
            return ""

        # Fire-and-forget summarize request; the worker queues it.
        last_assistant = ""
        for m in reversed(messages):
            if m.get("role") == "assistant" and isinstance(m.get("content"), str):
                last_assistant = m["content"][:4000]
                break

        def _bg():
            try:
                self._client.post_summarize(
                    content_session_id=self._session_id,
                    last_assistant_message=last_assistant,
                    platform_source=f"hermes-{self._platform}",
                )
            except Exception as e:
                logger.debug("claude-mem summarize failed: %s", e)

        threading.Thread(target=_bg, daemon=True, name="claude-mem-summarize").start()
        return ""  # we don't contribute to the compression prompt directly

    def system_prompt_block(self) -> str:
        return (
            "# Claude-Mem Memory\n"
            "Active. Persistent cross-session memory via local worker.\n"
            "Use claude_mem_recall(query) to search past sessions and "
            "claude_mem_save(text) to store a durable fact."
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RECALL_SCHEMA, SAVE_SCHEMA, TIMELINE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        from .client import ClaudeMemUnavailable

        try:
            if tool_name == "claude_mem_recall":
                res = self._client.search(
                    query=args["query"],
                    project=self._project,
                    limit=min(int(args.get("limit", 10)), 50),
                    obs_type=args.get("obs_type"),
                    order_by="relevance",
                )
                text = res.get("text", "")
                return json.dumps({"results_markdown": text}) if text else json.dumps({"results_markdown": "No matching observations."})

            if tool_name == "claude_mem_save":
                res = self._client.memory_save(
                    text=args["text"],
                    title=args.get("title"),
                    project=self._project,
                )
                return json.dumps(res)

            if tool_name == "claude_mem_timeline":
                res = self._client.timeline(
                    anchor=int(args["anchor_id"]),
                    depth_before=int(args.get("depth_before", 3)),
                    depth_after=int(args.get("depth_after", 3)),
                    project=self._project,
                )
                return json.dumps({"timeline_markdown": res.get("text", "")})

            return json.dumps({"error": f"unknown tool: {tool_name}"})

        except ClaudeMemUnavailable as e:
            return json.dumps({"error": "claude-mem worker unavailable", "detail": str(e)})
        except Exception as e:
            logger.exception("claude-mem tool call failed")
            return json.dumps({"error": str(e)})

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "base_url",
                "description": "Claude-mem worker URL",
                "default": _DEFAULT_BASE_URL,
                "required": False,
            },
            {
                "key": "default_project",
                "description": "Default project name for scoping (auto-detected from cwd if empty)",
                "default": "",
                "required": False,
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        cfg_path = Path(hermes_home) / "claude-mem.json"
        payload = {
            "base_url": values.get("base_url") or _DEFAULT_BASE_URL,
            "default_project": values.get("default_project") or "",
        }
        cfg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def register(ctx) -> None:
    """Register claude-mem as a memory provider plugin."""
    ctx.register_memory_provider(ClaudeMemMemoryProvider())
