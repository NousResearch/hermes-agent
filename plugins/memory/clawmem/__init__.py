"""ClawMem memory plugin — GitHub Issues API for Jane + Hermes shared memory.

API base: https://git.clawmem.ai/api/v3
Auth: Bearer token (CLAWMEM_TOKEN)

Repo: main-787c63/memory
Memories = GitHub Issues with labels: type:memory, kind:xxx, topic:xxx
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

import httpx

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_BASE_URL = "https://git.clawmem.ai/api/v3"
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    from hermes_constants import get_hermes_home
    from pathlib import Path

    config = {
        "token": os.environ.get("CLAWMEM_TOKEN", ""),
        "repo": os.environ.get("CLAWMEM_REPO", "main-787c63/memory"),
    }

    config_path = Path(get_hermes_home()) / "clawmem.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                          if v is not None and v != ""})
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

RECALL_SCHEMA = {
    "name": "memory_recall",
    "description": (
        "Search ClawMem for relevant prior facts, decisions, and context. "
        "Use when you need to recall something Jane or Hermes stored earlier."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "limit": {"type": "integer", "description": "Max results (default: 5, max: 20)."},
        },
        "required": ["query"],
    },
}

LIST_SCHEMA = {
    "name": "memory_list",
    "description": "List all memories, optionally filtered by kind or topic.",
    "parameters": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "description": "Filter by kind (e.g. core-fact, event, lesson)."},
            "topic": {"type": "string", "description": "Filter by topic tag."},
            "limit": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
        },
    },
}

STORE_SCHEMA = {
    "name": "memory_store",
    "description": (
        "Store a durable fact in ClawMem. Stored as a GitHub Issue with labels."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short title for the memory."},
            "detail": {"type": "string", "description": "The fact or content to store."},
            "kind": {"type": "string", "description": "Type: core-fact, event, lesson, preference, convention."},
            "topics": {"type": "array", "items": {"type": "string"}, "description": "Topic tags."},
        },
        "required": ["detail"],
    },
}

FORGET_SCHEMA = {
    "name": "memory_forget",
    "description": "Mark a memory as stale by closing its GitHub issue.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory ID (issue number) to close."},
        },
        "required": ["memory_id"],
    },
}


# ---------------------------------------------------------------------------
# GitHub Issues API client for ClawMem
# ---------------------------------------------------------------------------

class _ClawMemClient:
    """Lightweight GitHub Issues API client for ClawMem."""

    def __init__(self, token: str, repo: str):
        self.token = token
        self.repo = repo
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=_BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                timeout=30.0,
            )
        return self._client

    def close(self):
        if self._client:
            self._client.close()
            self._client = None

    def _repo_path(self, *parts) -> str:
        return f"/repos/{self.repo}/{'/'.join(parts)}"

    def search(self, query: str, limit: int = 5) -> List[dict]:
        """Search memories using GitHub search API."""
        q = f"repo:{self.repo} type:issue {query}"
        resp = self._get_client().get(
            "/search/issues",
            params={"q": q, "per_page": limit, "sort": "updated", "order": "desc"},
        )
        resp.raise_for_status()
        data = resp.json()
        return [self._parse_issue(i) for i in data.get("items", [])]

    def list_memories(self, kind: str = "", topic: str = "", limit: int = 10) -> List[dict]:
        """List memories with optional label filters."""
        labels = ["type:memory"]
        if kind:
            labels.append(f"kind:{kind}")
        if topic:
            labels.append(f"topic:{topic}")
        labels_str = ",".join(labels)
        resp = self._get_client().get(
            self._repo_path("issues"),
            params={"labels": labels_str, "per_page": limit, "state": "open", "sort": "updated", "direction": "desc"},
        )
        resp.raise_for_status()
        return [self._parse_issue(i) for i in resp.json()]

    def get_memory(self, number: int) -> Optional[dict]:
        """Get a single memory by issue number."""
        try:
            resp = self._get_client().get(self._repo_path(f"issues/{number}"))
            resp.raise_for_status()
            return self._parse_issue(resp.json())
        except Exception:
            return None

    def store(self, detail: str, title: str = "", kind: str = "core-fact",
              topics: List[str] = None) -> dict:
        """Create a new memory as a GitHub issue."""
        labels = [f"kind:{kind}"]
        if topics:
            labels.extend(f"topic:{t}" for t in topics)
        # Ensure labels exist
        for label in labels:
            self._ensure_label(label)
        body = json.dumps({"detail": detail}, ensure_ascii=False)
        issue_title = title or f"Memory: {detail[:80]}"
        resp = self._get_client().post(
            self._repo_path("issues"),
            json={"title": issue_title, "body": body, "labels": labels},
        )
        resp.raise_for_status()
        result = resp.json()
        return {"id": str(result["number"]), "number": result["number"]}

    def forget(self, memory_id: str) -> dict:
        """Close a memory issue."""
        number = int(memory_id)
        resp = self._get_client().patch(
            self._repo_path(f"issues/{number}"),
            json={"state": "closed"},
        )
        resp.raise_for_status()
        return {"result": "Memory closed"}

    def _ensure_label(self, label: str):
        """Create label if it doesn't exist."""
        try:
            self._get_client().post(
                self._repo_path("labels"),
                json={"name": label, "color": "5319e7" if label.startswith("kind:") else "fbca04"},
            )
        except Exception:
            pass  # Label already exists

    def _parse_issue(self, issue: dict) -> dict:
        """Parse a GitHub issue into a memory dict."""
        labels = []
        kind = ""
        memory_topics = []
        for lbl in issue.get("labels", []):
            name = lbl.get("name", "") if isinstance(lbl, dict) else str(lbl)
            labels.append(name)
            if name.startswith("kind:"):
                kind = name[5:]
            elif name.startswith("topic:"):
                memory_topics.append(name[6:])
        # Body might be JSON with detail
        body = issue.get("body", "")
        detail = body
        try:
            body_data = json.loads(body)
            detail = body_data.get("detail", body)
        except (json.JSONDecodeError, TypeError):
            pass
        return {
            "id": str(issue["number"]),
            "number": issue["number"],
            "title": issue.get("title", ""),
            "detail": detail,
            "kind": kind,
            "topics": memory_topics,
            "labels": labels,
            "updated_at": issue.get("updated_at", ""),
        }


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class ClawMemMemoryProvider(MemoryProvider):
    """Shared ClawMem memory between Jane and Hermes via GitHub Issues."""

    def __init__(self):
        self._config: Optional[dict] = None
        self._client: Optional[_ClawMemClient] = None
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "clawmem"

    def is_available(self) -> bool:
        cfg = _load_config()
        return bool(cfg.get("token"))

    def _is_breaker_open(self) -> bool:
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning("ClawMem breaker tripped after %d failures", self._consecutive_failures)

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._client = _ClawMemClient(
            token=self._config.get("token", ""),
            repo=self._config.get("repo", "main-787c63/memory"),
        )

    def system_prompt_block(self) -> str:
        return (
            "# ClawMem Shared Memory\n"
            f"Active. Repo: {self._config.get('repo', 'main-787c63/memory')}.\n"
            "Jane and Hermes share the same memory via ClawMem (GitHub Issues API). "
            "Use memory_recall to find context, memory_store to save important facts."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        return result

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _run():
            try:
                results = self._client.search(query=query, limit=5)
                if results:
                    lines = []
                    for r in results:
                        title = r.get("title", "")
                        detail = r.get("detail", "")[:200]
                        lines.append(f"- {title}: {detail}" if title else f"- {detail}")
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("ClawMem prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="clawmem-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        pass

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RECALL_SCHEMA, LIST_SCHEMA, STORE_SCHEMA, FORGET_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "ClawMem temporarily unavailable (multiple consecutive failures). Will retry later."
            })

        try:
            if tool_name == "memory_recall":
                query = args.get("query", "")
                if not query:
                    return tool_error("Missing required parameter: query")
                limit = min(int(args.get("limit", 5)), 20)
                results = self._client.search(query=query, limit=limit)
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = [{"id": r["id"], "title": r["title"], "detail": r["detail"][:500], "kind": r["kind"], "topics": r["topics"]} for r in results]
                return json.dumps({"results": items, "count": len(items)})

            elif tool_name == "memory_list":
                kind = args.get("kind", "")
                topic = args.get("topic", "")
                limit = min(int(args.get("limit", 10)), 50)
                results = self._client.list_memories(kind=kind, topic=topic, limit=limit)
                self._record_success()
                if not results:
                    return json.dumps({"result": "No memories found."})
                items = [{"id": r["id"], "title": r["title"], "detail": r["detail"][:500], "kind": r["kind"], "topics": r["topics"]} for r in results]
                return json.dumps({"results": items, "count": len(items)})

            elif tool_name == "memory_store":
                detail = args.get("detail", "")
                if not detail:
                    return tool_error("Missing required parameter: detail")
                result = self._client.store(
                    detail=detail,
                    title=args.get("title", ""),
                    kind=args.get("kind", "core-fact"),
                    topics=args.get("topics", []),
                )
                self._record_success()
                return json.dumps({"result": "Memory stored.", "id": result.get("id", "")})

            elif tool_name == "memory_forget":
                memory_id = args.get("memory_id", "")
                if not memory_id:
                    return tool_error("Missing required parameter: memory_id")
                self._client.forget(memory_id=memory_id)
                self._record_success()
                return json.dumps({"result": "Memory marked as stale."})

            return tool_error(f"Unknown tool: {tool_name}")

        except Exception as e:
            self._record_failure()
            return tool_error(f"ClawMem error: {e}")

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        if self._client:
            self._client.close()
            self._client = None


def register(ctx) -> None:
    """Register ClawMem as a memory provider plugin."""
    ctx.register_memory_provider(ClawMemMemoryProvider())
