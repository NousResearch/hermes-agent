"""Unforgit memory provider for Hermes.

Uses the local Unforgit CLI against a configured repository directory. This is
intended as a hybrid memory backend: Hermes built-in memory remains global user
profile context; Unforgit stores/searches structured, repo-style memories.

Config: $HERMES_HOME/unforgit.json
{
  "repo_path": "$HERMES_HOME/unforgit-memory",
  "cli_path": "unforgit",
  "recall_top_k": 5,
  "local_only": true,
  "mirror_builtin_writes": true,
  "sync_turns": false
}
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)


SEARCH_SCHEMA = {
    "name": "unforgit_search",
    "description": (
        "Search Unforgit long-term memory by meaning/keywords. Use for project "
        "conventions, decisions, gotchas, durable technical facts, and playbooks."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results (default from config; max 50)."},
            "types": {"type": "array", "items": {"type": "string", "enum": ["episodic", "semantic", "procedural"]}},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["query"],
    },
}

ADD_SCHEMA = {
    "name": "unforgit_remember",
    "description": (
        "Store a durable memory in Unforgit. Prefer semantic/procedural for facts, "
        "decisions, conventions, gotchas, and playbooks; avoid temporary task progress."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Memory content to store."},
            "type": {"type": "string", "enum": ["episodic", "semantic", "procedural"], "default": "semantic"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "template": {"type": "string", "description": "Optional Unforgit template: decision, gotcha, playbook, bug, convention, etc."},
        },
        "required": ["text"],
    },
}

STATUS_SCHEMA = {
    "name": "unforgit_status",
    "description": "Check the configured Unforgit memory repository status/health.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


def _expand_path(value: str, hermes_home: Path) -> Path:
    value = value.replace("$HERMES_HOME", str(hermes_home)).replace("${HERMES_HOME}", str(hermes_home))
    return Path(os.path.expanduser(value))



def _resolve_cli(value: str, hermes_home: Path) -> Optional[str]:
    """Resolve configured Unforgit CLI path.

    Accepts either an absolute/relative path (including $HERMES_HOME) or a
    command name discoverable on PATH. As a convenience for existing Hermes
    installs, falls back to $HERMES_HOME/scripts/unforgit when cli_path is the
    default command name and the wrapper exists there.
    """
    if not value:
        value = "unforgit"
    if "/" in value or value.startswith(("~", "$HERMES_HOME", "${HERMES_HOME}")):
        path = _expand_path(value, hermes_home)
        return str(path) if path.exists() and os.access(path, os.X_OK) else None
    found = shutil.which(value)
    if found:
        return found
    fallback = hermes_home / "scripts" / value
    if fallback.exists() and os.access(fallback, os.X_OK):
        return str(fallback)
    return None

def _load_config() -> Dict[str, Any]:
    from hermes_constants import get_hermes_home

    hermes_home = get_hermes_home()
    cfg: Dict[str, Any] = {
        "repo_path": str(hermes_home / "unforgit-memory"),
        "cli_path": "unforgit",
        "recall_top_k": 5,
        "local_only": True,
        "mirror_builtin_writes": True,
        "sync_turns": False,
    }
    path = hermes_home / "unforgit.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cfg.update({k: v for k, v in data.items() if v is not None and v != ""})
        except Exception as e:
            logger.warning("Failed to load Unforgit config %s: %s", path, e)
    return cfg


class UnforgitMemoryProvider(MemoryProvider):
    """Hermes MemoryProvider backed by a local Unforgit repository."""

    def __init__(self):
        self._config: Dict[str, Any] = _load_config()
        self._repo_path: Optional[Path] = None
        self._cli_path: Optional[str] = None
        self._top_k = 5
        self._local_only = True
        self._mirror_builtin_writes = True
        self._sync_turns = False
        self._session_id = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_result = ""
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "unforgit"

    def is_available(self) -> bool:
        try:
            from hermes_constants import get_hermes_home
            hermes_home = get_hermes_home()
            cfg = _load_config()
            cli_value = str(cfg.get("cli_path", ""))
            cli = _resolve_cli(cli_value, hermes_home)
            repo = _expand_path(str(cfg.get("repo_path", "")), hermes_home)
            return cli is not None and (repo / ".unforgit" / "unforgit.yaml").exists()
        except Exception:
            return False

    def get_config_schema(self) -> List[Dict[str, Any]]:
        from hermes_constants import display_hermes_home
        home = display_hermes_home()
        return [
            {"key": "repo_path", "description": "Unforgit repository path", "default": f"{home}/unforgit-memory"},
            {"key": "cli_path", "description": "Unforgit CLI executable or absolute path", "default": "unforgit"},
            {"key": "recall_top_k", "description": "Automatic recall result count", "default": "5"},
            {"key": "local_only", "description": "Use local Unforgit store only for recall", "default": "true", "choices": ["true", "false"]},
            {"key": "mirror_builtin_writes", "description": "Mirror Hermes built-in memory writes into Unforgit", "default": "true", "choices": ["true", "false"]},
            {"key": "sync_turns", "description": "Store every completed turn in Unforgit (usually noisy)", "default": "false", "choices": ["true", "false"]},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        path = Path(hermes_home) / "unforgit.json"
        existing: Dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        existing.update(values)
        path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home
        hermes_home = get_hermes_home()
        self._config = _load_config()
        self._repo_path = _expand_path(str(self._config.get("repo_path")), hermes_home)
        self._cli_path = _resolve_cli(str(self._config.get("cli_path")), hermes_home)
        self._top_k = max(1, min(int(self._config.get("recall_top_k", 5)), 50))
        self._local_only = str(self._config.get("local_only", True)).lower() != "false"
        self._mirror_builtin_writes = str(self._config.get("mirror_builtin_writes", True)).lower() != "false"
        self._sync_turns = str(self._config.get("sync_turns", False)).lower() == "true"
        self._session_id = session_id

    def system_prompt_block(self) -> str:
        repo = str(self._repo_path or self._config.get("repo_path", ""))
        return (
            "# Unforgit Memory\n"
            f"Active. Repository: {repo}.\n"
            "Use unforgit_search for durable project/repo memory and unforgit_remember for stable facts, "
            "decisions, conventions, gotchas, and playbooks. Avoid storing temporary progress."
        )

    def _run(self, args: List[str], timeout: int = 20) -> Dict[str, Any]:
        if not self._cli_path or not self._repo_path:
            raise RuntimeError("Unforgit provider is not initialized")
        cmd = [self._cli_path, "--json", *args]
        proc = subprocess.run(
            cmd,
            cwd=str(self._repo_path),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if proc.returncode != 0:
            raise RuntimeError(err or out or f"unforgit exited with {proc.returncode}: {shlex.join(cmd)}")
        if not out:
            return {"ok": True}
        try:
            return json.loads(out)
        except Exception:
            return {"raw": out}

    @staticmethod
    def _format_results(data: Dict[str, Any], limit: int) -> str:
        results = data.get("results") or []
        lines: List[str] = []
        for item in results[:limit]:
            text = item.get("text") or item.get("memory") or item.get("content") or ""
            if not text:
                continue
            mtype = item.get("type") or "memory"
            score = item.get("score")
            sid = str(item.get("id") or "")[:8]
            prefix = f"[{mtype}"
            if sid:
                prefix += f" {sid}"
            if isinstance(score, (int, float)):
                prefix += f" score={score:.2f}"
            prefix += "]"
            lines.append(f"- {prefix} {text}")
        return "\n".join(lines)

    def _search(self, query: str, *, top_k: Optional[int] = None, types: Optional[List[str]] = None, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        limit = max(1, min(int(top_k or self._top_k), 50))
        args = ["recall", query, "-k", str(limit)]
        if self._local_only:
            args.append("--local-only")
        if types:
            args.extend(["--types", ",".join(types)])
        if tags:
            args.extend(["--tags", ",".join(tags)])
        return self._run(args)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if result:
            return f"## Unforgit Memory\n{result}"
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not query:
            return

        def _run_prefetch() -> None:
            try:
                data = self._search(query, top_k=self._top_k)
                formatted = self._format_results(data, self._top_k)
                with self._prefetch_lock:
                    self._prefetch_result = formatted
            except Exception as e:
                logger.debug("Unforgit prefetch failed: %s", e)

        if self._prefetch_thread and self._prefetch_thread.is_alive():
            return
        self._prefetch_thread = threading.Thread(target=_run_prefetch, daemon=True, name="unforgit-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._sync_turns:
            return
        text = f"User: {user_content}\nAssistant: {assistant_content}"

        def _sync() -> None:
            try:
                self._add(text, mtype="episodic", tags=["conversation", "hermes"], timeout=30)
            except Exception as e:
                logger.debug("Unforgit sync_turn failed: %s", e)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="unforgit-sync")
        self._sync_thread.start()

    def _add(self, text: str, *, mtype: str = "semantic", tags: Optional[List[str]] = None, template: str = "", timeout: int = 20) -> Dict[str, Any]:
        args = ["add", text, "--type", mtype]
        if tags:
            args.extend(["--tags", ",".join(t for t in tags if t)])
        if template:
            args.extend(["--template", template])
        return self._run(args, timeout=timeout)

    def on_memory_write(self, action: str, target: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self._mirror_builtin_writes or action != "add" or not content:
            return
        tags = ["hermes-memory", target]
        if metadata and metadata.get("platform"):
            tags.append(str(metadata["platform"]))
        try:
            self._add(content, mtype="semantic", tags=tags)
        except Exception as e:
            logger.debug("Unforgit memory_write mirror failed: %s", e)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, ADD_SCHEMA, STATUS_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            if tool_name == "unforgit_search":
                query = args.get("query", "")
                if not query:
                    return tool_error("Missing required parameter: query")
                top_k = int(args.get("top_k") or self._top_k)
                data = self._search(query, top_k=top_k, types=args.get("types"), tags=args.get("tags"))
                return json.dumps(data)

            if tool_name == "unforgit_remember":
                text = args.get("text", "")
                if not text:
                    return tool_error("Missing required parameter: text")
                data = self._add(
                    text,
                    mtype=args.get("type") or "semantic",
                    tags=args.get("tags") or [],
                    template=args.get("template") or "",
                )
                return json.dumps({"result": "Memory stored in Unforgit.", "data": data})

            if tool_name == "unforgit_status":
                status = self._run(["status"])
                return json.dumps(status)
        except Exception as e:
            return tool_error(str(e))
        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        for thread in (self._prefetch_thread, self._sync_thread):
            if thread and thread.is_alive():
                thread.join(timeout=3.0)


def register(ctx) -> None:
    """Register Unforgit as a Hermes memory provider."""
    ctx.register_memory_provider(UnforgitMemoryProvider())
