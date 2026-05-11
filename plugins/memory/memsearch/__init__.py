"""MemSearch memory plugin — semantic long-term recall via Milvus + hybrid search.

Config: $HERMES_HOME/memsearch_config.json (use absolute path for milvus_uri).
Requires: pip install memsearch + embedding API key (or local/onnx provider).
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_DEFAULTS: Dict[str, Any] = {
    "milvus_uri": "~/.memsearch/milvus.db",
    "embedding_provider": "openai",
    "embedding_model": "",
    "collection": "hermes_memory",
    "auto_ingest": True,
    "auto_compact": True,
    "compact_model": "",
    "max_recall_results": 5,
    "context_budget_tokens": 800,
    "index_paths": "",
    "sync_mode": "daemon",
}

RECALL_SCHEMA = {
    "name": "memsearch_recall",
    "description": "Semantic search across indexed memory. Returns ranked excerpts.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results (default 5, max 20)."},
        },
        "required": ["query"],
    },
}

EXPAND_SCHEMA = {
    "name": "memsearch_expand",
    "description": "Expand a memory chunk to show full section context.",
    "parameters": {
        "type": "object",
        "properties": {
            "chunk_hash": {"type": "string", "description": "Chunk hash from memsearch_recall."},
        },
        "required": ["chunk_hash"],
    },
}

INGEST_SCHEMA = {
    "name": "memsearch_ingest",
    "description": "Index a file or directory into semantic memory.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File or directory path."},
            "force": {"type": "boolean", "description": "Re-index even unchanged content."},
        },
        "required": ["path"],
    },
}


def _load_plugin_config(hermes_home: str = "") -> dict:
    from hermes_constants import get_hermes_home
    _home = hermes_home or str(get_hermes_home())
    config_path = Path(_home) / "memsearch_config.json"
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    # Fallback: config.yaml plugins.memsearch
    try:
        import yaml
        from hermes_cli.config import cfg_get
        yaml_path = Path(_home) / "config.yaml"
        if yaml_path.exists():
            with open(yaml_path, encoding="utf-8-sig") as f:
                all_config = yaml.safe_load(f) or {}
            return cfg_get(all_config, "plugins", "memsearch", default={})
    except Exception:
        pass
    return {}


def _real_home() -> str:
    """Return real user home even inside sandboxed HERMES_HOME."""
    import pwd
    for candidate in (os.environ.get("SUDO_HOME"), os.environ.get("HOME")):
        if candidate and "/profiles/" not in candidate:
            return candidate
    try:
        return pwd.getpwuid(os.getuid()).pw_dir
    except Exception:
        return os.path.expanduser("~")


def _expand_paths(cfg: dict) -> dict:
    real_home = _real_home()
    for key in ("milvus_uri", "index_paths"):
        val = cfg.get(key)
        if isinstance(val, str) and "~" in val:
            cfg[key] = val.replace("~", real_home, 1)
    return cfg


def _default_config() -> dict:
    cfg = dict(_DEFAULTS)
    saved = _load_plugin_config()
    if saved:
        cfg.update(saved)
    return _expand_paths(cfg)


class MemSearchMemoryProvider(MemoryProvider):
    """MemSearch-backed semantic memory with hybrid search and auto-ingest."""

    def __init__(self, config: dict | None = None):
        self._config = config or _default_config()
        self._session_id: str = ""
        self._hermes_home: str = ""
        self._is_primary: bool = True
        self._memsearch_available: bool = False
        self._daemon_thread: threading.Thread | None = None
        self._pending_turns: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "memsearch"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        try:
            import memsearch  # noqa: F401
        except ImportError:
            return False
        provider = self._config.get("embedding_provider", "openai")
        if provider == "openai":
            return bool(os.environ.get("OPENAI_API_KEY", ""))
        elif provider == "google":
            return bool(
                os.environ.get("GOOGLE_API_KEY", "")
                or os.environ.get("GEMINI_API_KEY", "")
            )
        elif provider in ("local", "onnx", "ollama"):
            return True
        return bool(os.environ.get("OPENAI_API_KEY", ""))

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home", os.path.expanduser("~/.hermes"))
        self._is_primary = kwargs.get("agent_context", "primary") == "primary"

        saved = _load_plugin_config(self._hermes_home)
        if saved:
            self._config.update(saved)
        self._config = _expand_paths(self._config)

        try:
            import memsearch  # noqa: F401
            self._memsearch_available = True
        except ImportError:
            logger.warning("MemSearch not installed — memory disabled")

        if self._config.get("auto_ingest", True) and self._is_primary:
            for p in self._config.get("index_paths", "").split(","):
                p = p.strip().replace("$HERMES_HOME", self._hermes_home)
                p = os.path.expanduser(p)
                if Path(p).exists():
                    self._index_path(p)

        logger.info(
            "MemSearch init: session=%s primary=%s available=%s",
            session_id, self._is_primary, self._memsearch_available,
        )

    def shutdown(self) -> None:
        if self._pending_turns:
            self._flush_turns()

    # ------------------------------------------------------------------
    # Context injection
    # ------------------------------------------------------------------

    def system_prompt_block(self) -> str:
        if not self._memsearch_available:
            return ""
        collection = self._config.get("collection", "hermes_memory")
        try:
            result = subprocess.run(
                ["memsearch", "stats", "--collection", collection],
                capture_output=True, text=True, timeout=10,
            )
            m = re.search(r"Total indexed chunks:\s*(\d+)", (result.stdout or "") + (result.stderr or ""))
            count = int(m.group(1)) if m else 0
        except Exception:
            count = 0
        if count == 0:
            return "# MemSearch Memory\nActive. Empty index — conversations will be auto-indexed.\nUse memsearch_recall to search, memsearch_ingest to add files."
        return f"# MemSearch Memory\nActive. {count} chunks indexed with hybrid semantic search.\nUse memsearch_recall for search, memsearch_expand for full context, memsearch_ingest to add files."

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._memsearch_available or len(query) < 5:
            return ""
        results = self._search(query, top_k=int(self._config.get("max_recall_results", 5)))
        if not results:
            return ""
        budget = int(self._config.get("context_budget_tokens", 800)) * 4
        lines, total = [], 0
        for r in results:
            line = f"- [{r.get('score', 0):.2f}] {r.get('heading', '')} ({r.get('source', '')})\n  {r.get('content', '')[:500]}"
            if total + len(line) > budget:
                break
            lines.append(line)
            total += len(line)
        return "## MemSearch Recall\n" + "\n".join(lines) if lines else ""

    # ------------------------------------------------------------------
    # Turn sync
    # ------------------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._is_primary or not self._config.get("auto_ingest", True):
            return
        if not self._memsearch_available or not user_content or not assistant_content:
            return
        with self._lock:
            self._pending_turns.append((user_content, assistant_content))
        if self._daemon_thread is None or not self._daemon_thread.is_alive():
            self._daemon_thread = threading.Thread(target=self._background_sync, daemon=True)
            self._daemon_thread.start()

    def _background_sync(self) -> None:
        time.sleep(2)
        self._flush_turns()

    def _flush_turns(self) -> None:
        with self._lock:
            turns = list(self._pending_turns)
            self._pending_turns.clear()
        if not turns:
            return
        memory_dir = Path(self._hermes_home) / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        md_path = memory_dir / f"{ts}-{self._session_id[:8]}.md"
        lines = []
        for user, assistant in turns:
            lines.extend([f"## User\n\n{user}\n", f"## Assistant\n\n{assistant}\n"])
        with open(md_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
        self._index_path(str(md_path))

    # ------------------------------------------------------------------
    # Session hooks (stubs or minimal)
    # ------------------------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        self._flush_turns()
        if self._config.get("auto_compact", True) and self._is_primary:
            self._run_compact()

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        topics = {m.get("content", "").split(".")[0][:100] for m in messages[-5:] if m.get("role") == "user"}
        topics.discard("")
        return f"MemSearch context: {'; '.join(topics)}" if topics else ""

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        if not content or action == "remove" or not self._memsearch_available:
            return
        md_path = Path(self._hermes_home) / "memory" / f"memory-{target}-{action}.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(f"# Memory {action}: {target}\n\n{content}\n", encoding="utf-8")
        self._index_path(str(md_path))

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        pass

    def on_session_switch(self, old_session_id: str, new_session_id: str,
                          old_messages: List[Dict[str, Any]], **kwargs) -> None:
        pass

    def on_delegation(self, task: str, result: str, *,
                      subagent_session_id: str = "", **kwargs) -> None:
        pass

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RECALL_SCHEMA, EXPAND_SCHEMA, INGEST_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "memsearch_recall":
            return self._handle_recall(args)
        elif tool_name == "memsearch_expand":
            return self._handle_expand(args)
        elif tool_name == "memsearch_ingest":
            return self._handle_ingest(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def _handle_recall(self, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return tool_error("query is required")
        results = self._search(query, top_k=min(int(args.get("top_k", 5)), 20))
        if not results:
            return json.dumps({"results": [], "count": 0, "message": "No results found"})
        formatted = [
            {
                "content": r.get("content", "")[:500],
                "source": r.get("source", ""),
                "heading": r.get("heading", ""),
                "chunk_hash": r.get("chunk_hash", ""),
                "score": r.get("score", 0),
            }
            for r in results
        ]
        return json.dumps({"results": formatted, "count": len(formatted)})

    def _handle_expand(self, args: dict) -> str:
        chunk_hash = args.get("chunk_hash", "")
        if not chunk_hash:
            return tool_error("chunk_hash is required")
        collection = self._config.get("collection", "hermes_memory")
        try:
            result = subprocess.run(
                ["memsearch", "expand", chunk_hash, "--collection", collection, "--json-output"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            return tool_error(f"Expand failed: {result.stderr.strip()}")
        except Exception as e:
            return tool_error(f"Expand error: {e}")

    def _handle_ingest(self, args: dict) -> str:
        path = args.get("path", "")
        if not path:
            return tool_error("path is required")
        path_obj = Path(path).expanduser()
        if not path_obj.exists():
            return tool_error(f"Path not found: {path}")
        self._index_path(str(path_obj), force=args.get("force", False))
        return json.dumps({"status": "indexed", "path": str(path_obj)})

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        from hermes_constants import display_hermes_home
        _db = f"{display_hermes_home()}/.memsearch/milvus.db"
        return [
            {"key": "embedding_provider", "description": "Embedding provider", "default": "openai",
             "choices": ["openai", "google", "voyage", "jina", "mistral", "ollama", "local", "onnx"]},
            {"key": "milvus_uri", "description": "Milvus URI (use absolute path)", "default": _db},
            {"key": "collection", "description": "Collection name", "default": "hermes_memory"},
            {"key": "api_key", "description": "Embedding API key", "secret": True,
             "required": True, "env_var": "OPENAI_API_KEY", "url": "https://platform.openai.com/api-keys"},
            {"key": "auto_ingest", "description": "Auto-index turns", "default": "true", "choices": ["true", "false"]},
            {"key": "auto_compact", "description": "Compact at session end", "default": "true", "choices": ["true", "false"]},
            {"key": "max_recall_results", "description": "Max search results", "default": "10"},
            {"key": "index_paths", "description": "Paths to auto-index on init", "default": ""},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        (Path(hermes_home) / "memsearch_config.json").write_text(
            json.dumps({k: v for k, v in values.items() if k != "api_key"}, indent=2),
            encoding="utf-8",
        )
        key_map = {"milvus_uri": ("milvus", "uri"), "embedding_provider": ("embedding", "provider"),
                   "embedding_model": ("embedding", "model"), "collection": ("milvus", "collection")}
        for key, val in values.items():
            tk = key_map.get(key)
            if tk:
                try:
                    subprocess.run(["memsearch", "config", "set", f"{tk[0]}.{tk[1]}", str(val)],
                                   capture_output=True, timeout=10)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_cmd(self, subcommand: str, *args) -> list:
        cmd = ["memsearch", subcommand, *args]
        provider = self._config.get("embedding_provider", "openai")
        cmd.extend(["--provider", provider])
        model = self._config.get("embedding_model", "")
        if model:
            cmd.extend(["--model", model])
        return cmd

    def _index_path(self, path: str, force: bool = False) -> None:
        cmd = self._build_cmd("index", path, "--collection", self._config.get("collection", "hermes_memory"))
        if force:
            cmd.append("--force")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                logger.info("MemSearch indexed %s", path)
            else:
                logger.warning("MemSearch index failed for %s: %s", path, result.stderr.strip()[:200])
        except Exception as e:
            logger.warning("MemSearch index error for %s: %s", path, e)

    def _search(self, query: str, top_k: int = 5) -> list:
        cmd = self._build_cmd("search", query, "--top-k", str(top_k),
                              "--collection", self._config.get("collection", "hermes_memory"), "--json-output")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout.strip())
        except Exception as e:
            logger.debug("MemSearch search failed: %s", e)
        return []

    def _run_compact(self) -> None:
        cmd = self._build_cmd("compact", "--collection", self._config.get("collection", "hermes_memory"))
        compact_model = self._config.get("compact_model", "")
        if compact_model:
            cmd.extend(["--llm-model", compact_model])
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            logger.info("MemSearch compact completed")
        except Exception as e:
            logger.debug("MemSearch compact failed: %s", e)


def register(ctx) -> None:
    ctx.register_memory_provider(MemSearchMemoryProvider(config=_default_config()))
