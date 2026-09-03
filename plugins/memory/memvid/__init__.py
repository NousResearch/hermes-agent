"""Memvid memory plugin — single-file .mv2 MemoryProvider.

This provider fronts an existing Memvid ``.mv2`` memory file through the
``memvid`` CLI. It is intentionally read-first: Hermes can recall/search/ask a
portable brain file without adding another database or cloud service.

Config via config.yaml:
  memory:
    provider: memvid
    memvid:
      file_path: ~/.hermes/memvid/mind.mv2
      executable: memvid
      prefetch_top_k: 5
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_DEFAULT_EXECUTABLE = "memvid"
_DEFAULT_PREFETCH_TOP_K = 5
_QUERY_TIMEOUT = 10
_ASK_TIMEOUT = 30


def _config_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _config_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(parsed, 20))


def _load_plugin_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
        memory_config = config.get("memory", {}) if isinstance(config, dict) else {}
        provider_config = memory_config.get("memvid", {}) if isinstance(memory_config, dict) else {}
        return dict(provider_config) if isinstance(provider_config, dict) else {}
    except Exception:
        return {}


def _default_memory_file(hermes_home: Optional[str] = None) -> Path:
    if hermes_home:
        return Path(hermes_home).expanduser() / "memvid" / "mind.mv2"
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home() / "memvid" / "mind.mv2"
    except Exception:
        return Path.home() / ".hermes" / "memvid" / "mind.mv2"


def _resolve_memory_file(config: Dict[str, Any], hermes_home: Optional[str] = None) -> Path:
    configured = _config_str(config.get("file_path")) or _config_str(config.get("path"))
    if configured:
        return Path(os.path.expandvars(configured)).expanduser()
    return _default_memory_file(hermes_home)


def _resolve_executable(config: Dict[str, Any]) -> Optional[str]:
    configured = _config_str(config.get("executable")) or _DEFAULT_EXECUTABLE
    if os.sep in configured or (os.altsep and os.altsep in configured):
        p = Path(configured).expanduser()
        return str(p) if p.exists() else None
    return shutil.which(configured)


def _run_memvid(executable: str, args: List[str], timeout: int = _QUERY_TIMEOUT) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            [executable] + args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"memvid timed out after {timeout}s"}
    except FileNotFoundError:
        return {"success": False, "error": "memvid CLI not found. Install: npm install -g memvid-cli"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode == 0:
        return {"success": True, "output": stdout}
    return {"success": False, "error": stderr or stdout or f"memvid exited {result.returncode}"}


SEARCH_SCHEMA = {
    "name": "memvid_search",
    "description": "Search a local Memvid .mv2 memory file for relevant past context.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results to return (default: configured prefetch_top_k)."},
        },
        "required": ["query"],
    },
}

ASK_SCHEMA = {
    "name": "memvid_ask",
    "description": "Ask a question against a local Memvid .mv2 memory file.",
    "parameters": {
        "type": "object",
        "properties": {"question": {"type": "string", "description": "Question for the memory file."}},
        "required": ["question"],
    },
}

STATS_SCHEMA = {
    "name": "memvid_stats",
    "description": "Return stats for the configured Memvid .mv2 memory file.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

RECENT_SCHEMA = {
    "name": "memvid_recent",
    "description": "Return recent timeline entries from the configured Memvid .mv2 memory file.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


class MemvidMemoryProvider(MemoryProvider):
    """Read/search provider for one portable Memvid ``.mv2`` file."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = dict(config) if config is not None else _load_plugin_config()
        self._hermes_home = ""
        self._session_id = ""
        self._file_path = _resolve_memory_file(self._config)
        self._executable = _resolve_executable(self._config) or ""
        self._prefetch_top_k = _config_int(self._config.get("prefetch_top_k"), _DEFAULT_PREFETCH_TOP_K)

    @property
    def name(self) -> str:
        return "memvid"

    def is_available(self) -> bool:
        return bool(self._executable and self._file_path.is_file() and self._file_path.suffix == ".mv2")

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = str(kwargs.get("hermes_home") or "")
        self._file_path = _resolve_memory_file(self._config, self._hermes_home)
        self._executable = _resolve_executable(self._config) or ""

    def system_prompt_block(self) -> str:
        if not self.is_available():
            return ""
        return (
            "Memvid memory provider active. Long-term memory is stored in one "
            f"portable .mv2 file: {self._file_path}. Use memvid_search, "
            "memvid_ask, memvid_stats, or memvid_recent when past context helps."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query.strip() or not self.is_available():
            return ""
        result = self._search(query, top_k=self._prefetch_top_k)
        if not result.get("success") or not result.get("output"):
            return ""
        return f"## Memvid recall\n{result['output']}"

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, ASK_SCHEMA, STATS_SCHEMA, RECENT_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self.is_available():
            return tool_error(tool_name, "memvid CLI or .mv2 file not available")
        if tool_name == "memvid_search":
            query = _config_str(args.get("query"))
            if not query:
                return tool_error(tool_name, "query is required")
            return json.dumps(self._search(query, top_k=_config_int(args.get("top_k"), self._prefetch_top_k)))
        if tool_name == "memvid_ask":
            question = _config_str(args.get("question"))
            if not question:
                return tool_error(tool_name, "question is required")
            return json.dumps(_run_memvid(self._executable, ["ask", str(self._file_path), question], timeout=_ASK_TIMEOUT))
        if tool_name == "memvid_stats":
            return json.dumps(_run_memvid(self._executable, ["stats", str(self._file_path)]))
        if tool_name == "memvid_recent":
            return json.dumps(_run_memvid(self._executable, ["timeline", str(self._file_path)]))
        return tool_error(tool_name, f"unknown memvid tool: {tool_name}")

    def backup_paths(self) -> List[str]:
        return [str(self._file_path)] if self._file_path.is_file() else []

    def _search(self, query: str, *, top_k: int) -> Dict[str, Any]:
        # memvid-cli accepts ``find <file> <query>`` per upstream docs. ``top_k``
        # is enforced locally by line count so older CLIs stay compatible.
        result = _run_memvid(self._executable, ["find", str(self._file_path), query])
        if result.get("success") and isinstance(result.get("output"), str) and top_k > 0:
            lines = [line for line in result["output"].splitlines() if line.strip()]
            if len(lines) > top_k:
                result["output"] = "\n".join(lines[:top_k])
        return result


def register(ctx):
    ctx.register_memory_provider(MemvidMemoryProvider())
