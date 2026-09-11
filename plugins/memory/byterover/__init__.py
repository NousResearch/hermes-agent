"""ByteRover memory plugin — MemoryProvider interface.

Persistent memory via the ByteRover CLI (``brv``): hierarchical context tree with tiered retrieval
(fuzzy text → LLM-driven search), local-first with optional cloud sync (BRV_API_KEY). Requires the
``brv`` CLI (npm install -g byterover-cli, or byterover.dev/install.sh). Working directory is
$HERMES_HOME/byterover/ (profile-scoped); ``memory.byterover.auto_extract: false`` disables curate hooks.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_QUERY_TIMEOUT = 10    # brv query — should be fast
_CURATE_TIMEOUT = 120  # brv curate — may involve LLM processing
_MIN_QUERY_LEN = 10    # noise filters
_MIN_OUTPUT_LEN = 20

_BOOL_WORDS = {**dict.fromkeys(("1", "true", "yes", "on"), True), **dict.fromkeys(("0", "false", "no", "off"), False)}


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """bool/number -> truthiness; common true/false words parsed; anything else -> default."""
    if isinstance(value, (bool, int, float)):
        return bool(value)
    return _BOOL_WORDS.get(value.strip().lower(), default) if isinstance(value, str) else default


def _load_plugin_config() -> Dict[str, Any]:
    """Read ``memory.byterover``; fall back to legacy ``memory.provider_config`` (early docs used it)."""
    try:
        from hermes_cli.config import load_config
        memory_config = load_config().get("memory", {})
    except Exception:
        return {}
    for key in ("byterover", "provider_config") if isinstance(memory_config, dict) else ():
        block = memory_config.get(key, {})
        if isinstance(block, dict) and (block or key == "provider_config"):
            return dict(block)
    return {}


def _get_brv_cwd() -> Path:
    """Profile-scoped working directory for the brv context tree."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "byterover"


# brv binary resolution (cached, thread-safe): None = unresolved, "" = resolved-missing
_brv_path_lock = threading.Lock()
_cached_brv_path: Optional[str] = None


def _resolve_brv_path() -> Optional[str]:
    """Find the brv binary on PATH or well-known install locations (resolved once; lookup is outside the lock)."""
    global _cached_brv_path
    with _brv_path_lock:
        if _cached_brv_path is not None:
            return _cached_brv_path or None
    candidates = (Path.home() / ".brv-cli/bin/brv", Path("/usr/local/bin/brv"), Path.home() / ".npm-global/bin/brv")
    found = shutil.which("brv") or next((str(c) for c in candidates if c.exists()), None)
    with _brv_path_lock:
        if _cached_brv_path is None:
            _cached_brv_path = found or ""
        return _cached_brv_path or None


def _run_brv(args: List[str], timeout: int = _QUERY_TIMEOUT, cwd: str = None) -> dict:
    """Run a brv CLI command. Returns {success, output, error}."""
    global _cached_brv_path
    brv_path = _resolve_brv_path()
    if not brv_path:
        return {"success": False, "error": "brv CLI not found. Install: npm install -g byterover-cli"}
    effective_cwd = cwd or str(_get_brv_cwd())
    Path(effective_cwd).mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PATH": str(Path(brv_path).parent) + os.pathsep + os.environ.get("PATH", "")}
    try:
        result = subprocess.run(
            [brv_path] + args, capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=timeout, cwd=effective_cwd, env=env, stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"brv timed out after {timeout}s"}
    except FileNotFoundError:
        with _brv_path_lock:
            _cached_brv_path = None  # binary vanished; re-resolve next call
        return {"success": False, "error": "brv CLI not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    stdout, stderr = result.stdout.strip(), result.stderr.strip()
    if result.returncode == 0:
        return {"success": True, "output": stdout}
    return {"success": False, "error": stderr or stdout or f"brv exited {result.returncode}"}


def _schema(name: str, description: str, arg: str = "", arg_desc: str = "") -> dict:
    props = {arg: {"type": "string", "description": arg_desc}} if arg else {}
    return {"name": name, "description": description, "parameters": {"type": "object", "properties": props, "required": [arg] if arg else []}}


QUERY_SCHEMA = _schema(
    "brv_query", "Search ByteRover's persistent knowledge tree for relevant context. Returns memories, project knowledge, "
    "architectural decisions, and patterns from previous sessions. Use for any question where past context would help.",
    "query", "What to search for.")
CURATE_SCHEMA = _schema(
    "brv_curate", "Store important information in ByteRover's persistent knowledge tree. Use for architectural decisions, bug fixes, "
    "user preferences, project patterns — anything worth remembering across sessions. ByteRover's LLM automatically "
    "categorizes and organizes the memory.", "content", "The information to remember.")
STATUS_SCHEMA = _schema("brv_status", "Check ByteRover status — CLI version, context tree stats, cloud sync state.")


class ByteRoverMemoryProvider(MemoryProvider):
    """ByteRover persistent memory via the brv CLI."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = dict(config) if config is not None else _load_plugin_config()
        self._auto_extract = _coerce_bool(self._config.get("auto_extract"), True)
        self._cwd, self._session_id, self._turn_count = "", "", 0
        self._sync_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "byterover"

    def is_available(self) -> bool:
        """Check if brv CLI is installed. No network calls."""
        return _resolve_brv_path() is not None

    def get_config_schema(self):
        return [
            {"key": "api_key", "description": "ByteRover API key (optional, for cloud sync)", "secret": True,
             "env_var": "BRV_API_KEY", "url": "https://app.byterover.dev"},
            {"key": "auto_extract", "description": "Automatically curate completed turns and compression/memory hooks",
             "default": "true", "choices": ["true", "false"]},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        self._cwd, self._session_id, self._turn_count = str(_get_brv_cwd()), session_id, 0
        Path(self._cwd).mkdir(parents=True, exist_ok=True)

    def system_prompt_block(self) -> str:
        if not _resolve_brv_path():
            return ""
        return ("# ByteRover Memory\nActive. Persistent knowledge tree with hierarchical context.\n"
                "Use brv_query to search past knowledge, brv_curate to store important facts, brv_status to check state.")

    def _query(self, query: str) -> dict:
        return _run_brv(["query", "--", query.strip()[:5000]], timeout=_QUERY_TIMEOUT, cwd=self._cwd)

    def _curate(self, content: str) -> dict:
        return _run_brv(["curate", "--", content], timeout=_CURATE_TIMEOUT, cwd=self._cwd)

    def _curate_in_background(self, content: str, *, name: str, what: str, on_done: str = "") -> threading.Thread:
        """Spawn a daemon thread that curates ``content``; failures are logged at debug, never raised."""
        def _work():
            try:
                self._curate(content)
                if on_done:
                    logger.info(on_done)
            except Exception as e:
                logger.debug("ByteRover %s failed: %s", what, e)

        t = threading.Thread(target=_work, daemon=True, name=name)
        t.start()
        return t

    def _auto_extract_enabled(self, what: str) -> bool:
        if not self._auto_extract:
            logger.debug("ByteRover %s skipped (auto_extract disabled)", what)
        return self._auto_extract

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Run brv query synchronously (blocks up to _QUERY_TIMEOUT) so context is ready before the first LLM call."""
        if not query or len(query.strip()) < _MIN_QUERY_LEN:
            return ""
        result = self._query(query)
        output = (result.get("output") or "").strip() if result["success"] else ""
        return f"## ByteRover Context\n{output}" if len(output) > _MIN_OUTPUT_LEN else ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """No-op: prefetch() runs synchronously at turn start."""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Curate the conversation turn in background (non-blocking); only substantive turns."""
        self._turn_count += 1
        if not self._auto_extract_enabled("sync_turn") or len(user_content.strip()) < _MIN_QUERY_LEN:
            return
        if self._sync_thread and self._sync_thread.is_alive():  # wait for the previous sync so curates don't pile up
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = self._curate_in_background(f"User: {user_content[:2000]}\nAssistant: {assistant_content[:2000]}",
                                                       name="brv-sync", what="sync")

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes to ByteRover."""
        if self._auto_extract_enabled("memory mirror") and action in {"add", "replace"} and content:
            label = "User profile" if target == "user" else "Agent memory"
            self._curate_in_background(f"[{label}] {content}", name="brv-memwrite", what="memory mirror")

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract insights from the last 10 user/assistant messages before compression discards them."""
        if not self._auto_extract_enabled("pre-compression flush") or not messages:
            return ""
        parts = [f"{msg.get('role', '')}: {msg.get('content', '')[:500]}" for msg in messages[-10:]
                 if msg.get("role", "") in {"user", "assistant"} and isinstance(msg.get("content", ""), str) and msg.get("content", "").strip()]
        if parts:
            self._curate_in_background("[Pre-compression context]\n" + "\n".join(parts), name="brv-flush", what="pre-compression flush",
                                       on_done=f"ByteRover pre-compression flush: {len(parts)} messages")
        return ""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [QUERY_SCHEMA, CURATE_SCHEMA, STATUS_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name not in _TOOLS:
            return tool_error(f"Unknown tool: {tool_name}")
        arg, fail_msg, run, on_ok = _TOOLS[tool_name]
        value = args.get(arg, "") if arg else None
        if arg and not value:
            return tool_error(f"{arg} is required")
        result = run(self, value)
        return json.dumps(on_ok(result.get("output", ""))) if result["success"] else tool_error(result.get("error", fail_msg))

    def shutdown(self) -> None:
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)


def _format_query_output(output: str) -> dict:
    output = output.strip()
    if len(output) < _MIN_OUTPUT_LEN:
        return {"result": "No relevant memories found."}
    return {"result": output[:8000] + "\n\n[... truncated]" if len(output) > 8000 else output}


# tool name -> (required arg or None, failure message, run(provider, arg_value) -> brv result, on_ok(output) -> JSON payload)
_TOOLS = {
    "brv_query": ("query", "Query failed", lambda p, q: p._query(q), _format_query_output),
    "brv_curate": ("content", "Curate failed", lambda p, c: p._curate(c), lambda _: {"result": "Memory curated successfully."}),
    "brv_status": (None, "Status check failed", lambda p, _: _run_brv(["status"], timeout=15, cwd=p._cwd), lambda out: {"status": out}),
}


def register(ctx) -> None:
    """Register ByteRover as a memory provider plugin."""
    ctx.register_memory_provider(ByteRoverMemoryProvider())
