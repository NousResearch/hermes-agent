"""Hindsight memory plugin — MemoryProvider interface.

Long-term memory with knowledge graph, entity resolution, and multi-strategy
retrieval. Supports cloud (API key) and local modes.

Configurable request timeout via HINDSIGHT_TIMEOUT env var or config.json.
Configurable embedded daemon idle timeout via HINDSIGHT_IDLE_TIMEOUT env var
or config.json idle_timeout.

Original PR #1811 by benfrank241, adapted to MemoryProvider ABC.

Config via environment variables:
  HINDSIGHT_API_KEY                — API key for Hindsight Cloud
  HINDSIGHT_BANK_ID                — memory bank identifier (default: hermes)
  HINDSIGHT_BUDGET                 — recall budget: low/mid/high (default: mid)
  HINDSIGHT_API_URL                — API endpoint
  HINDSIGHT_MODE                   — cloud or local (default: cloud)
  HINDSIGHT_TIMEOUT                — API request timeout in seconds (default: 120)
  HINDSIGHT_IDLE_TIMEOUT           — embedded daemon idle timeout seconds; 0 disables shutdown (default: 300)
  HINDSIGHT_RETAIN_TAGS            — comma-separated tags attached to retained memories
  HINDSIGHT_RETAIN_SOURCE          — metadata source value attached to retained memories
  HINDSIGHT_RETAIN_USER_PREFIX     — label used before user turns in retained transcripts
  HINDSIGHT_RETAIN_ASSISTANT_PREFIX — label used before assistant turns in retained transcripts

Or via $HERMES_HOME/hindsight/config.json (profile-scoped), falling back to
~/.hindsight/config.json (legacy, shared) for backward compatibility.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import threading

from datetime import datetime, timezone
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from hermes_constants import get_hermes_home
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "https://api.hindsight.vectorize.io"
_DEFAULT_LOCAL_URL = "http://localhost:8888"
_MIN_CLIENT_VERSION = "0.4.22"
_DEFAULT_TIMEOUT = 120  # seconds — cloud API can take 30-40s per request
_DEFAULT_IDLE_TIMEOUT = 300  # seconds — Hindsight embedded daemon default
_VALID_BUDGETS = {"low", "mid", "high"}
_PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
    "gemini": "gemini-2.5-flash",
    "groq": "openai/gpt-oss-120b",
    "openrouter": "qwen/qwen3.5-9b",
    "minimax": "MiniMax-M2.7",
    "ollama": "gemma3:12b",
    "lmstudio": "local-model",
    "openai_compatible": "your-model-name",
}


def _parse_int_setting(value: Any, default: int) -> int:
    """Parse an integer config/env value, falling back on invalid input."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("Invalid integer Hindsight setting %r; using default %s", value, default)
        return default


def _check_local_runtime() -> tuple[bool, str | None]:
    """Return whether local embedded Hindsight imports cleanly.

    On older CPUs, importing the local Hindsight stack can raise a runtime
    error from NumPy before the daemon starts. Treat that as "unavailable"
    so Hermes can degrade gracefully instead of repeatedly trying to start
    a broken local memory backend.
    """
    try:
        importlib.import_module("hindsight")
        importlib.import_module("hindsight_embed.daemon_embed_manager")
        return True, None
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Dedicated event loop for Hindsight async calls (one per process, reused).
# Avoids creating ephemeral loops that leak aiohttp sessions.
# ---------------------------------------------------------------------------

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    """Return a long-lived event loop running on a background thread."""
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is not None and _loop.is_running():
            return _loop
        _loop = asyncio.new_event_loop()

        def _run():
            asyncio.set_event_loop(_loop)
            _loop.run_forever()

        _loop_thread = threading.Thread(target=_run, daemon=True, name="hindsight-loop")
        _loop_thread.start()
        return _loop


def _run_sync(coro, timeout: float = _DEFAULT_TIMEOUT):
    """Schedule *coro* on the shared loop and block until done."""
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


# ---------------------------------------------------------------------------
# Backward-compatible alias — instances use self._run_sync() instead.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

RETAIN_SCHEMA = {
    "name": "hindsight_retain",
    "description": (
        "Store information to long-term memory. Hindsight automatically "
        "extracts structured facts, resolves entities, and indexes for retrieval."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to store."},
            "context": {"type": "string", "description": "Short label (e.g. 'user preference', 'project decision')."},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional per-call tags to merge with configured default retain tags.",
            },
        },
        "required": ["content"],
    },
}

RECALL_SCHEMA = {
    "name": "hindsight_recall",
    "description": (
        "Search long-term memory. Returns memories ranked by relevance using "
        "semantic search, keyword matching, entity graph traversal, and reranking."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
        },
        "required": ["query"],
    },
}

REFLECT_SCHEMA = {
    "name": "hindsight_reflect",
    "description": (
        "Synthesize a reasoned answer from long-term memories. Unlike recall, "
        "this reasons across all stored memories to produce a coherent response."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The question to reflect on."},
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from profile-scoped path, legacy path, or env vars.

    Resolution order:
      1. $HERMES_HOME/hindsight/config.json  (profile-scoped)
      2. ~/.hindsight/config.json             (legacy, shared)
      3. Environment variables
    """
    from pathlib import Path

    # Profile-scoped path (preferred)
    profile_path = get_hermes_home() / "hindsight" / "config.json"
    if profile_path.exists():
        try:
            return json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Legacy shared path (backward compat)
    legacy_path = Path.home() / ".hindsight" / "config.json"
    if legacy_path.exists():
        try:
            return json.loads(legacy_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "mode": os.environ.get("HINDSIGHT_MODE", "cloud"),
        "apiKey": os.environ.get("HINDSIGHT_API_KEY", ""),
        "timeout": _parse_int_setting(os.environ.get("HINDSIGHT_TIMEOUT"), _DEFAULT_TIMEOUT),
        "idle_timeout": _parse_int_setting(os.environ.get("HINDSIGHT_IDLE_TIMEOUT"), _DEFAULT_IDLE_TIMEOUT),
        "retain_tags": os.environ.get("HINDSIGHT_RETAIN_TAGS", ""),
        "retain_source": os.environ.get("HINDSIGHT_RETAIN_SOURCE", ""),
        "retain_user_prefix": os.environ.get("HINDSIGHT_RETAIN_USER_PREFIX", "User"),
        "retain_assistant_prefix": os.environ.get("HINDSIGHT_RETAIN_ASSISTANT_PREFIX", "Assistant"),
        "banks": {
            "hermes": {
                "bankId": os.environ.get("HINDSIGHT_BANK_ID", "hermes"),
                "budget": os.environ.get("HINDSIGHT_BUDGET", "mid"),
                "enabled": True,
            }
        },
    }


def _normalize_retain_tags(value: Any) -> List[str]:
    """Normalize tag config/tool values to a deduplicated list of strings."""
    if value is None:
        return []

    raw_items: list[Any]
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                raw_items = parsed
            else:
                raw_items = text.split(",")
        else:
            raw_items = text.split(",")
    else:
        raw_items = [value]

    normalized = []
    seen = set()
    for item in raw_items:
        tag = str(item).strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
    return normalized


def _utc_timestamp() -> str:
    """Return current UTC timestamp in ISO-8601 with milliseconds and Z suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _embedded_profile_name(config: dict[str, Any]) -> str:
    """Return the Hindsight embedded profile name for this Hermes config."""
    profile = config.get("profile", "hermes")
    return str(profile or "hermes")


def _load_simple_env(path) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file, ignoring comments and blank lines."""
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _build_embedded_profile_env(config: dict[str, Any], *, llm_api_key: str | None = None) -> dict[str, str]:
    """Build the profile-scoped env file that standalone hindsight-embed consumes."""
    current_key = llm_api_key
    if current_key is None:
        current_key = (
            config.get("llmApiKey")
            or config.get("llm_api_key")
            or os.environ.get("HINDSIGHT_LLM_API_KEY", "")
        )

    current_provider = config.get("llm_provider", "")
    current_model = config.get("llm_model", "")
    current_base_url = config.get("llm_base_url") or os.environ.get("HINDSIGHT_API_LLM_BASE_URL", "")

    # The embedded daemon expects OpenAI wire format for these providers.
    daemon_provider = "openai" if current_provider in ("openai_compatible", "openrouter") else current_provider

    env_values = {
        "HINDSIGHT_API_LLM_PROVIDER": str(daemon_provider),
        "HINDSIGHT_API_LLM_API_KEY": str(current_key or ""),
        "HINDSIGHT_API_LLM_MODEL": str(current_model),
        "HINDSIGHT_API_LOG_LEVEL": "info",
    }
    if current_base_url:
        env_values["HINDSIGHT_API_LLM_BASE_URL"] = str(current_base_url)

    idle_timeout = (
        config.get("idle_timeout")
        if config.get("idle_timeout") is not None
        else os.environ.get("HINDSIGHT_IDLE_TIMEOUT")
    )
    if idle_timeout is not None and idle_timeout != "":
        env_values["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] = str(
            _parse_int_setting(idle_timeout, _DEFAULT_IDLE_TIMEOUT)
        )
    return env_values


def _embedded_profile_env_path(config: dict[str, Any]):
    from pathlib import Path

    return Path.home() / ".hindsight" / "profiles" / f"{_embedded_profile_name(config)}.env"


def _materialize_embedded_profile_env(config: dict[str, Any], *, llm_api_key: str | None = None):
    """Write the profile-scoped env file that standalone hindsight-embed uses."""
    profile_env = _embedded_profile_env_path(config)
    profile_env.parent.mkdir(parents=True, exist_ok=True)
    env_values = _build_embedded_profile_env(config, llm_api_key=llm_api_key)
    profile_env.write_text(
        "".join(f"{key}={value}\n" for key, value in env_values.items()),
        encoding="utf-8",
    )
    return profile_env

def _sanitize_bank_segment(value: str) -> str:
    """Sanitize a bank_id_template placeholder value.

    Bank IDs should be safe for URL paths and filesystem use. Replaces any
    character that isn't alphanumeric, dash, or underscore with a dash, and
    collapses runs of dashes.
    """
    if not value:
        return ""
    out = []
    prev_dash = False
    for ch in str(value):
        if ch.isalnum() or ch == "-" or ch == "_":
            out.append(ch)
            prev_dash = False
        else:
            if not prev_dash:
                out.append("-")
                prev_dash = True
    return "".join(out).strip("-_")


def _resolve_bank_id_template(template: str, fallback: str, **placeholders: str) -> str:
    """Resolve a bank_id template string with the given placeholders.

    Supported placeholders (each is sanitized before substitution):
      {profile}   — active Hermes profile name (from agent_identity)
      {workspace} — Hermes workspace name (from agent_workspace)
      {platform}  — "cli", "telegram", "discord", etc.
      {user}      — platform user id (gateway sessions)
      {session}   — current session id

    Missing/empty placeholders are rendered as the empty string and then
    collapsed — e.g. ``hermes-{user}`` with no user becomes ``hermes``.

    If the template is empty, resolution falls back to *fallback*.
    Returns the sanitized bank id.
    """
    if not template:
        return fallback
    sanitized = {k: _sanitize_bank_segment(v) for k, v in placeholders.items()}
    try:
        rendered = template.format(**sanitized)
    except (KeyError, IndexError) as exc:
        logger.warning("Invalid bank_id_template %r: %s — using fallback %r",
                       template, exc, fallback)
        return fallback
    while "--" in rendered:
        rendered = rendered.replace("--", "-")
    while "__" in rendered:
        rendered = rendered.replace("__", "_")
    rendered = rendered.strip("-_")
    return rendered or fallback


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class HindsightMemoryProvider(MemoryProvider):
    """Hindsight long-term memory with knowledge graph and multi-strategy retrieval."""

    def __init__(self):
        self._config = None
        self._api_key = None
        self._api_url = _DEFAULT_API_URL
        self._bank_id = "hermes"
        self._budget = "mid"
        self._mode = "cloud"
        self._llm_base_url = ""
        self._memory_mode = "hybrid"  # "context", "tools", or "hybrid"
        self._prefetch_method = "recall"  # "recall" or "reflect"
        self._retain_tags: List[str] = []
        self._retain_source = ""
        self._retain_user_prefix = "User"
        self._retain_assistant_prefix = "Assistant"
        self._platform = ""
        self._user_id = ""
        self._user_name = ""
        self._chat_id = ""
        self._chat_name = ""
        self._chat_type = ""
        self._thread_id = ""
        self._agent_identity = ""
        self._agent_workspace = ""
        self._turn_index = 0
        self._client = None
        self._timeout = _DEFAULT_TIMEOUT
        self._idle_timeout = _DEFAULT_IDLE_TIMEOUT
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        self._session_id = ""
        self._parent_session_id = ""
        self._document_id = ""

        # Tags
        self._tags: list[str] | None = None
        self._recall_tags: list[str] | None = None
        self._recall_tags_match = "any"

        # Retain controls
        self._auto_retain = True
        self._retain_every_n_turns = 1
        self._retain_async = True
        self._retain_context = "conversation between Hermes Agent and the User"
        self._turn_counter = 0
        self._session_turns: list[str] = []  # accumulates ALL turns for the session

        # Recall controls
        self._auto_recall = True
        self._recall_max_tokens = 4096
        self._recall_types: list[str] | None = None
        self._recall_prompt_preamble = ""
        self._recall_max_input_chars = 800

        # Bank
        self._bank_mission = ""
        self._bank_retain_mission: str | None = None
        self._bank_id_template = ""

    @property
    def name(self) -> str:
        return "hindsight"

    def is_available(self) -> bool:
        try:
            cfg = _load_config()
            mode = cfg.get("mode", "cloud")
            if mode in ("local", "local_embedded"):
                available, _ = _check_local_runtime()
                return available
            if mode == "local_external":
                return True
            has_key = bool(
                cfg.get("apiKey")
                or cfg.get("api_key")
                or os.environ.get("HINDSIGHT_API_KEY", "")
            )
            has_url = bool(cfg.get("api_url") or os.environ.get("HINDSIGHT_API_URL", ""))
            return has_key or has_url
        except Exception:
            return False

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/hindsight/config.json."""
        import json
        from pathlib import Path
        config_dir = Path(hermes_home) / "hindsight"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    def post_setup(self, hermes_home: str, config: dict) -> None:
        """Custom setup wizard — installs only the deps needed for the selected mode."""
        import getpass
        import subprocess
        import shutil
        import sys
        from pathlib import Path

        from hermes_cli.config import save_config

        from hermes_cli.memory_setup import _curses_select

        print("\n  Configuring Hindsight memory:\n")

        # Step 1: Mode selection
        mode_items = [
            ("Cloud", "Hindsight Cloud API (lightweight, just needs an API key)"),
            ("Local Embedded", "Run Hindsight locally (downloads ~200MB, needs LLM key)"),
            ("Local External", "Connect to an existing Hindsight instance"),
        ]
        mode_idx = _curses_select("  Select mode", mode_items, default=0)
        mode = ["cloud", "local_embedded", "local_external"][mode_idx]

        provider_config: dict = {"mode": mode}
        env_writes: dict = {}

        # Step 2: Install/upgrade deps for selected mode
        _MIN_CLIENT_VERSION = "0.4.22"
        cloud_dep = f"hindsight-client>={_MIN_CLIENT_VERSION}"
        local_dep = "hindsight-all"
        if mode == "local_embedded":
            deps_to_install = [local_dep]
        elif mode == "local_external":
            deps_to_install = [cloud_dep]
        else:
            deps_to_install = [cloud_dep]

        print("\n  Checking dependencies...")
        uv_path = shutil.which("uv")
        if not uv_path:
            print("  ⚠ uv not found — install it: curl -LsSf https://astral.sh/uv/install.sh | sh")
            print(f"  Then run manually: uv pip install --python {sys.executable} {' '.join(deps_to_install)}")
        else:
            try:
                subprocess.run(
                    [uv_path, "pip", "install", "--python", sys.executable, "--quiet", "--upgrade"] + deps_to_install,
                    check=True, timeout=120, capture_output=True,
                )
                print("  ✓ Dependencies up to date")
            except Exception as e:
                print(f"  ⚠ Install failed: {e}")
                print(f"  Run manually: uv pip install --python {sys.executable} {' '.join(deps_to_install)}")

        # Step 3: Mode-specific config
        if mode == "cloud":
            print("\n  Get your API key at https://ui.hindsight.vectorize.io\n")
            existing_key = os.environ.get("HINDSIGHT_API_KEY", "")
            if existing_key:
                masked = f"...{existing_key[-4:]}" if len(existing_key) > 4 else "set"
                sys.stdout.write(f"  API key (current: {masked}, blank to keep): ")
                sys.stdout.flush()
                api_key = getpass.getpass(prompt="") if sys.stdin.isatty() else sys.stdin.readline().strip()
            else:
                sys.stdout.write("  API key: ")
                sys.stdout.flush()
                api_key = getpass.getpass(prompt="") if sys.stdin.isatty() else sys.stdin.readline().strip()
            if api_key:
                env_writes["HINDSIGHT_API_KEY"] = api_key

            val = input(f"  API URL [{_DEFAULT_API_URL}]: ").strip()
            if val:
                provider_config["api_url"] = val

        elif mode == "local_external":
            val = input(f"  Hindsight API URL [{_DEFAULT_LOCAL_URL}]: ").strip()
            provider_config["api_url"] = val or _DEFAULT_LOCAL_URL

            sys.stdout.write("  API key (optional, blank to skip): ")
            sys.stdout.flush()
            api_key = getpass.getpass(prompt="") if sys.stdin.isatty() else sys.stdin.readline().strip()
            if api_key:
                env_writes["HINDSIGHT_API_KEY"] = api_key

        else:  # local_embedded
            providers_list = list(_PROVIDER_DEFAULT_MODELS.keys())
            llm_items = [
                (p, f"default model: {_PROVIDER_DEFAULT_MODELS[p]}")
                for p in providers_list
            ]
            llm_idx = _curses_select("  Select LLM provider", llm_items, default=0)
            llm_provider = providers_list[llm_idx]

            provider_config["llm_provider"] = llm_provider

            if llm_provider == "openai_compatible":
                val = input("  LLM endpoint URL (e.g. http://192.168.1.10:8080/v1): ").strip()
                if val:
                    provider_config["llm_base_url"] = val
            elif llm_provider == "openrouter":
                provider_config["llm_base_url"] = "https://openrouter.ai/api/v1"

            default_model = _PROVIDER_DEFAULT_MODELS.get(llm_provider, "gpt-4o-mini")
            val = input(f"  LLM model [{default_model}]: ").strip()
            provider_config["llm_model"] = val or default_model

            sys.stdout.write("  LLM API key: ")
            sys.stdout.flush()
            llm_key = getpass.getpass(prompt="") if sys.stdin.isatty() else sys.stdin.readline().strip()
            if llm_key:
                env_writes["HINDSIGHT_LLM_API_KEY"] = llm_key
            else:
                env_path = Path(hermes_home) / ".env"
                existing_llm_key = ""
                if env_path.exists():
                    for line in env_path.read_text().splitlines():
                        if line.startswith("HINDSIGHT_LLM_API_KEY="):
                            existing_llm_key = line.split("=", 1)[1]
                            break
                env_writes["HINDSIGHT_LLM_API_KEY"] = existing_llm_key

        # Step 4: Save everything
        provider_config["bank_id"] = "hermes"
        provider_config["recall_budget"] = "mid"
        # Read existing timeout from config if present, otherwise use default
        existing_timeout = self._config.get("timeout") if self._config else None
        timeout_val = existing_timeout if existing_timeout else _DEFAULT_TIMEOUT
        provider_config["timeout"] = timeout_val
        env_writes["HINDSIGHT_TIMEOUT"] = str(timeout_val)
        if mode == "local_embedded":
            existing_idle_timeout = self._config.get("idle_timeout") if self._config else None
            idle_timeout_val = existing_idle_timeout if existing_idle_timeout is not None else _DEFAULT_IDLE_TIMEOUT
            provider_config["idle_timeout"] = idle_timeout_val
            env_writes["HINDSIGHT_IDLE_TIMEOUT"] = str(idle_timeout_val)
        config["memory"]["provider"] = "hindsight"
        save_config(config)

        self.save_config(provider_config, hermes_home)

        if env_writes:
            env_path = Path(hermes_home) / ".env"
            env_path.parent.mkdir(parents=True, exist_ok=True)
            existing_lines = []
            if env_path.exists():
                existing_lines = env_path.read_text().splitlines()
            updated_keys = set()
            new_lines = []
            for line in existing_lines:
                key_match = line.split("=", 1)[0].strip() if "=" in line and not line.startswith("#") else None
                if key_match and key_match in env_writes:
                    new_lines.append(f"{key_match}={env_writes[key_match]}")
                    updated_keys.add(key_match)
                else:
                    new_lines.append(line)
            for k, v in env_writes.items():
                if k not in updated_keys:
                    new_lines.append(f"{k}={v}")
            env_path.write_text("\n".join(new_lines) + "\n")

        if mode == "local_embedded":
            materialized_config = dict(provider_config)
            config_path = Path(hermes_home) / "hindsight" / "config.json"
            try:
                materialized_config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass

            llm_api_key = env_writes.get("HINDSIGHT_LLM_API_KEY", "")
            if not llm_api_key:
                llm_api_key = _load_simple_env(Path(hermes_home) / ".env").get("HINDSIGHT_LLM_API_KEY", "")
            if not llm_api_key:
                llm_api_key = _load_simple_env(_embedded_profile_env_path(materialized_config)).get(
                    "HINDSIGHT_API_LLM_API_KEY",
                    "",
                )

            _materialize_embedded_profile_env(
                materialized_config,
                llm_api_key=llm_api_key or None,
            )

        print(f"\n  ✓ Hindsight memory configured ({mode} mode)")
        if env_writes:
            print("  API keys saved to .env")
        print("\n  Start a new session to activate.\n")

    def get_config_schema(self):
        return [
            {"key": "mode", "description": "Connection mode", "default": "cloud", "choices": ["cloud", "local_embedded", "local_external"]},
            # Cloud mode
            {"key": "api_url", "description": "Hindsight Cloud API URL", "default": _DEFAULT_API_URL, "when": {"mode": "cloud"}},
            {"key": "api_key", "description": "Hindsight Cloud API key", "secret": True, "env_var": "HINDSIGHT_API_KEY", "url": "https://ui.hindsight.vectorize.io", "when": {"mode": "cloud"}},
            # Local external mode
            {"key": "api_url", "description": "Hindsight API URL", "default": _DEFAULT_LOCAL_URL, "when": {"mode": "local_external"}},
            {"key": "api_key", "description": "API key (optional)", "secret": True, "env_var": "HINDSIGHT_API_KEY", "when": {"mode": "local_external"}},
            # Local embedded mode
            {"key": "llm_provider", "description": "LLM provider", "default": "openai", "choices": ["openai", "anthropic", "gemini", "groq", "openrouter", "minimax", "ollama", "lmstudio", "openai_compatible"], "when": {"mode": "local_embedded"}},
            {"key": "llm_base_url", "description": "Endpoint URL (e.g. http://192.168.1.10:8080/v1)", "default": "", "when": {"mode": "local_embedded", "llm_provider": "openai_compatible"}},
            {"key": "llm_api_key", "description": "LLM API key (optional for openai_compatible)", "secret": True, "env_var": "HINDSIGHT_LLM_API_KEY", "when": {"mode": "local_embedded"}},
            {"key": "llm_model", "description": "LLM model", "default": "gpt-4o-mini", "default_from": {"field": "llm_provider", "map": _PROVIDER_DEFAULT_MODELS}, "when": {"mode": "local_embedded"}},
            {"key": "bank_id", "description": "Memory bank name (static fallback when bank_id_template is unset)", "default": "hermes"},
            {"key": "bank_id_template", "description": "Optional template to derive bank_id dynamically. Placeholders: {profile}, {workspace}, {platform}, {user}, {session}. Example: hermes-{profile}", "default": ""},
            {"key": "bank_mission", "description": "Mission/purpose description for the memory bank"},
            {"key": "bank_retain_mission", "description": "Custom extraction prompt for memory retention"},
            {"key": "recall_budget", "description": "Recall thoroughness", "default": "mid", "choices": ["low", "mid", "high"]},
            {"key": "memory_mode", "description": "Memory integration mode", "default": "hybrid", "choices": ["hybrid", "context", "tools"]},
            {"key": "recall_prefetch_method", "description": "Auto-recall method", "default": "recall", "choices": ["recall", "reflect"]},
            {"key": "retain_tags", "description": "Default tags applied to retained memories (comma-separated)", "default": ""},
            {"key": "retain_source", "description": "Metadata source value attached to retained memories", "default": ""},
            {"key": "retain_user_prefix", "description": "Label used before user turns in retained transcripts", "default": "User"},
            {"key": "retain_assistant_prefix", "description": "Label used before assistant turns in retained transcripts", "default": "Assistant"},
            {"key": "recall_tags", "description": "Tags to filter when searching memories (comma-separated)", "default": ""},
            {"key": "recall_tags_match", "description": "Tag matching mode for recall", "default": "any", "choices": ["any", "all", "any_strict", "all_strict"]},
            {"key": "auto_recall", "description": "Automatically recall memories before each turn", "default": True},
            {"key": "auto_retain", "description": "Automatically retain conversation turns", "default": True},
            {"key": "retain_every_n_turns", "description": "Retain every N turns (1 = every turn)", "default": 1},
            {"key": "retain_async","description": "Process retain asynchronously on the Hindsight server", "default": True},
            {"key": "retain_context", "description": "Context label for retained memories", "default": "conversation between Hermes Agent and the User"},
            {"key": "recall_max_tokens", "description": "Maximum tokens for recall results", "default": 4096},
            {"key": "recall_max_input_chars", "description": "Maximum input query length for auto-recall", "default": 800},
            {"key": "recall_prompt_preamble", "description": "Custom preamble for recalled memories in context"},
            {"key": "timeout", "description": "API request timeout in seconds", "default": _DEFAULT_TIMEOUT},
            {"key": "idle_timeout", "description": "Embedded daemon idle timeout in seconds (0 disables auto-shutdown)", "default": _DEFAULT_IDLE_TIMEOUT, "when": {"mode": "local_embedded"}},
        ]

    def _get_client(self):
        """Return the cached Hindsight client (created once, reused)."""
        if self._client is None:
            if self._mode == "local_embedded":
                available, reason = _check_local_runtime()
                if not available:
                    raise RuntimeError(
                        "Hindsight local runtime is unavailable"
                        + (f": {reason}" if reason else "")
                    )
                from hindsight import HindsightEmbedded
                HindsightEmbedded.__del__ = lambda self: None
                llm_provider = self._config.get("llm_provider", "")
                if llm_provider in ("openai_compatible", "openrouter"):
                    llm_provider = "openai"
                logger.debug("Creating HindsightEmbedded client (profile=%s, provider=%s)",
                             self._config.get("profile", "hermes"), llm_provider)
                kwargs = dict(
                    profile=self._config.get("profile", "hermes"),
                    llm_provider=llm_provider,
                    llm_api_key=self._config.get("llmApiKey") or self._config.get("llm_api_key") or os.environ.get("HINDSIGHT_LLM_API_KEY", ""),
                    llm_model=self._config.get("llm_model", ""),
                )
                if self._llm_base_url:
                    kwargs["llm_base_url"] = self._llm_base_url
                idle_timeout = _parse_int_setting(
                    self._config.get("idle_timeout")
                    if self._config.get("idle_timeout") is not None
                    else os.environ.get("HINDSIGHT_IDLE_TIMEOUT", self._idle_timeout),
                    _DEFAULT_IDLE_TIMEOUT,
                )
                self._idle_timeout = idle_timeout
                kwargs["idle_timeout"] = idle_timeout
                self._client = HindsightEmbedded(**kwargs)
            else:
                from hindsight_client import Hindsight
                timeout = self._timeout or _DEFAULT_TIMEOUT
                kwargs = {"base_url": self._api_url, "timeout": float(timeout)}
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                logger.debug("Creating Hindsight cloud client (url=%s, has_key=%s, timeout=%s)",
                             self._api_url, bool(self._api_key), kwargs["timeout"])
                self._client = Hindsight(**kwargs)
        return self._client

    def _run_sync(self, coro):
        """Schedule *coro* on the shared loop using the configured timeout."""
        return _run_sync(coro, timeout=self._timeout)

    def _is_retriable_embedded_connection_error(self, exc: Exception) -> bool:
        """Return True for stale embedded-daemon connection failures."""
        if self._mode != "local_embedded":
            return False
        text = f"{type(exc).__name__}: {exc}".lower()
        return any(
            marker in text
            for marker in (
                "cannot connect to host",
                "connection refused",
                "connect call failed",
                "clientconnectorerror",
            )
        )

    def _run_hindsight_operation(self, operation):
        """Run an async Hindsight client operation, retrying once after idle shutdown."""
        client = self._get_client()
        try:
            return self._run_sync(operation(client))
        except Exception as exc:
            if not self._is_retriable_embedded_connection_error(exc):
                raise
            logger.info(
                "Hindsight embedded daemon appears unreachable; recreating client and retrying once: %s",
                exc,
            )
            self._client = None
            client = self._get_client()
            self._client = client
            return self._run_sync(operation(client))

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = str(session_id or "").strip()
        self._parent_session_id = str(kwargs.get("parent_session_id", "") or "").strip()

        # Each process lifecycle gets its own document_id. Reusing session_id
        # alone caused overwrites on /resume — the reloaded session starts
        # with an empty _session_turns, so the next retain would replace the
        # previously stored content. session_id stays in tags so processes
        # for the same session remain filterable together.
        start_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._document_id = f"{self._session_id}-{start_ts}"

        # Check client version and auto-upgrade if needed
        try:
            from importlib.metadata import version as pkg_version
            from packaging.version import Version
            installed = pkg_version("hindsight-client")
            if Version(installed) < Version(_MIN_CLIENT_VERSION):
                logger.warning("hindsight-client %s is outdated (need >=%s), attempting upgrade...",
                               installed, _MIN_CLIENT_VERSION)
                import shutil
                import subprocess
                import sys
                uv_path = shutil.which("uv")
                if uv_path:
                    try:
                        subprocess.run(
                            [uv_path, "pip", "install", "--python", sys.executable,
                             "--quiet", "--upgrade", f"hindsight-client>={_MIN_CLIENT_VERSION}"],
                            check=True, timeout=120, capture_output=True,
                        )
                        logger.info("hindsight-client upgraded to >=%s", _MIN_CLIENT_VERSION)
                    except Exception as e:
                        logger.warning("Auto-upgrade failed: %s. Run: uv pip install 'hindsight-client>=%s'",
                                       e, _MIN_CLIENT_VERSION)
                else:
                    logger.warning("uv not found. Run: pip install 'hindsight-client>=%s'", _MIN_CLIENT_VERSION)
        except Exception:
            pass  # packaging not available or other issue — proceed anyway

        self._config = _load_config()
        self._platform = str(kwargs.get("platform") or "").strip()
        self._user_id = str(kwargs.get("user_id") or "").strip()
        self._user_name = str(kwargs.get("user_name") or "").strip()
        self._chat_id = str(kwargs.get("chat_id") or "").strip()
        self._chat_name = str(kwargs.get("chat_name") or "").strip()
        self._chat_type = str(kwargs.get("chat_type") or "").strip()
        self._thread_id = str(kwargs.get("thread_id") or "").strip()
        self._agent_identity = str(kwargs.get("agent_identity") or "").strip()
        self._agent_workspace = str(kwargs.get("agent_workspace") or "").strip()
        self._turn_index = 0
        self._session_turns = []
        self._mode = self._config.get("mode", "cloud")
        # Read timeout from config or env var, fall back to default
        self._timeout = _parse_int_setting(
            self._config.get("timeout") if self._config.get("timeout") is not None else os.environ.get("HINDSIGHT_TIMEOUT"),
            _DEFAULT_TIMEOUT,
        )
        self._idle_timeout = _parse_int_setting(
            self._config.get("idle_timeout") if self._config.get("idle_timeout") is not None else os.environ.get("HINDSIGHT_IDLE_TIMEOUT"),
            _DEFAULT_IDLE_TIMEOUT,
        )
        # "local" is a legacy alias for "local_embedded"
        if self._mode == "local":
            self._mode = "local_embedded"
        if self._mode == "local_embedded":
            available, reason = _check_local_runtime()
            if not available:
                logger.warning(
                    "Hindsight local mode disabled because its runtime could not be imported: %s",
                    reason,
                )
                self._mode = "disabled"
                return
        self._api_key = self._config.get("apiKey") or self._config.get("api_key") or os.environ.get("HINDSIGHT_API_KEY", "")
        default_url = _DEFAULT_LOCAL_URL if self._mode in ("local_embedded", "local_external") else _DEFAULT_API_URL
        self._api_url = self._config.get("api_url") or os.environ.get("HINDSIGHT_API_URL", default_url)
        self._llm_base_url = self._config.get("llm_base_url", "")

        banks = self._config.get("banks", {}).get("hermes", {})
        static_bank_id = self._config.get("bank_id") or banks.get("bankId", "hermes")
        self._bank_id_template = self._config.get("bank_id_template", "") or ""
        self._bank_id = _resolve_bank_id_template(
            self._bank_id_template,
            fallback=static_bank_id,
            profile=self._agent_identity,
            workspace=self._agent_workspace,
            platform=self._platform,
            user=self._user_id,
            session=self._session_id,
        )
        budget = self._config.get("recall_budget") or self._config.get("budget") or banks.get("budget", "mid")
        self._budget = budget if budget in _VALID_BUDGETS else "mid"

        memory_mode = self._config.get("memory_mode", "hybrid")
        self._memory_mode = memory_mode if memory_mode in ("context", "tools", "hybrid") else "hybrid"

        prefetch_method = self._config.get("recall_prefetch_method") or self._config.get("prefetch_method", "recall")
        self._prefetch_method = prefetch_method if prefetch_method in ("recall", "reflect") else "recall"

        # Bank options
        self._bank_mission = self._config.get("bank_mission", "")
        self._bank_retain_mission = self._config.get("bank_retain_mission") or None

        # Tags
        self._retain_tags = _normalize_retain_tags(
            self._config.get("retain_tags")
            or os.environ.get("HINDSIGHT_RETAIN_TAGS", "")
        )
        self._tags = self._retain_tags or None
        self._recall_tags = self._config.get("recall_tags") or None
        self._recall_tags_match = self._config.get("recall_tags_match", "any")
        self._retain_source = str(
            self._config.get("retain_source") or os.environ.get("HINDSIGHT_RETAIN_SOURCE", "")
        ).strip()
        self._retain_user_prefix = str(
            self._config.get("retain_user_prefix") or os.environ.get("HINDSIGHT_RETAIN_USER_PREFIX", "User")
        ).strip() or "User"
        self._retain_assistant_prefix = str(
            self._config.get("retain_assistant_prefix") or os.environ.get("HINDSIGHT_RETAIN_ASSISTANT_PREFIX", "Assistant")
        ).strip() or "Assistant"

        # Retain controls
        self._auto_retain = self._config.get("auto_retain", True)
        self._retain_every_n_turns = max(1, int(self._config.get("retain_every_n_turns", 1)))
        self._retain_context = self._config.get("retain_context", "conversation between Hermes Agent and the User")

        # Recall controls
        self._auto_recall = self._config.get("auto_recall", True)
        self._recall_max_tokens = int(self._config.get("recall_max_tokens", 4096))
        self._recall_types = self._config.get("recall_types") or None
        self._recall_prompt_preamble = self._config.get("recall_prompt_preamble", "")
        self._recall_max_input_chars = int(self._config.get("recall_max_input_chars", 800))
        self._retain_async = self._config.get("retain_async", True)

        _client_version = "unknown"
        try:
            from importlib.metadata import version as pkg_version
            _client_version = pkg_version("hindsight-client")
        except Exception:
            pass
        logger.info("Hindsight initialized: mode=%s, api_url=%s, bank=%s, budget=%s, memory_mode=%s, prefetch_method=%s, client=%s",
                     self._mode, self._api_url, self._bank_id, self._budget, self._memory_mode, self._prefetch_method, _client_version)
        if self._bank_id_template:
            logger.debug("Hindsight bank resolved from template %r: profile=%s workspace=%s platform=%s user=%s -> bank=%s",
                         self._bank_id_template, self._agent_identity, self._agent_workspace,
                         self._platform, self._user_id, self._bank_id)
        logger.debug("Hindsight config: auto_retain=%s, auto_recall=%s, retain_every_n=%d, "
                     "retain_async=%s, retain_context=%s, recall_max_tokens=%d, recall_max_input_chars=%d, tags=%s, recall_tags=%s",
                     self._auto_retain, self._auto_recall, self._retain_every_n_turns,
                     self._retain_async, self._retain_context, self._recall_max_tokens, self._recall_max_input_chars,
                     self._tags, self._recall_tags)

        # For local mode, start the embedded daemon in the background so it
        # doesn't block the chat. Redirect stdout/stderr to a log file to
        # prevent rich startup output from spamming the terminal.
        if self._mode == "local_embedded":
            def _start_daemon():
                import traceback
                log_dir = get_hermes_home() / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / "hindsight-embed.log"
                try:
                    # Redirect the daemon manager's Rich console to our log file
                    # instead of stderr. This avoids global fd redirects that
                    # would capture output from other threads.
                    import hindsight_embed.daemon_embed_manager as dem
                    from rich.console import Console
                    dem.console = Console(file=open(log_path, "a"), force_terminal=False)

                    client = self._get_client()
                    profile = self._config.get("profile", "hermes")

                    # Update the profile .env to match our current config so
                    # the daemon always starts with the right settings.
                    # If the config changed and the daemon is running, stop it.
                    profile_env = _embedded_profile_env_path(self._config)
                    expected_env = _build_embedded_profile_env(self._config)
                    saved = _load_simple_env(profile_env)
                    config_changed = saved != expected_env

                    if config_changed:
                        profile_env = _materialize_embedded_profile_env(self._config)
                        if client._manager.is_running(profile):
                            with open(log_path, "a") as f:
                                f.write("\n=== Config changed, restarting daemon ===\n")
                            client._manager.stop(profile)

                    client._ensure_started()
                    with open(log_path, "a") as f:
                        f.write("\n=== Daemon started successfully ===\n")
                except Exception as e:
                    with open(log_path, "a") as f:
                        f.write(f"\n=== Daemon startup failed: {e} ===\n")
                        traceback.print_exc(file=f)

            t = threading.Thread(target=_start_daemon, daemon=True, name="hindsight-daemon-start")
            t.start()

    def system_prompt_block(self) -> str:
        if self._memory_mode == "context":
            return (
                f"# Hindsight Memory\n"
                f"Active (context mode). Bank: {self._bank_id}, budget: {self._budget}.\n"
                f"Relevant memories are automatically injected into context."
            )
        if self._memory_mode == "tools":
            return (
                f"# Hindsight Memory\n"
                f"Active (tools mode). Bank: {self._bank_id}, budget: {self._budget}.\n"
                f"Use hindsight_recall to search, hindsight_reflect for synthesis, "
                f"hindsight_retain to store facts."
            )
        return (
            f"# Hindsight Memory\n"
            f"Active. Bank: {self._bank_id}, budget: {self._budget}.\n"
            f"Relevant memories are automatically injected into context. "
            f"Use hindsight_recall to search, hindsight_reflect for synthesis, "
            f"hindsight_retain to store facts."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            logger.debug("Prefetch: waiting for background thread to complete")
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            logger.debug("Prefetch: no results available")
            return ""
        logger.debug("Prefetch: returning %d chars of context", len(result))
        header = self._recall_prompt_preamble or (
            "# Hindsight Memory (persistent cross-session context)\n"
            "Use this to answer questions about the user and prior sessions. "
            "Do not call tools to look up information that is already present here."
        )
        return f"{header}\n\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._memory_mode == "tools":
            logger.debug("Prefetch: skipped (tools-only mode)")
            return
        if not self._auto_recall:
            logger.debug("Prefetch: skipped (auto_recall disabled)")
            return
        # Truncate query to max chars
        if self._recall_max_input_chars and len(query) > self._recall_max_input_chars:
            query = query[:self._recall_max_input_chars]

        def _run():
            try:
                if self._prefetch_method == "reflect":
                    logger.debug("Prefetch: calling reflect (bank=%s, query_len=%d)", self._bank_id, len(query))
                    resp = self._run_hindsight_operation(lambda client: client.areflect(bank_id=self._bank_id, query=query, budget=self._budget))
                    text = resp.text or ""
                else:
                    recall_kwargs: dict = {
                        "bank_id": self._bank_id, "query": query,
                        "budget": self._budget, "max_tokens": self._recall_max_tokens,
                    }
                    if self._recall_tags:
                        recall_kwargs["tags"] = self._recall_tags
                        recall_kwargs["tags_match"] = self._recall_tags_match
                    if self._recall_types:
                        recall_kwargs["types"] = self._recall_types
                    logger.debug("Prefetch: calling recall (bank=%s, query_len=%d, budget=%s)",
                                 self._bank_id, len(query), self._budget)
                    resp = self._run_hindsight_operation(lambda client: client.arecall(**recall_kwargs))
                    num_results = len(resp.results) if resp.results else 0
                    logger.debug("Prefetch: recall returned %d results", num_results)
                    text = "\n".join(f"- {r.text}" for r in resp.results if r.text) if resp.results else ""
                if text:
                    with self._prefetch_lock:
                        self._prefetch_result = text
            except Exception as e:
                logger.debug("Hindsight prefetch failed: %s", e, exc_info=True)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="hindsight-prefetch")
        self._prefetch_thread.start()

    def _build_turn_messages(self, user_content: str, assistant_content: str) -> List[Dict[str, str]]:
        now = datetime.now(timezone.utc).isoformat()
        return [
            {
                "role": "user",
                "content": f"{self._retain_user_prefix}: {user_content}",
                "timestamp": now,
            },
            {
                "role": "assistant",
                "content": f"{self._retain_assistant_prefix}: {assistant_content}",
                "timestamp": now,
            },
        ]

    def _build_metadata(self, *, message_count: int, turn_index: int) -> Dict[str, str]:
        metadata: Dict[str, str] = {
            "retained_at": _utc_timestamp(),
            "message_count": str(message_count),
            "turn_index": str(turn_index),
        }
        if self._retain_source:
            metadata["source"] = self._retain_source
        if self._session_id:
            metadata["session_id"] = self._session_id
        if self._platform:
            metadata["platform"] = self._platform
        if self._user_id:
            metadata["user_id"] = self._user_id
        if self._user_name:
            metadata["user_name"] = self._user_name
        if self._chat_id:
            metadata["chat_id"] = self._chat_id
        if self._chat_name:
            metadata["chat_name"] = self._chat_name
        if self._chat_type:
            metadata["chat_type"] = self._chat_type
        if self._thread_id:
            metadata["thread_id"] = self._thread_id
        if self._agent_identity:
            metadata["agent_identity"] = self._agent_identity
        return metadata

    def _build_retain_kwargs(
        self,
        content: str,
        *,
        context: str | None = None,
        document_id: str | None = None,
        metadata: Dict[str, str] | None = None,
        tags: List[str] | None = None,
        retain_async: bool | None = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "bank_id": self._bank_id,
            "content": content,
            "metadata": metadata or self._build_metadata(message_count=1, turn_index=self._turn_index),
        }
        if context is not None:
            kwargs["context"] = context
        if document_id:
            kwargs["document_id"] = document_id
        if retain_async is not None:
            kwargs["retain_async"] = retain_async
        merged_tags = _normalize_retain_tags(self._retain_tags)
        for tag in _normalize_retain_tags(tags):
            if tag not in merged_tags:
                merged_tags.append(tag)
        if merged_tags:
            kwargs["tags"] = merged_tags
        return kwargs

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Retain conversation turn in background (non-blocking).

        Respects retain_every_n_turns for batching.
        """
        if not self._auto_retain:
            logger.debug("sync_turn: skipped (auto_retain disabled)")
            return

        if session_id:
            self._session_id = str(session_id).strip()

        turn = json.dumps(self._build_turn_messages(user_content, assistant_content), ensure_ascii=False)
        self._session_turns.append(turn)
        self._turn_counter += 1
        self._turn_index = self._turn_counter

        if self._turn_counter % self._retain_every_n_turns != 0:
            logger.debug("sync_turn: buffered turn %d (will retain at turn %d)",
                         self._turn_counter, self._turn_counter + (self._retain_every_n_turns - self._turn_counter % self._retain_every_n_turns))
            return

        logger.debug("sync_turn: retaining %d turns, total session content %d chars",
                     len(self._session_turns), sum(len(t) for t in self._session_turns))
        content = "[" + ",".join(self._session_turns) + "]"

        lineage_tags: list[str] = []
        if self._session_id:
            lineage_tags.append(f"session:{self._session_id}")
        if self._parent_session_id:
            lineage_tags.append(f"parent:{self._parent_session_id}")

        def _sync():
            try:
                client = self._get_client()
                item = self._build_retain_kwargs(
                    content,
                    context=self._retain_context,
                    metadata=self._build_metadata(
                        message_count=len(self._session_turns) * 2,
                        turn_index=self._turn_index,
                    ),
                    tags=lineage_tags or None,
                )
                item.pop("bank_id", None)
                item.pop("retain_async", None)
                logger.debug("Hindsight retain: bank=%s, doc=%s, async=%s, content_len=%d, num_turns=%d",
                             self._bank_id, self._document_id, self._retain_async, len(content), len(self._session_turns))
                self._run_hindsight_operation(
                    lambda client: client.aretain_batch(
                        bank_id=self._bank_id,
                        items=[item],
                        document_id=self._document_id,
                        retain_async=self._retain_async,
                    )
                )
                logger.debug("Hindsight retain succeeded")
            except Exception as e:
                logger.warning("Hindsight sync failed: %s", e, exc_info=True)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="hindsight-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if self._memory_mode == "context":
            return []
        return [RETAIN_SCHEMA, RECALL_SCHEMA, REFLECT_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name == "hindsight_retain":
            content = args.get("content", "")
            if not content:
                return tool_error("Missing required parameter: content")
            context = args.get("context")
            try:
                retain_kwargs = self._build_retain_kwargs(
                    content,
                    context=context,
                    tags=args.get("tags"),
                )
                logger.debug("Tool hindsight_retain: bank=%s, content_len=%d, context=%s",
                             self._bank_id, len(content), context)
                self._run_hindsight_operation(lambda client: client.aretain(**retain_kwargs))
                logger.debug("Tool hindsight_retain: success")
                return json.dumps({"result": "Memory stored successfully."})
            except Exception as e:
                logger.warning("hindsight_retain failed: %s", e, exc_info=True)
                return tool_error(f"Failed to store memory: {e}")

        elif tool_name == "hindsight_recall":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            try:
                recall_kwargs: dict = {
                    "bank_id": self._bank_id, "query": query, "budget": self._budget,
                    "max_tokens": self._recall_max_tokens,
                }
                if self._recall_tags:
                    recall_kwargs["tags"] = self._recall_tags
                    recall_kwargs["tags_match"] = self._recall_tags_match
                if self._recall_types:
                    recall_kwargs["types"] = self._recall_types
                logger.debug("Tool hindsight_recall: bank=%s, query_len=%d, budget=%s",
                             self._bank_id, len(query), self._budget)
                resp = self._run_hindsight_operation(lambda client: client.arecall(**recall_kwargs))
                num_results = len(resp.results) if resp.results else 0
                logger.debug("Tool hindsight_recall: %d results", num_results)
                if not resp.results:
                    return json.dumps({"result": "No relevant memories found."})
                lines = [f"{i}. {r.text}" for i, r in enumerate(resp.results, 1)]
                return json.dumps({"result": "\n".join(lines)})
            except Exception as e:
                logger.warning("hindsight_recall failed: %s", e, exc_info=True)
                return tool_error(f"Failed to search memory: {e}")

        elif tool_name == "hindsight_reflect":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            try:
                logger.debug("Tool hindsight_reflect: bank=%s, query_len=%d, budget=%s",
                             self._bank_id, len(query), self._budget)
                resp = self._run_hindsight_operation(
                    lambda client: client.areflect(
                        bank_id=self._bank_id, query=query, budget=self._budget
                    )
                )
                logger.debug("Tool hindsight_reflect: response_len=%d", len(resp.text or ""))
                return json.dumps({"result": resp.text or "No relevant memories found."})
            except Exception as e:
                logger.warning("hindsight_reflect failed: %s", e, exc_info=True)
                return tool_error(f"Failed to reflect: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        logger.debug("Hindsight shutdown: waiting for background threads")
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        if self._client is not None:
            try:
                if self._mode == "local_embedded":
                    # HindsightEmbedded.close() delegates to its sync client.close().
                    # When Hermes created/used that client on the shared async loop,
                    # closing it from this thread can raise "attached to a different
                    # loop" before aiohttp releases the session. Close the embedded
                    # inner async client on the shared loop first, then let the
                    # wrapper clean up daemon/UI bookkeeping.
                    inner_client = getattr(self._client, "_client", None)
                    if inner_client is not None and hasattr(inner_client, "aclose"):
                        _run_sync(inner_client.aclose())
                        try:
                            self._client._client = None
                        except Exception:
                            pass
                    try:
                        self._client.close()
                    except RuntimeError:
                        pass
                else:
                    self._run_sync(self._client.aclose())
            except Exception:
                pass
            self._client = None
        # The module-global background event loop (_loop / _loop_thread)
        # is intentionally NOT stopped here. It is shared across every
        # HindsightMemoryProvider instance in the process — the plugin
        # loader creates a new provider per AIAgent, and the gateway
        # creates one AIAgent per concurrent chat session. Stopping the
        # loop from one provider's shutdown() strands the aiohttp
        # ClientSession + TCPConnector owned by every sibling provider
        # on a dead loop, which surfaces as the "Unclosed client session"
        # / "Unclosed connector" warnings reported in #11923. The loop
        # runs on a daemon thread and is reclaimed on process exit;
        # per-session cleanup happens via self._client.aclose() above.


def register(ctx) -> None:
    """Register Hindsight as a memory provider plugin."""
    ctx.register_memory_provider(HindsightMemoryProvider())
