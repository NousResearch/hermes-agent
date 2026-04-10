"""Hindsight memory plugin — MemoryProvider interface.

Long-term memory with knowledge graph, entity resolution, and multi-strategy
retrieval. Supports cloud (API key) and local modes.

Original PR #1811 by benfrank241, adapted to MemoryProvider ABC.

Config via environment variables:
  HINDSIGHT_API_KEY   — API key for Hindsight Cloud
  HINDSIGHT_BANK_ID   — memory bank identifier (default: hermes)
  HINDSIGHT_BUDGET    — recall budget: low/mid/high (default: mid)
  HINDSIGHT_API_URL   — API endpoint
  HINDSIGHT_MODE      — cloud or local (default: cloud)

Or via $HERMES_HOME/hindsight/config.json (profile-scoped), falling back to
~/.hindsight/config.json (legacy, shared) for backward compatibility.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading

from hermes_constants import get_hermes_home
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from hermes_constants import get_hermes_home
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "https://api.hindsight.vectorize.io"
_DEFAULT_LOCAL_URL = "http://localhost:8888"
_MIN_CLIENT_VERSION = "0.4.22"
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


def _run_sync(coro, timeout: float = 120.0):
    """Schedule *coro* on the shared loop and block until done."""
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


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
        "banks": {
            "hermes": {
                "bankId": os.environ.get("HINDSIGHT_BANK_ID", "hermes"),
                "budget": os.environ.get("HINDSIGHT_BUDGET", "mid"),
                "enabled": True,
            }
        },
    }


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
        self._client = None
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        self._session_id = ""

        # Tags
        self._tags: list[str] | None = None
        self._recall_tags: list[str] | None = None
        self._recall_tags_match = "any"

        # Retain controls
        self._auto_retain = True
        self._retain_every_n_turns = 1
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
        self._retain_async = True

    @property
    def name(self) -> str:
        return "hindsight"

    def is_available(self) -> bool:
        try:
            cfg = _load_config()
            mode = cfg.get("mode", "cloud")
            if mode in ("local", "local_embedded", "local_external"):
                return True
            has_key = bool(cfg.get("apiKey") or os.environ.get("HINDSIGHT_API_KEY", ""))
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

        print(f"\n  Checking dependencies...")
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
                print(f"  ✓ Dependencies up to date")
            except Exception as e:
                print(f"  ⚠ Install failed: {e}")
                print(f"  Run manually: uv pip install --python {sys.executable} {' '.join(deps_to_install)}")

        # Step 3: Mode-specific config
        if mode == "cloud":
            print(f"\n  Get your API key at https://ui.hindsight.vectorize.io\n")
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

        # Step 4: Save everything
        provider_config["bank_id"] = "hermes"
        provider_config["recall_budget"] = "mid"
        bank_id = "hermes"
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

        print(f"\n  ✓ Hindsight memory configured ({mode} mode)")
        if env_writes:
            print(f"  API keys saved to .env")
        print(f"\n  Start a new session to activate.\n")

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
            {"key": "bank_id", "description": "Memory bank name", "default": "hermes"},
            {"key": "bank_mission", "description": "Mission/purpose description for the memory bank"},
            {"key": "bank_retain_mission", "description": "Custom extraction prompt for memory retention"},
            {"key": "recall_budget", "description": "Recall thoroughness", "default": "mid", "choices": ["low", "mid", "high"]},
            {"key": "memory_mode", "description": "Memory integration mode", "default": "hybrid", "choices": ["hybrid", "context", "tools"]},
            {"key": "recall_prefetch_method", "description": "Auto-recall method", "default": "recall", "choices": ["recall", "reflect"]},
            {"key": "tags", "description": "Tags applied when storing memories (comma-separated)", "default": ""},
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
        ]

    def _get_client(self):
        """Return the cached Hindsight client (created once, reused)."""
        if self._client is None:
            if self._mode == "local_embedded":
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
                self._client = HindsightEmbedded(**kwargs)
            else:
                from hindsight_client import Hindsight
                kwargs = {"base_url": self._api_url, "timeout": 30.0}
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                logger.debug("Creating Hindsight cloud client (url=%s, has_key=%s)",
                             self._api_url, bool(self._api_key))
                self._client = Hindsight(**kwargs)
        return self._client

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id

        # Check client version and auto-upgrade if needed
        try:
            from importlib.metadata import version as pkg_version
            from packaging.version import Version
            installed = pkg_version("hindsight-client")
            if Version(installed) < Version(_MIN_CLIENT_VERSION):
                logger.warning("hindsight-client %s is outdated (need >=%s), attempting upgrade...",
                               installed, _MIN_CLIENT_VERSION)
                import shutil, subprocess, sys
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
        self._mode = self._config.get("mode", "cloud")
        # "local" is a legacy alias for "local_embedded"
        if self._mode == "local":
            self._mode = "local_embedded"
        self._api_key = self._config.get("apiKey") or self._config.get("api_key") or os.environ.get("HINDSIGHT_API_KEY", "")
        default_url = _DEFAULT_LOCAL_URL if self._mode in ("local_embedded", "local_external") else _DEFAULT_API_URL
        self._api_url = self._config.get("api_url") or os.environ.get("HINDSIGHT_API_URL", default_url)
        self._llm_base_url = self._config.get("llm_base_url", "")

        banks = self._config.get("banks", {}).get("hermes", {})
        self._bank_id = self._config.get("bank_id") or banks.get("bankId", "hermes")
        budget = self._config.get("recall_budget") or self._config.get("budget") or banks.get("budget", "mid")
        self._budget = budget if budget in _VALID_BUDGETS else "mid"

        memory_mode = self._config.get("memory_mode", "hybrid")
        self._memory_mode = memory_mode if memory_mode in ("context", "tools", "hybrid") else "hybrid"

        prefetch_method = self._config.get("recall_prefetch_method", "recall")
        self._prefetch_method = prefetch_method if prefetch_method in ("recall", "reflect") else "recall"

        # Bank options
        self._bank_mission = self._config.get("bank_mission", "")
        self._bank_retain_mission = self._config.get("bank_retain_mission") or None

        # Tags
        self._tags = self._config.get("tags") or None
        self._recall_tags = self._config.get("recall_tags") or None
        self._recall_tags_match = self._config.get("recall_tags_match", "any")

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
        logger.debug("Hindsight config: auto_retain=%s, auto_recall=%s, retain_every_n=%d, "
                     "retain_async=%s, retain_context=%s, "
                     "recall_max_tokens=%d, recall_max_input_chars=%d, tags=%s, recall_tags=%s",
                     self._auto_retain, self._auto_recall, self._retain_every_n_turns,
                     self._retain_async, self._retain_context,
                     self._recall_max_tokens, self._recall_max_input_chars,
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
                    from pathlib import Path as _Path
                    profile_env = _Path.home() / ".hindsight" / "profiles" / f"{profile}.env"
                    current_key = self._config.get("llm_api_key") or os.environ.get("HINDSIGHT_LLM_API_KEY", "")
                    current_provider = self._config.get("llm_provider", "")
                    current_model = self._config.get("llm_model", "")
                    current_base_url = self._config.get("llm_base_url") or os.environ.get("HINDSIGHT_API_LLM_BASE_URL", "")
                    # Map openai_compatible/openrouter → openai for the daemon (OpenAI wire format)
                    daemon_provider = "openai" if current_provider in ("openai_compatible", "openrouter") else current_provider

                    # Read saved profile config
                    saved = {}
                    if profile_env.exists():
                        for line in profile_env.read_text().splitlines():
                            if "=" in line and not line.startswith("#"):
                                k, v = line.split("=", 1)
                                saved[k.strip()] = v.strip()

                    config_changed = (
                        saved.get("HINDSIGHT_API_LLM_PROVIDER") != daemon_provider or
                        saved.get("HINDSIGHT_API_LLM_MODEL") != current_model or
                        saved.get("HINDSIGHT_API_LLM_API_KEY") != current_key or
                        saved.get("HINDSIGHT_API_LLM_BASE_URL", "") != current_base_url
                    )

                    if config_changed:
                        # Write updated profile .env
                        profile_env.parent.mkdir(parents=True, exist_ok=True)
                        env_lines = (
                            f"HINDSIGHT_API_LLM_PROVIDER={daemon_provider}\n"
                            f"HINDSIGHT_API_LLM_API_KEY={current_key}\n"
                            f"HINDSIGHT_API_LLM_MODEL={current_model}\n"
                            f"HINDSIGHT_API_LOG_LEVEL=info\n"
                        )
                        if current_base_url:
                            env_lines += f"HINDSIGHT_API_LLM_BASE_URL={current_base_url}\n"
                        profile_env.write_text(env_lines)
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
                client = self._get_client()
                if self._prefetch_method == "reflect":
                    logger.debug("Prefetch: calling reflect (bank=%s, query_len=%d)", self._bank_id, len(query))
                    resp = _run_sync(client.areflect(bank_id=self._bank_id, query=query, budget=self._budget))
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
                    resp = _run_sync(client.arecall(**recall_kwargs))
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

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Retain conversation turn in background (non-blocking).

        Respects retain_every_n_turns for batching.
        """
        if not self._auto_retain:
            logger.debug("sync_turn: skipped (auto_retain disabled)")
            return

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()

        messages = [
            {"role": "user", "content": user_content, "timestamp": now},
            {"role": "assistant", "content": assistant_content, "timestamp": now},
        ]

        turn = json.dumps(messages)
        self._session_turns.append(turn)
        self._turn_counter += 1

        # Only retain every N turns
        if self._turn_counter % self._retain_every_n_turns != 0:
            logger.debug("sync_turn: buffered turn %d (will retain at turn %d)",
                         self._turn_counter, self._turn_counter + (self._retain_every_n_turns - self._turn_counter % self._retain_every_n_turns))
            return

        logger.debug("sync_turn: retaining %d turns, total session content %d chars",
                     len(self._session_turns), sum(len(t) for t in self._session_turns))
        # Send the ENTIRE session as a single JSON array (document_id deduplicates).
        # Each element in _session_turns is a JSON string of that turn's messages.
        content = "[" + ",".join(self._session_turns) + "]"

        def _sync():
            try:
                client = self._get_client()
                item: dict = {
                    "content": content,
                    "context": self._retain_context,
                }
                if self._tags:
                    item["tags"] = self._tags
                logger.debug("Hindsight retain: bank=%s, doc=%s, async=%s, content_len=%d, num_turns=%d",
                             self._bank_id, self._session_id, self._retain_async, len(content), len(self._session_turns))
                _run_sync(client.aretain_batch(
                    bank_id=self._bank_id,
                    items=[item],
                    document_id=self._session_id,
                    retain_async=self._retain_async,
                ))
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
        try:
            client = self._get_client()
        except Exception as e:
            logger.warning("Hindsight client init failed: %s", e)
            return tool_error(f"Hindsight client unavailable: {e}")

        if tool_name == "hindsight_retain":
            content = args.get("content", "")
            if not content:
                return tool_error("Missing required parameter: content")
            context = args.get("context")
            try:
                retain_kwargs: dict = {
                    "bank_id": self._bank_id, "content": content, "context": context,
                }
                if self._tags:
                    retain_kwargs["tags"] = self._tags
                logger.debug("Tool hindsight_retain: bank=%s, content_len=%d, context=%s",
                             self._bank_id, len(content), context)
                _run_sync(client.aretain(**retain_kwargs))
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
                resp = _run_sync(client.arecall(**recall_kwargs))
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
                resp = _run_sync(client.areflect(
                    bank_id=self._bank_id, query=query, budget=self._budget
                ))
                logger.debug("Tool hindsight_reflect: response_len=%d", len(resp.text or ""))
                return json.dumps({"result": resp.text or "No relevant memories found."})
            except Exception as e:
                logger.warning("hindsight_reflect failed: %s", e, exc_info=True)
                return tool_error(f"Failed to reflect: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        logger.debug("Hindsight shutdown: waiting for background threads")
        global _loop, _loop_thread
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        if self._client is not None:
            try:
                if self._mode == "local_embedded":
                    # Use the public close() API. The RuntimeError from
                    # aiohttp's "attached to a different loop" is expected
                    # and harmless — the daemon keeps running independently.
                    try:
                        self._client.close()
                    except RuntimeError:
                        pass
                else:
                    _run_sync(self._client.aclose())
            except Exception:
                pass
            self._client = None
        # Stop the background event loop so no tasks are pending at exit
        if _loop is not None and _loop.is_running():
            _loop.call_soon_threadsafe(_loop.stop)
            if _loop_thread is not None:
                _loop_thread.join(timeout=5.0)
            _loop = None
            _loop_thread = None


def register(ctx) -> None:
    """Register Hindsight as a memory provider plugin."""
    ctx.register_memory_provider(HindsightMemoryProvider())
