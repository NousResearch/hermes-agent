"""
Gateway runner - entry point for messaging platform integrations.

This module provides:
- start_gateway(): Start all configured platform adapters
- GatewayRunner: Main class managing the gateway lifecycle

Usage:
    # Start the gateway
    python -m gateway.run
    
    # Or from CLI
    python cli.py --gateway
"""

import asyncio
import logging
import os
import re
import sys
import signal
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Resolve Hermes home directory (respects HERMES_HOME override)
_hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))

# Load environment variables from ~/.hermes/.env first
from dotenv import load_dotenv
_env_path = _hermes_home / '.env'
if _env_path.exists():
    try:
        load_dotenv(_env_path, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(_env_path, encoding="latin-1")
# Also try project .env as fallback
load_dotenv()

# Bridge config.yaml values into the environment so os.getenv() picks them up.
# config.yaml is authoritative for terminal settings — overrides .env.
_config_path = _hermes_home / 'config.yaml'
if _config_path.exists():
    try:
        import yaml as _yaml
        with open(_config_path, encoding="utf-8") as _f:
            _cfg = _yaml.safe_load(_f) or {}
        # Top-level simple values (fallback only — don't override .env)
        for _key, _val in _cfg.items():
            if isinstance(_val, (str, int, float, bool)) and _key not in os.environ:
                os.environ[_key] = str(_val)
        # Terminal config is nested — bridge to TERMINAL_* env vars.
        # config.yaml overrides .env for these since it's the documented config path.
        _terminal_cfg = _cfg.get("terminal", {})
        if _terminal_cfg and isinstance(_terminal_cfg, dict):
            _terminal_env_map = {
                "backend": "TERMINAL_ENV",
                "cwd": "TERMINAL_CWD",
                "timeout": "TERMINAL_TIMEOUT",
                "lifetime_seconds": "TERMINAL_LIFETIME_SECONDS",
                "docker_image": "TERMINAL_DOCKER_IMAGE",
                "singularity_image": "TERMINAL_SINGULARITY_IMAGE",
                "modal_image": "TERMINAL_MODAL_IMAGE",
                "daytona_image": "TERMINAL_DAYTONA_IMAGE",
                "ssh_host": "TERMINAL_SSH_HOST",
                "ssh_user": "TERMINAL_SSH_USER",
                "ssh_port": "TERMINAL_SSH_PORT",
                "ssh_key": "TERMINAL_SSH_KEY",
                "container_cpu": "TERMINAL_CONTAINER_CPU",
                "container_memory": "TERMINAL_CONTAINER_MEMORY",
                "container_disk": "TERMINAL_CONTAINER_DISK",
                "container_persistent": "TERMINAL_CONTAINER_PERSISTENT",
                "docker_volumes": "TERMINAL_DOCKER_VOLUMES",
                "sandbox_dir": "TERMINAL_SANDBOX_DIR",
            }
            for _cfg_key, _env_var in _terminal_env_map.items():
                if _cfg_key in _terminal_cfg:
                    _val = _terminal_cfg[_cfg_key]
                    if isinstance(_val, list):
                        os.environ[_env_var] = json.dumps(_val)
                    else:
                        os.environ[_env_var] = str(_val)
        _compression_cfg = _cfg.get("compression", {})
        if _compression_cfg and isinstance(_compression_cfg, dict):
            _compression_env_map = {
                "enabled": "CONTEXT_COMPRESSION_ENABLED",
                "threshold": "CONTEXT_COMPRESSION_THRESHOLD",
                "summary_model": "CONTEXT_COMPRESSION_MODEL",
                "summary_provider": "CONTEXT_COMPRESSION_PROVIDER",
            }
            for _cfg_key, _env_var in _compression_env_map.items():
                if _cfg_key in _compression_cfg:
                    os.environ[_env_var] = str(_compression_cfg[_cfg_key])
        # Auxiliary model overrides (vision, web_extract).
        # Each task has provider + model; bridge non-default values to env vars.
        _auxiliary_cfg = _cfg.get("auxiliary", {})
        if _auxiliary_cfg and isinstance(_auxiliary_cfg, dict):
            _aux_task_env = {
                "vision":      ("AUXILIARY_VISION_PROVIDER",      "AUXILIARY_VISION_MODEL"),
                "web_extract": ("AUXILIARY_WEB_EXTRACT_PROVIDER",  "AUXILIARY_WEB_EXTRACT_MODEL"),
            }
            for _task_key, (_prov_env, _model_env) in _aux_task_env.items():
                _task_cfg = _auxiliary_cfg.get(_task_key, {})
                if not isinstance(_task_cfg, dict):
                    continue
                _prov = str(_task_cfg.get("provider", "")).strip()
                _model = str(_task_cfg.get("model", "")).strip()
                if _prov and _prov != "auto":
                    os.environ[_prov_env] = _prov
                if _model:
                    os.environ[_model_env] = _model
        _agent_cfg = _cfg.get("agent", {})
        if _agent_cfg and isinstance(_agent_cfg, dict):
            if "max_turns" in _agent_cfg:
                os.environ["HERMES_MAX_ITERATIONS"] = str(_agent_cfg["max_turns"])
        # Timezone: bridge config.yaml → HERMES_TIMEZONE env var.
        # HERMES_TIMEZONE from .env takes precedence (already in os.environ).
        _tz_cfg = _cfg.get("timezone", "")
        if _tz_cfg and isinstance(_tz_cfg, str) and "HERMES_TIMEZONE" not in os.environ:
            os.environ["HERMES_TIMEZONE"] = _tz_cfg.strip()
        # Security settings
        _security_cfg = _cfg.get("security", {})
        if isinstance(_security_cfg, dict):
            _redact = _security_cfg.get("redact_secrets")
            if _redact is not None:
                os.environ["HERMES_REDACT_SECRETS"] = str(_redact).lower()
    except Exception:
        pass  # Non-fatal; gateway can still run with .env values

# Gateway runs in quiet mode - suppress debug output and use cwd directly (no temp dirs)
os.environ["HERMES_QUIET"] = "1"

# Enable interactive exec approval for dangerous commands on messaging platforms
os.environ["HERMES_EXEC_ASK"] = "1"

# Set terminal working directory for messaging platforms.
# If the user set an explicit path in config.yaml (not "." or "auto"),
# respect it. Otherwise use MESSAGING_CWD or default to home directory.
_configured_cwd = os.environ.get("TERMINAL_CWD", "")
if not _configured_cwd or _configured_cwd in (".", "auto", "cwd"):
    messaging_cwd = os.getenv("MESSAGING_CWD") or str(Path.home())
    os.environ["TERMINAL_CWD"] = messaging_cwd

from gateway.config import (
    Platform,
    GatewayConfig,
    load_gateway_config,
)
from gateway.session import (
    SessionStore,
    SessionSource,
    SessionContext,
    build_session_context,
    build_session_context_prompt,
    build_session_key,
)
from gateway.delivery import DeliveryRouter, DeliveryTarget
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType

logger = logging.getLogger(__name__)


def _resolve_runtime_agent_kwargs() -> dict:
    """Resolve provider credentials for gateway-created AIAgent instances."""
    from hermes_cli.runtime_provider import (
        resolve_runtime_provider,
        format_runtime_provider_error,
    )

    try:
        runtime = resolve_runtime_provider(
            requested=os.getenv("HERMES_INFERENCE_PROVIDER"),
        )
    except Exception as exc:
        raise RuntimeError(format_runtime_provider_error(exc)) from exc

    return {
        "api_key": runtime.get("api_key"),
        "base_url": runtime.get("base_url"),
        "provider": runtime.get("provider"),
        "api_mode": runtime.get("api_mode"),
    }


def _resolve_gateway_model() -> str:
    """Read model from env/config — mirrors the resolution in _run_agent_sync.

    Without this, temporary AIAgent instances (memory flush, /compress) fall
    back to the hardcoded default ("anthropic/claude-opus-4.6") which fails
    when the active provider is openai-codex.
    """
    model = os.getenv("HERMES_MODEL") or os.getenv("LLM_MODEL") or "anthropic/claude-opus-4.6"
    try:
        import yaml as _y
        _cfg_path = _hermes_home / "config.yaml"
        if _cfg_path.exists():
            with open(_cfg_path, encoding="utf-8") as _f:
                _cfg = _y.safe_load(_f) or {}
            _model_cfg = _cfg.get("model", {})
            if isinstance(_model_cfg, str):
                model = _model_cfg
            elif isinstance(_model_cfg, dict):
                model = _model_cfg.get("default", model)
    except Exception:
        pass
    return model


class GatewayRunner:
    """
    Main gateway controller.
    
    Manages the lifecycle of all platform adapters and routes
    messages to/from the agent.
    """
    
    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or load_gateway_config()
        self.adapters: Dict[Platform, BasePlatformAdapter] = {}

        # Load ephemeral config from config.yaml / env vars.
        # Both are injected at API-call time only and never persisted.
        self._prefill_messages = self._load_prefill_messages()
        self._ephemeral_system_prompt = self._load_ephemeral_system_prompt()
        self._reasoning_config = self._load_reasoning_config()
        self._show_reasoning = self._load_show_reasoning()
        self._provider_routing = self._load_provider_routing()
        self._fallback_model = self._load_fallback_model()

        # Wire process registry into session store for reset protection
        from tools.process_registry import process_registry
        self.session_store = SessionStore(
            self.config.sessions_dir, self.config,
            has_active_processes_fn=lambda key: process_registry.has_active_for_session(key),
        )
        self.delivery_router = DeliveryRouter(self.config)
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Track running agents per session for interrupt support
        # Key: session_key, Value: AIAgent instance
        self._running_agents: Dict[str, Any] = {}
        self._pending_messages: Dict[str, str] = {}  # Queued messages during interrupt
        
        # Track pending exec approvals per session
        # Key: session_key, Value: {"command": str, "pattern_key": str}
        self._pending_approvals: Dict[str, Dict[str, str]] = {}

        # Persistent Honcho managers keyed by gateway session key.
        # This preserves write_frequency="session" semantics across short-lived
        # per-message AIAgent instances.
        self._honcho_managers: Dict[str, Any] = {}
        self._honcho_configs: Dict[str, Any] = {}

        # Ensure tirith security scanner is available (downloads if needed)
        try:
            from tools.tirith_security import ensure_installed
            ensure_installed()
        except Exception:
            pass  # Non-fatal — fail-open at scan time if unavailable
        
        # Initialize session database for session_search tool support
        self._session_db = None
        try:
            from hermes_state import SessionDB
            self._session_db = SessionDB()
        except Exception as e:
            logger.debug("SQLite session store not available: %s", e)
        
        # DM pairing store for code-based user authorization
        from gateway.pairing import PairingStore
        self.pairing_store = PairingStore()
        
        # Event hook system
        from gateway.hooks import HookRegistry
        self.hooks = HookRegistry()

    def _get_or_create_gateway_honcho(self, session_key: str):
        """Return a persistent Honcho manager/config pair for this gateway session."""
        if not hasattr(self, "_honcho_managers"):
            self._honcho_managers = {}
        if not hasattr(self, "_honcho_configs"):
            self._honcho_configs = {}

        if session_key in self._honcho_managers:
            return self._honcho_managers[session_key], self._honcho_configs.get(session_key)

        try:
            from honcho_integration.client import HonchoClientConfig, get_honcho_client
            from honcho_integration.session import HonchoSessionManager

            hcfg = HonchoClientConfig.from_global_config()
            if not hcfg.enabled or not hcfg.api_key:
                return None, hcfg

            client = get_honcho_client(hcfg)
            manager = HonchoSessionManager(
                honcho=client,
                config=hcfg,
                context_tokens=hcfg.context_tokens,
            )
            self._honcho_managers[session_key] = manager
            self._honcho_configs[session_key] = hcfg
            return manager, hcfg
        except Exception as e:
            logger.debug("Gateway Honcho init failed for %s: %s", session_key, e)
            return None, None

    def _shutdown_gateway_honcho(self, session_key: str) -> None:
        """Flush and close the persistent Honcho manager for a gateway session."""
        managers = getattr(self, "_honcho_managers", None)
        configs = getattr(self, "_honcho_configs", None)
        if managers is None or configs is None:
            return

        manager = managers.pop(session_key, None)
        configs.pop(session_key, None)
        if not manager:
            return
        try:
            manager.shutdown()
        except Exception as e:
            logger.debug("Gateway Honcho shutdown failed for %s: %s", session_key, e)

    def _shutdown_all_gateway_honcho(self) -> None:
        """Flush and close all persistent Honcho managers."""
        managers = getattr(self, "_honcho_managers", None)
        if not managers:
            return
        for session_key in list(managers.keys()):
            self._shutdown_gateway_honcho(session_key)
    
    def _flush_memories_for_session(self, old_session_id: str):
        """Prompt the agent to save memories/skills before context is lost.

        Synchronous worker — meant to be called via run_in_executor from
        an async context so it doesn't block the event loop.
        """
        try:
            history = self.session_store.load_transcript(old_session_id)
            if not history or len(history) < 4:
                return

            from run_agent import AIAgent
            runtime_kwargs = _resolve_runtime_agent_kwargs()
            if not runtime_kwargs.get("api_key"):
                return

            # Resolve model from config — AIAgent's default is OpenRouter-
            # formatted ("anthropic/claude-opus-4.6") which fails when the
            # active provider is openai-codex.
            model = _resolve_gateway_model()

            tmp_agent = AIAgent(
                **runtime_kwargs,
                model=model,
                max_iterations=8,
                quiet_mode=True,
                enabled_toolsets=["memory", "skills"],
                session_id=old_session_id,
            )

            # Build conversation history from transcript
            msgs = [
                {"role": m.get("role"), "content": m.get("content")}
                for m in history
                if m.get("role") in ("user", "assistant") and m.get("content")
            ]

            # Give the agent a real turn to think about what to save
            flush_prompt = (
                "[System: This session is about to be automatically reset due to "
                "inactivity or a scheduled daily reset. The conversation context "
                "will be cleared after this turn.\n\n"
                "Review the conversation above and:\n"
                "1. Save any important facts, preferences, or decisions to memory "
                "(user profile or your notes) that would be useful in future sessions.\n"
                "2. If you discovered a reusable workflow or solved a non-trivial "
                "problem, consider saving it as a skill.\n"
                "3. If nothing is worth saving, that's fine — just skip.\n\n"
                "Do NOT respond to the user. Just use the memory and skill_manage "
                "tools if needed, then stop.]"
            )

            tmp_agent.run_conversation(
                user_message=flush_prompt,
                conversation_history=msgs,
            )
            logger.info("Pre-reset memory flush completed for session %s", old_session_id)
            # Flush any queued Honcho writes before the session is dropped
            if getattr(tmp_agent, '_honcho', None):
                try:
                    tmp_agent._honcho.shutdown()
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Pre-reset memory flush failed for session %s: %s", old_session_id, e)

    async def _async_flush_memories(self, old_session_id: str):
        """Run the sync memory flush in a thread pool so it won't block the event loop."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._flush_memories_for_session, old_session_id)
    
    @staticmethod
    def _load_prefill_messages() -> List[Dict[str, Any]]:
        """Load ephemeral prefill messages from config or env var.
        
        Checks HERMES_PREFILL_MESSAGES_FILE env var first, then falls back to
        the prefill_messages_file key in ~/.hermes/config.yaml.
        Relative paths are resolved from ~/.hermes/.
        """
        import json as _json
        file_path = os.getenv("HERMES_PREFILL_MESSAGES_FILE", "")
        if not file_path:
            try:
                import yaml as _y
                cfg_path = _hermes_home / "config.yaml"
                if cfg_path.exists():
                    with open(cfg_path, encoding="utf-8") as _f:
                        cfg = _y.safe_load(_f) or {}
                    file_path = cfg.get("prefill_messages_file", "")
            except Exception:
                pass
        if not file_path:
            return []
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            path = _hermes_home / path
        if not path.exists():
            logger.warning("Prefill messages file not found: %s", path)
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            if not isinstance(data, list):
                logger.warning("Prefill messages file must contain a JSON array: %s", path)
                return []
            return data
        except Exception as e:
            logger.warning("Failed to load prefill messages from %s: %s", path, e)
            return []

    @staticmethod
    def _load_ephemeral_system_prompt() -> str:
        """Load ephemeral system prompt from config or env var.
        
        Checks HERMES_EPHEMERAL_SYSTEM_PROMPT env var first, then falls back to
        agent.system_prompt in ~/.hermes/config.yaml.
        """
        prompt = os.getenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "")
        if prompt:
            return prompt
        try:
            import yaml as _y
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as _f:
                    cfg = _y.safe_load(_f) or {}
                return (cfg.get("agent", {}).get("system_prompt", "") or "").strip()
        except Exception:
            pass
        return ""

    @staticmethod
    def _load_reasoning_config() -> dict | None:
        """Load reasoning effort from config or env var.
        
        Checks HERMES_REASONING_EFFORT env var first, then agent.reasoning_effort
        in config.yaml. Valid: "xhigh", "high", "medium", "low", "minimal", "none".
        Returns None to use default (medium).
        """
        effort = os.getenv("HERMES_REASONING_EFFORT", "")
        if not effort:
            try:
                import yaml as _y
                cfg_path = _hermes_home / "config.yaml"
                if cfg_path.exists():
                    with open(cfg_path, encoding="utf-8") as _f:
                        cfg = _y.safe_load(_f) or {}
                    effort = str(cfg.get("agent", {}).get("reasoning_effort", "") or "").strip()
            except Exception:
                pass
        if not effort:
            return None
        effort = effort.lower().strip()
        if effort == "none":
            return {"enabled": False}
        valid = ("xhigh", "high", "medium", "low", "minimal")
        if effort in valid:
            return {"enabled": True, "effort": effort}
        logger.warning("Unknown reasoning_effort '%s', using default (medium)", effort)
        return None

    @staticmethod
    def _load_show_reasoning() -> bool:
        """Load show_reasoning toggle from config.yaml display section."""
        try:
            import yaml as _y
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as _f:
                    cfg = _y.safe_load(_f) or {}
                return bool(cfg.get("display", {}).get("show_reasoning", False))
        except Exception:
            pass
        return False

    @staticmethod
    def _load_background_notifications_mode() -> str:
        """Load background process notification mode from config or env var.

        Modes:
          - ``all``    — push running-output updates *and* the final message (default)
          - ``result`` — only the final completion message (regardless of exit code)
          - ``error``  — only the final message when exit code is non-zero
          - ``off``    — no watcher messages at all
        """
        mode = os.getenv("HERMES_BACKGROUND_NOTIFICATIONS", "")
        if not mode:
            try:
                import yaml as _y
                cfg_path = _hermes_home / "config.yaml"
                if cfg_path.exists():
                    with open(cfg_path, encoding="utf-8") as _f:
                        cfg = _y.safe_load(_f) or {}
                    raw = cfg.get("display", {}).get("background_process_notifications")
                    if raw is False:
                        mode = "off"
                    elif raw not in (None, ""):
                        mode = str(raw)
            except Exception:
                pass
        mode = (mode or "all").strip().lower()
        valid = {"all", "result", "error", "off"}
        if mode not in valid:
            logger.warning(
                "Unknown background_process_notifications '%s', defaulting to 'all'",
                mode,
            )
            return "all"
        return mode

    @staticmethod
    def _load_provider_routing() -> dict:
        """Load OpenRouter provider routing preferences from config.yaml."""
        try:
            import yaml as _y
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as _f:
                    cfg = _y.safe_load(_f) or {}
                return cfg.get("provider_routing", {}) or {}
        except Exception:
            pass
        return {}

    @staticmethod
    def _load_fallback_model() -> dict | None:
        """Load fallback model config from config.yaml.

        Returns a dict with 'provider' and 'model' keys, or None if
        not configured / both fields empty.
        """
        try:
            import yaml as _y
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as _f:
                    cfg = _y.safe_load(_f) or {}
                fb = cfg.get("fallback_model", {}) or {}
                if fb.get("provider") and fb.get("model"):
                    return fb
        except Exception:
            pass
        return None

    async def start(self) -> bool:
        """
        Start the gateway and all configured platform adapters.
        
        Returns True if at least one adapter connected successfully.
        """
        logger.info("Starting Hermes Gateway...")
        logger.info("Session storage: %s", self.config.sessions_dir)
        
        # Warn if no user allowlists are configured and open access is not opted in
        _any_allowlist = any(
            os.getenv(v)
            for v in ("TELEGRAM_ALLOWED_USERS", "DISCORD_ALLOWED_USERS",
                       "WHATSAPP_ALLOWED_USERS", "SLACK_ALLOWED_USERS",
                       "GATEWAY_ALLOWED_USERS")
        )
        _allow_all = os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in ("true", "1", "yes")
        if not _any_allowlist and not _allow_all:
            logger.warning(
                "No user allowlists configured. All unauthorized users will be denied. "
                "Set GATEWAY_ALLOW_ALL_USERS=true in ~/.hermes/.env to allow open access, "
                "or configure platform allowlists (e.g., TELEGRAM_ALLOWED_USERS=your_id)."
            )
        
        # Discover and load event hooks
        self.hooks.discover_and_load()
        
        # Recover background processes from checkpoint (crash recovery)
        try:
            from tools.process_registry import process_registry
            recovered = process_registry.recover_from_checkpoint()
            if recovered:
                logger.info("Recovered %s background process(es) from previous run", recovered)
        except Exception as e:
            logger.warning("Process checkpoint recovery: %s", e)
        
        connected_count = 0
        
        # Initialize and connect each configured platform
        for platform, platform_config in self.config.platforms.items():
            if not platform_config.enabled:
                continue
            
            adapter = self._create_adapter(platform, platform_config)
            if not adapter:
                logger.warning("No adapter available for %s", platform.value)
                continue
            
            # Set up message handler
            adapter.set_message_handler(self._handle_message)
            
            # Try to connect
            logger.info("Connecting to %s...", platform.value)
            try:
                success = await adapter.connect()
                if success:
                    self.adapters[platform] = adapter
                    connected_count += 1
                    logger.info("✓ %s connected", platform.value)
                else:
                    logger.warning("✗ %s failed to connect", platform.value)
            except Exception as e:
                logger.error("✗ %s error: %s", platform.value, e)
        
        if connected_count == 0:
            logger.warning("No messaging platforms connected.")
            logger.info("Gateway will continue running for cron job execution.")
        
        # Update delivery router with adapters
        self.delivery_router.adapters = self.adapters
        
        self._running = True
        
        # Emit gateway:startup hook
        hook_count = len(self.hooks.loaded_hooks)
        if hook_count:
            logger.info("%s hook(s) loaded", hook_count)
        await self.hooks.emit("gateway:startup", {
            "platforms": [p.value for p in self.adapters.keys()],
        })
        
        if connected_count > 0:
            logger.info("Gateway running with %s platform(s)", connected_count)
        
        # Build initial channel directory for send_message name resolution
        try:
            from gateway.channel_directory import build_channel_directory
            directory = build_channel_directory(self.adapters)
            ch_count = sum(len(chs) for chs in directory.get("platforms", {}).values())
            logger.info("Channel directory built: %d target(s)", ch_count)
        except Exception as e:
            logger.warning("Channel directory build failed: %s", e)
        
        # Check if we're restarting after a /update command
        await self._send_update_notification()

        # Start background session expiry watcher for proactive memory flushing
        asyncio.create_task(self._session_expiry_watcher())

        logger.info("Press Ctrl+C to stop")
        
        return True
    
    async def _session_expiry_watcher(self, interval: int = 300):
        """Background task that proactively flushes memories for expired sessions.
        
        Runs every `interval` seconds (default 5 min).  For each session that
        has expired according to its reset policy, flushes memories in a thread
        pool and marks the session so it won't be flushed again.

        This means memories are already saved by the time the user sends their
        next message, so there's no blocking delay.
        """
        await asyncio.sleep(60)  # initial delay — let the gateway fully start
        while self._running:
            try:
                self.session_store._ensure_loaded()
                for key, entry in list(self.session_store._entries.items()):
                    if entry.session_id in self.session_store._pre_flushed_sessions:
                        continue  # already flushed this session
                    if not self.session_store._is_session_expired(entry):
                        continue  # session still active
                    # Session has expired — flush memories in the background
                    logger.info(
                        "Session %s expired (key=%s), flushing memories proactively",
                        entry.session_id, key,
                    )
                    try:
                        await self._async_flush_memories(entry.session_id)
                        self._shutdown_gateway_honcho(key)
                        self.session_store._pre_flushed_sessions.add(entry.session_id)
                    except Exception as e:
                        logger.debug("Proactive memory flush failed for %s: %s", entry.session_id, e)
            except Exception as e:
                logger.debug("Session expiry watcher error: %s", e)
            # Sleep in small increments so we can stop quickly
            for _ in range(interval):
                if not self._running:
                    break
                await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the gateway and disconnect all adapters."""
        logger.info("Stopping gateway...")
        self._running = False
        
        for platform, adapter in self.adapters.items():
            try:
                await adapter.disconnect()
                logger.info("✓ %s disconnected", platform.value)
            except Exception as e:
                logger.error("✗ %s disconnect error: %s", platform.value, e)

        self.adapters.clear()
        self._shutdown_all_gateway_honcho()
        self._shutdown_event.set()
        
        from gateway.status import remove_pid_file
        remove_pid_file()
        
        logger.info("Gateway stopped")
    
    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()
    
    def _create_adapter(
        self, 
        platform: Platform, 
        config: Any
    ) -> Optional[BasePlatformAdapter]:
        """Create the appropriate adapter for a platform."""
        if platform == Platform.TELEGRAM:
            from gateway.platforms.telegram import TelegramAdapter, check_telegram_requirements
            if not check_telegram_requirements():
                logger.warning("Telegram: python-telegram-bot not installed")
                return None
            return TelegramAdapter(config)
        
        elif platform == Platform.DISCORD:
            from gateway.platforms.discord import DiscordAdapter, check_discord_requirements
            if not check_discord_requirements():
                logger.warning("Discord: discord.py not installed")
                return None
            return DiscordAdapter(config)
        
        elif platform == Platform.WHATSAPP:
            from gateway.platforms.whatsapp import WhatsAppAdapter, check_whatsapp_requirements
            if not check_whatsapp_requirements():
                logger.warning("WhatsApp: Node.js not installed or bridge not configured")
                return None
            return WhatsAppAdapter(config)
        
        elif platform == Platform.SLACK:
            from gateway.platforms.slack import SlackAdapter, check_slack_requirements
            if not check_slack_requirements():
                logger.warning("Slack: slack-bolt not installed. Run: pip install 'hermes-agent[slack]'")
                return None
            return SlackAdapter(config)

        elif platform == Platform.SIGNAL:
            from gateway.platforms.signal import SignalAdapter, check_signal_requirements
            if not check_signal_requirements():
                logger.warning("Signal: SIGNAL_HTTP_URL or SIGNAL_ACCOUNT not configured")
                return None
            return SignalAdapter(config)

        elif platform == Platform.HOMEASSISTANT:
            from gateway.platforms.homeassistant import HomeAssistantAdapter, check_ha_requirements
            if not check_ha_requirements():
                logger.warning("HomeAssistant: aiohttp not installed or HASS_TOKEN not set")
                return None
            return HomeAssistantAdapter(config)

        elif platform == Platform.EMAIL:
            from gateway.platforms.email import EmailAdapter, check_email_requirements
            if not check_email_requirements():
                logger.warning("Email: EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_IMAP_HOST, or EMAIL_SMTP_HOST not set")
                return None
            return EmailAdapter(config)

        return None
    
    def _is_user_authorized(self, source: SessionSource) -> bool:
        """
        Check if a user is authorized to use the bot.
        
        Checks in order:
        1. Per-platform allow-all flag (e.g., DISCORD_ALLOW_ALL_USERS=true)
        2. Environment variable allowlists (TELEGRAM_ALLOWED_USERS, etc.)
        3. DM pairing approved list
        4. Global allow-all (GATEWAY_ALLOW_ALL_USERS=true)
        5. Default: deny
        """
        # Home Assistant events are system-generated (state changes), not
        # user-initiated messages.  The HASS_TOKEN already authenticates the
        # connection, so HA events are always authorized.
        if source.platform == Platform.HOMEASSISTANT:
            return True

        user_id = source.user_id
        if not user_id:
            return False

        platform_env_map = {
            Platform.TELEGRAM: "TELEGRAM_ALLOWED_USERS",
            Platform.DISCORD: "DISCORD_ALLOWED_USERS",
            Platform.WHATSAPP: "WHATSAPP_ALLOWED_USERS",
            Platform.SLACK: "SLACK_ALLOWED_USERS",
            Platform.SIGNAL: "SIGNAL_ALLOWED_USERS",
            Platform.EMAIL: "EMAIL_ALLOWED_USERS",
        }
        platform_allow_all_map = {
            Platform.TELEGRAM: "TELEGRAM_ALLOW_ALL_USERS",
            Platform.DISCORD: "DISCORD_ALLOW_ALL_USERS",
            Platform.WHATSAPP: "WHATSAPP_ALLOW_ALL_USERS",
            Platform.SLACK: "SLACK_ALLOW_ALL_USERS",
            Platform.SIGNAL: "SIGNAL_ALLOW_ALL_USERS",
            Platform.EMAIL: "EMAIL_ALLOW_ALL_USERS",
        }

        # Per-platform allow-all flag (e.g., DISCORD_ALLOW_ALL_USERS=true)
        platform_allow_all_var = platform_allow_all_map.get(source.platform, "")
        if platform_allow_all_var and os.getenv(platform_allow_all_var, "").lower() in ("true", "1", "yes"):
            return True

        # Check pairing store (always checked, regardless of allowlists)
        platform_name = source.platform.value if source.platform else ""
        if self.pairing_store.is_approved(platform_name, user_id):
            return True

        # Check platform-specific and global allowlists
        platform_allowlist = os.getenv(platform_env_map.get(source.platform, ""), "").strip()
        global_allowlist = os.getenv("GATEWAY_ALLOWED_USERS", "").strip()

        if not platform_allowlist and not global_allowlist:
            # No allowlists configured -- check global allow-all flag
            return os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in ("true", "1", "yes")

        # Check if user is in any allowlist
        allowed_ids = set()
        if platform_allowlist:
            allowed_ids.update(uid.strip() for uid in platform_allowlist.split(",") if uid.strip())
        if global_allowlist:
            allowed_ids.update(uid.strip() for uid in global_allowlist.split(",") if uid.strip())

        # WhatsApp JIDs have @s.whatsapp.net suffix — strip it for comparison
        check_ids = {user_id}
        if "@" in user_id:
            check_ids.add(user_id.split("@")[0])
        return bool(check_ids & allowed_ids)
    
    async def _handle_message(self, event: MessageEvent) -> Optional[str]:
        """
        Handle an incoming message from any platform.
        
        This is the core message processing pipeline:
        1. Check user authorization
        2. Check for commands (/new, /reset, etc.)
        3. Check for running agent and interrupt if needed
        4. Get or create session
        5. Build context for agent
        6. Run agent conversation
        7. Return response
        """
        source = event.source
        
        # Check if user is authorized
        if not self._is_user_authorized(source):
            logger.warning("Unauthorized user: %s (%s) on %s", source.user_id, source.user_name, source.platform.value)
            # In DMs: offer pairing code. In groups: silently ignore.
            if source.chat_type == "dm":
                platform_name = source.platform.value if source.platform else "unknown"
                code = self.pairing_store.generate_code(
                    platform_name, source.user_id, source.user_name or ""
                )
                if code:
                    adapter = self.adapters.get(source.platform)
                    if adapter:
                        await adapter.send(
                            source.chat_id,
                            f"Hi~ I don't recognize you yet!\n\n"
                            f"Here's your pairing code: `{code}`\n\n"
                            f"Ask the bot owner to run:\n"
                            f"`hermes pairing approve {platform_name} {code}`"
                        )
                else:
                    adapter = self.adapters.get(source.platform)
                    if adapter:
                        await adapter.send(
                            source.chat_id,
                            "Too many pairing requests right now~ "
                            "Please try again later!"
                        )
            return None
        
        # PRIORITY: If an agent is already running for this session, interrupt it
        # immediately. This is before command parsing to minimize latency -- the
        # user's "stop" message reaches the agent as fast as possible.
        _quick_key = build_session_key(source)
        if _quick_key in self._running_agents:
            running_agent = self._running_agents[_quick_key]
            logger.debug("PRIORITY interrupt for session %s", _quick_key[:20])
            running_agent.interrupt(event.text)
            if _quick_key in self._pending_messages:
                self._pending_messages[_quick_key] += "\n" + event.text
            else:
                self._pending_messages[_quick_key] = event.text
            return None
        
        # Check for commands
        command = event.get_command()
        
        # Emit command:* hook for any recognized slash command
        _known_commands = {"new", "reset", "help", "status", "stop", "model",
                          "personality", "retry", "undo", "sethome", "set-home",
                          "compress", "usage", "insights", "reload-mcp", "reload_mcp",
                          "update", "title", "resume", "provider", "rollback",
                          "background", "reasoning"}
        if command and command in _known_commands:
            await self.hooks.emit(f"command:{command}", {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "command": command,
                "args": event.get_command_args().strip(),
            })
        
        if command in ["new", "reset"]:
            return await self._handle_reset_command(event)
        
        if command == "help":
            return await self._handle_help_command(event)
        
        if command == "status":
            return await self._handle_status_command(event)
        
        if command == "stop":
            return await self._handle_stop_command(event)
        
        if command == "model":
            return await self._handle_model_command(event)
        
        if command == "provider":
            return await self._handle_provider_command(event)
        
        if command == "personality":
            return await self._handle_personality_command(event)
        
        if command == "retry":
            return await self._handle_retry_command(event)
        
        if command == "undo":
            return await self._handle_undo_command(event)
        
        if command in ["sethome", "set-home"]:
            return await self._handle_set_home_command(event)

        if command == "compress":
            return await self._handle_compress_command(event)

        if command == "usage":
            return await self._handle_usage_command(event)

        if command == "insights":
            return await self._handle_insights_command(event)

        if command in ("reload-mcp", "reload_mcp"):
            return await self._handle_reload_mcp_command(event)

        if command == "update":
            return await self._handle_update_command(event)

        if command == "title":
            return await self._handle_title_command(event)

        if command == "resume":
            return await self._handle_resume_command(event)

        if command == "rollback":
            return await self._handle_rollback_command(event)

        if command == "background":
            return await self._handle_background_command(event)

        if command == "reasoning":
            return await self._handle_reasoning_command(event)
        
        # User-defined quick commands (bypass agent loop, no LLM call)
        if command:
            quick_commands = self.config.get("quick_commands", {})
            if command in quick_commands:
                qcmd = quick_commands[command]
                if qcmd.get("type") == "exec":
                    exec_cmd = qcmd.get("command", "")
                    if exec_cmd:
                        try:
                            proc = await asyncio.create_subprocess_shell(
                                exec_cmd,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                            )
                            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                            output = (stdout or stderr).decode().strip()
                            return output if output else "Command returned no output."
                        except asyncio.TimeoutError:
                            return "Quick command timed out (30s)."
                        except Exception as e:
                            return f"Quick command error: {e}"
                    else:
                        return f"Quick command '/{command}' has no command defined."
                else:
                    return f"Quick command '/{command}' has unsupported type (only 'exec' is supported)."

        # Skill slash commands: /skill-name loads the skill and sends to agent
        if command:
            try:
                from agent.skill_commands import get_skill_commands, build_skill_invocation_message
                skill_cmds = get_skill_commands()
                cmd_key = f"/{command}"
                if cmd_key in skill_cmds:
                    user_instruction = event.get_command_args().strip()
                    msg = build_skill_invocation_message(
                        cmd_key, user_instruction, task_id=session_key
                    )
                    if msg:
                        event.text = msg
                        # Fall through to normal message processing with skill content
            except Exception as e:
                logger.debug("Skill command check failed (non-fatal): %s", e)
        
        # Check for pending exec approval responses
        session_key_preview = build_session_key(source)
        if session_key_preview in self._pending_approvals:
            user_text = event.text.strip().lower()
            if user_text in ("yes", "y", "approve", "ok", "go", "do it"):
                approval = self._pending_approvals.pop(session_key_preview)
                cmd = approval["command"]
                pattern_keys = approval.get("pattern_keys", [])
                if not pattern_keys:
                    pk = approval.get("pattern_key", "")
                    pattern_keys = [pk] if pk else []
                logger.info("User approved dangerous command: %s...", cmd[:60])
                from tools.terminal_tool import terminal_tool
                from tools.approval import approve_session
                for pk in pattern_keys:
                    approve_session(session_key_preview, pk)
                result = terminal_tool(command=cmd, force=True)
                return f"✅ Command approved and executed.\n\n```\n{result[:3500]}\n```"
            elif user_text in ("no", "n", "deny", "cancel", "nope"):
                self._pending_approvals.pop(session_key_preview)
                return "❌ Command denied."
            elif user_text in ("full", "show", "view", "show full", "view full"):
                # Show full command without consuming the approval
                cmd = self._pending_approvals[session_key_preview]["command"]
                return f"Full command:\n\n```\n{cmd}\n```\n\nReply yes/no to approve or deny."
            # If it's not clearly an approval/denial, fall through to normal processing
        
        # Get or create session
        session_entry = self.session_store.get_or_create_session(source)
        session_key = session_entry.session_key
        
        # Emit session:start for new or auto-reset sessions
        _is_new_session = (
            session_entry.created_at == session_entry.updated_at
            or getattr(session_entry, "was_auto_reset", False)
        )
        if _is_new_session:
            await self.hooks.emit("session:start", {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "session_id": session_entry.session_id,
                "session_key": session_key,
            })
        
        # Build session context
        context = build_session_context(source, self.config, session_entry)
        
        # Set environment variables for tools
        self._set_session_env(context)
        
        # Build the context prompt to inject
        context_prompt = build_session_context_prompt(context)
        
        # If the previous session expired and was auto-reset, prepend a notice
        # so the agent knows this is a fresh conversation (not an intentional /reset).
        if getattr(session_entry, 'was_auto_reset', False):
            context_prompt = (
                "[System note: The user's previous session expired due to inactivity. "
                "This is a fresh conversation with no prior context.]\n\n"
                + context_prompt
            )
            session_entry.was_auto_reset = False
        
        # Load conversation history from transcript
        history = self.session_store.load_transcript(session_entry.session_id)
        
        # -----------------------------------------------------------------
        # Session hygiene: auto-compress pathologically large transcripts
        #
        # Long-lived gateway sessions can accumulate enough history that
        # every new message rehydrates an oversized transcript, causing
        # repeated truncation/context failures.  Detect this early and
        # compress proactively — before the agent even starts.  (#628)
        #
        # Token source priority:
        # 1. Actual API-reported prompt_tokens from the last turn
        #    (stored in session_entry.last_prompt_tokens)
        # 2. Rough char-based estimate (str(msg)//4) with a 1.4x
        #    safety factor to account for overestimation on tool-heavy
        #    conversations (code/JSON tokenizes at 5-7+ chars/token).
        # -----------------------------------------------------------------
        if history and len(history) >= 4:
            from agent.model_metadata import (
                estimate_messages_tokens_rough,
                get_model_context_length,
            )

            # Read model + compression config from config.yaml.
            # NOTE: hygiene threshold is intentionally HIGHER than the agent's
            # own compressor (0.85 vs 0.50).  Hygiene is a safety net for
            # sessions that grew too large between turns — it fires pre-agent
            # to prevent API failures.  The agent's own compressor handles
            # normal context management during its tool loop with accurate
            # real token counts.  Having hygiene at 0.50 caused premature
            # compression on every turn in long gateway sessions.
            _hyg_model = "anthropic/claude-sonnet-4.6"
            _hyg_threshold_pct = 0.85
            _hyg_compression_enabled = True
            try:
                _hyg_cfg_path = _hermes_home / "config.yaml"
                if _hyg_cfg_path.exists():
                    import yaml as _hyg_yaml
                    with open(_hyg_cfg_path, encoding="utf-8") as _hyg_f:
                        _hyg_data = _hyg_yaml.safe_load(_hyg_f) or {}

                    # Resolve model name (same logic as run_sync)
                    _model_cfg = _hyg_data.get("model", {})
                    if isinstance(_model_cfg, str):
                        _hyg_model = _model_cfg
                    elif isinstance(_model_cfg, dict):
                        _hyg_model = _model_cfg.get("default", _hyg_model)

                    # Read compression settings — only use enabled flag.
                    # The threshold is intentionally separate from the agent's
                    # compression.threshold (hygiene runs higher).
                    _comp_cfg = _hyg_data.get("compression", {})
                    if isinstance(_comp_cfg, dict):
                        _hyg_compression_enabled = str(
                            _comp_cfg.get("enabled", True)
                        ).lower() in ("true", "1", "yes")
            except Exception:
                pass

            # Check env override for disabling compression entirely
            if os.getenv("CONTEXT_COMPRESSION_ENABLED", "").lower() in ("false", "0", "no"):
                _hyg_compression_enabled = False

            if _hyg_compression_enabled:
                _hyg_context_length = get_model_context_length(_hyg_model)
                _compress_token_threshold = int(
                    _hyg_context_length * _hyg_threshold_pct
                )
                _warn_token_threshold = int(_hyg_context_length * 0.95)

                _msg_count = len(history)

                # Prefer actual API-reported tokens from the last turn
                # (stored in session entry) over the rough char-based estimate.
                # The rough estimate (str(msg)//4) overestimates by 30-50% on
                # tool-heavy/code-heavy conversations, causing premature compression.
                _stored_tokens = session_entry.last_prompt_tokens
                if _stored_tokens > 0:
                    _approx_tokens = _stored_tokens
                    _token_source = "actual"
                else:
                    _approx_tokens = estimate_messages_tokens_rough(history)
                    # Apply safety factor only for rough estimates
                    _compress_token_threshold = int(
                        _compress_token_threshold * 1.4
                    )
                    _warn_token_threshold = int(_warn_token_threshold * 1.4)
                    _token_source = "estimated"

                _needs_compress = _approx_tokens >= _compress_token_threshold

                if _needs_compress:
                    logger.info(
                        "Session hygiene: %s messages, ~%s tokens (%s) — auto-compressing "
                        "(threshold: %s%% of %s = %s tokens)",
                        _msg_count, f"{_approx_tokens:,}", _token_source,
                        int(_hyg_threshold_pct * 100),
                        f"{_hyg_context_length:,}",
                        f"{_compress_token_threshold:,}",
                    )

                    _hyg_adapter = self.adapters.get(source.platform)
                    _hyg_meta = {"thread_id": source.thread_id} if source.thread_id else None
                    if _hyg_adapter:
                        try:
                            await _hyg_adapter.send(
                                source.chat_id,
                                f"🗜️ Session is large ({_msg_count} messages, "
                                f"~{_approx_tokens:,} tokens). Auto-compressing...",
                                metadata=_hyg_meta,
                            )
                        except Exception:
                            pass

                    try:
                        from run_agent import AIAgent

                        _hyg_runtime = _resolve_runtime_agent_kwargs()
                        if _hyg_runtime.get("api_key"):
                            _hyg_msgs = [
                                {"role": m.get("role"), "content": m.get("content")}
                                for m in history
                                if m.get("role") in ("user", "assistant")
                                and m.get("content")
                            ]

                            if len(_hyg_msgs) >= 4:
                                _hyg_agent = AIAgent(
                                    **_hyg_runtime,
                                    model=_hyg_model,
                                    max_iterations=4,
                                    quiet_mode=True,
                                    enabled_toolsets=["memory"],
                                    session_id=session_entry.session_id,
                                )

                                loop = asyncio.get_event_loop()
                                _compressed, _ = await loop.run_in_executor(
                                    None,
                                    lambda: _hyg_agent._compress_context(
                                        _hyg_msgs, "",
                                        approx_tokens=_approx_tokens,
                                    ),
                                )

                                self.session_store.rewrite_transcript(
                                    session_entry.session_id, _compressed
                                )
                                # Reset stored token count — transcript was rewritten
                                session_entry.last_prompt_tokens = 0
                                history = _compressed
                                _new_count = len(_compressed)
                                _new_tokens = estimate_messages_tokens_rough(
                                    _compressed
                                )

                                logger.info(
                                    "Session hygiene: compressed %s → %s msgs, "
                                    "~%s → ~%s tokens",
                                    _msg_count, _new_count,
                                    f"{_approx_tokens:,}", f"{_new_tokens:,}",
                                )

                                if _hyg_adapter:
                                    try:
                                        await _hyg_adapter.send(
                                            source.chat_id,
                                            f"🗜️ Compressed: {_msg_count} → "
                                            f"{_new_count} messages, "
                                            f"~{_approx_tokens:,} → "
                                            f"~{_new_tokens:,} tokens",
                                            metadata=_hyg_meta,
                                        )
                                    except Exception:
                                        pass

                                # Still too large after compression — warn user
                                if _new_tokens >= _warn_token_threshold:
                                    logger.warning(
                                        "Session hygiene: still ~%s tokens after "
                                        "compression — suggesting /reset",
                                        f"{_new_tokens:,}",
                                    )
                                    if _hyg_adapter:
                                        try:
                                            await _hyg_adapter.send(
                                                source.chat_id,
                                                "⚠️ Session is still very large "
                                                "after compression "
                                                f"(~{_new_tokens:,} tokens). "
                                                "Consider using /reset to start "
                                                "fresh if you experience issues.",
                                                metadata=_hyg_meta,
                                            )
                                        except Exception:
                                            pass

                    except Exception as e:
                        logger.warning(
                            "Session hygiene auto-compress failed: %s", e
                        )
                        # Compression failed and session is dangerously large
                        if _approx_tokens >= _warn_token_threshold:
                            _hyg_adapter = self.adapters.get(source.platform)
                            _hyg_meta = {"thread_id": source.thread_id} if source.thread_id else None
                            if _hyg_adapter:
                                try:
                                    await _hyg_adapter.send(
                                        source.chat_id,
                                        f"⚠️ Session is very large "
                                        f"({_msg_count} messages, "
                                        f"~{_approx_tokens:,} tokens) and "
                                        "auto-compression failed. Consider "
                                        "using /compress or /reset to avoid "
                                        "issues.",
                                        metadata=_hyg_meta,
                                    )
                                except Exception:
                                    pass

        # First-message onboarding -- only on the very first interaction ever
        if not history and not self.session_store.has_any_sessions():
            context_prompt += (
                "\n\n[System note: This is the user's very first message ever. "
                "Briefly introduce yourself and mention that /help shows available commands. "
                "Keep the introduction concise -- one or two sentences max.]"
            )
        
        # One-time prompt if no home channel is set for this platform
        if not history and source.platform and source.platform != Platform.LOCAL:
            platform_name = source.platform.value
            env_key = f"{platform_name.upper()}_HOME_CHANNEL"
            if not os.getenv(env_key):
                adapter = self.adapters.get(source.platform)
                if adapter:
                    await adapter.send(
                        source.chat_id,
                        f"📬 No home channel is set for {platform_name.title()}. "
                        f"A home channel is where Hermes delivers cron job results "
                        f"and cross-platform messages.\n\n"
                        f"Type /sethome to make this chat your home channel, "
                        f"or ignore to skip."
                    )
        
        # -----------------------------------------------------------------
        # Auto-analyze images sent by the user
        #
        # If the user attached image(s), we run the vision tool eagerly so
        # the conversation model always receives a text description.  The
        # local file path is also included so the model can re-examine the
        # image later with a more targeted question via vision_analyze.
        #
        # We filter to image paths only (by media_type) so that non-image
        # attachments (documents, audio, etc.) are not sent to the vision
        # tool even when they appear in the same message.
        # -----------------------------------------------------------------
        message_text = event.text or ""
        if event.media_urls:
            image_paths = []
            for i, path in enumerate(event.media_urls):
                # Check media_types if available; otherwise infer from message type
                mtype = event.media_types[i] if i < len(event.media_types) else ""
                is_image = (
                    mtype.startswith("image/")
                    or event.message_type == MessageType.PHOTO
                )
                if is_image:
                    image_paths.append(path)
            if image_paths:
                message_text = await self._enrich_message_with_vision(
                    message_text, image_paths
                )
        
        # -----------------------------------------------------------------
        # Auto-transcribe voice/audio messages sent by the user
        # -----------------------------------------------------------------
        if event.media_urls:
            audio_paths = []
            for i, path in enumerate(event.media_urls):
                mtype = event.media_types[i] if i < len(event.media_types) else ""
                is_audio = (
                    mtype.startswith("audio/")
                    or event.message_type in (MessageType.VOICE, MessageType.AUDIO)
                )
                if is_audio:
                    audio_paths.append(path)
            if audio_paths:
                message_text = await self._enrich_message_with_transcription(
                    message_text, audio_paths
                )

        # -----------------------------------------------------------------
        # Enrich document messages with context notes for the agent
        # -----------------------------------------------------------------
        if event.media_urls and event.message_type == MessageType.DOCUMENT:
            for i, path in enumerate(event.media_urls):
                mtype = event.media_types[i] if i < len(event.media_types) else ""
                if not (mtype.startswith("application/") or mtype.startswith("text/")):
                    continue
                # Extract display filename by stripping the doc_{uuid12}_ prefix
                import os as _os
                basename = _os.path.basename(path)
                # Format: doc_<12hex>_<original_filename>
                parts = basename.split("_", 2)
                display_name = parts[2] if len(parts) >= 3 else basename
                # Sanitize to prevent prompt injection via filenames
                import re as _re
                display_name = _re.sub(r'[^\w.\- ]', '_', display_name)

                if mtype.startswith("text/"):
                    context_note = (
                        f"[The user sent a text document: '{display_name}'. "
                        f"Its content has been included below. "
                        f"The file is also saved at: {path}]"
                    )
                else:
                    context_note = (
                        f"[The user sent a document: '{display_name}'. "
                        f"The file is saved at: {path}. "
                        f"Ask the user what they'd like you to do with it.]"
                    )
                message_text = f"{context_note}\n\n{message_text}"

        try:
            # Emit agent:start hook
            hook_ctx = {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "session_id": session_entry.session_id,
                "message": message_text[:500],
            }
            await self.hooks.emit("agent:start", hook_ctx)
            
            # Run the agent
            agent_result = await self._run_agent(
                message=message_text,
                context_prompt=context_prompt,
                history=history,
                source=source,
                session_id=session_entry.session_id,
                session_key=session_key
            )
            
            response = agent_result.get("final_response", "")
            agent_messages = agent_result.get("messages", [])

            # If the agent's session_id changed during compression, update
            # session_entry so transcript writes below go to the right session.
            if agent_result.get("session_id") and agent_result["session_id"] != session_entry.session_id:
                session_entry.session_id = agent_result["session_id"]

            # Prepend reasoning/thinking if display is enabled
            if getattr(self, "_show_reasoning", False) and response:
                last_reasoning = agent_result.get("last_reasoning")
                if last_reasoning:
                    # Collapse long reasoning to keep messages readable
                    lines = last_reasoning.strip().splitlines()
                    if len(lines) > 15:
                        display_reasoning = "\n".join(lines[:15])
                        display_reasoning += f"\n_... ({len(lines) - 15} more lines)_"
                    else:
                        display_reasoning = last_reasoning.strip()
                    response = f"💭 **Reasoning:**\n```\n{display_reasoning}\n```\n\n{response}"

            # Emit agent:end hook
            await self.hooks.emit("agent:end", {
                **hook_ctx,
                "response": (response or "")[:500],
            })
            
            # Check for pending process watchers (check_interval on background processes)
            try:
                from tools.process_registry import process_registry
                while process_registry.pending_watchers:
                    watcher = process_registry.pending_watchers.pop(0)
                    asyncio.create_task(self._run_process_watcher(watcher))
            except Exception as e:
                logger.error("Process watcher setup error: %s", e)

            # Check if the agent encountered a dangerous command needing approval
            try:
                from tools.approval import pop_pending
                pending = pop_pending(session_key)
                if pending:
                    self._pending_approvals[session_key] = pending
            except Exception as e:
                logger.debug("Failed to check pending approvals: %s", e)
            
            # Save the full conversation to the transcript, including tool calls.
            # This preserves the complete agent loop (tool_calls, tool results,
            # intermediate reasoning) so sessions can be resumed with full context
            # and transcripts are useful for debugging and training data.
            ts = datetime.now().isoformat()
            
            # If this is a fresh session (no history), write the full tool
            # definitions as the first entry so the transcript is self-describing
            # -- the same list of dicts sent as tools=[...] in the API request.
            if not history:
                tool_defs = agent_result.get("tools", [])
                self.session_store.append_to_transcript(
                    session_entry.session_id,
                    {
                        "role": "session_meta",
                        "tools": tool_defs or [],
                        "model": os.getenv("HERMES_MODEL", ""),
                        "platform": source.platform.value if source.platform else "",
                        "timestamp": ts,
                    }
                )
            
            # Find only the NEW messages from this turn (skip history we loaded).
            # Use the filtered history length (history_offset) that was actually
            # passed to the agent, not len(history) which includes session_meta
            # entries that were stripped before the agent saw them.
            history_len = agent_result.get("history_offset", len(history))
            new_messages = agent_messages[history_len:] if len(agent_messages) > history_len else []
            
            # If no new messages found (edge case), fall back to simple user/assistant
            if not new_messages:
                self.session_store.append_to_transcript(
                    session_entry.session_id,
                    {"role": "user", "content": message_text, "timestamp": ts}
                )
                if response:
                    self.session_store.append_to_transcript(
                        session_entry.session_id,
                        {"role": "assistant", "content": response, "timestamp": ts}
                    )
            else:
                # The agent already persisted these messages to SQLite via
                # _flush_messages_to_session_db(), so skip the DB write here
                # to prevent the duplicate-write bug (#860).  We still write
                # to JSONL for backward compatibility and as a backup.
                agent_persisted = self._session_db is not None
                for msg in new_messages:
                    # Skip system messages (they're rebuilt each run)
                    if msg.get("role") == "system":
                        continue
                    # Add timestamp to each message for debugging
                    entry = {**msg, "timestamp": ts}
                    self.session_store.append_to_transcript(
                        session_entry.session_id, entry,
                        skip_db=agent_persisted,
                    )
            
            # Update session with actual prompt token count from the agent
            self.session_store.update_session(
                session_entry.session_key,
                last_prompt_tokens=agent_result.get("last_prompt_tokens", 0),
            )
            
            return response
            
        except Exception as e:
            logger.exception("Agent error in session %s", session_key)
            return (
                "Sorry, I encountered an unexpected error. "
                "The details have been logged for debugging. "
                "Try again or use /reset to start a fresh session."
            )
        finally:
            # Clear session env
            self._clear_session_env()
    
    async def _handle_reset_command(self, event: MessageEvent) -> str:
        """Handle /new or /reset command."""
        source = event.source
        
        # Get existing session key
        session_key = self.session_store._generate_session_key(source)
        
        # Flush memories in the background (fire-and-forget) so the user
        # gets the "Session reset!" response immediately.
        try:
            old_entry = self.session_store._entries.get(session_key)
            if old_entry:
                asyncio.create_task(self._async_flush_memories(old_entry.session_id))
        except Exception as e:
            logger.debug("Gateway memory flush on reset failed: %s", e)

        self._shutdown_gateway_honcho(session_key)
        
        # Reset the session
        new_entry = self.session_store.reset_session(session_key)
        
        # Emit session:reset hook
        await self.hooks.emit("session:reset", {
            "platform": source.platform.value if source.platform else "",
            "user_id": source.user_id,
            "session_key": session_key,
        })
        
        if new_entry:
            return "✨ Session reset! I've started fresh with no memory of our previous conversation."
        else:
            # No existing session, just create one
            self.session_store.get_or_create_session(source, force_new=True)
            return "✨ New session started!"
    
    async def _handle_status_command(self, event: MessageEvent) -> str:
        """Handle /status command."""
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        
        connected_platforms = [p.value for p in self.adapters.keys()]
        
        # Check if there's an active agent
        session_key = session_entry.session_key
        is_running = session_key in self._running_agents
        
        lines = [
            "📊 **Hermes Gateway Status**",
            "",
            f"**Session ID:** `{session_entry.session_id[:12]}...`",
            f"**Created:** {session_entry.created_at.strftime('%Y-%m-%d %H:%M')}",
            f"**Last Activity:** {session_entry.updated_at.strftime('%Y-%m-%d %H:%M')}",
            f"**Tokens:** {session_entry.total_tokens:,}",
            f"**Agent Running:** {'Yes ⚡' if is_running else 'No'}",
            "",
            f"**Connected Platforms:** {', '.join(connected_platforms)}",
        ]
        
        return "\n".join(lines)
    
    async def _handle_stop_command(self, event: MessageEvent) -> str:
        """Handle /stop command - interrupt a running agent."""
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        session_key = session_entry.session_key
        
        if session_key in self._running_agents:
            agent = self._running_agents[session_key]
            agent.interrupt()
            return "⚡ Stopping the current task... The agent will finish its current step and respond."
        else:
            return "No active task to stop."
    
    async def _handle_help_command(self, event: MessageEvent) -> str:
        """Handle /help command - list available commands."""
        lines = [
            "📖 **Hermes Commands**\n",
            "`/new` — Start a new conversation",
            "`/reset` — Reset conversation history",
            "`/status` — Show session info",
            "`/stop` — Interrupt the running agent",
            "`/model [provider:model]` — Show/change model (or switch provider)",
            "`/provider` — Show available providers and auth status",
            "`/personality [name]` — Set a personality",
            "`/retry` — Retry your last message",
            "`/undo` — Remove the last exchange",
            "`/sethome` — Set this chat as the home channel",
            "`/compress` — Compress conversation context",
            "`/title [name]` — Set or show the session title",
            "`/resume [name]` — Resume a previously-named session",
            "`/usage` — Show token usage for this session",
            "`/insights [days]` — Show usage insights and analytics",
            "`/reasoning [level|show|hide]` — Set reasoning effort or toggle display",
            "`/rollback [number]` — List or restore filesystem checkpoints",
            "`/background <prompt>` — Run a prompt in a separate background session",
            "`/reload-mcp` — Reload MCP servers from config",
            "`/update` — Update Hermes Agent to the latest version",
            "`/help` — Show this message",
        ]
        try:
            from agent.skill_commands import get_skill_commands
            skill_cmds = get_skill_commands()
            if skill_cmds:
                lines.append(f"\n⚡ **Skill Commands** ({len(skill_cmds)} installed):")
                for cmd in sorted(skill_cmds):
                    lines.append(f"`{cmd}` — {skill_cmds[cmd]['description']}")
        except Exception:
            pass
        return "\n".join(lines)
    
    async def _handle_model_command(self, event: MessageEvent) -> str:
        """Handle /model command - show or change the current model."""
        import yaml
        from hermes_cli.models import (
            parse_model_input,
            validate_requested_model,
            curated_models_for_provider,
            normalize_provider,
            _PROVIDER_LABELS,
        )

        args = event.get_command_args().strip()
        config_path = _hermes_home / 'config.yaml'

        # Resolve current model and provider from config
        current = os.getenv("HERMES_MODEL") or "anthropic/claude-opus-4.6"
        current_provider = "openrouter"
        try:
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                model_cfg = cfg.get("model", {})
                if isinstance(model_cfg, str):
                    current = model_cfg
                elif isinstance(model_cfg, dict):
                    current = model_cfg.get("default", current)
                    current_provider = model_cfg.get("provider", current_provider)
        except Exception:
            pass

        # Resolve "auto" to the actual provider using credential detection
        current_provider = normalize_provider(current_provider)
        if current_provider == "auto":
            try:
                from hermes_cli.auth import resolve_provider as _resolve_provider
                current_provider = _resolve_provider(current_provider)
            except Exception:
                current_provider = "openrouter"

        # Detect custom endpoint: provider resolved to openrouter but a custom
        # base URL is configured — the user set up a custom endpoint.
        if current_provider == "openrouter" and os.getenv("OPENAI_BASE_URL", "").strip():
            current_provider = "custom"

        if not args:
            provider_label = _PROVIDER_LABELS.get(current_provider, current_provider)
            lines = [
                f"🤖 **Current model:** `{current}`",
                f"**Provider:** {provider_label}",
                "",
            ]
            curated = curated_models_for_provider(current_provider)
            if curated:
                lines.append(f"**Available models ({provider_label}):**")
                for mid, desc in curated:
                    marker = " ←" if mid == current else ""
                    label = f"  _{desc}_" if desc else ""
                    lines.append(f"• `{mid}`{label}{marker}")
                lines.append("")
            lines.append("To change: `/model model-name`")
            lines.append("Switch provider: `/model provider:model-name`")
            return "\n".join(lines)

        # Parse provider:model syntax
        target_provider, new_model = parse_model_input(args, current_provider)
        provider_changed = target_provider != current_provider

        # Resolve credentials for the target provider (for API probe)
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        base_url = "https://openrouter.ai/api/v1"
        if provider_changed:
            try:
                from hermes_cli.runtime_provider import resolve_runtime_provider
                runtime = resolve_runtime_provider(requested=target_provider)
                api_key = runtime.get("api_key", "")
                base_url = runtime.get("base_url", "")
            except Exception as e:
                provider_label = _PROVIDER_LABELS.get(target_provider, target_provider)
                return f"⚠️ Could not resolve credentials for provider '{provider_label}': {e}"
        else:
            # Use current provider's base_url from config or registry
            try:
                from hermes_cli.runtime_provider import resolve_runtime_provider
                runtime = resolve_runtime_provider(requested=current_provider)
                api_key = runtime.get("api_key", "")
                base_url = runtime.get("base_url", "")
            except Exception:
                pass

        # Validate the model against the live API
        try:
            validation = validate_requested_model(
                new_model,
                target_provider,
                api_key=api_key,
                base_url=base_url,
            )
        except Exception:
            validation = {"accepted": True, "persist": True, "recognized": False, "message": None}

        if not validation.get("accepted"):
            msg = validation.get("message", "Invalid model")
            tip = "\n\nUse `/model` to see available models, `/provider` to see providers" if "Did you mean" not in msg else ""
            return f"⚠️ {msg}{tip}"

        # Persist to config only if validation approves
        if validation.get("persist"):
            try:
                user_config = {}
                if config_path.exists():
                    with open(config_path, encoding="utf-8") as f:
                        user_config = yaml.safe_load(f) or {}
                if "model" not in user_config or not isinstance(user_config["model"], dict):
                    user_config["model"] = {}
                user_config["model"]["default"] = new_model
                if provider_changed:
                    user_config["model"]["provider"] = target_provider
                with open(config_path, 'w', encoding="utf-8") as f:
                    yaml.dump(user_config, f, default_flow_style=False, sort_keys=False)
            except Exception as e:
                return f"⚠️ Failed to save model change: {e}"

        # Set env vars so the next agent run picks up the change
        os.environ["HERMES_MODEL"] = new_model
        if provider_changed:
            os.environ["HERMES_INFERENCE_PROVIDER"] = target_provider

        provider_label = _PROVIDER_LABELS.get(target_provider, target_provider)
        provider_note = f"\n**Provider:** {provider_label}" if provider_changed else ""

        warning = ""
        if validation.get("message"):
            warning = f"\n⚠️ {validation['message']}"

        if validation.get("persist"):
            persist_note = "saved to config"
        else:
            persist_note = "this session only — will revert on restart"
        return f"🤖 Model changed to `{new_model}` ({persist_note}){provider_note}{warning}\n_(takes effect on next message)_"

    async def _handle_provider_command(self, event: MessageEvent) -> str:
        """Handle /provider command - show available providers."""
        import yaml
        from hermes_cli.models import (
            list_available_providers,
            normalize_provider,
            _PROVIDER_LABELS,
        )

        # Resolve current provider from config
        current_provider = "openrouter"
        config_path = _hermes_home / 'config.yaml'
        try:
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                model_cfg = cfg.get("model", {})
                if isinstance(model_cfg, dict):
                    current_provider = model_cfg.get("provider", current_provider)
        except Exception:
            pass

        current_provider = normalize_provider(current_provider)
        if current_provider == "auto":
            try:
                from hermes_cli.auth import resolve_provider as _resolve_provider
                current_provider = _resolve_provider(current_provider)
            except Exception:
                current_provider = "openrouter"

        # Detect custom endpoint
        if current_provider == "openrouter" and os.getenv("OPENAI_BASE_URL", "").strip():
            current_provider = "custom"

        current_label = _PROVIDER_LABELS.get(current_provider, current_provider)

        lines = [
            f"🔌 **Current provider:** {current_label} (`{current_provider}`)",
            "",
            "**Available providers:**",
        ]

        providers = list_available_providers()
        for p in providers:
            marker = " ← active" if p["id"] == current_provider else ""
            auth = "✅" if p["authenticated"] else "❌"
            aliases = f"  _(also: {', '.join(p['aliases'])})_" if p["aliases"] else ""
            lines.append(f"{auth} `{p['id']}` — {p['label']}{aliases}{marker}")

        lines.append("")
        lines.append("Switch: `/model provider:model-name`")
        lines.append("Setup: `hermes setup`")
        return "\n".join(lines)
    
    async def _handle_personality_command(self, event: MessageEvent) -> str:
        """Handle /personality command - list or set a personality."""
        import yaml

        args = event.get_command_args().strip().lower()
        config_path = _hermes_home / 'config.yaml'

        try:
            if config_path.exists():
                with open(config_path, 'r', encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                personalities = config.get("agent", {}).get("personalities", {})
            else:
                config = {}
                personalities = {}
        except Exception:
            config = {}
            personalities = {}

        if not personalities:
            return "No personalities configured in `~/.hermes/config.yaml`"

        if not args:
            lines = ["🎭 **Available Personalities**\n"]
            lines.append("• `none` — (no personality overlay)")
            for name, prompt in personalities.items():
                if isinstance(prompt, dict):
                    preview = prompt.get("description") or prompt.get("system_prompt", "")[:50]
                else:
                    preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
                lines.append(f"• `{name}` — {preview}")
            lines.append(f"\nUsage: `/personality <name>`")
            return "\n".join(lines)

        def _resolve_prompt(value):
            if isinstance(value, dict):
                parts = [value.get("system_prompt", "")]
                if value.get("tone"):
                    parts.append(f'Tone: {value["tone"]}')
                if value.get("style"):
                    parts.append(f'Style: {value["style"]}')
                return "\n".join(p for p in parts if p)
            return str(value)

        if args in ("none", "default", "neutral"):
            try:
                if "agent" not in config or not isinstance(config.get("agent"), dict):
                    config["agent"] = {}
                config["agent"]["system_prompt"] = ""
                with open(config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            except Exception as e:
                return f"⚠️ Failed to save personality change: {e}"
            self._ephemeral_system_prompt = ""
            return "🎭 Personality cleared — using base agent behavior.\n_(takes effect on next message)_"
        elif args in personalities:
            new_prompt = _resolve_prompt(personalities[args])

            # Write to config.yaml, same pattern as CLI save_config_value.
            try:
                if "agent" not in config or not isinstance(config.get("agent"), dict):
                    config["agent"] = {}
                config["agent"]["system_prompt"] = new_prompt
                with open(config_path, 'w', encoding="utf-8") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            except Exception as e:
                return f"⚠️ Failed to save personality change: {e}"

            # Update in-memory so it takes effect on the very next message.
            self._ephemeral_system_prompt = new_prompt

            return f"🎭 Personality set to **{args}**\n_(takes effect on next message)_"

        available = "`none`, " + ", ".join(f"`{n}`" for n in personalities.keys())
        return f"Unknown personality: `{args}`\n\nAvailable: {available}"
    
    async def _handle_retry_command(self, event: MessageEvent) -> str:
        """Handle /retry command - re-send the last user message."""
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        history = self.session_store.load_transcript(session_entry.session_id)
        
        # Find the last user message
        last_user_msg = None
        last_user_idx = None
        for i in range(len(history) - 1, -1, -1):
            if history[i].get("role") == "user":
                last_user_msg = history[i].get("content", "")
                last_user_idx = i
                break
        
        if not last_user_msg:
            return "No previous message to retry."
        
        # Truncate history to before the last user message and persist
        truncated = history[:last_user_idx]
        self.session_store.rewrite_transcript(session_entry.session_id, truncated)
        # Reset stored token count — transcript was truncated
        session_entry.last_prompt_tokens = 0
        
        # Re-send by creating a fake text event with the old message
        retry_event = MessageEvent(
            text=last_user_msg,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=event.raw_message,
        )
        
        # Let the normal message handler process it
        return await self._handle_message(retry_event)
    
    async def _handle_undo_command(self, event: MessageEvent) -> str:
        """Handle /undo command - remove the last user/assistant exchange."""
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
