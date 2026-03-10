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

            tmp_agent = AIAgent(
                **runtime_kwargs,
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
        }
        platform_allow_all_map = {
            Platform.TELEGRAM: "TELEGRAM_ALLOW_ALL_USERS",
            Platform.DISCORD: "DISCORD_ALLOW_ALL_USERS",
            Platform.WHATSAPP: "WHATSAPP_ALLOW_ALL_USERS",
            Platform.SLACK: "SLACK_ALLOW_ALL_USERS",
            Platform.SIGNAL: "SIGNAL_ALLOW_ALL_USERS",
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
                          "update", "title", "resume", "provider"}
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
        
        # Skill slash commands: /skill-name loads the skill and sends to agent
        if command:
            try:
                from agent.skill_commands import get_skill_commands, build_skill_invocation_message
                skill_cmds = get_skill_commands()
                cmd_key = f"/{command}"
                if cmd_key in skill_cmds:
                    user_instruction = event.get_command_args().strip()
                    msg = build_skill_invocation_message(cmd_key, user_instruction)
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
                pattern_key = approval.get("pattern_key", "")
                logger.info("User approved dangerous command: %s...", cmd[:60])
                from tools.terminal_tool import terminal_tool
                from tools.approval import approve_session
                approve_session(session_key_preview, pattern_key)
                result = terminal_tool(command=cmd, force=True)
                return f"✅ Command approved and executed.\n\n```\n{result[:3500]}\n```"
            elif user_text in ("no", "n", "deny", "cancel", "nope"):
                self._pending_approvals.pop(session_key_preview)
                return "❌ Command denied."
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
        # Thresholds are derived from the SAME compression config the
        # agent uses (compression.threshold × model context length) so
        # CLI and messaging platforms behave identically.
        # -----------------------------------------------------------------
        if history and len(history) >= 4:
            from agent.model_metadata import (
                estimate_messages_tokens_rough,
                get_model_context_length,
            )

            # Read model + compression config from config.yaml — same
            # source of truth the agent itself uses.
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

                    # Read compression settings
                    _comp_cfg = _hyg_data.get("compression", {})
                    if isinstance(_comp_cfg, dict):
                        _hyg_threshold_pct = float(
                            _comp_cfg.get("threshold", _hyg_threshold_pct)
                        )
                        _hyg_compression_enabled = str(
                            _comp_cfg.get("enabled", True)
                        ).lower() in ("true", "1", "yes")
            except Exception:
                pass

            # Also check env overrides (same as run_agent.py)
            _hyg_threshold_pct = float(
                os.getenv("CONTEXT_COMPRESSION_THRESHOLD", str(_hyg_threshold_pct))
            )
            if os.getenv("CONTEXT_COMPRESSION_ENABLED", "").lower() in ("false", "0", "no"):
                _hyg_compression_enabled = False

            if _hyg_compression_enabled:
                _hyg_context_length = get_model_context_length(_hyg_model)
                _compress_token_threshold = int(
                    _hyg_context_length * _hyg_threshold_pct
                )
                # Warn if still huge after compression (95% of context)
                _warn_token_threshold = int(_hyg_context_length * 0.95)

                _msg_count = len(history)
                _approx_tokens = estimate_messages_tokens_rough(history)

                _needs_compress = _approx_tokens >= _compress_token_threshold

                if _needs_compress:
                    logger.info(
                        "Session hygiene: %s messages, ~%s tokens — auto-compressing "
                        "(threshold: %s%% of %s = %s tokens)",
                        _msg_count, f"{_approx_tokens:,}",
                        int(_hyg_threshold_pct * 100),
                        f"{_hyg_context_length:,}",
                        f"{_compress_token_threshold:,}",
                    )

                    _hyg_adapter = self.adapters.get(source.platform)
                    if _hyg_adapter:
                        try:
                            await _hyg_adapter.send(
                                source.chat_id,
                                f"🗜️ Session is large ({_msg_count} messages, "
                                f"~{_approx_tokens:,} tokens). Auto-compressing..."
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
                                            f"~{_new_tokens:,} tokens"
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
                                                "fresh if you experience issues."
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
                            if _hyg_adapter:
                                try:
                                    await _hyg_adapter.send(
                                        source.chat_id,
                                        f"⚠️ Session is very large "
                                        f"({_msg_count} messages, "
                                        f"~{_approx_tokens:,} tokens) and "
                                        "auto-compression failed. Consider "
                                        "using /compress or /reset to avoid "
                                        "issues."
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
                for msg in new_messages:
                    # Skip system messages (they're rebuilt each run)
                    if msg.get("role") == "system":
                        continue
                    # Add timestamp to each message for debugging
                    entry = {**msg, "timestamp": ts}
                    self.session_store.append_to_transcript(
                        session_entry.session_id, entry
                    )
            
            # Update session
            self.session_store.update_session(session_entry.session_key)
            
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
        current = os.getenv("HERMES_MODEL") or os.getenv("LLM_MODEL") or "anthropic/claude-opus-4.6"
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
            for name, prompt in personalities.items():
                preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
                lines.append(f"• `{name}` — {preview}")
            lines.append(f"\nUsage: `/personality <name>`")
            return "\n".join(lines)

        if args in personalities:
            new_prompt = personalities[args]

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

        available = ", ".join(f"`{n}`" for n in personalities.keys())
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
        history = self.session_store.load_transcript(session_entry.session_id)
        
        # Find the last user message and remove everything from it onward
        last_user_idx = None
        for i in range(len(history) - 1, -1, -1):
            if history[i].get("role") == "user":
                last_user_idx = i
                break
        
        if last_user_idx is None:
            return "Nothing to undo."
        
        removed_msg = history[last_user_idx].get("content", "")
        removed_count = len(history) - last_user_idx
        self.session_store.rewrite_transcript(session_entry.session_id, history[:last_user_idx])
        
        preview = removed_msg[:40] + "..." if len(removed_msg) > 40 else removed_msg
        return f"↩️ Undid {removed_count} message(s).\nRemoved: \"{preview}\""
    
    async def _handle_set_home_command(self, event: MessageEvent) -> str:
        """Handle /sethome command -- set the current chat as the platform's home channel."""
        source = event.source
        platform_name = source.platform.value if source.platform else "unknown"
        chat_id = source.chat_id
        chat_name = source.chat_name or chat_id
        
        env_key = f"{platform_name.upper()}_HOME_CHANNEL"
        
        # Save to config.yaml
        try:
            import yaml
            config_path = _hermes_home / 'config.yaml'
            user_config = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    user_config = yaml.safe_load(f) or {}
            user_config[env_key] = chat_id
            with open(config_path, 'w', encoding="utf-8") as f:
                yaml.dump(user_config, f, default_flow_style=False)
            # Also set in the current environment so it takes effect immediately
            os.environ[env_key] = str(chat_id)
        except Exception as e:
            return f"Failed to save home channel: {e}"
        
        return (
            f"✅ Home channel set to **{chat_name}** (ID: {chat_id}).\n"
            f"Cron jobs and cross-platform messages will be delivered here."
        )
    
    async def _handle_compress_command(self, event: MessageEvent) -> str:
        """Handle /compress command -- manually compress conversation context."""
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        history = self.session_store.load_transcript(session_entry.session_id)

        if not history or len(history) < 4:
            return "Not enough conversation to compress (need at least 4 messages)."

        try:
            from run_agent import AIAgent
            from agent.model_metadata import estimate_messages_tokens_rough

            runtime_kwargs = _resolve_runtime_agent_kwargs()
            if not runtime_kwargs.get("api_key"):
                return "No provider configured -- cannot compress."

            msgs = [
                {"role": m.get("role"), "content": m.get("content")}
                for m in history
                if m.get("role") in ("user", "assistant") and m.get("content")
            ]
            original_count = len(msgs)
            approx_tokens = estimate_messages_tokens_rough(msgs)

            tmp_agent = AIAgent(
                **runtime_kwargs,
                max_iterations=4,
                quiet_mode=True,
                enabled_toolsets=["memory"],
                session_id=session_entry.session_id,
            )

            loop = asyncio.get_event_loop()
            compressed, _ = await loop.run_in_executor(
                None,
                lambda: tmp_agent._compress_context(msgs, "", approx_tokens=approx_tokens),
            )

            self.session_store.rewrite_transcript(session_entry.session_id, compressed)
            new_count = len(compressed)
            new_tokens = estimate_messages_tokens_rough(compressed)

            return (
                f"🗜️ Compressed: {original_count} → {new_count} messages\n"
                f"~{approx_tokens:,} → ~{new_tokens:,} tokens"
            )
        except Exception as e:
            logger.warning("Manual compress failed: %s", e)
            return f"Compression failed: {e}"

    async def _handle_title_command(self, event: MessageEvent) -> str:
        """Handle /title command — set or show the current session's title."""
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        session_id = session_entry.session_id

        if not self._session_db:
            return "Session database not available."

        title_arg = event.get_command_args().strip()
        if title_arg:
            # Sanitize the title before setting
            try:
                sanitized = self._session_db.sanitize_title(title_arg)
            except ValueError as e:
                return f"⚠️ {e}"
            if not sanitized:
                return "⚠️ Title is empty after cleanup. Please use printable characters."
            # Set the title
            try:
                if self._session_db.set_session_title(session_id, sanitized):
                    return f"✏️ Session title set: **{sanitized}**"
                else:
                    return "Session not found in database."
            except ValueError as e:
                return f"⚠️ {e}"
        else:
            # Show the current title
            title = self._session_db.get_session_title(session_id)
            if title:
                return f"📌 Session title: **{title}**"
            else:
                return "No title set. Usage: `/title My Session Name`"

    async def _handle_resume_command(self, event: MessageEvent) -> str:
        """Handle /resume command — switch to a previously-named session."""
        if not self._session_db:
            return "Session database not available."

        source = event.source
        session_key = build_session_key(source)
        name = event.get_command_args().strip()

        if not name:
            # List recent titled sessions for this user/platform
            try:
                user_source = source.platform.value if source.platform else None
                sessions = self._session_db.list_sessions_rich(
                    source=user_source, limit=10
                )
                titled = [s for s in sessions if s.get("title")]
                if not titled:
                    return (
                        "No named sessions found.\n"
                        "Use `/title My Session` to name your current session, "
                        "then `/resume My Session` to return to it later."
                    )
                lines = ["📋 **Named Sessions**\n"]
                for s in titled[:10]:
                    title = s["title"]
                    preview = s.get("preview", "")[:40]
                    preview_part = f" — _{preview}_" if preview else ""
                    lines.append(f"• **{title}**{preview_part}")
                lines.append("\nUsage: `/resume <session name>`")
                return "\n".join(lines)
            except Exception as e:
                logger.debug("Failed to list titled sessions: %s", e)
                return f"Could not list sessions: {e}"

        # Resolve the name to a session ID
        target_id = self._session_db.resolve_session_by_title(name)
        if not target_id:
            return (
                f"No session found matching '**{name}**'.\n"
                "Use `/resume` with no arguments to see available sessions."
            )

        # Check if already on that session
        current_entry = self.session_store.get_or_create_session(source)
        if current_entry.session_id == target_id:
            return f"📌 Already on session **{name}**."

        # Flush memories for current session before switching
        try:
            asyncio.create_task(self._async_flush_memories(current_entry.session_id))
        except Exception as e:
            logger.debug("Memory flush on resume failed: %s", e)

        # Clear any running agent for this session key
        if session_key in self._running_agents:
            del self._running_agents[session_key]

        # Switch the session entry to point at the old session
        new_entry = self.session_store.switch_session(session_key, target_id)
        if not new_entry:
            return "Failed to switch session."

        # Get the title for confirmation
        title = self._session_db.get_session_title(target_id) or name

        # Count messages for context
        history = self.session_store.load_transcript(target_id)
        msg_count = len([m for m in history if m.get("role") == "user"]) if history else 0
        msg_part = f" ({msg_count} message{'s' if msg_count != 1 else ''})" if msg_count else ""

        return f"↻ Resumed session **{title}**{msg_part}. Conversation restored."

    async def _handle_usage_command(self, event: MessageEvent) -> str:
        """Handle /usage command -- show token usage for the session's last agent run."""
        source = event.source
        session_key = build_session_key(source)

        agent = self._running_agents.get(session_key)
        if agent and hasattr(agent, "session_total_tokens") and agent.session_api_calls > 0:
            lines = [
                "📊 **Session Token Usage**",
                f"Prompt (input): {agent.session_prompt_tokens:,}",
                f"Completion (output): {agent.session_completion_tokens:,}",
                f"Total: {agent.session_total_tokens:,}",
                f"API calls: {agent.session_api_calls}",
            ]
            ctx = agent.context_compressor
            if ctx.last_prompt_tokens:
                pct = ctx.last_prompt_tokens / ctx.context_length * 100 if ctx.context_length else 0
                lines.append(f"Context: {ctx.last_prompt_tokens:,} / {ctx.context_length:,} ({pct:.0f}%)")
            if ctx.compression_count:
                lines.append(f"Compressions: {ctx.compression_count}")
            return "\n".join(lines)

        # No running agent -- check session history for a rough count
        session_entry = self.session_store.get_or_create_session(source)
        history = self.session_store.load_transcript(session_entry.session_id)
        if history:
            from agent.model_metadata import estimate_messages_tokens_rough
            msgs = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")]
            approx = estimate_messages_tokens_rough(msgs)
            return (
                f"📊 **Session Info**\n"
                f"Messages: {len(msgs)}\n"
                f"Estimated context: ~{approx:,} tokens\n"
                f"_(Detailed usage available during active conversations)_"
            )
        return "No usage data available for this session."

    async def _handle_insights_command(self, event: MessageEvent) -> str:
        """Handle /insights command -- show usage insights and analytics."""
        import asyncio as _asyncio

        args = event.get_command_args().strip()
        days = 30
        source = None

        # Parse simple args: /insights 7  or  /insights --days 7
        if args:
            parts = args.split()
            i = 0
            while i < len(parts):
                if parts[i] == "--days" and i + 1 < len(parts):
                    try:
                        days = int(parts[i + 1])
                    except ValueError:
                        return f"Invalid --days value: {parts[i + 1]}"
                    i += 2
                elif parts[i] == "--source" and i + 1 < len(parts):
                    source = parts[i + 1]
                    i += 2
                elif parts[i].isdigit():
                    days = int(parts[i])
                    i += 1
                else:
                    i += 1

        try:
            from hermes_state import SessionDB
            from agent.insights import InsightsEngine

            loop = _asyncio.get_event_loop()

            def _run_insights():
                db = SessionDB()
                engine = InsightsEngine(db)
                report = engine.generate(days=days, source=source)
                result = engine.format_gateway(report)
                db.close()
                return result

            return await loop.run_in_executor(None, _run_insights)
        except Exception as e:
            logger.error("Insights command error: %s", e, exc_info=True)
            return f"Error generating insights: {e}"

    async def _handle_reload_mcp_command(self, event: MessageEvent) -> str:
        """Handle /reload-mcp command -- disconnect and reconnect all MCP servers."""
        loop = asyncio.get_event_loop()
        try:
            from tools.mcp_tool import shutdown_mcp_servers, discover_mcp_tools, _load_mcp_config, _servers, _lock

            # Capture old server names before shutdown
            with _lock:
                old_servers = set(_servers.keys())

            # Read new config before shutting down, so we know what will be added/removed
            new_config = _load_mcp_config()
            new_server_names = set(new_config.keys())

            # Shutdown existing connections
            await loop.run_in_executor(None, shutdown_mcp_servers)

            # Reconnect by discovering tools (reads config.yaml fresh)
            new_tools = await loop.run_in_executor(None, discover_mcp_tools)

            # Compute what changed
            with _lock:
                connected_servers = set(_servers.keys())

            added = connected_servers - old_servers
            removed = old_servers - connected_servers
            reconnected = connected_servers & old_servers

            lines = ["🔄 **MCP Servers Reloaded**\n"]
            if reconnected:
                lines.append(f"♻️ Reconnected: {', '.join(sorted(reconnected))}")
            if added:
                lines.append(f"➕ Added: {', '.join(sorted(added))}")
            if removed:
                lines.append(f"➖ Removed: {', '.join(sorted(removed))}")
            if not connected_servers:
                lines.append("No MCP servers connected.")
            else:
                lines.append(f"\n🔧 {len(new_tools)} tool(s) available from {len(connected_servers)} server(s)")

            # Inject a message at the END of the session history so the
            # model knows tools changed on its next turn.  Appended after
            # all existing messages to preserve prompt-cache for the prefix.
            change_parts = []
            if added:
                change_parts.append(f"Added servers: {', '.join(sorted(added))}")
            if removed:
                change_parts.append(f"Removed servers: {', '.join(sorted(removed))}")
            if reconnected:
                change_parts.append(f"Reconnected servers: {', '.join(sorted(reconnected))}")
            tool_summary = f"{len(new_tools)} MCP tool(s) now available" if new_tools else "No MCP tools available"
            change_detail = ". ".join(change_parts) + ". " if change_parts else ""
            reload_msg = {
                "role": "user",
                "content": f"[SYSTEM: MCP servers have been reloaded. {change_detail}{tool_summary}. The tool list for this conversation has been updated accordingly.]",
            }
            try:
                session_entry = self.session_store.get_or_create_session(event.source)
                self.session_store.append_to_transcript(
                    session_entry.session_id, reload_msg
                )
            except Exception:
                pass  # Best-effort; don't fail the reload over a transcript write

            return "\n".join(lines)

        except Exception as e:
            logger.warning("MCP reload failed: %s", e)
            return f"❌ MCP reload failed: {e}"

    async def _handle_update_command(self, event: MessageEvent) -> str:
        """Handle /update command — update Hermes Agent to the latest version.

        Spawns ``hermes update`` in a separate systemd scope so it survives the
        gateway restart that ``hermes update`` triggers at the end.  A marker
        file is written so the *new* gateway process can notify the user of the
        result on startup.
        """
        import json
        import shutil
        import subprocess
        from datetime import datetime

        project_root = Path(__file__).parent.parent.resolve()
        git_dir = project_root / '.git'

        if not git_dir.exists():
            return "✗ Not a git repository — cannot update."

        hermes_bin = shutil.which("hermes")
        if not hermes_bin:
            return "✗ `hermes` command not found on PATH."

        # Write marker so the restarted gateway can notify this chat
        pending_path = _hermes_home / ".update_pending.json"
        output_path = _hermes_home / ".update_output.txt"
        pending = {
            "platform": event.source.platform.value,
            "chat_id": event.source.chat_id,
            "user_id": event.source.user_id,
            "timestamp": datetime.now().isoformat(),
        }
        pending_path.write_text(json.dumps(pending))

        # Spawn `hermes update` in a separate cgroup so it survives gateway
        # restart.  systemd-run --user --scope creates a transient scope unit.
        update_cmd = f"{hermes_bin} update > {output_path} 2>&1"
        try:
            systemd_run = shutil.which("systemd-run")
            if systemd_run:
                subprocess.Popen(
                    [systemd_run, "--user", "--scope",
                     "--unit=hermes-update", "--",
                     "bash", "-c", update_cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            else:
                # Fallback: best-effort detach with start_new_session
                subprocess.Popen(
                    ["bash", "-c", f"nohup {update_cmd} &"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
        except Exception as e:
            pending_path.unlink(missing_ok=True)
            return f"✗ Failed to start update: {e}"

        return "⚕ Starting Hermes update… I'll notify you when it's done."

    async def _send_update_notification(self) -> None:
        """If the gateway is starting after a ``/update``, notify the user."""
        import json
        import re as _re

        pending_path = _hermes_home / ".update_pending.json"
        output_path = _hermes_home / ".update_output.txt"

        if not pending_path.exists():
            return

        try:
            pending = json.loads(pending_path.read_text())
            platform_str = pending.get("platform")
            chat_id = pending.get("chat_id")

            # Read the captured update output
            output = ""
            if output_path.exists():
                output = output_path.read_text()

            # Resolve adapter
            platform = Platform(platform_str)
            adapter = self.adapters.get(platform)

            if adapter and chat_id:
                # Strip ANSI escape codes for clean display
                output = _re.sub(r'\x1b\[[0-9;]*m', '', output).strip()
                if output:
                    # Truncate if too long for a single message
                    if len(output) > 3500:
                        output = "…" + output[-3500:]
                    msg = f"✅ Hermes update finished — gateway restarted.\n\n```\n{output}\n```"
                else:
                    msg = "✅ Hermes update finished — gateway restarted successfully."
                await adapter.send(chat_id, msg)
                logger.info("Sent post-update notification to %s:%s", platform_str, chat_id)
        except Exception as e:
            logger.warning("Post-update notification failed: %s", e)
        finally:
            pending_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def _set_session_env(self, context: SessionContext) -> None:
        """Set environment variables for the current session."""
        os.environ["HERMES_SESSION_PLATFORM"] = context.source.platform.value
        os.environ["HERMES_SESSION_CHAT_ID"] = context.source.chat_id
        if context.source.chat_name:
            os.environ["HERMES_SESSION_CHAT_NAME"] = context.source.chat_name
    
    def _clear_session_env(self) -> None:
        """Clear session environment variables."""
        for var in ["HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID", "HERMES_SESSION_CHAT_NAME"]:
            if var in os.environ:
                del os.environ[var]
    
    async def _enrich_message_with_vision(
        self,
        user_text: str,
        image_paths: List[str],
    ) -> str:
        """
        Auto-analyze user-attached images with the vision tool and prepend
        the descriptions to the message text.

        Each image is analyzed with a general-purpose prompt.  The resulting
        description *and* the local cache path are injected so the model can:
          1. Immediately understand what the user sent (no extra tool call).
          2. Re-examine the image with vision_analyze if it needs more detail.

        Args:
            user_text:   The user's original caption / message text.
            image_paths: List of local file paths to cached images.

        Returns:
            The enriched message string with vision descriptions prepended.
        """
        from tools.vision_tools import vision_analyze_tool
        import json as _json

        analysis_prompt = (
            "Describe everything visible in this image in thorough detail. "
            "Include any text, code, data, objects, people, layout, colors, "
            "and any other notable visual information."
        )

        enriched_parts = []
        for path in image_paths:
            try:
                logger.debug("Auto-analyzing user image: %s", path)
                result_json = await vision_analyze_tool(
                    image_url=path,
                    user_prompt=analysis_prompt,
                )
                result = _json.loads(result_json)
                if result.get("success"):
                    description = result.get("analysis", "")
                    enriched_parts.append(
                        f"[The user sent an image~ Here's what I can see:\n{description}]\n"
                        f"[If you need a closer look, use vision_analyze with "
                        f"image_url: {path} ~]"
                    )
                else:
                    enriched_parts.append(
                        "[The user sent an image but I couldn't quite see it "
                        "this time (>_<) You can try looking at it yourself "
                        f"with vision_analyze using image_url: {path}]"
                    )
            except Exception as e:
                logger.error("Vision auto-analysis error: %s", e)
                enriched_parts.append(
                    f"[The user sent an image but something went wrong when I "
                    f"tried to look at it~ You can try examining it yourself "
                    f"with vision_analyze using image_url: {path}]"
                )

        # Combine: vision descriptions first, then the user's original text
        if enriched_parts:
            prefix = "\n\n".join(enriched_parts)
            if user_text:
                return f"{prefix}\n\n{user_text}"
            return prefix
        return user_text

    async def _enrich_message_with_transcription(
        self,
        user_text: str,
        audio_paths: List[str],
    ) -> str:
        """
        Auto-transcribe user voice/audio messages using OpenAI Whisper API
        and prepend the transcript to the message text.

        Args:
            user_text:   The user's original caption / message text.
            audio_paths: List of local file paths to cached audio files.

        Returns:
            The enriched message string with transcriptions prepended.
        """
        from tools.transcription_tools import transcribe_audio
        import asyncio

        enriched_parts = []
        for path in audio_paths:
            try:
                logger.debug("Transcribing user voice: %s", path)
                result = await asyncio.to_thread(transcribe_audio, path)
                if result["success"]:
                    transcript = result["transcript"]
                    enriched_parts.append(
                        f'[The user sent a voice message~ '
                        f'Here\'s what they said: "{transcript}"]'
                    )
                else:
                    error = result.get("error", "unknown error")
                    if "OPENAI_API_KEY" in error or "VOICE_TOOLS_OPENAI_KEY" in error:
                        enriched_parts.append(
                            "[The user sent a voice message but I can't listen "
                            "to it right now~ VOICE_TOOLS_OPENAI_KEY isn't set up yet "
                            "(';w;') Let them know!]"
                        )
                    else:
                        enriched_parts.append(
                            "[The user sent a voice message but I had trouble "
                            f"transcribing it~ ({error})]"
                        )
            except Exception as e:
                logger.error("Transcription error: %s", e)
                enriched_parts.append(
                    "[The user sent a voice message but something went wrong "
                    "when I tried to listen to it~ Let them know!]"
                )

        if enriched_parts:
            prefix = "\n\n".join(enriched_parts)
            if user_text:
                return f"{prefix}\n\n{user_text}"
            return prefix
        return user_text

    async def _run_process_watcher(self, watcher: dict) -> None:
        """
        Periodically check a background process and push updates to the user.

        Runs as an asyncio task. Stays silent when nothing changed.
        Auto-removes when the process exits or is killed.
        """
        from tools.process_registry import process_registry

        session_id = watcher["session_id"]
        interval = watcher["check_interval"]
        session_key = watcher.get("session_key", "")
        platform_name = watcher.get("platform", "")
        chat_id = watcher.get("chat_id", "")

        logger.debug("Process watcher started: %s (every %ss)", session_id, interval)

        last_output_len = 0
        while True:
            await asyncio.sleep(interval)

            session = process_registry.get(session_id)
            if session is None:
                break

            current_output_len = len(session.output_buffer)
            has_new_output = current_output_len > last_output_len
            last_output_len = current_output_len

            if session.exited:
                # Process finished -- deliver final update
                new_output = session.output_buffer[-1000:] if session.output_buffer else ""
                message_text = (
                    f"[Background process {session_id} finished with exit code {session.exit_code}~ "
                    f"Here's the final output:\n{new_output}]"
                )
                # Try to deliver to the originating platform
                adapter = None
                for p, a in self.adapters.items():
                    if p.value == platform_name:
                        adapter = a
                        break
                if adapter and chat_id:
                    try:
                        await adapter.send(chat_id, message_text)
                    except Exception as e:
                        logger.error("Watcher delivery error: %s", e)
                break

            elif has_new_output:
                # New output available -- deliver status update
                new_output = session.output_buffer[-500:] if session.output_buffer else ""
                message_text = (
                    f"[Background process {session_id} is still running~ "
                    f"New output:\n{new_output}]"
                )
                adapter = None
                for p, a in self.adapters.items():
                    if p.value == platform_name:
                        adapter = a
                        break
                if adapter and chat_id:
                    try:
                        await adapter.send(chat_id, message_text)
                    except Exception as e:
                        logger.error("Watcher delivery error: %s", e)

        logger.debug("Process watcher ended: %s", session_id)

    async def _run_agent(
        self,
        message: str,
        context_prompt: str,
        history: List[Dict[str, Any]],
        source: SessionSource,
        session_id: str,
        session_key: str = None
    ) -> Dict[str, Any]:
        """
        Run the agent with the given message and context.
        
        Returns the full result dict from run_conversation, including:
          - "final_response": str (the text to send back)
          - "messages": list (full conversation including tool calls)
          - "api_calls": int
          - "completed": bool
        
        This is run in a thread pool to not block the event loop.
        Supports interruption via new messages.
        """
        from run_agent import AIAgent
        import queue
        
        # Determine toolset based on platform.
        # Check config.yaml for per-platform overrides, fallback to hardcoded defaults.
        default_toolset_map = {
            Platform.LOCAL: "hermes-cli",
            Platform.TELEGRAM: "hermes-telegram",
            Platform.DISCORD: "hermes-discord",
            Platform.WHATSAPP: "hermes-whatsapp",
            Platform.SLACK: "hermes-slack",
            Platform.SIGNAL: "hermes-signal",
            Platform.HOMEASSISTANT: "hermes-homeassistant",
        }
        
        # Try to load platform_toolsets from config
        platform_toolsets_config = {}
        try:
            config_path = _hermes_home / 'config.yaml'
            if config_path.exists():
                import yaml
                with open(config_path, 'r', encoding="utf-8") as f:
                    user_config = yaml.safe_load(f) or {}
                platform_toolsets_config = user_config.get("platform_toolsets", {})
        except Exception as e:
            logger.debug("Could not load platform_toolsets config: %s", e)
        
        # Map platform enum to config key
        platform_config_key = {
            Platform.LOCAL: "cli",
            Platform.TELEGRAM: "telegram",
            Platform.DISCORD: "discord",
            Platform.WHATSAPP: "whatsapp",
            Platform.SLACK: "slack",
            Platform.SIGNAL: "signal",
            Platform.HOMEASSISTANT: "homeassistant",
        }.get(source.platform, "telegram")
        
        # Use config override if present (list of toolsets), otherwise hardcoded default
        config_toolsets = platform_toolsets_config.get(platform_config_key)
        if config_toolsets and isinstance(config_toolsets, list):
            enabled_toolsets = config_toolsets
        else:
            default_toolset = default_toolset_map.get(source.platform, "hermes-telegram")
            enabled_toolsets = [default_toolset]
        
        # Tool progress mode from config.yaml: "all", "new", "verbose", "off"
        # Falls back to env vars for backward compatibility
        _progress_cfg = {}
        try:
            _tp_cfg_path = _hermes_home / "config.yaml"
            if _tp_cfg_path.exists():
                import yaml as _tp_yaml
                with open(_tp_cfg_path, encoding="utf-8") as _tp_f:
                    _tp_data = _tp_yaml.safe_load(_tp_f) or {}
                _progress_cfg = _tp_data.get("display", {})
        except Exception:
            pass
        progress_mode = (
            _progress_cfg.get("tool_progress")
            or os.getenv("HERMES_TOOL_PROGRESS_MODE")
            or "all"
        )
        tool_progress_enabled = progress_mode != "off"
        
        # Queue for progress messages (thread-safe)
        progress_queue = queue.Queue() if tool_progress_enabled else None
        last_tool = [None]  # Mutable container for tracking in closure
        
        def progress_callback(tool_name: str, preview: str = None, args: dict = None):
            """Callback invoked by agent when a tool is called."""
            if not progress_queue:
                return
            
            # "new" mode: only report when tool changes
            if progress_mode == "new" and tool_name == last_tool[0]:
                return
            last_tool[0] = tool_name
            
            # Build progress message with primary argument preview
            tool_emojis = {
                "terminal": "💻",
                "process": "⚙️",
                "web_search": "🔍",
                "web_extract": "📄",
                "read_file": "📖",
                "write_file": "✍️",
                "patch": "🔧",
                "search": "🔎",
                "search_files": "🔎",
                "list_directory": "📂",
                "image_generate": "🎨",
                "text_to_speech": "🔊",
                "browser_navigate": "🌐",
                "browser_click": "👆",
                "browser_type": "⌨️",
                "browser_snapshot": "📸",
                "browser_scroll": "📜",
                "browser_back": "◀️",
                "browser_press": "⌨️",
                "browser_close": "🚪",
                "browser_get_images": "🖼️",
                "browser_vision": "👁️",
                "moa_query": "🧠",
                "mixture_of_agents": "🧠",
                "vision_analyze": "👁️",
                "skill_view": "📚",
                "skills_list": "📋",
                "todo": "📋",
                "memory": "🧠",
                "session_search": "🔍",
                "send_message": "📨",
                "schedule_cronjob": "⏰",
                "list_cronjobs": "⏰",
                "remove_cronjob": "⏰",
                "execute_code": "🐍",
                "delegate_task": "🔀",
                "clarify": "❓",
                "skill_manage": "📝",
            }
            emoji = tool_emojis.get(tool_name, "⚙️")
            
            # Verbose mode: show detailed arguments
            if progress_mode == "verbose" and args:
                import json as _json
                args_str = _json.dumps(args, ensure_ascii=False, default=str)
                if len(args_str) > 200:
                    args_str = args_str[:197] + "..."
                msg = f"{emoji} {tool_name}({list(args.keys())})\n{args_str}"
                progress_queue.put(msg)
                return
            
            if preview:
                # Truncate preview to keep messages clean
                if len(preview) > 80:
                    preview = preview[:77] + "..."
                msg = f"{emoji} {tool_name}: \"{preview}\""
            else:
                msg = f"{emoji} {tool_name}..."
            
            progress_queue.put(msg)
        
        # Background task to send progress messages
        # Accumulates tool lines into a single message that gets edited
        async def send_progress_messages():
            if not progress_queue:
                return

            adapter = self.adapters.get(source.platform)
            if not adapter:
                return

            progress_lines = []      # Accumulated tool lines
            progress_msg_id = None   # ID of the progress message to edit
            can_edit = True          # False once an edit fails (platform doesn't support it)

            while True:
                try:
                    msg = progress_queue.get_nowait()
                    progress_lines.append(msg)

                    if can_edit and progress_msg_id is not None:
                        # Try to edit the existing progress message
                        full_text = "\n".join(progress_lines)
                        result = await adapter.edit_message(
                            chat_id=source.chat_id,
                            message_id=progress_msg_id,
                            content=full_text,
                        )
                        if not result.success:
                            # Platform doesn't support editing — stop trying,
                            # send just this new line as a separate message
                            can_edit = False
                            await adapter.send(chat_id=source.chat_id, content=msg)
                    else:
                        if can_edit:
                            # First tool: send all accumulated text as new message
                            full_text = "\n".join(progress_lines)
                            result = await adapter.send(chat_id=source.chat_id, content=full_text)
                        else:
                            # Editing unsupported: send just this line
                            result = await adapter.send(chat_id=source.chat_id, content=msg)
                        if result.success and result.message_id:
                            progress_msg_id = result.message_id

                    # Restore typing indicator
                    await asyncio.sleep(0.3)
                    await adapter.send_typing(source.chat_id)

                except queue.Empty:
                    await asyncio.sleep(0.3)
                except asyncio.CancelledError:
                    # Drain remaining queued messages
                    while not progress_queue.empty():
                        try:
                            msg = progress_queue.get_nowait()
                            progress_lines.append(msg)
                        except Exception:
                            break
                    # Final edit with all remaining tools (only if editing works)
                    if can_edit and progress_lines and progress_msg_id:
                        full_text = "\n".join(progress_lines)
                        try:
                            await adapter.edit_message(
                                chat_id=source.chat_id,
                                message_id=progress_msg_id,
                                content=full_text,
                            )
                        except Exception:
                            pass
                    return
                except Exception as e:
                    logger.error("Progress message error: %s", e)
                    await asyncio.sleep(1)
        
        # We need to share the agent instance for interrupt support
        agent_holder = [None]  # Mutable container for the agent instance
        result_holder = [None]  # Mutable container for the result
        tools_holder = [None]   # Mutable container for the tool definitions
        
        # Bridge sync step_callback → async hooks.emit for agent:step events
        _loop_for_step = asyncio.get_event_loop()
        _hooks_ref = self.hooks

        def _step_callback_sync(iteration: int, tool_names: list) -> None:
            try:
                asyncio.run_coroutine_threadsafe(
                    _hooks_ref.emit("agent:step", {
                        "platform": source.platform.value if source.platform else "",
                        "user_id": source.user_id,
                        "session_id": session_id,
                        "iteration": iteration,
                        "tool_names": tool_names,
                    }),
                    _loop_for_step,
                )
            except Exception as _e:
                logger.debug("agent:step hook error: %s", _e)

        def run_sync():
            # Pass session_key to process registry via env var so background
            # processes can be mapped back to this gateway session
            os.environ["HERMES_SESSION_KEY"] = session_key or ""

            # Read from env var or use default (same as CLI)
            max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", "90"))
            
            # Map platform enum to the platform hint key the agent understands.
            # Platform.LOCAL ("local") maps to "cli"; others pass through as-is.
            platform_key = "cli" if source.platform == Platform.LOCAL else source.platform.value
            
            # Combine platform context with user-configured ephemeral system prompt
            combined_ephemeral = context_prompt or ""
            if self._ephemeral_system_prompt:
                combined_ephemeral = (combined_ephemeral + "\n\n" + self._ephemeral_system_prompt).strip()

            # Re-read .env and config for fresh credentials (gateway is long-lived,
            # keys may change without restart).
            try:
                load_dotenv(_env_path, override=True, encoding="utf-8")
            except UnicodeDecodeError:
                load_dotenv(_env_path, override=True, encoding="latin-1")
            except Exception:
                pass

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

            try:
                runtime_kwargs = _resolve_runtime_agent_kwargs()
            except Exception as exc:
                return {
                    "final_response": f"⚠️ Provider authentication failed: {exc}",
                    "messages": [],
                    "api_calls": 0,
                    "tools": [],
                }

            pr = self._provider_routing
            agent = AIAgent(
                model=model,
                **runtime_kwargs,
                max_iterations=max_iterations,
                quiet_mode=True,
                verbose_logging=False,
                enabled_toolsets=enabled_toolsets,
                ephemeral_system_prompt=combined_ephemeral or None,
                prefill_messages=self._prefill_messages or None,
                reasoning_config=self._reasoning_config,
                providers_allowed=pr.get("only"),
                providers_ignored=pr.get("ignore"),
                providers_order=pr.get("order"),
                provider_sort=pr.get("sort"),
                provider_require_parameters=pr.get("require_parameters", False),
                provider_data_collection=pr.get("data_collection"),
                session_id=session_id,
                tool_progress_callback=progress_callback if tool_progress_enabled else None,
                step_callback=_step_callback_sync if _hooks_ref.loaded_hooks else None,
                platform=platform_key,
                honcho_session_key=session_key,
                session_db=self._session_db,
                fallback_model=self._fallback_model,
            )
            
            # Store agent reference for interrupt support
            agent_holder[0] = agent
            # Capture the full tool definitions for transcript logging
            tools_holder[0] = agent.tools if hasattr(agent, 'tools') else None
            
            # Convert history to agent format.
            # Two cases:
            #   1. Normal path (from transcript): simple {role, content, timestamp} dicts
            #      - Strip timestamps, keep role+content
            #   2. Interrupt path (from agent result["messages"]): full agent messages
            #      that may include tool_calls, tool_call_id, reasoning, etc.
            #      - These must be passed through intact so the API sees valid
            #        assistant→tool sequences (dropping tool_calls causes 500 errors)
            agent_history = []
            for msg in history:
                role = msg.get("role")
                if not role:
                    continue
                
                # Skip metadata entries (tool definitions, session info)
                # -- these are for transcript logging, not for the LLM
                if role in ("session_meta",):
                    continue
                
                # Skip system messages -- the agent rebuilds its own system prompt
                if role == "system":
                    continue
                
                # Rich agent messages (tool_calls, tool results) must be passed
                # through intact so the API sees valid assistant→tool sequences
                has_tool_calls = "tool_calls" in msg
                has_tool_call_id = "tool_call_id" in msg
                is_tool_message = role == "tool"
                
                if has_tool_calls or has_tool_call_id or is_tool_message:
                    clean_msg = {k: v for k, v in msg.items() if k != "timestamp"}
                    agent_history.append(clean_msg)
                else:
                    # Simple text message - just need role and content
                    content = msg.get("content")
                    if content:
                        # Tag cross-platform mirror messages so the agent knows their origin
                        if msg.get("mirror"):
                            mirror_src = msg.get("mirror_source", "another session")
                            content = f"[Delivered from {mirror_src}] {content}"
                        agent_history.append({"role": role, "content": content})
            
            # Collect MEDIA paths already in history so we can exclude them
            # from the current turn's extraction. This is compression-safe:
            # even if the message list shrinks, we know which paths are old.
            _history_media_paths: set = set()
            for _hm in agent_history:
                if _hm.get("role") in ("tool", "function"):
                    _hc = _hm.get("content", "")
                    if "MEDIA:" in _hc:
                        for _match in re.finditer(r'MEDIA:(\S+)', _hc):
                            _p = _match.group(1).strip().rstrip('",}')
                            if _p:
                                _history_media_paths.add(_p)
            
            result = agent.run_conversation(message, conversation_history=agent_history, task_id=session_id)
            result_holder[0] = result
            
            # Return final response, or a message if something went wrong
            final_response = result.get("final_response")
            if not final_response:
                error_msg = f"⚠️ {result['error']}" if result.get("error") else "(No response generated)"
                return {
                    "final_response": error_msg,
                    "messages": result.get("messages", []),
                    "api_calls": result.get("api_calls", 0),
                    "tools": tools_holder[0] or [],
                    "history_offset": len(agent_history),
                }
            
            # Scan tool results for MEDIA:<path> tags that need to be delivered
            # as native audio/file attachments.  The TTS tool embeds MEDIA: tags
            # in its JSON response, but the model's final text reply usually
            # doesn't include them.  We collect unique tags from tool results and
            # append any that aren't already present in the final response, so the
            # adapter's extract_media() can find and deliver the files exactly once.
            #
            # Uses path-based deduplication against _history_media_paths (collected
            # before run_conversation) instead of index slicing. This is safe even
            # when context compression shrinks the message list. (Fixes #160)
            if "MEDIA:" not in final_response:
                media_tags = []
                has_voice_directive = False
                for msg in result.get("messages", []):
                    if msg.get("role") in ("tool", "function"):
                        content = msg.get("content", "")
                        if "MEDIA:" in content:
                            for match in re.finditer(r'MEDIA:(\S+)', content):
                                path = match.group(1).strip().rstrip('",}')
                                if path and path not in _history_media_paths:
                                    media_tags.append(f"MEDIA:{path}")
                            if "[[audio_as_voice]]" in content:
                                has_voice_directive = True
                
                if media_tags:
                    seen = set()
                    unique_tags = []
                    for tag in media_tags:
                        if tag not in seen:
                            seen.add(tag)
                            unique_tags.append(tag)
                    if has_voice_directive:
                        unique_tags.insert(0, "[[audio_as_voice]]")
                    final_response = final_response + "\n" + "\n".join(unique_tags)
            
            return {
                "final_response": final_response,
                "messages": result_holder[0].get("messages", []) if result_holder[0] else [],
                "api_calls": result_holder[0].get("api_calls", 0) if result_holder[0] else 0,
                "tools": tools_holder[0] or [],
                "history_offset": len(agent_history),
            }
        
        # Start progress message sender if enabled
        progress_task = None
        if tool_progress_enabled:
            progress_task = asyncio.create_task(send_progress_messages())
        
        # Track this agent as running for this session (for interrupt support)
        # We do this in a callback after the agent is created
        async def track_agent():
            # Wait for agent to be created
            while agent_holder[0] is None:
                await asyncio.sleep(0.05)
            if session_key:
                self._running_agents[session_key] = agent_holder[0]
        
        tracking_task = asyncio.create_task(track_agent())
        
        # Monitor for interrupts from the adapter (new messages arriving)
        async def monitor_for_interrupt():
            adapter = self.adapters.get(source.platform)
            if not adapter:
                return
            
            chat_id = source.chat_id
            while True:
                await asyncio.sleep(0.2)  # Check every 200ms
                # Check if adapter has a pending interrupt for this session
                if hasattr(adapter, 'has_pending_interrupt') and adapter.has_pending_interrupt(chat_id):
                    agent = agent_holder[0]
                    if agent:
                        pending_event = adapter.get_pending_message(chat_id)
                        pending_text = pending_event.text if pending_event else None
                        logger.debug("Interrupt detected from adapter, signaling agent...")
                        agent.interrupt(pending_text)
                        break
        
        interrupt_monitor = asyncio.create_task(monitor_for_interrupt())
        
        try:
            # Run in thread pool to not block
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, run_sync)
            
            # Check if we were interrupted and have a pending message
            result = result_holder[0]
            adapter = self.adapters.get(source.platform)
            
            # Get pending message from adapter if interrupted
            pending = None
            if result and result.get("interrupted") and adapter:
                pending_event = adapter.get_pending_message(source.chat_id)
                if pending_event:
                    pending = pending_event.text
                elif result.get("interrupt_message"):
                    pending = result.get("interrupt_message")
            
            if pending:
                logger.debug("Processing interrupted message: '%s...'", pending[:40])
                
                # Clear the adapter's interrupt event so the next _run_agent call
                # doesn't immediately re-trigger the interrupt before the new agent
                # even makes its first API call (this was causing an infinite loop).
                if adapter and hasattr(adapter, '_active_sessions') and source.chat_id in adapter._active_sessions:
                    adapter._active_sessions[source.chat_id].clear()
                
                # Don't send the interrupted response to the user — it's just noise
                # like "Operation interrupted." They already know they sent a new
                # message, so go straight to processing it.
                
                # Now process the pending message with updated history
                updated_history = result.get("messages", history)
                return await self._run_agent(
                    message=pending,
                    context_prompt=context_prompt,
                    history=updated_history,
                    source=source,
                    session_id=session_id,
                    session_key=session_key
                )
        finally:
            # Stop progress sender and interrupt monitor
            if progress_task:
                progress_task.cancel()
            interrupt_monitor.cancel()
            
            # Clean up tracking
            tracking_task.cancel()
            if session_key and session_key in self._running_agents:
                del self._running_agents[session_key]
            
            # Wait for cancelled tasks
            for task in [progress_task, interrupt_monitor, tracking_task]:
                if task:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
        
        return response


def _start_cron_ticker(stop_event: threading.Event, adapters=None, interval: int = 60):
    """
    Background thread that ticks the cron scheduler at a regular interval.
    
    Runs inside the gateway process so cronjobs fire automatically without
    needing a separate `hermes cron daemon` or system cron entry.

    Also refreshes the channel directory every 5 minutes and prunes the
    image/audio/document cache once per hour.
    """
    from cron.scheduler import tick as cron_tick
    from gateway.platforms.base import cleanup_image_cache, cleanup_document_cache

    IMAGE_CACHE_EVERY = 60   # ticks — once per hour at default 60s interval
    CHANNEL_DIR_EVERY = 5    # ticks — every 5 minutes

    logger.info("Cron ticker started (interval=%ds)", interval)
    tick_count = 0
    while not stop_event.is_set():
        try:
            cron_tick(verbose=False)
        except Exception as e:
            logger.debug("Cron tick error: %s", e)

        tick_count += 1

        if tick_count % CHANNEL_DIR_EVERY == 0 and adapters:
            try:
                from gateway.channel_directory import build_channel_directory
                build_channel_directory(adapters)
            except Exception as e:
                logger.debug("Channel directory refresh error: %s", e)

        if tick_count % IMAGE_CACHE_EVERY == 0:
            try:
                removed = cleanup_image_cache(max_age_hours=24)
                if removed:
                    logger.info("Image cache cleanup: removed %d stale file(s)", removed)
            except Exception as e:
                logger.debug("Image cache cleanup error: %s", e)
            try:
                removed = cleanup_document_cache(max_age_hours=24)
                if removed:
                    logger.info("Document cache cleanup: removed %d stale file(s)", removed)
            except Exception as e:
                logger.debug("Document cache cleanup error: %s", e)

        stop_event.wait(timeout=interval)
    logger.info("Cron ticker stopped")


async def start_gateway(config: Optional[GatewayConfig] = None, replace: bool = False) -> bool:
    """
    Start the gateway and run until interrupted.
    
    This is the main entry point for running the gateway.
    Returns True if the gateway ran successfully, False if it failed to start.
    A False return causes a non-zero exit code so systemd can auto-restart.
    
    Args:
        config: Optional gateway configuration override.
        replace: If True, kill any existing gateway instance before starting.
                 Useful for systemd services to avoid restart-loop deadlocks
                 when the previous process hasn't fully exited yet.
    """
    # ── Duplicate-instance guard ──────────────────────────────────────
    # Prevent two gateways from running under the same HERMES_HOME.
    # The PID file is scoped to HERMES_HOME, so future multi-profile
    # setups (each profile using a distinct HERMES_HOME) will naturally
    # allow concurrent instances without tripping this guard.
    import time as _time
    from gateway.status import get_running_pid, remove_pid_file
    existing_pid = get_running_pid()
    if existing_pid is not None and existing_pid != os.getpid():
        if replace:
            logger.info(
                "Replacing existing gateway instance (PID %d) with --replace.",
                existing_pid,
            )
            try:
                os.kill(existing_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # Already gone
            except PermissionError:
                logger.error(
                    "Permission denied killing PID %d. Cannot replace.",
                    existing_pid,
                )
                return False
            # Wait up to 10 seconds for the old process to exit
            for _ in range(20):
                try:
                    os.kill(existing_pid, 0)
                    _time.sleep(0.5)
                except (ProcessLookupError, PermissionError):
                    break  # Process is gone
            else:
                # Still alive after 10s — force kill
                logger.warning(
                    "Old gateway (PID %d) did not exit after SIGTERM, sending SIGKILL.",
                    existing_pid,
                )
                try:
                    os.kill(existing_pid, signal.SIGKILL)
                    _time.sleep(0.5)
                except (ProcessLookupError, PermissionError):
                    pass
            remove_pid_file()
        else:
            hermes_home = os.getenv("HERMES_HOME", "~/.hermes")
            logger.error(
                "Another gateway instance is already running (PID %d, HERMES_HOME=%s). "
                "Use 'hermes gateway restart' to replace it, or 'hermes gateway stop' first.",
                existing_pid, hermes_home,
            )
            print(
                f"\n❌ Gateway already running (PID {existing_pid}).\n"
                f"   Use 'hermes gateway restart' to replace it,\n"
                f"   or 'hermes gateway stop' to kill it first.\n"
                f"   Or use 'hermes gateway run --replace' to auto-replace.\n"
            )
            return False

    # Sync bundled skills on gateway start (fast -- skips unchanged)
    try:
        from tools.skills_sync import sync_skills
        sync_skills(quiet=True)
    except Exception:
        pass

    # Configure rotating file log so gateway output is persisted for debugging
    log_dir = _hermes_home / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / 'gateway.log',
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
    )
    from agent.redact import RedactingFormatter
    file_handler.setFormatter(RedactingFormatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.INFO)

    # Separate errors-only log for easy debugging
    error_handler = RotatingFileHandler(
        log_dir / 'errors.log',
        maxBytes=2 * 1024 * 1024,
        backupCount=2,
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(RedactingFormatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    logging.getLogger().addHandler(error_handler)

    runner = GatewayRunner(config)
    
    # Set up signal handlers
    def signal_handler():
        asyncio.create_task(runner.stop())
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass
    
    # Start the gateway
    success = await runner.start()
    if not success:
        return False
    
    # Write PID file so CLI can detect gateway is running
    import atexit
    from gateway.status import write_pid_file, remove_pid_file
    write_pid_file()
    atexit.register(remove_pid_file)
    
    # Start background cron ticker so scheduled jobs fire automatically
    cron_stop = threading.Event()
    cron_thread = threading.Thread(
        target=_start_cron_ticker,
        args=(cron_stop,),
        kwargs={"adapters": runner.adapters},
        daemon=True,
        name="cron-ticker",
    )
    cron_thread.start()
    
    # Wait for shutdown
    await runner.wait_for_shutdown()
    
    # Stop cron ticker cleanly
    cron_stop.set()
    cron_thread.join(timeout=5)

    # Close MCP server connections
    try:
        from tools.mcp_tool import shutdown_mcp_servers
        shutdown_mcp_servers()
    except Exception:
        pass

    return True


def main():
    """CLI entry point for the gateway."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hermes Gateway - Multi-platform messaging")
    parser.add_argument("--config", "-c", help="Path to gateway config file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    config = None
    if args.config:
        import json
        with open(args.config, encoding="utf-8") as f:
            data = json.load(f)
            config = GatewayConfig.from_dict(data)
    
    # Run the gateway - exit with code 1 if no platforms connected,
    # so systemd Restart=on-failure will retry on transient errors (e.g. DNS)
    success = asyncio.run(start_gateway(config))
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
