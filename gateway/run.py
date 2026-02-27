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
        with open(_config_path) as _f:
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
                "ssh_host": "TERMINAL_SSH_HOST",
                "ssh_user": "TERMINAL_SSH_USER",
                "ssh_port": "TERMINAL_SSH_PORT",
                "ssh_key": "TERMINAL_SSH_KEY",
                "container_cpu": "TERMINAL_CONTAINER_CPU",
                "container_memory": "TERMINAL_CONTAINER_MEMORY",
                "container_disk": "TERMINAL_CONTAINER_DISK",
                "container_persistent": "TERMINAL_CONTAINER_PERSISTENT",
            }
            for _cfg_key, _env_var in _terminal_env_map.items():
                if _cfg_key in _terminal_cfg:
                    os.environ[_env_var] = str(_terminal_cfg[_cfg_key])
    except Exception:
        pass  # Non-fatal; gateway can still run with .env values

# Gateway runs in quiet mode - suppress debug output and use cwd directly (no temp dirs)
os.environ["HERMES_QUIET"] = "1"

# Enable interactive exec approval for dangerous commands on messaging platforms
os.environ["HERMES_EXEC_ASK"] = "1"

# Set terminal working directory for messaging platforms
# Uses MESSAGING_CWD if set, otherwise defaults to home directory
# This is separate from CLI which uses the directory where `hermes` is run
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
)
from gateway.delivery import DeliveryRouter, DeliveryTarget
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType

logger = logging.getLogger(__name__)


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

        # Wire process registry into session store for reset protection
        from tools.process_registry import process_registry
        self.session_store = SessionStore(
            self.config.sessions_dir, self.config,
            has_active_processes_fn=lambda key: process_registry.has_active_for_session(key),
            on_auto_reset=self._flush_memories_before_reset,
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
    
    def _flush_memories_before_reset(self, old_entry):
        """Prompt the agent to save memories/skills before an auto-reset.
        
        Called synchronously by SessionStore before destroying an expired session.
        Loads the transcript, gives the agent a real turn with memory + skills
        tools, and explicitly asks it to preserve anything worth keeping.
        """
        try:
            history = self.session_store.load_transcript(old_entry.session_id)
            if not history or len(history) < 4:
                return

            from run_agent import AIAgent
            _flush_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
            _flush_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            _flush_model = os.getenv("HERMES_MODEL") or os.getenv("LLM_MODEL", "anthropic/claude-opus-4.6")

            if not _flush_api_key:
                return

            tmp_agent = AIAgent(
                model=_flush_model,
                api_key=_flush_api_key,
                base_url=_flush_base_url,
                max_iterations=8,
                quiet_mode=True,
                enabled_toolsets=["memory", "skills"],
                session_id=old_entry.session_id,
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
            logger.info("Pre-reset save completed for session %s", old_entry.session_id)
        except Exception as e:
            logger.debug("Pre-reset save failed for session %s: %s", old_entry.session_id, e)
    
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
                    with open(cfg_path) as _f:
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
                with open(cfg_path) as _f:
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
        Returns None to use default (xhigh).
        """
        effort = os.getenv("HERMES_REASONING_EFFORT", "")
        if not effort:
            try:
                import yaml as _y
                cfg_path = _hermes_home / "config.yaml"
                if cfg_path.exists():
                    with open(cfg_path) as _f:
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
        logger.warning("Unknown reasoning_effort '%s', using default (xhigh)", effort)
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
        
        logger.info("Press Ctrl+C to stop")
        
        return True
    
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
        user_id = source.user_id
        if not user_id:
            return False

        platform_env_map = {
            Platform.TELEGRAM: "TELEGRAM_ALLOWED_USERS",
            Platform.DISCORD: "DISCORD_ALLOWED_USERS",
            Platform.WHATSAPP: "WHATSAPP_ALLOWED_USERS",
            Platform.SLACK: "SLACK_ALLOWED_USERS",
        }
        platform_allow_all_map = {
            Platform.TELEGRAM: "TELEGRAM_ALLOW_ALL_USERS",
            Platform.DISCORD: "DISCORD_ALLOW_ALL_USERS",
            Platform.WHATSAPP: "WHATSAPP_ALLOW_ALL_USERS",
            Platform.SLACK: "SLACK_ALLOW_ALL_USERS",
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
        _quick_key = (
            f"agent:main:{source.platform.value}:{source.chat_type}:{source.chat_id}"
            if source.chat_type != "dm"
            else f"agent:main:{source.platform.value}:dm"
        )
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
        
        if command == "personality":
            return await self._handle_personality_command(event)
        
        if command == "retry":
            return await self._handle_retry_command(event)
        
        if command == "undo":
            return await self._handle_undo_command(event)
        
        if command in ["sethome", "set-home"]:
            return await self._handle_set_home_command(event)
        
        # Check for pending exec approval responses
        if source.chat_type != "dm":
            session_key_preview = f"agent:main:{source.platform.value}:{source.chat_type}:{source.chat_id}"
        elif source.platform and source.platform.value == "whatsapp" and source.chat_id:
            session_key_preview = f"agent:main:{source.platform.value}:dm:{source.chat_id}"
        else:
            session_key_preview = f"agent:main:{source.platform.value}:dm"
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
            
            # Find only the NEW messages from this turn (skip history we loaded)
            history_len = len(history)
            new_messages = agent_messages[history_len:] if len(agent_messages) > history_len else agent_messages
            
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
        session_key = f"agent:main:{source.platform.value}:" + \
                      (f"dm" if source.chat_type == "dm" else f"{source.chat_type}:{source.chat_id}")
        
        # Memory flush before reset: load the old transcript and let a
        # temporary agent save memories before the session is wiped.
        try:
            old_entry = self.session_store._sessions.get(session_key)
            if old_entry:
                old_history = self.session_store.load_transcript(old_entry.session_id)
                if old_history:
                    from run_agent import AIAgent
                    loop = asyncio.get_event_loop()
                    # Resolve credentials so the flush agent can reach the LLM
                    _flush_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
                    _flush_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
                    _flush_model = os.getenv("HERMES_MODEL") or os.getenv("LLM_MODEL", "anthropic/claude-opus-4.6")
                    def _do_flush():
                        tmp_agent = AIAgent(
                            model=_flush_model,
                            api_key=_flush_api_key,
                            base_url=_flush_base_url,
                            max_iterations=5,
                            quiet_mode=True,
                            enabled_toolsets=["memory"],
                            session_id=old_entry.session_id,
                        )
                        # Build simple message list from transcript
                        msgs = []
                        for m in old_history:
                            role = m.get("role")
                            content = m.get("content")
                            if role in ("user", "assistant") and content:
                                msgs.append({"role": role, "content": content})
                        tmp_agent.flush_memories(msgs)
                    await loop.run_in_executor(None, _do_flush)
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
        return (
            "📖 **Hermes Commands**\n"
            "\n"
            "`/new` — Start a new conversation\n"
            "`/reset` — Reset conversation history\n"
            "`/status` — Show session info\n"
            "`/stop` — Interrupt the running agent\n"
            "`/model [name]` — Show or change the model\n"
            "`/personality [name]` — Set a personality\n"
            "`/retry` — Retry your last message\n"
            "`/undo` — Remove the last exchange\n"
            "`/sethome` — Set this chat as the home channel\n"
            "`/help` — Show this message"
        )
    
    async def _handle_model_command(self, event: MessageEvent) -> str:
        """Handle /model command - show or change the current model."""
        args = event.get_command_args().strip()
        current = os.getenv("HERMES_MODEL", "anthropic/claude-opus-4.6")
        
        if not args:
            return f"🤖 **Current model:** `{current}`\n\nTo change: `/model provider/model-name`"
        
        os.environ["HERMES_MODEL"] = args
        return f"🤖 Model changed to `{args}`\n_(takes effect on next message)_"
    
    async def _handle_personality_command(self, event: MessageEvent) -> str:
        """Handle /personality command - list or set a personality."""
        args = event.get_command_args().strip().lower()
        
        try:
            import yaml
            config_path = _hermes_home / 'config.yaml'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
                personalities = config.get("agent", {}).get("personalities", {})
            else:
                personalities = {}
        except Exception:
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
            os.environ["HERMES_PERSONALITY"] = personalities[args]
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
        
        # Truncate history to before the last user message
        truncated = history[:last_user_idx]
        session_entry.conversation_history = truncated
        
        # Re-send by creating a fake text event with the old message
        retry_event = MessageEvent(
            text=last_user_msg,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=event.raw_message,
        )
        
        # Let the normal message handler process it
        await self._handle_message(retry_event)
        return None  # Response sent through normal flow
    
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
        session_entry.conversation_history = history[:last_user_idx]
        
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
                with open(config_path) as f:
                    user_config = yaml.safe_load(f) or {}
            user_config[env_key] = chat_id
            with open(config_path, 'w') as f:
                yaml.dump(user_config, f, default_flow_style=False)
            # Also set in the current environment so it takes effect immediately
            os.environ[env_key] = str(chat_id)
        except Exception as e:
            return f"Failed to save home channel: {e}"
        
        return (
            f"✅ Home channel set to **{chat_name}** (ID: {chat_id}).\n"
            f"Cron jobs and cross-platform messages will be delivered here."
        )
    
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
        }
        
        # Try to load platform_toolsets from config
        platform_toolsets_config = {}
        try:
            config_path = _hermes_home / 'config.yaml'
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
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
        }.get(source.platform, "telegram")
        
        # Use config override if present (list of toolsets), otherwise hardcoded default
        config_toolsets = platform_toolsets_config.get(platform_config_key)
        if config_toolsets and isinstance(config_toolsets, list):
            enabled_toolsets = config_toolsets
        else:
            default_toolset = default_toolset_map.get(source.platform, "hermes-telegram")
            enabled_toolsets = [default_toolset]
        
        # Check if tool progress notifications are enabled
        tool_progress_enabled = os.getenv("HERMES_TOOL_PROGRESS", "true").lower() in ("1", "true", "yes")
        progress_mode = os.getenv("HERMES_TOOL_PROGRESS_MODE", "all")  # "all" or "new" (only new tools)
        
        # Queue for progress messages (thread-safe)
        progress_queue = queue.Queue() if tool_progress_enabled else None
        last_tool = [None]  # Mutable container for tracking in closure
        
        def progress_callback(tool_name: str, preview: str = None):
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
            }
            emoji = tool_emojis.get(tool_name, "⚙️")
            
            if preview:
                # Truncate preview to keep messages clean
                if len(preview) > 40:
                    preview = preview[:37] + "..."
                msg = f"{emoji} {tool_name}... \"{preview}\""
            else:
                msg = f"{emoji} {tool_name}..."
            
            progress_queue.put(msg)
        
        # Background task to send progress messages
        async def send_progress_messages():
            if not progress_queue:
                return
            
            adapter = self.adapters.get(source.platform)
            if not adapter:
                return
            
            while True:
                try:
                    # Non-blocking check with small timeout
                    msg = progress_queue.get_nowait()
                    await adapter.send(chat_id=source.chat_id, content=msg)
                    # Restore typing indicator after sending progress message
                    await asyncio.sleep(0.3)
                    await adapter.send_typing(source.chat_id)
                except queue.Empty:
                    await asyncio.sleep(0.3)  # Check again soon
                except asyncio.CancelledError:
                    # Drain remaining messages
                    while not progress_queue.empty():
                        try:
                            msg = progress_queue.get_nowait()
                            await adapter.send(chat_id=source.chat_id, content=msg)
                        except Exception:
                            break
                    return
                except Exception as e:
                    logger.error("Progress message error: %s", e)
                    await asyncio.sleep(1)
        
        # We need to share the agent instance for interrupt support
        agent_holder = [None]  # Mutable container for the agent instance
        result_holder = [None]  # Mutable container for the result
        tools_holder = [None]   # Mutable container for the tool definitions
        
        def run_sync():
            # Pass session_key to process registry via env var so background
            # processes can be mapped back to this gateway session
            os.environ["HERMES_SESSION_KEY"] = session_key or ""

            # Read from env var or use default (same as CLI)
            max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", "60"))
            
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

            # Custom endpoint (OPENAI_*) takes precedence, matching CLI behavior
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
            base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            model = os.getenv("HERMES_MODEL") or os.getenv("LLM_MODEL") or "anthropic/claude-opus-4.6"

            try:
                import yaml as _y
                _cfg_path = _hermes_home / "config.yaml"
                if _cfg_path.exists():
                    with open(_cfg_path) as _f:
                        _cfg = _y.safe_load(_f) or {}
                    _model_cfg = _cfg.get("model", {})
                    if isinstance(_model_cfg, str):
                        model = _model_cfg
                    elif isinstance(_model_cfg, dict):
                        model = _model_cfg.get("default", model)
                        base_url = _model_cfg.get("base_url", base_url)
                    # Check if provider is nous — resolve OAuth credentials
                    provider = _model_cfg.get("provider", "") if isinstance(_model_cfg, dict) else ""
                    if provider == "nous":
                        try:
                            from hermes_cli.auth import resolve_nous_runtime_credentials
                            creds = resolve_nous_runtime_credentials(min_key_ttl_seconds=5 * 60)
                            api_key = creds.get("api_key", api_key)
                            base_url = creds.get("base_url", base_url)
                        except Exception as nous_err:
                            logger.warning("Nous Portal credential resolution failed: %s", nous_err)
            except Exception:
                pass

            agent = AIAgent(
                model=model,
                api_key=api_key,
                base_url=base_url,
                max_iterations=max_iterations,
                quiet_mode=True,
                enabled_toolsets=enabled_toolsets,
                ephemeral_system_prompt=combined_ephemeral or None,
                prefill_messages=self._prefill_messages or None,
                reasoning_config=self._reasoning_config,
                session_id=session_id,
                tool_progress_callback=progress_callback if tool_progress_enabled else None,
                platform=platform_key,
                session_db=self._session_db,
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
            
            result = agent.run_conversation(message, conversation_history=agent_history)
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
                }
            
            # Scan tool results for MEDIA:<path> tags that need to be delivered
            # as native audio/file attachments.  The TTS tool embeds MEDIA: tags
            # in its JSON response, but the model's final text reply usually
            # doesn't include them.  We collect unique tags from tool results and
            # append any that aren't already present in the final response, so the
            # adapter's extract_media() can find and deliver the files exactly once.
            if "MEDIA:" not in final_response:
                media_tags = []
                has_voice_directive = False
                for msg in result.get("messages", []):
                    if msg.get("role") == "tool" or msg.get("role") == "function":
                        content = msg.get("content", "")
                        if "MEDIA:" in content:
                            for match in re.finditer(r'MEDIA:(\S+)', content):
                                path = match.group(1).strip().rstrip('",}')
                                if path:
                                    media_tags.append(f"MEDIA:{path}")
                            if "[[audio_as_voice]]" in content:
                                has_voice_directive = True
                
                if media_tags:
                    # Deduplicate while preserving order
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
    image/audio cache once per hour.
    """
    from cron.scheduler import tick as cron_tick
    from gateway.platforms.base import cleanup_image_cache

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

        stop_event.wait(timeout=interval)
    logger.info("Cron ticker stopped")


async def start_gateway(config: Optional[GatewayConfig] = None) -> bool:
    """
    Start the gateway and run until interrupted.
    
    This is the main entry point for running the gateway.
    Returns True if the gateway ran successfully, False if it failed to start.
    A False return causes a non-zero exit code so systemd can auto-restart.
    """
    # Configure rotating file log so gateway output is persisted for debugging
    log_dir = _hermes_home / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / 'gateway.log',
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
    )
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.INFO)

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
        with open(args.config) as f:
            data = json.load(f)
            config = GatewayConfig.from_dict(data)
    
    # Run the gateway - exit with code 1 if no platforms connected,
    # so systemd Restart=on-failure will retry on transient errors (e.g. DNS)
    success = asyncio.run(start_gateway(config))
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
