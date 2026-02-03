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
import os
import sys
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from ~/.hermes/.env first
from dotenv import load_dotenv
_env_path = Path.home() / '.hermes' / '.env'
if _env_path.exists():
    load_dotenv(_env_path)
# Also try project .env as fallback
load_dotenv()

# Gateway runs in quiet mode - suppress debug output and use cwd directly (no temp dirs)
os.environ["HERMES_QUIET"] = "1"

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
from gateway.platforms.base import BasePlatformAdapter, MessageEvent


class GatewayRunner:
    """
    Main gateway controller.
    
    Manages the lifecycle of all platform adapters and routes
    messages to/from the agent.
    """
    
    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or load_gateway_config()
        self.adapters: Dict[Platform, BasePlatformAdapter] = {}
        self.session_store = SessionStore(self.config.sessions_dir, self.config)
        self.delivery_router = DeliveryRouter(self.config)
        self._running = False
        self._shutdown_event = asyncio.Event()
    
    async def start(self) -> bool:
        """
        Start the gateway and all configured platform adapters.
        
        Returns True if at least one adapter connected successfully.
        """
        print("[gateway] Starting Hermes Gateway...")
        print(f"[gateway] Session storage: {self.config.sessions_dir}")
        
        connected_count = 0
        
        # Initialize and connect each configured platform
        for platform, platform_config in self.config.platforms.items():
            if not platform_config.enabled:
                continue
            
            adapter = self._create_adapter(platform, platform_config)
            if not adapter:
                print(f"[gateway] No adapter available for {platform.value}")
                continue
            
            # Set up message handler
            adapter.set_message_handler(self._handle_message)
            
            # Try to connect
            print(f"[gateway] Connecting to {platform.value}...")
            try:
                success = await adapter.connect()
                if success:
                    self.adapters[platform] = adapter
                    connected_count += 1
                    print(f"[gateway] ✓ {platform.value} connected")
                else:
                    print(f"[gateway] ✗ {platform.value} failed to connect")
            except Exception as e:
                print(f"[gateway] ✗ {platform.value} error: {e}")
        
        if connected_count == 0:
            print("[gateway] No platforms connected. Check your configuration.")
            return False
        
        # Update delivery router with adapters
        self.delivery_router.adapters = self.adapters
        
        self._running = True
        print(f"[gateway] Gateway running with {connected_count} platform(s)")
        print("[gateway] Press Ctrl+C to stop")
        
        return True
    
    async def stop(self) -> None:
        """Stop the gateway and disconnect all adapters."""
        print("[gateway] Stopping gateway...")
        self._running = False
        
        for platform, adapter in self.adapters.items():
            try:
                await adapter.disconnect()
                print(f"[gateway] ✓ {platform.value} disconnected")
            except Exception as e:
                print(f"[gateway] ✗ {platform.value} disconnect error: {e}")
        
        self.adapters.clear()
        self._shutdown_event.set()
        print("[gateway] Gateway stopped")
    
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
                print(f"[gateway] Telegram: python-telegram-bot not installed")
                return None
            return TelegramAdapter(config)
        
        elif platform == Platform.DISCORD:
            from gateway.platforms.discord import DiscordAdapter, check_discord_requirements
            if not check_discord_requirements():
                print(f"[gateway] Discord: discord.py not installed")
                return None
            return DiscordAdapter(config)
        
        elif platform == Platform.WHATSAPP:
            from gateway.platforms.whatsapp import WhatsAppAdapter, check_whatsapp_requirements
            if not check_whatsapp_requirements():
                print(f"[gateway] WhatsApp: Node.js not installed or bridge not configured")
                return None
            return WhatsAppAdapter(config)
        
        return None
    
    def _is_user_authorized(self, source: SessionSource) -> bool:
        """
        Check if a user is authorized to use the bot.
        
        Authorization is checked via environment variables:
        - GATEWAY_ALLOWED_USERS: Comma-separated list of user IDs (all platforms)
        - TELEGRAM_ALLOWED_USERS: Telegram-specific user IDs
        - DISCORD_ALLOWED_USERS: Discord-specific user IDs
        
        If no allowlist is configured, all users are allowed (open access).
        """
        user_id = source.user_id
        if not user_id:
            return False  # Can't verify unknown users
        
        # Check platform-specific allowlist first
        platform_env_map = {
            Platform.TELEGRAM: "TELEGRAM_ALLOWED_USERS",
            Platform.DISCORD: "DISCORD_ALLOWED_USERS",
            Platform.WHATSAPP: "WHATSAPP_ALLOWED_USERS",
        }
        
        platform_allowlist = os.getenv(platform_env_map.get(source.platform, ""))
        global_allowlist = os.getenv("GATEWAY_ALLOWED_USERS", "")
        
        # If no allowlists configured, allow all (backward compatible)
        if not platform_allowlist and not global_allowlist:
            return True
        
        # Check if user is in any allowlist
        allowed_ids = set()
        if platform_allowlist:
            allowed_ids.update(uid.strip() for uid in platform_allowlist.split(","))
        if global_allowlist:
            allowed_ids.update(uid.strip() for uid in global_allowlist.split(","))
        
        return user_id in allowed_ids
    
    async def _handle_message(self, event: MessageEvent) -> Optional[str]:
        """
        Handle an incoming message from any platform.
        
        This is the core message processing pipeline:
        1. Check user authorization
        2. Check for commands (/new, /reset, etc.)
        3. Get or create session
        4. Build context for agent
        5. Run agent conversation
        6. Return response
        """
        source = event.source
        
        # Check if user is authorized
        if not self._is_user_authorized(source):
            print(f"[gateway] Unauthorized user: {source.user_id} ({source.user_name}) on {source.platform.value}")
            return None  # Silently ignore unauthorized users
        
        # Check for reset commands
        command = event.get_command()
        if command in ["new", "reset"]:
            return await self._handle_reset_command(event)
        
        if command == "status":
            return await self._handle_status_command(event)
        
        # Get or create session
        session_entry = self.session_store.get_or_create_session(source)
        
        # Build session context
        context = build_session_context(source, self.config, session_entry)
        
        # Set environment variables for tools
        self._set_session_env(context)
        
        # Build the context prompt to inject
        context_prompt = build_session_context_prompt(context)
        
        # Load conversation history from transcript
        history = self.session_store.load_transcript(session_entry.session_id)
        
        try:
            # Run the agent
            response = await self._run_agent(
                message=event.text,
                context_prompt=context_prompt,
                history=history,
                source=source,
                session_id=session_entry.session_id
            )
            
            # Append to transcript
            self.session_store.append_to_transcript(
                session_entry.session_id,
                {"role": "user", "content": event.text, "timestamp": datetime.now().isoformat()}
            )
            self.session_store.append_to_transcript(
                session_entry.session_id,
                {"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()}
            )
            
            # Update session
            self.session_store.update_session(session_entry.session_key)
            
            return response
            
        except Exception as e:
            print(f"[gateway] Agent error: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
        finally:
            # Clear session env
            self._clear_session_env()
    
    async def _handle_reset_command(self, event: MessageEvent) -> str:
        """Handle /new or /reset command."""
        source = event.source
        
        # Get existing session key
        session_key = f"agent:main:{source.platform.value}:" + \
                      (f"dm" if source.chat_type == "dm" else f"{source.chat_type}:{source.chat_id}")
        
        # Reset the session
        new_entry = self.session_store.reset_session(session_key)
        
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
        
        lines = [
            "📊 **Hermes Gateway Status**",
            "",
            f"**Session ID:** `{session_entry.session_id[:12]}...`",
            f"**Created:** {session_entry.created_at.strftime('%Y-%m-%d %H:%M')}",
            f"**Last Activity:** {session_entry.updated_at.strftime('%Y-%m-%d %H:%M')}",
            f"**Tokens:** {session_entry.total_tokens:,}",
            "",
            f"**Connected Platforms:** {', '.join(connected_platforms)}",
        ]
        
        return "\n".join(lines)
    
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
    
    async def _run_agent(
        self,
        message: str,
        context_prompt: str,
        history: List[Dict[str, Any]],
        source: SessionSource,
        session_id: str
    ) -> str:
        """
        Run the agent with the given message and context.
        
        This is run in a thread pool to not block the event loop.
        """
        from run_agent import AIAgent
        
        # Determine toolset based on platform
        toolset_map = {
            Platform.LOCAL: "hermes-cli",
            Platform.TELEGRAM: "hermes-telegram",
            Platform.DISCORD: "hermes-discord",
            Platform.WHATSAPP: "hermes-whatsapp",
        }
        toolset = toolset_map.get(source.platform, "hermes-telegram")
        
        def run_sync():
            agent = AIAgent(
                model=os.getenv("HERMES_MODEL", "anthropic/claude-sonnet-4"),
                quiet_mode=True,
                enabled_toolsets=[toolset],
                ephemeral_system_prompt=context_prompt,
                session_id=session_id,
            )
            
            # If we have history, we need to restore it
            # For now, we pass the message directly
            # TODO: Implement proper history restoration
            
            result = agent.run_conversation(message)
            return result.get("final_response", "(No response)")
        
        # Run in thread pool to not block
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, run_sync)
        
        return response


async def start_gateway(config: Optional[GatewayConfig] = None) -> None:
    """
    Start the gateway and run until interrupted.
    
    This is the main entry point for running the gateway.
    """
    runner = GatewayRunner(config)
    
    # Set up signal handlers
    def signal_handler():
        asyncio.create_task(runner.stop())
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass
    
    # Start the gateway
    success = await runner.start()
    if not success:
        return
    
    # Wait for shutdown
    await runner.wait_for_shutdown()


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
    
    # Run the gateway
    asyncio.run(start_gateway(config))


if __name__ == "__main__":
    main()
