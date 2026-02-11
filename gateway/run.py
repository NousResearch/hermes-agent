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
        
        # Track running agents per session for interrupt support
        # Key: session_key, Value: AIAgent instance
        self._running_agents: Dict[str, Any] = {}
        self._pending_messages: Dict[str, str] = {}  # Queued messages during interrupt
    
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
        3. Check for running agent and interrupt if needed
        4. Get or create session
        5. Build context for agent
        6. Run agent conversation
        7. Return response
        """
        source = event.source
        
        # Check if user is authorized
        if not self._is_user_authorized(source):
            print(f"[gateway] Unauthorized user: {source.user_id} ({source.user_name}) on {source.platform.value}")
            return None  # Silently ignore unauthorized users
        
        # Check for commands
        command = event.get_command()
        if command in ["new", "reset"]:
            return await self._handle_reset_command(event)
        
        if command == "status":
            return await self._handle_status_command(event)
        
        if command == "stop":
            return await self._handle_stop_command(event)
        
        # Get or create session
        session_entry = self.session_store.get_or_create_session(source)
        session_key = session_entry.session_key
        
        # Check if there's already a running agent for this session
        if session_key in self._running_agents:
            running_agent = self._running_agents[session_key]
            print(f"[gateway] ⚡ Interrupting running agent for session {session_key[:20]}...")
            running_agent.interrupt(event.text)
            # Store the new message to be processed after current agent finishes
            self._pending_messages[session_key] = event.text
            return None  # Don't respond yet - let the interrupt handle it
        
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
                session_id=session_entry.session_id,
                session_key=session_key
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
        session_id: str,
        session_key: str = None
    ) -> str:
        """
        Run the agent with the given message and context.
        
        This is run in a thread pool to not block the event loop.
        Supports interruption via new messages.
        """
        from run_agent import AIAgent
        import queue
        
        # Determine toolset based on platform
        toolset_map = {
            Platform.LOCAL: "hermes-cli",
            Platform.TELEGRAM: "hermes-telegram",
            Platform.DISCORD: "hermes-discord",
            Platform.WHATSAPP: "hermes-whatsapp",
        }
        toolset = toolset_map.get(source.platform, "hermes-telegram")
        
        # Check if tool progress notifications are enabled
        tool_progress_enabled = os.getenv("HERMES_TOOL_PROGRESS", "").lower() in ("1", "true", "yes")
        progress_mode = os.getenv("HERMES_TOOL_PROGRESS_MODE", "new")  # "all" or "new" (only new tools)
        
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
            
            # Build progress message
            tool_emojis = {
                "terminal": "💻",
                "web_search": "🔍",
                "web_extract": "📄",
                "read_file": "📖",
                "write_file": "✍️",
                "list_directory": "📂",
                "image_generate": "🎨",
                "browser_navigate": "🌐",
                "browser_click": "👆",
                "moa_query": "🧠",
            }
            emoji = tool_emojis.get(tool_name, "⚙️")
            
            if tool_name == "terminal" and preview:
                msg = f"{emoji} `{preview}`..."
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
                        except:
                            break
                    return
                except Exception as e:
                    print(f"[Gateway] Progress message error: {e}")
                    await asyncio.sleep(1)
        
        # We need to share the agent instance for interrupt support
        agent_holder = [None]  # Mutable container for the agent instance
        result_holder = [None]  # Mutable container for the result
        
        def run_sync():
            # Read from env var or use default (same as CLI)
            max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", "60"))
            
            agent = AIAgent(
                model=os.getenv("HERMES_MODEL", "anthropic/claude-opus-4.6"),
                max_iterations=max_iterations,
                quiet_mode=True,
                enabled_toolsets=[toolset],
                ephemeral_system_prompt=context_prompt,
                session_id=session_id,
                tool_progress_callback=progress_callback if tool_progress_enabled else None,
            )
            
            # Store agent reference for interrupt support
            agent_holder[0] = agent
            
            # Convert transcript history to agent format
            # Transcript has timestamps; agent expects {"role": ..., "content": ...}
            agent_history = []
            for msg in history:
                role = msg.get("role")
                content = msg.get("content")
                if role and content:
                    agent_history.append({"role": role, "content": content})
            
            result = agent.run_conversation(message, conversation_history=agent_history)
            result_holder[0] = result
            
            # Return final response, or a message if something went wrong
            final_response = result.get("final_response")
            if final_response:
                return final_response
            elif result.get("error"):
                # Agent couldn't recover - show the error
                return f"⚠️ {result['error']}"
            else:
                return "(No response generated)"
        
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
                        print(f"[gateway] ⚡ Interrupt detected from adapter, signaling agent...")
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
                print(f"[gateway] 📨 Processing interrupted message: '{pending[:40]}...'")
                # Add an indicator to the response
                if response:
                    response = response + "\n\n---\n_[Interrupted - processing your new message]_"
                
                # Send the interrupted response first
                if adapter and response:
                    await adapter.send(chat_id=source.chat_id, content=response)
                
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


async def start_gateway(config: Optional[GatewayConfig] = None) -> bool:
    """
    Start the gateway and run until interrupted.
    
    This is the main entry point for running the gateway.
    Returns True if the gateway ran successfully, False if it failed to start.
    A False return causes a non-zero exit code so systemd can auto-restart.
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
        return False
    
    # Wait for shutdown
    await runner.wait_for_shutdown()
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
