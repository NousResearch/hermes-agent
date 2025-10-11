"""
WebSocket Connection Pool - Persistent Connection Manager

This module provides a singleton WebSocket connection that persists across
multiple agent runs. This is a more robust architecture than creating a new
connection for each run.

Benefits:
- No timeout issues (connection stays alive indefinitely)
- No reconnection overhead (connect once)
- Supports parallel agent runs (multiple sessions share one socket)
- Proper shutdown handling (SIGTERM/SIGINT)
- Thread-safe concurrent sends
"""

import asyncio
import signal
import websockets
from typing import Optional, Dict, Any
import json
import atexit
import sys
import threading
from datetime import datetime


class WebSocketConnectionPool:
    """
    Singleton WebSocket connection manager.
    
    Maintains a single persistent connection to the logging server
    that all agent sessions can use. Handles graceful shutdown.
    
    Usage:
        # Get singleton instance
        pool = WebSocketConnectionPool()
        
        # Connect (idempotent - safe to call multiple times)
        await pool.connect()
        
        # Send events (thread-safe, multiple sessions can call concurrently)
        await pool.send_event("query", session_id, {...})
        
        # Shutdown handled automatically on SIGTERM/SIGINT
    """
    
    _instance: Optional['WebSocketConnectionPool'] = None
    
    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the connection pool (only once)."""
        if getattr(self, '_initialized', False):
            return
            
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.server_url: str = "ws://localhost:8000/ws"
        self.connected: bool = False
        # Store reference to loop for signal handlers
        # Agent code should never close event loops when using persistent connections
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        # Locks are created lazily when event loop exists
        self._send_lock: Optional[asyncio.Lock] = None
        self._connect_lock: Optional[asyncio.Lock] = None
        self._locks_loop: Optional[asyncio.AbstractEventLoop] = None  # Track which loop created locks
        self._init_lock = threading.Lock()  # Thread-safe lock initialization
        self._shutdown_in_progress = False
        self._initialized = True
        
        # Register shutdown handlers for graceful cleanup
        # These ensure WebSocket is closed properly on exit
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        atexit.register(self._cleanup_sync)
        
        print("🔌 WebSocket connection pool initialized")
    
    def _ensure_locks(self):
        """
        Lazy initialization of asyncio locks with thread safety and loop tracking.
        
        Locks must be created when an event loop exists, not at import time.
        If the event loop changes between runs, locks must be recreated because
        asyncio.Lock objects are bound to the loop that created them.
        
        This is called before any async operation that needs locks.
        Uses a threading.Lock to prevent race conditions during initialization.
        """
        with self._init_lock:  # Thread-safe initialization
            try:
                current_loop = asyncio.get_event_loop()
            except RuntimeError:
                # No event loop in current thread
                return
            
            # Recreate locks if:
            # 1. Locks don't exist yet, OR
            # 2. Event loop has changed (locks are bound to the loop that created them)
            if self._locks_loop is not current_loop or self._send_lock is None:
                self._send_lock = asyncio.Lock()
                self._connect_lock = asyncio.Lock()
                self._locks_loop = current_loop
    
    async def connect(self, server_url: str = "ws://localhost:8000/ws") -> bool:
        """
        Connect to WebSocket server.
        
        This is idempotent - safe to call multiple times. If already connected,
        does nothing. If connection failed previously, will retry.
        
        Args:
            server_url: WebSocket server URL (default: ws://localhost:8000/ws)
            
        Returns:
            bool: True if connected successfully, False otherwise
        """
        # Ensure locks exist (lazy initialization)
        self._ensure_locks()
        
        async with self._connect_lock:
            # Always update loop reference to current loop (even if already connected)
            # This ensures signal handlers and cleanup use the correct loop
            self.loop = asyncio.get_event_loop()
            
            # Already connected - nothing to do
            if self.connected and self.websocket:
                return True
            
            try:
                self.server_url = server_url
                
                # Establish persistent WebSocket connection
                # No ping/pong needed since connection stays open indefinitely
                self.websocket = await websockets.connect(
                    server_url,
                    ping_interval=None,  # Disable ping/pong (not needed for persistent connection)
                    max_size=10 * 1024 * 1024,  # 10MB max message size for large tool results
                    open_timeout=10,  # 10s timeout for initial connection
                    close_timeout=5   # 5s timeout for close handshake
                )
                
                self.connected = True
                
                print(f"✅ Connected to logging server (persistent): {server_url}")
                return True
                
            except Exception as e:
                print(f"⚠️ Failed to connect to logging server: {e}")
                self.connected = False
                self.websocket = None
                return False
    
    async def send_event(
        self,
        event_type: str,
        session_id: str,
        data: Dict[str, Any],
        retry: bool = True
    ) -> bool:
        """
        Send event to logging server (thread-safe).
        
        Multiple agent runs can call this concurrently. The send lock ensures
        only one message is sent at a time (WebSocket protocol requirement).
        
        Args:
            event_type: Type of event (query, api_call, response, tool_call, tool_result, error, complete)
            session_id: Unique session identifier
            data: Event-specific data dictionary
            retry: Whether to retry connection if disconnected (default: True)
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        # Try to connect if not connected (or reconnect if disconnected)
        if not self.connected or not self.websocket:
            if retry:
                await self.connect()
            if not self.connected:
                return False  # Give up if connection fails
        
        # Ensure locks exist (lazy initialization)
        self._ensure_locks()
        
        # Lock to prevent concurrent sends (WebSocket requires sequential sends)
        async with self._send_lock:
            try:
                # Create standardized message format
                message = {
                    "session_id": session_id,
                    "event_type": event_type,
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Send message as JSON
                await self.websocket.send(json.dumps(message))
                
                # Wait for server acknowledgment (with timeout)
                # This confirms the server received and processed the event
                try:
                    response = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=2.0  # Increased to 2s for busy servers
                    )
                    # Successfully received acknowledgment
                    return True
                    
                except asyncio.TimeoutError:
                    # No response within timeout - that's OK, message likely sent
                    # Server might be busy processing
                    return True
                    
            except websockets.exceptions.ConnectionClosed:
                print(f"⚠️ WebSocket connection closed unexpectedly")
                self.connected = False
                
                # Try to reconnect and resend (one retry)
                if retry:
                    print("🔄 Attempting to reconnect...")
                    if await self.connect():
                        # Recursively call with retry=False to avoid infinite loop
                        return await self.send_event(event_type, session_id, data, retry=False)
                
                return False
                
            except Exception as e:
                print(f"⚠️ Error sending event: {e}")
                self.connected = False
                return False
    
    async def disconnect(self):
        """
        Gracefully close the WebSocket connection.
        
        Called on shutdown (SIGTERM/SIGINT/exit). Ensures proper cleanup.
        """
        if self._shutdown_in_progress:
            return  # Already shutting down
        
        self._shutdown_in_progress = True
        
        if self.websocket and self.connected:
            try:
                await self.websocket.close()
                self.connected = False
                print("✅ WebSocket connection pool closed gracefully")
            except Exception as e:
                print(f"⚠️ Error closing WebSocket: {e}")
        
        self._shutdown_in_progress = False
    
    def _signal_handler(self, signum, frame):
        """
        Handle SIGTERM/SIGINT signals for graceful shutdown.
        
        When user presses Ctrl+C or system sends SIGTERM, this ensures
        the WebSocket is closed properly before exit.
        """
        print(f"\n🛑 Received signal {signum}, closing WebSocket connection pool...")
        
        # Check if we have a valid loop and are connected
        if self.loop and not self.loop.is_closed() and self.connected and not self._shutdown_in_progress:
            try:
                # If loop is not running, we can wait for disconnect
                if not self.loop.is_running():
                    self.loop.run_until_complete(self.disconnect())
                else:
                    # Loop is running, can't wait for task - just mark disconnected
                    # The disconnect task would be cancelled when we exit anyway
                    self.connected = False
                    print("⚠️ Loop is running, marking disconnected without waiting")
            except Exception as e:
                print(f"⚠️ Error during signal handler cleanup: {e}")
        
        # Exit gracefully
        sys.exit(0)
    
    def _cleanup_sync(self):
        """
        Cleanup at exit (atexit handler).
        
        This is a fallback in case signal handlers don't fire.
        Called when Python interpreter shuts down normally.
        """
        if self.loop and not self.loop.is_closed() and self.connected and not self._shutdown_in_progress:
            try:
                # Try to run disconnect synchronously
                self.loop.run_until_complete(self.disconnect())
            except Exception:
                # Ignore errors during exit cleanup
                pass
    
    def is_connected(self) -> bool:
        """Check if currently connected to server."""
        return self.connected and self.websocket is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics for debugging."""
        return {
            "connected": self.connected,
            "server_url": self.server_url,
            "shutdown_in_progress": self._shutdown_in_progress,
            "has_websocket": self.websocket is not None,
            "has_loop": self.loop is not None
        }


# Global singleton instance
# Import this in other modules: from websocket_connection_pool import ws_pool
ws_pool = WebSocketConnectionPool()


# Convenience functions for direct usage
async def connect(server_url: str = "ws://localhost:8000/ws") -> bool:
    """Connect to logging server (convenience function)."""
    return await ws_pool.connect(server_url)


async def send_event(event_type: str, session_id: str, data: Dict[str, Any]) -> bool:
    """Send event to logging server (convenience function)."""
    return await ws_pool.send_event(event_type, session_id, data)


async def disconnect():
    """Disconnect from logging server (convenience function)."""
    await ws_pool.disconnect()


def is_connected() -> bool:
    """Check if connected to logging server (convenience function)."""
    return ws_pool.is_connected()


# ============================================================================
# SYNCHRONOUS API FOR AGENT LAYER
# ============================================================================
# These functions provide a clean abstraction that hides event loop management
# from the agent layer. Agent code should ONLY use these functions.

def connect_sync(server_url: str = "ws://localhost:8000/ws") -> bool:
    """
    Synchronous connect - handles event loop internally.
    
    Creates a persistent event loop in a background thread if needed.
    This is thread-safe and can be called from any thread (including agent background threads).
    """
    import threading
    
    # If pool doesn't have a loop yet or it's closed, we need to start one
    if not ws_pool.loop or ws_pool.loop.is_closed():
        # Start connection in a background thread with its own loop
        result_container = {"success": False, "error": None, "connected": False}
        
        def run_in_thread():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                ws_pool.loop = loop  # Store the loop in the pool
                
                # Connect to WebSocket
                result_container["success"] = loop.run_until_complete(ws_pool.connect(server_url))
                result_container["connected"] = True
                
                # Keep loop running forever for future send_event calls
                # This is critical - the loop must stay alive for run_coroutine_threadsafe to work
                loop.run_forever()
                
            except Exception as e:
                result_container["error"] = str(e)
                print(f"❌ Error in WebSocket connection thread: {e}")
            finally:
                # Clean up if loop stops
                if loop.is_running():
                    loop.close()
        
        thread = threading.Thread(target=run_in_thread, daemon=True, name="WebSocket-EventLoop")
        thread.start()
        
        # Wait for connection to complete (but not for loop to exit - it runs forever)
        import time
        timeout = 10.0
        start = time.time()
        while not result_container["connected"] and (time.time() - start) < timeout:
            time.sleep(0.1)
        
        if result_container["error"]:
            print(f"⚠️  Connection failed: {result_container['error']}")
        
        return result_container["success"]
    else:
        # Pool already has a loop, use run_coroutine_threadsafe
        try:
            future = asyncio.run_coroutine_threadsafe(
                ws_pool.connect(server_url),
                ws_pool.loop
            )
            return future.result(timeout=10.0)
        except Exception as e:
            print(f"⚠️  Connection failed: {e}")
            return False


def send_event_sync(event_type: str, session_id: str, data: Dict[str, Any]) -> bool:
    """
    Synchronous send event - handles event loop internally.
    
    Uses the WebSocket pool's own event loop to avoid loop conflicts.
    This is critical when called from background threads (like agent execution).
    This is thread-safe and works correctly even when agent runs in a different thread.
    """
    if not ws_pool.loop or not ws_pool.loop.is_running():
        # No event loop running - can't send
        print("⚠️  WebSocket pool has no running event loop")
        return False
    
    try:
        # Use run_coroutine_threadsafe to submit to the WebSocket pool's loop
        # This works across threads - submits the coroutine to the correct loop
        future = asyncio.run_coroutine_threadsafe(
            ws_pool.send_event(event_type, session_id, data),
            ws_pool.loop  # ← Use the pool's loop, not current thread's loop
        )
        
        # Wait for completion (with timeout to avoid hanging)
        return future.result(timeout=5.0)
        
    except TimeoutError:
        print(f"⚠️  Timeout sending event {event_type}")
        return False
    except Exception as e:
        print(f"⚠️  Error sending event: {e}")
        return False


def disconnect_sync():
    """
    Synchronous disconnect - handles event loop internally.
    
    Thread-safe disconnect that works from any thread.
    """
    if ws_pool.loop and ws_pool.loop.is_running():
        try:
            future = asyncio.run_coroutine_threadsafe(
                ws_pool.disconnect(),
                ws_pool.loop
            )
            return future.result(timeout=5.0)
        except Exception as e:
            print(f"⚠️  Error disconnecting: {e}")
            return False
    return True
