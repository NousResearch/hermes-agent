#!/usr/bin/env python3
"""
WebSocket Logger Client

Simple client for sending agent events to the logging server via WebSocket.
Used by the agent to log events in real-time during execution.
"""

import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import websockets


class WebSocketLogger:
    """
    Client for logging agent events via WebSocket.
    
    Usage:
        logger = WebSocketLogger("unique-session-id")
        await logger.connect()
        await logger.log_query("What is Python?", model="gpt-4")
        await logger.log_api_call(call_number=1)
        await logger.log_response(call_number=1, content="Python is...")
        await logger.disconnect()
    """
    
    def __init__(
        self, 
        session_id: str,
        server_url: str = "ws://localhost:8000/ws",
        enabled: bool = True
    ):
        """
        Initialize WebSocket logger.
        
        Args:
            session_id: Unique identifier for this agent session
            server_url: WebSocket server URL (default: ws://localhost:8000/ws)
            enabled: Whether logging is enabled (default: True)
        """
        self.session_id = session_id
        self.server_url = server_url
        self.enabled = enabled
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self.reconnect_count = 0  # Track reconnections for debugging
    
    async def connect(self):
        """
        Connect to the WebSocket logging server.
        
        Establishes WebSocket connection and sends initial session_start event.
        If connection fails, gracefully disables logging (agent continues normally).
        """
        if not self.enabled:
            return
        
        try:
            # Establish WebSocket connection to the server
            # Use VERY LONG ping intervals to avoid timeout during long tool execution
            # The event loop is blocked during tool execution, so we can't process pings
            # Setting to very large values (1 hour) effectively disables it
            self.websocket = await websockets.connect(
                self.server_url,
                ping_interval=3600,      # 1 hour - effectively disabled (event loop blocked anyway)
                ping_timeout=3600,       # 1 hour timeout for pong response
                close_timeout=10,        # Timeout for close handshake
                max_size=10 * 1024 * 1024,  # 10MB max message size (for large raw_results)
                open_timeout=10          # Timeout for initial connection
            )
            self.connected = True
            print(f"✅ Connected to logging server (ping/pong: 3600s intervals): {self.server_url}")
            
            # Send initial session_start event
            # This tells the server to create a new SessionLogger for this session
            await self._send_event("session_start", {
                "session_id": self.session_id,
                "start_time": datetime.now().isoformat()
            })
            
        except Exception as e:
            # Connection failed - disable logging but don't crash the agent
            print(f"⚠️ Failed to connect to logging server: {e}")
            print(f"   Logging will be disabled for this session.")
            self.enabled = False
            self.connected = False
    
    async def disconnect(self):
        """Disconnect from the WebSocket server."""
        if self.websocket and self.connected:
            try:
                await self.websocket.close()
                self.connected = False
                print(f"✅ Disconnected from logging server")
            except Exception as e:
                print(f"⚠️ Error disconnecting: {e}")
    
    async def _send_event(self, event_type: str, data: Dict[str, Any]):
        """
        Send an event to the logging server.
        
        This is the core method that sends all events via WebSocket.
        Creates a standardized message format and handles acknowledgments.
        
        Args:
            event_type: Type of event (query, api_call, response, tool_call, tool_result, error, complete)
            data: Event data dictionary containing event-specific information
        """
        # Safety check: Don't send if logging is disabled
        if not self.enabled:
            return
        
        # Auto-reconnect if connection was lost
        if not self.connected or not self.websocket:
            try:
                self.reconnect_count += 1
                print(f"🔄 Reconnecting to logging server (attempt #{self.reconnect_count})...")
                await self.connect()
                print(f"✅ Reconnected successfully!")
            except Exception as e:
                print(f"⚠️ Failed to reconnect: {e}")
                self.enabled = False  # Disable logging after failed reconnect
                return
        
        try:
            # Create standardized message structure
            # All events follow this format for consistent server-side handling
            message = {
                "session_id": self.session_id,      # Links event to specific agent session
                "event_type": event_type,            # Identifies what kind of event this is
                "data": data                         # Event-specific payload
            }
            
            # Send message as JSON string over WebSocket
            await self.websocket.send(json.dumps(message))
            
            # Wait for server acknowledgment (with 1 second timeout)
            # This ensures the server received and processed the event
            try:
                response = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=1.0
                )
                # Server sends back: {"status": "logged", "session_id": "...", "event_type": "..."}
                # We don't need to process it, just confirms receipt
            except asyncio.TimeoutError:
                # No response within 1 second - that's okay, continue anyway
                # Server might be busy or network slow, but event was likely sent
                pass
                
        except Exception as e:
            # Log error but don't crash - graceful degradation
            # Agent should continue working even if logging fails
            error_str = str(e)
            
            # Check if connection was closed (error 1011 = keepalive ping timeout)
            if "1011" in error_str or "closed" in error_str.lower():
                print(f"⚠️ WebSocket connection closed: {error_str}")
                self.connected = False  # Mark as disconnected
                # Don't try to send more events - connection is dead
            else:
                print(f"⚠️ Error sending event to logging server: {e}")
            # Don't disable entirely or try to reconnect - just continue with logging disabled
    
    # Convenience methods for specific event types
    
    async def log_query(
        self, 
        query: str, 
        model: str = None,
        toolsets: list = None
    ):
        """
        Log a user query (the question/task given to the agent).
        
        This is typically the first event in a session after connection.
        Captures what the user asked and which model/tools will be used.
        """
        await self._send_event("query", {
            "query": query,          # The user's question/instruction
            "model": model,          # Which AI model is being used
            "toolsets": toolsets     # Which tool categories are enabled
        })
    
    async def log_api_call(
        self,
        call_number: int,
        message_count: int = None,
        has_tools: bool = None
    ):
        """
        Log an API call to the AI model.
        
        Called right before sending a request to the model (OpenAI/Anthropic/etc).
        Helps track how many API calls are being made and conversation length.
        """
        await self._send_event("api_call", {
            "call_number": call_number,      # Sequential number (1, 2, 3...)
            "message_count": message_count,  # How many messages in conversation so far
            "has_tools": has_tools          # Whether tools are available to the model
        })
    
    async def log_response(
        self,
        call_number: int,
        content: str = None,
        has_tool_calls: bool = False,
        tool_call_count: int = 0,
        duration: float = None
    ):
        """
        Log an assistant response from the AI model.
        
        Called after receiving a response from the API.
        Captures what the model said and whether it wants to use tools.
        """
        await self._send_event("response", {
            "call_number": call_number,          # Which API call this response is from
            "content": content,                   # What the model said (text response)
            "has_tool_calls": has_tool_calls,    # Did model request tool execution?
            "tool_call_count": tool_call_count,  # How many tools does it want to call?
            "duration": duration                  # How long the API call took (seconds)
        })
    
    async def log_tool_call(
        self,
        call_number: int,
        tool_index: int,
        tool_name: str,
        parameters: Dict[str, Any],
        tool_call_id: str = None
    ):
        """
        Log a tool call (before executing the tool).
        
        Captures which tool is being called and with what parameters.
        This happens BEFORE the tool runs, so no results yet.
        """
        await self._send_event("tool_call", {
            "call_number": call_number,      # Which API call requested this tool
            "tool_index": tool_index,        # Which tool in the sequence (if multiple)
            "tool_name": tool_name,          # Name of tool (e.g., "web_search", "web_extract")
            "parameters": parameters,        # Arguments passed to the tool (e.g., {"query": "Python", "limit": 5})
            "tool_call_id": tool_call_id    # Unique ID to link call with result
        })
    
    async def log_tool_result(
        self,
        call_number: int,
        tool_index: int,
        tool_name: str,
        result: str = None,
        error: str = None,
        duration: float = None,
        tool_call_id: str = None,
        raw_result: str = None  # NEW: Full untruncated result for verification
    ):
        """
        Log a tool result (output from tool execution).
        
        Captures both a truncated preview (for UI display) and the full raw result
        (for verification and debugging). This is especially important for web tools
        where you want to see what was scraped vs what the LLM processed.
        
        Args:
            call_number: Which API call this tool was part of
            tool_index: Which tool in the sequence (1st, 2nd, etc.)
            tool_name: Name of the tool that was executed
            result: Tool output (will be truncated to 1000 chars for preview)
            error: Error message if tool failed
            duration: How long the tool took to execute (seconds)
            tool_call_id: Unique ID linking this result to the tool call
            raw_result: NEW - Full untruncated result for verification/debugging
        """
        await self._send_event("tool_result", {
            "call_number": call_number,
            "tool_index": tool_index,
            "tool_name": tool_name,
            "result": result[:1000] if result else None,  # Truncated preview (1000 chars max)
            "raw_result": raw_result,  # NEW: Full result - can be 100KB+ for web scraping
            "error": error,
            "duration": duration,
            "tool_call_id": tool_call_id
        })
    
    async def log_error(
        self,
        error_message: str,
        call_number: int = None
    ):
        """
        Log an error that occurred during agent execution.
        
        Captures exceptions, API failures, or other issues.
        """
        await self._send_event("error", {
            "error_message": error_message,  # Description of what went wrong
            "call_number": call_number       # Which API call caused the error (if applicable)
        })
    
    async def log_complete(
        self,
        final_response: str = None,
        total_calls: int = None,
        completed: bool = True
    ):
        """
        Log session completion (final event before disconnecting).
        
        Marks the end of the agent's execution and provides summary info.
        """
        await self._send_event("complete", {
            "final_response": final_response[:500] if final_response else None,  # Truncated summary of final answer
            "total_calls": total_calls,      # How many API calls were made total
            "completed": completed           # Did it complete successfully? (true/false)
        })


# Synchronous wrapper for convenience
class SyncWebSocketLogger:
    """
    Synchronous wrapper around WebSocketLogger.
    
    For use in synchronous code - creates an event loop internally.
    """
    
    def __init__(self, session_id: str, server_url: str = "ws://localhost:8000/ws", enabled: bool = True):
        self.logger = WebSocketLogger(session_id, server_url, enabled)
        self.loop = None
    
    def connect(self):
        """Connect to server (synchronous)."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.logger.connect())
    
    def disconnect(self):
        """Disconnect from server (synchronous)."""
        if self.loop:
            self.loop.run_until_complete(self.logger.disconnect())
            self.loop.close()
    
    def _run_async(self, coro):
        """
        Run an async coroutine synchronously.
        
        Bridge between sync code (agent) and async code (WebSocket).
        Uses event loop to execute async operations in sync context.
        """
        if self.loop and self.loop.is_running():
            # Already in event loop, just await
            asyncio.create_task(coro)
        else:
            # Run in current loop
            if self.loop:
                self.loop.run_until_complete(coro)
    
    def log_query(self, query: str, model: str = None, toolsets: list = None):
        self._run_async(self.logger.log_query(query, model, toolsets))
    
    def log_api_call(self, call_number: int, message_count: int = None, has_tools: bool = None):
        self._run_async(self.logger.log_api_call(call_number, message_count, has_tools))
    
    def log_response(self, call_number: int, content: str = None, has_tool_calls: bool = False, 
                    tool_call_count: int = 0, duration: float = None):
        self._run_async(self.logger.log_response(call_number, content, has_tool_calls, 
                                                 tool_call_count, duration))
    
    def log_tool_call(self, call_number: int, tool_index: int, tool_name: str, 
                     parameters: Dict[str, Any], tool_call_id: str = None):
        self._run_async(self.logger.log_tool_call(call_number, tool_index, tool_name, 
                                                  parameters, tool_call_id))
    
    def log_tool_result(self, call_number: int, tool_index: int, tool_name: str,
                       result: str = None, error: str = None, duration: float = None,
                       tool_call_id: str = None, raw_result: str = None):
        self._run_async(self.logger.log_tool_result(call_number, tool_index, tool_name,
                                                    result, error, duration, tool_call_id, raw_result))
    
    def log_error(self, error_message: str, call_number: int = None):
        self._run_async(self.logger.log_error(error_message, call_number))
    
    def log_complete(self, final_response: str = None, total_calls: int = None, completed: bool = True):
        self._run_async(self.logger.log_complete(final_response, total_calls, completed))

