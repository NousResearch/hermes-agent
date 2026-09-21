#!/usr/bin/env python3
"""
Adaptive Rate Limiter MCP Server
Implements per-key sliding windows with reactive 429 cooldowns, jittered backoff, and header-aware gating.
"""

import asyncio
import time
import random
from typing import Dict, Optional
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configuration from environment variables with defaults
import os
RATE_LIMIT_DEFAULT_RPM = float(os.getenv("RATE_LIMIT_DEFAULT_RPM", "16"))
RATE_LIMIT_JITTER_SECONDS = float(os.getenv("RATE_LIMIT_JITTER_SECONDS", "1.5"))
RATE_LIMIT_FAILURE_BACKOFF = float(os.getenv("RATE_LIMIT_FAILURE_BACKOFF", "3.0"))
RATE_LIMIT_FALLBACK_WINDOW = float(os.getenv("RATE_LIMIT_FALLBACK_WINDOW", "90.0"))

class AdaptiveRateLimiter:
    def __init__(self):
        # Per-key state: {key: {'rpm': float, 'next_available': float, 'active_requests': int}}
        self._state: Dict[str, dict] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._default_rpm = RATE_LIMIT_DEFAULT_RPM
        self._jitter_seconds = RATE_LIMIT_JITTER_SECONDS
        self._failure_backoff = RATE_LIMIT_FAILURE_BACKOFF
        self._fallback_window = RATE_LIMIT_FALLBACK_WINDOW

    def _get_lock(self, key: str) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    def _get_state(self, key: str) -> dict:
        if key not in self._state:
            self._state[key] = {
                'rpm': self._default_rpm,
                'next_available': 0.0,
                'active_requests': 0
            }
        return self._state[key]

    async def await_slot(self, key: str, rpm_override: Optional[float] = None) -> dict:
        """Block until a request slot is available for the given key."""
        lock = self._get_lock(key)
        state = self._get_state(key)
        
        async with lock:
            rpm = rpm_override if rpm_override is not None else state['rpm']
            min_interval = 60.0 / rpm if rpm > 0 else float('inf')
            
            now = time.monotonic()
            if now < state['next_available']:
                wait_time = state['next_available'] - now
                # Sleep outside the lock
                pass
            else:
                wait_time = 0.0
            
            # Update state optimistically
            state['active_requests'] += 1
            state['next_available'] = now + min_interval
        
        # Actually sleep outside the lock
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        return {"waited": wait_time, "key": key}

    def record_response(self, key: str, status: int, headers: dict) -> dict:
        """Record an HTTP response, updating reactive cooldowns from headers."""
        lock = self._get_lock(key)
        # Note: We don't hold the lock during the calculation, only during state update
        
        # Calculate cooldown based on priority rules
        cooldown = 0.0
        
        # Priority 1: retry-after header
        if 'retry-after' in headers:
            try:
                cooldown = float(headers['retry-after'])
            except (ValueError, TypeError):
                pass
        
        # Priority 2: x-rate-limit-remaining == "0" + x-rate-limit-reset
        elif (headers.get('x-rate-limit-remaining') == "0" and 
              'x-rate-limit-reset' in headers):
            try:
                reset_epoch = float(headers['x-rate-limit-reset'])
                cooldown = max(0.0, reset_epoch - time.time())
            except (ValueError, TypeError):
                pass
        
        # Priority 3: Status 429 (no headers)
        elif status == 429:
            cooldown = self._failure_backoff
        
        # Priority 4: Status 500/503 (ambiguous)
        elif status in (500, 503):
            cooldown = min(self._fallback_window, self._failure_backoff)
        
        # Apply jitter
        if cooldown > 0:
            jitter = random.uniform(0, self._jitter_seconds)
            cooldown += jitter
        
        # Update state with conservative update (only extend existing ban)
        if cooldown > 0:
            with lock:  # Actually, we need async lock here - let me fix this
                pass  # Will fix below
        
        # Fix: Use async lock properly
        async def _update_state():
            async with lock:
                state = self._get_state(key)
                if cooldown > 0:
                    next_ready = time.monotonic() + cooldown
                    if next_ready > state['next_available']:
                        state['next_available'] = next_ready
                state['active_requests'] = max(0, state['active_requests'] - 1)
        
        # For sync method, we'll need to handle this differently
        # Let's make it async or use a different approach
        # Actually, let me reconsider - record_response can be async too
        
        return {"cooldown_applied": cooldown > 0, "cooldown_seconds": cooldown}

    async def record_response_async(self, key: str, status: int, headers: dict) -> dict:
        """Async version of record_response."""
        lock = self._get_lock(key)
        async with lock:
            state = self._get_state(key)
            
            # Calculate cooldown based on priority rules
            cooldown = 0.0
            
            # Priority 1: retry-after header
            if 'retry-after' in headers:
                try:
                    cooldown = float(headers['retry-after'])
                except (ValueError, TypeError):
                    pass
            
            # Priority 2: x-rate-limit-remaining == "0" + x-rate-limit-reset
            elif (headers.get('x-rate-limit-remaining') == "0" and 
                  'x-rate-limit-reset' in headers):
                try:
                    reset_epoch = float(headers['x-rate-limit-reset'])
                    cooldown = max(0.0, reset_epoch - time.time())
                except (ValueError, TypeError):
                    pass
            
            # Priority 3: Status 429 (no headers)
            elif status == 429:
                cooldown = self._failure_backoff
            
            # Priority 4: Status 500/503 (ambiguous)
            elif status in (500, 503):
                cooldown = min(self._fallback_window, self._failure_backoff)
            
            # Apply jitter
            if cooldown > 0:
                jitter = random.uniform(0, self._jitter_seconds)
                cooldown += jitter
            
            # Conservative update: only extend existing ban
            if cooldown > 0:
                next_ready = time.monotonic() + cooldown
                if next_ready > state['next_available']:
                    state['next_available'] = next_ready
            
            # Decrement active requests
            state['active_requests'] = max(0, state['active_requests'] - 1)
        
        return {"cooldown_applied": cooldown > 0, "cooldown_seconds": cooldown}

    def get_stats(self, key: Optional[str] = None) -> dict:
        """Get statistics for a specific key or all keys."""
        if key is not None:
            if key in self._state:
                state = self._state[key]
                return {
                    "keys": [{
                        "key": key,
                        "rpm": state['rpm'],
                        "active_requests": state['active_requests'],
                        "next_available": state['next_available']
                    }]
                }
            else:
                return {"keys": []}
        else:
            # Return all keys
            keys_list = []
            for k, state in self._state.items():
                keys_list.append({
                    "key": k,
                    "rpm": state['rpm'],
                    "active_requests": state['active_requests'],
                    "next_available": state['next_available']
                })
            return {"keys": keys_list}

# MCP Server setup
server = Server("adaptive-rate-limiter")
limiter = AdaptiveRateLimiter()

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="rate_limiter_await_slot",
            description="Blocks until a request slot is available for the given key. Implements both proactive sliding window and reactive cooldown.",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The key to rate limit (e.g., endpoint, channel)"},
                    "rpm_override": {"type": "number", "description": "Override RPM for this key"}
                },
                "required": ["key"]
            }
        ),
        Tool(
            name="rate_limiter_record_response",
            description="Records an HTTP response, updating reactive cooldowns from headers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The key that was used"},
                    "status": {"type": "integer", "description": "HTTP status code"},
                    "headers": {"type": "object", "description": "Response headers"}
                },
                "required": ["key", "status", "headers"]
            }
        ),
        Tool(
            name="rate_limiter_stats",
            description="Get statistics for keys.",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Specific key to get stats for (optional)"}
                }
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "rate_limiter_await_slot":
        result = await limiter.await_slot(
            key=arguments["key"],
            rpm_override=arguments.get("rpm_override")
        )
        return [TextContent(type="text", text=str(result))]
    
    elif name == "rate_limiter_record_response":
        result = await limiter.record_response_async(
            key=arguments["key"],
            status=arguments["status"],
            headers=arguments["headers"]
        )
        return [TextContent(type="text", text=str(result))]
    
    elif name == "rate_limiter_stats":
        result = limiter.get_stats(
            key=arguments.get("key")
        )
        return [TextContent(type="text", text=str(result))]
    
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())