#!/usr/bin/env python3
"""
Hermes Agent - Real-time Logging Server

A FastAPI server with WebSocket support that listens for agent execution events
and logs them to JSON files in real-time.

Events tracked:
- User queries
- API calls (requests to the model)
- Assistant responses
- Tool calls (name, parameters, timing)
- Tool results (outputs, errors, duration)
- Final responses
- Session metadata

Usage:
    python logging_server.py
    
Or with uvicorn directly:
    uvicorn logging_server:app --host 0.0.0.0 --port 8000 --reload
    
The server will listen for WebSocket connections at ws://localhost:8000/ws
"""

import json
import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

load_dotenv()




# Configuration
LOGS_DIR = Path(__file__).parent / "logs" / "realtime"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Hermes Agent Logging Server",
    description="Real-time WebSocket server for agent execution logging",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionLogger:
    """
    Manages logging for a single agent session.
    
    Each agent execution gets its own SessionLogger instance.
    Responsible for:
    - Collecting all events for the session
    - Saving events to JSON file in real-time
    - Managing session lifecycle (start -> events -> finalize)
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.now()
        self.events: List[Dict[str, Any]] = []  # In-memory list of all events
        self.log_file = LOGS_DIR / f"session_{session_id}.json"  # Where to save on disk
        
        # Initialize session data structure
        # This is what gets saved to the JSON file
        self.session_data = {
            "session_id": session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": None,  # Set when session completes
            "events": [],      # Will be populated as events come in
            "metadata": {}     # Model, toolsets, etc. (set via session_start event)
        }
    
    def add_event(self, event: Dict[str, Any]):
        """
        Add an event to the session log.
        
        Called every time a new event arrives (query, api_call, tool_call, etc).
        IMMEDIATELY saves to file for real-time persistence.
        """
        # Add timestamp if not present (should always be added, but safety check)
        if "timestamp" not in event:
            event["timestamp"] = datetime.now().isoformat()
        
        # Add to in-memory event list
        self.events.append(event)
        self.session_data["events"] = self.events
        
        # CRITICAL: Save to file immediately (real-time logging)
        # This ensures events are persisted even if agent crashes
        self._save()
    
    def set_metadata(self, metadata: Dict[str, Any]):
        """Set session metadata (model, toolsets, etc.)."""
        self.session_data["metadata"].update(metadata)
        self._save()
    
    def finalize(self):
        """Finalize the session and save."""
        self.session_data["end_time"] = datetime.now().isoformat()
        self._save()
    
    def _save(self):
        """
        Save current session data to JSON file.
        
        Called after EVERY event is added - provides real-time persistence.
        If file writing fails, logs error but continues (doesn't crash server).
        """
        try:
            # Write complete session data to JSON file
            # indent=2 makes it human-readable
            # ensure_ascii=False preserves Unicode characters
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving session log: {e}")


class ConnectionManager:
    """
    Manages WebSocket connections and active sessions.
    
    Global singleton that:
    - Tracks all active WebSocket connections (for broadcasting)
    - Manages all SessionLogger instances (one per agent session)
    - Coordinates between WebSocket events and file logging
    """
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []      # All connected WebSocket clients
        self.sessions: Dict[str, SessionLogger] = {}       # session_id -> SessionLogger mapping
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✅ WebSocket connected. Active connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"❌ WebSocket disconnected. Active connections: {len(self.active_connections)}")
    
    def get_or_create_session(self, session_id: str) -> SessionLogger:
        """
        Get existing session logger or create a new one.
        
        Called when an event arrives for a session. Creates SessionLogger
        on first event, reuses it for subsequent events from same session.
        """
        if session_id not in self.sessions:
            # First time seeing this session - create new logger
            self.sessions[session_id] = SessionLogger(session_id)
            print(f"📝 Created new session: {session_id}")
        return self.sessions[session_id]
    
    def finalize_session(self, session_id: str):
        """Finalize and clean up a session."""
        if session_id in self.sessions:
            self.sessions[session_id].finalize()
            print(f"✅ Session finalized: {session_id}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcast a message to all connected WebSocket clients.
        
        Allows multiple clients (e.g., multiple browser tabs) to watch
        the same agent session in real-time. Future UI feature.
        """
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                # Connection closed - mark for removal
                disconnected.append(connection)
        
        # Clean up disconnected clients silently
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)


# Global connection manager
manager = ConnectionManager()


# Request/Response models for API endpoints
class AgentRequest(BaseModel):
    """Request model for starting an agent run."""
    query: str
    model: str = "claude-sonnet-4-5-20250929"
    base_url: str = "https://api.anthropic.com/v1/"
    enabled_toolsets: Optional[List[str]] = None
    disabled_toolsets: Optional[List[str]] = None
    max_turns: int = 10
    mock_web_tools: bool = False
    mock_delay: int = 60
    verbose: bool = False


class AgentResponse(BaseModel):
    """Response model for agent run request."""
    status: str
    session_id: str
    message: str


@app.get("/")
async def root():
    """Root endpoint - server status."""
    return {
        "status": "running",
        "service": "Hermes Agent Logging Server",
        "websocket_url": "ws://localhost:8000/ws",
        "active_connections": len(manager.active_connections),
        "active_sessions": len(manager.sessions),
        "logs_directory": str(LOGS_DIR)
    }


@app.get("/sessions")
async def list_sessions():
    """List all active and recent sessions."""
    # Get all session log files
    session_files = list(LOGS_DIR.glob("session_*.json"))
    
    sessions = []
    for session_file in sorted(session_files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                sessions.append({
                    "session_id": session_data.get("session_id"),
                    "start_time": session_data.get("start_time"),
                    "end_time": session_data.get("end_time"),
                    "event_count": len(session_data.get("events", [])),
                    "file": str(session_file)
                })
        except Exception as e:
            print(f"⚠️ Error reading session file {session_file}: {e}")
    
    return {
        "total_sessions": len(sessions),
        "sessions": sessions
    }


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get detailed data for a specific session."""
    session_file = LOGS_DIR / f"session_{session_id}.json"
    
    if not session_file.exists():
        return {"error": "Session not found"}, 404
    
    try:
        with open(session_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Failed to load session: {str(e)}"}, 500


@app.post("/agent/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest, background_tasks: BackgroundTasks):
    """
    Start an agent run with specified parameters.
    
    This endpoint triggers an agent execution in the background and returns immediately.
    The agent will connect to the WebSocket endpoint to send real-time events.
    
    Args:
        request: AgentRequest with query and configuration
        background_tasks: FastAPI background tasks for async execution
        
    Returns:
        AgentResponse with session_id for tracking
    """
    import uuid
    import sys
    import os
    
    # Generate session ID for this run - we'll pass it to the agent
    session_id = str(uuid.uuid4())
    
    # Add parent directory to path to import run_agent
    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from run_agent import AIAgent
    
    # Run agent in background thread (not blocking the API)
    def run_agent_background():
        """Run agent in a separate thread."""
        try:
            # Initialize agent with WebSocket logging enabled
            agent = AIAgent(
                base_url=request.base_url,
                model=request.model,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                max_iterations=request.max_turns,
                enabled_toolsets=request.enabled_toolsets,
                disabled_toolsets=request.disabled_toolsets,
                save_trajectories=False,
                verbose_logging=request.verbose,
                enable_websocket_logging=True,  # Always enable for UI
                websocket_server="ws://localhost:8000/ws",
                mock_web_tools=request.mock_web_tools,
                mock_delay=request.mock_delay
            )
            
            # Run conversation with our session_id
            result = agent.run_conversation(
                request.query,
                session_id=session_id  # Pass session_id so it matches
            )
            
            print(f"✅ Agent run completed: {session_id[:8]}...")
            print(f"   Final response: {result['final_response'][:100] if result.get('final_response') else 'No response'}...")
            
        except Exception as e:
            print(f"❌ Error running agent {session_id[:8]}...: {e}")
            import traceback
            traceback.print_exc()
    
    # Start agent in background thread
    thread = threading.Thread(target=run_agent_background, daemon=True)
    thread.start()
    
    return AgentResponse(
        status="started",
        session_id=session_id,
        message=f"Agent started with session ID: {session_id}"
    )


@app.get("/tools")
async def get_available_tools():
    """Get list of available toolsets and tools."""
    try:
        import sys
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from toolsets import get_all_toolsets, get_toolset_info
        
        all_toolsets = get_all_toolsets()
        toolsets_info = []
        
        for name in all_toolsets.keys():
            info = get_toolset_info(name)
            if info:
                toolsets_info.append({
                    "name": name,
                    "description": info['description'],
                    "tool_count": info['tool_count'],
                    "resolved_tools": info['resolved_tools']
                })
        
        return {
            "toolsets": toolsets_info
        }
    except Exception as e:
        return {"error": f"Failed to load tools: {str(e)}"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for receiving real-time agent events.
    
    This is the main entry point for all logging. Agents connect here and send events.
    
    Message Flow:
    1. Agent connects to ws://localhost:8000/ws
    2. Agent sends events as JSON messages
    3. Server parses event_type and routes to appropriate handler
    4. Event is added to SessionLogger (saved to file)
    5. Event is broadcast to all connected clients
    6. Acknowledgment sent back to agent
    
    Expected message format:
    {
        "session_id": "unique-session-id",        // Links event to specific session
        "event_type": "query" | "api_call" | ..., // What kind of event
        "data": { ... event-specific data ... }   // Event payload
    }
    """
    # Accept the WebSocket connection
    await manager.connect(websocket)
    
    try:
        # Main event loop - runs until client disconnects
        while True:
            # Receive message from client (agent)
            # This is a blocking call - waits for next message
            message = await websocket.receive_json()
            
            # Parse the standard message structure
            session_id = message.get("session_id")  # Which agent session
            event_type = message.get("event_type")   # What kind of event
            data = message.get("data", {})           # Event payload
            
            # Validate: session_id is required
            if not session_id:
                await websocket.send_json({
                    "error": "session_id is required"
                })
                continue
            
            # Get or create SessionLogger for this session
            # First event creates it, subsequent events reuse it
            session = manager.get_or_create_session(session_id)
            
            # Route event to appropriate handler based on event_type
            # Each handler extracts relevant data and adds to session log
            
            if event_type == "session_start":
                # Initial event - sent when agent first connects
                # Contains metadata about the session (model, toolsets, etc.)
                session.set_metadata(data)
                print(f"🚀 Session started: {session_id}")
                
            elif event_type == "query":
                # User query
                session.add_event({
                    "type": "query",
                    "query": data.get("query"),
                    "toolsets": data.get("toolsets"),
                    "model": data.get("model")
                })
                print(f"📝 Query logged: {data.get('query', '')[:60]}...")
                
            elif event_type == "api_call":
                # API call to model
                session.add_event({
                    "type": "api_call",
                    "call_number": data.get("call_number"),
                    "message_count": data.get("message_count"),
                    "has_tools": data.get("has_tools")
                })
                print(f"🔄 API call #{data.get('call_number')} logged")
                
            elif event_type == "response":
                # Assistant response
                session.add_event({
                    "type": "response",
                    "call_number": data.get("call_number"),
                    "content": data.get("content"),
                    "has_tool_calls": data.get("has_tool_calls"),
                    "tool_call_count": data.get("tool_call_count"),
                    "duration": data.get("duration")
                })
                print(f"🤖 Response logged: {data.get('content', '')[:60]}...")
                
            elif event_type == "tool_call":
                # Tool execution
                session.add_event({
                    "type": "tool_call",
                    "call_number": data.get("call_number"),
                    "tool_index": data.get("tool_index"),
                    "tool_name": data.get("tool_name"),
                    "parameters": data.get("parameters"),
                    "tool_call_id": data.get("tool_call_id")
                })
                print(f"🔧 Tool call logged: {data.get('tool_name')}")
                
            elif event_type == "tool_result":
                # Tool result - captures output from tool execution
                # Now includes BOTH truncated preview AND full raw result
                session.add_event({
                    "type": "tool_result",
                    "call_number": data.get("call_number"),
                    "tool_index": data.get("tool_index"),
                    "tool_name": data.get("tool_name"),
                    "result": data.get("result"),              # Truncated preview (1000 chars)
                    "raw_result": data.get("raw_result"),      # NEW: Full untruncated result
                    "error": data.get("error"),
                    "duration": data.get("duration"),
                    "tool_call_id": data.get("tool_call_id")
                })
                
                # Enhanced logging with size information
                if data.get("error"):
                    print(f"❌ Tool error logged: {data.get('tool_name')}")
                else:
                    # Show size of raw result to indicate data volume
                    raw_size = len(data.get("raw_result", "")) if data.get("raw_result") else len(data.get("result", ""))
                    size_kb = raw_size / 1024
                    print(f"✅ Tool result logged: {data.get('tool_name')} ({size_kb:.1f} KB)")
                
            elif event_type == "error":
                # Error event
                session.add_event({
                    "type": "error",
                    "error_message": data.get("error_message"),
                    "call_number": data.get("call_number")
                })
                print(f"❌ Error logged: {data.get('error_message', '')[:60]}...")
                
            elif event_type == "complete":
                # Session complete
                session.add_event({
                    "type": "complete",
                    "final_response": data.get("final_response"),
                    "total_calls": data.get("total_calls"),
                    "completed": data.get("completed")
                })
                manager.finalize_session(session_id)
                print(f"🎉 Session complete: {session_id}")
                
            else:
                # Unknown event type - log it anyway
                session.add_event({
                    "type": event_type or "unknown",
                    **data
                })
                print(f"⚠️ Unknown event type: {event_type}")
            
            # Broadcast event to all connected clients (for future real-time UI)
            # Allows multiple browsers/dashboards to watch same session live
            await manager.broadcast({
                "session_id": session_id,
                "event_type": event_type,
                "timestamp": datetime.now().isoformat(),
                "data": data
            })
            
            # Send acknowledgment back to sender
            # Confirms event was received and logged
            # Handle case where client disconnects before we can ack
            try:
                await websocket.send_json({
                    "status": "logged",
                    "session_id": session_id,
                    "event_type": event_type
                })
            except Exception:
                # Connection closed before ack - this is normal for "complete" event
                # Client disconnects after sending, so we can't ack
                pass
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        manager.disconnect(websocket)


def main(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Start the logging server.
    
    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to run on (default: 8000)
        reload: Enable auto-reload on file changes (default: False)
    """
    print("🚀 Hermes Agent Logging Server")
    print("=" * 50)
    print(f"📂 Logs directory: {LOGS_DIR}")
    print(f"🌐 Server starting at http://{host}:{port}")
    print(f"🔌 WebSocket endpoint: ws://{host}:{port}/ws")
    print(f"🔄 Auto-reload: {'enabled' if reload else 'disabled'}")
    print("\n📡 Ready to receive agent events...")
    print("=" * 50)
    
    uvicorn.run(
        "logging_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        timeout_keep_alive=600      # Keep HTTP/WS connections alive for 10 minutes of inactivity
        # Note: WebSocket ping/pong disabled in client to avoid timeout during blocked event loop
    )


if __name__ == "__main__":
    import fire
    fire.Fire(main)

