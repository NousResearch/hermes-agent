"""
WebSocket & Live Dashboard Routes

Endpoints:
- WS /ws/live - Real-time event WebSocket
- GET /live - Live dashboard page
- GET /api/sessions - list active sessions
- GET /api/sessions/{session_id}/graph - Session action graph
- GET /health - Health check
"""

import asyncio
import logging
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_shared():
    """Import shared state from server module lazily."""
    from Cosmos.web import server
    return server


# ============================================
# WEBSOCKET ENDPOINTS
# ============================================

@router.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """WebSocket endpoint for real-time events."""
    s = _get_shared()
    await s.ws_manager.connect(websocket)
    try:
        await websocket.send_json({
            "type": "connected",
            "message": "Good news, everyone! Connected to Cosmos Live Feed!",
            "timestamp": datetime.now().isoformat()
        })

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)

                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif data.get("type") == "get_history":
                    session_id = data.get("session_id", "default")
                    history = s.ws_manager.get_session_history(session_id)
                    await websocket.send_json({
                        "type": "history",
                        "session_id": session_id,
                        "events": history
                    })

            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        s.ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        s.ws_manager.disconnect(websocket)


@router.get("/live", response_class=HTMLResponse)
async def live_dashboard(request: Request):
    """Live dashboard showing real-time action graphs."""
    s = _get_shared()
    return s.templates.TemplateResponse("live.html", {"request": request})


@router.get("/api/sessions")
async def get_sessions():
    """Get list of active sessions."""
    s = _get_shared()
    sessions = []
    for session_id, events in s.ws_manager.session_events.items():
        sessions.append({
            "session_id": session_id,
            "event_count": len(events),
            "last_event": events[-1]["timestamp"] if events else None
        })
    return JSONResponse({
        "sessions": sessions,
        "active_connections": len(s.ws_manager.active_connections)
    })


@router.get("/api/sessions/{session_id}/graph")
async def get_session_graph(session_id: str):
    """Get action chain graph data for a session."""
    s = _get_shared()
    events = s.ws_manager.get_session_history(session_id)

    nodes = []
    edges = []
    node_id = 0

    for event in events:
        event_type = event.get("type", "unknown")

        node = {
            "id": node_id,
            "type": event_type,
            "label": event_type.replace("_", " ").title(),
            "timestamp": event.get("timestamp"),
            "data": event.get("data", {})
        }
        nodes.append(node)

        if node_id > 0:
            edges.append({
                "from": node_id - 1,
                "to": node_id
            })

        node_id += 1

    return JSONResponse({
        "session_id": session_id,
        "nodes": nodes,
        "edges": edges
    })


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
