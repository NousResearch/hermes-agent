"""
cosmos Emotional Token Server - 12D CST Full Architecture

Streams cosmos_packet JSON to cosmos Companion AI

Endpoints:
    GET /              - Server status
    GET /state         - Current cosmos_packet
    GET /stream        - SSE stream of packets
    WS  /ws            - WebSocket for real-time packets
    GET /system_prompt - Get LLM system prompt
    
Usage:
    python emotion_server.py
    python emotion_server.py --port 8765
"""

import os
import sys
import json
import time
import asyncio
import argparse
import math
from datetime import datetime
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try FastAPI
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("[ERROR] FastAPI not installed. Run: pip install fastapi uvicorn")

# Import 12D CST Emotional API
from emotional_state_api import (
    EmotionalStateAPI,
    EmotionalState,
    IntentState,
    CSTPhaseState,
    LLMPersonaMode,
    PHASE_SYNCHRONY,
    PHASE_MASKING_THRESHOLD,
    PHASE_LEAKAGE_THRESHOLD,
    ENTANGLEMENT_HIGH,
    DECEPTION_HIGH
)

# Create API instance
emotion_api = EmotionalStateAPI()

# System prompt path
SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"

# Create FastAPI app
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="cosmos 12D CST Token Server",
        description="Streams cosmos_packet JSON with full CST physics to cosmos AI",
        version=emotion_api.version
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    connected_clients = []
    
    
    @app.get("/")
    async def root():
        """Server status and CST configuration."""
        return JSONResponse({
            "server": "cosmos 12D CST Token Server",
            "version": emotion_api.version,
            "architecture": emotion_api.architecture,
            "status": "online",
            "endpoints": {
                "GET /": "This status page",
                "GET /state": "Current cosmos_packet",
                "GET /stream": "SSE stream of packets",
                "WS /ws": "WebSocket for real-time packets",
                "GET /system_prompt": "LLM system prompt for Physics-to-LLM Bridge"
            },
            "cst_configuration": {
                "phase_synchrony_deg": round(math.degrees(PHASE_SYNCHRONY), 1),
                "phase_masking_deg": round(math.degrees(PHASE_MASKING_THRESHOLD), 1),
                "phase_leakage_deg": round(math.degrees(PHASE_LEAKAGE_THRESHOLD), 1),
                "entanglement_threshold": ENTANGLEMENT_HIGH,
                "deception_threshold": DECEPTION_HIGH
            },
            "persona_modes": [m.value for m in LLMPersonaMode],
            "cst_states": [s.value for s in CSTPhaseState]
        })
    
    
    @app.get("/state")
    async def get_state():
        """Get current cosmos_packet (simulation mode)."""
        packet = emotion_api.get_state()
        return JSONResponse(packet)
    
    
    @app.get("/system_prompt")
    async def get_system_prompt():
        """Get the Physics-to-LLM Bridge system prompt."""
        if SYSTEM_PROMPT_PATH.exists():
            content = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
            return PlainTextResponse(content)
        return PlainTextResponse("System prompt not found", status_code=404)
    
    
    @app.get("/stream")
    async def stream_packets():
        """
        Server-Sent Events (SSE) stream of cosmos_packets.
        cosmos AI subscribes to this for continuous emotional data.
        """
        async def generate():
            while True:
                packet = emotion_api.get_state()
                yield f"data: {json.dumps(packet)}\n\n"
                await asyncio.sleep(0.5)  # 2 packets per second
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket for real-time cosmos_packets."""
        await websocket.accept()
        connected_clients.append(websocket)
        
        print(f"[OK] Client connected. Total: {len(connected_clients)}")
        
        # Send welcome with CST info
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to cosmos 12D CST Token Server",
            "version": emotion_api.version,
            "architecture": emotion_api.architecture,
            "persona_modes": [m.value for m in LLMPersonaMode]
        })
        
        try:
            while True:
                packet = emotion_api.get_state()
                token = {
                    "type": "cosmos_packet",
                    **packet
                }
                await websocket.send_json(token)
                
                try:
                    msg = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=0.5
                    )
                    if msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif msg.get("type") == "get_system_prompt":
                        if SYSTEM_PROMPT_PATH.exists():
                            content = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
                            await websocket.send_json({
                                "type": "system_prompt",
                                "content": content
                            })
                except asyncio.TimeoutError:
                    pass
                
        except WebSocketDisconnect:
            connected_clients.remove(websocket)
            print(f"[DISCONNECT] Client disconnected. Total: {len(connected_clients)}")
        except Exception as e:
            print(f"WebSocket error: {e}")
            if websocket in connected_clients:
                connected_clients.remove(websocket)


def run_server(host: str = "0.0.0.0", port: int = 8765):
    """Run the 12D CST emotional token server."""
    if not FASTAPI_AVAILABLE:
        print("[ERROR] Cannot start server: FastAPI not installed")
        return
    
    print("\n" + "=" * 70)
    print("  [CST] cosmos 12D CST TOKEN SERVER")
    print("  Full Architecture - Physics-to-LLM Bridge")
    print("=" * 70)
    print(f"\n  Version: {emotion_api.version}")
    print(f"  Architecture: {emotion_api.architecture}")
    print(f"\n  Server: http://{host}:{port}")
    print(f"\n  Endpoints:")
    print(f"    GET  http://localhost:{port}/              - Status")
    print(f"    GET  http://localhost:{port}/state         - cosmos_packet")
    print(f"    GET  http://localhost:{port}/stream        - SSE token stream")
    print(f"    WS   ws://localhost:{port}/ws              - WebSocket tokens")
    print(f"    GET  http://localhost:{port}/system_prompt - LLM prompt")
    print(f"\n  CST Phase Mapping:")
    print(f"    SYNCHRONY:  PhiG ~ {math.degrees(PHASE_SYNCHRONY):.0f} deg -> RESONANCE")
    print(f"    MASKING:    PhiG < {math.degrees(PHASE_MASKING_THRESHOLD):.0f} deg -> VERIFICATION")
    print(f"    LEAKAGE:    PhiG > {math.degrees(PHASE_LEAKAGE_THRESHOLD):.0f} deg -> DE-ESCALATION")
    print(f"    JITTER:     High dPhi/dt -> GROUNDING")
    print("\n" + "=" * 70 + "\n")
    
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="cosmos 12D CST Token Server"
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Port to bind to (default: 8765)"
    )
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port)
