"""
Server Integration - API endpoints for VTuber control
Integrates with Cosmos web server for remote control and monitoring
"""

import asyncio
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
import json
import os
from loguru import logger

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .vtuber_core import CosmosVTuber, VTuberConfig, VTuberState
from .stream_manager import StreamQuality
from .avatar_controller import AvatarBackend


# Pydantic models for API
class StartStreamRequest(BaseModel):
    stream_key: str
    platform: str = "twitter"
    quality: str = "medium"
    simulate_chat: bool = False
    enable_swarm: bool = True


class SpeakRequest(BaseModel):
    text: str
    agent: str = "Cosmos"
    emotion: str = "neutral"


class ExpressionRequest(BaseModel):
    emotion: str
    intensity: float = 1.0


class ChatSimulateRequest(BaseModel):
    username: str
    message: str


# API key authentication for VTuber control endpoints
VTUBER_API_KEY = os.environ.get("VTUBER_API_KEY", "")


async def verify_api_key(authorization: Optional[str] = Header(None)):
    """Verify API key for VTuber control endpoints"""
    if not VTUBER_API_KEY:
        return  # No key configured = open access (dev mode)
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    token = authorization.replace("Bearer ", "").strip()
    if token != VTUBER_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


# Router for VTuber endpoints
router = APIRouter(prefix="/api/vtuber", tags=["vtuber"])

# Global VTuber instance
_vtuber_instance: Optional[CosmosVTuber] = None


def get_vtuber() -> Optional[CosmosVTuber]:
    """Get the current VTuber instance"""
    return _vtuber_instance


@router.post("/start", dependencies=[Depends(verify_api_key)])
async def start_stream(request: StartStreamRequest):
    """Start the VTuber stream"""
    global _vtuber_instance

    if _vtuber_instance and _vtuber_instance.is_live:
        raise HTTPException(status_code=400, detail="Stream already running")

    try:
        # Map quality string to enum
        quality_map = {
            "low": StreamQuality.LOW,
            "medium": StreamQuality.MEDIUM,
            "high": StreamQuality.HIGH,
            "ultra": StreamQuality.ULTRA,
        }
        quality = quality_map.get(request.quality, StreamQuality.MEDIUM)

        # Create config
        config = VTuberConfig(
            stream_key=request.stream_key,
            stream_quality=quality,
            simulate_chat=request.simulate_chat,
            use_swarm_collective=request.enable_swarm,
        )

        # Create and start VTuber
        _vtuber_instance = CosmosVTuber(config)
        success = await _vtuber_instance.start()

        if success:
            return {"status": "live", "message": "VTuber stream started"}
        else:
            raise HTTPException(status_code=500, detail="Failed to start stream")

    except Exception as e:
        logger.error(f"Stream start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop", dependencies=[Depends(verify_api_key)])
async def stop_stream():
    """Stop the VTuber stream"""
    global _vtuber_instance

    if not _vtuber_instance:
        raise HTTPException(status_code=400, detail="No stream running")

    try:
        await _vtuber_instance.stop()
    except Exception as e:
        logger.error(f"Error during stream stop: {e}")
    finally:
        _vtuber_instance = None

    return {"status": "offline", "message": "VTuber stream stopped"}


@router.get("/status")
async def get_status():
    """Get current VTuber status"""
    if not _vtuber_instance:
        return {
            "status": "offline",
            "is_live": False,
        }

    return {
        "status": _vtuber_instance.state.value,
        "is_live": _vtuber_instance.is_live,
        "stats": _vtuber_instance.stats,
    }


@router.post("/speak", dependencies=[Depends(verify_api_key)])
async def make_speak(request: SpeakRequest):
    """Make the VTuber speak"""
    if not _vtuber_instance or not _vtuber_instance.is_live:
        raise HTTPException(status_code=400, detail="VTuber not live")

    try:
        await _vtuber_instance._speak(
            text=request.text,
            agent=request.agent,
            emotion=request.emotion
        )
        return {"status": "ok", "message": "Speaking"}

    except Exception as e:
        logger.error(f"Speak error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/expression", dependencies=[Depends(verify_api_key)])
async def set_expression(request: ExpressionRequest):
    """Set avatar expression"""
    if not _vtuber_instance or not _vtuber_instance.is_live:
        raise HTTPException(status_code=400, detail="VTuber not live")

    try:
        await _vtuber_instance.avatar.set_expression(
            request.emotion,
            request.intensity
        )
        return {"status": "ok", "emotion": request.emotion}

    except Exception as e:
        logger.error(f"Expression error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/simulate", dependencies=[Depends(verify_api_key)])
async def simulate_chat(request: ChatSimulateRequest):
    """Simulate a chat message (for testing)"""
    if not _vtuber_instance or not _vtuber_instance.is_live:
        raise HTTPException(status_code=400, detail="VTuber not live")

    from .chat_reader import ChatMessage
    from datetime import datetime

    message = ChatMessage(
        id=f"sim_{datetime.now().timestamp()}",
        username=request.username,
        display_name=request.username,
        content=request.message,
        timestamp=datetime.now(),
        platform="simulated",
    )

    _vtuber_instance._on_priority_chat_message(message)

    return {"status": "ok", "message": "Chat message simulated"}


@router.get("/config")
async def get_config():
    """Get current VTuber configuration"""
    if not _vtuber_instance:
        return {"config": None}

    return {
        "config": {
            "name": _vtuber_instance.config.name,
            "persona": _vtuber_instance.config.persona,
            "avatar_backend": _vtuber_instance.config.avatar_backend.value,
            "stream_quality": _vtuber_instance.config.stream_quality.value,
            "enable_chat": _vtuber_instance.config.enable_chat,
            "use_swarm_collective": _vtuber_instance.config.use_swarm_collective,
            "swarm_agents": _vtuber_instance.config.swarm_agents,
        }
    }


@router.get("/health")
async def vtuber_health():
    """Get VTuber system health including process supervisor"""
    health = {
        "vtuber_active": _vtuber_instance is not None,
        "is_live": _vtuber_instance.is_live if _vtuber_instance else False,
    }

    # Add supervisor health if available
    if _vtuber_instance and hasattr(_vtuber_instance, 'stream') and _vtuber_instance.stream:
        stream = _vtuber_instance.stream
        health["stream"] = {
            "status": stream.stats.status,
            "frames_sent": stream.stats.frames_sent,
            "dropped_frames": stream.stats.dropped_frames,
            "uptime": stream.stats.uptime_seconds,
        }

    return health


# WebSocket for real-time updates
@router.websocket("/ws")
async def vtuber_websocket(websocket: WebSocket):
    """WebSocket for real-time VTuber updates"""
    await websocket.accept()

    try:
        while True:
            # Send status updates
            if _vtuber_instance:
                status = {
                    "type": "status",
                    "state": _vtuber_instance.state.value,
                    "is_speaking": _vtuber_instance.state == VTuberState.SPEAKING,
                    "current_agent": _vtuber_instance._current_agent,
                    "stats": _vtuber_instance.stats,
                }
            else:
                status = {
                    "type": "status",
                    "state": "offline",
                    "is_speaking": False,
                }

            try:
                await websocket.send_json(status)
            except Exception:
                break
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        logger.debug("VTuber WebSocket disconnected")
    except Exception as e:
        logger.debug(f"VTuber WebSocket error: {e}")


# Control panel HTML endpoint
@router.get("/panel", response_class=JSONResponse)
async def control_panel():
    """Return control panel configuration"""
    return {
        "title": "Cosmos VTuber Control",
        "endpoints": {
            "start": "/api/vtuber/start",
            "stop": "/api/vtuber/stop",
            "status": "/api/vtuber/status",
            "speak": "/api/vtuber/speak",
            "expression": "/api/vtuber/expression",
            "websocket": "/api/vtuber/ws",
        },
        "emotions": [
            "neutral", "happy", "sad", "angry", "surprised",
            "thinking", "excited", "confused", "smug", "curious"
        ],
        "agents": [
            "Cosmos", "Grok", "DeepSeek", "Gemini",
            "Claude", "Kimi", "Phi", "Swarm-Mind"
        ],
    }


def register_vtuber_routes(app):
    """Register VTuber routes with FastAPI app"""
    app.include_router(router)
    logger.info("VTuber API routes registered")


# CLI interface for standalone testing
async def run_standalone_vtuber(stream_key: str, simulate: bool = True):
    """Run VTuber in standalone mode for testing"""
    config = VTuberConfig(
        stream_key=stream_key,
        simulate_chat=simulate,
        debug_mode=True,
    )

    vtuber = CosmosVTuber(config)

    try:
        await vtuber.start()

        # Keep running
        while vtuber.is_live:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down...")

    finally:
        await vtuber.stop()


if __name__ == "__main__":
    import sys

    stream_key = sys.argv[1] if len(sys.argv) > 1 else ""

    if not stream_key:
        print("Usage: python server_integration.py <stream_key>")
        print("       Use 'test' for simulated mode without streaming")
        sys.exit(1)

    asyncio.run(run_standalone_vtuber(
        stream_key=stream_key,
        simulate=stream_key == "test"
    ))
