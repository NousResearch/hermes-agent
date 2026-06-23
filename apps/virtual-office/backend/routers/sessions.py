from fastapi import APIRouter, HTTPException

from backend.services.hermes_bridge import HermesBridge

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
async def get_sessions() -> list[dict]:
    bridge = HermesBridge()
    try:
        return bridge.list_sessions()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Unable to list sessions: {exc}") from exc


@router.get("/{session_id:path}")
async def get_session(session_id: str) -> dict:
    bridge = HermesBridge()
    try:
        return bridge.resume_session(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Unable to resume session: {exc}") from exc
