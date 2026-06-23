from typing import Any

from fastapi import APIRouter, HTTPException

from backend.services.handoff_manager import HandoffManager
from backend.services.json_store import HANDOFFS_PATH, find_by_id, read_list_store

router = APIRouter(prefix="/api/handoffs", tags=["handoffs"])


@router.get("")
async def get_handoffs() -> list[dict[str, Any]]:
    return read_list_store(HANDOFFS_PATH)


@router.get("/{handoff_id}")
async def get_handoff(handoff_id: str) -> dict[str, Any]:
    handoff = find_by_id(read_list_store(HANDOFFS_PATH), handoff_id)
    if handoff is None:
        raise HTTPException(status_code=404, detail="Handoff not found")
    return handoff


@router.post("")
async def create_handoff(payload: dict[str, Any]) -> dict[str, Any]:
    from_agent = str(payload.get("from_agent") or "").strip()
    to_agent = str(payload.get("to_agent") or "").strip()
    raw_payload = payload.get("payload")
    auto_run = bool(payload.get("auto_run"))

    if not from_agent:
        raise HTTPException(status_code=400, detail="from_agent is required")
    if not to_agent:
        raise HTTPException(status_code=400, detail="to_agent is required")

    handoff_payload = raw_payload if isinstance(raw_payload, dict) else {"text": str(raw_payload or "")}
    return HandoffManager().create_handoff(
        from_agent=from_agent,
        to_agent=to_agent,
        payload=handoff_payload,
        auto_run=auto_run,
    )


@router.post("/{handoff_id}/run-again")
async def run_handoff_again(handoff_id: str) -> dict[str, Any]:
    return HandoffManager().run_handoff_again(handoff_id)
