import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.services.log_store import LOGS_PATH, get_event, list_events

router = APIRouter(prefix="/api/logs", tags=["logs"])


@router.get("")
async def get_logs(
    level: str | None = Query(default=None),
    agent: str | None = Query(default=None),
    task_id: str | None = Query(default=None),
    handoff_id: str | None = Query(default=None),
) -> list[dict]:
    return list_events(limit=100, level=level, agent=agent, task_id=task_id, handoff_id=handoff_id)


@router.get("/{log_id}")
async def get_log(log_id: str) -> dict:
    event = get_event(log_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Log entry not found")
    return event


@router.get("/stream")
async def stream_logs() -> StreamingResponse:
    async def event_generator():
        LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not LOGS_PATH.exists():
            Path(LOGS_PATH).touch()
        offset = LOGS_PATH.stat().st_size
        while True:
            current_size = LOGS_PATH.stat().st_size
            if current_size > offset:
                with LOGS_PATH.open("r", encoding="utf-8") as handle:
                    handle.seek(offset)
                    for line in handle:
                        raw = line.strip()
                        if not raw:
                            continue
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        yield f"data: {json.dumps(payload)}\n\n"
                    offset = handle.tell()
            elif current_size < offset:
                offset = current_size
            await asyncio.sleep(3)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)
