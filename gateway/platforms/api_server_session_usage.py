"""Bounded, opt-in model ledger reads for the session detail API."""

import asyncio
from typing import Any, Dict


async def session_usage_page(db: Any, session_id: str, *, limit: int, offset: int) -> Dict[str, Any]:
    rows = await asyncio.to_thread(db.session_model_usage_page, session_id, limit=limit + 1, offset=offset)

    return {
        "data": rows[:limit],
        "pagination": {"limit": limit, "offset": offset, "has_more": len(rows) > limit},
    }
