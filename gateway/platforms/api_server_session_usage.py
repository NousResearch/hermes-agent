"""Bounded, opt-in model ledger reads for the session detail API."""

import asyncio


async def session_usage_page(db, session_id, query):
    from gateway.platforms.api_server import APIServerAdapter

    limit = APIServerAdapter._parse_nonnegative_int(query.get("usage_limit"), default=100, maximum=500)
    limit = max(1, limit)
    offset = APIServerAdapter._parse_nonnegative_int(query.get("usage_offset"), default=0, maximum=1_000_000)
    rows = await asyncio.to_thread(db.session_model_usage_page, session_id, limit=limit + 1, offset=offset)

    return {
        "data": rows[:limit],
        "pagination": {"limit": limit, "offset": offset, "has_more": len(rows) > limit},
    }
