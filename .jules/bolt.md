# Performance optimization learnings

- **FastAPI with SQLite**: When working with FastAPI endpoints (`async def`), any synchronous database calls using sqlite3 (like the ones made via `SessionDB`) must be offloaded to a separate thread to prevent blocking the async event loop.
- **Pattern**: Refactor the synchronous logic into an inner `_sync_...` function, and wrap the original endpoint logic with `return await asyncio.to_thread(_sync_..., *args)`.
- **Measurement**: Using concurrent loops with asyncio.sleep alongside the blocking calls makes it very clear how much the event loop is blocked. For example, gap delays jumped from 0.01s expected to >0.22s when blocked. Offloading the work brings latency down strictly by order of magnitude.
