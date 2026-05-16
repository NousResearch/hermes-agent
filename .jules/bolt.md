# Performance optimization learnings

- **FastAPI with SQLite**: When working with FastAPI endpoints (`async def`), any synchronous database calls using sqlite3 (like the ones made via `SessionDB`) must be offloaded to a separate thread to prevent blocking the async event loop.
- **Pattern**: Refactor the synchronous logic into an inner `_sync_...` function, and wrap the original endpoint logic with `return await asyncio.to_thread(_sync_..., *args)`.
- **Measurement**: Using concurrent loops with asyncio.sleep alongside the blocking calls makes it very clear how much the event loop is blocked. For example, gap delays jumped from 0.01s expected to >0.22s when blocked. Offloading the work brings latency down strictly by order of magnitude.

## 2024-05-16 - [Fast batch tokenization in python]
**Learning:** HuggingFace `tokenizer` batch encoding (passing a list of texts to `tokenizer(texts)`) is approximately 3x faster than calling `tokenizer.encode(text)` in a loop, as it delegates the iteration down to the optimized Rust implementation within the `transformers` library. This is critical when computing token counts across multiple turns within a chat trajectory.
**Action:** When working with tokenizers and a list of texts, prefer passing the entire list to the tokenizer at once rather than looping over items, making sure to fallback gracefully to character limits if the tokenizer object is absent or errors out.
