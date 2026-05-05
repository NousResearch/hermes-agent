# HZCA serve REST/SSE contract for Hermes Zalo adapter

Verified against local `/home/hafo_creative/hzca` smoke server with:

```bash
npm run smoke:serve -- --no-build --sse-timeout-ms 2000
```

Smoke result on 2026-05-03:
- `/api`, `/api/health`, `/api/metrics`, `/api/events`: OK
- SSE emits `event: connected` with JSON payload `{"type":"connected","version":...}`

Adapter MVP contract:
- Health: `GET /api/health` (adapter also accepts `/health` fallback for fake tests only)
- Self ID: prefer `GET /api/me/id -> {"id":"..."}`; fallback `GET /api/me` and read `userId|uid|id|zaloId` from top-level or nested data/user/profile/me
- Send text: `POST /api/messages/text` body `{threadId, message, isGroup, autoMarkdown:false}`
- Typing: no stable HZCA typing endpoint found in serve route source; adapter treats typing as best-effort no-op unless backend adds `/api/typing` later
- SSE: `GET /api/events`, frames may include `id:`, `event:`, `retry:`, comments, and multi-line `data:`. Message frames use `event: message` and JSON payload.

Privacy rule: adapter must not store raw SSE blobs in `MessageEvent.raw_message`; only sanitized IDs/thread/sender/content/type/quote ids are retained.
