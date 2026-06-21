# ChatGPT Review Paper Chrome Extension

Manifest V3 Chrome extension that generates a local review paper from the signed-in user's own ChatGPT history.

## Build

```bash
npm install
npm test
npm run build
```

Load `dist/` in Chrome via `chrome://extensions` → Developer mode → Load unpacked.

## Live verification note

The ChatGPT history endpoints are undocumented. This implementation isolates every ChatGPT backend detail in:

- `src/source/chatgpt-backend-api-source.ts`
- `src/content/chatgpt-content.ts`

In this execution environment, direct live verification against a signed-in `chatgpt.com` DevTools Network session was blocked by Cloudflare before login. Before relying on the adapter, open ChatGPT in Chrome, inspect Network, and confirm/update:

- session check/token shape: `https://chatgpt.com/api/auth/session`
- conversation list endpoint, params, and response items shape
- conversation detail endpoint and message `mapping` shape

The rest of the app (normalization, cache, summarization, rendering, export) is independent of those details.

## Privacy posture

- No credential entry for ChatGPT; it uses the existing browser session on `chatgpt.com`.
- Conversation cache and generated reviews stay in IndexedDB / `chrome.storage.local`.
- Network destinations during a run are `chatgpt.com` plus the selected LLM provider.
