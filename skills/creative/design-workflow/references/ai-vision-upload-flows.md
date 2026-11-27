# AI Vision Upload Flow Case Notes

Use these notes when a product UI includes image upload, screenshot analysis, screenshot-to-design restoration, style extraction, or any AI vision workflow.

## Failure pattern observed

AI image analysis may keep failing even after generic retry/fallback when either of these is true:

1. **The client only compresses by width:**

```ts
if (img.width <= maxWidth) return originalDataUrl
```

Narrow-but-tall screenshots (especially mobile screenshots or long screenshots) can have width below the threshold but still carry high total pixels, high height, and dense text/UI detail.

2. **The browser calls the AI provider directly:** client-side calls to a provider such as APIMart can fail with CORS/network/browser transport errors (`TypeError: Load failed`, `NetworkError`, `Failed to fetch`) and may bypass server-side retry, provider error mapping, credential protection, logging, or fallback behavior.

Vision providers may return transient/internal/truncation errors such as:

- `finish_reason=length`
- `Internal error`
- `UnknownError`
- browser network errors like `TypeError: Load failed`, `NetworkError`, `Failed to fetch`
- localized retry messages like `临时失败`, `请再试一次`, `稍后重试`, `服务繁忙`

## Robust implementation pattern

1. **Keep provider calls server-side.** The browser should upload/compress locally if needed, then call a same-origin route such as `/api/analyze-style`. Do not expose provider keys or call `https://api.../chat/completions` from client code unless the product explicitly accepts that architecture and its trade-offs.
2. **Compress by bounding box and total pixels, not width only.** Calculate scale from max width, max height, and max megapixels; use the minimum scale.
3. **Use mode-specific plans.** Screenshot restoration often needs more vertical detail than generic image style analysis, but still needs a total pixel cap.
4. **Retry progressively on the server route.** Preferred sequence:
   - primary client-side preprocessing/compression
   - same-origin server analysis route with provider retry/error mapping
   - compact retry with stricter max width/height/pixels if the failure is retryable
5. **Propagate retry metadata.** Server API responses should include a machine-readable `retryable: true` (and usually HTTP 503 for transient AI errors), not only human copy.
6. **Classify truncation, browser transport, and localized transient errors as retryable.** Include `finish_reason=length`, `truncated`, `Load failed`, `NetworkError`, `Failed to fetch`, and Chinese retry text if the app is localized.
7. **Show state per tab/mode.** For multi-tab analysis pages, keep upload/result/progress state scoped to the active mode so a task from one tab does not leak into another.
8. **Make upload areas compact but complete.** Combine upload affordance, file metadata, remove action, preview, and optional URL input into one card instead of scattering them across multiple modules.

## Verification recipe

For production-like confidence, do all of these before calling the task done:

- Unit-test dimension planning: small image unchanged, wide image scaled, tall/narrow screenshot scaled, compact fallback stricter than primary.
- Unit-test retry classification: provider internal errors, truncation/`finish_reason=length`, HTTP 429/5xx, browser transport failures, and localized retry messages.
- Run build/tests.
- Use an actual uploaded screenshot fixture, not only mocked API responses.
- Instrument browser `fetch` or network logs during E2E and verify the client calls only same-origin APIs (for example `/api/analyze-style`), not `/api/token` or external provider endpoints.
- Verify desktop and a narrow mobile viewport (e.g. 390px) for overflow, tab state, and upload-card usability.
- Verify live/production after deploy when credentials/provider behavior differ from local.
- If a preview/deployment URL is protected by Vercel SSO or similar access control, resolve and test the public production alias before calling production E2E blocked.

## Regression test pattern

Add a lightweight client-pipeline regression test that reads the analyze page/source and asserts:

- same-origin route is present, e.g. `'/api/analyze-style'`
- external provider URLs are absent, e.g. `https://api.apimart.ai/v1/chat/completions`
- browser token bootstrap is absent, e.g. `'/api/token'`
- client authorization headers for provider keys are absent

This catches future refactors that accidentally reintroduce direct browser-to-provider calls.

## UX copy guidance

Use plain progress copy that explains the automatic recovery without alarming the user:

- Chinese: `AI 服务不稳定，正在自动压缩并重试…`
- English: `AI service is unstable. Compressing further and retrying…`

Avoid implying that the user caused the failure unless validation proves the uploaded file is invalid.