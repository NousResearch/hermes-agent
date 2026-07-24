# Vertex AI Gemini — Feature Roadmap

Status of Gemini model features through the Vertex OpenAI-compatible endpoint
after API key (Express Mode) auth is established.

## Legend

- ✅ **Works** — tested and confirmed
- 🟡 **Partial** — works in theory but needs verification/tuning
- 🔧 **Needs work** — requires implementation

---

## Core Features

| Feature | Status | Notes |
|---------|--------|-------|
| **Chat completions** | ✅ | Tested: `"Okay, sure."` with full token accounting |
| **Streaming (SSE)** | 🟡 | Chat_completions transport supports SSE; needs live test with Vertex |
| **Authentication** | ✅ | `x-goog-api-key` header, confirmed 200 OK |
| **Reasoning / Thinking** | ✅ | Wired via `extra_body.google.thinking_config` in the plugin |
| **Token tracking** | ✅ | `prompt_tokens`, `completion_tokens`, `reasoning_tokens` all reported |
| **Model selection** | ✅ | Curated list (13 models) + `/model` slash command |

---

## Tool Calling

| Feature | Status | Notes |
|---------|--------|-------|
| **Function calling** | 🟡 | OpenAI-compat transport handles `tool_calls` generically. Works if Vertex endpoint accepts OpenAI-format `tools` array. Needs live test. |
| **Parallel tool calls** | 🟡 | Depends on tool calling working first |
| **Structured output (JSON mode)** | 🟡 | OpenAI-compat transport supports `response_format: {type: 'json_object'}`. Gemini supports it natively. Needs verification. |

**Implementation effort:** Low. The chat_completions transport already handles
tool calls for any OpenAI-compatible endpoint. If Vertex returns 400 on the
OpenAI tool format, we may need a thin adapter.

---

## Multimodal / Image Upload

| Feature | Status | Notes |
|---------|--------|-------|
| **Image input (base64)** | 🔧 | Hermes sends images as `data:` URIs or base64 in messages. Vertex OpenAI-compat endpoint expects `inlineData` format (Gemini native), NOT OpenAI `image_url`. A format adapter is needed. |
| **Image input (URL)** | 🔧 | Same as base64 — format mismatch |
| **Audio input** | 🔧 | Gemini supports audio natively; needs transport adapter |
| **Video input** | 🔧 | Gemini supports video (GCS URIs); needs transport adapter |
| **PDF input** | 🔧 | Gemini supports PDF natively |

**The problem:** The Vertex OpenAI-compatible endpoint accepts OpenAI-format
messages (`role`, `content` with text), but for multimodal input it uses
Gemini's native format, not OpenAI's `image_url` / `input_audio` format.

**Fix:** Add a `vertex` message adapter in `agent/transports/` that converts
OpenAI-format multimodal messages to Gemini's `inlineData` parts before
sending. This is the same adapter pattern used for Anthropic's transport.

---

## Prompt Caching

| Feature | Status | Notes |
|---------|--------|-------|
| **Automatic server-side caching** | 🟡 | Google may cache transparently. Not configurable. |
| **Cached Content API** | 🔧 | Google's `/v1/cachedContents` endpoint for explicit cache management |
| **Cache breakpoints** | ❌ | Not supported by Gemini. `cache_control` markers are Anthropic-only. |

**Vertex approach:** Google's caching is managed through a separate
`cachedContents` API — create a cache entry, get a `cachedContent` name,
then reference it in `generateContent` requests via `cachedContent` field
in the request body.

**Implementation effort:** Medium. Requires:
1. New module: `agent/vertex_cache.py` — create/update/delete cached contents
2. Transport mod: pass `cachedContent` field in request body when cache is active
3. Integration with Hermes' context compression / session management

---

## Advanced Features

| Feature | Status | Notes |
|---------|--------|-------|
| **System instructions** | ✅ | OpenAI-compat transport sends `system` role → Vertex maps to `systemInstruction` |
| **Safety settings** | 🔧 | Gemini supports `safetySettings` array; needs `extra_body` passthrough |
| **Frequency / presence penalty** | 🟡 | OpenAI-compat transport sends these; Vertex may ignore or accept |
| **Max tokens / stop sequences** | ✅ | Standard OpenAI params, handled by chat_completions transport |
| **Temperature / top_p / top_k** | 🟡 | OpenAI `temperature`/`top_p` map automatically. Gemini `top_k` needs `extra_body` |
| **Seed** | 🟡 | OpenAI-compat transport passes `seed`; Gemini accepts it |
| **Logprobs** | ❓ | Not tested. Gemini supports but may need `extra_body` |

---

## Provider-Specific Gaps

| Area | What's missing |
|------|----------------|
| **OpenAI-compatible messages → Gemini format** | Multimodal parts (images, audio, video) need inlineData conversion |
| **Vertex cachedContents API** | Full lifecycle: create cache, reference, update TTL, delete |
| **`extra_body.google` passthrough** | The plugin already emits `thinking_config`. Need to also pass `safety_settings`, `top_k` |
| **Safety settings** | `safetySettings: [{category, threshold}]` — Gemini-native, needs `extra_body` |
| **Model listing (API key)** | Google doesn't expose `models.list` for Express Mode. Curated list is the only option. |

---

## Implementation Plan

### Phase 1 — Core (done)
- [x] API key auth (`GOOGLE_VERTEX_API_KEY`, `x-goog-api-key` header)
- [x] Reasoning / thinking config
- [x] Model picker with curated list
- [x] `hermes doctor` detection
- [x] Desktop settings card
- [x] Web dashboard env vars

### Phase 2 — Tool calling & structured output
- [ ] Verify tool calling works with Vertex OpenAI-compat endpoint
- [ ] Add adapter if needed (Vertex tool format ≠ OpenAI format)
- [ ] Test JSON mode (`response_format: {type: 'json_object'}`)

### Phase 3 — Multimodal support
- [ ] Create `agent/transports/vertex.py` — message adapter
- [ ] Convert OpenAI `image_url` parts to Gemini `inlineData` parts
- [ ] Test with image upload in chat
- [ ] Add support for audio, video, PDF via GCS URIs

### Phase 4 — Prompt caching
- [ ] Implement `agent/vertex_cache.py`
- [ ] Create/query cached contents via Vertex API
- [ ] Wire cache references into request body
- [ ] Integrate with session lifecycle

### Phase 5 — Advanced features
- [ ] Safety settings passthrough via `extra_body`
- [ ] `top_k` support in generation config
- [ ] Logprobs support
- [ ] Context caching for long sessions
