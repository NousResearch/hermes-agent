# OpenAI Chat Completions Adapter Pattern

Bridge an external OpenAI-compatible client (e.g. OpenClaw Dashboard, OpenAI JS SDK 6.x) to an internal brain-ask pipeline without modifying the client or switching providers.

## When to use

- An external service speaks OpenAI `/chat/completions` format and needs to reach a custom brain/ask endpoint
- The internal pipeline does reasoning, governance, safety, memory, and classification before calling the actual LLM
- You cannot (or should not) modify the external client's API call format
- You need to preserve the internal identity, safety envelope, and provider selection

## Architecture (non-tool requests)

```
POST /chat/completions  { messages: [...], stream: false }
         |
  [auth middleware]  -- 401 if unauthorized
         |
  [determine route: body.tools present?]
         |
         NO
         |
  [extract latest user message]  -- backwards iteration
         |
  buildBrainAskResponse({ env, prompt, fetchImpl })
         |
  [response mapper]  -- OpenAI JSON + _cloe metadata
         |
  200 JSON or SSE
```

## Architecture (tool-enabled requests)

```
POST /chat/completions  { messages, tools: [...], tool_choice: "auto" }
         |
  [auth middleware]
         |
  [determine route: body.tools present?]
         |
         YES
         |
  [validate tool definitions + tool_choice]  -- 400 if malformed
         |
  buildNativeChatCompletions({ messages, tools, tool_choice, ... })
         |
  [resolve provider config + build payload]
         |
  POST /v1/chat/completions to provider (messages + tools forwarded)
         |
  [build OpenAI response preserving tool_calls + finish_reason]
         |
  200 JSON or SSE with tool_calls chunks
```

## Implementation steps

### 1. Auth middleware

Accept both `x-admin-api-key` (legacy) and `Authorization: Bearer` (OpenAI SDK standard).

### 2. Create native tool protocol service

A dedicated module (`chat-completions-service.js`) exporting:

```javascript
buildNativeChatCompletions({ messages, tools, tool_choice, parallel_tool_calls, model, stream, env, fetchImpl, timeoutMs })
validateToolDefinitions(tools)        // each must be { type: "function", function: { name } }
validateToolChoice(tool_choice)        // "auto" | "none" | "required" | { type:"function", function: { name } }
buildProviderPayload({ messages, tools, ... })  // OpenAI-compatible body for provider
resolveProviderConfig(env)             // reads OPENCLAW_BRAIN_PROVIDER, DEEPSEEK_API_KEY, etc.
```

### 3. Bifurcate the route handler

```javascript
const hasNativeTools = Array.isArray(body.tools) && body.tools.length > 0;

if (hasNativeTools) {
  // Full OpenAI lifecycle: messages, tools, tool_choice forwarded to provider
  const result = await buildNativeChatCompletions({...});
  if (!result.ok) return res.status(500).json({ error: { message, type, code } });
  // SSE: emit role, content, tool_calls, finish_reason chunks
  return res.status(200).json(result.value);
} else {
  // Legacy: extract latest user text → CLOE cognitive pipeline
  // ... existing buildBrainAskResponse logic
}
```

### 4. SSE streaming with tool_calls

When `stream=true` and the response has tool_calls, emit OpenAI SSE chunks:

```
data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}
data: {"choices":[{"delta":{"content":"text"},"finish_reason":null}]}
data: {"choices":[{"delta":{"tool_calls":[...]},"finish_reason":null}]}  // single chunk
data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}
data: [DONE]
```

### 5. Never return HTML

Wrap in try/catch. On uncaught exception return 500 JSON.

## Tool lifecycle for governed action preparation

```
1. WebUI sends messages + tools (cloe_operator_action) + tool_choice: "auto"
2. Route detects tools → buildNativeChatCompletions
3. Service validates, resolves provider (DeepSeek), forwards tools
4. DeepSeek returns tool_calls with cloe_operator_action
5. Adapter preserves tool_calls verbatim in response
6. OpenClaw receives structured tool_calls, executes via tool executor
7. Tool result → second turn (tool message forwarded to provider)
8. DeepSeek returns final text answer based on tool result
```

Safety: `execution.allowed: false` enforced at tool definition level. No approval ID or LAHB state invented by the adapter.

## Pitfalls

| Pitfall | Fix |
|---------|-----|
| Tools silently discarded | Add `hasNativeTools` bifurcation — do NOT pass tools through text-extraction pipeline |
| `finish_reason` rewritten to `"stop"` | Preserve from provider verbatim |
| `tool_calls` serialized into text | Preserve as structured JSON in `message.tool_calls` |
| Normal text forced into tool calling | Only use native path when `body.tools` is present AND non-empty |
| GIT_COMMIT metadata stale after Docker build | Pass `GIT_COMMIT=$(git rev-parse HEAD)` explicitly as build arg |
| WebUI trial cannot verify adapter from CLI | Test protocol parity via direct HTTP to deployed container |
| Provider return value without tool_choice not forwarded | Include `tool_choice` only when supplied — omit for text-only requests |
