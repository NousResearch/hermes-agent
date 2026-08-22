# Chat-Completions Tool-Call Path Divergence — Diagnostic Pattern

Established 2026-08-05 in CLOE_OURDREAM_CHAT_COMPLETIONS_TOOLCALL_ADAPTER_PATH_DIAGNOSTIC_V1 (DIAGNOSTIC-ONLY, no fix applied).

## Symptom shape

Same session, same provider (`cloe` = MVP /chat/completions on 127.0.0.1:4000, api=openai-completions):

- Successful turns: `assistant stopReason=toolUse` (toolCallId `call_00_...`, toolName read/exec) → toolResult → continuation → stop.
- Failed turns: `assistant stopReason=stop`, no toolCall, no toolResult, ~1s latency, text reply that exposes internal retrieval layers.

Mission-level symptom: "assistant stop=stop / no toolCall / no toolResult / safe fallback" while other sessions complete the full tool lifecycle.

## Read-only diagnostic method (no docker, no process inspection)

1. **Worktree at the deployed SHA, no rm -rf:** `git worktree add --detach /tmp/cloe-diag-<unique-ts> <sha>`. `rm -rf` requires explicit operator consent even in /tmp — use a fresh unique path instead. Treat mission-provided container state (restart count, health, mounts, guards) as acquired; do NOT run `docker inspect` / `pm2 jlist` / process listing during read-only diagnostics (operator denies these).
2. **Session JSONL metadata extraction** (never print full prompts/tool results): role, stopReason, api/provider/model, toolCallId/toolName, idempotencyKey, responseId, errorCode, timestamps.
3. **Entrypoint marker:** `idempotencyKey=<uuid>:user` on user messages = dashboard SPA (get-reply recorder, idempotent chokepoint PR #668). Absence ≠ different entrypoint by itself — correlate per-session.
4. **responseId fingerprint:** all replies carry `responseId = chatcmpl-<epoch>-<rand>` (MVP buildOpenAiResponse format) → EVERYTHING flows through MVP /chat/completions. The MVP's ws gateway / canonical service (session-adapter in-memory Map) is NOT the path that writes the runtime JSONL — do not chase it.
5. **usage=0 is NOT a discriminator:** runtime openclaw persists usage=0 on ALL replies for the custom `cloe` provider (successes AND failures). Check the provider payload `tools` key instead.
6. **Stable-block violation fingerprint:** if the reply opens with a phrase the stable-block forbids (e.g. "Réponse basée sur la connaissance indexée (retrieval-first)") and lists layer names (HOT_CURRENT_STATE, EXACT_REGISTRY, CAMPAIGN_MEMORY), the request took the provider-bypass/deterministic branch — NOT the legacy `buildBrainAskResponse` path, which injects the stable-block via `buildOpenClawBrainContext`.
7. **Deployment-timing correlation:** `git log --all -S "<string>" -- <file>` + `git show -s --format='%h %ci %s' <sha>` to date the introducing commit; correlate successes (before) vs failures (after).
8. **Registry feedback loop:** inspect `data/knowledge/lah-knowledge-registry.json` for entities named `retrieval-*` whose summary is `Answered from retrieval (LAYER): <user query>`. `created_at` timestamps show the loop feeding itself (each user turn indexed → next turn matches → ANSWER_SUFFICIENT → tools stripped).

## Root cause (confirmed)

`src/services/chat-completions-service.js` → `buildNativeChatCompletions`, ANSWER_SUFFICIENT branch:

```js
effectiveTools = undefined;
effectiveToolChoice = 'none';
forcedSynthesis = true;
```

→ provider payload without `tools` → DeepSeek cannot emit tool_calls → text terminal reply → OpenClaw runtime persists stop=stop; the native tool-loop runner is never entered (nothing to dispatch).

Primary classification: **PROVIDER_ADAPTER_CONTRACT_DIVERGENCE**.

The runtime openai-completions adapter (`openai-completions-DTj6G8AI.js`) and parser (`"tool_calls" → { stopReason: "toolUse" }`) are CORRECT — proven by the successful turns. Do not blame DeepSeek tool support; the request never carried tools.

## Secondary factors

- `b582b08` (CLOE_RETRIEVAL_FIRST_TOOL_BUDGET_AND_PROGRESSIVE_REPLY_V1, 2026-08-01 16:54) introduced BOTH the retrieval-first bypass AND indexing user messages as `Answered from retrieval (LAYER): <query>` entities (chat-completions-service.js:593 at that SHA) → subsequent turns matched → ANSWER_SUFFICIENT → tools stripped.
- `b3acd3e` (19:11 same day) later restored the LLM (no more deterministic non-LLM answers) and added the `looksConversationalEphemera` guard, but did NOT restore tools.
- Failures at 17:31-17:32 UTC are ~37 min after the b582b08 deploy; successful turns at 15:00 / 15:49 precede it.
- L204 stop=error (provider abort) before the failures was a red herring — the gap was a redeploy, not a runtime degradation.

## Repair ranking (diagnostic output — not applied)

1. Keep client-supplied tools in the ANSWER_SUFFICIENT payload: only strip `tool_choice` when the client sent no tools. Grounding retrieval stays injected as internal context. Smallest change, restores the native tool loop.
2. Per-request `never-strip-tools` flag from the runtime client (bigger contract change).
3. Read-only evidence collector before synthesis (belt-and-suspenders only — does not fix the tool strip).

## Verification notes

- Assert on the provider payload `tools` key, not on usage.
- Test suite shape: ANSWER_SUFFICIENT + client tools → payload keeps tools; provider tool_calls → buildOpenAiResponse preserves them; DSML rejection unchanged (NFKC guard, U+FF5C family); greeting ("salut cloe") → LLM synthesis without tool strip; the 3 exact failed requests (diagnostic / microtest / paused-plan) → toolUse or honest synthesis, never empty stop text.
