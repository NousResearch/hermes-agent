# Provider True-Streaming Transport Pattern (SSE + staged timeouts)

Established during `HERMES_CLOE_TRUE_PROVIDER_STREAMING_AND_TIMEOUT_REPAIR_V1`
(2026-08-08) in lah-openclaw-mvp. Successor to the "provider timeout budget"
missions that kept `stream:false` + a single absolute deadline.

## When to use

Any LAH provider-response path that must:
- NOT destroy long-running DeepSeek/OpenAI-compatible generations with a fixed
  35s/70s hard deadline while chunks keep arriving;
- deliver TRUE progressive SSE to a client instead of buffering the full JSON
  and replaying it;
- keep a deterministic, non-blank fallback on every failure mode.

## Architecture

Shared transport module (`src/services/provider-chat-transport.js`) consumed by
BOTH the native-tool path (`chat-completions-service.js`) and the direct
brain/ask path (`readonly-operator-cli-client.js`) — parity via SHARED code,
never a second implementation.

Exports: `createStagedTimeoutPolicy`, `parseSseEvents`, `parseSseJson`,
`createStreamAccumulator`, `reconstructCompletion`,
`streamProviderChatCompletion`, `buildClientDelta`, `isProviderStreamFailure`,
`STREAM_FAILURE_CODES`.

## Staged timeout policy (non-destructive)

- `connectTimeoutMs` — connection/request-start bound (default 10s, env
  `CLOE_PROVIDER_CONNECT_TIMEOUT_MS`)
- `firstTokenTimeoutMs` — = the round's effective timeout (35s standard / 70s
  post-tool / explicit override). The differentiated doctrine is preserved AS A
  FIRST-TOKEN BOUND, not a total deadline.
- `idleTimeoutMs` — gap between chunks; RESET on every valid chunk (default 30s,
  env `CLOE_PROVIDER_IDLE_TIMEOUT_MS`)
- `absoluteCeilingMs` — generous last-resort total bound (default 300s, env
  `CLOE_PROVIDER_ABSOLUTE_CEILING_MS`); `governance.maxProviderTimeoutMs`
  overrides the ceiling (operator hard deadline = last-resort protection)
- `combinedPreStreamMs` — when the caller passes an explicit `timeoutMs`
  (tests/scripts/harnesses), the WHOLE pre-first-token window uses that single
  budget from request start, and both connect+first_token stages abort with the
  legacy `PROVIDER_TIMEOUT` code → existing T4/T5-style timeout-fallback tests
  keep their exact contract.

Key invariant: a healthy response is NEVER aborted merely because total
generation exceeds 35s/70s while chunks continue arriving.

## SSE parser

Incremental, buffer split on `/\r?\n\r?\n/`, `data:` lines, `[DONE]` sentinel,
CRLF tolerant, chunk-boundary splits handled, comments (`: ...`) skipped.
Must accept an ALREADY-RESOLVED iterator (see pitfall 1).

## Accumulator + reconstruction

- content fragments joined; `tool_calls` accumulated per index with argument
  fragment concatenation; finish_reason; usage
- `usage` arrives on the FINAL SSE chunk at TOP level (with
  `stream_options.include_usage: true`), never inside `choices`
- `reconstructCompletion` produces the SAME shape as a `stream:false` provider
  JSON response → downstream consumers (`buildOpenAiResponse`, budget gates,
  protocol recovery, denial-synthesis) are unchanged

## Dual-mode body handling

- `response.body` async-iterable → TRUE incremental SSE (production)
- only `response.text()` (legacy test mocks) → read text; starts with `{` →
  plain-JSON single-shot completion; starts with `data:` → SSE parse of the
  buffered text. Production always has a body stream, so production is always
  true streaming.

## Failure semantics (LOT 5)

Codes: `PROVIDER_CONNECT_TIMEOUT`, `PROVIDER_TIMEOUT` (first-token, legacy),
`PROVIDER_IDLE_TIMEOUT`, `PROVIDER_ABSOLUTE_TIMEOUT`,
`PROVIDER_STREAM_MALFORMED`, `PROVIDER_STREAM_INCOMPLETE`.
Every failure returns `stream_state { interrupted, reason, chunks_received,
partial_content, partial_tool_calls }` — partial provider output is NEVER
silently presented as a complete answer. Callers route ALL these codes to the
deterministic fallback (never blank) and surface `_cloe.provider_stream`.

## Pitfalls (each cost real debugging time)

1. **Web ReadableStream double-lock.** Calling `body[Symbol.asyncIterator]()`
   LOCKS the stream; passing `response.body` to a second parser that calls the
   iterator again throws `Invalid state: ReadableStream is locked`. Fix: create
   the iterator ONCE (`getAsyncIterator(body)`), pass the ITERATOR into the
   parser, and make the parser accept either (detect `typeof x.next ===
   'function'`).

2. **`unref()` stage timers hang node:test.** A stage timer with `.unref()`
   does not keep the event loop alive; a never-resolving fetch mock + unref'd
   timer → test cancelled with "Promise resolution is still pending but the
   event loop has already resolved" / `cancelledByParent`. Fix: do NOT unref
   the abort-critical stage timers — they are cleared in `finally` anyway.

3. **setInterval poll to detect timer fire LEAKS.** Racing fetch against a
   `setInterval` that polls `timer.current` keeps the event loop alive forever
   when the fetch wins. Fix: a rejectable race promise
   (`timer.createRacePromise()`); the timer's `fire()` rejects it; `clear()`
   nulls `rejectFn`.

4. **Usage field location.** OpenAI SSE usage is top-level on the final chunk,
   NOT `choices[0].usage`. Reading the wrong location silently yields zero
   usage.

5. **env not threaded → test overrides silently ignored.**
   `createStagedTimeoutPolicy(env = process.env)` ignores a per-call env object
   unless every layer threads it explicitly (`callProvider` opts.env → policy
   env). Symptom: `CLOE_PROVIDER_IDLE_TIMEOUT_MS` override in test env had no
   effect; idle never fired and the stream ended as `incomplete`.

6. **SSE test mock string-vs-object.** A stream builder that reads `ev.text`
   breaks when given plain strings. Normalize: `const text = (ev && typeof ev
   === 'object') ? ev.text : ev;`.

7. **Stalling body must race the stage timer.** `for await` over a body that
   never yields blocks forever; race each `iterator.next()` against the active
   stage timer's rejection, and call `iterator.return()` in `finally` to avoid
   a dangling generator.

## Testing pattern

- SSE body mocks via `new ReadableStream({ async start(controller) { ... } })`
  with `{ text, delayMs }` event objects for timing.
- Real-time tests B (>35s) and C (>70s) with 2s/4s chunk cadence prove the
  mission's core claim; acceptable suite cost (~110s). Keep them — they are the
  certification bar, not an optimization target.
- Explicit `timeoutMs` in tests preserves legacy single-deadline semantics
  (combined pre-stream budget) so old timeout-fallback tests stay green.
