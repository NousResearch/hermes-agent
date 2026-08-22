# True Provider Streaming (SSE) — Pitfalls & Certification Pattern

Established during HERMES_CLOE_TRUE_PROVIDER_STREAMING_AND_TIMEOUT_REPAIR_V1
(PR #736, merged b7fbddd, 2026-08-08). Successor to
CLOE_POST_TOOL_PROVIDER_TIMEOUT_BUDGET_REPAIR_V1 (#715).

## Shared transport architecture (LOT 3 parity by construction)

- ONE module (e.g. `provider-chat-transport.js`) consumed by BOTH the
  native-tool path (`chat-completions-service.js`) and the direct brain/ask
  path (`readonly-operator-cli-client.js`). Parity by construction, never a
  second implementation.
- Provider payloads: `stream: true` + `stream_options: { include_usage: true }`.
  SSE parsed incrementally; the canonical `chat.completion` is reconstructed
  from content + tool_calls deltas.
- Staged timeouts: connect → first-token (= round budget) → idle (resets on
  every valid chunk) → generous absolute ceiling. When a caller passes an
  explicit combined pre-stream budget (legacy tests/harnesses), connect +
  first-token collapse to the legacy timeout code so old fallback semantics
  survive exactly.

## Pitfall: web ReadableStream double-lock

`getAsyncIterator(response.body)` LOCKS a web ReadableStream. Calling the
SSE parser with `response.body` again afterwards throws
`Invalid state: ReadableStream is locked` → misclassified as
PROVIDER_REQUEST_FAILED with 0 chunks. Fix: resolve the iterator ONCE
(`const it = getAsyncIterator(response.body)`), pass `it` to the parser, and
let the parser accept either a raw body or an already-resolved iterator
(`typeof x.next === 'function'` → use it directly).

## Pitfall: `unref()` on stage timers + never-resolving fetch

Calling `.unref()` on timeout-stage timers lets the event loop drain when the
fetch is a never-resolving mock (`await new Promise(() => {})`) → node --test
cancels with "Promise resolution is still pending but the event loop has
already resolved". Fix: do NOT unref stage timers — they are always cleared
in the function's finally block, so they cannot leak. unref is only safe when
another handle (a real fetch stream) keeps the loop alive.

## Pitfall: setInterval poll inside Promise.race leaks

Racing the fetch against a `setInterval` poll (checking `timer.current` every
10ms) leaks the interval when the fetch wins the race → the test runner hangs
forever at exit. Fix: the timer owns a rejectable promise
(`createRacePromise()`); `fire()` rejects it. Callers race
`[fetchPromise, timer.createRacePromise()]` and abandon the loser. Use the
SAME pattern for per-chunk iteration when a stream stalls (first-token / idle
gap): race `iterator.next()` against the active stage timer's rejection, and
call `iterator.return()` in a finally block to release the suspended
generator (a pending `next()` on an aborted stream would otherwise dangle).

## Pitfall: SSE usage field location

With `stream_options.include_usage`, the `usage` object arrives on the FINAL
chunk at TOP level — NOT inside `choices[0]`. Read `chunk.usage`, never
`choice.usage`, or reconstructed responses report `total_tokens: 0` and tests
asserting usage fail.

## Pitfall: forbidden-token static scans match COMMENT text

`test/openclaw-operator-cli.test.js` reads `src/services/readonly-operator-cli-client.js`
raw and asserts it contains NONE of: `child_process, exec(, spawn(, docker
exec, gh , git , PUT, PATCH, DELETE, /execute, fs.writeFile, appendFile,
createWriteStream, anthropic, ollama` — COMMENTS included. "**through** the
SHARED transport" contains "gh " (t-h-r-o-u-**gh** + space) → the security
scan fails. Fix: reword comments (e.g. "via the SHARED transport"). Before
committing provider/service changes, grep every changed file for these
tokens (including comment lines) — the scan is raw-file, not AST.

## Pitfall: legacy failure-code contract drift

Switching a provider path to the new transport changes OBSERVABLE failure
codes. Existing tests assert legacy codes (e.g. `INVALID_PROVIDER_JSON` in
`cloe-post-tool-raw-dsml-single-retry.test.js` T6). Fix: preserve legacy
codes on the buffered compatibility path (plain-JSON mocks →
`INVALID_PROVIDER_JSON`); use the new codes (e.g. `PROVIDER_STREAM_MALFORMED`)
only on the true-streaming path. Catch these with a FULL-suite run
before/after — bounded suites alone miss sibling-file contract assertions.

## Full-suite no-regression proof (baseline worktree diff)

1. Run the full suite on a CLEAN worktree at the base commit; save failure
   names (`grep '^not ok' log | sed 's/^not ok [0-9]* - //' | sort`).
2. Run the full suite on the mission worktree; same extraction.
3. `comm -13 <baseline> <mine>` = the TRUE regression candidates.
4. OOM-crashing test files (e.g. `readonly-operator-cli-client.test.js`,
   `openclaw-aionui.test.js` at the default ~2GB heap, SIGABRT) produce
   CASCADE differences: subtests that happened to run before/after the crash
   differ between runs. Classify those as pre-existing — verify the file
   OOM-crashes IDENTICALLY on the clean base standalone.
5. State-pollution tests (e.g. campaign-memory "duplicate_skipped") fail
   identically on a clean base standalone → pre-existing/ordering, not
   regression. Note they can PASS in the baseline full-suite yet FAIL in
   yours purely from cross-file data-store ordering.
6. Fix real regressions, re-run the bounded suites, re-run the full suite
   ONCE more, re-diff. Accept the bounded proof when the residual diff is
   only OOM-cascade + state pollution + pre-existing.

## Live certification harness

Against the DEPLOYED container, a streaming client that records per-chunk
arrival timestamps proves all LIVE criteria at once:
- first chunk before completion (time-to-first-token ≪ total duration)
- true progressive delivery (hundreds/thousands of content deltas, NOT one
  buffered burst)
- no PROVIDER_TIMEOUT while chunks arrive (total duration may exceed the
  former hard deadline — observed 66.9s vs former 35s, no timeout)
- no duplicate final output (exactly one finish chunk + one `[DONE]`)
- no business/provider writes (zero tool_call deltas, no campaign_id in the
  answer)

Extract the admin key to a 600-perm temp file (`docker exec ... > /tmp/key &&
chmod 600`), pass via env var, delete after — never print the value. Bounded
A-N suite with real-time 38s/72s stream proofs: `test/cloe-true-provider-streaming-timeout.test.js`.
