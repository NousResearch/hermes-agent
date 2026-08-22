# Idempotent User-Message Persistence at the Transcript Append Chokepoint

Established 2026-08-03 in CLOE_DASHBOARD_REPLY_SESSION_CONFLICT_REPAIR_V1
PHASE 2 (PR #668). Generalizes the "duplicate delivery persists two user
messages" class of defect in OpenClaw/CLOE.

## Root-cause shape (recognize this)

1. The dashboard SPA sends chat.send with `idempotencyKey: t.runId`.
2. The gateway in-memory dedupe (`chat:${runId}` cache + abort-controller map)
   is PROCESS-LOCAL and only catches SAME-runId duplicates.
3. `buildActiveChatSendDedupeKey` returns null when the originating channel is
   not "webchat" — so dashboard/internal sends get NO content-based gateway
   dedupe at all.
4. The runtime pipeline (attempt-execution `persistTextTurnTranscript`) and the
   get-reply recorder build user turns as `{role:"user", content, timestamp}`
   WITHOUT an idempotencyKey.
5. `appendSessionTranscriptMessageLocked` (session-accessor bundle) only dedupes
   when `message.idempotencyKey && idempotencyLookup === "scan"` — keyless
   appends always pass.

Live evidence signature: two identical `role:user` entries in the session
`.jsonl`, 668ms apart, BOTH with `idem=None` (no key persisted).

## Fix pattern (ESM loader patches on the dist bundles)

Two layers, applied at the SINGLE append chokepoint all user-turn appends flow
through (`appendSessionTranscriptMessageLocked` in `session-accessor-DvSc996e.js`):

### Layer 1 — PRIMARY key (durable, file-backed)
Stamp the SPA-provided key onto the runtime recorder input so the EXISTING
primary scan (`findTranscriptMessageByIdempotencyKey`, reads the transcript FILE)
dedupes same-key duplicates across reconnect, tabs, and gateway restarts.

Patch target in get-reply bundle (`userTurnInput` construction):
```
idempotencyKey: ctx.MessageSid.trim() + ":user"
```
`ctx.MessageSid` === `clientRunId` (the SPA runId) — set in the chat.send
handler. The transcript file IS the durable store: file-backed scan survives
restart, so no in-memory Map is needed for persistence dedup.

### Layer 2 — FALLBACK (bounded content dedup, keyless channels)
Only when the message has NO idempotencyKey AND `role === "user"`, dedupe by:
`agentId + sessionKey` (transcript file scope) + `normalizedBodyHash` +
`boundedTimeBucket` (default 30000ms, env `CLOE_DEDUP_FALLBACK_BUCKET_MS`).

- Reverse-scan the transcript file; stop at the first entry older than the
  bucket start (file is FIFO → chronological).
- MUST NOT merge: distinct bodies; same body with DIFFERENT keys (mission test 3);
  intentional identical sends outside the bucket (mission test 4).
- Assistant messages are never content-deduped (role guard).

## Design invariants (mission requirements to preserve)

- Historical duplicates in existing transcripts are NEVER deleted — only new
  appends are skipped.
- Single-flight reply init (phase 1) must remain untouched — dedup happens at
  persistence, BEFORE reply initialization.
- No UI-renderer-only dedup: the authoritative layer is the append path.

## Test harness shape (behavioral, no provider)

- Extract the REAL injected helper functions from the patched bundle and run
  them against REAL temp transcript JSONL files.
- Mirrors the patched decision sequence: primary scan → fallback scan → append.
- Covers: same-key simultaneous, same-key after reconnect, different keys,
  delayed intentional send, two tabs, process interruption (fresh harness,
  same file), single-flight regression, dashboard repro (greeting + long prompt
  + duplicate → exactly one prompt in file), non-regression (new session,
  resume, self-audit, Telegram, rapid distinct ordering).
- Watch the `new Function` factory traps: async-keyword strip breaks `for await`;
  `'\n'` in template literals becomes a real newline; module imports
  (`readFileSync`) are undefined in global factory scope — pass them as args.
  See behavioral-operator-simulation skill.

## Deploy

1. Merge loader changes via PR (canonical repo), then copy the loader file to
   the ACTIVE workspace clone (gateway loads from there via
   `NODE_OPTIONS=--experimental-loader=...`).
2. Bounded gateway restart.
3. Verify transcript shows exactly ONE user entry after a duplicate send.
