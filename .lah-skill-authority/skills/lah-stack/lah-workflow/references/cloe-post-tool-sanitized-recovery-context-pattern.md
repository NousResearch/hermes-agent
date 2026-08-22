# CLOE Post-Tool Sanitized Recovery Context Pattern

Established 2026-08-07 on HERMES_CLOE_POST_TOOL_SANITIZED_RECOVERY_CONTEXT_V1
(openclaw-runtime lah-openclaw-mvp, deploy 0908681).

## When to use

Any CLOE mission where the single bounded protocol-recovery retry (raw DSML on
post-tool rounds) fails with `RECOVERY_RAW_TOOL_MARKUP` even though tools are
disabled and `tool_choice=none`. Root cause is NOT the timeout, NOT the retry
count, NOT `tool_choice` — it is the **context shape**:

> `CLOE_POST_TOOL_RECOVERY_CONTEXT_RETRIGGERS_PROVIDER_TOOL_PROTOCOL_V1`:
> DeepSeek is reconditioned by the native post-tool transcript structure
> itself (assistant.tool_calls + role:tool + tool_call_id). Reusing that shape
> in the recovery call re-triggers DSML generation.

## Diagnosis that proved it (V5 A/B/C)

Run the exact failing transcript through three recovery shapes:

| Case | Recovery context | Result |
|------|------------------|--------|
| A | native transcript + directive PREFIXED | HTTP 200, ~2.4s, markup again |
| B | native transcript + directive at END | HTTP 200, ~2.0s, markup again |
| C | SANITIZED evidence context | HTTP 200, ~58s, clean, 0 tool_calls |

Only C produces `clean_natural_language_candidate = true`.

## The sanitizer design (rules A–F)

Implement `buildProtocolRecoveryMessages(messages)` in
`src/services/chat-completions-service.js`:

- **A. Preserve user authority** — keep relevant user messages verbatim,
  especially the latest request. Never rewrite the business task.
- **B. Preserve safe grounding** — keep ordinary system/context messages
  (temporal authority, retrieval grounding, governance, read-only
  constraints). Never preserve tool invocation syntax.
- **C. Remove native tool protocol** — drop assistant messages whose purpose
  is tool invocation (`role === 'assistant' && Array.isArray(m.tool_calls) &&
  length > 0`), drop `role: 'tool'`, `tool_call_id`, function envelopes, DSML.
- **D. Convert tool results to textual evidence** — build one system message:
  ```
  COLLECTED READ-ONLY EVIDENCE
  Evidence 1 — <tool name>
  <tool result text>
  ```
  Tool name is resolved from the transcript's OWN `assistant.tool_calls`
  (tool_call_id → function.name map) — never invented.
- **E. Never invent evidence** — only transform messages already present.
- **F. Final synthesis directive** — appended as the LAST message:
  "Produce the final user-facing answer from the collected evidence only.
  Do not call tools. Do not emit DSML, XML-like tool markup, function
  invocation syntax, tool_calls structures, internal protocol instructions,
  or internal reasoning. Return ordinary natural-language text only."

Insert the evidence block at the position where the first `role:tool` was seen
(splice index), then append the directive. Return a NEW array — never mutate
the caller's.

## Wiring (what to change / not change)

- Replace `injectProtocolRecoveryDirective(effectivePayloadMessages)` with
  `buildProtocolRecoveryMessages(effectivePayloadMessages)` — that is the ONLY
  change to the recovery path.
- KEEP `MAX_PROTOCOL_RECOVERY_RETRIES = 1` — no loops, no recursion.
- KEEP timeout policy: standard 35000 ms, post-tool 70000 ms. Do NOT raise
  preemptively; the sanitized path completes well under 70s (~3-58s observed).
- Recovery payload: `tools: undefined`, `tool_choice: 'none'`,
  `parallel_tool_calls: false`, `stream: false`.

## Observability to add

`protocol_recovery_context_mode: 'sanitized_evidence'`,
`protocol_recovery_original_tool_messages_count` (role:tool count in source),
`protocol_recovery_sanitized_evidence_count` (Evidence N lines),
`original_raw_tool_markup_detected: true`. Never log keys/prompts/raw markup.

## Test shape (20 requirements T1–T20)

Split into: clean-path regression (no recovery), sanitizer unit tests
(NO assistant.tool_calls / role:tool / tool_call_id keys — assert on JSON keys
`/"tool_calls"\s*:/` NOT substring, because the directive TEXT mentions
"tool_calls structures"), evidence preservation, latest-user-preserved,
no-invention, recovery payload shape, clean recovery success, DSML-again →
exactly one retry + fallback, recovery tool_calls → reject, timeout, HTTP/JSON
failure, governance/budget/retrieval/timeout unchanged, no DSML in output,
`MAX_PROTOCOL_RECOVERY_RETRIES === 1` (prove no 3rd call even when a clean
3rd response is queued in the mock).

## Regression fixture from the failing transcript

The mission REQUIRES a fixture built from the real session
(`~/.openclaw/agents/cloe-poc/sessions/<id>.jsonl`): original user message +
tool calls + tool results. Do NOT re-execute tools. If the operator blocks
reading the transcript, build a shape-exact fixture from the documented
failure chain (tool names, round types, markup kind, timeout) — the swap is
1:1 and assertions unchanged, but the strict verdict stays PARTIAL until the
real-transcript fixture passes.

## Real-provider proof harness

Mock the FIRST call to return the raw obfuscated DSML (the production
failure), let subsequent calls hit the real DeepSeek. Assert: exactly 2 calls,
`protocol_recovery_context_mode === 'sanitized_evidence'`, success, no
fallback, `effective_timeout_ms === 70000`, 3 original tool messages → 3
evidence entries, answer > 80 chars, `detectRawToolControlMarkup(answer) ===
false`. Run from the DEPLOYED worktree (same SHA as the container) for the
post-deploy cert.

## Post-deploy certification (A–G)

health 200; standard response unchanged; native tool lifecycle unchanged;
known failing transcript → sanitized recovery + clean synthesis + no fallback;
repeated bad recovery → exactly one retry + fallback preserved; governance
(mutation fail-closed, budgets, retrieval-first, ExoClick safety); real
authority-reconciliation replay completes without protocol failure (2 rounds:
tools → post-tool synthesis, 0 DSML, 0 protocol_failure, recovery_loop_count
≤ 1). Reuse the existing `docs/evidence/cloe-live-certification/
cloe-live-smoke-harness.mjs` A–H harness for the live scenarios — note its
scenario-G verdict false-positives when Cloé's defensive answer CITES
`<tool_call>` textually (0 executions, 0 violations — that's a verifier regex
hit, not DSML leakage).

## Deployment

Canonical exact-SHA deployer (`bin/deploy-lah-openclaw-mvp-exact-sha.mjs
<FULL_SHA>`), tracked on main, workdir = clean worktree at the target SHA.
Build with LITERAL `--build-arg GIT_COMMIT=<full-40-hex>` (never
`GIT_COMMIT=$GIT_COMMIT` — shell-expansion trap), verify image ENV
immediately, tag `rollout-<sha>` before anything else. Expect
`DEPLOYED_EXACT_SHA` then idempotent `ALREADY_DEPLOYED`.
