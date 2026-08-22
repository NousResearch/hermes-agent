# Exact-Transcript Fixture Derivation Pattern

Use when a mission requires proving a regression against the EXACT content of a
retained production transcript (session JSONL), not just a shape-equivalent
fixture. Established during HERMES_CLOE_POST_TOOL_SANITIZED_RECOVERY_CONTEXT_V1
closure (PR #717 code + PR #718 exact fixture, 2026-08-07).

## OpenClaw session JSONL schema (session `bec14190-…` example)

Each line is an object with `type`:
- `session`, `model_change`, `thinking_level_change`, `custom` — metadata, skip
- `message` — real conversation; `obj.message` carries the payload

`message.role` values:
- `user` — `content` is a plain string
- `assistant` — `content` is an ARRAY of items:
  - `{type: 'toolCall', id, name, arguments, partialArgs}` — native tool call
    (NOT OpenAI-style `tool_calls`; `arguments` may be an object OR a JSON string)
  - `{type: 'text', text}` — ordinary reply
- `toolResult` — THE TRAP: OpenClaw uses `toolResult` (10 chars), NOT `tool`.
  A filter on `role === 'tool'` silently finds 0 results. Fields:
  `toolCallId`, `toolName`, `content` (array of `{type:'text', text}`),
  `isError`, `details`.

## Extraction pipeline (never dump the transcript)

1. **Structural inspect first**: per-line type/role/shape + counts only, no
   content (python3 helper with a `safe_shape()` that returns
   `{type, keys, str(len)}` descriptors).
2. **Extract minimal fields** to a working JSON: user content, assistant
   toolCall items (id/name/arguments), toolResult items (tool_call_id,
   tool_name, content text), original ordering.
3. **Verify ordering**: `call_ids == result_ids` — tool results must reference
   the same call ids in order; fail the build otherwise.
4. **Secret scan** on ALL extracted content before committing the fixture
   (sk-, ghp_/gho_, xox, Bearer, api_key, password, private key, AKIA,
   EXOCLICK_API_TOKEN/LIVE_ENABLED patterns). 0 hits required.
5. **Build the 1:1 fixture** in OpenAI message format:
   - user: verbatim
   - assistant: `{role:'assistant', content:null, tool_calls:[{id,
     type:'function', function:{name, arguments}}]}` — serialize `arguments`
     to a JSON string if it was an object
   - tool results: `{role:'tool', tool_call_id, content}` in original order

## Pitfalls

- **toolResult vs tool**: role is `toolResult`; filter on `('tool','toolResult')`.
- **toolCall accumulation bug**: `native_tool_calls = []` assigned INSIDE the
  content-item loop is overwritten per item — initialize ONCE before the loop,
  append inside. Symptom: counts=3 but only 1 call captured.
- **Shape-equivalent ≠ exact**: a spec-verbatim fixture is a valid regression
  but does NOT prove content-exactness. State BOTH explicitly in the final
  receipt ("previously proven shape-equivalent" vs "newly proven
  exact-transcript-derived"). Never claim the earlier fixture was exact.
- **Assertion fragments must match REAL content**: when supplementing an
  existing shape-equivalent test with an exact fixture, verify each assertion
  fragment (user head/tail, evidence markers like `camp-ourdream`,
  `AGENTS.md`) exists in the REAL transcript content BEFORE writing the test.
  The shape-equivalent fixture's invented evidence strings (e.g.
  `CAMP-CONV PLANNED`) will NOT be present — assert their ABSENCE to prove no
  evidence was invented.
- **User message integrity**: assert byte-identical preservation
  (`assert.equal(users[0], fixtureUser)`), not just substring.

## Closure mission gate shape (transcript-proof closure)

Gate 0 read (report transcript_read, user_message_found, tool_calls_count,
tool_results_count, ordering_consistent, exact_fixture_constructible) →
Gate 1 build fixture → Gate 2 focused regression (+ directly related suites) →
Gate 3 verify deployed authority read-only (origin/main vs image GIT_COMMIT,
health, restart count, EXOCLICK_LIVE_ENABLED=false, recovery code present,
retry bound, timeout policy; if origin/main advanced, classify code-affecting
vs docs-only; NO redeploy for test-only deltas) → Gate 4 closure evidence
(receipt + continuity JSON + operator packet + memory lock).

Verdict CERTIFIED only if all pass and no contradiction with prior
real-provider/live proofs; do NOT create a follow-up repair mission when
closure gates pass.
