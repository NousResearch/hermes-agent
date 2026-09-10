# Codex Responses Source Replay Design

## Status and scope

This design repairs one security-sensitive transport defect in the protected
remote-worker path: an exact, locally validated `read_file` presentation can
be replayed through Chat messages, but the equivalent Codex Responses
`function_call_output` is unconditionally replaced by the read-file elision
marker. The repair extends the existing source-provenance contract to the
Responses input shape. It does not add a model tool, authorize terminal
stdout, parse arbitrary conflict output, or weaken path, secret, size,
request-identity, or policy gates.

The implementation branch starts from deployed runtime commit
`e0b848fed2f0bb446237c066eaa991e454096a43`. The completed worker-outcome
branch and PR #98205 are separate work and must remain unchanged.

## Goal

Allow a protected Codex Responses worker to receive the exact bounded JSON
presentation produced by a locally authorized `read_file` call when, and
only when, the existing content-free provenance envelope matches the current
request, tool call, source grant, policy, and presentation bytes.

This enables future LunaBot repair workers to inspect conflict files and
resolve semantic hunks without granting them broad terminal replay or
unbounded filesystem visibility.

## Existing authority

`agent/source_provenance_tools.py` already activates provenance only around
the trusted `read_file` implementation, issues grants for the resolved path
and requested line range, hashes the exact rendered JSON presentation, and
builds an internal content-free sidecar before provider conversion strips
private metadata.

`agent/llm_egress_runtime.py` already validates Chat-style tool messages with
`_segment_read_file_presentation()`. That validator checks the presentation
kind, exact content hash, request ID, grant digest, current request identity,
and the line-numbered rendering reconstructed from the granted source bytes.
The Responses repair must reuse this validator rather than create a second
authorization policy.

## Architecture

### Sidecar identity

Extend each sidecar record with a transport-independent result identity:

- exact provider-issued tool call ID;
- exact SHA-256 of the scalar JSON presentation;
- following API request ID;
- source grant digests;
- presentation kind `read_file_json_v1`.

The sidecar remains internal and content-free. It is removed from SDK kwargs
before provider serialization. No source bytes, path, line contents, secret,
or model-visible authority flag may enter it.

### Responses reattachment

Generalize `_restore_source_provenance_sidecar()` so it can reattach a copied
`_source_provenance` envelope to exactly one matching result in either:

- `messages[index]`, for the existing Chat path; or
- a Codex Responses `input` item whose type is `function_call_output`.

Responses matching is by exact call ID plus exact scalar presentation hash,
never by array position alone. A record that matches zero or more than one
candidate is ignored. Reattachment mutates only an internal copied body.

Responses tool output can be represented as a scalar string or as a
structured list of `input_text`/media items. Only a single scalar text
presentation, or a list that normalizes unambiguously to one text
presentation with no image/audio/file item, is eligible. Mixed or malformed
content stays elided.

### Typing and authorization

In `_typed_payload()`, a structured `read_file` Responses result with valid
reattached metadata must normalize to its exact scalar presentation and pass
through `_segment_read_file_presentation()`. The existing source grant and
request rebinding logic remains the sole authorization decision.

If metadata is missing, stale, forged, ambiguous, mismatched, or malformed,
the output remains `_READ_FILE_REPLAY_ELISION` or fails closed through the
existing untrusted-provenance decision. No fallback may treat raw output as
generated or sanitized context.

### Security invariants

1. The provider never receives `_hermes_source_provenance` or
   `_source_provenance` metadata.
2. Call ID, content hash, request ID, grant digest, and current request
   identity all bind the replay; position alone never authorizes it.
3. Only `read_file` is eligible. Terminal, search, mutation, scratch-file,
   web, and arbitrary tool output retain their current replay boundaries.
4. Existing secret, encoding, path, receipt, byte-cap, and source-grant checks
   run unchanged after reattachment.
5. One sidecar record authorizes at most one result, and one result consumes
   at most one unambiguous sidecar record.
6. Image, audio, file, multi-text, truncated, and malformed structured output
   cannot be normalized into trusted source.
7. Prompt history and provider-visible schemas remain byte-stable; this is a
   request-body authorization fix, not a prompt or toolset change.

## Error handling

The transport adapter fails closed. Invalid sidecar entries are ignored and
the corresponding read result is elided. It must not raise merely because an
old client omitted the new transport fields. A validly bound presentation
that later fails the source validator follows the existing
`untrusted_provenance` path and cannot be downgraded to ordinary text.

Audit receipts must continue to report only bounded reason codes and source
grant/segment counts; raw file contents never enter logs or durable receipts.

## Test strategy

Add behavior tests in `tests/agent/test_llm_egress_runtime.py` using the real
`source_provenance_activation()`, `read_file_tool()`,
`attach_trusted_source_provenance_metadata()`, Chat-to-Responses conversion
shape, and `authorize_agent_sdk_kwargs()`.

Positive cases:

- exact scalar Responses `function_call_output` survives with one source
  grant and one source presentation segment;
- the one-text-item structured Responses form normalizes to the same exact
  JSON presentation;
- Chat-style replay remains unchanged.

Fail-closed cases:

- missing, stale, forged, wrong-request, wrong-call-ID, wrong-content-hash,
  duplicate-match, and already-consumed provenance;
- altered line numbering, changed bytes, out-of-range grant, and policy
  digest mismatch;
- multiple text items, image/text mixtures, malformed mappings, truncated
  JSON, scratch-file calls, and absent metadata;
- terminal and arbitrary structured tool results remain outcome-only or
  elided exactly as before;
- internal sidecar keys are absent from final authorized SDK kwargs and
  receipts contain no raw source.

Run the focused egress runtime, source-provenance, file-tool, and protected
remote-Kanban suites. Run compilation and diff hygiene. A broad suite result
is reported separately and is never inferred from focused success.

## Acceptance canary

After tests and independent review, update the local runtime only through the
normal reviewed branch workflow. Restart the protected worker on the exact
committed source, then dispatch one explicitly authorized, no-push/no-merge
LunaBot conflict canary at an unchanged PR head.

Acceptance requires that the worker:

1. reproduces the merge conflict in an isolated pooled checkout;
2. receives exact bounded `read_file` presentations for every conflicted
   file it requests;
3. does not receive raw terminal output, unauthorized files, or secrets;
4. can describe and edit semantic conflict hunks from the granted source;
5. stops before push or merge and produces a typed durable result.

The canary does not prove receipt recovery, pool reclamation, detached-head
push, or LunaBot CI. Those remain separately scoped follow-ups.

## Non-goals

- No retry of the 73 failed repair receipts.
- No PR-feedback plugin reconciliation or installation mutation.
- No worktree-pool recreation or object-fetch repair.
- No detached-head push contract and no GitHub write.
- No automatic conflict resolution, broad `git` output replay, or
  `-X ours`/`-X theirs` strategy.
- No weakening of exact-head, CI, identity, review, security, or live-trading
  gates.
