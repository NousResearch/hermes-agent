# Incident: verification/session interruption and compression churn

**Date:** 2026-07-10
**Status:** active root-cause investigation; a partial identity patch is deployed, but the end-to-end interruption is not resolved.
**Audience:** Hermes maintainers and an external audit/fix agent.

## Purpose

A coding task was repeatedly interrupted after it appeared to have completed. The user saw a system verification message interleaved into the session, even after focused tests passed. The same investigation also exposed a gateway process that compacted a session dozens of times while shutting down and performed unrelated skill writes.

This document is the authoritative continuity record for the incident. It separates observed facts from hypotheses. Do not call the issue fixed merely because a unit test passes or because a SQLite ledger row has been manually reconciled.

## User-visible symptom

After source edits and an apparently complete response, the session injected this class of message again:

```text
[System: You edited code in this turn, but the workspace does not have fresh
passing verification evidence yet.]
```

The message names the edited source files and instructs the agent to run pytest. It can arrive after the agent has already reported passing tests, interrupting the intended task flow. The user specifically observed that the task was not completely finished and suspected that an internal worker, CLI verification output, or a sub-agent-style process had interrupted progress.

## Scope and non-goals

In scope:

- Stable identity for edits, verification evidence, and stop/continue gates during a turn.
- Compression and background-review concurrency that can fork or rotate a session.
- Whether verification output is surfaced to the user when it should remain internal.
- Durable automated reproduction and an externally reviewed remediation.

Out of scope for the first fix:

- Disabling verification entirely.
- Treating manual SQLite evidence writes as a product solution.
- Broad refactoring of skills, memory, or gateway behavior unrelated to identity/turn ownership.
- Claiming the obsolete `post_turn` config warning is the primary root cause without evidence.

## Timeline and direct evidence

### A. Initial partial patch

Commit `8092c8b1820676535614ace1c4b1dd6e04eec651` at `2026-07-10 02:12:33 -0700`:

```text
fix: preserve verification identity across compression
```

Changed files:

```text
agent/agent_runtime_helpers.py
agent/conversation_loop.py
agent/tool_executor.py
agent/turn_context.py
tests/run_agent/test_run_agent.py
```

The patch added `agent._verification_session_id`, captured once in `build_turn_context`, and used it when dispatching generic tools and evaluating both the verification-stop and `pre_verify` gates.

The added test, `TestRunConversation.test_verification_session_survives_compression_rotation`, simulates a parent session changing to a child session before tool dispatch. It asserts that the mocked `handle_function_call` and verification-stop nudge receive `parent-session`.

### B. Gateway compression churn during restart

The old gateway process (`python[1401094]`) was stopped at `02:13:57 PDT` during the rollout. Between then and the service exit at `02:14:01 PDT`, journal evidence recorded repeated compression warnings in the same second:

```text
Session compressed 21 times ...
...
Session compressed 49 times ...
```

During the same sequence it logged unrelated self-improvement writes:

```text
Self-improvement review: Patched SKILL.md in skill 'sassy-memory-bridge'
Self-improvement review: Patched SKILL.md in skill 'sassy-codi-client-operations'
```

The old process then exited with status 1. Systemd started a replacement gateway at `02:14:01 PDT`; current receipt during investigation:

```text
MainPID=2657426
ActiveState=active
SubState=running
Result=success
```

The restart and compaction churn are temporally associated. The log does not prove that SIGTERM caused the loop, nor that the loop caused the restart; it does prove that uncontrolled repeated compression and unrelated writes occurred in the old live process.

### C. Verification ledger split and repeated system message

The SQLite evidence ledger at `~/.hermes/verification_evidence.db` recorded the Hermes worktree under multiple session IDs:

```text
20260709_192113_72e273  # notifier/legacy session identity
20260710_015909_c94071  # edit-tool session identity
20260709_193151_6f3128  # earlier evidence session identity
```

The current changed-file state first existed under `20260710_015909_c94071`, while the notifier continued to evaluate `20260709_192113_72e273`. This is direct evidence that the running CLI/harness had split identity ownership.

Focused suite runs returned success repeatedly:

```text
109 passed in 39.27s
109 passed in 46.38s
109 passed in 55.03s
```

However, the chat-host `functions.terminal` interface did not call the source repository's `tools/terminal_tool.py`, so those commands did not automatically write fresh evidence through the product path. Evidence events 7 and 8 were subsequently written with the real observed test outputs to both live identities solely to inspect the stop gate:

```text
Event 7: session 20260710_015909_c94071, pytest, passed, 109 passed in 46.38s
Event 8: session 20260709_192113_72e273, pytest, passed, 109 passed in 55.03s
```

For each identity, `verification_status(...)` returned `passed` and `build_verify_on_stop_nudge(...)` returned `None`.

**Important limitation:** the reconciliation proves ledger semantics for those inputs. It does not prove that a normal CLI/gateway terminal invocation, compression rotation, and final response use one identity end to end.

## Code-path map

```text
run_conversation
  -> build_turn_context
     -> captures agent._verification_session_id
     -> performs preflight setup/compression later in the prologue
  -> tool dispatch
     -> tool_executor / agent_runtime_helpers
     -> run_agent.handle_function_call(... session_id=verification owner)
     -> tools.terminal_tool.terminal_tool
        -> record_terminal_result(... session_id or task_id or effective_task_id)
  -> final response branch
     -> build_verify_on_stop_nudge(verification owner, changed paths)
     -> get_pre_verify_continue_message(verification owner, changed paths)
```

Relevant source locations at the investigation commit:

- `agent/turn_context.py:261-263` — captures `_verification_session_id`.
- `agent/tool_executor.py:1400-1407` and `1446-1453` — registry dispatch passes the captured identity.
- `agent/agent_runtime_helpers.py:2115-2122` — generic helper dispatch passes it.
- `agent/conversation_loop.py:5069-5075` and `5131-5139` — stop and `pre_verify` gates use it.
- `tools/terminal_tool.py:2707-2724` — terminal evidence records `session_id or task_id or effective_task_id or "default"`.
- `agent/verification_stop.py:245-310` — builds the exact synthetic `[System: ... fresh passing verification evidence ...]` message when status is not `passed`.
- `hermes_cli/plugins.py:2100-2148` — `pre_verify` hooks can independently return a synthetic continuation message.

## Proven facts

1. The observed verification message is produced by `build_verify_on_stop_nudge`; it is not ordinary subprocess stdout.
2. A normal source terminal-tool result is intended to call `record_terminal_result`.
3. The chat-host terminal interface used during this investigation bypassed the repository's source terminal tool, so it is not a valid production-path test of automatic evidence recording.
4. The old in-memory CLI/harness process used different identities for edit tracking and the stop notifier, even after the source patch was committed and gateway restarted.
5. The source test covers mocked generic tool dispatch and mocked stop-gate ownership after one simulated rotation. It does not exercise `tools.terminal_tool`, SQLite evidence, actual preflight compression, background review, or gateway session routing together.
6. `hermes_state.py:1914-1927` documents an existing race for two `AIAgent` instances that share a parent session ID. The current background-review implementation explicitly sets `review_agent.compression_enabled = False` (`agent/background_review.py:731-741`), so that fork is not a proven direct compressor in the normal current path. The race remains a guardrail/risk if another peer compresses or if that invariant regresses. A lock API exists to prevent it.
7. The gateway log demonstrated pathological repeated compression and self-improvement writes in a live process.
8. Live config contains an obsolete `hooks.post_turn` entry. Gateway logs correctly warn that `post_turn` is unknown. It is a cleanup item, but no evidence presently ties it to this incident.

## Root-cause hypotheses, ordered by evidence

### H1 — Incomplete propagation of the per-turn verification owner

**Evidence:** The patch only changes selected dispatch and stop-gate call sites. There remain identity-bearing middleware/hook boundaries and terminal fallback values (`task_id`, `effective_task_id`) that need a full data-flow audit.

**What would prove/disprove it:** A deterministic integration test that mutates live `agent.session_id` during preflight compression, performs a real source terminal-tool test command, and asserts exactly one parent-owned state/event and no verifier nudge.

### H2 — Concurrent compression can create orphan lineage when its locking invariant is unavailable

**Evidence:** The state-layer source explicitly describes this race, and the live gateway logged 49 compressions plus self-improvement behavior during the affected process lifetime. The normal current background-review fork sets `compression_enabled = False`, so it is not a demonstrated active contender; the observed skill writes are still possible because the fork is intentionally permitted to use skill and memory tools.

**What would prove/disprove it:** A deterministic two-peer compression test using the same state DB and session ID, including the lock-subsystem-unavailable branch. Assert that exactly one agent obtains the compression lock when available, one successor identity exists, and an unavailable lock does not permit unbounded retries or orphan lineage. Separately, assert that a cancelled/shutting-down review fork performs no skill/memory write.

### H3 — Synthetic verification continuation is leaking into user-visible output

**Evidence:** `conversation_loop.py` intends the verification-stop nudge to be internal/silent, but the user received it as a system message. This may be a CLI/harness transport behavior rather than the conversation-loop branch itself.

**What would prove/disprove it:** An end-to-end CLI and gateway transport test asserting a verifier nudge is fed back to the model but never emitted as user-visible progress/system content; a genuine final answer appears only after verification completes or the bounded retry policy ends.

### H4 — Old in-memory process confounds deployment verification

**Evidence:** The gateway was restarted and runs the commit. The current long-lived CLI/harness continued to exhibit legacy behavior because source imports are process-local. Manual ledger reconciliation was necessary to inspect the old process's gate.

**What would prove/disprove it:** Fresh-process CLI and gateway reproduction with PID/commit receipts, compared to the current long-lived session.

## Required external-agent audit: read-only first

Run from `/home/lfdm/.hermes/hermes-agent` in a fresh external harness. Do **not** edit source, config, skills, or SQLite during the first pass. Do not commit.

```text
You are auditing an active Hermes Agent incident. Read docs/incidents/2026-07-10-verification-session-interruption.md first. Perform root-cause investigation only; do not modify files.

Trace every identity from AIAgent.session_id through build_turn_context, preflight compression, tool execution middleware, run_agent.handle_function_call, terminal_tool.record_terminal_result, mark_workspace_edited, verification_status, verification-stop, pre_verify hooks, gateway routing, and background_review forks.

Answer with:
1. a source-backed identity-flow diagram naming every session/task/turn identifier;
2. the minimal root cause(s), ranked and explicitly separated from conjecture;
3. all call paths missed by commit 8092c8b182;
4. a smallest safe implementation slice, including exactly which tests must be RED first;
5. whether compression locks are actually acquired/released around every rotation;
6. how to prevent a cancelled/shutting-down process from looping compression or making self-improvement writes;
7. whether synthetic verification nudges can leak to CLI/gateway output;
8. any config cleanup that is safe but unrelated.

Do not propose a broad rewrite. Do not call the incident fixed. Cite file:line evidence for every claim.
```

Recommended invocation on this host (Codex sandbox is known to fail here):

```bash
codex exec --yolo '<paste the audit prompt above>'
```

Use `--yolo` only for this bounded read-only audit and verify the final result independently. Codex must not be given a prompt that authorizes edits in this first pass.

## Required automated reproduction suite

Before another implementation attempt, add focused tests that fail on the current incomplete behavior:

1. **Real terminal evidence under rotation**
   - Start a parent session.
   - Mark a verifiable file edited.
   - Rotate `agent.session_id` during the turn/preflight path.
   - Invoke the source `terminal_tool` with a successful canonical test command.
   - Assert one evidence event/state under the immutable owner; child/task IDs do not own the event; stop nudge is `None`.

2. **All dispatch paths**
   - Cover quiet sequential executor, normal sequential executor, generic runtime helper, and middleware/hook route.
   - Assert each receives the same immutable verification owner.

3. **Concurrent-compression safety**
   - Run two compression-capable peer contenders against one state DB/session ID (the normal review fork is compression-disabled).
   - Assert exactly one lock holder and one session rotation.
   - Assert no orphan session writes and bounded, safe behavior if the lock subsystem is unavailable.

4. **Shutdown cancellation**
   - Interrupt a compression/review path.
   - Assert no repeated compression loop, no automatic skill/memory writes after cancellation, lock cleanup, and bounded shutdown.

5. **Transport behavior**
   - In CLI and gateway test harnesses, assert synthetic verification nudges do not render to the user as a response or progress message.

## Acceptance criteria for a future fix

A fix is not complete until all are true:

- One immutable verification-owner ID exists per turn and every relevant writer/checker uses it.
- A real terminal invocation records passing evidence under that same ID after a compression rotation.
- No fallback silently substitutes `task_id` for a session-owned verification event without an explicit, tested reason.
- Parent/review concurrency cannot produce sibling/orphan rotations or mutate the user's active session.
- Cancellation/shutdown stops compression/review work before any further writes.
- Verification nudges remain internal and bounded; they cannot interrupt a user-visible completed task as raw system output.
- Fresh CLI and fresh gateway process tests pass, not only an already-running process or manually reconciled SQLite state.
- The full relevant test suite and focused end-to-end tests pass with no manual ledger edits.

## Current operational posture

- Gateway: fresh process on commit `8092c8b182`, active after restart.
- Current old CLI/harness session: not trustworthy for rollout verification; it retained pre-patch runtime behavior and split state ownership.
- Do not apply a third in-process identity patch before an independent audit identifies the complete ownership boundary.
- The next implementation should be done in a fresh external harness/worktree, commit only the smallest verified slice, then roll out with a fresh-process receipt.

## Independent external audit receipt (verified)

On 2026-07-10, `codex-cli 0.140.0` ran a read-only audit in a fresh ephemeral session (`gpt-5.4`, `sandbox=read-only`). Its final report was preserved at `/tmp/hermes-verification-interruption-codex-audit-20260710.md`; the repository remained clean after the audit.

The following audit findings were independently checked in source:

1. **Confirmed identity miss:** `agent/agent_runtime_helpers.py:2135-2145` invokes `run_tool_execution_middleware` with mutable `agent.session_id`, while the nested registry dispatch uses `_verification_session_id`. This is a real split boundary missed by `8092c8b182`.
2. **Confirmed cancellation miss:** `agent/turn_finalizer.py:470-480` only prevents review when the completed turn is already marked interrupted. A process shutdown can occur after that decision. `agent/background_review.py:677-741` disables transcript persistence and compression but deliberately permits skills/memory tool writes; the daemon review thread has no parent cancellation token. This explains the observed self-improvement writes during the old process shutdown.
3. **Confirmed compression caveat:** `agent/conversation_compression.py:487-529` intentionally proceeds without a lock when the lock subsystem is missing or stale in memory. It is a bounded trade-off against a no-progress retry loop, not a proof that the normal review fork caused the incident.
4. **Not confirmed as a product leak:** `agent/conversation_loop.py:5069-5150` appends verification nudges as synthetic model turns. The audit found no standard CLI/gateway emitter for those flags. Raw `[System: ...]` appearance therefore remains a harness/transport hypothesis and requires an end-to-end test rather than a speculative code change.

### External implementation contract

The next coding pass must remain narrow and test-driven:

1. Add failing regressions for the execution-middleware identity, real file/terminal evidence after a rotation, and shutdown cancellation of background review.
2. Pass the immutable verification owner to execution middleware as well as nested dispatch. Preserve explicit compatibility semantics for any genuine task-scoped call; do not silently change unrelated task IDs.
3. Introduce a cancellation/shutdown signal shared with background-review work, check it before review spawn and immediately before any review skill/memory write, and prevent new compression work/rotation after interruption while always releasing any acquired lock.
4. Do not remove verification, weaken freshness, or make the `post_turn` cleanup part of the behavioral fix.
5. Run focused tests, the relevant wider suite, and a fresh-process verification before service rollout. No manual ledger reconciliation may be used as acceptance evidence.
