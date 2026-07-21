# Durable goal execution and verified outcome learning

## Purpose

This milestone extends Hermes with two deliberately narrow foundations for an
agent operating system:

1. a durable, same-session continuation after `/goal wait`; and
2. a verified outcome receipt that can become reusable only after explicit
   human confirmation.

It is not a new generic task database, automatic self-modification system, or
memory-injection mechanism. Hermes already has a Kanban task/event store,
session routing, verification evidence, and a goal state record. Duplicating
those ownership domains would create conflicting recovery semantics.

## Goal wait delivery contract

`GoalState` persists the wait barrier plus a small delivery obligation:

```text
waiting → pending → claimed (lease) → next same-session turn → idle
```

- A PID/session trigger or elapsed deadline clears the barrier and writes
  `pending` before a frontend queue is touched.
- `SessionDB.compare_and_set_meta()` makes the `pending → claimed` transition
  atomic across a CLI and a gateway sharing the same profile database.
- The CLI polls while idle after process notifications. The gateway polls
  persisted session routes, rechecks authorization, reserves its existing
  running slot, and dispatches a normal synthetic event for the original
  session.
- `/goal unwait`, pause, resume, clear, and a normal follow-up goal evaluation
  cancel or acknowledge the obligation.

The delivery contract is **at-least-once**, not falsely advertised as exactly
once. If Hermes dies after an external model/tool turn begins but before its
acknowledgement is recorded, the expired lease may replay the continuation.
The continuation still passes the existing active-goal check, and normal
agent-side tool receipts/idempotency remain the boundary for external effects.

Cron is intentionally not used: its one-shot jobs run a separate `cron_*`
session and its execution ledger classifies owner loss for audit rather than
restarting this conversation.

## Verified outcome receipts

`verification_evidence.db` now stores append-only `outcome_receipts` beside,
not inside, recalled memory. Each receipt contains a goal SHA-256 digest, a
versioned digest of the final completion contract and ordered subgoals,
terminal classification, current verification status/event, actor, confirmation
time, and `reusable` flag. It never stores raw goal, contract, or subgoal text.

| State | Reusable? | Reason |
| --- | --- | --- |
| `judge_done_unconfirmed` | No | Judge output is a candidate, not a success label. |
| `blocked` or `cancelled` | No | These are audit outcomes, not demonstrations. |
| `achieved_confirmed` + fresh `passed` verification | Yes | A human confirmed it and the workspace has current evidence. |
| `achieved_confirmed` + stale/failed/unverified evidence | No | Confirmation is recorded, but it is unsafe to reuse. |

`list_reusable_outcome_receipts()` is explicit pull-only retrieval. This change
does not alter memory prompts, background learning writes, or skill files.

`/goal outcomes` (with `/goal learning` as an alias) exposes that pull-only
view on CLI, gateway, and TUI. It is restricted to the active session and
current workspace, presents only receipts with currently passing evidence, and
shows a short criteria-digest prefix for newly recorded receipts. It does not
disclose raw goal text, mutate a receipt, or inject a model prompt.

`/goal learn <receipt-id> <lesson>` is the explicit, user-authored bridge from
one such candidate to procedural memory. It always stages a V3 memory proposal
instead of writing directly, even when ordinary memory approval is disabled.
The proposal carries only the receipt id plus its session/workspace scope and a
locked memory revision; its approval replay reserves the evidence ledger while
it re-checks eligibility and applies the reviewed memory revision, so a
workspace edit cannot land between those operations. The immutable approval
receipt records the numeric outcome reference for redacted lineage, never the
goal, contract, or lesson text.

When a `/goal` judge returns `done`, the shared `GoalManager` first persists
the goal terminal state, then records a `judge_done_unconfirmed` receipt for
the frontend's actual workspace. Hermes reports the receipt id but does not
promote it: the same active session in the same workspace must run
`/goal confirm <receipt-id>`. A receipt id from another session or workspace
is treated as unavailable. Retrying the same confirmation from its owning
session/workspace is idempotent: Hermes returns the first confirmation without
rewriting its actor, timestamp, or verification snapshot. Retrieval rechecks the receipt's current workspace
evidence and excludes a receipt after a later edit by **any** session in that
workspace makes its supporting evidence stale. This workspace edit marker is
monotonic: a later verification can establish a new receipt, but cannot revive
an older receipt whose proof predates the edit. The read path does not update
receipt state, inject a prompt, or create a memory/skill artifact.

Every attempted foreground terminal command records the same workspace
freshness marker before its normal terminal result is classified, and before a
timeout or final execution error is returned. This is deliberately
conservative: a formatter, script, or version-control command can write files
without using a Hermes file tool, and shell syntax is not a trustworthy way to
infer write intent. A later recognized verification command records fresh
evidence; background-process provenance remains outside this milestone.

## Immutable approval-decision receipts

Pending memory and skill changes have a separate, profile-scoped audit trail in
the same `verification_evidence.db`. `approval_decision_receipts` is append
only: database triggers reject every update and delete, while a unique proposal
digest makes a terminalization retry return the first receipt rather than add a
second decision.

Each row retains only the pending subsystem/id, canonical proposal digest,
staged origin, decision, terminal outcome, safe terminal-noop code, and time.
It deliberately stores no proposal payload, summary, user id, session id, or
raw content. The shared approval handler does not receive a trustworthy actor
identity on every CLI, gateway, TUI, and desktop route, so profile scope is the
honest attribution boundary for this milestone.

An approval receipt records an authorization decision, not demonstrated task
success. It never changes `GoalState`, verification evidence, an outcome
receipt's reusable flag, memory, or skills. In particular, it cannot promote a
goal-learning candidate. A receipt is written only after a claimed proposal is
known to be terminal: applied, explicitly rejected, or rejected as a stale or
invalid memory proposal. Retryable application failures are requeued without a
receipt. If persistence fails after a terminal mutation, Hermes holds the
private non-actionable claim and reports its path for manual reconciliation;
automatic replay is prohibited.

Operators can inspect this narrow audit boundary with `/memory receipts` or
`/skills receipts` (the `receipt` and `history` aliases are equivalent). The
shared CLI and gateway handler exposes only receipt id, pending id, terminal
decision/outcome, trusted proposal origin, safe terminal-noop code, and time.
It neither reveals proposal content nor changes pending, goal, learning, or
receipt state. The response explicitly states that it is read-only and cannot
replay a terminal decision.

## Manual wait controls

The same durable wait state is available on CLI, gateway, and TUI without new
tools or schema:

```text
/goal wait <pid> [reason]
/goal wait session <process-session-id> [reason]
/goal wait for <seconds> [reason]
```

The session form releases on the registered process trigger or exit; the time
form releases at its deadline. All three reuse the existing persisted barrier
and CAS-backed continuation flow.

## Approved learning proposal freshness

Memory proposals are an approval boundary, not an automatic bridge from a
verified outcome into prompt memory.  A newly staged memory proposal carries a
versioned, canonical record digest and a target-bound raw-byte revision of
`MEMORY.md` or `USER.md`.  At approval, Hermes claims the record, verifies the
review record, then compares the revision while holding the same target lock
used for the mutation.  A changed, malformed, or legacy proposal is not
applied and is terminally removed from the actionable queue; ordinary write
errors still retain the existing release-and-retry behavior.

The rule is deliberately strict: two proposals for the same target that were
staged against the same old revision cannot both be applied by a later
`approve all`.  After the first changes that target, the next proposal must be
restaged against the current reviewed state.  This avoids silently applying a
model-authored learning change that no longer matches the user-reviewed file.

An approved staged skill proposal also replays with its original trusted
`foreground` or `background_review` provenance.  The existing ownership,
pin, bundled, and external-skill protections therefore remain active at the
action sink.  This does not claim that skill review diffs are immutable:
action-specific snapshots and conditional writes are intentionally separate
future design work.

## Research basis and bounded decisions

The local ULR journals are under `.omo/ultraresearch/`; they record codebase
and independent-review evidence. The following primary sources informed the
boundaries rather than being copied into an implementation wholesale:

- SQLite's [atomic commit design](https://sqlite.org/atomiccommit.html) and
  [WAL documentation](https://www.sqlite.org/wal.html) support a small
  transactionally claimed local record.
- Temporal's [durable execution architecture](https://github.com/temporalio/temporal/blob/main/docs/architecture/README.md)
  and Dapr's [activity lifecycle](https://docs.dapr.io/contributing/protocol-reference/workflow-protocol/workflow-protocol-activity-lifecycle/)
  distinguish durable intent from non-transactional external effects.
- LangGraph's [persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
  and [interrupt](https://docs.langchain.com/oss/python/langgraph/interrupts)
  guidance supports checkpointing an interrupt rather than assuming an
  in-memory queue survives.
- The [MCP authorization specification](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization)
  and [OWASP guidance for agentic systems](https://genai.owasp.org/download/52117/?tmstv=1765059207)
  support retaining authorization and capability checks on synthetic work.
- OpenAI Agents' [human-in-the-loop](https://openai.github.io/openai-agents-python/human_in_the_loop/)
  and [tracing](https://openai.github.io/openai-agents-python/tracing/)
  guidance supports retaining approval/evidence boundaries rather than
  treating a model verdict as autonomous learning.

## Verification

The implementation has focused tests for deadline persistence, duplicate claim
suppression, expired-lease recovery, explicit cancellation, CLI idle queueing,
gateway same-session dispatch, and confirmed outcome receipt gating. The final
commit records the exact command and result in its message/history.
