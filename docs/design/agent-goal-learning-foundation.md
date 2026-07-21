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
not inside, recalled memory. Each receipt contains a goal SHA-256 digest,
terminal classification, current verification status/event, actor, confirmation
time, and `reusable` flag. It never stores the raw goal text.

| State | Reusable? | Reason |
| --- | --- | --- |
| `judge_done_unconfirmed` | No | Judge output is a candidate, not a success label. |
| `blocked` or `cancelled` | No | These are audit outcomes, not demonstrations. |
| `achieved_confirmed` + fresh `passed` verification | Yes | A human confirmed it and the workspace has current evidence. |
| `achieved_confirmed` + stale/failed/unverified evidence | No | Confirmation is recorded, but it is unsafe to reuse. |

`list_reusable_outcome_receipts()` is explicit pull-only retrieval. This change
does not alter memory prompts, background learning writes, or skill files.

When a `/goal` judge returns `done`, the shared `GoalManager` first persists
the goal terminal state, then records a `judge_done_unconfirmed` receipt for
the frontend's actual workspace. Hermes reports the receipt id but does not
promote it: the same active session in the same workspace must run
`/goal confirm <receipt-id>`. A receipt id from another session or workspace
is treated as unavailable. Retrieval rechecks the receipt's current workspace
evidence and excludes a receipt after a later edit by **any** session in that
workspace makes its supporting evidence stale. This workspace edit marker is
monotonic: a later verification can establish a new receipt, but cannot revive
an older receipt whose proof predates the edit. The read path does not update
receipt state, inject a prompt, or create a memory/skill artifact.

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

## Research basis and bounded decisions

The local ULR journal is under
`.omo/ultraresearch/20260721-183500-ai-agent-os/`; it records codebase and
independent-review evidence. The following primary sources informed the
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
