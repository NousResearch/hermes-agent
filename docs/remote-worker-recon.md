# Remote worker board access — implementation recon

Grounded in the code at base `0aff0259bc` (branch `task/remote-worker-spawn`).

## The write seam is genuinely single

`kanban_db_connect.write_txn()` is the only write path. 166 call sites, every one
`BEGIN IMMEDIATE`. `kanban_db_connect.connect()` is the connection entry point.

## The read bypasses — all read-only, all fail-closed

| site | mode | note |
|---|---|---|
| `agent/kanban_turn_recovery.py:223` | `?mode=ro` | the bypass the review flagged |
| `hermes_cli/kanban_db_connect.py:728` | `?mode=ro` | fenced-descendant read path |
| `hermes_cli/kanban_db_notify.py:242` | `?mode=ro` | notification ID cursor |
| `hermes_cli/kanban_db_workspace.py:158` | `uri=True, timeout=1.0` | sibling-board path check |
| `hermes_cli/kanban_transfer.py:65,66,142` | direct | export/import only |

**No module opens the board read-write outside `kanban_db_connect`.** That makes
the interception story tractable: one write seam, and a small number of read
seams that are already read-only.

On a remote host the board file does not exist, so the read paths fail closed by
design — `kanban_turn_recovery` logs *"pinned board db … not found — not retrying
in place"* and returns `False`. Nothing is corrupted; the worker simply has no
local board, which is the intended state. It does mean the worker must obtain its
task context **through the client**, not by reading anything local.

## Domain functions the client must reach (owner-side)

| function | location | signature highlights |
|---|---|---|
| `claim_task` | `kanban_db.py:2031` | the CAS |
| `complete_task` | `kanban_db.py:2493` | `expected_run_id=` is the ownership proof; `force=True` is the human override to NOT use |
| `block_task` | `kanban_db.py:2995` | |
| `add_comment` | `kanban_db.py:1522` | `(conn, task_id, author, body)` |
| `request_changes` | `kanban_db.py:3282` | reads the run's `claimed` + prior `review_requested` events |

`complete_task` already carries the ownership proof the review demanded
(`expected_run_id`), and `LiveClaimError` is its fence. The client should pass the
worker's run id through and let the owner's existing CAS decide — never `force`.

## Tools that must route through the client

15 handlers in `tools/kanban_tools.py`: `show`, `list`, `complete`, `block`,
`schedule`, `request_review`, `request_changes`, `heartbeat`, `comment`, `attach`,
`attach_url`, `attachments`, `create`, `unblock`, `link`.

The review established that `create`/`link`/cross-task `comment` are legitimate
worker operations, so the client surface cannot be T-only.

## Worker identity carrier

Already defined by the dispatcher and required by lifecycle tools:

- `HERMES_KANBAN_TASK`
- `HERMES_KANBAN_RUN_ID`
- `HERMES_KANBAN_CLAIM_LOCK`

`agent/kanban_turn_recovery.py` reads exactly these three and refuses to act when
any is missing. The client should carry the same triple, so the owner verifies a
request belongs to run R of task T rather than trusting a bare task id.

## Two findings that constrain the liveness model

**1. The worker writes to the board at startup.** `adopt_worker_pid`
(`kanban_db_dispatch.py:1514`) is the "worker-side half of `_set_worker_pid`, run
by the worker before its first model call" — a WRITE, and its docstring explains
why: a dispatcher killed between spawning the worker and `_set_worker_pid` leaves
the run with no pid, so no liveness check can see the worker and a TTL expiry
reclaims the card and spawns a second worker beside it.

Its own comment states the problem for us directly:

> Liveness checks are host-local: a pid from another host (or pid namespace)
> proves nothing here.

So a remote worker's PID is **meaningless to the owner**. The whole
pid-based liveness model (`_process_fingerprint`, dead-pid reclamation,
`worker_started_at`) does not transfer. The client must supply an equivalent
liveness signal — and the review already flagged that a live SSH process can
*both* mask remote failure *and* create false failure, so "SSH child alive" is
not by itself a sufficient answer.

**2. The heartbeat is already a side effect of API traffic.**
`kanban_db.py:337`: *"`_touch_activity` bridges chunk-level liveness into
`last_heartbeat_at` via #31752, so any genuinely active worker keeps its heartbeat
fresh as a side effect of normal API traffic."*

That is a gift for this design: if **every** worker board access is routed through
the client, the owner touches activity on each request and the heartbeat stays
fresh with no separate heartbeat protocol to invent. It also means the failure
mode is informative — a remote worker that stops making requests genuinely stops
being live, which is the property we want.

## Open — awaiting the transport decision

Transport (dedicated HTTP service / SSH-exec per op / persistent SSH stdio /
extend the webhook server) is with astra for a decisive call before any code is
written. The recon above is transport-independent.

