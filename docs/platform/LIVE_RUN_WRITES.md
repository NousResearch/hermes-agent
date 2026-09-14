# Live run — the control plane's write path

An end-to-end run of Phase 8 against a real board: real profiles, real `kanban.db`, the real
HTTP transport, two named principals. **Core patches: still 1.** Two defects found, both in
NOVA, both fixed and regression-tested.

The pattern from [`LIVE_RUN.md`](LIVE_RUN.md) repeated exactly: every defect was in the
**seam**, and no unit test could have found either — because the unit tests used a stub
runtime, and a stub returns whatever the stub was written to return.

---

## Setup

```bash
export HERMES_HOME=/tmp/live8 NOVA_HOME=/tmp/live8
python -m nova apply nova/examples/acme
python -m nova objective submit nova/examples/acme quarterly-refund-audit
python -m nova serve nova/examples/acme --port 8893
```

The objective lands as four real tasks with real dependency gating — two `ready`, two `todo`
because they wait on the first two:

```
t_3a601a2d  ready   Extract last quarter's refunds
t_67973cce  ready   Restate the current approval thresholds
t_1d158137  todo    Find refunds that missed their approval
t_c56705a2  todo    Write the audit findings
```

No model provider exists in this environment, so no worker consumes the board. That is fine
for this phase: the question is whether a **human decision** reaches the runtime's real state
machine and is recorded against that human, and every part of that is exercisable without a
model.

---

## What worked first time

**The runtime refused an unsafe promotion, and explained itself.** Releasing a step whose
parents are unfinished:

```
HTTP 409
{ "action": "release", "applied": false,
  "reason": "unsatisfied parent dependencies: t_3a601a2d, t_67973cce (the ready -> running
             claim re-checks parents, so promotion cannot bypass them; complete the parents
             or drop the link with `hermes kanban unlink <parent_id> t_1d158137`)",
  "resulting_status": "todo" }
```

This is the design argument made concrete. NOVA asked for a transition; the runtime — which
owns the accounting — said no and said why, in a sentence nobody wrote into NOVA. A control
plane that set `status = "ready"` itself would have silently produced a task that runs before
the work it depends on.

**A note reached the board and the audit.**

```
board:  [local] OPERATOR: Use the Q3 ledger export, not the live table — the live table
                was re-keyed on 1 Sep.
audit:  work.annotated  intent     model_visible=True
        work.annotated  committed  model_visible=True
```

**`resume` moved a blocked step.** blocked → ready, `200`.

**An empty rejection was refused before it reached the runtime**, with the reason the
refusal exists: *"a rejection needs a reason — it is what the worker reads and acts on."*

---

## Defect 1 — `reject` used the wrong primitive

```
HTTP 409
{ "action": "reject", "applied": false,
  "reason": "task is not in an active review run", "resulting_status": "review" }
```

The task **was** in `review`. The call was wrong.

`request_changes` closes an *active reviewer run*: a reviewer agent claimed the task out of
`review` and is handing it back with changes. A human looking at a queue of tasks awaiting
review has claimed nothing, so there is no run to close.

Two rejections wear the same word, and only a live board shows it. The operator's rejection
is `reopen_review_task` plus a comment carrying the reason — which is exactly what the
runtime's own operator command does, and which I would have found by reading
`_cmd_reopen_review` if I had thought to look for a *second* implementation of the same word.

*Fixed:* try the reviewer path first, so a genuine reviewer run is still closed properly;
fall back to the operator path rather than reporting a refusal to a human who did nothing
wrong. The reason goes through the runtime's own `redact_review_value` on the way — a
rejection reason is durable, a human wrote it in a hurry, and secrets end up in exactly that
kind of text.

After:

```
HTTP 200
{ "action": "reject", "applied": true, "resulting_status": "ready", "actor": "local" }

board:  [local] CHANGES REQUESTED: Refunds under GBP 50 were excluded; the policy
                threshold is GBP 25. Re-run with the correct floor.
```

---

## Defect 2 — the audit named the process, not the person

Everything was being recorded. The event looked like this:

```json
{ "actor": "nova-control",
  "detail": { "actor": "local", "action": "reject", "applied": true, ... } }
```

The human was there — one field too deep. **An auditor filtering on `actor` would have seen
`nova-control` for every decision this deployment ever made**, which is the precise failure
the identity work existed to prevent, reintroduced one field away from where it was fixed.

Found by reading the log rather than by a test assertion, which is worth noting: the tests
asserted the human was recorded, and the human *was* recorded. They were checking the field I
had decided to check.

*Fixed:* `AuditLog.with_actor()` binds the log to the principal for the duration of a write.
A long-running server is one process writing on behalf of many people, and the event has to
say which.

---

## The named-principal run

With a real principals file and `--behind-tls-proxy` (which turns off loopback trust, so both
callers are treated as remote):

```
dan-read  (viewer)  POST /work/t_1d158137/decide  ->  403
                    "role 'viewer' may not call this route"

priya-ops (admin)   POST /work/t_c56705a2/decide  ->  200
                    { "applied": true, "actor": "priya-ops" }

audit:  actor=priya-ops  work.annotated  intent
        actor=priya-ops  work.annotated  committed
board:  [priya-ops] OPERATOR: Findings go to the audit committee pack, not the client report.
```

The named human appears in NOVA's audit log **and** in the runtime's own board comment. That
chain — a token, a role check, a runtime transition, two independent records naming the same
person — is what "an approval with no identity is not an approval" was asking for.

---

## The CLI agrees with the API

```
$ nova work note <bundle> t_1d158137 "Cross-check against the Q3 close pack." --actor priya-ops
annotate t_1d158137 -> todo   (by priya-ops)

$ nova work resume <bundle> t_3a601a2d --actor priya-ops
t_3a601a2d: cannot be resumed from 'ready'
```

Both paths go through `runtime.decide_work`, so the audit record is identical and there is no
"the CLI does it differently" to discover during an incident. The second command is a correct
refusal: a `ready` task is not held, so there is nothing to resume.

---

## Regression cover

`tests/platform/test_work_decisions.py` — eight tests against a real board, including one
named for each defect above. They skip when the runtime is not importable, which is the
normal state of NOVA's own test environment.

612 platform tests pass, with and without the runtime installed.
