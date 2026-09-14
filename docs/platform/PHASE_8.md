# Phase 8 — The control plane learns to act

The dashboard could see everything and change nothing. **Core patches: still 1.**

This closes the first entry under *Future capability* in
[`PRODUCTION_READINESS_AUDIT.md`](PRODUCTION_READINESS_AUDIT.md), which was explicit about
what it was waiting for:

> Approvals, retries and objective submission from the dashboard all need finding 3
> (identity) first — **an approval with no identity is not an approval.**

Identity arrived with the principals store. This is what it was for. And it needs its other
half stated, because the phase turns on it: **an identity that can act without leaving a
record is not an identity either.**

---

## Four verbs, not "set status"

| Verb | What it means | Runtime transition |
|---|---|---|
| `release` | let a queued or held item through to run | `promote_task`, carrying the actor |
| `reject` | send an item awaiting review back, with a reason | reviewer run closed, or review reopened |
| `resume` | return a held item to whatever phase it was in | `unblock_task` |
| `annotate` | attach a note a worker will read | `add_comment` |

A control plane that can write any state can write an inconsistent one — a task marked ready
whose parents never finished, a review closed with a run still claimed. So NOVA asks for the
transition it wants and lets the runtime decide whether it is legal. **"No" is a normal
answer here**, returned as `409` with the runtime's own explanation, not an exception:

```json
{ "action": "release", "task_id": "t_1d158137", "applied": false,
  "reason": "unsatisfied parent dependencies: t_3a601a2d, t_67973cce (the ready -> running
             claim re-checks parents, so promotion cannot bypass them...)",
  "resulting_status": "todo", "actor": "priya-ops" }
```

That is a live response. Nobody wrote that sentence into NOVA; the runtime refused a
promotion that would have let a step run before the work it depends on, and said why.

## Which writes are model-visible, and which are not

`annotate` goes through `audit.model_visible_change` — intent before, committed after —
because the note becomes part of what a worker reads, exactly as a work item's body does.

`release`, `reject` and `resume` are recorded, not write-ahead. They change **when** a
worker runs, not **what it reads**. That distinction is the entire basis of the audit's
honesty, and blurring it in either direction makes the log mean less: mark everything
write-ahead and "model-visible" stops selecting anything; mark nothing and the invariant is
a slogan.

Refusals are logged too. An approval somebody attempted and was refused is worth exactly as
much to an auditor as one that succeeded.

## Security: the phase created an attack and had to close it

A loopback caller is a local admin. That was harmless while everything was a read. It is not
harmless now, because **any page in the operator's browser can make their browser POST to
`127.0.0.1`** — and that browser has the loopback admin rights.

Three defences, each blocking a different technique, none sufficient alone:

1. **`Content-Type: application/json` required.** An HTML form can only send urlencoded,
   multipart or text/plain, so a form on a hostile page cannot reach the API at all.
2. **Same-origin check on any `Origin` header.** Present and mismatched is a refusal.
   `Referer` is deliberately not required — privacy tooling strips it often enough that
   requiring it would break real operators rather than real attacks.
3. **No CORS preflight is answered.** `OPTIONS` returns 405, so a scripted cross-origin
   fetch never gets to send the real request.

Plus: a 64 KiB body cap checked from the header before anything is read, `PUT`/`PATCH`/
`DELETE` still refused before a handler runs, and a write with no audit log configured
refused outright with `503` — a write path that can run without leaving a record is one
somebody will run without leaving a record.

### Two permission tables, not one

`ROUTE_ROLES` gates reads. `WRITE_ROUTES` gates writes. They are separate because reading a
route and changing what it describes are different permissions, and one table would make
them the same permission by accident the first time somebody added an endpoint.

The defaults differ too, and asymmetrically on purpose: an undeclared **read** requires
admin; an undeclared **write** is a `404` — unroutable, not merely restricted. Forgetting to
declare a read exposes data. Forgetting to declare a write hands out an action.

---

## What the live run found

Two defects, both invisible to the unit tests, both found by
[`LIVE_RUN_WRITES.md`](LIVE_RUN_WRITES.md) — because those tests used a stub runtime, and a
stub returns whatever the stub was written to return.

### 1. `reject` used the wrong primitive

`request_changes` closes an **active reviewer run**: a reviewer agent claimed the task out of
`review` and is handing it back. A human looking at a queue of tasks awaiting review has
claimed nothing, so the call refused — correctly, and unhelpfully:

```
"applied": false, "reason": "task is not in an active review run"
```

The operator's rejection is `reopen_review_task` plus the comment carrying the reason, which
is what the runtime's own operator command does. Both paths are now tried, reviewer first,
so a genuine reviewer run is still closed properly. The reason is passed through the
runtime's `redact_review_value` on the way: a rejection reason is durable, a human wrote it
in a hurry, and secrets end up in exactly that kind of text.

### 2. The audit named the process, not the person

The human was in `detail.actor` and the top-level `actor` said `nova-control`. Everything was
recorded and **an auditor filtering on `actor` would have seen none of it** — which is the
precise failure the identity work existed to prevent, reintroduced one field away.

`AuditLog.with_actor()` now binds the log to the principal for the duration of a write. A
long-running server is one process writing on behalf of many people, and the event has to
say which.

---

## Verified end to end

Against a real board, real profiles, a real HTTP transport and two named principals:

- A dependent step's release **refused by the runtime** with its own reason, `409`.
- A note reaching the board as `[priya-ops] OPERATOR: ...`, with `intent` + `committed`.
- A rejection reaching it as `[priya-ops] CHANGES REQUESTED: ...`, review → ready.
- A blocked step resumed, blocked → ready.
- `dan-read` (viewer) refused `403` on the same route `priya-ops` (admin) was served.
- The same audit record from the CLI as from HTTP — no "the CLI does it differently" to
  discover during an incident.

612 platform tests pass, with and without the runtime installed.

## What this is not

**Not an approval workflow.** There is no escalation queue, no notification, no "this task
is waiting for you". An operator finds the work and acts on it. Building the queue before
anybody has used the verbs would be guessing at which queue.

**Not multi-tenant.** One deployment per customer, unchanged.

**Not a dashboard feature yet.** The API and the CLI have the verbs; the browser UI still
only reads. That is the next small piece, and it is small precisely because the transport,
the permissions and the audit are already done.
