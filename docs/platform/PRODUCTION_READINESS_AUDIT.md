# Production readiness audit — NOVA after Phase 6

Every Phase 1–5 capability re-verified against its actual runtime call site, plus the gaps
that stand between NOVA and a real customer AWS account. **Nothing was implemented during
this audit.** 440 platform tests pass (577 after the work the findings prompted); the findings below are things tests do not cover
because they are about seams, deployment and claims rather than logic.

Severity is about *deploying to a paying customer*, not about code quality:

- **Blocking** — a customer deployment is unsafe, impossible, or silently wrong.
- **Production-hardening** — deployable, but will hurt in month two.
- **Future capability** — deliberately absent; listed so nobody assumes it exists.

---

## Summary

| # | Finding | Area | Severity |
|---|---|---|---|
| 1 | ~~No TLS on the control plane~~ | Security | **FIXED** |
| 2 | ~~Bearer token passed as a CLI argument~~ | Secrets | **FIXED** |
| 3 | ~~No identity, users or RBAC~~ (OIDC still future) | Auth | **FIXED** |
| 4 | ~~Tenant collision is silent~~ | Isolation | **FIXED** |
| 5 | ~~No AWS deployment artifacts of any kind~~ (never applied for real) | AWS | **FIXED** |
| 6 | ~~`credential_isolation=True` overclaims~~ | Enforcement | **FIXED** |
| 7 | ~~Control plane returns every task regardless of tenant~~ | Isolation | **FIXED** |
| 8 | ~~Audit log: no rotation or retention~~ | Observability | **FIXED** |
| 9 | ~~Audit log tampering undetectable~~ (evidence, not prevention) | Security | **FIXED** |
| 10 | ~~Effectively no operational logging~~ | Observability | **FIXED** |
| 11 | ~~Interrupted apply never reconciles~~ | Failure recovery | **FIXED** |
| 12 | ~~`PROVENANCE_VERSION` never checked~~ | Upgrades | **FIXED** |
| 13 | ~~No runtime-version compatibility check~~ | Upgrades | **FIXED** |
| 14 | ~~Agent-level `max_task_runtime_seconds` ignored~~ | Enforcement | **FIXED** |
| 15 | ~~Two limit classifications are stale~~ | Enforcement | **FIXED** |
| 16 | ~~No backup or restore story~~ | Failure recovery | **FIXED** |
| 17 | ~~Control API is read-only — no write path~~ | Future | **DONE** |
| 18 | No multi-tenancy | Future |
| 19 | No cost ceiling (structurally impossible) | Future |
| 20 | No semantic retrieval | Future |

---

## Blocking

### 1. No TLS on the control plane — **FIXED**

> Closed. `--tls-cert`/`--tls-key` terminate TLS here (TLS 1.2 floor), or
> `--behind-tls-proxy` acknowledges a proxy in front. A non-loopback bind without one
> of the two is refused before the socket accepts. A bad certificate fails at
> construction, not at the first request.


`nova/control/server.py` is a `ThreadingHTTPServer` with no TLS anywhere — `grep` for
`ssl|https|certfile` across `nova/control/` returns nothing. The bind guard forces a token
for non-loopback binds, but that token then crosses the network **in cleartext**.

Loopback-plus-SSH-tunnel is a legitimate posture and is what the code recommends. But the
moment a customer fronts this with an ALB, the token is sniffable inside the VPC.

*Fix shape:* terminate TLS at the load balancer and refuse to bind non-loopback without an
explicit `--behind-tls-proxy` acknowledgement, or accept a certfile. The first is less code
and matches how it will actually be run.

### 2. Bearer token passed as a CLI argument — **FIXED**

> Closed. `--token` is gone. `nova token new` mints a token, prints it once, and
> stores only its SHA-256 digest in `<home>/control-principals.yaml` — so NOVA never
> holds the credential and a leaked principals file is an inconvenience, not an
> incident.


`nova serve --token <secret>`. The token is visible in `ps aux` to every user on the host and
lands in shell history. This directly contradicts the principle Phase 6 was built on — NOVA
writes the *name* of a secret, never its value — and is the one place NOVA still handles a
credential as a literal.

*Fix shape:* read it from an environment variable or a file path, exactly as Phase 6 does for
model credentials. `--token` should be refused, not merely deprecated.

### 3. No identity, users or RBAC — **FIXED** (OIDC remains future)

> Closed as far as it can be without a dependency. Callers are named `Principal`s
> with a role; `viewer` sees operational state, `admin` also sees policy, decisions
> and cost. Routes are gated per-role and an undeclared route requires admin, so a new
> endpoint must be *declared* viewer-readable rather than becoming readable by
> forgetting. The access log records the principal's name, and denials are logged.
> Revoking one principal leaves the rest. A real IdP replaces `authenticate()` and
> touches nothing else.
>
> One consequence worth knowing: a browser cannot send a bearer token, so a loopback
> caller is treated as a local admin — they can already read the principals file and
> every profile off disk, so a token protects nothing. `--behind-tls-proxy` disables
> that, because a local proxy makes every forwarded request look loopback.


There is one shared bearer token and no concept of a user. Consequences: no per-user audit
(the log records `actor="nova-control"` for everyone), no read/approve separation, no
revocation short of restarting with a new token, and no lockout or rate limiting.

This matters more than usual because the control plane is where a human would eventually
*approve* an escalated business action. An approval gate with no identity cannot record who
approved.

*Fix shape:* OIDC in front of the control plane, with the subject carried into the audit
record. Phase 2's approval design already anticipates a human gate; this is the missing half.

### 4. Tenant collision is silent — **FIXED**

> Closed. `nova-agent.json` now records `tenant_id`; materializing over another
> tenant's profile raises before anything is written. An unstamped profile is adopted
> with a warning, and adoption forces a marker rewrite so it actually completes.
> `tests/platform/test_tenant_isolation.py` (13 tests) reproduces the original
> scenario. Two bugs surfaced while fixing it: adoption silently never completed
> because the unchanged fast path skipped the rewrite, and `apply_bundle` was
> discarding every per-agent `MaterializeResult.warning` — both fixed.


Verified empirically. Applying two different tenant bundles to one `$NOVA_HOME`:

```
$ nova apply .../acme      → tenant=acme   created=2
$ nova apply .../globex    → tenant=globex unchanged=2     ← adopted acme's agents
```

`globex` reported acme's profiles as its own because `nova-agent.json` records
`version`, `agent_id`, `digest` and `nova_version` — **and no tenant id**. There is nothing
to compare, so nothing detects the collision.

The blast radius is not cosmetic: globex's control plane would display acme's agents and
tasks, globex's objectives would dispatch onto acme-materialized profiles, and those profiles
carry acme's `<profile>/.env` credentials.

One-tenant-per-deployment *is* the documented design (`HERMES_PLATFORM_AUDIT.md` §13.5,
`PHASE_4.md`), so this is a violated constraint rather than a missing feature. But a
documented constraint whose violation is silent and cross-contaminating is not a constraint —
it is a trap, and an operator will fall into it by putting two bundles on one box.

*Fix shape:* stamp `tenant_id` into the provenance file and refuse to materialize over a
profile belonging to a different tenant, in the same way materialization already refuses a
profile with no provenance at all.

### 5. No AWS deployment artifacts — **FIXED**

> A Terraform root module in `deploy/aws`, and `nova deploy render`, which derives the
> per-integration IAM from the tenant bundle rather than leaving it to be maintained
> alongside. The derivation is the point: a grant that lives beside the declaration drifts
> from it, and the drift is only ever in one direction.
>
> The refusals are what makes §6 a control instead of a sentence. A wildcard action, a `"*"`
> resource, and any `iam`/`sts`/`organizations`/`account`/`kms` action are refused — in the
> generator, *and again* in the module's own `variable "integrations"` validation, because a
> check that lives only in the generator is one an operator skips by hand-editing the tfvars,
> and the operator who does that is the one under time pressure.
>
> **What is verified, and what is not.** Verified offline against the real AWS provider:
> `terraform validate` passes, `fmt -check` is clean, every refusal fires at plan time
> against deliberately over-broad input, and — applied against a mock AWS API — the rendered
> IAM is what is claimed. The runtime role's policy contains no customer-data permission of
> any kind; its only `sts:AssumeRole` names the declared integration roles one by one; the
> integration trust policy names the runtime role plus an external id; the permissions
> boundary denies `iam:*`.
>
> **Not verified: this has never been applied to a real AWS account.** `aws_instance` and the
> Session Manager policy attachment went unexercised (the mock has neither a real AMI nor the
> AWS-managed policy catalogue), and `user_data.sh.tftpl` has never run on a booting host.
> The finding moves to FIXED because the artifacts exist and their security properties are
> demonstrated; the first real deployment is still a first real deployment, and
> `deploy/aws/README.md` says so where somebody will read it.

There was no Terraform, no CDK, no CloudFormation, no container image, no systemd unit, no
AMI recipe. `ARCHITECTURE_BOUNDARIES.md` §6 described the IAM model in prose — per-integration
roles assumed by a named runtime role, least privilege, no permanent admin — and **none of it
existed as code**.

"Deploy NOVA to a customer AWS account" meant a human following prose. The IAM design was
sound and unimplemented; that gap was the single largest piece of work remaining.

### 6. `credential_isolation=True` overclaims — **FIXED**

> Corrected rather than removed: the flag does mean something true — two agents given
> different keys get different keys — and the contract now states the boundary it does
> *not* cover, with the empirical evidence. The actionable half is that
> `deployment_readiness` reports `host_wide`: which credentials resolved from the host
> environment and are therefore shared by every agent. `nova doctor` prints it on a
> multi-agent host, so an operator sees the isolation they actually have rather than
> the boolean they were promised.


`HERMES_CAPABILITIES` asserts `credential_isolation=True`. Verified: a dispatcher-spawned
worker has `is_multiplex_active() == False` (it is set only by `gateway/run.py` and
`cron/scheduler.py`), so `agent/secret_scope.py::get_secret` falls through to `os.environ`.
And `_default_spawn` builds the child env with `scrub_secrets=is_multiplex_active()` — i.e.
`False` — so the worker inherits the dispatcher's entire environment.

Proven:

```
multiplex active in a worker-like process: False
get_secret sees a PARENT-only var: 'leaked-from-parent'
```

So the claim is **conditionally true**: credentials in `<profile>/.env` are per-agent, and
credentials in the process environment are visible to every agent on the host. Phase 6's
readiness report already distinguishes the two, which was the right instinct — but the
capability flag promises more than the runtime delivers, and a capability flag is exactly
what a reviewer will read.

*Fix shape:* qualify the capability (a `credential_isolation` that means "per-agent file
scope, not process scope"), and have `nova doctor` warn when a required variable resolves
from the process environment on a host with more than one agent.

---

## Production-hardening

> **All seven closed.** See the per-finding notes below. The one qualification worth
> carrying forward: finding 9 delivers tamper *evidence*, not prevention — NOVA cannot stop
> a process sharing its uid from rewriting a file, and a seal is only worth where it is
> kept.


### 7. Control plane returns every task regardless of tenant — **FIXED**

> Closed alongside 4. `list_tasks` and `get_task` filter by tenant; rows written
> before stamping stay visible so a deployment never appears to lose its own work.


`work.py::list_tasks` filters only by `assignee`; there is no `tenant` predicate, though the
column exists and NOVA stamps it. Harmless under one-tenant-per-home; it is the second half
of finding 4 and should be fixed with it.

### 8. Audit log grows without bound

`audit.jsonl` has no rotation, no size cap and no retention policy. A busy deployment writes
on every materialization, every knowledge search and every policy decision. This eventually
fills a disk, and the failure mode is an agent platform that stops being able to record what
it did — while still doing it.

### 9. The audit log is writable by the process it audits

The compiled policy hands each worker the log path, and the enforcement plugin appends to it
with `O_APPEND`. The file is `0600`, but the worker runs as the same OS user, so the mode is
no barrier. An agent granted any file-write or shell tool could rewrite the record of its own
refusals.

Today the policy prevents that (`operations` resolves to `allow: ['erp_stock_query']`,
`unlisted_tool: deny`), which makes this a circular trust: the integrity of the log depends on
the policy the log exists to evidence.

*Fix shape:* a separate append-only sink — a distinct OS user, `chattr +a`, or shipping
events off-host — so integrity does not depend on the control being audited.

### 10. Effectively no operational logging

One `logging.getLogger` in the entire `nova/` tree, in the control server. No structured
logs, no correlation ids on stderr, nothing a CloudWatch agent could usefully ship. The audit
log is a *governance* record and is not a substitute: it deliberately excludes the successful,
boring events that operational debugging needs.

### 11. Interrupted apply warns and never reconciles

`AuditLog.open_intents()` finds write-ahead intents with no commit, and the CLI prints a
warning. Nothing reconciles them. A crash mid-apply leaves a half-materialized agent and the
next run says so and proceeds.

The write-ahead invariant was built to make this recoverable; the recovery half was never
written.

### 12. `PROVENANCE_VERSION` is written but never checked

`Provenance.read` parses `version` and no caller compares it to `PROVENANCE_VERSION`. This is
inconsistent with the two places that do fail closed on a version mismatch —
`policy/decide.py` (`POLICY_SCHEMA_VERSION`) and `knowledge/index.py` (`SCHEMA_VERSION`). A
future NOVA reading an old profile marker will not notice.

### 13. No runtime-version compatibility check

NOVA's adapter was verified against Hermes v0.21.1 and nothing pins or checks that. The
adapter calls `kanban_db.create_task` with fifteen keyword arguments and reads the `tasks`
table by column name. An upstream change to either surfaces at task-submission time on a
customer's box, not at `nova apply`.

The manifest mechanism the runtime offers plugins (`requires_hermes`) is the obvious model.

### 14. An agent's declared task-runtime limit is ignored — **FIXED**

> Closed with 15, and necessarily before it: reclassifying the limit as enforcing
> while NOVA still dropped it would have been the exact sin the audit was looking for.
> A step with no cap now inherits its assignee's; a step that states one overrides it.


`AgentSpec.limits.max_task_runtime_seconds` is compiled into the profile config under `nova:`
where the runtime ignores it, and **is not applied to work NOVA submits** — `submit.py` reads
`max_runtime_seconds` only from the objective *step*.

This is worth more than it looks, because the dispatcher genuinely enforces the per-task value
(`kanban_db_dispatch.py::enforce_max_runtime` — SIGTERM, grace, SIGKILL). So NOVA has a
declared limit it could make `hard_preemptive` today and instead silently drops.

### 15. Two limit classifications are stale — **FIXED**

> `max_task_runtime_seconds` and `max_retries` are now `hard_preemptive`, scoped to
> work NOVA submits — the qualification is real and survives, because a task a human
> creates carries whatever that command was given. A test now asserts that every
> enforcing classification names a call site.


`nova/runtime/hermes/limits.py` still classifies `max_task_runtime_seconds` and `max_retries`
as `recorded_only`, annotated "not settable by NOVA". Phase 5 made them settable —
`WorkItem` carries both and `submit.py` passes them to `create_task`.

The error is in the safe direction (understating enforcement), but it is still inaccurate, and
`/budget` renders these classifications to customers.

### 16. No backup or restore

NOVA writes profiles, a knowledge index, an audit log and a work board, and offers no way to
back any of it up or restore it. `backups` appears only on `NEVER_WRITE`.

---

## Verified sound — no action

Worth recording, because these were claims and are now evidence:

- **Policy enforcement is real.** A live worker's `terminal` call was denied and recorded
  (`LIVE_RUN.md`). Four decisions, one by explicit deny and three by allow-list.
- **Knowledge scoping is real.** The grant is a SQL predicate resolved before the worker
  starts; four attempts to widen it through tool arguments returned nothing.
- **`max_tool_calls_per_run` is a genuine hard boundary** — `pre_tool_call` is the only hook
  in `_HOOK_TIMEOUT_FAIL_CLOSED_HOOKS`.
- **Delegation limits are all `hard_preemptive`**, each verified at a `tools/delegate_tool.py`
  call site.
- **Secrets stay out of everything NOVA writes.** Verified by grep across a live deployment:
  the credential appears only in operator-owned `.env` files.
- **The dashboard cannot be made to execute agent-written content.** No `innerHTML` anywhere;
  every value is inserted with `textContent`, so a hostile task title is inert.
- **Write paths are closed.** The control server refuses every method but GET and HEAD before
  a handler runs.
- **Token comparison is constant-time** (`hmac.compare_digest`).

---

## Future capability — absent by decision

1. ~~**Control API writes.**~~ **Built in Phase 8** ([`PHASE_8.md`](PHASE_8.md)), once
   finding 3 made it possible. Four verbs — release, reject, resume, annotate — plus
   objective submission, each gated on an admin principal, each recorded against the human
   who made it, and each asking the runtime for a transition rather than writing a status.
   The dashboard's *browser* UI is still read-only; the API and the CLI are not.
2. **Multi-tenancy.** One deployment per customer is the design. Finding 4 is about enforcing
   that boundary, not removing it.
3. **A cost ceiling.** Structurally impossible on this runtime: LLM-boundary hooks discard
   their return values (`BUDGET_ENFORCEMENT_AUDIT.md`). Usage is reported and labelled
   `observed_only`. This should never be promised.
4. **Semantic retrieval.** BM25 only, by choice (`PHASE_4.md`).
5. **Scheduled objectives.** Submission is manual; recurrence is the runtime's `cron/` or an
   external scheduler.

---

## Recommended order

The six blocking findings are not equal work. Grouped by what they share:

1. **Tenant identity** (4, 7) — small, self-contained, and the highest risk-per-line here.
   Stamp the tenant into provenance, refuse a mismatch, filter tasks.
2. **Control-plane security** (1, 2, 3) — one coherent piece: identity in front, TLS
   terminated at the proxy, token out of argv. Do it once rather than three times.
3. **Capability honesty** (6, 15) — a doc-and-flag change, no mechanism. Cheap, and it stops
   a reviewer being misled in the meantime.
4. **AWS deployment** (5) — the largest, and the one that benefits most from 1–3 being
   settled first, since the IaC has to encode the trust boundaries they define.

Findings 8–16 are real but none of them should reorder the four above.

**All four are now done**, in that order — the last of them, AWS deployment, in `deploy/aws`
and `nova/deploy/`. What remains on this list is the four entries under *Future capability*,
which are absent by decision rather than by omission, and the one honest caveat on finding 5:
the module has never met a real AWS account.
