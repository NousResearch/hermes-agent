# Specialist-discovery operations

## Authority and boundaries

Specialist discovery is a local Hermes orchestration feature.  It does not
grant trading, runtime, acceptance, CI, PR, merge, deployment, credential, or
provider authority.  A candidate is inert until every durable receipt is
present: independent benchmark, one-task disposable sandbox, independent
verification, staged operator approval, a local no-send canary, and a second
active operator approval.

Hermes now recognizes an approval only when it is submitted through the gated
dashboard by an OAuth-authenticated session whose exact
`<provider>:<user_id>` subject is in
`kanban.specialist_operator_approvals.allowed_subjects`.  The allowlist is
empty by default, and loopback dashboard session tokens are deliberately not
an operator identity. `operator_identity` text supplied by a worker, model,
or API body is not authority.

Configure the intended operator before using this feature:

```yaml
kanban:
  specialist_operator_approvals:
    allowed_subjects:
      - "portal:YOUR_AUTHENTICATED_USER_ID"
```

The authenticated dashboard can then POST the exact candidate ID, durable
verification receipt hash, and target (`staged` or `active`) to
`/api/plugins/kanban/specialist-approvals`. The server derives the operator
identity from its verified session and records an append-only approval. A
separate approval is required for each target state.

The status read model is deliberately reconstructed from append-only SQLite
records.  It never retries Codex, Claude, Sol, an advisory adapter, a
benchmark, a sandbox, or a canary after restart.

## Inspecting one candidate

Use `gateway.status.specialist_discovery_status(candidate_id, db_path=...)`.
It returns only opaque receipt hashes and stage/status rows for:

- source task and candidate request;
- benchmark, sandbox, verification, staged/active approvals, and no-send
  canary;
- active resolution, expiry, and rollback/revocation.

`recovery_action` is `active_resolution` only for an unexpired, unrevoked
durable profile.  Any missing, expired, revoked, malformed, or unavailable
state returns `task_orchestrator` or `normal_triage`; it must not dispatch a
generated profile.

Recovery always opens an existing mutable Hermes SQLite ledger with a fresh
`mode=ro` snapshot, so a committed WAL revocation is visible immediately even
if the WAL appears between earlier filesystem observations and the read. It
never mutates the database or WAL. SQLite may attach or create its `-shm`
mapping while observing an active writer; that mapping is not lifecycle
evidence. Any fresh-read failure is `normal_triage`, never a stale active
resolution.

## Synthetic manual no-send canary

This is an operator-run local verification, not a provider or deployment
operation. After the staged dashboard approval is durably recorded, use a new,
synthetic diagnostic-only signature and opaque test-case hashes. Confirm all
of the following before the active approval:

1. The no-match handoff created exactly one candidate request and preserved
   source idempotency.
2. The original task was assigned to `task-orchestrator`; no generated profile
   is active or dispatchable.
3. The durable benchmark, disposable one-task sandbox, verification, and
   staged approval are present.
4. `run_local_no_send_canary` produced a `local-no-send` receipt.  It only
   reads/writes local Kanban SQLite records: no provider/model/network call,
   adapter fallback, task worker, notification send, external write, or
   registry activation occurs.
5. Registry activation still fails until the separately recorded active
   operator approval is present.

The canary never proves runtime, acceptance, trading, CI, merge, or deployment
safety.

## Expiry and rollback

An elapsed profile expiry is rejected during lookup.  To roll back before
expiry, call `CapabilityRegistry.revoke` with the exact local profile and
capability signature; it appends an immutable revocation receipt.  Revoked
profiles immediately stop resolving.  Subsequent specialist handoffs use
their safe `task-orchestrator`/normal-triage fallback and may create only an
inert candidate request.

No receipt table is updated or deleted during recovery, expiry, or rollback.

## Verification evidence

The focused specialist-discovery suite was run locally with the Hermes project
venv:

```text
pytest tests/gateway/test_capability_registry.py tests/gateway/test_candidate_profile_requests.py tests/gateway/test_specialist_routing.py tests/gateway/test_specialist_handoff_orchestration.py tests/agent/test_profile_candidate_review.py tests/agent/test_profile_benchmark.py tests/gateway/test_specialist_discovery_e2e.py -q
```

Result: `75 passed in 4.74s` (local diagnostic evidence only; no external
adapter, provider, model, or live-trading action was invoked).
