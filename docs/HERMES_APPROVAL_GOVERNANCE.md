# Hermes Approval Governance (P2)

P2 adds persistent approval governance for Code Mode actions, with GitHub
write safety as the first concrete use case.

## Scope

- Persistent approval requests in `state.db` (`code_approval_requests` table).
- Approval lifecycle state machine with transition validation.
- Redacted API responses and event payloads.
- Integration with `ExecutionPolicyEngine` risk classes.
- GitHub write endpoints gated by approval lifecycle (no ad-hoc `approved=true`).

## Lifecycle

Statuses:

- `pending`
- `approved`
- `rejected`
- `expired`
- `cancelled`
- `executed`
- `failed`

Allowed transitions:

- `pending -> approved | rejected | cancelled | expired`
- `approved -> executed | failed | cancelled`
- `rejected`, `expired`, `cancelled`, `executed`, `failed` are terminal

Invalid transitions return safe errors.

## Risk Classes

Risk classes align with Code Mode policy:

- `safe_readonly`
- `safe_local_write`
- `network`
- `git_write`
- `secret_sensitive`
- `remote_mutating`
- `destructive`
- `production_sensitive`

GitHub write actions are classified as `git_write` and tagged with
`network + git_write`.

## API

- `GET /api/code/approvals`
- `POST /api/code/approvals`
- `GET /api/code/approvals/{approval_id}`
- `POST /api/code/approvals/{approval_id}/approve`
- `POST /api/code/approvals/{approval_id}/reject`
- `POST /api/code/approvals/{approval_id}/cancel`
- `POST /api/code/approvals/expire`
- `GET /api/code/approvals/summary`

All responses are redacted for secret-bearing keys and token-like values.

## GitHub Write Flow

For `POST /api/code/github/comments` and
`POST /api/code/github/pull-requests/prepare`:

1. Call without `approval_id`.
2. Backend creates persistent `pending` approval and returns:
   - `requires_approval: true`
   - `approval_id`
   - `status: pending`
3. Approve via `POST /api/code/approvals/{approval_id}/approve`.
4. Call the same GitHub endpoint with `approval_id`.
5. Backend validates kind/action/resource/payload binding and expiry.
6. On success: write executes once and approval becomes `executed`.
7. On execution failure: approval becomes `failed`.

Rejected/expired/cancelled/executed approvals never execute writes.

## Event Model

Persisted in `code_events` with normalized envelope:

- `code.approval.created`
- `code.approval.approved`
- `code.approval.rejected`
- `code.approval.cancelled`
- `code.approval.expired`
- `code.approval.executed`
- `code.approval.failed`
- `github.write.approval_required`
- `github.write.executed`
- `github.write.rejected`
- `github.write.failed`

Realtime fanout is deferred; persistence is authoritative in P2.

## Security / Redaction

Never exposed in API/events:

- Authorization headers
- GitHub/PAT/install tokens
- private keys, webhook secrets
- client secrets, passwords

Keys matching `token`, `authorization`, `secret`, `password`, and related
variants are redacted recursively.

## Known Limitations

- No dashboard approval panel (`hermesWeb` UI remains deferred).
- Approval actor identity is best-effort (`local`) where strong identity is
  unavailable.
- Optional `/execute` endpoint is intentionally omitted; action endpoints
  perform their own execution after validation.

## P3 Recommendations

- Add signed user identity/claims for `requested_by`/`approved_by`.
- Add policy-driven TTL profiles by risk class.
- Add realtime event fanout for approval changes.
- Add dashboard approval management UI on top of existing APIs.
