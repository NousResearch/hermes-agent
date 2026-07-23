# Operations runbook

All examples are read-only or local state changes. Never enable production
workers during this runbook.

## Account health and re-auth

1. Run `accounts show`, `accounts status`, and `accounts capabilities` for the
   exact account ID.
2. If auth is `reauth_required` or connector health is failed, disable the
   account. Do not select a sibling account.
3. Repair credentials/session through the owning platform setup flow, then
   re-check health before sync.

## Sync issue and retry

1. Run `sync status <account>` and record the run/issue IDs only.
2. Correct the account-specific connector/auth/rate-limit problem.
3. Run `sync retry <same-account>`. Cursors and locks for other accounts must
   remain unchanged.
4. If a lock is stale, verify no owner process exists before allowing its TTL
   to expire; never delete another account's lock.

## Route audit or account disable

Use `routes dry-run` before `routes set`, then inspect `routes audit`. A denied
link stays denied until the directed account link is explicitly allowed.
`accounts disable` pauses endpoints and routes without fallback or data move.

## Stuck outbox

Production execution is disabled. Inspect IDs/status/event evidence only. A
claim past TTL is marked `uncertain`; do not reset it to pending without an
external postcondition check and a separately authorized production-write
procedure. Test sink records can be discarded with the test database.

## News source quarantine/recovery

Use the News intelligence health view to inspect source success/failure
history. Repeated failures quarantine the source. Recovery requires a later
successful observation and explicit recovery transition; publication remains
moderated dry-run/test-sink only.

## Controlled-smoke gate

Before any browser/network smoke, require green deterministic suites, named
synthetic/test accounts, exact profile targets, read-only operations, bounded
timeouts, cleanup, and rollback. Re-check orphan processes, browser contexts,
leases, locks, and outbox claims afterward. The current dating pilot lacks a
confirmed provider/test account, so its smoke is blocked.
