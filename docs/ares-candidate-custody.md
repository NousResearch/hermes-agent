# Ares Candidate Custody

`CandidateStore` is the sole Ares-owned durable custody store for release
candidates. Its root is `get_ares_state_root()/candidates`, where
`get_ares_state_root()` is the installation-scoped `get_default_hermes_root() /
"ares"`. It is intentionally not profile runtime state and is not a Context
Governor store.

A builder may create and certify artifacts in scratch, but scratch is never a
sealed candidate. `CandidateStore.publish()` copies an explicit artifact
allowlist into `candidates/.incoming/<uuid>`, reopens and hashes destination
bytes, verifies the archive and identity chain, fsyncs all data, and commits
only with a same-filesystem rename followed by an fsync of `candidates/`.
`SEALED` is recorded only after that commit point.

The candidate root contains immutable `artifacts/`, append-only `events/`, and
the canonical `custody.json` snapshot. `custody.json` is strict canonical JSON
and binds the archive, identity manifests, complete artifact inventory, source
repository identities, lifecycle, audit, authorization, rollback, and
retention summaries. Lifecycle changes first persist an event, then atomically
replace the snapshot. An interrupted incoming tree is `INCOMPLETE_PUBLICATION`
and never listed as sealed.

Hostile audit receives a handoff that names the absolute persistent candidate
root, identity chain, archive digest, custody digest, inventory digest, and
lifecycle sequence. Missing custody is `BLOCKED_PENDING_REPAIR /
CUSTODY_UNAVAILABLE`; a digest or identity mismatch is `AUDIT_FAILED /
CUSTODY_CORRUPT`. Audit leases recover only to `AUDIT_BLOCKED`, never pass or
fail automatically. Candidate certification and the candidate-bundled
activation input are both explicitly `NON_AUTHORIZING`, even when
certification passes. Missing, unknown, duplicate, or contradictory authority
fields fail closed during publication. The sole positive authorization state
is recorded by the CandidateStore-owned explicit transition from
`AUDIT_PASSED` to `AWAITING_ACTIVATION`; that transition does not activate a
runtime.

GC is possible only for explicitly terminal eligible candidates with a valid
`AresCandidateGcApprovalV1`. Audit-, activation-, rollback-, active-, and
incident-held candidates are protected. GC renames to same-filesystem
quarantine, persists a durable tombstone, then deletes the quarantined bytes.

For local development, one verified persistent store copy is acceptable only
before a stable public release, unattended update authorization,
originating-state deletion, or an independent-backup requirement. Once any of
those boundaries applies, a second independently retained byte-identical copy
must be evidenced before the original custody can be removed or treated as the
sole recovery source.
