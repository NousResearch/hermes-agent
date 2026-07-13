# Honest live full-canary driver

`gateway.canonical_full_canary_live_driver` is the root-only observation and
orchestration layer for one approved isolated Muncho canary. It does not alter
the normal production gateway and it does not claim Discord ingress. The
canary prompt enters through the authenticated loopback API server.

The driver deliberately contains no semantic task router, keyword guard,
classifier, dispatcher, or external effort chooser. GPT-5.6-sol authors the
task plan, success criteria, plan transitions, tool calls, verification
receipts, and the adaptive `high` to `xhigh` reasoning directive. Deterministic
code checks only identities, bounded protocols, lifecycle order, canonical
state, and receipts.

## Operational callback flow

There is intentionally no standalone live-driver CLI. A CLI would need either
to persist the raw one-shot API session key between plan creation and owner
approval, or to invent a second secret transport. The deployment coordinator
instead runs this exact same-process callback flow as root:

1. Load the reviewed writer configuration and E2E fixture.
2. Call `prepare_session_bound_plan(...)`.
3. The root process first disables process dumping and core dumps. Its
   session-key factory then creates the raw key in memory. Only its SHA-256 is
   written into the staged writer configuration.
4. Its plan-builder callback builds and atomically publishes the exact plan
   after the staged writer-config digest is final.
   If a different staged writer config already exists, the driver does not
   overwrite it. The coordinator must supply a bounded pre-stage reconciler;
   the driver captures the prior receipt generation, invokes reconciliation,
   then requires a fresh writer-owned terminal `retired` or `claimed` receipt
   with `authority_active=false`, bound to the old source bytes, database
   identity, and exact scope. It re-reads the old inode and digest immediately
   before replacement. `not_preapproved`, a stale receipt, active authority,
   or a replacement race remains blocked owner cleanup.
5. Its approval-provider callback calls
   `wait_for_fresh_owner_approval(plan)`. The generic helper has a 900-second
   upper bound, but the packaged coordinator always supplies the shorter
   request-bound remainder: at most 240 seconds, with owner input closing 30
   seconds before the request deadline. It accepts only a newly published
   root-owned `0400` approval at the fixed path for that exact plan digest.
6. Only after that approval, the owner-operated coordinator opens one
   short-lived managed-administrator PostgreSQL connection, independently
   verifies its TLS peer, and wraps the already-open connection in
   `PreopenedSessionBootstrapProvisioner`. No password or connection factory
   is passed to Hermes.
7. The returned `SessionBoundPlan` and that provisioner are passed directly to
   `HonestFullCanaryDriver(..., bootstrap_provisioner=...).run()` in the same
   process. Supplying both a provisioner and a custom lifecycle factory is
   rejected. Omitting both leaves the runtime's default database gate blocked.
   The raw key has
   never been serialized. `run()` atomically consumes and removes it from the
   caller-owned holder before any live work; the holder cannot be run twice.
   Both raw API keys are also cleared from the loopback client and driver
   locals on every success or failure path. The driver aborts the provisioner
   on every path, including process-hardening or fixture failures before the
   lifecycle exists, retries one transient close failure, and removes its own
   reference afterward.

The plan-builder remains deployment-owned because it binds release artifacts,
service identities, and root paths. The approval is owner-authored and remains
a separate gate; neither callback infers approval from historical state.

The live Cloud run therefore has one deliberate external prerequisite: the
owner-operated coordinator must be able to hand over that already-open
ephemeral administrator session. The implementation does not reuse or invent
a persistent administrator secret to bypass this gate. Without the pre-opened
session, packaging and read-only validation can complete, but service
activation fails closed before mutation.

## Live evidence barriers

The collector is an in-process root AF_UNIX server. It accepts frames only
from the exact gateway UID/GID/MainPID/start-time identity obtained through
Linux `SO_PEERCRED`. Every frame is canonical JSON, sequence-bound, tied to the
plan, fixture, collector readiness, and Discord-edge readiness, and appended
to a SHA-256 hash chain.

The driver independently checks the loopback listener against the gateway
MainPID both immediately before and immediately after the long API turn. The
SSE stream must end with exactly one honest `run.completed` followed by
`done`; partial, interrupted, failed, cancelled, or error terminals fail the
run.

The private-target probe has an ACK barrier around two coherent read-only
SQLite snapshots. The journal must remain the same logical database, owned by
the exact Discord-edge UID/GID with mode `0600`, and retain the same inode
through each read. The public route-back is accepted only from the committed
`verified` journal row and its matching signed receipt history.

After `run.completed`, the trusted writer runs the plan-bound one-shot
projection export. The plugin's real pre-revocation `resume_bundle` readback
must be bounded, untruncated, have no support-incomplete reason or missing
verification ID, and match the exported events byte-for-byte. The only event
allowed to appear after that readback is the durable scope-revocation event.
Plan completion and verification satisfaction are derived from those records;
they are never hard-coded by the driver.

Evidence is atomically written only to the plan-addressed root-owned `0400`
path. It is checked in-process and then by the independently packaged offline
verifier. Services are always stopped in reverse order on success or failure.
Collector cleanup unlinks only the exact runtime inodes it created and refuses
to remove a replacement.

Stopping the processes is not itself treated as cleanup proof. The writer's
sealed `ExecStopPost` reconciliation must publish a fresh exact receipt for
the plan-bound preapproval, and lifecycle accepts the stop only when durable
Canonical state proves that no authority remains active. The matching
`ExecStartPre` call performs the same reconciliation before a restart, covering
SIGKILL and power-loss recovery as long as the interrupted plan's staged source
has not been replaced. Missing proof is reported as blocked owner cleanup, not
as a successful canary result.

## Deliberate safety boundaries

- No merge, deployment, restart, or production configuration is performed by
  importing this module.
- No raw API control key or session key is included in evidence or hashed into
  a provenance receipt. Control-key provenance hashes only root-owned file
  metadata.
- Discord DMs remain forbidden. The canary proves the private target is
  rejected without a signed response or journal mutation.
- The driver does not convert an incomplete Canonical Brain readback, a
  partial task, or an unsigned route-back into success evidence.
