# Honest live full-canary driver

`gateway.canonical_full_canary_live_driver` is the root-only observation and
orchestration layer for one approved isolated Muncho canary. It does not alter
the normal production gateway and it does not claim Discord ingress. The
canary prompt enters through the authenticated loopback API server.

The driver deliberately contains no semantic task router, keyword guard,
classifier, dispatcher, or external effort chooser. GPT-5.6-sol authors the
task plan, success criteria, plan transitions, tool calls, verification
receipts, and the adaptive `high` to `max` reasoning directive. Deterministic
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
   after the unchanged writer-config digest and session-bound fixture digest
   are final. A different staged writer config is replaced only after the
   coordinator proves services are stopped and re-reads the same inode and
   digest immediately before replacement.
5. The same sealed coordinator process emits the exact plan-bound owner request
   over its existing session. The owner-input cutoff and hard deadline are
   carried in that request; no remote approval file or second remote process is
   involved. The local launcher consumes the separate one-shot owner-only
   approval file and sends its bounded frame over this session.
6. The returned `SessionBoundPlan` is passed directly to
   `HonestFullCanaryDriver(prepared).run()`. The driver accepts no database
   administrator, login, password, provisioning object, or connection factory.
   The raw API key has never been serialized. `run()` atomically consumes and
   removes it from the caller-owned holder before any live work; the holder
   cannot be run twice. The key is dropped from loopback-client and driver
   locals on every success or failure path.

The plan-builder remains deployment-owned because it binds release artifacts,
service identities, and root paths. The approval is owner-authored and remains
a separate gate; neither callback infers approval from historical state.

The live Cloud run therefore has one deliberate external prerequisite: the
reviewed stopped-only Phase-B foundation must already have a terminal,
in-process readiness descendant bound to this release. The live run can read
that proof but cannot create or recover foundation authority.

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

Stopping the processes is not itself treated as completion proof. The driver
requires the plan-bound Canonical task, append-only event chain, verification
records, session/capability revocation, and public route-back receipt before it
accepts the run. It then proves all live services and the Phase-B readiness
unit are stopped. Missing durable truth is reported as blocked cleanup, never
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
