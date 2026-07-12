# Muncho privileged Canonical Writer boundary

This fork treats Canonical Brain events, task plans, dangerous-plan
capabilities, and route-back receipts as security-sensitive operational truth.
GPT/Hermes decides meaning. Deterministic code is limited to typed transport,
schema and scope validation, permissions, compare-and-swap transitions,
idempotency, safety boundaries, and receipts.

Publication of this code does **not** activate the boundary. Cloud cutover is a
separate owner/passkey-approved deployment gate and is currently blocked by the
items in [Cutover blockers](#cutover-blockers).

## Runtime boundary

The production gateway and writer must be different Unix users and different
systemd services:

```text
GPT / Hermes tool call
        |
        | fixed typed operation; no SQL, bearer token, or DB password
        v
hermes-cloud-gateway.service (unprivileged)
        |
        | /run/muncho-canonical-writer/writer.sock
        | bounded canonical-JSON frames
        | mutual SO_PEERCRED + exact current systemd MainPID
        v
muncho-canonical-writer.service (privileged writer identity)
        |
        | verified TLS; explicit writer-only credential
        | one pinned SECURITY DEFINER routine per typed operation
        v
Canonical Brain append-only event log + private append-only writer ledgers
```

Both peers re-check the other's exact current systemd `MainPID` before every
request. The gateway socket is non-inheritable and is closed in a forked child.
The gateway becomes non-dumpable and disables core dumps before any
model-controlled child can run. A terminal, MCP, cron, or `execute_code` child
may share the gateway UID, but it has a different PID and cannot use the writer
socket.

Exact-PID authorization is useful only when the authorized process cannot load
mutable code. The deployment preflight therefore requires the gateway and
writer to run from the same root-owned, read-only, revision- and digest-pinned
release. It attests exact `python -I -m` entry points, interpreter and import
closure, live module origins and executable mappings, systemd `MainPID` plus
start time, and exact mount carve-outs. Cloud dynamic Python loading must be
disabled, or every configured discovery path must be complete, immutable, and
inside that release. Writable plugin, hook, provider, or other in-process code
paths are forbidden.

The socket path, gateway unit, writer unit, and writer Unix user are pinned in
code. Mutable environment variables cannot redirect the client or restore a
direct database path. Canonical tool availability and boundary policy are
frozen once per gateway process so a live config edit cannot mutate the model
tool schema or invalidate prompt caching. An invalid enabled configuration
fails closed.

## Fixed operation surface

The protocol exposes sixteen operations:

- exact case query, route-back context, active-plan match, and bounded
  projection reads;
- model-authored event append, task-plan transition, and verification append;
- atomic route-back claim, sent finalization, and blocked finalization;
- durable capability grant, consume, plan revoke, and session revoke;
- internal lease-shadow receipt and health ping.

There is no raw SQL operation. Writer-owned route-back intent/terminal,
capability grant/check/revocation, lease, task-transition, and verification
receipts are reserved from the generic append routine. Task-plan and
verification events remain model-authored, but their tool adapter selects the
dedicated typed routine with exact revision, provenance, criteria, and receipt
invariants. A blocked route-back that fails before a claim uses the existing
typed blocked-finalization operation in an explicit `preclaim` shape; it is not
forged as a generic model event.

No operation classifies text, scans keywords, chooses a business route, or
interprets intent. The operation dispatcher is a mechanical enum-to-handler
map only; GPT/Hermes remains the semantic authority.

Every gateway-initiated mutation also carries an exact process-memory session
epoch. Session retirement and each new model event, plan transition,
verification, route-back claim, or preclaim-blocked lifecycle serialize on the
same scope lock. If retirement wins, the paused old worker receives
`session_epoch_retired` and cannot create new Canonical state. If the mutation
wins, it commits before retirement. The one deliberate exception is terminal
truth: an already-claimed route-back may still record its exact sent or blocked
receipt under the original claimant epoch after rotation.

## Canonical truth and legacy quarantine

The migration gives `public.canonical_event_log` one offline `NOLOGIN`
migration owner, removes every other table and column ACL, forbids RLS, user
triggers, rules, policies, and inheritance, and attests the exact table
identity in the same serializable, advisory-locked transaction as every writer
call. Canonical schema, side-table, sequence, and function ACLs are likewise
closed to every non-owner grantee except the runtime writer's exact `USAGE` and
sixteen `EXECUTE` grants. The migration does not revoke unrelated shared
database/schema/function privileges as a side effect; unexpected external ACLs
inside the Canonical boundary fail the contract closed.

Every new event is inserted, read back, content-hash verified, and then linked
to a private writer-provenance row in one transaction. Queries, active-plan
checks, route-back context, projection export, completion receipts, and
capability decisions consider only rows with matching writer provenance.

The provenance table intentionally starts empty. Existing events written by the
legacy shared helper are quarantined from current operational truth until an
owner reviews their exact schema and content and applies a separate, explicit
reseed/reconciliation migration. They are not silently trusted or deleted.

## Route-back receipts and Discord boundary

Canonical route-back follows this order:

1. Resolve and authorize an exact owner-approved public channel or public
   thread.
2. Atomically create a durable claim bound to the globally unique case plus
   idempotency key, exact claimant session/epoch, target, and rendered content
   hash. A retry from another authorized lane therefore observes the same
   lifecycle instead of creating a second send authority.
3. Only the process that inserted the claim may call the live adapter.
4. Read the message back through the live Discord adapter.
5. Finalize `route_back.sent` only when the receipt has exact platform,
   channel, message ID, content hash, adapter acceptance, and verified readback.

If Discord accepted the outbound call but readback cannot be proven, the writer
stores `route_back.blocked` with an exact six-field partial receipt and marks
the delivery as `accepted_unverified`. A post-send timeout or exception in the
claimant is treated as delivery-outcome-uncertain, finalized without a resend,
and reported as such. A different request that encounters an existing
unterminated claim may neither infer failure from its age nor send again: the
lifecycle remains pending reconciliation until the original claimant or an
owner-reviewed receipt reconciliation supplies a terminal fact. If adapter
acceptance and live readback were verified but the sent terminal could not be
persisted after bounded reconciliation/retry, the writer records the exact
verified receipt as a distinct ledger-persistence blocker; it does not falsely
call delivery uncertain and still forbids resend. Pre-claim validation failures
use the typed preclaim-blocked path, which creates a terminal lifecycle and
cannot later be converted into send authority.

The Canonical route-back path and the primary Discord adapter reject DMs,
group DMs, private channels, private threads, and guild surfaces that the live
Discord `@everyone` role cannot view. When the privileged writer policy is
declared enabled, inbound DM/private interactions are ignored, native slash
and component callbacks fail closed before responding, prompt/edit/media paths
repeat the public-target proof, the send-message tool prefers the live adapter,
standalone Discord REST delivery is disabled, typing re-attests before every
request, and voice egress re-attests immediately before playback and tears down
on bot moves, channel changes, or `@everyone` role changes. Voice proof also
requires the default role's effective `CONNECT`, preventing a bot-role override
from speaking into a visible but closed-audience channel. Raw Discord channel
mutations also re-read the current channel, parent, role, and overwrite data;
missing or malformed permission data fails closed. Channel-scoped raw content
reads use the same live proof, so a DM/private snowflake cannot bypass the
gateway's ignored private ingress. Chunked text, reactions, media batches, and
download-backed sends re-attest immediately before each Discord mutation and
stop after permission revocation.

The model-facing raw `create_thread`, pin/unpin/delete, and role mutation
actions are hidden and denied while the privileged writer policy is enabled.
They do not yet carry an exact owner/passkey capability and Canonical terminal
receipt, and deterministic code must not guess whether arbitrary thread text is
a handoff. Existing public channels/threads remain usable through the typed
Canonical route-back lifecycle. A future thread/handoff executor must add its
own writer-owned claim and terminal receipt before these raw actions can be
enabled in production. The legacy SQLite/synthetic gateway handoff watcher and
its silent home-channel fallback are likewise disabled under the privileged
policy, so returning `None` from thread creation cannot bypass Canonical truth.

These are defense-in-depth code gates, not yet a global token-egress guarantee:
the gateway still owns the Discord token, so a compromised gateway or an
insufficiently isolated child could bypass Python call sites. A separate
privileged Discord-token/egress boundary remains required before claiming that
*all* Muncho Discord sends are public-only.

## Dangerous-plan capability lifetime

Dangerous plan authority is granted only from the current authenticated,
configured Discord owner message and is bound to that exact user, Canonical
case, active plan ID and revision, command hashes, normalized expiry, session
digest, and a cryptographically random capability epoch. Uses are bounded,
expire within eight hours, are transactionally consumed, and append a receipt.
Plan/session revocation creates durable scope tombstones, so a retry or later
grant cannot revive authority in the revoked generation. Superseded or
terminal plan revisions revoke their grants.

The raw capability epoch exists only in gateway process memory. It is omitted
from model-facing `SessionContext`, prompts, IPC, `sessions.json`, and
`state.db`; only its SHA-256 digest reaches the writer. `/new`, reset, resume,
branch, ordinary session switches, recovery without an exact continuation, and
every gateway restart rotate it. Before any live epoch-changing routing entry
is published, the gateway binds a fresh internal context to the exact old
session, user, and epoch and requires the writer to confirm a durable
session-scope tombstone. Explicit transitions fail without changing the
session/epoch if that confirmation is unavailable; automatic policy resets
remain on the old epoch and retry later. Publication uses compare-and-swap so a
concurrent winner cannot be overwritten after the privileged call. Only a
trusted compression transition observed and applied directly to the live
in-process `SessionEntry` preserves the epoch. Compression lineage inferred
later from `state.db`, routing JSON, or a topic binding is never authority:
those stores are writable by same-UID model children, so recovery/switching
from them performs a durable revoke and mints a fresh epoch.

The running-agent `/new` path executes that durable gate before interrupting
the old turn, releasing its slot, evicting its cache, clearing queues, or
stopping delegations. A writer outage therefore leaves both the old routing
entry and the running task untouched and reports the blocked reset instead of
performing a partial boundary transition.

Consequently, dangerous approvals intentionally expire on gateway restart.
This prevents a same-UID child from restoring an old routing snapshot and
reviving durable authority. Restart-persistent approvals must not return until
a future writer-owned monotonic generation protocol can provide equivalent
rollback resistance.

## Database and credential policy

The gateway must have none of the following:

- a database password, password file, `.pgpass`, or DB secret environment
  variable;
- access to the retired Cloud SQL helper;
- effective Cloud SQL or Secret Manager IAM permissions;
- table/routine ownership, DDL, table `SELECT`/`INSERT`/`UPDATE`/`DELETE`, or a
  general SQL execution surface.

The runtime database role is not `postgres`, owns no object, has no dangerous
role attributes or memberships, and receives only database `CONNECT`, schema
`USAGE`, and `EXECUTE` on the exact sixteen pinned routines. It has no direct
table grants. The offline migration owner is `NOLOGIN`, has no dangerous role
attributes, has neither memberships nor inheriting members, and alone owns the
event table, private ledgers, schema, and routines.

The migration never revokes ambient `PUBLIC` privileges from unrelated shared
database objects. Instead, startup attestation fails closed if the writer role
inherits `TEMP`, public-schema/function capability, or `CONNECT` to another
database. Production therefore needs a dedicated Canonical database/cluster,
or a separately owner-reviewed ambient-ACL hardening change. A default shared
cluster is a cutover blocker, not something this migration silently rewrites.

PostgreSQL 12 or newer is required for the pinned catalog contract. Routine
owner, language, `SECURITY DEFINER`/invoker mode, exact safe
`search_path`, definition SHA-256, PUBLIC ACL absence, helper catalog, owner
attributes, memberships, table identity, and all effective privileges are
attested at startup and again inside every serializable advisory-locked call.
The root-owned writer config also pins a SHA-256 over the complete private
schema identity: schema/table/index owners and dangerous owner attributes,
the exact relation set, columns, defaults, constraints, indexes, persistence,
storage, RLS, triggers, rules, policies, and inheritance. Any owner or
structural drift fails closed on startup and on every call.

The password comes from an explicit writer-owned regular file or an
already-open descriptor. It is never accepted through a model payload, gateway
config, command argument, or new behavioral environment variable. TLS requires
a trusted CA and hostname/IP verification.

## Configuration and projection split

Gateway `config.yaml` contains only static non-secret policy:

```yaml
canonical_brain:
  tools_enabled: true
  writer_boundary:
    enabled: true
```

The writer consumes a separate root-owned, secret-free strict JSON file. It
pins service identities, verified database coordinates, the writer-owned
credential path, Discord owner IDs, deployment lock, and exact routine
identities plus the reviewed private-schema identity SHA-256. Unknown fields,
embedded secret-shaped keys, mutable modes,
symlinks, untrusted parents, unpinned units, and shared gateway/writer UIDs are
rejected.

The projector has no database credential. A privileged one-shot writer job may
create an atomic, bounded, writer-owned event export for the unprivileged
projector group. Every page is read inside one PostgreSQL `SERIALIZABLE READ
ONLY` transaction/snapshot, so concurrent live commits cannot create a skipped
or internally inconsistent export. The exporter and timer are disabled by
default. If enabled,
preflight requires an exact one-shot argv, the same immutable release and
config, a dedicated output directory, an exact hardened unit/timer schedule,
and no other writer-UID cron, `at`, transient-unit, timer, or process surface.

## Required deployment evidence

The deterministic preflight consumes three strict deployment blocks in
addition to DB, IAM, socket, credential, and systemd evidence:

- `writer_deployment`: immutable writer artifact, unit, live process, complete
  import/mapping closure, and minimal writer/export writable paths;
- `gateway_deployment`: the authorized gateway's exact shared immutable
  release, unit, live process, dynamic-Python policy, complete code closure,
  and explicitly allowlisted writable state/log/workspace paths that cannot
  contain in-process code discovery;
- `writer_authority_surface`: fresh UID-0-collected identities and groups,
  denial of systemd/transient-unit/timer/cron/UID-switch/polkit/sudo/doas and
  capability authority for the gateway and every child, plus an exact
  privileged service/timer/cron/`at`/process inventory.

These blocks are validation contracts, not evidence collectors. A JSON field
that merely says `collected_by_uid: 0` is self-assertion. Production requires a
trusted root-side collector that obtains the values from kernel/systemd/polkit
state, binds them to the current boot and process start times, authenticates the
snapshot, and invokes preflight within its freshness window.

## Cutover blockers

Cloud deployment must remain blocked until all of the following are complete:

1. **Live schema attestation and reconciliation.** The checked-in migration
   intentionally requires an exact fourteen-column event-log contract. Legacy
   Cloud currently has five additional columns (`inserted_at`,
   `idempotency_key`, `source_spool`, `spool_line_number`, and
   `raw_event_sha256`), producing a nineteen-column shape with additional
   defaults and indexes. Collect the exact live
   types, order, nullability, defaults, owner, constraints, ACLs, triggers,
   rules, policies, and row count read-only; then review a reconciliation
   migration and derive/review the private-schema identity digest from the
   resulting production-shaped PostgreSQL instance. Do not apply
   `canonical_writer_v1.sql` beforehand.
2. **Dedicated database authority.** Prove that the writer role has no ambient
   `PUBLIC`/`TEMP`, public-schema/function, or other-database authority. If the
   current Cloud database is shared/default, provision a dedicated Canonical
   database/cluster or approve a separate ACL-hardening migration; this writer
   migration will not alter unrelated workloads.
3. **Real PostgreSQL E2E.** Apply the migration twice to a production-shaped
   ephemeral PostgreSQL instance and execute all sixteen routines through the
   real wire client, including idempotent retries, CAS conflicts, provenance
   quarantine, completion verification, route-back partial receipts, legacy
   column ACLs, arbitrary grantee/function ACL drift, malformed preexisting
   writer objects, migration rollback, revocation tombstones, competing
   grant/consume/revoke transactions, cross-lane route-back claim races,
   preclaim terminal fencing, one-snapshot multi-page export, and concurrent
   transactions. Static parsing and Python fakes do not validate PL/pgSQL
   execution.
4. **Trusted root evidence collector.** Implement and review the authenticated,
   fresh collector for gateway/writer code closure, local authority, systemd,
   polkit, IAM, secret sources, socket, and privileged execution inventory.
5. **Legacy truth decision.** Owner-review and explicitly reseed accepted legacy
   events, or formally start a new trusted truth epoch. Until then legacy rows
   remain quarantined.
6. **Global Discord egress boundary.** Isolate the Discord token and route every
   outbound primitive through one public-channel/thread-only privileged egress
   process before asserting the no-DM rule globally.
7. **Owner-approved Cloud mutation plan.** Provision roles, immutable release,
   users, groups, config, CA, credential, systemd units, exporter state, IAM
   removals, migration, canary, and rollback under the applicable owner/passkey
   gate.
8. **Real isolated canary and forward-only rollback.** The current
   `muncho-auto-deploy-release` helper switches the sole production symlink,
   restarts the production gateway, has no traffic isolation, and does not
   restore the previous symlink after a failed post-start check. It is not a
   canary. Because peer authentication pins the exact gateway and writer unit
   names, the canary needs an isolated host with those same unit names, its own
   dedicated database, and a separate Discord identity/public test channel.
   After truth cutover, rollback may disable Canonical mutations and select
   only a boundary-compatible artifact; reverting to a legacy gateway with
   superuser/helper authority is forbidden.

## Deployment sequence after the blockers close

1. Stop Canonical mutations and take the migration form of the global advisory
   lock.
2. Apply the reviewed reconciliation and writer migration with the offline
   migration owner.
3. Provision the immutable shared release, writer identity/config/credential,
   socket group, and exact hardened units. Keep dynamic Python disabled unless
   every extension is inside the immutable release.
4. Remove the legacy helper and all gateway Cloud SQL/Secret Manager authority.
5. Run the trusted root collector and require every preflight invariant to pass.
6. Start the writer, then the gateway.
7. Canary exact query/append, plan retry/CAS, restart-invalidated approval,
   public route-back sent receipt, accepted-unverified blocked receipt, DM and
   private-thread rejection, child-PID rejection, writer outage behavior,
   projection export if enabled, and a multi-step task resume.

Rollback is fail-closed: disable Canonical mutations and repair the writer. Do
not restore the shared helper, `postgres` credential, broad IAM, mutable code,
or a direct database fallback.

## Trust limit

The boundary protects against out-of-process model-controlled tools only when
the deployment preflight guarantees the immutable gateway code closure and
local authority denials above. Owner-reviewed Python executing inside the
gateway shares its exact PID and is trusted. Writable/unreviewed in-process
plugins are forbidden by the Cloud deployment policy; a future general plugin
isolation host remains a separate architectural improvement.
