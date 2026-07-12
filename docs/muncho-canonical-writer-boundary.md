# Muncho privileged Canonical Writer boundary

This fork treats Canonical Brain events, task plans, dangerous-plan
capabilities, and route-back receipts as security-sensitive operational truth.
GPT/Hermes decides meaning. Deterministic code is limited to typed transport,
schema and scope validation, permissions, compare-and-swap transitions,
idempotency, safety boundaries, and receipts.

Publication of this code does **not** activate the boundary. Cloud cutover is a
separate owner-approved gate whose current bootstrap receipt records an exact
root-provisioned, out-of-band confirmation and explicitly no cryptographic
owner proof. A future runtime passkey capability is a separate authority and
must not be inferred from this bootstrap. Cutover remains blocked by the items
in [Cutover blockers](#cutover-blockers).

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

The protocol exposes seventeen operations:

- exact case query, route-back context, active-plan match, and bounded
  projection reads;
- model-authored event append, task-plan transition, and verification append;
- atomic route-back claim, exact restart recovery, sent finalization, and
  blocked finalization;
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
seventeen `EXECUTE` grants. The migration does not revoke unrelated shared
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

`scripts/sql/canonical_writer_legacy_reconcile_v1.sql` is the structural half
of that reconciliation for an isolated PostgreSQL 18 canary copy only. It
requires the exact database name `muncho_canary_brain`, a root-collected server
identity SHA-256, source owner, frozen row count,
fourteen-field digest, nineteen-field digest, `occurred_at` cutoff, and hashed
owner-approval receipt as explicit session settings, and refuses the production
database name. In one serializable, advisory-locked transaction it moves the
untouched nineteen-column relation into the private legacy-quarantine schema,
creates the exact fourteen-column public contract, copies the fourteen canonical
values without semantic transformation, and records both content receipts and
cutoffs. Reruns re-attest the same result and make no further change. A failure
before `COMMIT` restores the pristine nineteen-column state atomically. After a
trusted write begins, rollback is forward-only: disable mutations and repair or
discard the isolated canary; never move the legacy table back into the runtime
path.

The reconciliation acquires its exact administrator only as a PostgreSQL 16+
SET-only/non-inheriting member of the offline owner and temporarily grants that
owner `CREATE` on `public` before the target ownership transfer. Both grants are
revoked and their absence is attested before `COMMIT`; the SQL does not rely on
Cloud SQL's provider-specific owner-transfer bypass.

That structural migration intentionally inserts zero legacy IDs into
`writer_event_provenance`. Legacy UUID derivation, content hashing, source and
decision envelopes differ from the privileged v1 writer, so bulk provenance
promotion would falsely claim a privileged origin and could duplicate a logical
event after retry. Continuity therefore requires a later owner-reviewed typed
reseed manifest, or an explicit decision to begin a new trusted truth epoch;
neither choice is inferred by deterministic migration code.

## Route-back receipts and Discord boundary

Canonical route-back follows this order:

1. Resolve and authorize an exact owner-approved public channel or public
   thread.
2. Authenticate the credential-free gateway client to the exact current
   privileged Discord-edge systemd `MainPID`. This preconnect happens before
   any durable dispatch authority is created.
3. Ask the edge's exact mutation-free reconciliation endpoint before changing
   the Canonical claim. Current signed journal evidence enters the dedicated
   recovery path and is never resent. Only an authenticated exact `no record`
   result permits a new claim or an epoch-only takeover to mint fresh dispatch
   authority.
4. Atomically create a durable claim bound to the globally unique case plus
   caller idempotency key, exact claimant session/epoch, target, and rendered
   content hash. The edge idempotency key is independently derived from the
   case and caller key, so equal caller keys in different cases cannot collide.
   A successful nonterminal claim in the exact original runtime scope rechecks
   the current public-target ACL and returns a fresh, short-lived,
   writer-signed Ed25519 request for the exact `public.message.send` intent.
   The edge journal is first-wins: an exact PREPARED intent may atomically bind
   to a strictly newer capability, while any accepted or dispatching record is
   fenced from rebinding or a second mutation.
5. The token-owning REST edge independently re-reads the Discord guild,
   channel/thread parent, `@everyone` visibility, bot identity and exact
   permissions. It commits `dispatching` before the HTTP mutation, persists a
   signed `accepted_unverified` receipt with the returned Discord object ID
   immediately after acceptance, then performs exact author/content/reply
   readback. A verified result atomically upgrades that receipt; all prior
   signed receipts remain in append-only history.
6. The writer verifies both its own capability and the edge signature, plus
   authorization, target, content and idempotency bindings. Only a signed
   `verified` receipt can finalize `route_back.sent`. Before any Canonical
   finalization, a `dispatching` observation is re-read through the exact
   mutation-free reconciliation endpoint. A current `accepted_unverified`
   receipt is nonterminal and leaves the claim pending because later readback
   may still upgrade it to `verified`; a delayed older receipt therefore cannot
   permanently win over newer journal truth. Signed `dispatch_uncertain` or
   pre-dispatch blocked/failed evidence may finalize `route_back.blocked`.
   Gateway booleans, adapter objects and caller-authored blocker text are not
   delivery truth.

If the local transport closes after request bytes may have reached the edge but
before its signed receipt reaches the gateway, the claim remains pending exact
journal reconciliation. A retry always asks the mutation-free reconciliation
endpoint first. Only an authenticated exact `no record` result permits the
writer to issue a fresh short-lived request for the unchanged intent; the edge
journal's first-wins/CAS rules still permit at most one Discord mutation.
Existing accepted records perform readback-only reconciliation, including
after the original capability deadline, and are never dispatched again. The
gateway never fabricates `route_back.blocked`. A legacy `dispatching` record
without accepted object evidence remains conservatively uncertain. Pre-claim
validation or edge-preconnect failures use the typed preclaim-blocked path,
which creates no send authority.

A gateway restart rotates the process capability epoch but does not change the
canonical session. One pending route-back may cross that epoch boundary only
through `routeback.recover`, only for the exact same session, runtime platform,
source thread/lane, immutable case, target, rendered-content digest, and
case-scoped idempotency key, and only after the current runtime re-proves case
scope. A current signed edge receipt is paired with the original writer-signed
request and may finalize already-observed truth even if the public-target ACL
was disabled after dispatch. An authenticated exact edge `no record` result
may instead obtain a fresh short-lived request, but it must recheck the current
public-target ACL. Recovery creates no second lifecycle and does not rewrite
the append-only authorization ledger. A different session, platform, or source
lane remains blocked.

This exception is route-back-specific. It never restores, copies, or advances
a dangerous-plan approval or its usage budget. Those capabilities remain bound
to their original session epoch and intentionally expire on gateway restart.

The Canonical route-back edge rejects DMs, group DMs, private channels, private
threads, mismatched guild/parent relationships, and guild surfaces that the
live Discord `@everyone` role cannot view. Its API surface contains only fixed
send/edit/thread operations; no caller controls an HTTP method or URL. It uses
safe mentions, bounded strict JSON, no environment proxy, no redirect, and a
hard total request deadline. Non-empty two-step text-channel thread creation is
rejected before HTTP because it cannot provide one atomic accepted-object
receipt; an empty thread followed by a separately receipted send, or an atomic
forum post, is required.

When the privileged writer policy is
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

The Canonical route-back gateway no longer reads its Discord token, writer
private key, or edge private receipt key. The canary uses a distinct Discord
application whose token is readable only by the privileged egress identity.
This is not yet a claim that every unrelated Hermes Discord surface is globally
token-isolated: ordinary platform ingress/reply and legacy non-route-back paths
must either be disabled in that deployment or migrated to their own signed edge
capabilities before production may call the entire Discord integration sealed.

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
`USAGE`, and `EXECUTE` on the exact seventeen pinned routines. It has no direct
table grants. The offline migration owner is `NOLOGIN`, has no dangerous role
attributes, is `NOINHERIT`, has neither memberships nor inheriting members, and
alone owns the event table, private ledgers, schema, and routines. A managed
database administrator is not assumed to be PostgreSQL superuser. An approved
migration first attests the zero-membership state, grants its exact login a
transaction-scoped SET-only/non-inheriting owner membership, executes owner-only
DDL through `SET LOCAL ROLE`, then resets, revokes, and re-attests zero
membership before `COMMIT`. Temporary archive reads and database `TEMP` needed
by migration checks are treated the same way. A crash or error rolls every
temporary grant back atomically; no runtime process receives this path.

The migration never revokes ambient `PUBLIC` privileges from unrelated shared
database objects. Instead, startup attestation fails closed if the writer role
inherits `TEMP`, public-schema/function capability, or `CONNECT` to another
database. Production therefore needs a dedicated Canonical database/cluster,
or a separately owner-reviewed ambient-ACL hardening change. A default shared
cluster is a cutover blocker, not something this migration silently rewrites.
The one managed Cloud SQL exception is the provider-owned `cloudsqladmin`
maintenance database: its database owner, complete ACL, `cloudsqladmin` and
`cloudsqlsuperuser` role attributes must exactly match the pinned catalog
fingerprint, and a fresh trusted preflight receipt must prove that direct TLS
login is rejected by `pg_hba`. The canonical receipt binds the same writer
host, port, CA-verified peer certificate SHA-256, user and credential path to
database `cloudsqladmin`, exact SQLSTATE `28000`, the exact PostgreSQL TLS HBA
rejection, observation/expiry times, and result. It is collected by root,
digest-verified and freshness-checked by deployment preflight. If this managed
exception is configured, writer startup repeats the active verified-TLS probe
on every process start and fails before opening the production session when the
receipt is stale, the certificate changes, the error differs, or login succeeds.
A name match, catalog match, arbitrary 64-hex value, or caller-supplied boolean
alone is insufficient; every other connectable non-template database remains
forbidden.

PostgreSQL 16 or newer is required because the owner boundary depends on
independent `ADMIN FALSE`, `INHERIT FALSE`, and `SET TRUE` membership options.
Routine
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
  discord_edge:
    enabled: true
```

The writer consumes a separate root-owned, secret-free strict JSON file. It
pins service identities, verified database coordinates, the writer-owned
credential path, Discord owner IDs, deployment lock, and exact routine
identities plus the reviewed private-schema identity SHA-256. Unknown fields,
embedded secret-shaped keys, mutable modes,
symlinks, untrusted parents, unpinned units, and shared gateway/writer UIDs are
rejected.

The same strict writer JSON must make Discord route-back authority explicit.
The enabled shape contains paths and a pinned public-key identifier, never key
material:

```json
{
  "discord_edge_authority": {
    "enabled": true,
    "capability_private_key_file": "/etc/muncho/credentials/discord-writer-capability-private.pem",
    "edge_receipt_public_key_file": "/etc/muncho/trust/discord-edge-receipt-public.pem",
    "edge_receipt_public_key_id": "<lowercase-ed25519-public-key-sha256>",
    "request_timeout_seconds": 15
  }
}
```

The capability private key is an unencrypted Ed25519 PKCS#8 PEM owned by the
writer UID/GID with exact mode `0400`. The pinned edge receipt key is an
Ed25519 SubjectPublicKeyInfo PEM owned by root, grouped to the writer, with
exact mode `0440`; its observed key ID must equal the configured digest. Both
must be regular non-symlink, single-link files at distinct absolute paths,
opened with no-follow semantics and re-attested by descriptor before parsing.
Production parent directories remain root-controlled and non-writable by
group/world. Key paths are accepted only from this explicit config—never from
environment variables, Secret Manager lookup, a caller payload, or automatic
discovery. The only valid disabled shape is
`{"discord_edge_authority":{"enabled":false}}`. If authority is enabled but
either key is missing, malformed, misowned, incorrectly permissioned, reused
for both roles, or does not match the pinned edge key ID, bootstrap fails
before exposing the writer socket. With authority disabled, route-back claims
remain fail-closed while unrelated typed writer operations can still run.

The projector has no database credential. A privileged one-shot writer job may
create an atomic, bounded, writer-owned event export for the unprivileged
projector group. Every page is read inside one PostgreSQL `SERIALIZABLE READ
ONLY` transaction/snapshot, so concurrent live commits cannot create a skipped
or internally inconsistent export. The exporter and timer are disabled by
default. If enabled,
preflight requires an exact one-shot argv, the same immutable release and
config, a dedicated output directory, an exact hardened unit/timer schedule,
and no other writer-UID cron, `at`, transient-unit, timer, or process surface.

## Packaged activation and readiness gate

The VM never imports the source-only `scripts.canary` namespace. The sealed
wheel provides the sole native-observation and final-activation entry point at
`gateway.canonical_writer_activation`, and the source planner exposes one
deployable v3 plan builder that requires the exact stopped native receipt.

The bootstrap approval receipt is an exact plan/scope/source/nonce/TTL binding,
but its trust description is deliberately narrow:
`authority_kind=trusted_root_bootstrap_out_of_band_owner` and
`cryptographic_owner_proof=false`. It records a root-provisioned out-of-band
owner confirmation; it does not claim a passkey or signature proof. Renewals
are append-only and receipt-addressed, so a new TTL/nonce never overwrites the
older evidence.

The complete host lifecycle is serialized under a root-owned lock. Before any
native identity change or final service start, the packaged preflight re-hashes
the release, manifest, units, configs, CA and credential provenance; performs
the exact TLS/PostgreSQL startup privilege attestation; verifies current NSS
identities and complete primary/supplementary group membership; rejects
systemd overrides/drop-ins, Discord authority and the retired helper; checks
the same-host stopped receipt and its append-only stage; and requires fresh
exact-policy IAM evidence. Passing and failing reports are sealed append-only
under the plan evidence root with separate report and file digests.

The fresh IAM receipt is archived before the replaceable `/run` copy can be
renewed. A large minimum lifetime is required before mutation, and the current
same-policy receipt is re-read and archived immediately before service start.
The root preflight receipt binds the exact IAM digest actually used.

Native live observation remains boot- and time-bounded through stage
finalization. The resulting stopped receipt is durable: a later consumer may
use it after its original boot/TTL only on the same host, after re-hashing the
current release/config/library closure and comparing current native/kernel
mapping policy. Cross-host replay and any artifact or mapping drift fail
closed. Successful replay performs no second service mutation.

Live canary services remain intentionally disabled until the installed-wheel
probe and this comprehensive in-process preflight both pass on the isolated
Cloud host. A preflight-only blocker is retryable and does not create a
forensic mutation quarantine; a failure after approved mutation begins seals a
unique failure receipt and the fixed quarantine marker.

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
   ephemeral PostgreSQL instance and execute all seventeen routines through the
   real wire client, including idempotent retries, CAS conflicts, provenance
   quarantine, completion verification, route-back partial receipts, legacy
   column ACLs, arbitrary grantee/function ACL drift, malformed preexisting
   writer objects, migration rollback, revocation tombstones, competing
   grant/consume/revoke transactions, cross-lane route-back claim races,
   preclaim terminal fencing, one-snapshot multi-page export, and concurrent
   transactions. Static parsing and Python fakes do not validate PL/pgSQL
   execution.
4. **Trusted root evidence collector.** Deploy the packaged, reviewed collector
   and verify its fresh gateway/writer code-closure, local-authority, systemd,
   polkit, IAM, secret-source, socket, and privileged-execution evidence on the
   isolated Cloud host.
5. **Legacy truth decision.** Owner-review and explicitly reseed accepted legacy
   events, or formally start a new trusted truth epoch. Until then legacy rows
   remain quarantined.
6. **Global Discord egress boundary.** Isolate the Discord token and route every
   outbound primitive through one public-channel/thread-only privileged egress
   process before asserting the no-DM rule globally.
7. **Owner-approved Cloud mutation plan.** Provision roles, immutable release,
   users, groups, config, CA, credential, systemd units, exporter state, IAM
   removals, migration, canary, and rollback under the exact out-of-band owner
   gate. Any future passkey-backed runtime gate must issue its own distinct
   cryptographic receipt.
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
   migration owner. Never provide the Cloud SQL HBA digest manually or invoke
   the SQL directly: the root-controlled probe-and-apply driver must actively
   collect the receipt with the writer host/port/CA/user/credential and inject
   only that canonical digest into the transaction.
3. Provision the immutable shared release, writer identity/config/credential,
   socket group, and exact hardened units. Keep dynamic Python disabled unless
   every extension is inside the immutable release.
4. Remove the legacy helper and all gateway Cloud SQL/Secret Manager authority.
5. Run the trusted root collector and require every preflight invariant to pass.
6. Start the writer, then the gateway.
7. Canary exact query/append, plan retry/CAS, restart-invalidated approval,
   public route-back sent receipt, accepted-unverified pending reconciliation,
   DM and private-thread rejection, child-PID rejection, writer outage
   behavior, projection export if enabled, and a multi-step task resume.

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
