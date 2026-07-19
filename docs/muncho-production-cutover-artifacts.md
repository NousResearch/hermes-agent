# Muncho production cutover artifacts

The production legacy-truth coordinator never executes loose SQL, imports a
mutable release module, or inherits an operator environment.  Every immutable
release now contains six generated, self-contained mechanical executables:

| Artifact | Accepted actions |
| --- | --- |
| `production-observe` | `observe_initial`, `observe_final_tail`, `observe_before_apply` |
| `production-database-apply` | `database_apply` |
| `production-database-rollback` | `database_rollback` |
| `production-database-postflight` | `database_preflight`, `database_terminal` |
| `production-host-activation` | `host_apply_stopped`, `host_start_prerequisites`, `host_start_writer`, `host_commit_boot` |
| `production-host-rollback` | `host_rollback` |

They are generated under
`ops/muncho/cutover/artifacts/` while the release is still staged. The deploy
helper verifies the canonical `manifest.json` before the release can become a
deployment target. Each executable embeds its disjoint action allowlist, the
reviewed legacy reconciliation SQL, and the reviewed Canonical Writer SQL.
There is no runtime import from the mutable checkout. The manifest records the
source and executable SHA-256 digests, sizes, actions, exact release revision,
and its own digest. The source set also binds the reviewed privileged Discord
connector unit template, gateway `BindsTo`/Relay drop-in, and connector config
template. The rendered live host bytes are separately bound by the signed
`host_transition` manifest because their release SHA, numeric service IDs, and
exact Discord guild/root-channel ACL allowlist are production inputs rather
than build-time facts.
Its `plan_bindings` object is already in the coordinator's exact
`{path, sha256}` shape for `observe`, `database_apply`,
`database_rollback`, `database_postflight`, `host_activation`, and
`host_rollback`; operators do not transcribe or recompute those values.

## Approval binding

The freeze plan binds the exact `production-observe` path and digest. The
cutover plan separately binds all six logical artifact slots. The coordinator
copies the approved bytes into its root-owned plan journal before execution.
The child then independently verifies:

- canonical request and plan self-digests;
- exact release revision and target fields;
- its own bytes against the applicable plan artifact SHA-256;
- its action against the action allowlist sealed into those bytes;
- the final stopped-tail row, content, relation/OID/owner, ACL, and index
  identities;
- an empty inherited environment and root/Linux execution identity; and
- absence of secret material from every response schema.

The owner must use the manifest paths and digests exactly when constructing the
freeze and cutover plans. A PR approval, canary receipt, manifest, or old chat
approval is not production mutation authority. The final signed approval is
issued only after the stopped final-tail receipt and the resulting exact plan
digest exist.

## OS Login pre-cutover gate

The production cutover transport intentionally refuses IAP/SSH unless the exact
production instance has `enable-oslogin=TRUE`, has no instance-level
`ssh-keys`, and the pinned owner profile contains the expected POSIX identity
and public key. A separate owner-signed metadata migration gate handles the
one-time transition; it is not folded into the cutover state machine.

Before its first metadata write, the gate binds and re-reads the exact project,
zone, instance ID, metadata fingerprint, project metadata fingerprint, owner
profile/key, and effective IAM decisions for instance read, metadata update,
OS Admin Login, and IAP tunnel access. Its only forward operations are setting
`enable-oslogin=TRUE` and removing the single instance `ssh-keys` entry. It
then reads the full metadata state back, proves unrelated metadata unchanged,
and runs the fixed `/usr/bin/true` probe through pinned IAP/OS Login. If any
mutation or access proof fails, it restores the exact prior `ssh-keys` value
and prior `enable-oslogin` state and verifies the full metadata map again. No
caller-supplied remote command or owner private key crosses the transport.

The fixed owner actions are deliberately split so the signed authority can be
reviewed before the metadata mutation:

```bash
python -m scripts.canary.production_cutover_owner_launcher \
  os-login-preflight \
  --revision <exact-40-char-release-sha> \
  --owner-private-key "$HOME/.ssh/skyvision_mac_ops_ed25519" \
  --output /absolute/owner/path/os-login-authority.json

python -m scripts.canary.production_cutover_owner_launcher \
  os-login-migrate \
  --revision <same-exact-release-sha> \
  --authority /absolute/owner/path/os-login-authority.json \
  --output /absolute/owner/path/os-login-receipt.json
```

The first action is Cloud read-only and accepts the existing unencrypted
Ed25519 OpenSSH owner key locally. The second action accepts only that exact,
self-hashed, signed authority bundle. Both actions reconstruct the production
transport from the release-pinned owner runtime and verify the launcher and OS
Login module against the same commit. The private key is neither printed nor
staged.

## Database boundary

Observe and postflight run serializable read-only transactions under the same
global advisory lock used by the writer migration. They connect only to the
plan's exact IP, verified TLS server name, database, port, and frozen source
owner. The CA and password transport live at fixed root-only paths:

- `/etc/muncho-production-cutover/cloudsql-server-ca.pem` (`0400` or `0444`);
- `/etc/muncho-production-cutover/pgpass` (`0400`).

Neither value, its digest, subprocess output, SQL text, nor database error text
is returned or journaled. Before database apply, the artifact performs a fresh
verified-TLS startup probe to the managed `cloudsqladmin` database and requires
the exact SQLSTATE `28000` HBA rejection. Only that fresh receipt digest is
injected into the writer transaction.

Database apply is resumable across three independently idempotent states:
legacy nineteen-column truth, reconciled fourteen-column truth with the moved
archive, and terminal Canonical Writer schema plus exact writer membership.
Every entry rechecks the signed final snapshot. Database rollback is a separate
artifact. It is permitted only while both the canonical table and archive still
match the approved frozen truth and writer provenance contains zero rows. It
then restores the original relation object (and therefore its OID, owner, ACL,
indexes, defaults, and five legacy columns) rather than reconstructing it.

## Host and privileged Discord boundary

Before the owner signs the cutover plan, every reviewed host target is staged as
root-owned mode `0400` files below
`/var/lib/muncho-production-legacy-cutover/staged/host/`:

- the production gateway unit and normal GPT-5.6 production config;
- the Canonical Writer unit;
- the exact-SHA-rendered privileged Discord connector unit;
- the reviewed gateway `BindsTo`/Relay/`UnsetEnvironment` drop-in; and
- the numeric-ID-rendered production `guild_acl` connector JSON config (the
  separate synthetic canary config remains `public_only`);
- the Phase-B, route-back, macOS edge, browser, and isolated-worker units and
  configs; and
- nine credential-scoped operational-edge units and configs plus the exact
  root-owned client map; and
- the root-only API bearer and approval verifiers.

The plan's self-hashed `host_transition` binds each staged and target path,
SHA-256, owner, group, mode, and exact pre-state. The gateway target service
identity also binds the drop-in path and digest. The generated executable
independently requires the connector unit to be an exact rendering of the
packaged reviewed template, requires the drop-in bytes to equal the packaged
source, validates the connector JSON's exact shape, mode, guild/root-channel/
user/role ID lists, and rejects Discord credential names in the staged gateway
unit or config.

### Two-stage host authority

The initial collector intentionally has no owner-authored input. It can report
only facts already observable on the production host: the verified release and
artifact bindings, the three current service identities, the legacy snapshot,
the cron inventory, and mechanical-rail host/package facts. It therefore
cannot safely invent the three target service identities, the target host
transition, the capability topology, or the owner-reviewed cron continuity
plan. Those are exactly the fields needed to turn an observation into full
cutover authority.

The release manifest now carries a self-hashed host-artifact contract covering
the exact 38-file host transition. Twenty-seven static runtime payloads have
their final byte digests sealed by the release package, and the reviewed static
gateway connector drop-in contributes one more package digest. The remaining
eight production-rendered unit/config outputs and two root-only verifier files
depend on owner-controlled live inputs, so packaging cannot truthfully know
their final bytes. Instead, the owner submits one canonical, self-hashed host
plan to the fixed read-only host-authority collector. That collector verifies
the release contract, reads back all 38 fixed staged files, compares all 38
target pre-states, validates the topology and executable cron plan, and returns
the full per-file evidence plus its aggregate digest. Any omitted, extra,
changed, wrongly owned, wrongly permissioned, or package-mismatched file fails
closed.

The owner-side workflow composes the initial and host receipts, signs the
resulting freeze authority locally, and only then performs its first mutation:
staging that signed freeze publication. Its order is fixed as initial
collection, host-authority collection, authority composition, freeze signing,
freeze staging, final-tail capture, stopped-state collection, cutover-plan
staging, Phase-B preflight, and apply. Before a cutover plan is staged, a
failure invokes the fixed `abort-freeze` recovery path. The private signing key
is never passed to the production transport.

The isolated-canary prerequisite is not hand-authored JSON. Build it from the
four immutable canonical public receipts with the edge author:

```bash
python -m scripts.canary.owner_gate_release_author \
  author-isolated-canary-prerequisite \
  --release-revision <exact-40-character-release-sha> \
  --fixture /absolute/immutable/fixture.json \
  --workspace-gateway /absolute/immutable/workspace-gateway.json \
  --cleanup-receipt /absolute/immutable/cleanup-receipt.json \
  --production-diff /absolute/immutable/production-diff.json
```

It reuses the production validator, derives the fixture digest itself, and
publishes only to the fixed mode-`0444` owner path
`~/.hermes/owner-gate-production-cutover/isolated-canary-prerequisites/<release-sha>.json`.
There is no output-path argument.

### Fixed host-authority plan production

The owner launcher does not accept the seven semantic host-authority fields as
JSON. Signed v3 unit inputs bind the exact identities, distinct public key IDs,
and the reviewed legacy-to-target Discord policy reconciliation. The fixed
root-side producer then:

- re-verifies the immutable release and renders or copies every contracted host
  artifact to its fixed create-only staging path;
- records only public verifier/key identities and never secret content or a
  secret digest;
- derives live user/group, target-file, token/passkey metadata, lease-directory,
  service-target, topology, and transition facts; and
- copies the initial collector's already owner-approved cron continuity plan
  byte-for-byte into the exact seven-field host-authority plan.

`stage-host-artifacts` is inert and cleanly resumable only for exact bytes.
`collect-host-plan` is read-only. The downstream host-authority collector still
re-reads every staged file, rejects target or boot drift, and binds the result
to the signed FreezePlan. Caller-authored `host_transition`, target identity,
or `capability_topology` remains outside the authority boundary.

### Rollback authority boundary

Rollback has three deliberately separate phases:

1. Before a cutover plan has been staged, `abort-freeze` may restart only the
   exact legacy gateway that the freeze stopped. It must attest that database
   authority, host state, tokens, and Caddy were never changed.
2. After plan staging but before the fsynced `activation_commit_intent`, the
   signed cutover transaction performs its own exact preimage rollback for the
   approved database, host, token, and service changes. Caddy is recovered by
   its separate signed journal/state machine; `abort-freeze` is no longer a
   valid recovery command.
3. The fsynced `activation_commit_intent` is the irreversible forward-only
   authority boundary. Recovery after it must never restore or route to v1.
   It may only converge to verified v2 or to the fixed 503 maintenance route
   while preserving the v2 authority database and mutation journal, followed
   by forward recovery.

The existing `muncho-auto-deploy-release run <SHA> <PR>` action remains valid
only before cutover while the loaded production unit still uses the legacy
mutable-release symlink topology. It is a reversible pre-cutover deploy, not a
cutover stage action and not a launch blocker. Once the SHA-pinned cutover
identity or any ambiguous cutover state is present, that helper remains
fail-closed; no new stage-only variant is introduced.

This layer deliberately does not invent a production gateway `ExecStart`.
The production model-sovereignty startup-contract renderer must supply the
normal GPT-5.6 agent loop + API/Relay + Canonical Writer target unit/config;
neither the writer-only `--require-canonical-writer` contract nor the bounded
`--require-capability-canary` contract is a valid production substitute. Its
exact output is handed to this layer through:

- `host_transition.files.gateway_unit` at staged
  `.../staged/host/hermes-cloud-gateway.service`; and
- `host_transition.files.gateway_config` at staged
  `.../staged/host/config.yaml`.

Host apply requires gateway, writer, and connector stopped. Before the first
replacement it records exact root-only backups for every target file. It
then installs only the signed bytes, atomically moves the ordinary-session
Discord credential from the plan's exact stopped-gateway source lease to the
connector-owned one-link mode-`0400` target, proves every other gateway token
path absent, and proves the separate route-back-only lease remains non-gateway
owned. Token content and token digests are never emitted or journaled. A crash
with source and target both present is resumable only when their bytes compare
equal internally; the source is retired before an apply receipt is possible.

The same stopped action re-reads every sealed operational helper and manifest,
installs the exact nine pre-staged Ed25519 receipt-key pairs, and starts only the
nine credential-scoped operational-edge services under the reboot fence. It
proves nine distinct non-root service UIDs/GIDs, nine distinct per-domain
socket groups, every systemd fragment, each Unix socket owner/group/mode, and a
fresh boot-bound readiness receipt collected through the real signed socket
protocol. Every service is a member of only its own socket group; its config
admits only the gateway UID, its state directory is mode `0700`, and its
systemd credential projection contains only that domain's leases. The root
publisher drops the probe subprocess to the gateway UID/GID with exactly the
nine client groups—never to an edge identity—so a compromised edge cannot
invoke a sibling socket or read a sibling state/credential projection.
Gateway, writer, Discord connector, and the normal prerequisite services remain
stopped throughout this isolated canary gate.

The second host action starts the connector first and requires its in-process
readiness before starting the writer. It never starts the gateway. The
coordinator starts the gateway only after database postflight; the concrete
service observer then fails closed if the live gateway process has a Discord
token environment variable, an open connector/route-back credential file, a
root process identity, or a gateway-owned privileged token lease. Terminal
evidence requires gateway, writer, and connector active with the exact target
unit/drop-in identities.

Rollback requires all three services stopped, accepts only exact target or
already-restored file identities, restores every exact backup, moves the
ordinary-session token back to its exact pre-cutover owner/path without
recording it, reloads systemd, and proves all three legacy identities stopped.
It is intentionally fail-closed. The rollback contract sets
`restart_legacy_gateway=true`, but only after exact legacy host/database
identities have been restored and the DM-safe legacy boundary has been
revalidated.

Rollback also stops and removes the exact nine operational-edge units, removes
the published readiness receipt, and verifies that both staged and final key
copies remain exact but dormant. It never regenerates, replaces, exports, or
deletes those private keys. The nine service identities and nine socket groups
remain dormant, with every pre-existing membership restored and no cross-domain
membership widening.

## Cron continuity and mechanical rail

The read-only initial receipt also binds the exact legacy cron inventory,
redaction-safe root metadata for `/usr/bin/gh`, `/usr/bin/git`, and the GitHub
credential file, plus the exact release-addressed mechanical-job package
manifest. No credential value, size, or digest is recorded. Before a freeze
plan can be authored, the owner continuity plan must account for every exact
record. The legacy upstream-sync job may only be replaced by the matching
packaged rail manifest; a blanket inert migration is non-executable. The
cutover authority refuses to stop production unless the reviewed continuity
plan is explicitly executable.

## Live prerequisites

Packaging closes the missing executable boundary but does not fabricate live
facts. Before production execution, collect and owner-review:

1. the release manifest and six exact artifact digests;
2. the stopped final-tail receipt and exact Cloud SQL target/TLS identity;
3. an exact least-privilege migration owner, writer role, and enabled writer
   login with no unrelated authority;
4. the root-only CA/password transport and every staged host target file;
5. the reviewed production model-sovereignty gateway unit/config producer and
   exact SHA-bound output (writer-only/canary startup modes are forbidden);
6. connector user/group, credential directory, pre-initialized connector
   journal, exact ordinary-token source lease, and distinct preserved
   route-back-only lease;
7. trusted preflight evidence for every remaining non-file host prerequisite;
8. an exhaustive owner-reviewed cron continuity plan and matching mechanical
   rail package manifest;
9. successful clean-room and production-shaped canary evidence; and
10. a fresh out-of-band signature over the final cutover plan.

The cutover authority v3 embeds the isolated-canary prerequisite v2 rather
than accepting a terminal digest by itself. The owner signature therefore
binds the complete reviewed fixture, signed goal-continuation gateway
envelope, signed cleanup observer envelope, and canonical native
`production-diff.json`. The verifier rechecks the cleanup receipt's native
`production_diff_observation` binding and requires the same diff digest in the
terminal, cleanup receipt, run, release, fixture, capability plan, full-canary
plan, and owner approval. Legacy prerequisite shapes and missing evidence have
no compatibility bypass.

Promotion also requires byte-exact equality for the semantic configuration,
ordered toolsets, and capability-role/service topology. Only the reviewed
canary-versus-production Discord channel ID is mechanically normalized; guild,
roles, service units, model route, `goals.max_turns=0`, Kanban-off policy,
privileged-writer boundary, and DM/direct-Discord prohibitions must remain
exact.

The two no-mutation statements are intentionally separate. The owner-bound
native canary diff proves zero production mutation before production cutover
intent. After host files, cron continuity, and prerequisite services are
staged, those production lifecycle mutations are explicitly acknowledged; the
pre-database receipt v2 claims only
`canonical_database_mutation_observed=false`, backed by the stopped writer,
unchanged frozen legacy snapshot, and still-legacy schema. Database apply
cannot run until that receipt and `capability_prerequisites_validated` are in
the append-only journal.

No production service, database, secret, or Cloud resource is changed by the
packager or its tests.
