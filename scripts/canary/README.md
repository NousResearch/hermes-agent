# Isolated Cloud Muncho canary

These fork-only helpers build the temporary privileged-writer/Discord canary in
three digest-bound phases. The canary does not reuse the production VM,
network, subnet, service account, Cloud SQL instance, Hermes home, Discord bot,
or credentials.

No command accepts or prints a secret value. In particular, the canary never
creates `muncho-canary-db-password` or `muncho-canary-discord-bot-token` in the
shared project's Secret Manager. The production runtime identity already has a
project-wide Secret Accessor grant; a resource-level policy cannot deny an IAM
grant inherited from the project. A new canary secret in that project would
therefore be readable by production even if the secret itself had no explicit
production binding. Both forbidden names must remain absent.

The database password and distinct Discord application token are instead
owner-provisioned directly to the privileged host identities as root/systemd
credentials outside shared-project Secret Manager. The database credential's
host shape is `/etc/muncho/credentials/canonical-writer-db-password`, owned by
the pinned `muncho-canonical-writer` identity with mode `0400`; this document
never contains its value.

## Phase 1 — dedicated foundation

```bash
python -m scripts.canary.foundation plan
python -m scripts.canary.foundation_preflight
python -m scripts.canary.foundation apply \
  --preflight /root/approved-canary-foundation-preflight.json \
  --approved-plan-sha256 <exact-plan-sha256>
```

Phase 1 creates only:

- custom VPC `muncho-canary-vpc`;
- regional subnet `muncho-canary-europe-west3` (`10.90.0.0/24`);
- private Service Networking range `muncho-canary-sql-range`
  (`10.91.0.0/24`) and its single Google-managed peering;
- runtime service account `muncho-canary-v2-runtime`, with only Logging Writer,
  Monitoring Metric Writer, and the conditional custom
  `munchoCanaryCloudSqlReadinessV1` role. The custom role contains only
  `cloudsql.instances.get` and its IAM condition binds it to
  `muncho-canary-pg18-v2`; the identity has no user-managed keys;
- private-only PostgreSQL 18 instance `muncho-canary-pg18-v2` and database
  `muncho_canary_brain`.

The only permitted peering is `servicenetworking-googleapis-com`. There is no
peering or route to `ai-platform-vpc` or its `10.80.0.0/24` production subnet.
Preflight permits only the generated internet-default, local-subnet, and exact
Service Networking routes. The SQL private address must be inside
`10.91.0.0/24`.

Phase 1 cannot create a VM, firewall rule, or Secret Manager resource.

## Phase 2 — network boundary

After phase 1 is re-attested with every step satisfied:

```bash
python -m scripts.canary.network_boundary plan --sql-private-ip <private-ip>
python -m scripts.canary.network_preflight
python -m scripts.canary.network_boundary apply \
  --sql-private-ip <private-ip> \
  --preflight /root/approved-canary-network-preflight.json \
  --approved-plan-sha256 <exact-network-plan-sha256>
```

Phase 2 creates exactly three logged rules on the dedicated VPC:

- IAP SSH ingress from `35.235.240.0/20` to tagged TCP 22;
- priority 800 egress allow from the runtime service account to only the exact
  SQL private `/32` on TCP 5432;
- priority 900 egress deny from that identity to all other RFC1918 addresses.

The preflight reads Compute Engine's effective-firewall view, not merely the
project's ordinary firewall-rule list. Any other applicable VPC, network
firewall policy, folder policy, or organization policy rule fails closed. Rules
explicitly targeted at unrelated identities or tags are ignored.

## Phase 3 — host, only after live firewall proof

Only a fresh phase-2 report with all three exact rules satisfied can authorize
the VM plan:

```bash
python -m scripts.canary.host plan \
  --sql-private-ip <private-ip> \
  --network-plan-sha256 <exact-network-plan-sha256>
python -m scripts.canary.host_preflight
python -m scripts.canary.host apply \
  --sql-private-ip <private-ip> \
  --network-plan-sha256 <exact-network-plan-sha256> \
  --preflight /root/approved-canary-host-preflight.json \
  --approved-plan-sha256 <exact-host-plan-sha256>
```

This phase has one mutation: create `muncho-canary-v2-01` as an `e2-medium`
Shielded VM with the pinned Debian 12 image, 40 GB balanced disk, exact
`cloud-platform` OAuth scope, OS Login, disabled legacy metadata endpoints, and
only the `iap-ssh` tag. Effective authority remains limited by the exact three
project IAM bindings attested in phase 1; the broad predefined Cloud SQL Viewer
role is forbidden. It has an ephemeral premium-tier public address for IAP
reachability and no deletion protection because it is a temporary canary.

The foundation and network plans contain no VM-create command. The host
preflight nests a fresh effective-firewall attestation and requires all network
steps satisfied; consequently the VM cannot be created before the firewall
boundary is live. Every apply receipt still requires a post-apply read-only
attestation before the next phase or runtime enablement.

### One-time existing-host storage transition

The already-created v2 canary predates the 40 GB host contract and has one
identity-pinned 20 GB boot disk. Its one-way transition is a separate bounded
gate. The resize half does not deploy code, start a service, create a snapshot,
run a guest command, or change IAM:

```bash
python -m scripts.canary.host_storage plan
python -m scripts.canary.host_storage_preflight \
  > /root/approved-canary-host-storage-preflight.json
python -m scripts.canary.host_storage apply \
  --preflight /root/approved-canary-host-storage-preflight.json \
  --approved-plan-sha256 <exact-storage-plan-sha256> \
  > /root/canary-host-storage-apply-receipt.json
```

The source preflight permits 20 GB only for VM instance ID
`9153645328899914617` and disk ID `4195397669213846393`, with the exact normal
`persistent-disk-0` boot attachment, and the exact normal host shape failing
solely because its canonical target is now 40 GB. The apply
contains exactly one `gcloud compute disks resize` command, fixed to the owner
account, project, zone, disk, and 40 GB target. It returns hashes of command
output, never raw provider output.

Guest evidence reuses the owner launcher's sealed IAP transport material:
`google_compute_engine`, its matching public key, and the exact
`google_compute_known_hosts` entry for the pinned VM ID. Before and after the
read, the collector proves that same public key is already present in the
owner's OS Login profile and that instance/project authorization did not
change. It uses `gcloud compute ssh --plain`, never imports or uploads a key,
and fails closed before disk mutation if the existing profile does not match.
The separate `skyvision_mac_ops_ed25519` key remains owner-signing authority;
it is not SSH transport material.

Live execution established the narrower contract for the pinned Debian image:
the disk resize is online, but its `google-disk-expand` path expands the root
partition and ext4 filesystem during the next boot. It did not expand the
online root immediately. The storage plan remains byte-identical to the
already approved resize-only plan; a separate digest-bound boot plan now owns
the conditional stop/start and makes the boot dependency explicit.

Collect the boot preflight only after the resize receipt exists and a fresh
storage report is exactly `transition_pending`. It proves the fixed VM and disk
IDs, the source-sized root on the 40 GB disk, the current boot digest, and the
complete runtime-unit inventory. Every canary runtime unit must be absent or
disabled and inactive before the VM can stop:

```bash
python -m scripts.canary.host_storage_boot plan
python -m scripts.canary.host_storage_boot preflight \
  --storage-apply-receipt /root/canary-host-storage-apply-receipt.json \
  > /root/canary-host-storage-boot-preflight.json
python -m scripts.canary.host_storage_boot apply \
  --preflight /root/canary-host-storage-boot-preflight.json \
  --approved-plan-sha256 <exact-storage-boot-plan-sha256> \
  > /root/canary-host-storage-boot-receipt.json
```

The boot executor contains only the exact identity-pinned `gcloud compute
instances stop` and `start` calls through the sealed owner SDK. It grants no
`growpart`, `resize2fs`, guest shell, cleanup, repair, or service-start
authority. Before `stop`, it publishes and fsyncs an owner-only, append-only
intent below
`/Users/emillomliev/.hermes/canary-storage-boot-transactions`. The transaction
ID binds the approved plan, preflight, resize receipt, VM/disk IDs, prior boot,
and stopped service-state digest. Every stop/start observation and command
completion is then appended with its own self-digest and monotonic chronology.
A pre-existing `TERMINATED` VM is never enough to authorize `start`: recovery
requires that exact unexpired pre-stop intent. An unrelated or stale stopped
state fails closed. A crash after either command can only append the evidence
that remains observable; it cannot manufacture a missing command receipt. Once
the terminal receipt is durable, retries replay it without another observation
or mutation.

After the boot, collect a new report until normal host preflight passes exact
40 GB and the read-only guest fact proves `/dev/sda1` is ext4 at `/`, the
filesystem exposes at least 39,000,000,000 bytes, and at least 8 GiB remains
available. Bind the resize, boot preflight, boot receipt, and terminal
postflight into one readiness receipt:

```bash
python -m scripts.canary.host_storage_preflight \
  > /root/canary-host-storage-postflight.json
python -m scripts.canary.host_storage attest \
  --apply-receipt /root/canary-host-storage-apply-receipt.json \
  --boot-preflight /root/canary-host-storage-boot-preflight.json \
  --boot-receipt /root/canary-host-storage-boot-receipt.json \
  --postflight /root/canary-host-storage-postflight.json \
  > /root/canary-host-storage-readiness-receipt.json
```

Only `muncho-isolated-canary-host-storage-readiness-receipt.v2` with a valid
self-digest and the required boot-receipt binding establishes storage
readiness after a 20 → 40 GB mutation. An already target-ready 40 GB host stays
idempotent and does not manufacture a boot dependency. Readiness explicitly
does not open a runtime gate; the remaining stopped-release and canary
preflights still apply.

The live v2 canary completed its historical resize and reboot before the
journaled boot contract existed. Do not reboot it again to fabricate missing
history. Its one exact reconciliation path accepts only the three fixed
owner-only archives and a newly collected target-ready postflight:

```bash
python -m scripts.canary.host_storage_preflight \
  > /root/canary-host-storage-postflight.json
python -m scripts.canary.host_storage attest \
  --apply-receipt /Users/emillomliev/.hermes/owner-approvals/canary-storage/apply-b2ada08f473cf67dee9c738852373d19d9dba473fdc8ee3cdf834f14d951dbd5.json \
  --legacy-boot-receipt /Users/emillomliev/.hermes/owner-approvals/canary-storage/boot-expansion-7430a8e859c6f24261ef89182dc49c64c2f5832b50e8918bdae8008b8c0a0cb8.json \
  --legacy-readiness-receipt /Users/emillomliev/.hermes/owner-approvals/canary-storage/readiness-de2da85103d578073b795828fbad2db2a602d63cbb079352e9c90e74d6400777.json \
  --postflight /root/canary-host-storage-postflight.json \
  > /root/canary-host-storage-readiness-receipt-v2.json
```

This special receipt records
`boot_evidence_kind=legacy_live_receipt_reconciliation`, binds both legacy
receipt hashes, and states `preboot_boot_id_evidence=false` and
`preboot_service_state_evidence=false`; those archived facts do not exist. It
also keeps `opens_runtime_gate=false` and requires the follow-on stopped-runtime
gate. Journaled and legacy evidence arguments are mutually exclusive.

### One-time 40 -> 80 GB storage growth

The completed 20 -> 40 GB contracts above remain historical evidence.  Any
later expansion uses the canonical passkey-v2 storage-growth gate documented
in [host-storage-growth.md](host-storage-growth.md).  The local
`host_storage_growth` module is only a pure plan/read-only validation facade;
it has no approval, journal, recovery, or mutation authority.  The passkey-v2
executor pins the current bc37 host/stopped receipts, external IAM digest, and
historical 40 GB readiness; requires the complete twelve-unit stopped runtime
superset; performs one exact 40 -> 80 GB disk resize; and journal-reboots only
when a fresh read-only post-resize observation proves boot expansion is
required.  Its readiness receipt requires at least 84,000,000,000 root bytes
and 32 GiB free and never opens a runtime gate.

The gate is callable only through the exact release-bound full owner launcher;
its direct module entry points fail closed.  The local launcher has read-only
source-observation authority and no gcloud storage mutator.  Production
approval is an exact-action, single-use WebAuthn/passkey grant from Emil, not
the local UID-501 SSH key, legacy same-UID Cloud guard, or same-UID TOTP
fallback.  The initial passkey is live for at most five minutes and, once
atomically consumed, opens only the plan-pinned 3600-second execution window.
An expired incomplete mutation needs a domain-separated passkey resume request
built from the privileged executor's authoritative append-only stage and fresh
versioned live/IAM projection; already-proven commands are reconciled
mechanically without being repeated.

Do not operate this gate until the dedicated private `muncho-owner-gate-01`
attests split no-shell passkey/executor identities, root-owned code and units,
passkey-only approval, a complete canonical payload/full-hash UI, strict
path/token validation, atomic challenge/grant claims, a durable append-only
ledger, and a narrow attached service account inaccessible to the shared
gateway.  That executor owns the mutation credential, journal, and every Cloud
mutation; local files and local gcloud are never replay authority or fallback.

### Owner-side foundation author-and-apply entrypoint

The initial nine-operation owner-gate foundation is an owner-side,
pre-foundation mutation and is intentionally excluded from the Cloud owner-gate
package, `ROOT_RUNTIME_FILES`, and `REQUIRED_ENTRYPOINTS`.  Run the existing
entrypoint directly by its absolute path inside the reviewed, immutable
release-bound owner-support tree. Replace the SHA placeholder once with the
exact reviewed 40-character release SHA; do not use `-m`, a worktree path, a
relative path, a symlink, or caller-authored evidence/key/journal flags:

```bash
/Users/emillomliev/.local/share/uv/python/\
cpython-3.11.15-macos-aarch64-none/bin/python3.11 \
  -I -S -B \
  /Users/emillomliev/.hermes/trusted/\
owner-support-<exact-40-character-release-sha>/source/scripts/canary/\
owner_gate_author_and_apply.py \
  author-and-apply
```

This is the sole production Foundation operation. It derives the release revision and
source tree only from the sealed owner-support path and manifest, loads the
fixed pinned release/network keys and trusted gcloud runtime, performs the
interactive owner gcloud reauthentication, collects the fixed interpreter,
network, and organization-ancestry evidence, mechanically authors the exact
nine-operation plan, and invokes the fixed foundation apply boundary. There is
deliberately no SHA, private-key, runtime, provider, journal, clock, evidence,
or output-path argument. The append-only outer journal reconciles the fixed
inner journal before recovery; an ambiguous partial mutation becomes a durable
manual-reconciliation block and is never skipped in favor of fresh authoring.
Stdout contains only the stable terminal envelope, never a bearer token, key
material, raw evidence, or signed receipt body.

`success.json` is a signed journal transition wrapper, not a raw foundation
apply receipt.  Downstream owner gates must not copy it, loosen its mode, or
extract `.receipt` with `jq`.  They receive the opaque
`ValidatedFoundationAChain` and call
`load_validated_foundation_apply_chain(foundation_a)`.  That read-only boundary
derives the transaction ID and fixed journal path, performs no pending-file
recovery or write, verifies the transition signature/domain and exact chain,
rejects any conflicting failure terminal, validates the nested signed apply
receipt, and returns the opaque `ValidatedFoundationApplyChain` capability.
There is no caller-selected journal path, raw receipt, output path, or semantic
field in this handoff.

### Inert owner-gate HOST/CLOUD observation

The owner-side release author is a separate edge command. It never accepts a
private key or a caller-selected publication path. First publish an exact clean
`main` checkout whose sole `origin` is the reviewed fork and whose
`origin/main`, `HEAD`, tree, worktree, submodules, and outer-stage-zero sources
all resolve to the release SHA:

```bash
python -m scripts.canary.owner_gate_release_author \
  publish-release-source \
  --source-root /absolute/clean/fork-main \
  --release-revision <exact-40-character-release-sha>
```

This performs a local `--no-hardlinks --no-checkout` clone, checks out the
exact SHA detached, resets and re-verifies the fixed fork origin, and publishes
with one atomic no-replace rename at
`~/.hermes/trusted/owner-gate-release-sources/<release-sha>`. A conflicting
destination or fixed `.pending` path fails closed.

After Foundation A, the same edge author can build the canonical unsigned
release trust manifest from the exact offline package inventory, the one
Foundation collector key used by both network and ancestry evidence, the three
distinct final collector keys, and the signed credential-migration envelope:

```bash
python -m scripts.canary.owner_gate_release_author \
  author-unsigned-trust \
  --source-root /absolute/clean/fork-main \
  --release-revision <final-release-sha> \
  --wheelhouse-root /absolute/immutable/wheelhouse \
  --wheelhouse-manifest /absolute/immutable/wheelhouse-manifest.json \
  --interpreter-sha256 <sha256> \
  --foundation-source-revision <foundation-release-sha> \
  --foundation-source-tree-oid <foundation-tree-oid> \
  --pre-foundation-authority /absolute/immutable/pre-foundation.json \
  --owner-reauth-receipt /absolute/immutable/owner-reauth.json \
  --network-evidence /absolute/immutable/network-evidence.json \
  --foundation-collector-public-key /absolute/immutable/foundation.pub \
  --project-ancestry-evidence /absolute/immutable/project-ancestry.json \
  --direct-iam-identity-authority /absolute/immutable/direct-iam.json \
  --network-collector-public-key /absolute/immutable/final-network.pub \
  --cloud-collector-public-key /absolute/immutable/final-cloud.pub \
  --host-collector-public-key /absolute/immutable/final-host.pub \
  --credential-migration-envelope /absolute/immutable/credential.json
```

The fixed output is
`~/.hermes/owner-gate-release-authority/manifests/<release-sha>.trust.unsigned.json`
at mode `0444`. Sign it only through the same fixed release author, repeating
the exact inputs used to author it:

```bash
python -m scripts.canary.owner_gate_release_author \
  sign-trust \
  --source-root /absolute/clean/fork-main \
  --release-revision <final-release-sha> \
  --wheelhouse-root /absolute/immutable/wheelhouse \
  --wheelhouse-manifest /absolute/immutable/wheelhouse-manifest.json \
  --interpreter-sha256 <same-sha256> \
  --foundation-source-revision <same-foundation-release-sha> \
  --foundation-source-tree-oid <same-foundation-tree-oid> \
  --pre-foundation-authority /absolute/immutable/pre-foundation.json \
  --owner-reauth-receipt /absolute/immutable/owner-reauth.json \
  --network-evidence /absolute/immutable/network-evidence.json \
  --foundation-collector-public-key /absolute/immutable/foundation.pub \
  --project-ancestry-evidence /absolute/immutable/project-ancestry.json \
  --direct-iam-identity-authority /absolute/immutable/direct-iam.json \
  --network-collector-public-key /absolute/immutable/final-network.pub \
  --cloud-collector-public-key /absolute/immutable/final-cloud.pub \
  --host-collector-public-key /absolute/immutable/final-host.pub \
  --credential-migration-envelope /absolute/immutable/credential.json
```

`owner_gate_trust_author` has no `sign` CLI action; its only CLI action is
`init-key`. `owner_gate_release_author sign-trust` re-authors and compares the
unsigned manifest before signing, revalidates the complete signed Foundation
apply journal lineage and package inventory, and writes only the fixed
`<release-sha>.trust.json` path. It accepts neither a Foundation apply receipt
path nor an output path.

The first combined inert observation is a separate exact action on the
release-bound full owner launcher. It does not replay the Foundation operation
and accepts no evidence, key, plan, journal, stream, output, or remote-command
argument. Package *authoring* remains separate. The fixed owner-side input
preparer validates an already-materialized package and publishes exactly these
owner-owned, mode-`0400`, single-link regular files below an owner-owned
mode-`0700` release directory:

```text
~/.hermes/owner-gate-inert-observation-inputs/<release-sha>/
  stream-pins.json
  outer-stage0.tree-stream
  owner-gate-bundle.tree-stream
```

`stream-pins.json` is canonical JSON with the exact schema
`muncho-owner-gate-inert-observation-input-pins.v1`, the release SHA, the
outer-stage0 kit release ID, both tree-manifest SHA-256 digests, and its own
`pins_sha256`. It also pins both complete stream SHA-256 digests, so a
same-size payload substitution is rejected before credentials or IAP. The
launcher derives every path and filename from the reviewed
release; symlinks, extra links, wrong owners or modes, changed files, and
caller-selected alternatives fail closed. It resolves the one
cryptographically validated successful outer Foundation transaction, then
loads Foundation B only through
`load_validated_foundation_apply_chain(foundation_a)`.

The preparer accepts no path. Before it can write anything, these two exact
owner-only prerequisites must already exist:

```text
~/.hermes/trusted/owner-gate-release-sources/<release-sha>/
~/.hermes/trusted/owner-gate-offline-bundles/<release-sha>/
```

The release source root is mode `0700`, has exact `HEAD=<release-sha>`, has no
tracked or untracked change, and its Git tree must equal the source tree in the
sealed release-bound owner-support manifest. The bundle root is the immutable
mode-`0555` result of the reviewed `owner_gate_package.materialize_bundle`
builder. Its inventory is closed: every directory is mode `0555`; every file
has the exact builder mode and is owner-owned and single-link; the signed
release-trust manifest, its bound Foundation apply receipt digest, direct-IAM
authority, package manifest, payload/wheel inventory, collector keys, and
credential-migration envelope must all verify. This action does not download,
author, repair, or select a bundle.

First run the read-only prerequisite check. It creates neither the input root
nor a lock or pending directory and reports the exact missing or invalid fixed
prerequisite:

```bash
/Users/emillomliev/.local/share/uv/python/\
cpython-3.11.15-macos-aarch64-none/bin/python3.11 \
  -I -S -B -X pycache_prefix=/var/empty/muncho-canary \
  /absolute/clean/hermes-agent/scripts/canary/full_canary_owner_launcher.py \
  --release-sha <exact-40-character-release-sha> \
  --preflight-owner-gate-inert-inputs
```

Then publish the two reviewed exact-tree streams and their canonical
self-hashed pins with the pathless action:

```bash
/Users/emillomliev/.local/share/uv/python/\
cpython-3.11.15-macos-aarch64-none/bin/python3.11 \
  -I -S -B -X pycache_prefix=/var/empty/muncho-canary \
  /absolute/clean/hermes-agent/scripts/canary/full_canary_owner_launcher.py \
  --release-sha <exact-40-character-release-sha> \
  --prepare-owner-gate-inert-inputs
```

The preparer fsyncs each file and the private pending directory before one
atomic no-replace directory rename, then fsyncs the parent and loads the result
through the same pinned reader used by the observation. A valid final directory
is an exact replay and is never rebuilt. A crash after the rename is therefore
recoverable by replay. Any `.pending` directory means publication did not reach
the atomic boundary and requires manual reconciliation; it is never removed or
silently resumed.

After the preparation receipt succeeds, run:

```bash
/Users/emillomliev/.local/share/uv/python/\
cpython-3.11.15-macos-aarch64-none/bin/python3.11 \
  -I -S -B -X pycache_prefix=/var/empty/muncho-canary \
  /absolute/clean/hermes-agent/scripts/canary/full_canary_owner_launcher.py \
  --release-sha <exact-40-character-release-sha> \
  --observe-owner-gate-inert
```

The action runs one fixed stage-zero composite, obtains the signed HOST report
and attached-service-account permission probe, authors the read-only Cloud
snapshot, sends only that canonical unsigned Cloud report to the fixed target
UID-29103 signer, and immediately consumes the opaque bound pair in the inert
preflight. It atomically publishes the exact activation-ready objects below the
fixed owner-only append-only namespace; there is no output-path or filename
argument:

```text
~/.hermes/owner-gate-inert-observation-evidence/<release-sha>/inert/
  <evidence-set-sha256>/
    inert-cloud-observation.json
    inert-host-observation.json
    inert-preflight.json
    receipt.json
```

The transaction directory is mode `0500`; its four single-link files are mode
`0400`. Every evidence digest is bound into the self-hashed receipt and the
evidence-set directory name. Files and directories are fsynced before one
atomic no-replace publication. A partial `.pending` transaction, an unexpected
name, wrong ownership or mode, link alias, substitution, or conflicting fresh
transaction requires manual reconciliation and is never silently removed.
A valid fresh transaction is revalidated from its signed observations and
replayed byte-for-byte without another IAP collection; valid stale history is
retained and a new transaction is collected.

Stdout is only the compact canonical secret-free receipt and digest inventory,
not the three evidence documents. It never contains an access token, private
key, generic command, or mutable pair.

### Fixed owner-gate activation-seal install

After `--stage-owner-gate-activation-evidence` has successfully staged and
validated the complete release-bound evidence set, install the host activation
seal with the sole fixed action:

```bash
/Users/emillomliev/.local/share/uv/python/\
cpython-3.11.15-macos-aarch64-none/bin/python3.11 \
  -I -S -B -X pycache_prefix=/var/empty/muncho-canary \
  /absolute/clean/hermes-agent/scripts/canary/full_canary_owner_launcher.py \
  --release-sha <exact-40-character-release-sha> \
  --install-owner-gate-activation-seal
```

There is no host, path, command, seal, receipt, or payload argument. The
launcher uses only the signed numeric identity and host key for
`muncho-owner-gate-01`, the pinned owner account, the sealed gcloud runtime,
IAP, and the complete fixed SSH option tuple. The only permitted remote command
is:

```text
/usr/bin/sudo --non-interactive --user=root -- /opt/muncho-owner-gate/current/venv/bin/python -I -B /opt/muncho-owner-gate/current/bin/muncho-owner-gate-activate-storage install
```

The remote release validates the already-staged evidence before installing or
exactly replaying `/etc/muncho-owner-gate/storage-executor-enabled` and the
release-addressed receipt below
`/var/lib/muncho-owner-gate/activation-receipts/`. Before any mutation, it also
requires the two-field canonical release request sent on stdin by the fixed
launcher and rejects it unless that SHA equals its own resolved immutable
release. The launcher accepts only one canonical newline-terminated
`muncho-owner-gate-storage-activation-response.v1` result whose release equals
the requested SHA, disposition is `installed` or `exact_replay`, seal and
receipt paths and SHA-256 values are exact, the service contract is accepted,
`cloud_mutation_performed=false`, and `response_sha256` verifies. This is the
single bounded host activation mutation; it grants no generic SSH or Cloud
mutation surface.

## Later host/runtime gates

1. Install one root-owned immutable release by exact 40-character Git SHA;
   gateway and writer execute that path directly, never a mutable symlink.
2. Create distinct gateway, Canonical writer, socket-client, projector, and
   Discord-egress identities and install the reviewed pinned units.
3. Discover the Cloud SQL certificate DNS SAN over the private IP, verify it
   against the downloaded server CA, and install a root-owned exact hostname
   mapping. The writer uses that SAN as `database.host`; the raw IP fails TLS
   hostname verification.
4. Create the migration owner, group role, and sole runtime login; revoke
   `PUBLIC` authority; apply the reviewed migration twice; then attest all
   routines and the private-schema digest.
5. Use a distinct Discord application and an allowlisted public canary channel.
   The token belongs only to the privileged egress unit. The gateway uses the
   authenticated local relay boundary and never loads the Discord token. DMs
   remain mechanically blocked.
6. Run the fresh root collector and deterministic deployment preflight before
   gateway, writer, or Discord egress is enabled.

The package supply-chain path has a rerunnable real-platform check. It uses a
digest-pinned Debian 12 amd64 container, verifies the exact runtime-lock
artifacts, disconnects Docker networking, corrupts the disposable host pip
wheel, and then exercises the signed bootstrap install, inventory/replay, and
hostile `.pth` rejection:

```bash
.venv/bin/python scripts/canary/run_owner_gate_debian12_e2e.py
```

## Stopped release publication

The sealed release is published through the local owner launcher, never with a
Cloud-side `python -c`, heredoc, shell fragment, or caller-selected SSH command.
The action is a mutation because it creates a revision-addressed root-owned
source checkout, managed Python, virtual environment, wheel, manifest, and
append-only publication evidence. It does not install, start, enable, stop, or
restart a service. It uses the attested owner gcloud identity for IAP access but
does not request or disclose a database, Discord, or runtime credential.

After the fork revision has passed review and CI, first create its local
release-bound trusted gcloud receipt as described below. Then invoke the same
attested launcher with the explicit stopped-release action:

```bash
/Users/emillomliev/.local/share/uv/python/\
cpython-3.11.15-macos-aarch64-none/bin/python3.11 \
  -I -S -B -X pycache_prefix=/var/empty/muncho-canary \
  /absolute/clean/hermes-agent/scripts/canary/full_canary_owner_launcher.py \
  --publish-stopped-release \
  --release-sha <exact-40-character-release-sha>
```

The launcher is fixed to the fork repository, the dedicated v2 canary VM, the
existing `/opt/muncho-canary-source/<revision>` source namespace, and
`/opt/muncho-canary-releases/<revision>`. It first obtains a deterministic
read-only `muncho-canary-stopped-release-plan.v1`, then returns that exact plan
digest to the root/Linux `apply` entry point. `apply` re-observes the dedicated
GCE/host/boot identity, the complete fixed activation inventory, and the
stopped service states before writing. It builds only a clean root-owned checkout whose
HEAD and origin are exact, validates the sealed result, and publishes or
same-boot revalidates the trusted host receipt at
`/etc/muncho/full-canary/host-identity.json` as root-owned mode `0400`. The
root-owned `0755` parent deliberately permits the future gateway identity to
traverse to its separately protected `0440` observer and fixture files. The
gate then re-observes the complete stopped plan; any drift blocks publication
evidence. The host receipt file and internal digests are bound into the
root-owned mode `0400` publication receipt at:

```text
/var/lib/muncho-canary-release-evidence/<revision>/stopped-release-publication.json
```

An unfamiliar source, release, activation path, unit state, receipt, incomplete
build, or cross-revision collision is preserved and blocks the gate. No generic
cleanup or repair authority is part of this action. An exact terminal retry may
only revalidate the same source, sealed release, live stopped state, and receipt
bytes.

## Phase 4 — stopped writer activation through the owner launcher

The operator entry point is the local, release-bound owner launcher. Do not run
the packaged VM modules by hand. The launcher accepts no remote command, path,
SQL statement, credential, or semantic instruction. It selects only reviewed
packaged commands from the sealed wheel and validates one canonical JSON result
from each command.

The first writer action is a stopped preflight:

```bash
/Users/emillomliev/.local/share/uv/python/\
cpython-3.11.15-macos-aarch64-none/bin/python3.11 \
  -I -S -B -X pycache_prefix=/var/empty/muncho-canary \
  /absolute/clean/hermes-agent/scripts/canary/full_canary_owner_launcher.py \
  --release-sha <exact-40-character-release-sha> \
  --publish-writer-preflight \
  --external-iam-policy-sha256 <exact-reviewed-policy-sha256>
```

It collects the fixed writer configuration and live PostgreSQL facts, stages
the native plan and reviewed units, and publishes an append-only preflight
receipt. It does not install a unit, create an approval, or start Discord.

After that receipt succeeds, the stopped activation bridge runs the complete
native-to-final writer lifecycle:

```bash
/Users/emillomliev/.local/share/uv/python/\
cpython-3.11.15-macos-aarch64-none/bin/python3.11 \
  -I -S -B -X pycache_prefix=/var/empty/muncho-canary \
  /absolute/clean/hermes-agent/scripts/canary/full_canary_owner_launcher.py \
  --release-sha <exact-40-character-release-sha> \
  --activate-writer-stopped \
  --external-iam-policy-sha256 <exact-reviewed-policy-sha256>
```

For each of the native and final plans, the launcher:

1. pins the current human gcloud account and rechecks the trusted local
   interpreter, SDK, configuration, SSH identity, and authorization snapshots;
2. executes only the exact read-only IAM inventory used by the foundation and
   host collectors;
3. authors the existing `muncho-writer-owner-approval.v1` receipt, bound to
   the exact plan, hashed owner account, and pinned
   `skyvision_mac_ops_ed25519` public-key fingerprint;
4. records `cryptographic_owner_proof=false` honestly—the fingerprint binding
   is not presented as a passkey or signature verification;
5. projects a fresh external-IAM receipt with at least 720 seconds remaining;
6. sends only the bounded, secret-free `MWA1` authority frame to the packaged
   fixed-path bridge; and
7. validates every packaged receipt and the terminal stopped service state.

The bridge journals the final transition before replacing anything, archives
both native authority files append-only, and then atomically replaces the two
fixed staged files with the final generation. A retry may finish only that
same byte-identical transition. It has no service-start command. The activation
lifecycle itself performs the reviewed temporary writer/gateway observations,
always stops them again, never starts Discord, and succeeds only with
`muncho-writer-stopped-owner-activation.v1`.

## Exact stopped-to-live operator sequence

Use one clean checkout whose `HEAD` is the exact reviewed release SHA. Every
step is digest-bound and must succeed before the next. A failure is a gate:
preserve its receipt and do not skip ahead.

First bootstrap the fixed local gcloud runtime:

If an earlier read-only SDK invocation left Python bytecode and the bootstrap
fails specifically with `trusted_gcloud_sdk_bytecode_forbidden`, run the
separate bounded repair once before retrying bootstrap:

```bash
/Users/emillomliev/.local/share/uv/python/\
cpython-3.11.15-macos-aarch64-none/bin/python3.11 \
  -I -S -B -X pycache_prefix=/var/empty/muncho-canary \
  /absolute/clean/hermes-agent/scripts/canary/full_canary_owner_launcher.py \
  --release-sha <exact-40-character-release-sha> \
  --repair-trusted-sdk-bytecode
```

This action has no fallback cleanup path. Before deleting anything it proves
that the exact pinned SDK, with only owner-owned non-hardlinked `*.pyc` leaves
inside `__pycache__`, still equals its sealed publication intent. It removes
only those leaves and now-empty cache directories, revalidates the complete
publication, and durably records a self-hashed release-bound receipt. The exact
audited set is journaled and fsynced before the first descriptor-relative
unlink, so an interrupted retry can remove only the proven survivors and still
bind the terminal receipt to the original complete set. A completed retry only
revalidates and returns that same receipt.

```bash
/Users/emillomliev/.local/share/uv/python/\
cpython-3.11.15-macos-aarch64-none/bin/python3.11 \
  -I -S -B -X pycache_prefix=/var/empty/muncho-canary \
  /absolute/clean/hermes-agent/scripts/canary/full_canary_owner_launcher.py \
  --release-sha <exact-40-character-release-sha> \
  --bootstrap-trusted-runtime
```

Then publish the immutable stopped release:

```bash
/Users/emillomliev/.local/share/uv/python/\
cpython-3.11.15-macos-aarch64-none/bin/python3.11 \
  -I -S -B -X pycache_prefix=/var/empty/muncho-canary \
  /absolute/clean/hermes-agent/scripts/canary/full_canary_owner_launcher.py \
  --release-sha <exact-40-character-release-sha> \
  --publish-stopped-release
```

Run `--publish-writer-preflight` and `--activate-writer-stopped` exactly as
shown above. Next publish the secret-free, public-channel fixture:

```bash
/Users/emillomliev/.local/share/uv/python/\
cpython-3.11.15-macos-aarch64-none/bin/python3.11 \
  -I -S -B -X pycache_prefix=/var/empty/muncho-canary \
  /absolute/clean/hermes-agent/scripts/canary/full_canary_owner_launcher.py \
  --release-sha <exact-40-character-release-sha> \
  --publish-full-canary-fixture \
  --external-iam-policy-sha256 <exact-reviewed-policy-sha256>
```

The fixture describes only a genuinely complex objective and public route-back
policy. It does not mention a reasoning level, adaptive control, or a required
`todo` call. The live verifier still requires GPT-5.6-sol itself to request
`max` and proves the later real request used it.

Publish the fixed coordinator input, then apply the standalone Phase-B
foundation:

```bash
/Users/emillomliev/.local/share/uv/python/\
cpython-3.11.15-macos-aarch64-none/bin/python3.11 \
  -I -S -B -X pycache_prefix=/var/empty/muncho-canary \
  /absolute/clean/hermes-agent/scripts/canary/full_canary_owner_launcher.py \
  --release-sha <exact-40-character-release-sha> \
  --publish-coordinator-input
```

```bash
/Users/emillomliev/.local/share/uv/python/\
cpython-3.11.15-macos-aarch64-none/bin/python3.11 \
  -I -S -B -X pycache_prefix=/var/empty/muncho-canary \
  /absolute/clean/hermes-agent/scripts/canary/full_canary_owner_launcher.py \
  --release-sha <exact-40-character-release-sha> \
  --apply-phase-b-foundation
```

Phase B is the only bounded foundation-mutation protocol. Any temporary
database authority exists only inside that protocol and must be retired before
its terminal readiness receipt opens the live gate. The normal live path
accepts no database administrator credential, bootstrap credential, or
recovery worker.

Finally invoke the launcher with no action flag:

```bash
/Users/emillomliev/.local/share/uv/python/\
cpython-3.11.15-macos-aarch64-none/bin/python3.11 \
  -I -S -B -X pycache_prefix=/var/empty/muncho-canary \
  /absolute/clean/hermes-agent/scripts/canary/full_canary_owner_launcher.py \
  --release-sha <exact-40-character-release-sha>
```

The final invocation obtains the distinct canary Discord token only from the
masked interactive prompt or the documented `MDO1` stdin frame. The token is
never placed in argv, environment, config, logs, or Secret Manager. The
launcher installs it only through the privileged Discord-egress boundary.

The same remote coordinator process then builds the exact live plan and emits
one canonical final-approval request. There is no remote approval file and no
second remote coordinator. The owner decision is supplied through the one-shot
local file
`~/.hermes/approvals/muncho-full-canary-final-approval.json`; it must be an
exact owner-authored receipt bound to the emitted plan and current request. The
launcher reads a stable owner-only mode-`0600` file, consumes it once, and
sends the bounded `MFA1` frame. The approval window is 30–900 seconds,
bounded by fixture expiry, and reserves its final five seconds for delivery.
A stale, early, late, mismatched, or reused approval is rejected.
The 30–900 second window is deliberately session-bound. There is no offline
approval-renewal artifact: once the request window closes, retry the fixed
full-canary launch so the coordinator emits a fresh request bound to the new
live session. Re-signing or extending an old receipt would sever that binding
and is therefore unsupported.

Success requires the real complex API turn, Canonical Task Workspace
transitions, model-authored `max` escalation, writer readback, public Discord
delivery and receipt, DM pre-dispatch denial, terminal service stop, and
Discord-token retirement. `route_back.sent` is accepted only after the public
delivery receipt; blocked delivery remains `route_back.blocked`.

The launcher is idempotent only where a byte-identical append-only receipt
proves the prior operation. Conflicting state is preserved and blocks. Do not
manually delete plans, journals, credentials, or receipts: that would destroy
the evidence needed for safe reconciliation.

## Production-shaped capability plan publication

The capability overlay is a separate stopped-only gate after the clean-room
canary. Its owner launcher creates a chain of local owner-only mode-`0600`
canonical files and self-digested receipts. The packaged collectors, not the
caller, provide the fixed Linux identities and exact artifact SHA-256 values.
The only caller-authored plan values are two public canary bot user IDs. No
artifact contains task text, route choice, token, token digest, credential, or
caller-selected target filesystem path.

The connector and Canonical route-back bot user IDs must be distinct from each
other and from the fixed production Muncho bot user ID. These are public
Discord snowflakes only. The two canary credentials remain separate leases and
never enter the plan or its receipt.

Do not hand-assemble identity or artifact inputs. First collect the pre-plan
Bitrix foundation inputs from the packaged read-only collector. The collector
binds the complete terminal full-canary receipt, fixed reviewed service IDs,
the observed host identity, exact staged plan/release, and packaged asset
manifest. The owner launcher creates the canonical local file once at mode
`0600` and emits a self-digested authoring receipt:

```bash
capability_canary_owner_launcher.py \
  <exact-40-character-release-sha> author-bitrix-foundation-inputs \
  --full-canary-receipt-file /absolute/owner-only/full-canary-terminal.json \
  --output-file /absolute/owner-only/bitrix-foundation-inputs.json
```

Save canonical stdout as the mode-`0600` foundation-authoring receipt. Then
bootstrap the stopped Bitrix precursor using only that exact chain:

```bash
capability_canary_owner_launcher.py \
  <exact-40-character-release-sha> bootstrap-bitrix-foundation \
  --full-canary-receipt-file /absolute/owner-only/full-canary-terminal.json \
  --foundation-file /absolute/owner-only/bitrix-foundation-inputs.json \
  --foundation-authoring-receipt-file \
    /absolute/owner-only/bitrix-foundation-authoring.json
```

Save its canonical stdout as the mode-`0600` Bitrix bootstrap receipt. Only
then author the capability plan inputs:

```bash
capability_canary_owner_launcher.py \
  <exact-40-character-release-sha> author-plan-inputs \
  --full-canary-receipt-file /absolute/owner-only/full-canary-terminal.json \
  --foundation-authoring-receipt-file \
    /absolute/owner-only/bitrix-foundation-authoring.json \
  --bitrix-foundation-receipt-file \
    /absolute/owner-only/bitrix-foundation-bootstrap.json \
  --output-file /absolute/owner-only/capability-plan-inputs.json \
  --connector-bot-user-id <public-connector-bot-id> \
  --routeback-bot-user-id <public-routeback-bot-id>
```

The only caller-authored plan inputs are the two public canary bot snowflakes.
The packaged collector mechanically derives all reviewed UIDs/GIDs and
recollects every executable, wrapper, runtime-manifest, and Bitrix artifact
digest. It embeds the complete terminal/foundation/bootstrap chain and the
observed host identities. Save the emitted plan-authoring receipt as an
owner-only file; no plan or full-plan digest is copied back onto argv.

That fixed inventory includes the producer principals `2109/2212`,
`2110/2213`, `2111/2214`, and `2112/2215`, plus the empty persistent
receipt-writer group at GID `2216`. The collector rejects partial principals,
name/numeric collisions, drift, unrelated primary-GID users, or persistent
supplementary authority. The sole retryable partial state is an exact empty
pinned group whose user name and UID remain absent; this closes the unavoidable
crash window between `groupadd` and `useradd`. After plan publication,
bootstrap creates or resumes only those exact create-only slots with explicit
numeric IDs and publishes a root-owned mode-`0400` before/after receipt. All
cross-service access is supplied at service time by systemd
`SupplementaryGroups=`; bootstrap never runs `usermod`.

Run the sealed publication through the local owner launcher:

```bash
/Users/emillomliev/.local/share/uv/python/\
cpython-3.11.15-macos-aarch64-none/bin/python3.11 \
  -I -S -B -X pycache_prefix=/var/empty/muncho-canary \
  /absolute/clean/hermes-agent/scripts/canary/\
capability_canary_owner_launcher.py \
  <exact-40-character-release-sha> publish-plan \
  --full-canary-receipt-file /absolute/owner-only/full-canary-terminal.json \
  --plan-file /absolute/owner-only/capability-plan-inputs.json \
  --plan-authoring-receipt-file \
    /absolute/owner-only/capability-plan-authoring.json
```

The remote runtime validates the complete terminal and authoring context,
rebuilds the plan mechanically, re-observes every bound host identity, and
requires the computed digest to match the authoring receipt. It then atomically
publishes only the fixed root-owned mode-`0400` path
`/etc/muncho/capability-canary/runtime-plan.json`. Its self-digested receipt is
append-only below `/var/lib/muncho-capability-canary-control/plan-publications/`.
A retry succeeds only when both existing plan and receipt are byte-identical;
an absent half, changed authority, changed plan, or tampered receipt blocks.

Bootstrap the producer foundation with the same terminal receipt and the exact
published plan bindings, then save its canonical stdout receipt as a
mode-`0600` owner file:

```bash
capability_canary_owner_launcher.py \
  <exact-40-character-release-sha> bootstrap-producer-foundation \
  --full-canary-receipt-file /absolute/owner-only/full-canary-terminal.json \
  --plan-publication-receipt-file \
    /absolute/owner-only/capability-plan-publication.json \
  --plan-sha256 <exact-published-capability-plan-sha256> \
  --full-canary-plan-sha256 <terminal-full-canary-plan-sha256>
```

Create the exact live-fixture authority with that receipt and the pinned owner
sshsig key:

```bash
capability_canary_owner_launcher.py \
  <exact-40-character-release-sha> author-live-fixture \
  --full-canary-receipt-file /absolute/owner-only/full-canary-terminal.json \
  --producer-receipt-file /absolute/owner-only/producer-bootstrap.json \
  --plan-publication-receipt-file \
    /absolute/owner-only/capability-plan-publication.json \
  --output-file /absolute/owner-only/live-fixture-authority.json \
  --run-id <fixed-public-run-id> --valid-for-seconds 900
```

The action fixes the owner and Discord target to the approved public canary
channel, binds the producer foundation back to the terminal full-canary
receipt, signs the canonical authority with namespace
`muncho-production-capability-canary-owner-v1`, and creates the local file
without replacement. Pipe that file to `publish-live-fixture`; it contains no
secret or secret digest.

After fixture publication, provision all six required credential leases, run
the stopped preflight, install the exact fresh owner approval, and then invoke
the owner launcher's `run-live-observed` action directly. Do not run the
standalone `start`, `preflight-live`, or `run-live` actions first: the sealed
live driver owns startup and live preflight and always runs reverse-order stop
and secret retirement in its `finally` boundary. A pre-started service is
contract drift, not a shortcut.

`run-live-observed` keeps the live command attached while it waits for the
exact before and after markers, collects and signs the pinned read-only
production state, and stages both envelopes through fixed packaged runtime
actions. Its terminal receipt is valid only when the live evidence binds the
staged before observation and the published no-change diff. The internal
marker-wait and observation-staging actions are not exposed as standalone owner
CLI choices. Run the explicit `stop`, `retire-secrets`, and stopped-preflight
actions afterward as terminal verification/idempotent cleanup; a partial live
result is never promotion evidence.

## Production cutover owner state chain

After every production prerequisite in
`docs/muncho-production-cutover-artifacts.md` is satisfied, the executable
owner surface is one `prepare-cutover` followed by five one-state
`resume-cutover` calls. All paths are absolute, each output is create-only,
and each resume reads the preceding immutable output and writes a new filename:

```bash
python -m scripts.canary.production_cutover_owner_launcher \
  prepare-cutover \
  --revision <exact-40-character-release-sha> \
  --isolated-canary-goal-prerequisite \
    /absolute/owner-only/cutover/isolated-canary-prerequisite.json \
  --owner-private-key /absolute/owner-only/cutover-owner-ed25519 \
  --truth-mode start_new_truth_epoch \
  --output /absolute/owner-only/cutover/00-awaiting-bridge-bootstrap.json

python -m scripts.canary.production_cutover_owner_launcher \
  resume-cutover --revision <same-exact-release-sha> \
  --workspace /absolute/owner-only/cutover/00-awaiting-bridge-bootstrap.json \
  --output /absolute/owner-only/cutover/01-awaiting-bridge-passkey.json

# Review 01 and approve only bridge_request.legacy_approval_url.
python -m scripts.canary.production_cutover_owner_launcher \
  resume-cutover --revision <same-exact-release-sha> \
  --workspace /absolute/owner-only/cutover/01-awaiting-bridge-passkey.json \
  --output /absolute/owner-only/cutover/02-awaiting-cutover-passkey.json

# Review 02 and approve only advertised_approval_url with the exact v2 passkey.
python -m scripts.canary.production_cutover_owner_launcher \
  resume-cutover --revision <same-exact-release-sha> \
  --workspace /absolute/owner-only/cutover/02-awaiting-cutover-passkey.json \
  --output /absolute/owner-only/cutover/03-passkey-claim-recorded.json

python -m scripts.canary.production_cutover_owner_launcher \
  resume-cutover --revision <same-exact-release-sha> \
  --workspace /absolute/owner-only/cutover/03-passkey-claim-recorded.json \
  --output /absolute/owner-only/cutover/04-cutover-staged.json

python -m scripts.canary.production_cutover_owner_launcher \
  resume-cutover --revision <same-exact-release-sha> \
  --workspace /absolute/owner-only/cutover/04-cutover-staged.json \
  --output /absolute/owner-only/cutover/05-cutover-terminal.json
```

The two approvals are distinct and ordered: the legacy passkey authorizes only
the fixed temporary approval bridge; the v2 passkey authorizes the exact
release-bound FreezePlan and is consumed into `passkey_claim_recorded`. Never
reuse an output filename for a different state, input, retry, or release. An
expired approval starts a fresh `prepare-cutover` chain under fresh filenames;
old workspaces and journals remain evidence.

The fourth resume durably produces state `cutover_staged`; it binds the final
tail, stopped services, cron continuity, authored and staged cutover plan, and
their receipts. The fifth resume is the only route to the fixed idempotent
internal `converge-cutover` root action and the terminal receipt.

Recovery follows that boundary. Before `cutover_staged`, the fixed internal
`abort-freeze` may restore only the signed exact legacy/Caddy pre-state; once
its validated `freeze_aborted` terminal exists, that attempt is closed and a
fresh `prepare-cutover` and approval chain is required. From
`cutover_staged` but before a cutover `passkey_intent`, `abort-freeze` is valid
only through the maintenance-proven Caddy restore handoff; exact Caddy bytes
are restored only after that gateway abort is terminal. After a cutover
`passkey_intent` but before `activation_commit_intent`, the cutover transaction
must first publish its validated `rollback_terminal`; that path does not call
`abort-freeze`. Either pre-intent recovery may close the approved attempt as
`freeze_aborted` or `cutover_rolled_back_restored`. Preserve its workspace and
journals, then start a fresh `prepare-cutover` and approval chain under fresh
filenames; retrying the same fifth-stage workspace cannot resume forward
apply. Same-workspace fifth-stage retries are only for incomplete recovery or
an already forward-only transaction. After `activation_commit_intent`, such
retries from the preserved `04-cutover-staged.json`, each into a new absolute
recovery-attempt output, may converge only to
`private_v2_active` or persistent fixed 503 `maintenance_active`, never v1.
There is no public owner `recover` or `converge` subcommand; do not invoke the
sealed `converge-cutover` action directly. The production cutover artifact
document is the normative detailed recovery matrix and runbook.
