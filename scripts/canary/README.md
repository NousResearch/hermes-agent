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
