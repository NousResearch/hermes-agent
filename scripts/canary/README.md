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
- runtime service account `muncho-canary-v2-runtime`, with only Logging Writer
  and Monitoring Metric Writer and no user-managed keys;
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
Shielded VM with the pinned Debian 12 image, 20 GB balanced disk, exact
logging/monitoring scopes, OS Login, disabled legacy metadata endpoints, and
only the `iap-ssh` tag. It has an ephemeral premium-tier public address for IAP
reachability and no deletion protection because it is a temporary canary.

The foundation and network plans contain no VM-create command. The host
preflight nests a fresh effective-firewall attestation and requires all network
steps satisfied; consequently the VM cannot be created before the firewall
boundary is live. Every apply receipt still requires a post-apply read-only
attestation before the next phase or runtime enablement.

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

## Phase 4 — packaged writer-only activation

Runtime mutation is performed by the sealed wheel, not by importing this
source-only `scripts.canary` package on the VM. The source
`scripts.canary.writer_activation` entrypoint is only a delegate to the same
packaged planner and contains no alternate preview or deployment logic.

Before any unit is installed or started, root runs the packaged trusted
collector against the sealed release and live PostgreSQL authority. The
collector reads the already provisioned credential through its pinned file
descriptor, but its append-only receipt records only file provenance and
digests—never credential content or a credential digest:

```bash
<sealed-python> -B -I -m gateway.canonical_writer_config_collector collect \
  --revision <exact-40-character-release-sha> \
  --release-artifact-sha256 <sealed-release-artifact-sha256> \
  --release-manifest-file-sha256 <sealed-manifest-file-sha256> \
  --tls-server-name <verified-cloud-sql-certificate-san> \
  --owner-discord-user-id <owner-id>
```

The result supplies `receipt_sha256`. While that receipt's live HBA evidence
is still fresh, the packaged planner derives the SQL private IP and TLS name
from the receipt itself; they are deliberately not accepted again as CLI
facts. It renders and exclusively stages the two reviewed units plus the
discovery plan at their fixed paths. Output contains digests only and grants
no approval:

```bash
<sealed-python> -B -I -m gateway.canonical_writer_planner build-native-plan \
  --revision <exact-40-character-release-sha> \
  --external-iam-policy-sha256 <exact-reviewed-policy-sha256> \
  --config-collector-receipt-sha256 <receipt-sha256>
```

Root then installs that strict staged discovery plan at the only accepted
path, still without starting a service:

```bash
<sealed-python> -B -I -m gateway.canonical_writer_activation install-native-plan \
  --plan /etc/muncho/writer-activation/staged/native-observation-plan.json
```

The bootstrap owner confirmation is root-provisioned and out of band. Its
receipt truthfully records
`authority_kind=trusted_root_bootstrap_out_of_band_owner` and
`cryptographic_owner_proof=false`; this is not a claim that a passkey or
signature was verified. Renewals are installed append-only at
`approvals/<scope>/<plan-sha>/<receipt-sha>.json`:

```bash
<sealed-python> -B -I -m gateway.canonical_writer_activation install-approval \
  --staged-receipt /etc/muncho/writer-activation/staged/owner-approval.json
<sealed-python> -B -I -m gateway.canonical_writer_activation install-external-iam \
  --staged-receipt /etc/muncho/writer-activation/staged/external-iam-receipt.json \
  --external-iam-policy-sha256 <exact-reviewed-policy-sha256> \
  --plan /etc/muncho/writer-activation/native-observation-plan.json \
  --approved-plan-sha256 <exact-native-plan-sha256> \
  --owner-approval-receipt \
    /etc/muncho/writer-activation/approvals/native_observation/<plan-sha>/<approval-sha>.json
```

The IAM helper archives both old and new generations before atomically
replacing only the fixed live file under `/run`. It accepts no unapproved
refresh: the staged IAM receipt's `source_approval_sha256` must equal the exact
owner receipt for the pinned native or final plan, and the lifecycle rechecks
that binding again before mutation and immediately before service start. The
native observation plan
contains no guessed mapping list or placeholder manifest. It binds the exact
release, configs, canary users/groups/homes, SQL `/32`, TLS name and CA,
retired-helper and Discord absence, and root-owned discovery policy. Before any
identity mutation, the packaged lifecycle re-hashes every staged input and
release file, verifies CA and credential provenance, performs TLS/PostgreSQL
startup privilege attestation, and proves the services are stopped or absent.

```bash
<sealed-python> -B -I -m gateway.canonical_writer_activation observe-native \
  --plan /etc/muncho/writer-activation/native-observation-plan.json \
  --approved-plan-sha256 <exact-native-plan-sha256> \
  --owner-approval-receipt \
    /etc/muncho/writer-activation/approvals/native_observation/<plan-sha>/<approval-sha>.json \
  --external-iam-receipt \
    /run/muncho-canonical-preflight/external-iam-receipt.json
```

The observer starts writer then gateway without enabling either and always
stops gateway then writer. Its durable stopped receipt binds the append-only
live stage, host-preparation receipt and exact IAM receipt. Later consumers
accept an expired receipt after reboot only on the same host and only after
re-hashing current release/config/library inputs and rechecking current mapping
policy; a cross-host replay remains invalid.

Only that stopped receipt may build the single deployable v3 activation plan.
The packaged planner reloads the fixed installed native plan and its durable
append-only stopped receipt, revalidates the current host/release/config/unit
bindings, and exclusively stages the final plan. It accepts the expected
receipt digest only; it does not accept or infer owner approval:

```bash
<sealed-python> -B -I -m gateway.canonical_writer_planner build-final-plan \
  --native-observation-receipt-sha256 \
    <exact-durable-stopped-native-receipt-sha256>
```

The final plan is then installed, without starting a service:

```bash
<sealed-python> -B -I -m gateway.canonical_writer_activation install-plan \
  --plan /etc/muncho/writer-activation/staged/activation-plan.json
```

After the owner separately approves the exact final plan digest, a new
activation-scoped approval receipt and fresh IAM receipt are staged. The IAM
receipt must set `source_approval_sha256` to that exact approval receipt SHA;
the earlier native-scoped IAM receipt cannot authorize final activation:

```bash
<sealed-python> -B -I -m gateway.canonical_writer_activation install-approval \
  --staged-receipt /etc/muncho/writer-activation/staged/owner-approval.json
<sealed-python> -B -I -m gateway.canonical_writer_activation install-external-iam \
  --staged-receipt /etc/muncho/writer-activation/staged/external-iam-receipt.json \
  --external-iam-policy-sha256 <exact-reviewed-policy-sha256> \
  --plan /etc/muncho/writer-activation/activation-plan.json \
  --approved-plan-sha256 <exact-activation-plan-sha256> \
  --owner-approval-receipt \
    /etc/muncho/writer-activation/approvals/activation/<plan-sha>/<approval-sha>.json
<sealed-python> -B -I -m gateway.canonical_writer_activation validate-plan \
  --plan /etc/muncho/writer-activation/activation-plan.json \
  --approved-plan-sha256 <exact-activation-plan-sha256> \
  --owner-approval-receipt \
    /etc/muncho/writer-activation/approvals/activation/<plan-sha>/<approval-sha>.json
<sealed-python> -B -I -m gateway.canonical_writer_activation apply \
  --plan /etc/muncho/writer-activation/activation-plan.json \
  --approved-plan-sha256 <exact-activation-plan-sha256> \
  --owner-approval-receipt \
    /etc/muncho/writer-activation/approvals/activation/<plan-sha>/<approval-sha>.json
```

`validate-plan` runs only after the exact activation-scoped owner approval and
its freshly bound IAM receipt are installed. It runs the same bounded preflight
used under the activation lock.
Each report is sealed append-only with distinct report-content and file
digests. A failed preflight is blocked and retryable, not a forensic mutation
quarantine. IAM must retain at least 720 seconds before mutation and is
re-read/re-archived immediately before services start.

`apply` serializes the whole lifecycle at the root-controlled
`/run/muncho-writer-activation.lock`. It accepts only absent or byte-identical
artifacts, runs a temporary non-enableable exporter, verifies the canonical
export and `999:991 0640` identity, then removes the exporter and proves it
absent. It starts writer and gateway, runs the packaged root collector, archives
the exact root receipt, and always stops gateway followed by writer.

Success and failure evidence is append-only and plan-addressed. A mutation
failure creates a unique receipt plus fixed quarantine marker; preflight-only
failures do not. No command enables a unit, creates a timer, starts Discord,
invokes a shell, accepts a secret value, or infers a semantic decision.

The installed systemd units execute the packaged bootstraps directly with
the sealed interpreter and `-B -I`. Writer readiness is emitted from inside the
writer process only after database startup attestation, socket creation, and
runtime/module identity attestation succeed. Gateway readiness is emitted
only after its in-process writer PING/readiness proof succeeds. A process that
merely exists, or an external probe that only sees an open socket, cannot mark
either unit ready.

## Packaged full-canary coordinator and owner edge

The full canary uses the packaged coordinator from the exact sealed release.
There is no Cloud-side `python -c`, `tee`, heredoc, mutable checkout, or
free-form remote command. Before the owner launcher is run, the separate root
bootstrap invokes the coordinator's fixed, no-argument command:

```text
/usr/bin/sudo --non-interactive -- \
  /opt/muncho-canary-releases/<release-sha>/venv/bin/python \
  -B -I -m gateway.canonical_full_canary_coordinator \
  publish-coordinator-input
```

`publish-coordinator-input` reads only the installed fixed-path writer plan and
success receipt, staged writer/gateway/Discord-edge configs, fixture and host
receipts, and sealed SQL/NSS facts. It accepts no stdin. It atomically creates
root-owned `0400` files at
`/etc/muncho/full-canary/coordinator-input.json` and
`/etc/muncho/full-canary/coordinator-input-publication.json`; an identical retry
is accepted and a conflicting replacement is rejected. Its terminal receipt
schema is `muncho-full-canary-coordinator-input-publication.v1`.

The local owner launcher is never started with ambient `python`, `-m`,
`PYTHONPATH`, or site packages. From a clean checkout whose `HEAD` equals the
exact release SHA, first publish the release-bound trusted runtime receipt with
the fixed standalone interpreter and the launcher's absolute path:

```bash
/Users/emillomliev/.local/share/uv/python/\
cpython-3.11.15-macos-aarch64-none/bin/python3.11 \
  -I -S -B -X pycache_prefix=/var/empty/muncho-canary \
  /absolute/clean/hermes-agent/scripts/canary/full_canary_owner_launcher.py \
  --bootstrap-trusted-runtime \
  --release-sha <exact-40-character-release-sha>
```

That secret-free, explicit bootstrap downloads only Google's versioned
`google-cloud-cli-569.0.0-darwin-arm.tar.gz` archive, requiring exactly
`60,511,521` bytes and SHA-256
`2d4ab8eb0a9362a69feabade6df4163763cd989cb840dc3f7ced5ac24dde6e67`.
Proxy use, redirects, unsafe tar paths, hard links, special files, escaping
symlinks, `.pyc`, `.pyo`, and `__pycache__` are rejected. Files are extracted
with exclusive creation into owner-only staging, every implicit parent is
created privately, and all files and directories are synced. Before SDK
publication, a version-and-archive-hash-specific canonical intent durably binds
the release, launcher SHA-256, reviewed archive, destination, and deterministic
complete extracted-tree fingerprint. Darwin `renamex_np(RENAME_EXCL)` publishes
that intent, the SDK at `~/.hermes/trusted/google-cloud-sdk-569.0.0`, and the
release receipt atomically without replacing a file, directory, or symlink.
Reruns recover exact crashes after the intent, SDK, or receipt publication;
destination state without the exact intent, or a mismatching tree, fails closed.
The release-specific receipt binds the intent, current tracked launcher SHA-256,
exact Python version `3.11.15`, the full SDK/Python trees, and the fixed macOS
dependency set. An identical retry is accepted; conflicting state is preserved
and rejected.

After the bootstrap receipt succeeds, run the canary with the same exact
interpreter, isolation flags, absolute tracked script, and release SHA, but
without the bootstrap flag:

```bash
/Users/emillomliev/.local/share/uv/python/\
cpython-3.11.15-macos-aarch64-none/bin/python3.11 \
  -I -S -B -X pycache_prefix=/var/empty/muncho-canary \
  /absolute/clean/hermes-agent/scripts/canary/full_canary_owner_launcher.py \
  --release-sha <exact-40-character-release-sha>
```

Before git, auth, network access, or secret input, the launcher proves its
interpreter path and all isolation flags, re-hashes the fixed Python/SDK trees
and release receipt, and then binds its own tracked bytes to the exact commit.
The pre-existing same-UID owner state and fixed uv Python are explicit initial
trust anchors for this gate. These checks detect drift and other-admin/path
injection; they do not claim independent authenticity against an already
compromised owner UID or owner process.

It invokes gcloud directly as fixed Python plus the attested `lib/gcloud.py`;
the mutable shell wrapper and gcloud virtualenv are never executed. The active
gcloud configuration is parsed without gcloud and may contain exactly one
human account, project `adventico-ai-platform`, and zone `europe-west3-a`—no
impersonation, credential override, proxy, endpoint, custom CA, or logging
property. The closed subprocess environment disables HTTP/file logging, usage
reporting, component update checks, prompts, ambient proxy/SSH-agent/askpass
state, and is re-attested after every auth/IAP process and at launcher exit.

The launcher pins the private key, public key, and
`google_compute_known_hosts` inside the exact private owner SSH directory. A
read-only preflight proves the exact OS Login identity
`lomliev_adventico_com`, instance ID, `enable-oslogin=TRUE`, and that the pinned
public key is already provisioned; complete project metadata, instance
metadata, and OS Login profile snapshots must be unchanged after dry-run and
after the terminal SSH exit. `gcloud compute ssh --plain` prevents gcloud from
adding keys to OS Login or project/instance metadata. The only identity path is
the explicit pinned SSH `-i`; `IdentitiesOnly=yes`, `IdentityAgent=none`,
`CertificateFile=none`, public-key-only authentication, exact user known-hosts,
disabled global/DNS/update host trust, ambient config, local commands,
connection sharing, forwarding, and canonicalization close all other paths.
Before every connection, a bounded real `--dry-run` is parsed exactly and must
produce `/usr/bin/ssh` plus an IAP ProxyCommand containing only the fixed Python
isolation flags and attested `lib/gcloud.py`. The remote command uses exact
`/usr/bin/sudo` and the sealed interpreter. The launcher never accepts a secret
through argv, environment, a shell expansion, a log, or Secret Manager.

On an interactive terminal, the distinct canary Discord token is requested by
a masked prompt. A non-interactive trusted secret source must provide exactly
`MDO1 + u32be(length) + opaque token + EOF` on file descriptor 0; do not build
that frame with a shell `echo` or command-line literal. The temporary Cloud SQL
admin password is generated only in process memory, sent as the `MCA2` stdin
frame, wiped, and deleted only after a validated terminal receipt makes deletion
safe. An explicit create rejection never authorizes deletion of a concurrently
created same-name account.

The launcher emits the exact final-approval request as a canonical JSON line.
The coordinator has already produced that request and the launcher has
validated its bindings and stable owner identity before exposing it. The
launcher then opens and validates the independent read-only remote install gate
while the owner may decide; it does not read the local approval or disclose an
`MFA1` byte until that gate exactly matches the published request. Transport
setup therefore remains inside the bounded coordinator-authored window, and
the cutoff is checked again after setup. The request binds every upstream
expiry and defines
`owner_input_cutoff_unix = approval_deadline_unix - 30`; an owner decision or
local delivery after that cutoff is rejected before any `MFA1` byte is sent.
The whole request window is capped at exactly 240 seconds.
Only after the request is visible should the matching canonical approval be
atomically installed at
`~/.hermes/approvals/muncho-full-canary-final-approval.json`, owned by the local
owner, regular/unlinked, and mode `0600`. The launcher verifies a stable file
identity, consumes it once, sends it as `MFA1`, and accepts success only after
the coordinator emits and closes on the exact terminal receipt. An approval
that misses the request's cutoff or plan/owner/digest bindings is rejected.

If the wait times out, a signal arrives, or another local failure occurs before
the first `MFA1` byte is disclosed, the launcher closes that dedicated stdin
without sending data. The coordinator must return the exact append-only
`muncho-full-canary-final-approval-cancel-receipt.v2`, prove zero frame bytes,
no new owner-approval installation, and no owner-approval artifact mutation,
and terminate with status 2. The v2 receipt is bound to the request, staged
plan, and prior owner-approval snapshots captured before the gate. A clean
cancel requires either the matching request plus matching staged plan, or both
artifacts retired; mixed, superseded, same-bytes/replaced-inode, or owner-path
drift is reported as `cancelled_no_secret_state_conflict`. A partial `MFA1`
disclosure is instead a hard ambiguous failure and is never reported as a clean
cancellation. The broader coordinator terminal cleanup is still awaited for a
bounded interval through the request deadline plus a 300-second cleanup grace,
capped at 540 seconds total, so cleanup evidence is not abandoned merely
because the shorter owner-input window closed.

Every invocation begins with the read-only `preflight-recovery` command:

- An active run lease or recovery-worker lease yields a stage-one takeover gate.
  The launcher returns only the canonical no-secret `MRA1` acknowledgement.
  The coordinator then terminates the exact predecessor through pidfd, acquires
  the process lock, and replaces the full journal snapshot with durable worker
  transition 1 (`claimed_awaiting_admin`). It durably CASes transition 2
  (`admin_authority_may_be_in_use`) before emitting any secret gate.
- Only the stage-one CAS winner receives the stage-two secret gate. The launcher
  binds a fresh in-memory password to that gate and its unpredictable nonce in
  `MRC2 + gate_sha256 + nonce_sha256 + username + password + EOF`. Legacy
  `MCA2` is rejected after its four-byte magic, before credential bytes are
  read. A contending loser returns an exact zero-secret claim-lost receipt and
  never receives the stage-two gate.
- Cleanup persists
  `muncho-full-canary-recovery-worker-completion.v1` with the full original run
  lease, zeroized frame, closed admin session, and cleanup receipts. Completion
  explicitly sets worker-exit proof and safe-delete to false. The separate
  no-secret `finalize-recovery` command proves exact worker exit, reacquires the
  process lock, and CASes that exact completion to
  `muncho-full-canary-recovery-receipt.v2`; only v2 may set worker-exit proof and
  safe-delete true.
- A strictly valid legacy `muncho-full-canary-recovery-receipt.v1` is detected
  but is not consumed, upgraded, or treated as current terminal success. Every
  operational entry point returns the explicit fail-closed
  `legacy_recovery_receipt_reconciliation_required` blocker without opening a
  credential path.
- An exact persisted recovery receipt v2 is terminal truth. The launcher does
  not repeat recovery and only finishes the receipt-authorized Cloud SQL
  deletion.
- The exact no-journal/fully-stopped receipt permits an idempotent absence proof
  and deletion of only the approval-derived temporary username before a fresh
  run.
- Token residue with no process lease uses the separate causal `DRA1`
  owner-acknowledged token-retirement path. This includes a durable
  install-intent journal whose token device/inode pair is exactly null, and a
  crash after `retirement_prepared`; the owner ACK mirrors the exact causal
  digest and the same invocation continues through terminal retirement when
  the bounded proof permits it.

Success requires stopped and disabled services, an exact recovery v2 receipt,
a closed admin session, disabled/removed bootstrap authority, removed Discord
token and install receipt, proven recovery-worker exit, and a proven-absent
temporary admin. A
`cleanup_blocked` result is intentionally retryable: preserve the root journal
and receipts and rerun this same approved launcher. Do not manually remove the
journal, token files, receipts, or SQL user, because doing so destroys the
causal evidence needed for safe reconciliation.
