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
source-only `scripts.canary` package on the VM. Root first installs one strict
staged discovery plan at the only accepted path:

```bash
<sealed-python> -I -m gateway.canonical_writer_activation install-native-plan \
  --plan /etc/muncho/writer-activation/staged/native-observation-plan.json
```

The bootstrap owner confirmation is root-provisioned and out of band. Its
receipt truthfully records
`authority_kind=trusted_root_bootstrap_out_of_band_owner` and
`cryptographic_owner_proof=false`; this is not a claim that a passkey or
signature was verified. Renewals are installed append-only at
`approvals/<scope>/<plan-sha>/<receipt-sha>.json`:

```bash
<sealed-python> -I -m gateway.canonical_writer_activation install-approval \
  --staged-receipt /etc/muncho/writer-activation/staged/owner-approval.json
<sealed-python> -I -m gateway.canonical_writer_activation install-external-iam \
  --staged-receipt /etc/muncho/writer-activation/staged/external-iam-receipt.json \
  --external-iam-policy-sha256 <exact-reviewed-policy-sha256>
```

The IAM helper archives both old and new generations before atomically
replacing only the fixed live file under `/run`. The native observation plan
contains no guessed mapping list or placeholder manifest. It binds the exact
release, configs, canary users/groups/homes, SQL `/32`, TLS name and CA,
retired-helper and Discord absence, and root-owned discovery policy. Before any
identity mutation, the packaged lifecycle re-hashes every staged input and
release file, verifies CA and credential provenance, performs TLS/PostgreSQL
startup privilege attestation, and proves the services are stopped or absent.

```bash
<sealed-python> -I -m gateway.canonical_writer_activation observe-native \
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
The final plan is installed, comprehensively preflighted, and applied as:

```bash
<sealed-python> -I -m gateway.canonical_writer_activation install-plan \
  --plan /etc/muncho/writer-activation/staged/activation-plan.json
<sealed-python> -I -m gateway.canonical_writer_activation validate-plan \
  --plan /etc/muncho/writer-activation/activation-plan.json
<sealed-python> -I -m gateway.canonical_writer_activation apply \
  --plan /etc/muncho/writer-activation/activation-plan.json \
  --approved-plan-sha256 <exact-activation-plan-sha256> \
  --owner-approval-receipt \
    /etc/muncho/writer-activation/approvals/activation/<plan-sha>/<approval-sha>.json
```

`validate-plan` runs the same bounded preflight used under the activation lock.
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
