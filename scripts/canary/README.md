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
