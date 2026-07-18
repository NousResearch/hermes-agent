# Isolated canary 40 -> 80 GB storage growth

This is a one-time owner-approved gate for VM `muncho-canary-v2-01`, instance
ID `9153645328899914617`, boot disk ID `4195397669213846393`, device
`persistent-disk-0`, project `adventico-ai-platform`, zone `europe-west3-a`,
and owner account `lomliev@adventico.com`.  It does not change or reinterpret
the completed 20 -> 40 GB plan, constants, journals, or receipts.

## Fixed authority

The plan contains only these possible mutations:

1. one exact `gcloud compute disks resize ... --size=80GB`;
2. the exact VM stop; and
3. the exact VM start, only if a read-only post-resize observation still sees
   the 40 GB-sized ext4 root.

There is no `growpart`, `parted`, `resize2fs`, guest shell, generic command,
cleanup, snapshot, delete, service enablement, or service-start authority.
Every running observation proves all twelve units in
`runtime_units.CANARY_RUNTIME_UNITS` absent or disabled+inactive.  This is the
seven stopped-release units plus the isolated worker socket/service,
capability browser, Discord connector, and Mac ops edge.

## Pinned one-time boundary

The canonical passkey-v2 action rejects any variation from these current facts:

- stopped release SHA:
  `bc37d4252c46f6780e10552580fedb5147157bee`;
- host receipt raw/self:
  `ecb53958439984bb317578f8495358c04db01669df06dc7e0f3af8c7eb982f55` /
  `4b6a6716c27a52659f204fb8a796657aeb370426c80fe21c5b470bfa763f74c7`;
- stopped receipt raw/self:
  `180aee0ee954d114e26b509bf8af78dab5e8896da05d00f5e275200b94b4f2ed` /
  `47c79dbc36d2c13009af82572885ce8481c82079f7a2694ccf5ec209ee30541f`;
- external IAM:
  `236924140942a99e6162ae6492261ddd8b3a3f61013691a44c9b5e79bfcddb16`;
- historical 40 GB readiness raw/self:
  `e382d03f1179c028698ab4073d378bde32106558bcc510a35e5ee3ec1339a35a` /
  `5a8e63f0ca5df03c4bf40222861d4828515cb73aca7e5dc17e102aa176a62202`.

The historical readiness archive is read and validated without modification.
The new release SHA in the approval must differ from bc37.

## Approval and execution

`host_storage_growth.py` is only a pure canonical plan/read-only validation
facade, and `host_storage_growth_preflight.py` is only an injected read-only
collector; both direct entry points fail closed.  The local release-bound
`full_canary_owner_launcher.py` may collect read-only source evidence and talk
to the privileged boundary, but it has no approval store, mutation journal, or
storage mutation runner.  The dedicated private `muncho-owner-gate-01`
passkey-v2 executor owns the narrow Cloud identity, authoritative SQLite
ledger, atomic grant claim, and every resize/stop/start attempt.

The storage executor is the fixed no-shell UID/GID `29103:29103`.  Its
mode-`0700` mutation ledger root is
`/var/lib/muncho-owner-gate/executor`.  The web identity (`29101`) and
passkey-authority identity (`29102`) cannot read or write that ledger or the
executor's Cloud credential.

Storage growth is a dangerous `runtime_config_mutation`.  Production approval
must be an exact-action, single-use grant from Emil's enrolled WebAuthn passkey
at `https://auth.lomliev.com`.  The UID-501-readable local SSH key is not a
privileged approval boundary and must not be used for production authoring.
The old same-UID Cloud helper/guard and its TOTP fallback are not an approval
boundary either: gateway, passkey service, helper code, TOTP seeds, and grant
state all share UID 999.  A successful interaction with that current approval
page proves user presence only for the legacy service; it does not authorize
this mutation.

This gate stays fail-closed until `muncho-owner-gate-01` attests distinct
no-shell passkey and cutover-executor identities; root-owned code, venv, and
units; passkey-only approval; strict token/path handling; atomic challenge and
grant consumption; an append-only durable executor ledger; a narrow attached
service account inaccessible to the gateway; and an owner UI that shows the
complete canonical payload and full hash.  The commands below are the target
interface, not authorization to use the legacy guard.

Bringing this boundary online requires two separate owner-presence gates.  The
first is a fresh GCP owner reauthentication/passkey approval for the inert v2
foundation only (VM, identities, root-owned package, ledger, proxy, and
readiness attestation).  The legacy guard cannot authorize its own
replacement.  After the v2 readiness receipt passes, the second is a new
passkey approval for this exact storage action.  Bootstrap approval never
authorizes resize, stop, or start.

The release-bound flow is:

```bash
R='<exact 40-character fork release SHA>'

# Collect the exact source and create a passkey exact-action request.
python scripts/canary/full_canary_owner_launcher.py \
  --release-sha "$R" \
  --author-storage-growth

# After passkey approval, consume its request exactly once and apply.
python scripts/canary/full_canary_owner_launcher.py \
  --release-sha "$R" \
  --apply-storage-growth \
  --storage-growth-transaction-id '<transaction_id>' \
  --storage-growth-passkey-request-id '<request_id>'
```

The authoring output supplies only canonical identifiers, hashes, the request
ID, and the passkey approval URL.  It never prints a credential or grant.  The
action envelope uses strict compact UTF-8 JSON with duplicate keys, non-finite
values, floats, traversal-shaped IDs, and non-canonical bytes rejected.  Its
full hash binds scope, case, target, summary, risk, rollback, plan SHA and argv,
transaction ID, exact release, 40/80 resource identities, current authoritative
journal stage, receipt evidence, the versioned live-equivalence projection,
expiry, and expected verification.  The approval page shows that complete
material and full hash.  A twelve-character prefix or a legacy payload-only
hash is insufficient.

The passkey request/grant is live for 30–300 seconds.  Consuming the initial
grant within that window durably opens only the plan's explicit
`transaction_authorization_max_age_seconds=3600` execution window.  It does
not create a general one-hour approval.  Every command remains fixed by the
canonical plan, and fresh IAM/host evidence is rechecked immediately before
each mutation.

The remote executor atomically consumes the grant and permanently marks the
request used before mutation.  It durably commits the exact approval envelope,
authorization, and append-only intent before resize.  Resize/stop/start and
completion records are no-replace, self-hashed, and fsync/readback verified in
the privileged ledger.  Terminal completion also binds the ordered
resume-authorization chain.  A terminal replay performs a read-only check
against that authoritative remote ledger; a local same-UID JSON journal is
never sufficient.  There is no local gcloud mutation fallback and no
user-managed service-account key.

### Interrupted or expired transaction

First rerun `--apply-storage-growth` with the same release, transaction, and
request ID.  The privileged executor reconciles already-completed commands
from its ledger and fresh live facts, and appends missing mechanical stages
without asking for another approval.  It never repeats a mutation whose effect
is already proven.  Local files cannot drive recovery.

If an actual remaining mutation is needed after the 3600-second window, apply
fails closed and reports the exact current stage.  Create and approve a new
domain-separated `resume_incomplete` passkey request, then apply it once:

```bash
python scripts/canary/full_canary_owner_launcher.py \
  --release-sha "$R" \
  --author-storage-growth-resume \
  --storage-growth-transaction-id '<transaction_id>'

python scripts/canary/full_canary_owner_launcher.py \
  --release-sha "$R" \
  --apply-storage-growth \
  --storage-growth-transaction-id '<transaction_id>' \
  --storage-growth-passkey-request-id '<resume_request_id>'
```

Each resume request is built from the remote executor's authoritative intent,
authorization/stage hashes, exact release, fresh evidence, explicit
live-equivalence policy, remaining action, sequence, previous resume record,
and short TTL.  Stage advancement, expiry, or an IAM/resource mismatch needs a
new request.  Completed/inferred stages and terminal receipt publication are
mechanical and do not consume a new grant.  Crash recovery accepts only exact
atomic-ledger state; missing, ambiguous, tampered, cross-release, or
cross-transaction evidence blocks.

## Terminal meaning

Final readiness requires an 80 GB disk, ext4 `/dev/sda1` at `/`, at least
84,000,000,000 filesystem bytes, at least 32 GiB available, and the complete
stopped runtime inventory.

- `online_expansion_observed` proves the boot ID did not change.  The bc37
  host/stopped identity remains usable.
- `reboot_expansion_completed` proves a new boot ID and sets
  `host_identity_rotation_required`,
  `fresh_post_reboot_release_sha_required`, and
  `same_revision_republication_forbidden` to true.

After a reboot, rotate the host identity receipt and publish a new stopped
release at this gate's merged SHA before any database bootstrap.  Never
overwrite or republish the old revision.  Storage readiness itself always has
`opens_runtime_gate=false`.
