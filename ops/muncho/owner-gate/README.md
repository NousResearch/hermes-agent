# Muncho owner gate foundation

This directory is the inert, offline-deployable foundation for the dedicated
private `muncho-owner-gate-01` VM.  It does not authorize a Cloud mutation and
does not switch `auth.lomliev.com`.

The VM uses three fixed no-shell identities.  `muncho-passkey-web` can only
serve the public WebAuthn UI and reach the authority Unix socket.  It cannot
read the SQLite authority database, consume a grant, or reach the metadata
server.  `muncho-passkey-authority` owns the SQLite database, runs without a
network namespace, verifies the exact WebAuthn assertion, and can reach only
the executor Unix socket.  `muncho-storage-executor` cannot read the authority
database; it owns only `/var/lib/muncho-owner-gate/executor`, whose canonical
SQLite execution ledger is the append-only mutation journal, and is the sole
non-root UID allowed to reach instance metadata and the private Google API VIP.

The release is transferred over IAP as a complete offline artifact, installed
root-owned beneath `/opt/muncho-owner-gate/releases/<revision>`, and sealed
read-only.  No `apt`, `pip`, public IP, service-account key, generic shell, or
local `gcloud` fallback is part of runtime execution.  The dedicated subnet is
in `ai-platform-vpc`, uses `10.80.3.0/28`, and enables Private Google Access.
The plan must freshly prove that CIDR does not overlap any subnet, route,
peering, or connector range before creation.

`auth.lomliev.com` and Caddy remain on `ai-platform-runtime-01`.  Only after
offline split-UID, WebAuthn, concurrency, metadata-firewall, and no-op executor
smoke may Caddy atomically reload the private upstream template.  The old v1
unit is masked before cutover.  Once the public-key-only credential migration
has committed, v1 can never be re-enabled; rollback becomes a fail-closed
maintenance response while the v2 authority database and mutation journal are
preserved.

The Caddy switch is owned by
`scripts.canary.owner_gate_caddy_cutover`, not by an operator-supplied reload
command.  Its caller selects one of exactly four fixed phases; it cannot supply
a path, hostname, upstream, executable, or Caddy bytes.  `prepare-bridge` reads
one canonical, self-digested v2 request binding from standard input, snapshots
the exact live Caddyfile, and creates or adopts exactly one legacy-v1 passkey
request without changing ingress.  After that passkey is approved,
`activate-bridge` temporarily stops the legacy verifier, atomically consumes
the grant, installs only the exact v2 request UI/API paths, restarts and locally
health-checks the legacy verifier, and proves the default local-v1 route is
unchanged.  A pre-migration failure or interrupted retry restores the exact
preimage and the legacy verifier; the bridge authorization cannot authorize
the database or full route cutover.

The input-free `prepare` phase reuses the separately consumed v2
production-cutover passkey claim from the root journal, binds it to the exact
FreezePlan, CutoverPlan, bridge, and release, then stores an fsynced no-clobber
preimage plus validated private-v2 and maintenance candidates.  It does not
edit live ingress.  `commit` additionally requires the durable legacy
`activation_commit_intent` and terminal lineage, atomically replaces only the
verified local upstream token with `10.80.3.2:8080`, validates and reloads
Caddy, and verifies public TLS readiness.  Before the irreversible intent,
recovery restores the exact preimage.  After it, any validation, reload, or
public-verification failure can only converge to the fixed 503 maintenance
route; v1 bytes are never restored.  Both successful and maintenance terminals
bind the same passkey claim journal entry and consumption identity, preventing
database and ingress authority from being replayed independently.

The mutation IAM binding is a second gate.  Bootstrap creates no usable Cloud
mutation authority.  The deferred custom-role binding is resource-conditioned
to the exact canary disk and instance. Completion is observed by polling those
same conditioned resources; no zonal-operation permission is granted. It is
allowed only after all v2 security smoke receipts are valid.

The final storage-executor activation is a separate root-only command and is
not called by bootstrap.  It accepts no paths, digests, or resource names.  It
revalidates the release-signed foundation/apply lineage, every installed
package payload, the exact owner-gate VM topology receipt, the split-UID/offline
security smoke, and the post-binding IAM/API/numeric-target re-preflight from
the fixed root-owned evidence directory.  It also requires a new canonical,
release-key-signed owner reauthentication receipt issued after that
re-preflight and still fresh at activation; its release and project number
must match the immutable lineage.  The executor-readable
`/etc/muncho-owner-gate/storage-executor-enabled` file is itself the complete
canonical authorization record: its self-hash covers the signed release
manifest, source tree, package inventory, terminal foundation-apply receipt,
both owner reauthentications, ancestry, numeric VM/targets, smoke, post-IAM
receipt, and every fixed evidence-file digest.  It is published as root, group
`muncho-storage-executor`, mode `0440`, using an fsynced no-replace protocol;
only after that complete record exists is a root-only append-only audit mirror
published under `activation-receipts`.  A seal-only interrupted recovery never
receives a freshness waiver.  A fully published exact replay is idempotent;
any stale, missing, symlinked, or divergent artifact fails closed.

Bootstrap and runtime IAP are separate surfaces.  The owner-side bootstrap
transport may invoke only the package-pinned stage-zero entrypoint; the normal
IAP intake exists only after installation and accepts fixed JSON on stdin for
the authority UID.  Before any venv code runs, stage zero verifies the stable
fork-pinned release signer with Debian OpenSSL, the canonical manifest, and
every package byte.  The runtime lock and signed package bind one exact
bootstrap-pip wheel separately from the 22 target wheels.  Stage zero creates
the venv with `--without-pip`, executes only that carried wheel under
`-I -S -B` and an exact standard-library path, and installs itself plus the
target closure with no network, dependencies, source builds, or bytecode
compilation.  The mutable host `/usr/share/python-wheels` directory is never an
authority or execution input.  It also proves `venv --without-pip` and required
systemd helpers locally; it never downloads a package.  Bootstrap never edits the
machine-wide `/etc/hosts`.  Instead it installs one exact root-owned, mode-0444,
single-link file at `/etc/muncho-owner-gate/compute-api-hosts`, containing only
the three `199.36.153.8` mappings for `compute.googleapis.com`,
`cloudresourcemanager.googleapis.com`, and `iam.googleapis.com`.  Only the
privileged executor receives that file as its read-only `/etc/hosts` via
systemd `BindReadOnlyPaths`; every other owner-gate service masks the source
path.  Installation is byte-identical on replay, and rollback removes only
that exact managed path after digest, owner, mode, link-count, and symlink
checks, leaving all unrelated host state untouched.
