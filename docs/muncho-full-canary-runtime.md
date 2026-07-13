# Muncho isolated full-canary runtime

This gate extends the completed writer-only canary. It does not replace the
writer-only activation plan or infer a new owner approval from its success.
The full plan binds the same sealed release, the stopped writer-only receipt,
exact configuration artifacts, and exact disabled systemd unit bytes.

This clean-room gate is deliberately narrower than the intended everyday
Muncho runtime. A successful result must be followed by the separate
[production-shaped capability canary](./muncho-production-capability-canary.md);
the clean-room configuration is never copied directly to production.

The one-shot database bootstrap and compensating retirement SQL are not
discovered from the checkout or from a mutable operator path. The
sealed-release builder copies their exact tracked-index bytes to
`scripts/sql/canonical_writer_canary_bootstrap_v1.sql` and
`scripts/sql/canonical_writer_canary_bootstrap_retire_v1.sql` inside that
release before any project build tooling runs. Each path, size, mode, and
SHA-256 is therefore part of the release manifest and artifact digest.
Runtime accepts only those manifest-declared files and rechecks their bytes
before execution.

## Dedicated-host gate

The full canary is mechanically restricted to one compile-time GCE identity;
the plan cannot choose or override it:

- project ID `adventico-ai-platform`;
- project number `39589465056`;
- zone `europe-west3-a`;
- instance name `muncho-canary-v2-01`;
- instance ID `9153645328899914617`;
- service account
  `muncho-canary-v2-runtime@adventico-ai-platform.iam.gserviceaccount.com`.

A trusted root bootstrap collects those values directly from bounded GCE
metadata reads, plus SHA-256 bindings for `/etc/machine-id`, the kernel
hostname, their combined host identity, and the current kernel boot ID. It
seals the canonical receipt at
`/etc/muncho/full-canary/host-identity.json` as a single-link regular file
owned by `root:root`, mode `0400`. The receipt is a required `ExactArtifact`
of the full-canary plan; its source and target are the same fixed path, so the
lifecycle never manufactures or installs its own host authority.

Stopped preflight validates this gate before invoking even a read-only
service-state runner. It re-reads the sealed file without following symlinks,
checks its exact digest/schema/self-digest, and re-queries metadata and local
host/boot identity. Lifecycle repeats that complete validation immediately
before the first artifact install. A changed VM, service account, machine ID,
hostname, boot, file owner/mode, symlink, replacement, unavailable metadata,
or arbitrary plan value fails closed before installation or service start.
The same host gate protects explicit stop and verify-and-stop entry points so
the canary command cannot target the production VM's similarly named units.
No `gcloud` subprocess and no self-asserted `dedicated=true` field is trusted.

The three runtime services remain:

1. `muncho-discord-egress.service` — owns the Discord token and permits only
   public guild channel/forum/thread operations.
2. `muncho-canonical-writer.service` — owns the database mutation role and
   signs bounded public-egress capabilities.
3. `hermes-cloud-gateway.service` — runs GPT/Hermes and has no Discord token or
   writer private key.

There is no timer and no unit is enabled. Every service is `Restart=no` with a
900-second maximum runtime. Service start order is edge, writer, gateway, with
the trusted collector/config gate between edge and writer. Every normal and
failure stop path is gateway, writer, edge.

## Model-sovereignty configuration

The full gateway preflight requires all of the following:

- model `gpt-5.6-sol` through provider `openai-codex`;
- initial reasoning effort `high`, with model-authored adaptive escalation
  enabled only up to `xhigh`;
- an exact 90-turn ceiling for the sustained task;
- Kanban auxiliary planning, auto-decomposition, notifier, and dispatcher
  disabled;
- clean-room gateway isolation enabled, with cron and built-in memory/profile
  loading disabled;
- curator mutation and built-in pruning disabled;
- no fallback model or fallback provider;
- no Discord platform adapter in the gateway;
- only the sealed `muncho_canary_evidence` observer plugin may be present;
- API concurrency exactly one;
- API model tools exactly `canonical_brain` and `todo` (no terminal, file, or
  messaging authority in this canary);
- Canonical Writer, privileged Discord edge, and model tools enabled.

These checks are mechanical configuration invariants. They do not classify
task text, choose routes, decompose work, or make semantic decisions for the
model.

## Sealed effective gateway process

Preflight validation of staged YAML is not treated as proof of the config the
long-running gateway actually sees. The canary installs the exact file at
`/var/lib/hermes-gateway/.hermes/config.yaml`, sets that directory as
`HERMES_HOME`, points `HERMES_MANAGED_DIR` at an intentionally absent child of
the root-owned mode `0750 root:<gateway-group>` collector runtime, and hides
`/etc/hermes`. The gateway can traverse that parent but cannot create the child,
so the fail-open managed-scope loader observes no managed directory. Preflight
and live readiness require the child to remain absent and revalidate the exact
parent identity and mode. In required-writer mode, `gateway.run`
strictly validates the same stable file again, requires managed scope to be
absent, loads the effective gateway config through the normal runtime path,
and requires semantic equality before it constructs `GatewayConfig`.

The service also starts from an exact environment boundary. Its one
`UnsetEnvironment=` inventory removes inherited provider/model/base-URL,
prompt, Kanban-worker, plugin-loader, task-timeout, alternate credential,
alternate CA-bundle, and proxy selectors. Tmpfiles truncates all gateway-home
and profile-home `.env` and `.op.env` locations to exact zero-byte, mode `0444`,
root-owned regular files, and the unit mounts each path read-only. This is
compatible with Hermes' import-time dotenv loader while proving that none can
inject a second credential surface. Preflight and live readiness re-read each
through a stable no-follow descriptor and require the empty-file SHA-256,
ownership, link count, and mode. The system CA path is pinned to
`/etc/ssl/certs/ca-certificates.crt`, asserted to exist, and read-only in the
service namespace.

Hermes normally lets `$HERMES_HOME/plugins` override a bundled plugin with the
same key. Before exec, tmpfiles therefore materializes the fixed canary user
plugin root as `root:root` mode `0000` and the gateway unit mounts that exact
path inaccessible. A stale user `muncho_canary_evidence` manifest cannot be
scanned or execute before the later immutable-module readiness attestation.

The same pre-exec boundary covers every legacy user-authored startup/context
path reachable from the gateway profile: `$HERMES_HOME/SOUL.md`,
`processes.json`, `plugins/`, `hooks/`, `cron/`, `scripts/`, `memories/`, and
`skills/`, plus the gateway-home `.hermes.md`, `HERMES.md`,
`AGENTS.md`/`agents.md`, `CLAUDE.md`/`claude.md`, `.cursorrules`, and `.cursor`.
Each fixed file or directory is root-created mode `0000` and mounted
inaccessible, so stale code, jobs, process watchers, or text cannot execute or
enter the system prompt before the requested model call.

The sealed YAML additionally pins `gateway.isolated_runtime: true`, literal
`cron.enabled: false`, both built-in memory flags false, and
`kanban.dispatch_in_gateway: false`. The generic gateway keeps its historical
defaults; only explicit isolation suppresses shell/event hooks, process
checkpoint recovery, prior-session mutation and synthetic auto-resume,
Tirith installation, relay provisioning, MCP discovery, Nous keepalive,
startup lifecycle notifications, housekeeping, and continuity/Kanban/handoff
watchers. API-created agents mechanically receive `skip_memory=True` and
`skip_context_files=True`. Cron and housekeeping never start, so the masked
job/script paths cannot create an error loop.

Plugin discovery is sticky-narrow in this process. It scans only immutable
top-level bundled manifests and loads exactly the sealed
`muncho_canary_evidence` standalone observer. User/project/entrypoint plugins,
bundled backends, and bundled platform adapters are not enumerated or
registered; a prior broad discovery, missing observer, failed import/register,
safe-mode skip, or later attempt to broaden discovery fails closed and keeps
the gateway offline. The separate model-provider registry is one-way pinned to
the bundled `openai-codex` profile before its first lookup; it neither scans
user/legacy providers nor imports unrelated bundled profiles, and any prior or
later broadening is fatal. The sole API control adapter is core and must connect
during startup; isolated mode never waits for a reconnect watcher.

Readiness verifies an exact allowlist of effective environment names plus
SHA-256 bindings for every fixed or config-derived value, including the CA
path and 90-turn budget. Dynamic systemd values are allowed only by explicit
name; raw environment values never enter receipts or failure messages. Any
extra semantic selector, alternate credential, proxy, CA hash drift, managed
overlay, or config drift blocks readiness.

Finally, exact fragment paths are insufficient because systemd drop-ins can
override a valid base unit. Every stopped and live service-state gate, plus
the edge-to-collector handoff, therefore requires `DropInPaths` to be empty
for all three units.

## Authenticated loopback control

The canary accepts its one external task through the API server bound to
`127.0.0.1:8642`. The bearer key is loaded once from the systemd credential
basename `api-server.key` sourced from
`/etc/muncho/keys/api-server-control.key`.

The source file must be a root-owned, single-link, mode `0400` regular file.
The gateway receives it through exactly one `LoadCredential=` directive. It
does not accept a simultaneous inline key or `API_SERVER_KEY` environment
value. The key value and its digest are never logged, persisted, or placed in
a receipt.

Stopped preflight also checks, without reading or hashing secret contents:

- the Discord bot token is a non-empty, single-link, mode `0400` regular file
  owned by the edge UID/GID; and
- the OpenAI Codex auth store is a single-link, mode `0600` regular file owned
  by the gateway UID/GID.

Each API run receives a fresh cryptographically random gateway-owned
capability epoch. Only the epoch SHA-256 is placed in request-scoped context;
the HTTP client cannot provide it. Before context is cleared, the exact
session/epoch is durably revoked through the Canonical Writer. When the writer
boundary is required, an unconfirmed revoke fails the API run closed.

The session chat SSE boundary preserves the complete assistant/tool transcript
but emits `run.failed` or `run.partial` for structured failed, partial, or
interrupted agent results. Only an actual complete result can emit
`run.completed`.

## Sealed observer timing

The evidence observer uses fixed paths:

- config: `/etc/muncho/full-canary/observer.json`;
- fixture: `/etc/muncho/full-canary/fixture.json`;
- root collector socket: `/run/muncho-full-canary/collector.sock`;
- Discord edge socket: `/run/muncho-discord-egress/edge.sock`.

The secret-free observer config and fixture are both exact mode `0440`, owned
by `root:<gateway-group>`. This preserves root-only mutation while allowing the
unprivileged gateway plugin to read them. The collector directory is exact
mode `0750 root:<gateway-group>`. The collector socket is `0660
root:<gateway-group>`, while its authenticated peer remains UID/GID `0:0`;
socket ownership and peer credentials are intentionally separate facts.

The final observer config contains exact live collector and edge PIDs, so it
cannot honestly be a fully static pre-start artifact. The permitted sequence
is:

1. validate the sealed plan and stopped state;
2. install exact configs and disabled units;
3. start the Discord edge;
4. the trusted live driver starts the packaged root collector; lifecycle waits
   for and validates its fixed root-owned readiness receipt and both live peer
   identities;
5. mechanically materialize the final root-owned observer config from the
   sealed template plus observed PIDs, and persist a digest receipt;
6. start the writer and gateway;
7. require the plugin to synchronously send sequence-1 `plugin_ready` over the
   authenticated collector socket before hooks register;
8. wait for the collector's distinct canonical, root-owned mode `0400`
   `/run/muncho-full-canary/plugin-readiness.json` receipt. It binds the exact
   gateway MainPID/start time, sealed plugin module, observer config, fixture,
   one-shot scope, edge/collector identities, authenticated frame digest, and
   collector hash-chain head;
9. execute the one authenticated loopback session;
10. verify plan-addressed collected evidence;
11. stop gateway, writer, and edge even if verification fails.

No operator-supplied evidence path is accepted. The verifier consumes only:

`/var/lib/muncho-full-canary/plans/<release>/<plan>/live/evidence.json`

and binds it to the exact start-receipt file SHA-256 and all three service
identity receipt digests.

## One-shot Canonical scope

Normal API identity is intentionally not treated as Discord or owner identity.
Before the full plan is built and approved, the trusted driver generates the
raw API session key in memory and places only its SHA-256 in the exact writer
config's `canary_scope_preapproval`. That immutable config is itself a digested
plan artifact; lifecycle never edits it after plan approval.

The preapproval is bound to grant, case, sealed release artifact, fixture,
canary run, session-key digest, owner, approval-source digest, and an expiry
exactly equal to the fixture end time. The plugin claims it once through the
writer with `canary-scope-claim:<grant_id>`. The trusted API runtime supplies
the session and capability epoch; no Discord/user identity is spoofed. Before
request context is cleared, the epoch is durably revoked. Missing claim,
registration receipt, tombstone, or exact evidence fails the canary closed.

### Private bootstrap and mandatory retirement

The normal writer role cannot grant itself the private preapproval function.
That one-time authority is installed by the sealed
`canonical_writer_canary_bootstrap_v1.sql` and is always reconciled by the
separate sealed `canonical_writer_canary_bootstrap_retire_v1.sql`. The runtime
does not expose a generic administrator SQL runner, accept an administrator
password, invoke a shell, or put a secret in argv, logs, or receipts.

The writer config names the fixed login
`canonical_brain_canary_bootstrap_login` and the separate credential path
`/etc/muncho/credentials/canonical-canary-bootstrap-db-password`. Stopped
preflight requires that credential to be a single-link `0400` file owned by
the exact writer UID/GID. It also requires a digest-bound managed-HBA rejection
receipt for that same login, database host, TLS server name, and port. This is
separate from the ordinary writer login and from the ephemeral administrator
session used to apply the sealed SQL.

Database mutation is blocked by default. The only live entry is an explicitly
supplied `PreopenedSessionBootstrapProvisioner` wrapping an already-open,
owner-operated managed-administrator PostgreSQL session plus its verified TLS
peer-certificate digest. Connection establishment and authentication remain
outside the runtime, so there is no persistent administrator credential or
session factory in Hermes. The runtime independently rechecks database,
session/current user, and backend PID. It rechecks the same identity after a
`ROLLBACK` fence immediately before retirement SQL, rejecting a silently
reconnected backend.

The provisioning request sets exactly eleven digest-bound PostgreSQL settings:
ten immutable scope/database values plus the non-circular provisioning
authorization digest. After owner approval and a final dedicated-host recheck,
the lifecycle applies the sealed provisioning SQL immediately before writer
start. Writer readiness must then contain the exact Canonical consumption
receipt. The staged writer config is atomically replaced with the same sealed
config minus `canary_scope_preapproval`, and a plan-addressed append-only
root-owned `0400` tombstone binds the original config, retired config, writer
readiness, and consumption receipt.

Retirement is mandatory after every provisioning attempt, including a failed
or malformed provisioning response. It sets the original eleven values plus
exact plan, owner-approval, and current-executor digests (fourteen total), then
applies the second sealed SQL. Its thirteen-column result has only these
terminal states:

- `consumed`: authorization and consumption events exist; no retirement event
  is written; reason is `bootstrap_consumed`;
- `retired`: authorization exists but consumption does not; the fixed
  `activation_failed_before_consumption` retirement event is appended once or
  replayed idempotently;
- `not_authorized`: provisioning never committed; no terminal event exists;
  reason is `provisioning_not_committed`.

All three prove the bootstrap role/login have no remaining schema/function ACL
and that temporary migration-owner membership is absent. A recovery attempt
may use a new pre-opened administrator backend after an uncertain commit. Its
receipt binds the current executor session, while replay compares durable
terminal truth without treating the previous backend PID as durable identity.
Administrator close is idempotent, retried once after a transient failure, and
the lifecycle/driver retains no administrator object after the attempt.

Bootstrap consumption is not the final authority boundary. Once the writer
has inserted the owner-bound preapproval, a later startup, gateway, or
verification failure must not leave that still-claimable row behind. The full
writer unit therefore invokes the same sealed writer executable with the
private `--reconcile-canary-preclaim` mode at both boundaries:

- `ExecStartPre` reconciles the preserved staged plan source before any writer
  restart. On the first activation the exact scope is not yet preapproved, so
  this is a bounded no-op. After a crash or power loss it retires prior durable
  state before a writer or gateway can become live again.
- `ExecStopPost` repeats the same fixed call after every writer stop attempt,
  including a failed start. It uses a fresh writer-UID database connection;
  no public writer operation, model tool, administrator credential, SQL text,
  shell, or caller-selected scope is exposed.

Both calls read only the preserved plan-bound staged writer configuration and
atomically replace the fixed writer-owned mode `0600` receipt at
`/run/muncho-canonical-writer/preclaim-reconciliation.json`. Claim and
retirement serialize in PostgreSQL. An unclaimed preapproval receives one
append-only retirement event. A committed claim receives a durable tombstone
for its exact session/capability epoch before cleanup can be reported as
complete. Replay returns the original durable event identities.

Lifecycle accepts cleanup only when the receipt is exact for the sealed grant
and proves `authority_active=false`. A claimed scope that is still active, a
missing or stale receipt generation, a mismatched event identity, or an
unproven preapproval state is `cleanup_blocked` and requires owner recovery;
it is never converted into terminal success. The old staged source must remain
unchanged until this next-start recovery has completed, so staging a newer plan
cannot erase the only exact binding for an interrupted one.

## Operator entry points

The packaged module is invoked from the sealed interpreter:

```text
python -B -I -m gateway.canonical_full_canary_runtime validate --phase stopped
python -B -I -m gateway.canonical_full_canary_runtime start
python -B -I -m gateway.canonical_full_canary_runtime verify-and-stop \
  --start-receipt <plan-addressed-start-receipt> \
  --evidence-sha256 <collected-evidence-sha256>
python -B -I -m gateway.canonical_full_canary_runtime stop
```

`start` requires the canonical root-owned plan and a fresh root-owned owner
approval at their fixed paths. The standalone entry point remains blocked
unless its owner-operated process supplies the already-open provisioner; it
never reads an administrator secret itself. It never enables a unit. Receipt
output includes the exact on-disk receipt SHA-256 so the live evidence can bind
the file rather than a self-declared internal field.

## Owner coordinator and crash recovery

The packaged `gateway.canonical_full_canary_coordinator` is the only remote
owner-control entry point. The local owner launcher remains a separately
attested, source-only program; neither component is imported by the normal
gateway/model loop.

Every owner invocation first runs the read-only `preflight-recovery` command.
An active run or recovery-worker journal is recovered in two distinct stages:

1. The coordinator emits a no-secret takeover gate. The owner returns only the
   exact `MRA1` acknowledgement. After exact predecessor termination, the
   exclusive process lock and a full journal CAS, the worker durably advances
   through `claimed_awaiting_admin` and
   `admin_authority_may_be_in_use`.
2. Only then does the worker emit a fresh nonce-bound secret gate. The owner
   may send the recovery-only `MRC2` frame, whose header binds that exact gate
   and nonce before any username or password byte is read. The normal-run
   `MCA2` frame is never accepted on this recovery path.

Successful cleanup first publishes
`muncho-full-canary-recovery-worker-completion.v1` with
`recovery_worker_exit_proven=false` and
`safe_to_delete_temporary_admin=false`. The worker cannot attest its own exit.
A separate no-secret `finalize-recovery` process proves the exact worker has
exited (or terminates that exact PID through pidfd), reacquires the lock, and
CAS-publishes `muncho-full-canary-recovery-receipt.v2`. Only that terminal v2
receipt authorizes deletion of an administrator credential that may have been
disclosed. A valid legacy v1 receipt is recognized only to return the explicit
`legacy_recovery_receipt_reconciliation_required` blocker; it is never
silently consumed, upgraded, or used to request a secret.

Final runtime approval uses an independent request capped at 240 seconds, with
an owner-input cutoff 30 seconds before its deadline. EOF before the first
`MFA1` byte produces the zero-secret cancellation receipt v2. That receipt
reports the exact active, expired, retired, superseded, or drifted state of the
request, staged plan, and prior approval artifacts. Mixed or conflicting
artifact states are `cancelled_no_secret_state_conflict`, never a fabricated
clean cancellation. Partial `MFA1` is a hard ambiguous failure.

### Temporary Cloud SQL authority

Cloud SQL user mutations are asynchronous. The owner launcher sends each
`CREATE_USER`, `UPDATE_USER`, or `DELETE_USER` request at most once and never
turns a timeout, redirect, rate limit, HTTP 499, or server response into an
implicit retry. Only a narrow, reviewed set of definite client rejections can
prove that a create was not committed. Every other uncertain response enters
reconciliation, and no `MCA2` or `MRC2` credential byte is sent from that run.

Positive credential authority requires the exact response-known operation,
expected operation type, owner identity, successful terminal outcome, exact
temporary user presence, and unchanged complete paginated operation/user
snapshots. The same live proof is repeated inside the local first-byte write
guard, followed by owner-identity and expiry checks with a 30-second delivery
margin. Any intervening user operation or evidence drift sends zero credential
bytes. The Discord-token `DCT1` path uses the same first-byte expiry boundary.

This proof assumes the isolated canary SQL instance has no concurrent user
mutator after the final read. The canary must be stopped if that operational
assumption cannot be guaranteed; API reads cannot provide a transactional lock
against a mutation accepted after their final snapshot.

Negative cleanup truth is deliberately separate from positive authority. An
unknown-response run may prove the exact temporary user absent without ever
claiming that a candidate operation belonged to it. It must observe complete,
warning-free paginated operation and user snapshots for one continuous
180-second quiet window, polling no faster than every five seconds and resetting
the window on any operation identity, type, status, actor, outcome, user
presence, or delete change. A late change can extend observation to the bounded
360-second hard horizon; insufficient quiet remains `cleanup_blocked`.

Cloud SQL publishes no visibility-latency guarantee that turns this operational
horizon into a mathematical proof. Accordingly, a no-candidate result may only
record bounded absence, zero secret disclosure, and a blocked run; a fresh
invocation must start from a new preflight. Receipts keep preflight, recovery,
and fresh-run evidence phase-bound and expose the response-known-candidate flag,
post-baseline operation count, quiet window, and a secret-free evidence digest.
