# Muncho isolated full-canary runtime

This gate extends the completed writer-only canary. It does not replace the
writer-only activation plan or infer a new owner approval from its success.
The full plan binds the same sealed release, the stopped writer-only receipt,
exact configuration artifacts, and exact disabled systemd unit bytes.

This clean-room gate is deliberately narrower than the intended everyday
Muncho runtime. A successful result must be followed by the separate
[production-shaped capability canary](./muncho-production-capability-canary.md);
the clean-room configuration is never copied directly to production.

The live canary has no one-shot database role, login, grant, or compensating
SQL path. Writer foundation changes happen separately in the stopped-only
Phase-B protocol. Its exact tracked migration artifacts, packaged Python
runtime, terminal receipt, and in-process readiness handoff are bound into the
sealed release before a live plan can be built. The live coordinator accepts
only that terminal readiness descendant and never receives a database
administrator credential.

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

The three live runtime services remain:

1. `muncho-discord-egress.service` — owns the Discord token and permits only
   public guild channel/forum/thread operations.
2. `muncho-canonical-writer.service` — owns the database mutation role and
   signs bounded public-egress capabilities.
3. `hermes-cloud-gateway.service` — runs GPT/Hermes and has no Discord token or
   writer private key.

The separate
`muncho-canonical-writer-phase-b-readiness.service` attests the already
completed writer foundation in process; it is not a model or routing service.
There is no timer and no unit is enabled. Every live service is `Restart=no`
with a 900-second maximum runtime. Service start order is edge, writer,
gateway, with the trusted collector/config gate between edge and writer.
Every normal and failure stop path is gateway, writer, edge, then the readiness
unit.

## Model-sovereignty configuration

The full gateway preflight requires all of the following:

- model `gpt-5.6-sol` through provider `openai-codex`;
- initial reasoning effort `high`, with model-authored adaptive escalation
  enabled only up to `max`;
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

## Session-bound Canonical authority

Normal API identity is intentionally not treated as Discord or owner identity.
The trusted coordinator binds the generated API session digest, capability
epoch, sealed release, fixture, run, plan, and exact owner approval before any
live service starts. The raw session key remains in memory and never becomes
database authorization material, a receipt field, or an operator-editable
writer setting.

The writer exposes only the fixed least-privilege operation catalog. Case
access is derived mechanically from server-observed session, thread, owner,
handoff, and capability state. Dangerous mutations require the applicable
plan-addressed owner capability; ordinary reads do not. At terminal cleanup
the exact session/capability epoch is durably revoked, so a replayed request
cannot regain authority.

There is no canary-specific database role, login, one-shot SQL grant, or
administrator credential in the runtime. Stopped preflight verifies the
packaged writer, fixed configuration, single writer credential, database
identity, privilege attestation, and in-process readiness. The live lifecycle
then starts edge, collector, writer, and gateway in the sealed plan order and
stops them in reverse order on either success or failure.

Readiness and final evidence must bind the same release, plan, owner approval,
session digest, capability epoch, fixture, service identities, and append-only
Canonical receipts. A missing binding, stale generation, unexpected authority,
or incomplete session retirement fails the run closed and leaves an explicit
owner-recoverable failure receipt.

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

These commands are mechanical runtime primitives. The target live path is the
session-bound coordinator, which supplies the exact in-memory owner approval
and one-shot API key to the driver. No runtime command accepts a database
administrator or provisioning object, and no command enables a unit. Receipt
output includes the exact on-disk receipt SHA-256 so live evidence binds the
file rather than a self-declared internal field.

## Session-bound owner coordinator

The packaged `gateway.canonical_full_canary_coordinator` is the only remote
owner-control entry point. The local owner launcher remains a separately
attested, source-only program; neither component is imported by the normal
gateway/model loop.

The owner launcher first validates the terminal Phase-B readiness gate. It then
installs the Discord token through its dedicated framed boundary and opens one
sealed coordinator process for the complete live run. That same process builds
the final plan, emits the plan-bound approval request, receives the owner's
`MFA1` frame, runs the driver, proves stopped terminal state, and retires the
Discord credential. There is no remote approval file, second remote coordinator
process, database administrator handoff, bootstrap credential, or external
recovery worker. The local launcher consumes one owner-only, one-shot approval
file bound to the just-emitted request.

On failure, the coordinator emits one self-digested terminal receipt containing
only observable cleanup facts: exact command and release bindings, whether all
services are stopped, whether the Discord token is gone, and whether an
obsolete process journal is absent. It never fabricates successful cleanup.
The owner closes the sealed session and, if token installation had completed,
uses only the dedicated token-retirement boundary. A later attempt begins with
fresh read-only Phase-B/live preflight rather than resuming hidden authority.

Phase-B foundation mutation remains a separate stopped-only owner protocol.
Its bounded Cloud SQL operations may exist only while applying the reviewed
writer foundation and must end in a terminal readiness receipt before the live
gate opens. No Phase-B credential or mutation capability is accepted by the
live coordinator, gateway, model, or Canonical Writer runtime.
