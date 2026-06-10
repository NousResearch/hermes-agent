# Bot Factory Hardening — Implementation Plan

## Goal

Make the multi-tenant architecture (client registry + soft isolation + claude-proxy)
safe and durable enough to onboard paying clients on. The control plane
(`hermes_cli/clients.py` and friends) stays as-is — it's the right design. The
work concentrates on the data plane (`deploy/claude-proxy/`) and on turning
documented invariants into mechanical ones.

Each phase is independently shippable and leaves the system better than before;
nothing in a later phase blocks an earlier one.

## Decision log (what we are committing to)

| Decision | Choice | Why |
|---|---|---|
| Client control plane | Keep the declarative registry + soft isolation | One-flag graduation shared→container, no data migration; scales to multi-host later (registry becomes scheduler input) |
| Proxy tool access | **Pure text completion — no tools, ever** | The proxy serves untrusted end-user prompts from all tenants; an agent with file/Bash tools + shared volume = cross-tenant exfiltration via prompt injection |
| Tenant identity | Per-client bearer keys on :11435, riding the existing custom-provider `api_key` mechanism | Gateways already send `Authorization: Bearer <api_key>` to OpenAI-compatible endpoints; no new protocol needed |
| Proxy backend | Pluggable backend interface; CLI today, Anthropic API as the target (Phase 4 decision point) | Retires the OAuth/ToS + single-account-ban risk; per-tier model map finally makes the registry's `tier`/`model` fields real |
| Proxy code location | Baked into the image, not the data volume | The volume copy drifts from `deploy/claude-proxy/` in the repo; deploys must update the proxy |
| Double-poll prevention | Enforced at boot from the registry, not documented in comments | Telegram-409 has bitten twice (template clone, isolated+shared); invariants this load-bearing must be mechanical |

---

## Phase 0 — Close the cross-tenant hole (ship first, smallest diff)

The proxy currently runs `claude -p --permission-mode bypassPermissions` with
`cwd=/opt/data`. Prompts originate from tenants' end users; Claude Code's
default toolset can read `/opt/data/secrets/*.env` and every profile's data.

Changes in `deploy/claude-proxy/claude_proxy.py`:

1. **Strip all tools** from both the main call and the Haiku classifier call:
   add `--tools ""` (pure text completion; verified against the CLI reference
   2026-06). Belt-and-braces: also pass `--disallowedTools "*"` so a future
   `--tools` regression still denies everything.
2. **Drop `--permission-mode bypassPermissions`.** With no tools requested
   nothing needs permissions; if a tool call ever slips through, non-interactive
   default mode denies it instead of executing it.
3. **Move the workdir off the data volume**: default `CLAUDE_PROXY_WORKDIR` to a
   dedicated empty dir (`/opt/data/proxy/workdir`, created 0700 at startup) so
   even a flag regression has nothing to look at.
4. **Resolve `CLAUDE_BIN` robustly**: env override → `shutil.which("claude")` →
   glob of the current hardcoded node_modules path. Today's deep hardcoded path
   breaks silently on a claude-code update.
5. **Atomic + debounced cache persistence**: `_cache_save` writes temp file then
   `os.replace` (same pattern as `add_client.write_doc`); coalesce saves to at
   most one per ~30s instead of one thread per `cache_set`.

Acceptance:
- Unit test asserts the spawned argv contains `--tools ""` and does **not**
  contain `bypassPermissions` (extend `test_claude_proxy.py`'s command-builder
  coverage).
- Manual probe on dev: a prompt asking to read `/opt/data/secrets` returns
  refusal text, not file contents.
- `/health` still green through the deploy health gate.

## Phase 1 — Tenant identity, auth, and tenant-scoped caching

Today :11435 is unauthenticated (any process in the container) and the 24h
cache is keyed only on `(tier, system, prompt)` — shared across all tenants.

1. **Key issuance at provision time** (`hermes_cli/provision_client.py`):
   generate a random per-client key, write it into the profile's provider
   config for the :11435 endpoint (the gateway sends it as
   `Authorization: Bearer`), and append `key → client` to a host-side map
   `/opt/data/proxy/keys.json` (0600). Idempotent: existing key is kept.
2. **Proxy-side auth**: requests without a valid bearer key get 401. Key map
   reloaded on mtime change so provisioning never restarts the proxy.
   A `CLAUDE_PROXY_ALLOW_ANON=1` escape hatch defaults **off** in prod, on in
   single-user dev so existing setups don't break mid-migration.
3. **Tenant in the cache key**: `_cache_key(tenant, tier, system, prompt)` for
   the good cache and negative cache. Breaker stays global (it measures Claude
   health, not tenant behavior).
4. **Tenant in observability**: tenant id on every `claude_call:` log line and
   per-tenant request/token counters in `/health`. This is the seed of usage
   metering for billing later.
5. **Registry**: document the key convention in `clients.example.yaml`; the
   secret itself follows the existing `secrets/` pattern, never the registry.

Acceptance: 401 without key; same prompt from two tenants produces two cache
entries; provisioning a client twice keeps the same key; tests for key load,
auth, and tenant-scoped keys.

## Phase 2 — Make the "never run a bot twice" invariant mechanical

`compose_gen.py` warns in comments that isolated clients must be excluded from
the shared gateway; nothing enforces it, and runtime profile discovery ignores
the registry entirely.

1. **Registry-aware boot reconcile**: when `$HERMES_CLIENTS_REGISTRY` is set,
   `02-reconcile-profiles` (or a small `04-registry-gate` hook to avoid
   touching upstream's script) skips seeding per-profile gateway services for
   clients marked `isolation: container`, and logs a `DRIFT:` warning for any
   on-disk profile absent from the registry (visibility first; no auto-delete).
2. **Generator emits the exclusion artifact**: `compose_gen --output` also
   writes `<data_root>/isolated.list` consumed by the boot hook, so the compose
   file and the exclusion can't disagree.
3. **Deploy gate add-on**: after health passes, assert via `s6-svstat` that the
   set of running per-profile gateways matches the registry's
   shared-prod set — duplicates or strays fail the deploy loudly.

Acceptance: flip a client to `isolation: container`, redeploy → shared gateway
does not start it (test the filter function); hand-made unregistered profile →
`DRIFT:` line in boot log; gate test in the workflow.

## Phase 3 — Proxy lifecycle into the image (kill volume drift)

The running proxy is whatever copy sits on the data volume under
`$HERMES_HOME/services/claude-proxy/`; the repo's `deploy/claude-proxy/` is
just documentation. They *will* diverge.

1. **Bake** `claude_proxy.py` and its s6 service definition into the image
   (`COPY` in the Dockerfile next to the existing cont-init hooks). The s6
   `run` script `exec`s the image copy; only env/config stays on the volume.
2. `03-seed-data-services` keeps working for any *other* volume-defined
   service, but seeds the proxy def from the image so a rebuilt container
   always runs the code that was reviewed and tested in the repo.
3. One-time migration in the deploy script: retire the volume copy (rename to
   `services/claude-proxy.legacy`), keeping rollback trivial.

Acceptance: editing `deploy/claude-proxy/claude_proxy.py` + deploy changes the
running proxy (verify via a version string in `/health`); health gate green;
volume copy no longer load-bearing.

## Phase 4 — Backend swap: OAuth CLI → Anthropic API (the decision point)

This retires the two existential risks on the revenue path: consumer-OAuth
terms-of-service exposure, and a single account ban taking every bot down.
It also removes the per-request Node.js subprocess tax.

1. **Extract a backend interface** inside the proxy:
   `Backend.complete(system, prompt, tier, tenant) -> result`, with
   `ClaudeCliBackend` (current behavior) and `AnthropicApiBackend`
   (direct `/v1/messages` call with an API key; per-tier model map, e.g.
   haiku→`claude-haiku-4-5`, sonnet→`claude-sonnet-4-6`, opus→`claude-opus-4-8`).
   Selected by `CLAUDE_PROXY_BACKEND=cli|api`. Cache/negative-cache/breaker/
   logging all sit above the interface and don't change.
2. **Fix the classifier cost** regardless of backend: classify keyword-first,
   call Haiku only for ambiguous prompts, and cache classifications by prompt
   hash (today every uncached request spawns a *second* claude subprocess with
   a 15s timeout).
3. **Per-tenant budgets**: daily token counters per tenant in the proxy; over
   budget → 429, which the gateway's existing fallback chain turns into a
   cheap-model answer instead of an outage. This is what the registry's `tier`
   field finally drives (tier → budget + default model).
4. **Rollout**: dev env first with `backend=api`, compare `claude_call:` logs
   and breaker behavior for a few days, then flip prod. The CLI backend stays
   as an emergency fallback for one release, then is removed.

Open business input needed before flipping prod: API spend per client vs.
current subscription cost — the per-tenant counters from Phase 1 give the
real numbers to decide with.

## Phase 5 — Scale posture (do when load demands, not before)

- Replace `ThreadingHTTPServer` with an ASGI app (uvicorn) and add streaming
  when bot latency starts to matter; not worth it at current volume.
- Registry-driven *continuous* reconciliation (registry as runtime authority)
  once Phase 2's drift warnings prove the registry is being kept honest.
- Multi-host: the registry grows a `host` field and `compose_gen` becomes the
  per-host scheduler input. Don't build this before a second host exists.
- Ops hygiene riders: scheduled `client_split --backup` via cron; prune without
  `ignore_errors` (log each failed removal); non-root deploy SSH user.

## Sequencing & effort

| Phase | Scope | Risk if skipped | Size |
|---|---|---|---|
| 0 | Proxy sandbox + bin/cache robustness | Cross-tenant secret exfiltration | S (1 file + tests) |
| 1 | Tenant keys, auth, scoped cache, metering seed | Tenant data bleed, no usage data for pricing | M |
| 2 | Registry-enforced no-double-run + drift visibility | Repeat Telegram-409 outages, silent drift | M |
| 3 | Proxy baked into image | Prod runs unreviewed code; deploys don't deploy | S |
| 4 | API backend + budgets | ToS/account-ban on the revenue path; subprocess tax | L |
| 5 | ASGI/streaming, multi-host | None today | — |

Phases 0–3 are pure hardening of what exists and have no business dependencies.
Phase 4 needs the cost decision; everything before it makes that decision
cheaper and reversible.

## Phase 6 — Hermes owns the fleet (privileged orchestrator)

Principle: **Hermes is the only actor that maintains the fleet** — it owns the
code, spins new bots, and operates existing ones. Client bots stay unprivileged
(consistent with the proxy lock: only Hermes reaches Claude).

Two channels, both self-provisioned by the deploy (no manual key handling):

1. **Code ownership — git-ops.** Hermes edits the repo and pushes to `main`; the
   existing gated pipeline (`deploy-contabo.yml`: build → health gate →
   proxy-lock gate) ships it. No new host surface.
2. **Host fleet ops — a bounded SSH broker.** The deploy creates an
   unprivileged `hermes-ops` host user whose SSH key is forced
   (`command="…/fleet_broker.py"`, no-pty/no-forwarding) to run ONLY an
   allowlisted set of `docker`/`compose` actions — never a shell. The broker
   keypair is generated on the host and the private key dropped onto the gateway
   volume; the in-container `hermes fleet` client
   (`hermes_cli/fleet.py`, `python -m hermes_cli.fleet`) reaches it over loopback
   ssh (the gateway runs `network_mode: host`). Every call is audited.

Division of labor: the **registry-aware brain stays in-container** —
`hermes fleet generate` runs the tested `compose_gen` to write
`docker-compose.clients.yml` + `isolated.list` onto the volume; the host broker
only runs `up`/`down`/`restart`/`apply`/`status`/`logs`/`ps` against those
generated artifacts. This is what makes "graduate a client to its own container"
(`isolation: container` → `fleet generate` → `fleet up <client>`) a Hermes-driven
operation rather than a human host step.

Known limitation (soft isolation): co-located client gateways run as the same
`hermes` UID, so filesystem perms can't hide the broker key from them. The
forced-command allowlist — not key secrecy — is the real boundary: the worst a
co-located bot can do with the key is run the same bounded fleet ops. Graduating
sensitive clients to their own containers (exactly what this tooling enables)
restores hard isolation.
