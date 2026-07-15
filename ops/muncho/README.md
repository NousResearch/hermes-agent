# Muncho fork-only operations

This directory contains fork-only operational helpers for Cloud Muncho. It is
not an upstream Hermes product surface and must not be proposed to
`NousResearch/hermes-agent`.

The helpers deliberately reuse existing Hermes behavior instead of widening
the core:

- `auto_sync_hardening.py` proves superseded automation-owned sync PR states
  from explicit Git ancestry and deduplicates unchanged structured blockers.
- `mechanical_job_rail.py` packages the fork-only sync routine as one exact,
  release-addressed systemd service/timer.  The rail has a single code-owned
  allowlist entry, a GitHub-only systemd credential, digest-bound sources,
  private idempotent receipts, and no model/provider/Discord dependency.  Its
  timer waits at least 30 minutes after activation and packaging never starts
  the job.  It creates fork branches/PRs only; auto-merge/deploy remains a
  separate gate and the public upstream is read-only.
- `planned_gateway_restart.sh` writes the existing Hermes planned-stop marker
  before an external service manager restarts the gateway.
- `production_config_model_sovereignty.py` plans, applies, and rolls back the
  exact model-sovereignty config delta. Mutation is digest-bound, requires the
  gateway to be stopped, publishes crash-safe exact backups, and returns a
  stable receipt without interpreting task meaning.

Cloud integration keeps the mutable wrappers outside the active release:

- `/opt/adventico-ai-platform/hermes-home/scripts/fork_upstream_auto_sync_pr_routine.py`
- `/usr/local/sbin/muncho-auto-deploy-release`

Rollout of any mutable helper is a separate exact-action production change.
The deployment wrapper restores the previously active release automatically
when target restart or post-start health verification fails; its caller must
then invoke the approved config rollback receipt path and restart that previous
release once more so it reloads its previous config. A
`deploy_rolled_back` receipt therefore proves code-link/service restoration,
not config restoration; a production transaction that changed config is
complete only after the separate config rollback receipt and final service
health/readback both pass.

Legacy Hermes cron scripts are never copied into the mechanical rail.  The
production cutover first uses `gateway.production_cron_migration` to inventory
the existing store with the same static startup validator.  An apply requires
an unchanged inventory, exhaustive per-record owner continuity dispositions,
an exact rail package manifest, and a stopped gateway.  The read-only
inventory defaults every incompatible record to `pending_review` and is never
executable.  The explicit dispositions distinguish compatible agent jobs,
agent/mechanical migrations, stale retirement, and preservation as inert
history; a blanket pause of multiple incompatible jobs is rejected.  The
inert apply path accepts at most one exact `preserve_inert` record, archives
the original store, and never executes, deletes, translates, or interprets a
prompt.  Its rollback restores the exact archive while keeping gateway start
blocked until a fresh production inventory is reviewed.

The host collector proves `/usr/bin/gh` and `/usr/bin/git` are exact
root-owned, one-link, non-group/world-writable executables and binds their
digests into the unit.  It proves the GitHub credential is one root-owned
`0400` file without reading or recording its content, size, or digest.  The
public package manifest and host/inventory validators are pure and suitable
for the owner-signed cutover authority bundle.

## Legacy deployment boundary

`muncho-auto-deploy-release` is retained only for the trusted, pre-cutover
gateway topology whose unit starts Python through the mutable active-release
symlink. Before either scheduling or running a deployment it reads the exact
loaded systemd identity. Once that identity is the signed, SHA-pinned gateway,
the legacy path is permanently fail-closed before clone, packaging, link
changes, service restart, release cleanup, or a `deploy_pass` receipt.

Partial cutover state, a missing or non-canonical cutover plan, or any unit,
drop-in, release-marker, or digest drift is treated as ambiguous and is also
blocked. The only write on either blocked path is its stable operational
receipt. Production deployment then belongs exclusively to the separately
owner-approved signed cutover transaction. Pull-request synchronization and
its merge/queue decisions do not invoke this deployment boundary and remain
unchanged.

## Node runtime scope

The Node version pinned by Muncho's release-local browser dependency is only
the runtime for `agent-browser` and Chrome for Testing. It is not authority for
the SkyVision website build, PM2 process, or production deployment.

The website merge/deploy gate stays fail-closed until the SkyVision repository
and production host independently prove one exact Node/npm contract across the
build user, privileged build path, and PM2 runtime, together with the
incident-required canary and multi-hour soak evidence. A Muncho dependency
manifest must never satisfy or override that separate gate, and this repository
must not hard-code a website Node patch version that belongs to the SkyVision
release contract.

The February incident degraded only after sustained runtime, so a short
``online`` probe is not sufficient evidence.  The website gate requires a
5--10% canary and a 2--4 hour soak with RSS/heap, event-loop delay, p95 latency,
5xx responses, and process restarts observed, followed by a tested rollback.
Exact Node, npm, lockfile, install command, and executable paths must match
across build, staging, the privileged deployment path, and the PM2 process.
The incident involved application resource growth plus proxy/WAF pressure; it
must not be reduced to an unsupported claim that a Node version alone caused
the outage.
