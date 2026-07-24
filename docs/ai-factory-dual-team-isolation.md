# AI Factory dual-team isolation (HER-96)

HER-96 is a **control-plane gate**, not prompt policy and not a second backlog.
Linear remains selection/backlog; the shared Kanban board remains execution. A loop
must run `team-admit` successfully before it creates or mutates its Kanban task.
The command stores the team lease, canonical worktree, validated freshness verdict,
and worker identity in the existing factory registry.

This is deliberately separate from generic Kanban dispatch: imposing HER/SCA mapping
on every board task would break unrelated profiles. A HER/SCA loop controller owns
the narrow pre-create sequence below.

## Deterministic controller config

Store a non-secret JSON file outside the repository's tracked config, owned by the
operator. It maps exactly one profile to each team and declares the issue-team
mandates that profile may own. Unknown, ambiguous, or unmandated mappings fail
closed; a profile can take a non-default lane only when the controller config
explicitly authorizes that lane's team.

```json
{
  "freshness": {
    "canonical_branch": "origin/main",
    "max_age_seconds": 3600
  },
  "teams": {
    "HER": {
      "profiles": ["default"],
      "allowed_teams": ["HER"],
      "job_id": "<default-loop-cron-job-id>",
      "gateway_started_at": "2026-07-23T14:00:00Z"
    },
    "SCA": {
      "profiles": ["hermes-immo"],
      "allowed_teams": ["SCA"],
      "job_id": "<hermes-immo-loop-cron-job-id>",
      "gateway_started_at": "2026-07-23T14:00:00Z"
    }
  }
}
```

`gateway_started_at` is updated by the loop controller when its gateway starts.
Status rejects a builtin tick that predates this time; a historical or manual run
cannot prove the current loop is alive.

`freshness.max_age_seconds` is a whole number from 1 through **86,400** (24 hours).
This bounds freshness as short-lived evidence; epoch zero and non-finite or
fractional windows are invalid and cannot create an admission lease.

## Freshness evidence (required before `team-admit`)

### Portable filesystem trust boundary

All evidence, controller configuration, registry owner, and journal reads first
walk their ancestors with `O_NOFOLLOW` directory descriptors. Linux further
pins the leaf with `O_PATH`, checks that the pinned inode is regular, then
reopens that same inode for content. This is the strongest path and remains
mandatory when available.

Darwin has no `O_PATH`. On that platform a content read is permitted only from
an already-pinned immediate parent directory that is not group- or
other-writable. Such a directory is the explicit trust boundary: another local
UID cannot replace its leaf between validation and open. The leaf is then
opened with `O_NOFOLLOW|O_NONBLOCK` and checked as regular before any content
is consumed. A group/world-writable parent fails before any content-capable
leaf open, so an attacker-controlled FIFO, character device, or block device
cannot be opened. This preserves normal macOS operation for operator-owned
controller files and factory registries without claiming that same-UID hostile
writers are isolatable by portable POSIX APIs.

Freshness is a fixed schema, not a pass-through document. Only the documented
keys below are accepted; unknown fields, oversized text, newlines, and
credential-shaped text (Authorization/Bearer, JWT, provider or GitHub prefixes,
and PEM keys) are rejected before owner persistence. `team-status` emits the
normalized allowlisted evidence, never the raw manifest or a raw cron error.

The loop first searches newer Linear issues, newer PRs/commits, and current-main
behavior for the same feature/tool. It writes the result as a bounded, non-symlink
JSON evidence file:

```json
{
  "issue": "HER-96",
  "checked_at": "2026-07-23T14:45:00Z",
  "canonical_branch": "origin/main",
  "canonical_head": "<40 lowercase hex SHA>",
  "sources": {
    "newer_linear_issues": ["HER-97"],
    "newer_prs_commits": ["<40 lowercase hex SHA>"],
    "current_main_behavior": {
      "checked": true,
      "summary": "Current main behavior was probed."
    }
  },
  "verdict": "current"
}
```

Allowed verdicts are `current`, `superseded`, `duplicate`, and `needs-rebase`.
`superseded` and `duplicate` block admission before an owner or Kanban task exists.
`needs-rebase` is allowed only with this additional proof that the bug reproduces
on current main:

```json
"current_main_red": {
  "reproduced": true,
  "evidence": "RED: <focused current-main reproduction>"
}
```

The manifest is required on every team admission, which is stricter than only
checking old issues and prevents an unclassified item from becoming mutable work.
The validated, normalized evidence is persisted in `owner.json` and returned by
runtime status.

## Pre-create sequence

For a HER loop selection, use the same sequence with the corresponding SCA values:

```bash
python scripts/factory_lane.py \
  --registry /ABS/registry \
  team-admit HER-96 \
  --team-config /ABS/dual-team.json \
  --freshness-evidence /ABS/her-96-freshness.json \
  --profile default \
  --agent hermes-code \
  --session <loop-session> \
  --owner-pid <long-lived-loop-pid> \
  --worktree /ABS/hermes-her-96-dual-team-loop-isolation
```

Only after exit 0 may the loop call `kanban_create` / create a task. On a non-zero
exit it must report the gate and make no task/worktree mutation. This guarantees:

- a profile cannot select a lane unless its team's `allowed_teams` explicitly
  contains that lane's team;
- one live lease exists per team;
- a physical worktree cannot be used by both teams, including a symlink alias;
- unrelated profile lanes keep using the legacy admission path.

## Runtime truth

```bash
python scripts/factory_lane.py \
  --registry /ABS/registry \
  team-status --team-config /ABS/dual-team.json --json
```

Each team record is derived only from registry owners and cron execution/job data.
It contains `profile`, `team`, `job`, `lane`, `worktree`, `worker`, `heartbeat`,
`gate`, `next_run_at`, the latest execution, and the latest builtin tick after
`gateway_started_at`. `source=direct` remains visible as an audit attempt but never
counts as recurrence proof. A chat answer/transcript is deliberately absent and
cannot substitute for this status payload.

Builtin `cron.scheduler.tick()` also calls
`recover_interrupted_executions()` before due-job selection. A dead owner therefore
becomes terminal `unknown` on the next builtin tick without a scheduler restart or
a retry side effect.

## Local canary (no install or restart)

1. Create a temporary registry, two empty temporary worktrees, config, and freshness
   manifests.
2. Admit `default/HER-*` and `hermes-immo/SCA-*` simultaneously; verify status has
   two leases.
3. Attempt both cross-routes, a second team owner, and a shared worktree; verify each
   fails and creates no new owner file.
4. Create a direct cron execution after a builtin one; verify status shows it only as
   `latest_execution`, while `last_builtin_tick_after_gateway_start` stays builtin.
5. Simulate a dead execution owner and run a scheduler tick; verify it becomes
   `unknown` before due-job lookup and future builtin ticks remain eligible.
6. Do not wire the controller into a live gateway, create live Kanban work, install
   scripts, or restart a service as part of this canary.

## Rollback

The change is additive. Before installation, discard this branch's changes and keep
the existing HER-95 admission gate. If a controller config has been staged, remove
its `team-admit` call and the `dual-team.json`/manifest files; then no new HER-96
team leases can be created.

Do **not** delete the existing registry as rollback: it is audit evidence. If an
already-admitted test lane must be released, use the existing `factory_lane close`
flow only after its worker is confirmed stopped. Existing generic Kanban dispatch,
profiles, and cron jobs are unaffected by removing the optional controller wiring.
