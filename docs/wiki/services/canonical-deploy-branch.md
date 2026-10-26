# Canonical Deploy Branch (t_7161d9a9)

> Wiki page for the structural-loop root-cause remedy.
>
> Decision: **Option A** (Bryan, 2026-09-23 — verbatim reply: "Option A").
> Pin the deployed tree to one canonical branch. All fixes merge there
> before deploy. Workers use separate worktrees.
>
| Status: shipped on branch `canonical-deploy-t_7161d9a9` @ `4f914e1d23`.
|> Acceptance pass at the time of writing: 146 passed, 1 skipped across
|> `test_deploy_guard` + `test_kanban_db*` + `test_kanban_decompose*` +
|> `test_kanban_specify*` + `test_kanban_block_kinds` +
|> `test_kanban_auto_decompose_quarantine_regression` +
|> `test_interrupt_scaffold_echo`.
|>
|> **Layer 2 enforcement (t_ff7d30cc):** the convention above is now
|> mechanically enforced — every kanban-spawned worker is launched inside an
|> unprivileged user+mount namespace in which the production tree is
|> read-only at the syscall level (EROFS). The launcher (`hermes_cli.worker_isolate`)
|> is applied by the dispatcher to the worker argv with the proven nesting
|> `systemd-run --user --scope` → `unshare -Urm` → launcher → worker. Config
|> gate: `kanban.worker_isolation` = `off|warn|enforce`, default `enforce`
|> on Linux+userns; env kill switch `HERMES_WORKER_ISOLATION=off` for
|> emergencies. See the `Worker isolation` section below for the full carve-out
|> set and the caveats that ship with the sandbox (lost supplementary groups;
|> no `git fetch`/`pull` inside a sandboxed worker; cron scheduler is a
|> second spawn path assessed separately by t_efef6176).
>
> **Doc-in-commit gap:** the Obsidian vault write path on this host
> has been returning 201 but not persisting since 2026-09-18 (LXC
> in-process cache, see t_f1551b54). This stage-file in the worktree
> is the source of truth; the wiki page will be re-issued when the
> vault cache clears.

## Overview

The hermes-agent production tree at
`/home/serveradmin/.hermes/hermes-agent` is the live source for the
whole agent fleet: gateways, dispatcher, kanban engine, auto-decomposer,
watchers. It is a mutable git checkout. Workers that pick up a card
checkout their fix branch into the **production** tree, which silently
wipes whatever was on the previous branch.

This was the root cause of two confirmed silent-fix-loss incidents:

| Date | Lost fix | Source branch | Discovered |
|---|---|---|---|
| 2026-09-17 22:27 | human-gate eligibility + `hub_escalation` quarantine gate (t_8b48a01f / t_334f608b) | `fix/state-db-busy-timeout` → `fix/kanban-recompute-ready-blocker-gate` | 2026-09-23 |
| 2026-09-23 17:36 | the restoration itself (t_efc7769a, 4a18baa139) + its regression test (t_9f3c5d4a, 6067b5a712) | `fix/kanban-human-gate-and-quarantine-restoration` → `fix/respawn-guard-decay-t_cfbbb112` | 9 minutes later, in a hub audit |

`done` previously meant "the fix existed somewhere at the time", not
"the fix is in production". Every closed fix was provisional.

## Architecture

Three pieces, all in the canonical-deploy branch:

1. **Canonical ref.** `canonical-deploy-t_7161d9a9` is the durable
   tip that the deployed tree tracks. Today its tip is `4f914e1d23`,
   branched from `fork/fix/kanban-recompute-ready-blocker-gate` (the
   prior de-facto canonical lineage) and merged with:
   - the gate-fix lineage (`4a18baa139` + `6067b5a712`),
   - the quota-cooldown split (`29bf7edfb5`, t_cfbbb112),
   - the interrupt-scaffold echo filter (`ed6dab63ba`),
   - this deploy-branch guard (`4f914e1d23`).

2. **Worker isolation.** Workers that need to do fix-branched work
   operate in `git worktree add` siblings under
   `~/.hermes/hermes-agent/.worktrees/<task-id>`. The production
   tree's working directory is NOT shared with workers.

   **As of t_ff7d30cc the isolation is mechanically enforced**: every
   worker spawned by the dispatcher enters an unprivileged
   user+mount namespace where the production tree is `EROFS` at the
   syscall level. A worker attempting `git checkout`, `git switch -c`,
   `git reset --hard`, `git pull`, `git update-ref canonical-deploy`,
   `git branch -f canonical-deploy`, `git worktree add`, or a nested
   `unshare -Urm` + remount-rw escape is denied by the kernel, not by
   convention. The launcher carves out exactly this rw set (no more):

   - the dispatcher-provisioned worktree (the worker can write here),
   - `<agent_home>/.git/objects` (shared object store a commit needs),
   - `<agent_home>/.git/refs/heads/<branch-dir>` (ONLY the directory
     containing the worker's own branch ref, e.g. `wt` for a branch
     named `wt/<task-id>`; `refs/heads` as a whole is NOT carved out
     because a worker that can update arbitrary refs can move
     `canonical-deploy` to a different commit — the narrow carve-out
     refuses cross-branch moves while still letting the worker update
     its own branch ref via the standard git plumbing),
   - `<agent_home>/.git/logs` (reflog),
   - `<agent_home>/.git/worktrees/<this-worktree-name>` (this worktree's
     `index`/`HEAD` — carved out per name, never the whole `.git/worktrees`).

   **Caveats** (measured, not surprises):

   - Supplementary groups are lost inside the user namespace; local
     group-gated sockets (`/var/snap/lxd/common/lxd/unix.socket`,
     group `lxd`) keep working, but the kernel maps the worker's
     in-namespace uid 0 back to the host's `serveradmin` via
     `SO_PEERCRED`, so lxd sees the worker as fully-trusted
     `serveradmin` (MORE trust than expected, not less). Mitigation
     if a flow needs lxd-as-another-user: map the gid via
     `newgidmap`/subgid, or run that one op outside the sandbox.
     Remote access is unaffected (docker on this host is already an
     ssh context).
   - `git fetch`/`pull` inside the worker's own worktree fails —
     `FETCH_HEAD` lives in the ro main `.git`. The dispatcher must
     provision the worktree before the sandbox exists; `hermes -w`
     from inside a worker is unavailable by design.
   - The cron scheduler (`cron/scheduler.py`) is a separate spawn
     path; coverage is owned by `t_efef6176` (verification card).
   - Sandbox-safe `mount` exits with distinct codes (91=stage bind,
     92=prod bind, 93=remount-ro, 94=carve bind) so a worker log
     surfaces the first failing mount immediately.

   Configuration: `kanban.worker_isolation = off|warn|enforce` in the
   resolved config; `HERMES_WORKER_ISOLATION=off` env var as a kill
   switch. The gate degrades HONESTLY when the kernel cannot deliver
   isolation (no unprivileged userns, non-Linux host): the worker
   still spawns but a single warning fires once per dispatcher
   process, never a per-spawn flood, never a silent pretend.

3. **Self-enforcing gate.** `hermes_cli.deploy_guard.check_deploy_branch()`
   runs on every dispatcher tick. A wrong branch, detached HEAD, or
   missing repo emits a structured WARNING + stderr banner. Drift is
   never fatal from the dispatcher's perspective (a stalled board is
   worse than a drifted one), but the cron pump captures the
   banner within one tick window (≤ 2 minutes) instead of the
   multi-day window that hid the 2026-09-23 incident.

The companion drift watcher (polls `git rev-parse HEAD` between cron
ticks and alerts on SHA change) is filed as a child card for Ultron.

## Operation

### Verify the deployed tree is on canonical

```bash
cd /home/serveradmin/.hermes/hermes-agent
./venv/bin/hermes deploy-guard check
```

Exit 0 = match. Exit 2 = drift. The output names `expected_branch`,
`observed_branch`, `observed_sha`, `reason`, and `agent_home`.

### One-time setup for a new operator

```bash
./venv/bin/hermes deploy-guard init
```

Appends `HERMES_DEPLOY_BRANCH=canonical-deploy-t_7161d9a9` to
`~/.hermes/.env` (mode 0600) if not already present. Idempotent.

### Promote a fix to canonical (Option A workflow)

```bash
cd /home/serveradmin/.hermes/hermes-agent
# 1. Worker already did the work in a sibling worktree:
#    .worktrees/<task-id>/ on a fix/* branch, all tests green.
# 2. From the canonical branch tip, fast-forward or merge:
git checkout canonical-deploy-t_7161d9a9
git merge --no-ff fix/<task-id>-<short-title>
git push fork canonical-deploy-t_7161d9a9
# 3. Re-point the deployed tree:
git checkout canonical-deploy-t_7161d9a9
# 4. Restart gateways so the next import resolves the merged tip:
bash /home/serveradmin/.hermes/start-gateway.sh
# 5. Verify on the deployed tree (acceptance must measure here):
./venv/bin/hermes deploy-guard check
./venv/bin/python -m pytest tests/hermes_cli/test_kanban_db.py tests/hermes_cli/test_kanban_decompose.py
```

### Force the dispatcher to abort on drift (testing only)

```bash
export HERMES_DEPLOY_GUARD=strict
# The next dispatch tick raises DeployBranchDriftError.
# DO NOT enable in production — a stalled board is worse than a drifted one.
```

## Configuration

| Env var | Default | Effect |
|---|---|---|
| `HERMES_DEPLOY_BRANCH` | `canonical-deploy-t_7161d9a9` | The expected branch name. |
| `HERMES_DEPLOY_GUARD` | `warn` | `off`/`warn`/`strict`. See deploy_guard.py. |
| `HERMES_AGENT_HOME` | `$HERMES_AGENT` or `/home/serveradmin/.hermes/hermes-agent` | The working tree to inspect. |
| `HERMES_WORKER_ISOLATION` | `enforce` on Linux+userns | Emergency kill switch for the worker sandbox. `off` falls back to the worktree convention (no kernel-level EROFS). Read `kanban.worker_isolation` from the resolved `config.yaml` to set the steady-state mode. |

The deploy guard reads `HERMES_AGENT_HOME` first, then `HERMES_AGENT`,
then the default path. This layering keeps the gate usable inside
containerized deploys (`HERMES_AGENT_HOME=/var/lib/hermes-agent`) and
on the host (`HERMES_AGENT` set by `start-gateway.sh`).

## Troubleshooting

### Symptom: every dispatch tick logs `[deploy-guard] DRIFT`

The deployed tree is on a non-canonical branch. Either:

- A worker checked out their fix branch into the production tree
  (the bug we just fixed — they should be using `.worktrees/`).
- An operator manually checked out a feature branch.

Fix:

```bash
cd /home/serveradmin/.hermes/hermes-agent
git checkout canonical-deploy-t_7161d9a9
```

If the deployed tree has uncommitted work (`git status --porcelain` is
non-empty), capture it first:

```bash
git stash push -u -m "pre-canonical-restore-<date>"
git checkout canonical-deploy-t_7161d9a9
git stash branch <branch-name>  # if the work should land
```

### Symptom: `hermes deploy-guard check` reports `detached`

A review worktree is checked out at the canonical SHA without a
branch ref. Either reattach:

```bash
git checkout canonical-deploy-t_7161d9a9
```

Or, if the detached state is intentional (a sandbox review), set
`HERMES_DEPLOY_GUARD=off` for that worktree.

### Symptom: `reason=missing_repo` or `reason=git_unavailable`

The deploy-guard cannot find a git repo at `HERMES_AGENT_HOME` (or
`git` is not on PATH). For a packaged install (no `.git/`), set:

```bash
export HERMES_DEPLOY_GUARD=off
```

For a misconfigured `HERMES_AGENT_HOME`, unset it and rely on the
default path detection.

### Symptom: dispatch tick returns `DeployBranchDriftError`

`HERMES_DEPLOY_GUARD=strict` is set. This is intentional (test
environments only). Production should always use the default `warn`.

### Symptom: worker logs show `[worker_isolate] degraded to off: ...`

The host cannot deliver unprivileged mount-namespace isolation (no
`unshare --user --map-root-user true`, non-Linux, or already inside a
user namespace). The worker still spawns and the carve-out convention
still applies; you lose the kernel-level EROFS backstop. Acceptable
on dev machines that don't ship the agent tree; investigate before
disabling on a production host. Set `HERMES_WORKER_ISOLATION=off`
only as a temporary kill switch, never as the steady state.

### Note: lxd unix.socket is NOT sandboxed

The lxd unix socket (`/var/snap/lxd/common/lxd/unix.socket`, group `lxd`)
**keeps working** inside the sandbox. Even though supplementary groups
are lost (`ogroups=` → `nogroup` in the new namespace), the kernel
maps the worker's in-namespace uid 0 back to the host's `serveradmin`
via `SO_PEERCRED`, so lxd's trust check returns fully-trusted
`serveradmin`. A worker running under `unshare -Urm` therefore has
MORE lxd trust than a normal host shell would have as a non-root user
— the opposite of what a casual reading of "groups lost" suggests.
This is not currently a live flow (no worker uses lxd today), but if
a future flow needs to invoke lxd as a *lesser* identity, it must
either map the gid via `newgidmap`/subgid, or run that one op outside
the sandbox by setting `HERMES_WORKER_ISOLATION=off` for the task.

## Dependencies

- `git` binary on the host (the guard shells out to `git rev-parse`).
- The dispatcher (`hermes kanban dispatch`) as the integration point;
  every other CLI command is gated only via `hermes deploy-guard check`.
- The cron pump `/home/serveradmin/.hermes/scripts/kanban_dispatch_pump.sh`
  for visibility — the WARNING log + stderr banner land in the cron
  stdout that already gets delivered to ops channels.

## See Also

- [Kanban Block-Loop Quarantine](kanban-block-loop-quarantine.md) — the
  gate-fix lineage that this branch carries (`4a18baa139` + `6067b5a712`).
- [Hermes Kanban Dispatcher](../hermes-cli/kanban.md) (t_7161d9a9 staging
  note: this page is still being authored; reference commit hashes
  instead until it lands).
- t_8b48a01f / t_334f608b — original 2026-08-28 guard pair.
- t_efc7769a — restoration that triggered this card.
- t_cfbbb112 — quota-cooldown split, also staged on the canonical branch.
- t_ea3bc1a1 — respawn-guard event storm; the worktree pattern is
  already in use by this card (`fix/respawn-guard-event-storm-t_ea3bc1a1`).
- t_4ac398f0 — restart card; sequencing corrected to "merge to canonical
  → deploy → verify → restart" by this card.
- t_f4806674 — design decision: chose `unshare -Urm` + ro bind + rw
  carve-outs after measuring that `systemd-run --user --scope` refuses
  `ReadOnlyPaths=` (`Unknown assignment`) and a transient systemd
  service cannot inherit the dispatcher-built worker env.
- t_ff7d30cc — implementation: `hermes_cli/worker_isolate.py` +
  `_isolation_worker_argv` wired into `_default_spawn`; worker argv is
  now wrapped in `unshare -Urm <launcher> -- <worker argv>` so the
  production tree is `EROFS` inside the worker.
- t_7161d9a9 — this card.

## Doc-in-commit Caveat

The Obsidian vault REST write path on this host has been returning
201 Created but not persisting since 2026-09-18 (LXC in-process cache,
t_f1551b54). This stage-file in the worktree is the source of truth;
a follow-up kanban comment + dispatch cycle will re-issue the wiki
page when the vault cache clears.