# Worker mount-namespace isolation

Long form for the `Worker mount-namespace isolation` section of `cron/AGENTS.md`. Kept as a
sibling doc so the AGENTS.md chain stays inside its context budget
(`scripts/ci/check_agents_md_size.py`).

**The production tree is read-only at the syscall level for every kanban-spawned worker.**
`hermes_cli/worker_isolate.py` is the launcher: the dispatcher wraps the worker argv with
`unshare -Urm <launcher> -- <worker argv>`; the launcher binds the production tree
`mount -o bind,ro`, then carves the writable paths back out of an rw staging bind. So
`git checkout` / `reset --hard` / `pull` / `update-ref` against the production tree are denied
by the kernel (`EROFS`), not by convention — the rule three silent-revert incidents
(2026-09-17, 2026-09-23, 2026-09-24) needed enforced.

## Gate

`kanban.worker_isolation` = `off|warn|enforce` (default `enforce` on Linux+userns);
kill switch `HERMES_WORKER_ISOLATION=off`. Degradation is honest: if isolation cannot be
applied the worker spawns UNWRAPPED and the helper warns once.

## Carve-out set

The only writable paths are the per-task worktree, plus `<agent_home>/.git/objects`,
`<agent_home>/.git/logs`, and the worker's own branch ref dir under `refs/heads/`.
`.git/packed-refs` is deliberately NOT carved out, and the branch carve-out is narrow enough
that a worker cannot `update-ref refs/heads/canonical-deploy` to a different commit — a
validated escape during integration.

## Spawn paths

- **Kanban dispatcher** — `hermes_cli/kanban_db_dispatch.py` wraps the worker argv.
- **Cron** — `cron/scheduler.py::_cron_isolation_argv` wraps the cron-restart-safe argv the
  same way; a cron worker has no per-task worktree, so its carve-out set comes from the agent
  home's own `.git/HEAD` and the production WORKING TREE stays read-only. Only worktree-anchored
  tasks are wrapped; scratch tasks, `dir` workspaces and an unresolvable agent home are skipped
  with a warning.

## Kernel caveats (this host)

`mount -o remount,bind,*` on a bind mount HANGS in uninterruptible sleep instead of returning
`EACCES`/`EPERM`. The launcher uses the single-step `-o bind,ro` form, and `_mount` bounds every
mount call with a 15s timeout so a hung mount is a failed mount rather than a wedged dispatcher.
Supplementary groups are also lost inside the namespace; no `git fetch` inside a sandboxed
worker.

## See also

- Wiki: `docs/wiki/services/canonical-deploy-branch.md` (carve-out set, spawn paths, caveats).
- Drift guard: `hermes_cli/deploy_guard.py`.
