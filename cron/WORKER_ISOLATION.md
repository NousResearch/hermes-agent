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

## The agent home IS the production code tree

`<agent_home>` is the tree the sandbox protects, and it must be the **production CODE tree** —
the checkout the agent's own source lives in. On this fleet that is `~/.hermes/hermes-agent`
(the `hermes-agent` repo), which is a DIFFERENT git checkout from `~/.hermes` itself (the
`hermes-config` repo that holds the profiles and the kanban workspaces).

Resolution order (`hermes_cli/kanban_db_dispatch.py::_resolve_worker_agent_home`):

1. `HERMES_AGENT_HOME` — explicit operator override; wins outright.
2. The git toplevel of the RUNNING `hermes_cli` package (`_production_code_repo_root`) — the
   authoritative source, because it cannot drift.
3. `HERMES_AGENT`, then `HERMES_HOME` and its parent — last-resort fallbacks.

Getting this wrong is not a harmless mismatch. The launcher flips `<agent_home>` read-only and
carves the writable paths out of a staging bind of `dirname(<agent_home>)`, so:

- Pointing it at the config repo leaves the CODE tree writable, and
- every scratch workspace (`<config-repo>/kanban/boards/<board>/workspaces/<id>`) looks like it
  lives INSIDE the protected subtree, so `_is_outside_agent_home` refuses the wrap outright and
  the dispatcher spawns the worker UNWRAPPED.

That second failure is what shipped on 2026-10-26: the resolver derived the agent home from
`HERMES_HOME` (the config repo), so **zero** workers were isolated while the code looked
correct. The regression test is
`tests/hermes_cli/test_worker_isolation_agent_home.py` — it builds the fleet's two-repo layout
and asserts the dispatcher WRAPS a scratch task.

## Carve-out set

The only writable paths are the per-task worktree (or, for a scratch task, the per-task scratch
directory), plus `<agent_home>/.git/objects`, `<agent_home>/.git/logs`, and the worker's own
branch ref dir under `refs/heads/`.
`.git/packed-refs` is deliberately NOT carved out, and the branch carve-out is narrow enough
that a worker cannot `update-ref refs/heads/canonical-deploy` to a different commit — a
validated escape during integration.

## Spawn paths

- **Kanban dispatcher** — `hermes_cli/kanban_db_dispatch.py` wraps the worker argv.
- **Cron** — `cron/scheduler.py::_cron_isolation_argv` wraps the cron-restart-safe argv the
  same way; a cron worker has no per-task worktree, so its carve-out set comes from the agent
  home's own `.git/HEAD` and the production WORKING TREE stays read-only.

Both worktree-anchored AND scratch-anchored tasks are wrapped. A scratch workspace must
resolve OUTSIDE the agent home (the fleet layout puts it at
`<parent>/kanban/boards/<board>/workspaces/<id>`, a sibling of the code tree under the same
parent). A `dir` workspace, a `manual`/`none` kind, and a scratch path inside the agent home
are SKIPPED with a warning — wrapping them would either over-grant write access to production
or hide a bug.

## Kernel caveats (this host)

`mount -o remount,bind,*` on a bind mount HANGS in uninterruptible sleep instead of returning
`EACCES`/`EPERM`. The launcher uses the single-step `-o bind,ro` form, and `_mount` bounds every
mount call with a 15s timeout so a hung mount is a failed mount rather than a wedged dispatcher.
Supplementary groups are also lost inside the namespace; no `git fetch` inside a sandboxed
worker.

## See also

- Wiki: `docs/wiki/services/canonical-deploy-branch.md` (carve-out set, spawn paths, caveats).
- Drift guard: `hermes_cli/deploy_guard.py`.
