# Hermes PM Dependency Environment

Hermes' dependency manager (PM) owns every Python environment the agent runs on: the managed
CPython store, the per-install dependency generations under `~/.hermes/installs/<key>/`, and
the `facts.json` record that says which generation an install has selected. This page is the
runbook for that subsystem — how to read its state, how to repair it, and what each failure
message means. It is the page you want when a worker dies at boot with "no dependency
environment is committed for this install" or when `hermes pm doctor` reports a damaged
install.

Written 2026-10-26 by @friday, alongside the fix for the worktree boot defect
(kanban `t_boot_defect_619cbb`).

## Overview

PM replaces the old in-tree `venv/` with a managed model:

- **The store** — `~/.hermes/tools/<package>-<version>-<platform>/` holds published artifacts
  (Python, node, ffmpeg, ripgrep, …). It is a *published artifact*: its realized content is
  hashed and the hash is recorded, so any hand-edit breaks the whole install's verification.
- **Installs** — `~/.hermes/installs/<install_key>/` holds one install's dependency state.
  `<install_key>` is `sha256(resolved checkout path)[:16]` — see
  [[Services/hermes-pm-dependency-environment#The install key is the checkout path]].
- **Generations** — `installs/<key>/environments/<uuid>/venv/` are the built dependency trees.
  One is *selected* at a time; the selection is recorded in `installs/<key>/facts.json`.

Who uses it: every Hermes process (gateway, CLI, cron worker, kanban worker) resolves its
dependency tree through PM before any third-party import. Nothing in Hermes imports a
third-party package outside a PM-selected generation.

## Architecture

```
~/.hermes/
├── tools/                                  # the STORE — published, digest-verified
│   ├── python-3.14.7+20260901-linux-x64/
│   └── facts.json                          # records each artifact's realized digest
├── installs/                               # per-install dependency state
│   ├── 9fe26f155893f1b4/                   # one per checkout path
│   │   ├── facts.json                      # ← the selection record
│   │   ├── environments/
│   │   │   └── 18f83ea57e984d06921b85e28c48c390/venv/
│   │   │       └── lib/python3.14/site-packages/
│   │   ├── inputs/                         # activation-input mtime stamps
│   │   └── pm-runtime/                     # the uv runtime PM uses to build
│   └── 85580d5f6fe54fad/                   # a worktree's key — no facts.json (see below)
└── hermes-agent/                           # the checkout
    └── .worktrees/<task-id>/               # linked git worktrees
```

Resolution chain at process boot (`pm/environments.py::activate_dependencies`):

1. `install_state_dir(project_root)` — `<installs>/<install_key(project_root)>`.
2. If that directory exists → read `facts.json` → the recorded generation is selected,
   under the install's runtime lock.
3. If it does **not** exist → a sealed payload's own venv, else the process keeps whatever
   packages it booted with — and a bare store Python (which owns no dependencies) **refuses**
   with `no dependency environment is committed for this install`.

### The install key is the checkout path

`install_key(project_root) = sha256(str(Path(project_root).resolve()))[:16]`.

Consequence: **a linked git worktree is a distinct install key from the checkout that owns
it**, and it has no `facts.json` of its own. Ten of the twelve worktrees on the hermes host
were in exactly that state as of 2026-10-25.

## Operation

### Reading the state

```bash
# Which generation is selected for a checkout?
cat ~/.hermes/installs/<install_key>/facts.json

# What is the install key of a checkout?
python3 -c "import sys; sys.path.insert(0,'/home/serveradmin/.hermes/hermes-agent'); \
from pathlib import Path; from pm.environments import install_key; \
print(install_key(Path('/home/serveradmin/.hermes/hermes-agent')))"

# Is a given checkout a worktree, and who owns it?
python3 -c "import sys; sys.path.insert(0,'/home/serveradmin/.hermes/hermes-agent'); \
from pathlib import Path; from pm.environments import worktree_owner_root; \
print(worktree_owner_root(Path('<worktree-path>')))"
```

### Health check

```bash
hermes pm doctor
```

Reports each store package and its digest. A healthy install shows `✓ python
3.14.7+20260901`. The realized tree digest must equal the recorded digest in
`~/.hermes/tools/facts.json` (`packages.python.digest`).

### Repair

```bash
hermes pm repair          # rebuild a damaged install's dependencies
hermes pm install         # full dependency sync
```

Never run a bare `pip install` against the managed store — see
[[Services/hermes-pm-dependency-environment#The pip guard]].

### Logs and state locations

| What | Where |
|---|---|
| Selection record | `~/.hermes/installs/<key>/facts.json` |
| Built generations | `~/.hermes/installs/<key>/environments/<uuid>/venv/` |
| Store + digests | `~/.hermes/tools/facts.json` |
| Activation input stamps | `~/.hermes/installs/<key>/inputs/` |
| PM runtime (uv) | `~/.hermes/installs/<key>/pm-runtime/` |
| Worker logs | `~/.hermes/kanban/boards/<board>/workspaces/<task-id>/` (the `worker.log`) |

### Scaling / adding a checkout

A new checkout gets its own install key and must sync its own dependencies
(`hermes pm install`). A new **linked worktree** does not: it borrows the owner checkout's
committed generation (see Troubleshooting).

### Update procedure

`hermes update` rebuilds the generation in place and re-records the selection. A worktree
picking up a new generation needs no action — it borrows whatever the owner has selected.

## Configuration

| Setting | Where | Notes |
|---|---|---|
| `HERMES_HOME` | env | Selects the data root; dependency state is scoped per root. |
| `HERMES_RUNTIME_DIR` | env | Points PM at a non-default store/runtime location. |
| `HERMES_INSTALL_ROOT` | env | The tree a steward wrapper is running from (Nix). |
| `HERMES_DISABLE_LAZY_INSTALLS` | env | Disables the lazy dependency install on first use. |
| install key | derived | `sha256(resolved checkout path)[:16]` — not configurable. |

No secrets live in PM state. `facts.json` holds a path and a stamp, nothing sensitive.

## Troubleshooting

### `hermes: no dependency environment is committed for this install`

**Symptom.** The process exits 1 with exactly that line and nothing else. The worker never
starts; the board card auto-blocks. This was the failure behind ~64 burned worker spawns on
the isolation probe (card `t_86e4721f`, comment 6897).

**Cause.** PM's store Python owns no third-party dependencies. A process that boots on it
must select a committed generation, or PM refuses rather than run on whatever
`PYTHONPATH` it inherited.

**Decision tree.**

1. Is the checkout a **linked git worktree** (`.git` is a *file*, not a directory)?

   ```bash
   test -f <checkout>/.git && echo "WORKTREE" || echo "main checkout"
   ```

   - **WORKTREE** → the worktree has no install state of its own. It must **borrow the owner
     checkout's committed generation**. As of 2026-10-26 this is automatic (`pm/environments.py`
     `worktree_owner_root` + the borrowed-activation path) **provided the worktree's branch
     contains that fix**. If the worktree's branch predates it, merge/rebase the fix into
     that branch — the fix must live in the code the worker *imports*, which is the
     worktree's own checkout (see below).
   - **main checkout** → continue to step 2.

2. Does `~/.hermes/installs/<install_key>/facts.json` exist?

   ```bash
   KEY=$(python3 -c "import sys; sys.path.insert(0,'/home/serveradmin/.hermes/hermes-agent'); \
     from pathlib import Path; from pm.environments import install_key; \
     print(install_key(Path('<checkout>')))")
   ls -l ~/.hermes/installs/$KEY/facts.json
   ```

   - **missing** → nothing was ever committed for this checkout. Run `hermes pm install`
     (or `hermes pm repair`) *from that checkout*.
   - **present but `packages.venv.environment` is null** → the install has a record but no
     committed generation. Same remedy: `hermes pm repair` from that checkout.

3. `packages.venv.environment` points at a tree that no longer exists → the generation was
   collected or the record is stale. `hermes pm repair` from that checkout.

### Why the worker's *own* checkout wins over the dispatcher's pin

This is the part that surprised everyone, and it is the reason the obvious fix (pin
`PYTHONPATH`) does not work.

The dispatcher spawns a worker as `sys.executable -m hermes_cli.main` with `cwd` set to the
task's workspace, and `_propagate_module_import_root` pins the running install's repo root
onto `PYTHONPATH`. But for `python -m`, **`sys.path[0]` is the current directory, and it
outranks `PYTHONPATH`**:

```
$ cd /some/worktree && PYTHONPATH=/home/serveradmin/.hermes/hermes-agent python3 -c 'import sys; print(sys.path[:2])'
['', '/home/serveradmin/.hermes/hermes-agent']
```

`''` resolves to the cwd. So `hermes_cli` is imported **from the worktree**, and the worktree
is the checkout whose install state does not exist. The `PYTHONPATH` pin is a no-op for this
case.

Practical consequence: **a fix to PM only helps a worktree if the fix is in the worktree's
own branch.** Fixing the owner checkout alone changes nothing for a worker that imports
from the worktree.

### `hermes pm doctor` reports the install as damaged

**Symptom.** `pm doctor` shows a digest mismatch for one of the store packages.

**Cause.** Something wrote into the published store tree. Two real incidents on this host:
`psycopg`/`psycopg_binary` on 2026-10-24 (124 files) and `pyyaml` on 2026-10-25 (26 files),
both from a bare `pip install` run against the store.

**Recovery.**

```bash
# 1. Find what changed (files newer than the store's publication).
find ~/.hermes/tools/python-3.14.7+20260901-linux-x64 -newermt '-3 days' -type f \
  | grep -v __pycache__

# 2. Re-publish the artifact through PM (never hand-edit the store).
hermes pm repair
```

### The pip guard

`~/.local/bin/pip` is a wrapper that refuses to install into the managed store, because the
store is a published artifact and mutating it invalidates the digest every later doctor
check verifies against.

| Invocation | Behaviour |
|---|---|
| `pip install <pkg>` (bare) | **REFUSED** with the supported alternatives listed. |
| `pip install --python <other-venv>/bin/python <pkg>` | passes through. |
| `pip install --target <dir> <pkg>` | passes through (deliberately scoped). |
| `pip check`, `pip list`, `pip --version`, `pip download`, … | pass through unchanged. |

Note pip's own ordering rule: `--python` must come **before** the subcommand
(`pip --python <path> install ...`, not `pip install --python <path> ...`).

### The five install states on the hermes host (2026-10-25, before the fix)

| Checkout | install key | facts.json |
|---|---|---|
| `~/.hermes/hermes-agent` | `9fe26f155893f1b4` | yes |
| `.worktrees/t_4d12c8eb` | `2d822d9d2b0f1b21` | yes (its own `pm repair` had been run) |
| `.worktrees/t_efef6176-probe` | `85580d5f6fe54fad` | no |
| `.worktrees/t_ff7d30cc` | `e5b1809050ce8b54` | no |
| `.worktrees/t_3d00480e` | `d4fbbb5b45d78fe3` | no |
| `.worktrees/t_832840f6` | `4f2e09d8db5c04fd` | no |
| `.worktrees/t_399f6b0a` | `b9aac4f2fcdd59f0` | no |
| `.worktrees/t_d2133284` | `63b73dba7a23bce2` | no |
| `.worktrees/t_76ca260a` | `8e0a0c4040ab4974` | no |
| `.worktrees/t_bc6bbb7c` | `baf3a9c710fe9f03` | no |
| `.worktrees/t_258dc2c3` | `8a353efbe7d21f60` | no |
| `.worktrees/t_102333c4` | `8b957b16f6f37ff4` | no |
| `.worktrees/t_15748fe5` | `949d9d29ac8b8fd7` | no |

**Ten worktrees had no install state.** The path prefix `202****0901` in older notes is a
redaction of the build date; the real directory is `python-3.14.7+20260901-linux-x64`.

## Dependencies & Topology

**Depends on:**

- The managed CPython store (`~/.hermes/tools/python-3.14.7+20260901-linux-x64`) — the
  interpreter every Hermes process boots on.
- `uv` from the same store (`~/.hermes/tools/uv-0.12.3-linux-x64`) — PM's builder. The
  runtime copy lives at `~/.hermes/installs/<key>/pm-runtime/`.
- The checkout's `uv.lock` / `pyproject.toml` / `pm/lock.json` — the *activation inputs*;
  their mtimes are stamped in `installs/<key>/inputs/` and compared on every launch to
  decide whether the environment is stale.

**Depended on by:** every Hermes process — the gateway, `hermes` CLI, cron workers, kanban
workers, the ACP adapter.

**Topology:** everything lives under `HERMES_HOME` on the local filesystem. No network
service, no port, no VLAN.

## See Also

- [[Services/fallback-provider-configuration]] — the other half of agent boot: which
  *provider* a worker resolves once its dependencies are selected.
- [[Services/Agent_Vault]] — where model-provider API keys live.
- [[Homelab/50-Agent-Memory/Friday/index|Friday — Memory Index]] — Friday's durable lessons.
- `pm/environments.py` in `ultron-aquabutler/hermes-agent` — the implementation.
- `tests/pm/test_worktree_install_state.py` — the behaviour contract for worktree borrowing.
- Kanban `t_boot_defect_619cbb` — the fix card; `t_86e4721f` comment 6897 — the original
  root-cause write-up.
