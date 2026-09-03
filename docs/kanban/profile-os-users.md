# Kanban profile-to-Linux-user execution

Optional `kanban.profile_os_users` maps a canonical Hermes profile id to a
POSIX account so dispatcher-spawned workers run as that account instead of
the gateway UID.

This is **off by default**. An empty or missing mapping preserves the existing
trusted-local-user behaviour (workers inherit the gateway UID). Do not enable
the mapping until `hermes kanban os-users check` passes **and** the live
gateway/dispatcher argv is this reviewed commit.

Isolation is **false** while the running gateway still uses the installed/canonical
Hermes without this feature.

## Contract

```yaml
kanban:
  profile_os_users:
    dev: hermes-dev
    sysadmin: hermes-sysadmin
  # Optional companion: Hermes *runtime root* (NOT OS HOME).
  # Default for a mapped user is {passwd_dir}/.hermes
  # profile_os_homes:
  #   dev: /home/hermes-dev/.hermes
```

Pin the shared board with the supported env (do not chmod `~/.hermes` group-writable):

```bash
export HERMES_KANBAN_DB=/home/matt/.hermes/kanban/kanban.db
```

- Launch mechanism: argv list `sudo -n -H -E -u <user> -- <hermes...>` (no shell,
  no setuid helper, no configurable command prefix).
- Fail closed: a configured mapping never falls back to the gateway UID.
- Root mappings are rejected. Same-UID mappings are rejected and are **not**
  reported as isolation.
- OS `HOME` is the passwd home. `HERMES_HOME` is `{root}/profiles/<id>` (or
  `{root}` for `default`). `profile_os_homes` overrides the Hermes root only.

Gateway/default retains administrative visibility of specialist homes if the
host operator wants emergency SSH. Specialists must not read one another's
`HERMES_HOME` or secrets. Workspaces are granted per-project (for example Dev
gets WorkoutTracker; Sys-admin is not granted that ACL). World-readable trees
(for example `/home/matt/Documents` or WorkoutTracker if mode allows other-read)
are **not** an isolation claim.

## Account and group design

Each specialist gets a **private primary group** with the same name as the
account, plus **supplemental** membership in the shared `hermes-kanban` group:

1. `groupadd --system hermes-kanban`
2. `groupadd --system hermes-dev` (private primary)
3. `useradd --system --create-home --home-dir /home/hermes-dev --shell /usr/sbin/nologin -g hermes-dev -G hermes-kanban hermes-dev`
4. Same for `hermes-sysadmin`.

Homes stay `0700` owned by that private group. Do **not** use
`useradd --gid hermes-kanban` — that makes the shared group the primary group
and breaks `install -g hermes-dev` plus cross-profile home isolation.

A pre-existing account whose primary group is `hermes-kanban` (or any other
shared group) is **incompatible**. `setup --apply` treats useradd/groupadd
"already exists" return codes as success only after proving the account matches
this design.

## Shared SQLite board (dedicated directory)

The live board file today is `/home/matt/.hermes/kanban.db`. SQLite WAL/SHM
sidecars are created **next to that file**. Granting `g:hermes-kanban:wx` (plus a
default ACL) on `/home/matt/.hermes` lets specialists create/remove/rename known
entries in the entire Hermes root even without listdir. That is not least privilege.

Least privilege:

- Dedicated shared directory: `/home/matt/.hermes/kanban/` owned `2770`
  `matt:hermes-kanban`. The board file is `/home/matt/.hermes/kanban/kanban.db`.
- Configure it explicitly: `HERMES_KANBAN_DB=/home/matt/.hermes/kanban/kanban.db`.
- Traverse-only (`g:hermes-kanban:--x`, no listdir, **no write**) on `/home/matt/.hermes`
  and ancestors.
- Write/execute without listdir (`g:hermes-kanban:wx`) **only** on the dedicated
  `kanban/` directory so WAL/SHM can be created there.
- Default ACL `d:g:hermes-kanban:rw` on that dedicated directory.
- `g:hermes-kanban:rw` on `kanban.db` itself.

Never put a write ACL on `/home/matt/.hermes`. Never `cp` a live DB or its
`-wal`/`-shm` sidecars. Cutover uses `hermes kanban os-users migrate-db`
(sqlite backup API) after a bounded gateway stop you control.

`hermes kanban os-users check` must prove both mapped users can complete a SQLite
WAL lifecycle **in the dedicated directory** (fail-closed `sudo -n` probe). Being
a directory is not enough. Check fails if the write parent is the Hermes root.

## Dev workspace ACL

Canonical path: `/home/matt/Documents/WorkoutTracker`.

`/home/matt` is typically mode `0750`, so a leaf-only ACL is unusable. Setup
grants:

- Traverse-only `u:hermes-dev:--x` on ancestors (`/home/matt`, `/home/matt/Documents`).
- Recursive `u:hermes-dev:rwx` on the repo.
- Default ACL `u:hermes-dev:rwx` so newly created files inherit access.

Sys-admin is **not** granted this tree. That is not confidentiality if the tree
is world-readable; isolation proofs are deny of `/home/matt/.ssh`, specialist
homes, and credentials.

## Toolchain (narrow, not `/home/matt`)

A separate UID does not inherit Matt-owned Flutter/Android/JDK paths. Setup may
grant **read/execute only** on:

- `/home/matt/flutter`
- `/home/matt/Android/Sdk`
- `/home/matt/.local/opt/jdk-17`

plus traverse `--x` on ancestors. Caches are **private** under
`/home/hermes-dev/.cache/{flutter,pub,gradle,android}` (`0700`). Do not
recursively ACL `/home/matt`. Check proves as `hermes-dev`: workspace traverse,
toolchain `x`, and writable private cache. Mapped env keeps `PUB_CACHE`,
`GRADLE_USER_HOME`, `ANDROID_SDK_ROOT`, `ANDROID_HOME`, `JAVA_HOME`, `FLUTTER_ROOT`.

## GitHub continuity (manual, secret-safe)

Copying profile `.env`/`config.yaml`/`SOUL.md`/`skills` does **not** provision
`~/.config/gh` or git HTTPS helpers. Do not copy Matt's SSH keys. As root in a
tty you start:

```bash
sudo -u hermes-dev -H gh auth login --hostname github.com --git-protocol https
# Prove (stdout omitted from audit; never print tokens):
sudo -n -u hermes-dev -- /usr/bin/gh api user
sudo -n -u hermes-dev -- /usr/bin/git ls-remote https://github.com/NousResearch/hermes-agent.git HEAD
```

Check fails closed if those probes fail.

## Deploy the reviewed commit first

PR code lives in this worktree/fork. `sudo hermes` may invoke the **old installed
CLI** without `os-users`. Do not use it.

Preferred: wait for upstream merge + upgrade, then provision.

Alternative local path: install this SHA into a versioned runtime
`/opt/hermes/kanban-os-users/<sha>/` and point `HERMES_BIN`, the gateway unit,
and generated sudoers at **that** argv. Check's `runtime-sha` gate proves the
worker/gateway command covers this tree; isolation is false until it does.

This worktree uses `.venv`, not `venv`:

```bash
cd /home/matt/.hermes/worktrees/kanban-profile-os-users
.venv/bin/python -m hermes_cli.main kanban os-users setup \
  --gateway-user matt \
  --dev-workspace /home/matt/Documents/WorkoutTracker
```

Apply with the same argv under sudo (never `sudo hermes`):

```bash
sudo /home/matt/.hermes/worktrees/kanban-profile-os-users/.venv/bin/python -m hermes_cli.main \
  kanban os-users setup --apply \
  --gateway-user matt \
  --dev-workspace /home/matt/Documents/WorkoutTracker
```

`--apply` requires euid 0. It does **not** copy profile files unless you pass
`--migrate-profile-files` (bounded `copy-tree`, not per-file flood). It does
**not** migrate the live DB unless `--migrate-shared-db`. Contents are never printed.
Dry-run summarizes planned skill file/dir counts per root.

## Ordered rollout (mid-step failure must leave the live board recoverable)

1. Deploy this reviewed commit (merge+upgrade, or versioned runtime + `HERMES_BIN`).
   Do not enable `profile_os_users` yet.
2. Create private groups/users and specialist homes (`0700`).
3. Create `~/.hermes/kanban/` with group wx. Hermes root gets traverse-only (`--x`), never write.
4. Quiesce the gateway (stop/restart window **you** control). Live `~/.hermes/kanban.db` stays until backup succeeds.
5. `kanban os-users migrate-db --from /home/matt/.hermes/kanban.db --to /home/matt/.hermes/kanban/kanban.db`
   (sqlite backup API, not `cp` of WAL/SHM).
6. Pin `HERMES_KANBAN_DB` on the gateway unit. Restart onto the versioned runtime.
7. Toolchain r-x ACLs + private caches. No recursive ACL on `/home/matt`.
8. `--migrate-profile-files` (bounded copy-tree), then `gh auth login` as `hermes-dev`.
9. `hermes kanban os-users check` must pass SHA/runtime, WAL-in-dedicated-dir, gh/git,
   toolchain, deny `.ssh`/credentials. World-readable WorkoutTracker is not isolation.
10. Only then enable `kanban.profile_os_users` in the **gateway** config.yaml and restart.

Do not enable mappings from a Kanban worker. Do not run sudo from this worker.

## Rollback

```bash
.venv/bin/python -m hermes_cli.main kanban os-users rollback    # print-only
# Then as root, after removing profile_os_users from config.yaml:
# 1. Restart onto the previous HERMES_BIN (mapping-off is the recoverability gate).
# 2. Point HERMES_KANBAN_DB back at the pre-cutover file if the dedicated copy is untrusted.
#    Keep the backup; do not delete the live DB first.
sudo rm -f /etc/sudoers.d/hermes-kanban-os-users
sudo visudo -c
```

Rollback **always** reverses recorded ACLs (`setfacl -x` / `-k`). It `userdel`s /
`groupdel`s **only** principals this setup created (recorded in
`/var/lib/hermes/kanban-os-users-state.json`). If that state file is missing,
rollback will **not** delete pre-existing users or groups. Group membership and
toolchain ACLs can wait; the board is already served by the previous runtime.

## Known limitations

- Linux only. A non-empty mapping on Windows fails closed.
- Sudoers command match is the resolved hermes argv at generation time;
  rebuild the drop-in if `HERMES_BIN` / venv path changes.
- The dispatcher still observes worker PIDs on the gateway host; sudo wraps
  the child so the recorded PID is the sudo process (acceptable for reclaim).
- Shared-board access is group/ACL based. Do not chmod the entire `~/.hermes`
  tree group-readable. Do not put write ACLs on the Hermes root.
