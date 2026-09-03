# Kanban profile-to-Linux-user execution

Optional `kanban.profile_os_users` maps a canonical Hermes profile id to a
POSIX account so dispatcher-spawned workers run as that account instead of
the gateway UID.

This is **off by default**. An empty or missing mapping preserves the existing
trusted-local-user behaviour (workers inherit the gateway UID). Do not enable
the mapping until `hermes kanban os-users check` passes.

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
gets WorkoutTracker; Sys-admin does not).

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

## Shared SQLite board

The default board file is `/home/matt/.hermes/kanban.db` (not under
`~/.hermes/kanban/`). SQLite WAL/SHM sidecars are created next to that file.
`~/.hermes` is typically mode `0700`, so mapped users cannot traverse it or
create sidecars unless ACLs are granted on the **actual DB parent**.

Least privilege:

- Traverse-only (`g:hermes-kanban:--x`, no listdir) on ancestors of the DB parent.
- Write/execute without listdir (`g:hermes-kanban:wx`) on the DB parent so WAL/SHM
  can be created without making the whole Hermes tree readable.
- Default ACL `d:g:hermes-kanban:rw` on the DB parent for new sidecar files.
- `g:hermes-kanban:rw` on `kanban.db` itself.

Do **not** chmod the entire `~/.hermes` tree group-readable.

`hermes kanban os-users check` must prove both mapped users can open the DB in
WAL mode and write sidecars (fail-closed `sudo -n` probe). Being a directory is
not enough.

## Dev workspace ACL

Canonical path: `/home/matt/Documents/WorkoutTracker`.

`/home/matt` is typically mode `0750`, so a leaf-only ACL is unusable. Setup
grants:

- Traverse-only `u:hermes-dev:--x` on ancestors (`/home/matt`, `/home/matt/Documents`).
- Recursive `u:hermes-dev:rwx` on the repo.
- Default ACL `u:hermes-dev:rwx` so newly created files inherit access.

Sys-admin is **not** granted this tree.

## Host setup (Matt, manual sudo)

Dry-run first (no privilege required). This worktree uses `.venv`, not `venv`:

```bash
cd /home/matt/.hermes/worktrees/kanban-profile-os-users
.venv/bin/python -m hermes_cli.main kanban os-users setup \
  --gateway-user matt \
  --dev-workspace /home/matt/Documents/WorkoutTracker

hermes kanban os-users sudoers
hermes kanban os-users rollback    # print-only reverse plan
```

Review the argv list. Then, in a root shell **you** start (this tool never
prompts for a password):

```bash
sudo hermes kanban os-users setup --apply \
  --gateway-user matt \
  --dev-workspace /home/matt/Documents/WorkoutTracker
```

`--apply` requires euid 0. It does **not** copy profile files unless you pass
`--migrate-profile-files` (manual gate). Contents are never printed.

Equivalent manual steps the dry-run prints:

1. Shared group `hermes-kanban` plus private primary groups `hermes-dev` /
   `hermes-sysadmin`.
2. `useradd … -g <private> -G hermes-kanban <user>`.
3. `install -d -m 0700 -o hermes-dev -g hermes-dev` for that user's `.hermes`
   and `profiles/dev` (and the sysadmin equivalents).
4. Least-privilege ACLs on the **actual** DB parent (`~/.hermes`) and on
   `kanban.db` — not only on `~/.hermes/kanban/`.
5. Ancestor traverse + recursive/default ACL on
   `/home/matt/Documents/WorkoutTracker` for `hermes-dev` only.
6. `usermod -aG hermes-kanban matt` so default retains admin visibility.
7. Install `/etc/sudoers.d/hermes-kanban-os-users` after `visudo -c`. The drop-in
   allows `id`, `/usr/bin/test` (audit probes), and the resolved hermes argv.

### Credential migration (manual gate)

`--apply` prints `install(1)` copy commands and **does not execute them** unless
you also pass `--migrate-profile-files`. Never cat/print `.env`, `auth.json`, or
SSH keys.

Required in each mapped `HERMES_HOME` **before check can report ready**:
`config.yaml`, `.env`, `SOUL.md`, `skills/`. Check fails closed if any are
missing.

```bash
# Optional explicit migration (still never prints file contents):
sudo hermes kanban os-users setup --apply \
  --migrate-profile-files \
  --gateway-user matt \
  --dev-workspace /home/matt/Documents/WorkoutTracker

# Or copy yourself with install(1):
sudo install -m 0600 -o hermes-dev -g hermes-dev \
  /home/matt/.hermes/profiles/dev/config.yaml \
  /home/hermes-dev/.hermes/profiles/dev/config.yaml
# Repeat for .env, SOUL.md, skills/. Repeat for sysadmin.
# Skip SSH keys; specialists get their own credentials.
```

Then audit **before** enabling the mapping:

```bash
hermes kanban os-users check
hermes kanban os-users check --json
```

Check must prove, fail-closed via `sudo -n` as each target UID:

- users exist, sudo -n works, homes are 0700 and owned by the mapped uid
- mapped `HERMES_HOME` has `config.yaml`, `.env`, `SOUL.md`, `skills/`
- target users can traverse/write the actual DB parent and complete a SQLite
  WAL lifecycle
- specialist homes are distinct **and** each user is denied read of the other

Only then add `profile_os_users` to the **gateway** profile's `config.yaml` and
restart the gateway yourself. Do not enable it from a Kanban worker.

## Rollback

```bash
hermes kanban os-users rollback    # print-only
# Then as root, after removing profile_os_users from config.yaml:
sudo rm -f /etc/sudoers.d/hermes-kanban-os-users
sudo visudo -c
```

Rollback **always** reverses recorded ACLs (`setfacl -x` / `-k`). It `userdel`s /
`groupdel`s **only** principals this setup created (recorded in
`/var/lib/hermes/kanban-os-users-state.json`). If that state file is missing,
rollback will **not** delete pre-existing users or groups.

## Known limitations

- Linux only. A non-empty mapping on Windows fails closed.
- Sudoers command match is the resolved `hermes` argv at generation time;
  rebuild the drop-in if `HERMES_BIN` / venv path changes.
- The dispatcher still observes worker PIDs on the gateway host; sudo wraps
  the child so the recorded PID is the sudo process (acceptable for reclaim).
- Shared-board access is group/ACL based. Do not chmod the entire `~/.hermes`
  tree group-readable.
