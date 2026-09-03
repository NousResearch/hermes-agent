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

## Host setup (Matt, manual sudo)

Dry-run first (no privilege required):

```bash
# From a Hermes install that contains this feature (worktree or later release).
cd /home/matt/.hermes/worktrees/kanban-profile-os-users   # or the installed tree
./venv/bin/python -m hermes_cli.main kanban os-users setup \
  --gateway-user matt \
  --dev-workspace /home/matt/WorkoutTracker

hermes kanban os-users sudoers
hermes kanban os-users rollback    # print-only reverse plan
```

Review the argv list. Then, in a root shell **you** start (this tool never
prompts for a password):

```bash
sudo hermes kanban os-users setup --apply \
  --gateway-user matt \
  --dev-workspace /home/matt/WorkoutTracker
```

`--apply` requires euid 0. Equivalent manual steps the dry-run prints:

1. `groupadd --system hermes-kanban`
2. `useradd --system --create-home --home-dir /home/hermes-dev --shell /usr/sbin/nologin --gid hermes-kanban hermes-dev`
3. Same for `hermes-sysadmin`.
4. `install -d -m 0700 -o hermes-dev -g hermes-dev /home/hermes-dev/.hermes` and
   `/home/hermes-dev/.hermes/profiles/dev` (and the sysadmin equivalents).
5. Group + default ACLs on the shared board dir so SQLite can create `-wal`/`-shm`
   siblings: `setfacl -m g:hermes-kanban:rwx,d:g:hermes-kanban:rwx <kanban-dir>`.
6. Narrow ACL on WorkoutTracker (and any other Dev-only tree) for `hermes-dev`
   only — do **not** grant `hermes-sysadmin`.
7. `usermod -aG hermes-kanban matt` so default retains admin visibility.
8. Install `/etc/sudoers.d/hermes-kanban-os-users` after `visudo -c`.

Credential migration uses `install(1)` and **never prints file contents**:

```bash
sudo install -m 0600 -o hermes-dev -g hermes-dev \
  /home/matt/.hermes/profiles/dev/config.yaml \
  /home/hermes-dev/.hermes/profiles/dev/config.yaml
sudo install -m 0600 -o hermes-dev -g hermes-dev \
  /home/matt/.hermes/profiles/dev/.env \
  /home/hermes-dev/.hermes/profiles/dev/.env
# Repeat for sysadmin. Skip SSH keys; specialists get their own credentials.
```

Then audit **before** enabling the mapping:

```bash
hermes kanban os-users check
hermes kanban os-users check --json
```

Check must prove: users exist, sudo -n works, homes are 0700 and owned by the
mapped uid, specialist homes are distinct, shared board is present. Only then
add `profile_os_users` to the **gateway** profile's `config.yaml` and restart
the gateway yourself. Do not enable it from a Kanban worker.

## Rollback

```bash
hermes kanban os-users rollback    # print-only
# Then as root, after removing profile_os_users from config.yaml:
sudo rm -f /etc/sudoers.d/hermes-kanban-os-users
sudo visudo -c
sudo userdel -r hermes-dev
sudo userdel -r hermes-sysadmin
sudo groupdel hermes-kanban
```

Remove ACLs on WorkoutTracker / board paths if you added them.

## Known limitations

- Linux only. A non-empty mapping on Windows fails closed.
- Sudoers command match is the resolved `hermes` argv at generation time;
  rebuild the drop-in if `HERMES_BIN` / venv path changes.
- The dispatcher still observes worker PIDs on the gateway host; sudo wraps
  the child so the recorded PID is the sudo process (acceptable for reclaim).
- Shared-board access is group/ACL based. Do not chmod the entire `~/.hermes`
  tree group-readable.
