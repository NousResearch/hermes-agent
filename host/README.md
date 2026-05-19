# Host-side Hermes restart hook

This directory contains Ansible assets for configuring a host machine so Hermes can trigger a **single restricted restart action**.

## Playbook

- `dell.yml` — installs a restricted restart script and a sudoers rule allowing one user to run it without a password.

## Default behavior

The installed script will:
1. `git pull --ff-only` if the Hermes stack directory is a git checkout
2. Restart a Docker Compose stack if it finds a compose file
3. Otherwise restart the systemd services listed in `hermes_systemd_services`

## Host CLI wrapper

If `hermes_install_host_cli_wrapper` is true, the playbook also installs a host-side wrapper that lets you run Hermes CLI commands directly on the host terminal.

- Wrapper path: `/usr/local/bin/hermes-host`
- Optional symlink to `/usr/local/bin/hermes`: controlled by `hermes_install_default_cli_symlink`

The wrapper prefers:
1. `{{ hermes_stack_dir }}/.venv/bin/hermes`
2. `{{ hermes_stack_dir }}/venv/bin/hermes`
3. `python -m hermes_cli.main` from the matching venv

## Variables

Override these when you run the playbook:

- `hermes_stack_dir` — path to the Hermes checkout on the host
- `hermes_restart_user` — user allowed to invoke the restart script via sudoers
- `hermes_systemd_services` — list of systemd units to restart if no compose file is present
- `hermes_compose_files` — filenames to probe for Docker Compose
- `hermes_install_host_cli_wrapper` — install a host-side CLI wrapper
- `hermes_install_default_cli_symlink` — make `/usr/local/bin/hermes` point to the wrapper

## Example

```bash
ansible-playbook -i inventory.ini host/dell.yml \
  -e hermes_stack_dir=/opt/data/hermes-agent \
  -e hermes_restart_user=hermes \
  -e hermes_install_host_cli_wrapper=true
```

## Result

After the playbook runs, the allowed user can execute:

```bash
sudo /usr/local/bin/restart-hermes-stack
/usr/local/bin/hermes-host chat -q "hello"
```

If `hermes_install_default_cli_symlink` is enabled, you can use `hermes` directly on the host CLI.

That host will then perform only the scripted restart workflow.
