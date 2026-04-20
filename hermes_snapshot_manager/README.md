# Hermes Snapshot Manager

Independent snapshot and rollback web app for protecting `~/.hermes` inside Windows WSL.

## MVP scope

- Snapshot the entire `~/.hermes` tree
- Compress each snapshot payload as `files.tar.gz` to reduce disk usage
- Include `.env`, session/state DBs, skills, memories, scripts, and profiles
- Restore the entire package in one action
- Create a pre-restore safeguard snapshot automatically
- Keep a SQLite catalog of snapshots and restore history
- Provide a local Web UI via FastAPI + Jinja templates

## Install

```bash
source venv/bin/activate
pip install -e '.[snapshot-manager]'
```

## Run

```bash
hermes-snapshot-manager
```

Then open `http://127.0.0.1:8876`.

## systemd user service (WSL)

A sample unit file is included at `hermes_snapshot_manager/deploy/hermes-snapshot-manager.service`.

Install it like this:

```bash
mkdir -p ~/.config/systemd/user
cp hermes_snapshot_manager/deploy/hermes-snapshot-manager.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now hermes-snapshot-manager.service
systemctl --user status hermes-snapshot-manager.service
```

The service uses `hermes_snapshot_manager/scripts/run_snapshot_manager.sh` to activate `venv` and launch uvicorn.

## Environment variables

- `HERMES_HOME` — source directory to protect (defaults to `~/.hermes`)
- `HERMES_SNAPSHOT_HOME` — app storage root (defaults to `/mnt/d/hermes_snapshot`, i.e. `D:\hermes_snapshot` on Windows)

## Notes

This first scaffold uses logical snapshots, not filesystem-level WSL/VHDX snapshots. That is intentional: the goal is reliable rollback of Hermes state, not full OS rollback.
