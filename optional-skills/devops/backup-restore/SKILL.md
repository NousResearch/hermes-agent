---
name: backup-restore
description: Create dated tarball backups of Hermes configs, profiles, and skills, pushed to git.
version: 1.0.0
author: Vijay Selvaraj (vijays365), Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [devops, backup, restore, git, automation, cron]
    category: devops
    requires_toolsets: [terminal, cronjob]
    related_skills: [config-validator, hermes-diag]
---

# Backup & Restore

Create dated tarball backups of your Hermes setup and push them to a git remote. Works as a one-shot command or a recurring cron job.

## When to Use

- User asks to back up Hermes configuration or data
- Setting up automated backups for a self-hosted Hermes instance
- Before making major config changes (backup first, then modify)
- Recovering from data loss or corruption

## What it backs up

| Source | Path (configurable) |
|--------|---------------------|
| Main config | `$HERMES_HOME/config.yaml` |
| SOUL file | `$HERMES_HOME/SOUL.md` |
| Profiles | `$HERMES_HOME/profiles/` |
| Skills | `$HERMES_HOME/skills/` |

### Excluded automatically

- `*_key.txt`, `*_credentials` (secrets)
- `.env` / `.env.*`
- `__pycache__` / `*.pyc`
- `.git/` directories
- `node_modules/`

## Usage

### Manual run

```bash
bash $HERMES_HOME/skills/devops/backup-restore/scripts/backup.sh
```

### Weekly cron

```bash
cronjob(action='create', script='$HERMES_HOME/skills/devops/backup-restore/scripts/backup.sh', no_agent=True, schedule='0 2 * * 0')
```

## What the script does

1. Creates a timestamped tarball: `backups/hermes-backup-YYYY-MM-DD-HHMMSS.tar.gz`
2. Writes a manifest alongside it (sources, sizes, hostname)
3. Commits to git (`git add`, `git commit`)
4. Pushes to the configured remote
5. Prunes backups older than the retention period (default: 4 weeks)

Tarball paths are relative (portable across systems).

## Configuration

Edit the top of `backup.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `REPO_DIR` | `$HERMES_HOME` | Git repo to push backups to |
| `SOURCES` | config, SOUL, profiles, skills | Paths to include |
| `RETENTION_WEEKS` | 4 | How long to keep old backups |

## Restore procedure

1. List available backups:
   ```bash
   ls $HERMES_HOME/backups/hermes-backup-*.tar.gz
   ```

2. Extract to a temp directory:
   ```bash
   tar xzf $HERMES_HOME/backups/hermes-backup-YYYY-MM-DD-HHMMSS.tar.gz -C /tmp/restore/
   ```

3. Copy files back:
   ```bash
   cp -r /tmp/restore/config.yaml $HERMES_HOME/
   cp -r /tmp/restore/profiles/* $HERMES_HOME/profiles/
   cp -r /tmp/restore/skills/* $HERMES_HOME/skills/
   ```

4. Restore any cron jobs from the manifest.

5. Re-add secrets (API keys, credentials — not backed up).

6. Restart Hermes after config restore.

## Common Pitfalls

1. **Secrets are NOT backed up.** Re-add API keys and credentials after restore.
2. **Git push fails if offline.** The backup is saved locally; push retries on the next run.
3. **Backups older than retention period are auto-pruned.** Adjust `RETENTION_WEEKS` if needed.
4. **Don't restore config while Hermes is running.** Restart Hermes after a config restore.

## Verification Checklist

- [ ] Run the script manually and confirm a tarball is created in `$HERMES_HOME/backups/`
- [ ] Verify tarball content: `tar tzf $HERMES_HOME/backups/hermes-backup-*.tar.gz`
- [ ] Check that the manifest was written alongside the tarball
- [ ] Confirm secrets files are NOT in the tarball
