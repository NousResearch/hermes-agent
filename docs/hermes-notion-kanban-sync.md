# Hermes ↔ Notion Kanban Sync

Production-safe sync for the SoLoVision Notion Task Board and Hermes Kanban.

## Dry run

```bash
python -m hermes_cli.notion_kanban_sync --dry-run
```

The dry run writes a timestamped JSON report with:
- current Notion status distribution
- proposed legacy → canonical status migrations
- proposed Hermes task creations
- proposed two-way updates

## Safe sample/backfill

```bash
python -m hermes_cli.notion_kanban_sync --apply --limit 3
```

For batched permanent backfill without flooding workers:

```bash
python -m hermes_cli.notion_kanban_sync --apply --quiet --max-creates 25
```

No hard deletes are performed. Legacy Notion statuses are not mass-rewritten unless `--status-migration` is explicitly passed after reviewing a dry-run report.

## Permanent timer

Install the operational script and systemd timer:

```bash
mkdir -p ~/.hermes/profiles/dev/scripts ~/.config/systemd/user
install -m 700 scripts/notion_kanban_sync_watchdog.sh ~/.hermes/profiles/dev/scripts/notion_kanban_sync_watchdog.sh
cp plugins/kanban/systemd/hermes-notion-kanban-sync.service ~/.config/systemd/user/
cp plugins/kanban/systemd/hermes-notion-kanban-sync.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now hermes-notion-kanban-sync.timer
systemctl --user list-timers hermes-notion-kanban-sync.timer --no-pager
```

The watchdog runs `python -m hermes_cli.notion_kanban_sync` from `~/.hermes/hermes-agent` by default. Override `HERMES_NOTION_SYNC_REPO`, `HERMES_NOTION_SYNC_PYTHON`, `HERMES_NOTION_SYNC_REPORT_DIR`, or `HERMES_NOTION_SYNC_MAX_CREATES` in the systemd service environment if needed. If `HERMES_NOTION_SYNC_REPORT_DIR` is unset, the module writes to the profile-safe default `~/.hermes/reports/hermes-notion-sync/`.

The watchdog is quiet when no changes occur; non-empty stdout means a sync changed something or hit an error.
