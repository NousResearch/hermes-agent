# Gateway service repair plan (M5 proposal)

> **Proposal only.** No `systemctl restart`, no unit write, no `hermes gateway restart` executed.

## Problem

Live gateway PID was started with:

`hermes_cli.main --profile auto-coder gateway run --replace`

User unit file `~/.config/systemd/user/hermes-gateway-auto-coder.service` has:

`ExecStart=... gateway run` (missing `--replace`)

After an unplanned restart, behavior may differ from the currently running process.

## Goal

Refresh the user unit so **ExecStart matches** the supported Hermes gateway invocation for `auto-coder`, without changing gateway ownership.

## Preconditions

- [ ] Operator confirms single-owner policy (`auto-coder` only).
- [ ] No second gateway profile started.
- [ ] Briefing window: avoid 02:30 / 08:00 cron peaks if possible (optional).

## Proposed steps (approval-gated)

1. **Backup** current unit:
   ```bash
   cp ~/.config/systemd/user/hermes-gateway-auto-coder.service \
      ~/.config/systemd/user/hermes-gateway-auto-coder.service.bak-$(date -u +%Y%m%d)
   ```
2. **Regenerate or edit** unit via Hermes-supported path (preferred):
   ```bash
   HERMES_HOME=~/.hermes/profiles/auto-coder hermes gateway install --help
   # or documented: hermes gateway restart  # when CLI advertises auto-refresh
   ```
   Operator runs the command Hermes docs recommend for “refresh installed unit” (M2b suggested `hermes gateway restart` auto-refreshes — **only after explicit go**).
3. **Reload** user systemd:
   ```bash
   systemctl --user daemon-reload
   ```
4. **Restart** (single owner):
   ```bash
   HERMES_HOME=~/.hermes/profiles/auto-coder hermes gateway restart
   ```
5. **Verify**:
   ```bash
   hermes gateway status
   systemctl --user show hermes-gateway-auto-coder.service -p ExecStart --value
   pgrep -af 'hermes_cli.main.*gateway run'
   ```

## Rollback

```bash
systemctl --user stop hermes-gateway-auto-coder.service
cp ~/.config/systemd/user/hermes-gateway-auto-coder.service.bak-<date> \
   ~/.config/systemd/user/hermes-gateway-auto-coder.service
systemctl --user daemon-reload
systemctl --user start hermes-gateway-auto-coder.service
hermes gateway status
```

## Out of scope (separate approval)

- Migrating gateway to `gateway-minimal`
- Changing MCP server set or profile `HERMES_HOME`
- Merging M1 worktree / reinstalling CLI on PATH

## When to execute

Operator message: **gateway repair go** (or equivalent explicit approval).