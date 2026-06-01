# Obsidian Safe Bridge

Opt-in bridge for local Hermes-to-Obsidian work.

Enable:

```bash
hermes plugins enable obsidian-safe-bridge
```

Set the vault path when needed:

```bash
HERMES_OBSIDIAN_VAULT=/Users/rattanasak/ObsidianVault/HermesAgent
```

Safety model:

- reads allow only safe local vault paths;
- sensitive paths such as `.env`, `.obsidian`, `.git`, and auth files are denied;
- owner-only zones (`90-Owner-Private/` and any `owner-private/` folder) are denied unless `HERMES_OWNER=1`;
- writes are limited to `95-Inbox-Lab/review/`;
- every write attempt is recorded under `$HERMES_HOME/obsidian-safe-bridge/audit.jsonl`.

Owner mode:

```bash
# allow reading owner-private zones (owner only)
HERMES_OWNER=1
```
