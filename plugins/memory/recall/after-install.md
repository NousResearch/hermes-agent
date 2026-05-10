# Hermes Recall installed

Recall is installed as a Hermes memory provider plugin.

## Recommended setup

Run the Hermes memory setup wizard and select `recall`:

```bash
hermes memory setup
```

Then start a fresh Hermes process.

## Non-interactive setup

```bash
hermes config set memory.provider recall
hermes config set plugins.recall.db_path '$HERMES_HOME/recall_memory.sqlite'
hermes config set plugins.recall.auto_capture true
hermes config set plugins.recall.prefetch_enabled true
hermes config set plugins.recall.max_prefetch_results 3
hermes config set plugins.recall.audit_enabled true
```

Verify:

```bash
hermes memory status
hermes chat -q "Use memory_archive_stats and tell me if Recall is active."
```

Recall is a lower-trust searchable archive. Built-in `MEMORY.md` and `USER.md` remain authoritative; promotion into built-in memory requires explicit review and confirmation.
