# Rule: [Subsystem Name] — Example Rule File

<!-- provenance: structural pattern from Motion-Granted-Production/.claude/rules/; content is illustrative -->
<!--
  This file demonstrates the rule-file pattern. Copy it for each subsystem that
  needs standing constraints. Name the file descriptively: caching.md, payments.md, etc.

  Scope rules to specific paths so they only load when relevant.
-->

## paths

```
src/your-subsystem/
lib/your-module/
```

## DO NOT

- **Never bypass the subsystem's primary validation layer.** All inputs must pass through `validate()` before processing. No direct DB writes from untrusted input.
- **Never hardcode configuration values** that belong in the config registry (`lib/config/`). Use the registered constant.
- **Never write to the primary table without idempotency.** All writes must be safe to retry. Use upsert or check-before-insert.
- **Never silently swallow errors** in this subsystem. All catch blocks must log the error and return the restrictive default — not the happy-path default.

## Architecture Notes

- Entry point for this subsystem: `{path/to/entrypoint.ts}`
- The source of truth for this subsystem's config: `{path/to/config.ts}`
- All state changes flow through `{path/to/state-manager.ts}` — never write state directly

## Thresholds

| Parameter | Value | Reason |
|-----------|-------|--------|
| Max retry attempts | 3 | Prevents infinite loops on persistent failures |
| Timeout (ms) | 30000 | SLA requirement |

## Key Files

| File | Role |
|------|------|
| `{entrypoint}` | Public API for this subsystem |
| `{config}` | Configuration source of truth |
| `{types}` | Type definitions — edit before changing data shapes |
