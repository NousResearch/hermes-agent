# Obsidian Architecture Stance

Decision: keep Obsidian integration as a script layer with one maintained canonical entrypoint,
not a first-class Hermes core plugin (for now).

Why:
- Lowest operational risk for user-specific vault paths, keys, and note layout.
- Keeps Hermes core tool surface minimal while still enabling deterministic automation.
- Easier to swap/iterate note formatting and routing without touching gateway/runtime internals.

Canonical entrypoints:
- Sync: `scripts/ops/obsidian_canonical_sync.py`
- Healthcheck/recovery: `scripts/ops/obsidian_pipeline_healthcheck.py`

Operational contract:
- No hardcoded absolute vault paths in scripts or cron prompts.
- Use env/args (`OBSIDIAN_VAULT`, `OBSIDIAN_REST_URL`, `OBSIDIAN_REST_KEY_FILE`, `OBSIDIAN_API_KEY`).
- Bookmark-ID ledger (`~/.hermes/state/obsidian_bookmark_ledger.json`) is the idempotency source.

Revisit trigger:
- Promote to first-class tool/plugin only if 3+ independent automations require shared runtime semantics
  that cannot be safely handled by script entrypoints.
