# Obsidian Canonical Pipeline

Canonical ingestion route for notes and memory synchronization:

1. Source events/messages enter Hermes tools.
2. Durable user/profile facts go to Hermes memory stores.
3. Structured long-form artifacts are written to Obsidian via Local REST API.
4. Obsidian remains the canonical human-readable archive; Hermes memory remains concise operational recall.

Operational defaults:
- Endpoint: `https://127.0.0.1:27124` (Obsidian Local REST API).
- Auth key location: `<vault>/.obsidian/plugins/obsidian-local-rest-api/data.json`.
- Markdown writes must set `Content-Type: text/markdown`.

Canonical script entrypoints:
- Sync pipeline: `scripts/ops/obsidian_canonical_sync.py`
- Healthcheck/recovery: `scripts/ops/obsidian_pipeline_healthcheck.py`

Recommended flow for automated jobs:
- Build final markdown report in temp content.
- Write/update deterministic note path (`Inbox/`, `Reports/`, etc.).
- Use bookmark-ID ledger (`~/.hermes/state/obsidian_bookmark_ledger.json`) for idempotent re-runs.
- Store only stable routing facts in Hermes memory, not full report contents.

Failure handling:
- If REST API unavailable, queue local fallback artifact and retry on next maintenance cycle.
- Only escalate to user if retries are exhausted and report is critical.
