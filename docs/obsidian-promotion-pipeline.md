# Obsidian Promotion Pipeline

## Purpose

The Obsidian promotion pipeline produces a dry-run plan for selected Hermes
artifacts before anything writes to an Obsidian vault. It is meant for Forge,
memory governance, and future CLI/API callers that need to decide whether a
summary, decision, source excerpt, or artifact is worth promoting.

The implementation lives in `agent/obsidian_promotion.py`.

Primary entry points:

- `plan_obsidian_promotion(candidate, vault_path=None)`
- `write_obsidian_promotion(plan, vault_path=..., approved=True)`

Planning is always safe: it classifies, redacts, detects likely duplicates, and
returns markdown/frontmatter previews. It never writes to Obsidian.

## Candidate Shape

Callers can pass an `ObsidianPromotionCandidate` or a mapping with these fields:

- `title`
- `content`
- `source_type`
- `source_path`
- `source_url`
- `profile`
- `project`
- `tags`
- `created_at`

Source fields are used as provenance in the preview where available.

## Classification Targets

- `RAW_EVIDENCE` -> `Raw/`
- `DEV_SYNTHESIS` -> `Dev/`
- `PROJECT_SUMMARY` -> `Projects/`
- `KNOWLEDGE_NOTE` -> `Knowledge/`
- `MEDIA_ARTIFACT` -> `Media/`
- `REJECT` -> no target and no write

`Raw/` is for source evidence only. `Dev/`, `Knowledge/`, `Projects/`, and
`Media/` are for curated synthesis or finished artifacts.

Forge defaults to `Dev/` for reusable Hermes/system output, unless the content is
source evidence (`Raw/`) or a final project/decision summary (`Projects/`).
The `vault` profile is treated as investment, risk, and strategy work. It does
not imply Obsidian governance ownership and does not map to `90. setting/`.

## Rejection Rules

The planner rejects candidates that look like:

- whole transcripts or complete chat logs
- temporary logs and debug logs
- active `plan.md`, `tasks.md`, or similar task files
- scratch outputs and raw workspace dumps
- full copies of intermediate tool output
- secret-looking material such as API keys, tokens, passwords, credentials, or
  private keys

Secret-looking values are redacted in every preview as `[REDACTED]`.

## Approval Boundary

Obsidian writes require an explicit API call:

```python
plan = plan_obsidian_promotion(candidate, vault_path=tmp_vault)
write_obsidian_promotion(plan, vault_path=tmp_vault, approved=True)
```

Calling `write_obsidian_promotion` without `approved=True` raises
`PermissionError`. Rejected plans cannot be written. Existing notes are not
overwritten.

The write helper only accepts a caller-supplied vault path and a relative target
path from the plan. It refuses absolute targets and `..` traversal outside the
vault.

## Duplicate Detection

When `vault_path` is supplied, planning scans only the target folder in that
vault. It reports duplicate candidates when an existing markdown note has the
same target basename or the same `title:` frontmatter / first heading.

Tests use `tmp_path`; no test writes to the real Obsidian vault.

## Non-Goals

This is not a workflow engine, queue, sync daemon, or live Obsidian integration.
It does not move or delete files, modify the real vault by default, manage
wikilinks, or decide that a duplicate should be merged. Callers must review the
plan and decide whether promotion is appropriate.

## Examples

Forge development synthesis:

```python
candidate = {
    "title": "Hermes Memory Governance Gate",
    "content": "Reusable Hermes development synthesis...",
    "source_type": "dev_synthesis",
    "source_path": "docs/memory-governance-gate.md",
    "profile": "forge",
    "project": "hermes-agent",
    "tags": ["hermes", "memory-governance"],
}
plan = plan_obsidian_promotion(candidate)
assert plan.target_relative_path == "Dev/hermes-memory-governance-gate.md"
```

Raw source evidence:

```python
candidate = {
    "title": "Hermes Source Evidence",
    "content": "Source evidence: targeted tests passed.",
    "source_type": "source_evidence",
    "source_url": "https://example.com/evidence",
    "profile": "forge",
}
plan = plan_obsidian_promotion(candidate, vault_path="/tmp/test-vault")
assert plan.target_relative_path.startswith("Raw/")
```
