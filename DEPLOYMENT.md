# Drumbeat Pivot Deployment Guide

## Overview
This PR integrates source-grounding, meta-preamble stripping, and quality gates directly into the production Drumbeat draft generator.

## Changes

### Core Features
1. **Source-grounding gate**: Fetches article content before drafting using `hermes -z` with web toolset
2. **Meta-preamble stripping**: Removes "I don't have", "No browser available", etc. from generated text
3. **Quality gates (BLOCKING)**: Runs deterministic checks from content-social.yaml and prevents draft save on failure
4. **Style parameter support**: Optional `--style` argument for theme variations (e.g., `deadpan_systems_parable`)
5. **Skip reason tracking**: Idempotent migration adds `skip_reason` column to candidates table

### Files Changed
- `plugins/drumbeat/scripts/draft.py` - Integrated pivot implementation (replaces sidecar)
- `plugins/drumbeat/scripts/test_pivot.py` - Focused test suite

## Deployment

### Prerequisites
1. Python 3.10+ with existing Drumbeat dependencies
2. Optional: `pip install pyyaml` for quality gate YAML parsing (graceful fallback if missing)
3. Quality gates config at `/home/ubuntu/.hermes/quality-gates/rubrics/content-social.yaml`

### Installation
```bash
# Deploy to production
cp plugins/drumbeat/scripts/draft.py /home/ubuntu/.hermes/drumbeat/scripts/draft.py

# The skip_reason column migration runs automatically on first use (idempotent)
```

### Rollback
```bash
# Restore previous version from git history
cd /home/ubuntu/.hermes/drumbeat
git checkout HEAD~1 scripts/draft.py
```

## Testing

### Run tests
```bash
cd /home/ubuntu/.hermes/hermes-agent/.worktrees/t_0a77347d/plugins/drumbeat/scripts
python3 test_pivot.py
```

### Dry-run draft generation
```bash
# Test with 1 candidate (no digest sent)
cd /home/ubuntu/.hermes/drumbeat/scripts
python3 draft.py -k 1

# Test with style parameter
python3 draft.py -k 1 --style deadpan_systems_parable
```

## Quality Gates

Quality gates now **block** draft creation on failure. Candidates that fail are marked as `skipped` with the failure reason in `skip_reason` column.

Example failures caught:
- HN discussion links (`news.ycombinator.com/item?id=`)
- Meta-preamble patterns in output
- Source content fetch failures (< 100 chars, timeout, etc.)

## Differences from t_308a1c7d Sidecar

| Aspect | t_308a1c7d Sidecar | This PR (Integrated) |
|--------|-------------------|---------------------|
| Implementation | `draft_pivoted.py` separate file | Integrated into production `draft.py` |
| Quality gates | Logged warnings only | **BLOCKING** - prevents save on failure |
| Schema migration | Manual `ALTER TABLE` required | **Idempotent** - runs automatically |
| Dependencies | Required `pip install pyyaml` | **Graceful fallback** if pyyaml missing |
| Documentation | 6 large generated docs | This single focused guide |

## Verification

After deployment, verify:
1. Draft generation still works: `cd /home/ubuntu/.hermes/drumbeat/scripts && python3 draft.py -k 1`
2. Candidates table has `skip_reason` column: `sqlite3 /home/ubuntu/.hermes/drumbeat/drumbeat.db ".schema candidates" | grep skip_reason`
3. Quality gate failures block save (test with a candidate containing HN link)
4. Approval-first behavior preserved (no direct LinkedIn/X posts)

## Risk Assessment

**Low risk** - Changes are additive with fallbacks:
- Schema migration is idempotent (safe to run multiple times)
- pyyaml import wrapped in try/except (continues without quality gates if missing)
- Source fetch failures mark candidate as skipped (doesn't crash the run)
- Quality gate failures skip candidate (doesn't block other candidates)
