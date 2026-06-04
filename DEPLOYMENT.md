# Drumbeat Pivot Deployment Guide

## Overview

This deployment replaces the old draft.py with a pivoted version that adds:
- Source-grounding gate (fetches article content before drafting)
- Meta-preamble stripping (removes "I don't have", "No browser available", etc.)
- Quality-gate checks (detects HN links, meta-text violations)
- Optional style support (--style deadpan_systems_parable)

## Pre-Deployment Checklist

### 1. Database Migration

Add `skip_reason` column to candidates table:

```bash
cd /home/ubuntu/.hermes/drumbeat
sqlite3 drumbeat.db "ALTER TABLE candidates ADD COLUMN skip_reason TEXT;"
```

Verify:
```bash
sqlite3 drumbeat.db "PRAGMA table_info(candidates);" | grep skip_reason
```

Expected output: `12|skip_reason|TEXT|0||0`

### 2. Install Dependencies

```bash
pip install pyyaml
```

### 3. Verify Quality Gates Config

```bash
ls -lh /home/ubuntu/.hermes/quality-gates/rubrics/content-social.yaml
```

Should exist and be readable.

### 4. Backup Current Version

```bash
cd /home/ubuntu/.hermes/drumbeat/scripts
cp draft.py draft.py.backup-$(date +%s)
```

## Deployment Steps

### Step 1: Copy Pivoted Version

```bash
cd /home/ubuntu/.hermes/hermes-agent/.worktrees/t_308a1c7d
cp draft_pivoted.py /home/ubuntu/.hermes/drumbeat/scripts/draft.py
```

### Step 2: Verify Syntax

```bash
cd /home/ubuntu/.hermes/drumbeat/scripts
python3 -m py_compile draft.py
```

Should exit 0 with no errors.

### Step 3: Dry Run Test (Single Candidate)

```bash
cd /home/ubuntu/.hermes/drumbeat/scripts
python3 draft.py -k 1
```

Expected output:
- `[run_id] skipping candidate N: source fetch failed: ...` for candidates where fetch fails
- `[run_id] draft d_xxx: text ok, image skipped` for successful drafts

### Step 4: Check Draft Quality

```bash
cd /home/ubuntu/.hermes/drumbeat
ls -lt drafts/ | head -5
```

Read the most recent draft:
```bash
cat drafts/d_*.md | head -50
```

Verify:
- No meta-preambles like "I don't have", "No browser available"
- Content is grounded in actual article (not just title)
- No HN discussion links in body

### Step 5: Check Skipped Candidates

```bash
sqlite3 drumbeat.db "SELECT id, url, skip_reason FROM candidates WHERE status='skipped' ORDER BY fetched_at DESC LIMIT 5;"
```

Should show reasons like "source fetch failed: fetch timeout" or "fetched content too short".

### Step 6: Full Test (3 Candidates)

```bash
python3 draft.py -k 3
```

Review all generated drafts for quality.

### Step 7: Style Test (Optional)

```bash
python3 draft.py -k 1 --style deadpan_systems_parable
```

Check if style instruction was added to prompt (visible in logs if verbose).

## Post-Deployment Verification

### Check Run Logs

```bash
sqlite3 /home/ubuntu/.hermes/drumbeat/drumbeat.db "SELECT run_id, status, notes FROM run_log WHERE phase='draft' ORDER BY started_at DESC LIMIT 10;"
```

Look for:
- `status='ok'` for successful runs
- `notes` showing created drafts and style if used

### Check Draft Database

```bash
sqlite3 drumbeat.db "SELECT id, candidate_id, substr(post_text, 1, 100) FROM drafts ORDER BY generated_at DESC LIMIT 5;"
```

Verify no meta-preambles in post_text preview.

### Monitor Quality Gate Warnings

```bash
grep "quality gate failures" /home/ubuntu/.hermes/drumbeat/logs/drumbeat-*.log | tail -10
```

Quality gate failures are logged but don't block draft creation (manual review still happens).

## Rollback Plan

If issues occur:

```bash
cd /home/ubuntu/.hermes/drumbeat/scripts
mv draft.py draft.py.failed
cp draft.py.backup-TIMESTAMP draft.py
```

Then restart the cron job.

## Configuration Options

### CLI Arguments

- `-k N` or `--count N`: Number of candidates to draft (default: 3)
- `--send-digest`: Send Telegram notification with file paths
- `--style NAME`: Optional style override (e.g., `deadpan_systems_parable`)

### Environment Variables

None required beyond existing Drumbeat setup.

### Quality Gates

Edit `/home/ubuntu/.hermes/quality-gates/rubrics/content-social.yaml` to add/modify checks.

Only deterministic checks (pattern matching) are used. LLM judge checks return "needs_review" in v0.

## Troubleshooting

### Issue: "yaml module not found"

```bash
pip install pyyaml
```

### Issue: "skip_reason column not found"

Run the database migration:
```bash
sqlite3 /home/ubuntu/.hermes/drumbeat/drumbeat.db "ALTER TABLE candidates ADD COLUMN skip_reason TEXT;"
```

### Issue: "hermes CLI not found"

Verify hermes is in PATH:
```bash
which hermes
hermes --version
```

If not, check `/home/ubuntu/.local/bin` is in PATH.

### Issue: All candidates skipped with "fetch timeout"

Check network connectivity:
```bash
curl -I https://simonwillison.net
```

If slow, increase `FETCH_TIMEOUT_SECONDS` in draft.py (default: 120s).

### Issue: Drafts still have meta-preambles

Check the META_PATTERNS list in draft.py. Add new patterns if needed:

```python
META_PATTERNS = [
    r"^I don't have.*?(?:\\n|$)",
    r"^Your new pattern here.*?(?:\\n|$)",
]
```

## Performance Notes

- Source fetching adds ~10-30s per candidate (depends on article load time)
- Expect 1-2 minutes per draft with fetch + generation
- For `-k 3`, total runtime: 3-6 minutes (vs 1-2 minutes without fetching)

## Next Steps

After stable deployment:

1. Monitor rejection rate of generated drafts vs old version
2. Add more deterministic quality gates as patterns emerge
3. Consider enabling LLM judge checks (currently stub)
4. Add optional image generation back if needed
5. Consider auto-approval for drafts that pass all quality gates

## Contact

If issues persist, check:
- `/home/ubuntu/.hermes/drumbeat/logs/drumbeat-*.log`
- `/home/ubuntu/.hermes/logs/agent.log`
- Kanban task t_308a1c7d comments

Or ask in Team Ops Telegram.
