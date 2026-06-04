# Draft.py Pivot - Quick Start Guide

## What Was Created

All files are in: `/home/ubuntu/.hermes/hermes-agent/.worktrees/t_308a1c7d/`

1. **draft_pivoted.py** (22K, 598 lines) - Main implementation
2. **PIVOT_SUMMARY.md** (7.0K) - Detailed change documentation
3. **TESTING_CHECKLIST.md** (5.8K) - Comprehensive test suite
4. **FLOW_COMPARISON.md** (6.6K) - Old vs new flow comparison
5. **QUICK_START.md** (this file) - Integration guide

## What's Different

### 🎯 Source-Grounding
- Fetches actual article content via `hermes` CLI before drafting
- Failed fetches skip candidate with reason in DB (no bad drafts)
- Articles truncated to 8K chars max

### 🧹 Meta-Preamble Stripping  
- Removes "I don't have the article", "No browser available", etc.
- Strips content before "---" separator
- Cleans 8 common meta-commentary patterns

### ✅ Quality Gates
- Runs deterministic checks from `/home/ubuntu/.hermes/quality-gates/rubrics/content-social.yaml`
- Detects HN discussion links in citations
- Logs violations but still creates draft for manual review

### 🎨 Style Support
- Optional `--style` CLI argument
- Adds style instruction to theme prompt
- Example: `--style deadpan_systems_parable`

## Quick Integration (5 minutes)

### Step 1: Database Migration
```bash
sqlite3 /home/ubuntu/.hermes/drumbeat/drumbeat.db <<SQL
ALTER TABLE candidates ADD COLUMN skip_reason TEXT;
.quit
SQL
```

### Step 2: Install Dependencies
```bash
pip install pyyaml
```

### Step 3: Test Run
```bash
cd /home/ubuntu/.hermes/drumbeat/scripts

# Copy pivoted version
cp /home/ubuntu/.hermes/hermes-agent/.worktrees/t_308a1c7d/draft_pivoted.py .

# Test with single candidate
python3 draft_pivoted.py -k 1

# Check output
ls -lah ../drafts/d_* | tail -1
```

### Step 4: Verify Output
```bash
# Check the draft file for:
# 1. No meta-text like "I don't have..."
# 2. Content is substantive (not just title rehash)
# 3. No HN discussion links

cat ../drafts/$(ls -t ../drafts/d_*.md | head -1)
```

### Step 5: Check Logs
```bash
# Look for:
# - "source fetch failed" (if any URLs were unreachable)
# - "quality gate failures" (if any violations detected)

tail -50 ../logs/drumbeat-$(date +%Y-%m-%d).log
```

## Full Production Deployment

### Step 1: Backup
```bash
cd /home/ubuntu/.hermes/drumbeat/scripts
cp draft.py draft.py.backup-$(date +%Y%m%d-%H%M%S)
```

### Step 2: Replace
```bash
cp /home/ubuntu/.hermes/hermes-agent/.worktrees/t_308a1c7d/draft_pivoted.py draft.py
```

### Step 3: Update Cron (if applicable)
```bash
# If using --style in production, update cron entry
# Example: 
# */30 * * * * cd /home/ubuntu/.hermes/drumbeat/scripts && python3 draft.py -k 3 --style deadpan_systems_parable --send-digest
```

### Step 4: Monitor First Run
```bash
# Watch logs in real-time
tail -f /home/ubuntu/.hermes/drumbeat/logs/drumbeat-$(date +%Y-%m-%d).log
```

## Usage Examples

### Basic (same as before, but now with source fetching)
```bash
python3 draft.py -k 3
```

### With Style Override
```bash
python3 draft.py -k 5 --style deadpan_systems_parable
```

### Production Cron Mode
```bash
python3 draft.py -k 3 --send-digest
```

### Dry Run (check candidates without drafting)
```bash
sqlite3 /home/ubuntu/.hermes/drumbeat/drumbeat.db <<SQL
SELECT id, title, url, engagement_velocity 
FROM candidates 
WHERE COALESCE(status, 'new') != 'skipped'
  AND NOT EXISTS (
      SELECT 1 FROM drafts d 
      WHERE d.candidate_id = candidates.id 
        AND d.status IN ('pending', 'approved')
  )
ORDER BY engagement_velocity DESC 
LIMIT 5;
SQL
```

## Troubleshooting

### "hermes CLI not found"
```bash
which hermes
# If missing:
pip install hermes-agent  # or your install method
```

### "Quality gates YAML not found"
```bash
ls -la /home/ubuntu/.hermes/quality-gates/rubrics/content-social.yaml
# If missing, quality gates will be skipped (non-fatal)
```

### "source fetch failed" for all candidates
Check hermes web tool:
```bash
hermes -z "Fetch content from https://example.com" --model gemini-2.5-pro --provider google
```

### Database "skip_reason" column error
```bash
sqlite3 /home/ubuntu/.hermes/drumbeat/drumbeat.db ".schema candidates" | grep skip_reason
# If missing, re-run Step 1 from Quick Integration
```

## Performance Notes

### Expected Slowdown
- **Before**: ~5-10s per candidate (just LLM generation)
- **After**: ~15-25s per candidate (fetch + LLM generation)
- **Reason**: Article fetching adds 5-15s per candidate

### Scaling Guidance
- Small batches (k=3): ~1-2 minutes total
- Medium batches (k=10): ~3-5 minutes total
- Large batches (k=20): ~6-10 minutes total

### Resource Usage
- Network: 1 HTTP request per candidate (via hermes)
- LLM tokens: 2-5x more (full article in prompt)
- Memory: +50-100MB (fetched content buffering)

## Quality Improvements

### Expected Outcomes
- ✅ **Fewer hallucinations** (grounded in source)
- ✅ **No meta-text leaks** (stripped before save)
- ✅ **Fewer HN link violations** (caught by gates)
- ✅ **More substantial posts** (full context vs title only)

### Failure Modes
- 🔴 **Unreachable URLs**: Skipped, logged, no draft
- 🟡 **Paywalled content**: May get partial/summary only
- 🟡 **Very long articles**: Truncated to 8K chars
- 🟢 **Quality violations**: Logged, draft still created

## Rollback Plan

If issues arise:
```bash
cd /home/ubuntu/.hermes/drumbeat/scripts

# Restore backup
cp draft.py.backup-YYYYMMDD-HHMMSS draft.py

# Optional: reset skipped candidates
sqlite3 /home/ubuntu/.hermes/drumbeat/drumbeat.db <<SQL
UPDATE candidates 
SET status = 'new', skip_reason = NULL 
WHERE status = 'skipped';
SQL
```

## Verification Checklist

After integration, verify:
- [ ] Drafts contain substantial content (not just title rehash)
- [ ] No "I don't have..." text in draft files
- [ ] Skipped candidates have reasons in DB
- [ ] Quality gate violations logged (if any)
- [ ] Run completes without crashes
- [ ] Digest notification works (if --send-digest used)

## Next Steps

1. **Run test batch** (k=3) and review drafts manually
2. **Check skip_reason** for any failed fetches
3. **Review quality gate logs** for common violations
4. **Adjust MAX_SOURCE_CHARS** if needed (currently 8000)
5. **Add custom meta-patterns** to META_PATTERNS list if needed
6. **Deploy to production** cron once validated

## Support

If you encounter issues:
1. Check logs: `/home/ubuntu/.hermes/drumbeat/logs/`
2. Review DB: `sqlite3 /home/ubuntu/.hermes/drumbeat/drumbeat.db`
3. Test components individually (see TESTING_CHECKLIST.md)
4. Restore backup if needed (see Rollback Plan above)

## Files Reference

- Implementation: `draft_pivoted.py`
- Documentation: `PIVOT_SUMMARY.md`
- Testing: `TESTING_CHECKLIST.md`
- Comparison: `FLOW_COMPARISON.md`
- This guide: `QUICK_START.md`
