# Testing Checklist for draft_pivoted.py

## Pre-Testing Setup

1. **Database Migration**
   ```sql
   -- Add skip_reason column to candidates table
   ALTER TABLE candidates ADD COLUMN skip_reason TEXT;
   ```

2. **Install Dependencies**
   ```bash
   pip install pyyaml
   ```

3. **Backup Current Implementation**
   ```bash
   cp /home/ubuntu/.hermes/drumbeat/scripts/draft.py \
      /home/ubuntu/.hermes/drumbeat/scripts/draft.py.backup-$(date +%Y%m%d-%H%M%S)
   ```

## Unit Tests

### 1. Meta-Preamble Stripping
Test cases:
- [ ] Text with "I don't have the article" → removed
- [ ] Text with "No browser available" → removed
- [ ] Text with "---" separator → content before removed
- [ ] Text with "Now I have the content" → removed
- [ ] Clean text (no meta) → unchanged
- [ ] Multiple meta patterns → all removed

Example test:
```python
from draft_pivoted import strip_meta_preamble

text = "I don't have the article\n\nHere's my actual post content."
cleaned = strip_meta_preamble(text)
assert "I don't have" not in cleaned
assert "Here's my actual post content" in cleaned
```

### 2. Article Fetching
Test cases:
- [ ] Valid URL with accessible article → returns (True, content)
- [ ] Invalid URL → returns (False, error_reason)
- [ ] Timeout URL → returns (False, "fetch timeout...")
- [ ] Very short content (< 100 chars) → returns (False, "content too short")
- [ ] Very long content → truncated to MAX_SOURCE_CHARS

Mock test (requires mocking subprocess):
```python
from draft_pivoted import fetch_article_content
# Test with known good URL
success, content = fetch_article_content("https://example.com/article")
assert success == True
assert len(content) > 100
```

### 3. Quality Gate Checks
Test cases:
- [ ] Post with HN link → fails "source_grounded" check
- [ ] Post with article URL only → passes
- [ ] Empty post → passes (no violations)
- [ ] Multiple violations → all captured in failures list

Example test:
```python
from draft_pivoted import check_quality_gates, Candidate

candidate = Candidate(...)
post_text = "Check out https://news.ycombinator.com/item?id=12345"
result = check_quality_gates(post_text, candidate)
assert result.passed == False
assert any("HN discussion link" in f for f in result.failures)
```

### 4. Skip Candidate Tracking
Test cases:
- [ ] Candidate skipped with reason → status='skipped' in DB
- [ ] skip_reason stored correctly
- [ ] Skipped candidates excluded from future picks

## Integration Tests

### 5. End-to-End Draft Generation
Test cases:
- [ ] Run with valid candidate → draft created, source fetched
- [ ] Run with unreachable URL → candidate skipped, no draft
- [ ] Run with --style argument → style instruction in prompt
- [ ] Run with multiple candidates → all processed sequentially
- [ ] Quality gate failure → draft still created, warning logged

Test script:
```bash
cd /home/ubuntu/.hermes/drumbeat/scripts

# Test with 1 candidate (safe)
python3 draft_pivoted.py -k 1

# Check draft was created
ls -lah /home/ubuntu/.hermes/drumbeat/drafts/d_*

# Check logs for fetch/gate messages
grep "source fetch" /home/ubuntu/.hermes/drumbeat/logs/*.log
grep "quality gate" /home/ubuntu/.hermes/drumbeat/logs/*.log
```

### 6. Style Parameter
Test cases:
- [ ] No --style → no style instruction in prompt
- [ ] --style deadpan_systems_parable → instruction added
- [ ] Style logged in run_log notes

Test:
```bash
python3 draft_pivoted.py -k 1 --style deadpan_systems_parable
# Check run_log for style note
sqlite3 /home/ubuntu/.hermes/drumbeat/drumbeat.db \
  "SELECT notes FROM run_log WHERE phase='draft' ORDER BY started_at DESC LIMIT 1"
```

### 7. Error Handling
Test cases:
- [ ] Hermes CLI unavailable → graceful error message
- [ ] Quality gates YAML missing → proceeds with empty gates
- [ ] Database error during skip → warning logged, continues
- [ ] Fetch timeout → candidate skipped with timeout reason

## Performance Tests

### 8. Fetch Performance
- [ ] Single article fetch completes within FETCH_TIMEOUT_SECONDS
- [ ] Multiple fetches don't block each other (sequential is OK)
- [ ] Large articles truncated without memory issues

### 9. Memory Usage
- [ ] Processing 10 candidates doesn't exhaust VM memory
- [ ] Fetched content properly released after drafting

## Regression Tests

### 10. Existing Functionality Preserved
- [ ] Draft files still written to DRAFTS_DIR
- [ ] Database inserts work as before
- [ ] Digest notification still works
- [ ] Run log entries created correctly
- [ ] Go/no-go gates still enforced

## Production Readiness

### 11. Logging & Observability
- [ ] Fetch failures logged with context
- [ ] Quality gate failures logged per candidate
- [ ] Skip reasons visible in database
- [ ] Run success/failure tracked in run_log

### 12. Failure Modes
- [ ] Single fetch failure doesn't crash entire run
- [ ] Quality gate failure doesn't block drafting
- [ ] Database errors gracefully handled
- [ ] All errors include candidate ID for debugging

## Acceptance Criteria

Before replacing draft.py in production:
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Successfully drafted 5+ posts with real candidates
- [ ] Verified meta-preamble stripped from at least 2 drafts
- [ ] Confirmed at least 1 candidate skipped due to fetch failure
- [ ] Quality gates detected at least 1 violation (if applicable)
- [ ] Style parameter tested and confirmed in logs
- [ ] No performance degradation vs original draft.py

## Rollback Plan

If issues arise:
```bash
# Restore backup
cp /home/ubuntu/.hermes/drumbeat/scripts/draft.py.backup-YYYYMMDD-HHMMSS \
   /home/ubuntu/.hermes/drumbeat/scripts/draft.py

# Clear any problematic drafts
# (manual review recommended)

# Reset skipped candidates if needed
sqlite3 /home/ubuntu/.hermes/drumbeat/drumbeat.db \
  "UPDATE candidates SET status = 'new', skip_reason = NULL WHERE status = 'skipped'"
```
