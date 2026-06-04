# Draft.py Flow Comparison

## Original Flow (draft.py)

```
1. Check go/no-go gates (paused, memory, hermes CLI, theme)
2. Load theme prompt
3. Pick top K candidates from DB
4. FOR EACH candidate:
   a. Build prompt with title/URL/summary only
   b. Call hermes to generate draft
   c. Strip markdown code fences
   d. Write draft to file and DB (as-is, no cleaning)
   e. Commit to DB
5. Send digest notification
6. Log run
```

### Problems:
- **No source content**: Drafts based on metadata only
- **Meta-text leakage**: "I don't have the article" appears in drafts
- **No quality checks**: HN links, unsupported claims make it through
- **No style flexibility**: Single theme only

## Pivoted Flow (draft_pivoted.py)

```
1. Check go/no-go gates (paused, memory, hermes CLI, theme)
2. Load theme prompt
3. Pick top K candidates from DB
4. FOR EACH candidate:
   a. *** FETCH ARTICLE CONTENT via hermes web tool ***
      - If fetch fails: skip candidate, mark in DB, continue
   b. Build prompt with FULL SOURCE CONTENT + optional style
   c. Call hermes to generate draft
   d. Strip markdown code fences
   e. *** STRIP META-PREAMBLE from generated text ***
   f. *** RUN QUALITY GATE CHECKS ***
      - If violations found: log warnings (but still save draft)
   g. Write CLEANED draft to file and DB
   h. Commit to DB
5. Send digest notification
6. Log run (including style if used)
```

### Improvements:
- ✅ **Source-grounded**: Fetches actual article before drafting
- ✅ **Meta-text cleaned**: Strips "I don't have..." patterns
- ✅ **Quality-checked**: Detects HN links and other violations
- ✅ **Style-aware**: Optional --style parameter
- ✅ **Skip tracking**: Failed fetches recorded in DB

## Key Function Differences

### build_draft_prompt()

**Before:**
```python
def build_draft_prompt(theme: str, candidate: Candidate) -> str:
    # Uses only title, URL, summary
    # No source content
    # No style support
```

**After:**
```python
def build_draft_prompt(theme: str, candidate: Candidate, 
                       source_content: str, style: str | None = None) -> str:
    # Includes full source content
    # Optional style instruction
    # Explicit grounding requirement
```

### call_hermes()

**Before:**
```python
def call_hermes(prompt: str) -> str:
    # Just generates draft
    # No content fetching
```

**After:**
```python
def call_hermes(prompt: str) -> str:
    # Same, but called AFTER fetch_article_content()
    # Prompt now includes source material

def fetch_article_content(url: str) -> tuple[bool, str]:
    # NEW: Fetches article via hermes web tool
    # Returns (success, content_or_error)
```

### write_draft()

**Before:**
```python
def write_draft(conn, candidate, post_text, version):
    # Writes post_text as-is
    # No cleaning
    # No quality checks
```

**After:**
```python
def write_draft(conn, candidate, post_text, version):
    cleaned_text = strip_meta_preamble(post_text)  # NEW
    gate_result = check_quality_gates(cleaned_text, candidate)  # NEW
    if not gate_result.passed:
        # Log warnings
    # Write CLEANED text
```

### main()

**Before:**
```python
def main(argv):
    parser.add_argument("-k", "--count", ...)
    parser.add_argument("--send-digest", ...)
    # No style support
```

**After:**
```python
def main(argv):
    parser.add_argument("-k", "--count", ...)
    parser.add_argument("--send-digest", ...)
    parser.add_argument("--style", ...)  # NEW
```

## New Functions

### strip_meta_preamble(text: str) -> str
Removes meta-commentary patterns:
- "I don't have..."
- "No browser available..."
- "Based on the title..."
- Content before "---" separator

### fetch_article_content(url: str) -> tuple[bool, str]
Fetches article text via hermes CLI:
- Returns (True, content) on success
- Returns (False, error_reason) on failure
- Validates content length and quality

### load_quality_gates() -> dict
Loads quality gate config from YAML:
- Returns parsed YAML structure
- Gracefully handles missing file

### check_quality_gates(post_text: str, candidate: Candidate) -> QualityGateResult
Runs deterministic checks:
- HN link detection
- Pattern matching from content-social.yaml
- Returns pass/fail + violation list

### skip_candidate(conn, candidate: Candidate, reason: str)
Marks candidate as skipped:
- Updates DB status to 'skipped'
- Stores reason for audit trail

## CLI Usage Comparison

### Original
```bash
# Basic draft generation
python draft.py -k 3

# With digest
python draft.py -k 3 --send-digest
```

### Pivoted
```bash
# Basic draft generation (now with source fetching)
python draft_pivoted.py -k 3

# With style override
python draft_pivoted.py -k 5 --style deadpan_systems_parable

# With digest and style
python draft_pivoted.py -k 3 --send-digest --style deadpan_systems_parable
```

## Database Changes

### New Columns
```sql
-- candidates table
ALTER TABLE candidates ADD COLUMN skip_reason TEXT;
```

### New Data Patterns
- **candidates.status**: Now includes 'skipped' value
- **candidates.skip_reason**: Stores fetch failure reasons
- **run_log.notes**: Includes style parameter when used

## Error Handling Comparison

### Original
- Fetch errors: N/A (no fetching)
- LLM errors: Fatal, stops run
- Quality issues: Undetected

### Pivoted
- Fetch errors: Skip candidate, continue run
- LLM errors: Fatal, stops run (unchanged)
- Quality issues: Log warnings, create draft anyway

## Performance Impact

### Expected Changes
- **Slower**: Each candidate requires article fetch (~5-10s per fetch)
- **More robust**: Failed fetches don't block other candidates
- **Better quality**: Source-grounded drafts should need less manual editing

### Resource Usage
- **Network**: Additional HTTP requests per candidate
- **LLM tokens**: ~2-5x more (due to source content in prompt)
- **Memory**: Slight increase (storing fetched content)

## Migration Checklist

1. [ ] Back up original draft.py
2. [ ] Add skip_reason column to candidates table
3. [ ] Install pyyaml dependency
4. [ ] Test draft_pivoted.py with 1-2 candidates
5. [ ] Verify source fetching works
6. [ ] Verify meta-preamble stripping works
7. [ ] Check quality gates detect violations
8. [ ] Test --style parameter
9. [ ] Run full batch with --send-digest
10. [ ] Replace draft.py with draft_pivoted.py

## Rollback Procedure

If problems occur:
```bash
# Restore backup
cp /home/ubuntu/.hermes/drumbeat/scripts/draft.py.backup-* \
   /home/ubuntu/.hermes/drumbeat/scripts/draft.py

# Reset skipped candidates (optional)
sqlite3 /home/ubuntu/.hermes/drumbeat/drumbeat.db <<SQL
UPDATE candidates 
SET status = 'new', skip_reason = NULL 
WHERE status = 'skipped';
SQL
```
