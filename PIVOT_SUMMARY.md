# Draft.py Pivot Summary

## Overview
Created pivoted draft.py at `/home/ubuntu/.hermes/hermes-agent/.worktrees/t_308a1c7d/draft_pivoted.py` with source-grounding, meta-preamble stripping, and quality gates.

## Key Changes from Original

### 1. Source-Grounding Gate
**Problem**: Original draft.py generated posts from title/URL/summary only, causing hallucinated content and meta-commentary like "I don't have the article content".

**Solution**: 
- Added `fetch_article_content(url)` function that uses hermes CLI with web tool to fetch actual article text
- Fetching happens BEFORE drafting in the main loop
- If fetch fails, candidate is marked as 'skipped' in DB with reason (no draft created)
- Added `FETCH_TIMEOUT_SECONDS = 120` constant for fetch operations
- Added `MAX_SOURCE_CHARS = 8000` to truncate very long articles

**Implementation**:
```python
fetch_success, content_or_reason = fetch_article_content(candidate.url)
if not fetch_success:
    skip_candidate(conn, candidate, f"source fetch failed: {content_or_reason}")
    continue
```

### 2. Meta-Preamble Stripping
**Problem**: LLM-generated drafts often include meta-commentary like "I don't have the article", "No browser available", "Now I have the content", etc.

**Solution**:
- Added `strip_meta_preamble(text)` function with regex patterns for common meta phrases
- Patterns include:
  - "I don't have...", "I'll draft based on...", "Now I have the content..."
  - "No browser available...", "Based on the title...", "Since I don't have..."
  - "Let me draft...", "Here's a draft...", "I've drafted..."
- Also removes content before "---" separator (if present)
- Applied in `write_draft()` before saving to DB

**Implementation**:
```python
META_PATTERNS = [
    r"^I don't have.*?(?:\n|$)",
    r"^I'll draft based on.*?(?:\n|$)",
    # ... more patterns
]

def strip_meta_preamble(text: str) -> str:
    if "---" in text:
        text = text.split("---", 1)[1].strip()
    for pattern in META_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()
```

### 3. Quality-Gate Checks
**Problem**: No automated quality checks before saving drafts.

**Solution**:
- Added `load_quality_gates()` to read `/home/ubuntu/.hermes/quality-gates/rubrics/content-social.yaml`
- Added `check_quality_gates(post_text, candidate)` that runs deterministic checks
- Returns `QualityGateResult(passed, failures)` dataclass
- Currently checks for:
  - HN discussion links visible as citations (pattern: `news\.ycombinator\.com`)
  - Other deterministic patterns from content-social.yaml
- Failures are logged but don't block draft creation (manual review still happens)

**Implementation**:
```python
@dataclass(frozen=True)
class QualityGateResult:
    passed: bool
    failures: list[str]

def check_quality_gates(post_text: str, candidate: Candidate) -> QualityGateResult:
    gates = load_quality_gates()
    failures = []
    for criterion in gates.get("criteria", []):
        for check in criterion.get("checks", []):
            if check.get("type") == "deterministic":
                pattern = check.get("pattern", "")
                if check.get("fail_if_present") and re.search(pattern, post_text):
                    failures.append(f"[{criterion['id']}] {check['reason']}")
    return QualityGateResult(passed=len(failures) == 0, failures=failures)
```

### 4. Optional Style Support
**Problem**: No way to vary post style beyond the base theme.

**Solution**:
- Added `--style` CLI argument (optional)
- Modified `build_draft_prompt()` to accept `style` parameter
- When style is set, adds instruction to theme prompt: "STYLE OVERRIDE: Apply the '{style}' style variation to this draft."
- Example usage: `python draft_pivoted.py --style deadpan_systems_parable`

**Implementation**:
```python
def build_draft_prompt(theme: str, candidate: Candidate, source_content: str, style: str | None = None) -> str:
    style_instruction = ""
    if style and style.strip():
        style_instruction = f"\n\nSTYLE OVERRIDE: Apply the '{style}' style variation to this draft."
    # ... rest of prompt
```

### 5. Candidate Skip Tracking
**Problem**: No way to track why candidates weren't drafted.

**Solution**:
- Added `skip_candidate(conn, candidate, reason)` function
- Updates candidate status to 'skipped' with reason in DB
- Called when source fetch fails

### 6. Enhanced Error Handling
- Fetch failures are non-fatal (skip candidate, continue to next)
- Quality gate failures are logged but don't block drafts
- All errors include context for debugging

## Database Schema Requirements
The code expects a `skip_reason` column on the candidates table:
```sql
ALTER TABLE candidates ADD COLUMN skip_reason TEXT;
```

## New Dependencies
- `yaml` module for parsing quality-gates YAML

## Function Changes Summary

### Modified Functions
- `build_draft_prompt()`: Added `source_content` and `style` parameters
- `write_draft()`: Now calls `strip_meta_preamble()` and `check_quality_gates()`
- `run()`: Added `style` parameter, calls `fetch_article_content()` before drafting
- `main()`: Added `--style` argument

### New Functions
- `fetch_article_content(url)`: Fetch article text via hermes CLI
- `strip_meta_preamble(text)`: Clean meta-commentary from drafts
- `load_quality_gates()`: Load quality gates YAML config
- `check_quality_gates(post_text, candidate)`: Run deterministic checks
- `skip_candidate(conn, candidate, reason)`: Mark candidate as skipped in DB

## Usage Examples

```bash
# Basic usage (fetches articles, strips meta-text, runs quality gates)
python draft_pivoted.py -k 3

# With style override
python draft_pivoted.py -k 5 --style deadpan_systems_parable

# With digest notification
python draft_pivoted.py -k 3 --send-digest --style deadpan_systems_parable
```

## Testing Checklist
- [ ] Verify article content is fetched before drafting
- [ ] Confirm meta-preamble is stripped from generated posts
- [ ] Check quality gate failures are logged
- [ ] Test skip_candidate() marks candidates correctly
- [ ] Verify --style argument affects prompt
- [ ] Ensure failed fetches don't crash the script
- [ ] Validate candidates table has skip_reason column

## Migration Path
1. Back up current draft.py
2. Add `skip_reason` column to candidates table
3. Install `pyyaml` if not already present: `pip install pyyaml`
4. Test draft_pivoted.py on staging/test candidates first
5. Once validated, replace draft.py with draft_pivoted.py

## Known Limitations
- Source fetching uses hermes CLI which may have rate limits
- Quality gates only run deterministic checks (no LLM judge checks yet)
- Meta-preamble patterns are regex-based and may not catch all variations
- Style parameter just adds instruction to prompt (no dedicated style system)

## Future Enhancements
- Add LLM judge checks from quality-gates YAML
- Implement style templates/prompts instead of just instructions
- Add retry logic for failed article fetches
- Cache fetched article content to avoid re-fetching
- Add metrics/logging for gate failures
