# no-agent Conversion Recipe — Structured Data Dump Fix

Worked example: `github-radar-merged` cron, 2026-05-27.

## Problem

Cron `github-radar-merged` had `no_agent=false`. The prompt instructed the LLM to process raw JSON repo data and emit a Discord message containing:
- A full `---REPOS---` section with every kept repo
- Inline per-repo descriptions, scores, and URLs
- Raw JSON fragments

Result: ~50KB of message text split across multiple Discord messages, completely violating the cron-output-contract.

## Fix

### 1. Move formatting into the script

Rewrote `github-radar-discover.py` (v4.0) to handle all scoring, classification, and rendering internally.

### 2. Script outputs three artifacts

| Artifact | Destination | Format |
|---|---|---|
| Stdout | Discord message | Plain-text summary + `MEDIA:/path.html` |
| HTML report | `~/.hermes/runbooks/github-radar/YYYY-MM-DD/github-radar-YYYY-MM-DD-HHMM.html` | Dark-theme HTML with full repo detail |
| Structured text | `~/.hermes/runbooks/github-radar/YYYY-MM-DD/github-radar-repos.txt` | `---REPOS---` block for downstream parsing |

### 3. Convert cron to `no_agent=true`

- `script: github-radar-discover.py`
- `no_agent: true`
- `model: null, provider: null`

### 4. Rewrite downstream cron prompts

**Before:** `context_from: github-radar-merged` — raw repo dump injected into LLM context.
**After:** Prompt includes an explicit instruction to read the text file:
```
Read the full repo list from /home/kensei/.hermes/runbooks/github-radar/YYYY-MM-DD/github-radar-repos.txt
```

This removes the dependency on context injection, making the pipeline resilient to large or malformed upstream output.

### 5. Verification

Run script manually:
```bash
cd /home/kensei/.hermes/scripts
python3 github-radar-discover.py
```

Check:
- Stdout is <500 chars of plain text + MEDIA tag.
- HTML file exists and uses dark theme.
- Text file exists and contains structured `---REPOS---` block.
- Exit code is 0.

## Checklist for future conversions

- [ ] Script produces Discord-safe stdout (no Telegram HTML tags — `<b>`, `<code>`, etc.)
- [ ] Script writes date-stamped HTML to an allowlisted directory (`runbooks/`, `reports/`)
- [ ] Script writes machine-parseable structured file for downstream consumers
- [ ] Downstream crons read from file, not raw context injection
- [ ] Cron flagged `no_agent=true`, model/provider cleared
- [ ] `cron-output-lint.py` passes on a test run
