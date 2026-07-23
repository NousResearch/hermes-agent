---
name: bulk-data-enrichment
description: Enrich large datasets (100-1000+ records) via web search waterfall queries in batches. Handles tool-call limits, rate limiting, progress checkpointing, and algorithmic fallbacks when web data is missing.
triggers:
  - enrich a list of companies/prospects/people
  - bulk research on many records
  - waterfall search across a dataset
  - ICP scoring or lead scoring for many records
  - batch web research with progress saving
---

# Bulk Data Enrichment via Web Search Waterfall

## When to use
When you have a JSON/CSV dataset with 50+ records that need enrichment from the web — company info, contact data, sponsorship history, industry classification, scoring, etc.

## Core constraint
`execute_code` has a **50 tool-call limit** per invocation. Each `web_search` call counts as 1 tool call. Plan batches accordingly:
- 3 searches per record → batch of 15-16 records per execute_code
- 2 searches per record → batch of 24 records per execute_code
- 1 search per record → batch of 49 records per execute_code

## Pitfalls

### USE hermes_tools.web_search, NOT direct API calls
The built-in `web_search` from `hermes_tools` works reliably. Do NOT call Brave/Google/SerpAPI directly via `urllib` or `requests` from standalone Python scripts — this consistently fails with HTTP 422 or authentication errors even when API keys are present in `.env`. The hermes_tools layer handles auth, retries, and formatting internally.

### Handle None vs missing in JSON
When iterating records, always use `r.get('field')` and check for both `None` and absence. Some records have `"field": null` vs missing entirely. Use `safe_len()` pattern:
```python
def safe_len(val):
    if val is None: return 0
    if isinstance(val, list): return len(val)
    return 0
```

### Don't let one failed search kill the batch
Wrap each record's enrichment in try/except. A 422, timeout, or parse error on one record should log and continue, not abort the batch.

### Normalize empty strings to null
Before writing enriched data back, normalize `""` → `None` for optional fields to prevent downstream display bugs:
```python
for field in ['email', 'phone', 'contact_person']:
    if record.get(field) == '':
        record[field] = None
```

## Waterfall search strategy

For each record, run searches in order. Move to the next step only if the previous returned nothing useful:

**Step 1 — Primary enrichment** (sponsorship, events, history):
```
"{company} sponsorship college events hackathon India 2024 2025"
```
Look for keywords: sponsor, hackathon, event, summit, partner, college, fest, conference, CSR, innovation. Extract year with regex `r'20(2[0-9])'`. Infer role from context (Partner, Host/Organizer, Sponsor, Platform Provider, CSR Sponsor).

**Step 2 — Contact enrichment** (if no email/contact_person):
```
"{company} marketing manager sponsorship contact email India"
```
Extract emails with `r'[\w.+-]+@[\w-]+\.[\w.-]+'`. Filter out noreply/example.com/test.com. Look for name patterns: `r'([A-Z][a-z]+ [A-Z][a-z]+),?\s+(CEO|CTO|CMO|VP|Director|Manager|Founder|Head)'`.

**Step 3 — Company metadata** (industry, size, location):
```
"{company} company about India <your-city>"
```
Map to categories using keyword lists (see templates/industry-classifier.md pattern below).

## Batching with progress saving

```python
import json, time
from hermes_tools import web_search

DATA_FILE = "/path/to/data.json"

with open(DATA_FILE) as f:
    data = json.load(f)

needs_work = [r for r in data if not r.get('enriched')]

for i, record in enumerate(needs_work[:15]):  # batch of 15
    enrichment = enrich_record(record)  # your waterfall function
    for k, v in enrichment.items():
        if v is not None:
            record[k] = v
    record['enriched'] = True

# Save after EVERY batch (survives interruption)
with open(DATA_FILE, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Batch done. Remaining: {sum(1 for r in data if not r.get('enriched'))}")
```

For resumability across sessions, save a checkpoint:
```python
CHECKPOINT = "/tmp/enrichment-checkpoint.json"
# ... save completed_ids after each batch
```

## Algorithmic fallback scoring

When web search returns nothing useful, don't leave records unscored. Apply algorithmic scoring based on available fields:

```python
def calculate_score(record):
    score = 0; reasons = []
    if record.get('sponsorship_history'):
        score += min(3, len(record['sponsorship_history']))
        reasons.append(f"{len(record['sponsorship_history'])} sponsorships")
    if record.get('local_presence'):
        score += 3; reasons.append("<your-city> presence")
    elif 'india' in (record.get('location') or '').lower():
        score += 2; reasons.append("India-based")
    tech_kw = ['edtech','developer','tech','hackathon','ai','ml','cloud','software','fintech','platform','saas','coding']
    tc = ((record.get('type','') or '') + ' ' + (record.get('industry_vertical','') or '')).lower()
    if any(k in tc for k in tech_kw):
        score += 2; reasons.append("Relevant industry")
    if record.get('company_size_tier') == 'enterprise':
        score += 2; reasons.append("Enterprise")
    elif record.get('company_size_tier') == 'mid-market':
        score += 1; reasons.append("Mid-market")
    return min(10, max(1, score)), '; '.join(reasons[:3])
```

## Industry classification keyword map

```python
INDUSTRY_MAP = [
    (['edtech', 'education', 'learning', 'training', 'academy'], 'EdTech'),
    (['fintech', 'finance', 'payment', 'banking', 'insurance'], 'Fintech'),
    (['healthtech', 'health', 'medical', 'pharma'], 'HealthTech'),
    (['saas', 'software', 'platform', 'devtool', 'api', 'cloud'], 'SaaS'),
    (['ecommerce', 'marketplace', 'retail', 'shopping'], 'E-Commerce'),
    (['ai', 'machine learning', 'genai', 'artificial intelligence'], 'AI/ML'),
    (['blockchain', 'web3', 'crypto'], 'Web3'),
    (['gaming', 'esports'], 'Gaming'),
    (['consulting', 'services', 'outsourcing', 'bpo'], 'IT Services'),
    (['media', 'marketing', 'advertising'], 'Media'),
    (['hackathon', 'developer', 'coding', 'open source'], 'Developer Ecosystem'),
    (['startup', 'incubator', 'accelerator', 'vc'], 'Startup Ecosystem'),
    (['government', 'public sector', 'ministry'], 'Government'),
]

SIZE_MAP = [
    (['unicorn', 'fortune 500', 'multinational', 'global', 'billion'], 'enterprise'),
    (['large', '1000+', '5000+'], 'enterprise'),
    (['mid-size', 'growing', 'series b', 'series c'], 'mid-market'),
    (['startup', 'early stage', 'seed', 'series a'], 'startup'),
]
```

## Targeted queries for known orgs

For well-known companies, generic queries often miss sponsorship data. Use targeted queries:
- `"GeeksforGeeks sponsorship coding event college India"` (not just "GeeksforGeeks India")
- `"Microsoft Imagine Cup sponsorship hackathon college India"`
- `"IBM CSR sponsorship hackathon college India"`

Build a lookup dict of known orgs → targeted search queries for the domain you're enriching.

## Rate limiting

- Add 0.3-0.5s delay between searches within a batch (`time.sleep(0.5)`)
- Add 1-2s delay between batches
- If you hit 422 errors, increase delay. The hermes_tools layer handles some retries, but sustained rapid calls will still fail.
- Process the highest-value records first (sorted by importance) in case you get rate-limited partway through.

## Execution pattern

Run as multiple `execute_code` invocations, each processing one batch:
1. Load data → filter to records needing work → process batch → save → print progress
2. Call execute_code again for next batch
3. Repeat until remaining = 0

Each invocation reads the latest saved state, so interruptions are safe.
