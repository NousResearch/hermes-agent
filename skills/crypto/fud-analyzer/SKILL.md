---
name: fud-analyzer
description: Analyze a Twitter/X user's FUD level against crypto projects. Feed it a handle, get a FUD percentage score. Zero dependencies — pure Python stdlib.
version: 1.0.0
author: Deniz Alagoz
license: MIT
metadata:
  hermes:
    tags: [crypto, twitter, social, fud, sentiment, analysis]
    related_skills: []
---

# FUD Analyzer

Measures how much a Twitter/X account FUDs crypto projects.
Uses keyword-weighted sentiment analysis on search engine snippets.
No API keys. No pip install. Pure Python stdlib.

Source: https://github.com/gizdusum/hermes-fud-analyzer

## Quick Start

```bash
# Clone and run
git clone https://github.com/gizdusum/hermes-fud-analyzer
cd hermes-fud-analyzer
python fud_analyzer.py elonmusk
```

Output example:

```
Analyzing @elonmusk for FUD...

  Username:        @elonmusk
  Posts analyzed:  24
  FUD Score:       23.5%
  Shill Score:     76.5%
  Verdict:         MILD FUD — occasional negativity, mostly chill
```

## How It Works

1. Searches DuckDuckGo / Google / Bing for recent posts from the handle
2. Scores each snippet against two keyword banks:
   - FUD keywords: scam, rug, dump, dead, worthless, hack... (40+ terms, weighted)
   - Shill keywords: moon, bullish, gem, 100x, wagmi... (25+ terms)
3. Calculates FUD vs Shill percentage
4. Returns a verdict with the spiciest evidence

## Verdict Scale

| Score   | Label                                     |
|---------|-------------------------------------------|
| 80%+    | MAXIMUM FUD SPREADER — certified chaos agent |
| 60-80%  | HEAVY FUDDER — really hates crypto projects |
| 40-60%  | MODERATE FUDDER — skeptical but not unhinged |
| 20-40%  | MILD FUD — occasional negativity, mostly chill |
| 10-20%  | LOW FUD — pretty balanced actually        |
| 0-10%   | NO FUD — either a shill or just vibing    |

## Use Inside Hermes Agent

```python
from fud_analyzer import analyze_fud
result = analyze_fud("SomeAccount")
print(result["verdict"])
# -> HEAVY FUDDER — this person really hates crypto projects
```

## Pitfalls

- Account must be public and indexed by search engines
- Results depend on what search engines surface (not a direct Twitter API)
- Very new or very small accounts may return no results
- Run again if you get "Could not find posts" — search engines sometimes rate-limit
