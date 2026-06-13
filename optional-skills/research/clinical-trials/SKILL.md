---
name: clinical-trials
description: Search and analyze clinical trials via ClinicalTrials.gov API v2. Free, no API key required. Search by condition, drug, sponsor, status, phase. Get trial details, eligibility criteria, locations, outcomes.
platforms: [linux, macos, windows]
---

# Clinical Trials Research

Search clinical trials via ClinicalTrials.gov public API. **No API key required.**

## Helper script

This skill includes `scripts/clinical_trials.py` — a complete CLI tool.

```bash
# Search by condition
python3 SKILL_DIR/scripts/clinical_trials.py search "COVID-19" --limit 5

# Search by drug/intervention
python3 SKILL_DIR/scripts/clinical_trials.py drug "Paxlovid" --status RECRUITING

# Get trial details
python3 SKILL_DIR/scripts/clinical_trials.py detail NCT05373043

# Search by sponsor
python3 SKILL_DIR/scripts/clinical_trials.py sponsor "Pfizer" --phase 3 --limit 10

# Advanced search
python3 SKILL_DIR/scripts/clinical_trials.py search "lung cancer" --status ACTIVE --phase 2
```

Commands: search, drug, detail, sponsor. Output is structured JSON.
