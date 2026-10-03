---
name: clinical-trials
description: Search clinical trials via the ClinicalTrials.gov API v2.
version: 1.0.0
author: Kewe63
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [clinical-trials, research, medical]
    related_skills: [drug-discovery, bioinformatics, patent-research]
prerequisites:
  commands: [python3]
---

# Clinical Trials Skill

Search clinical trials via the free **ClinicalTrials.gov API v2** — no API key required. Search by condition, drug, sponsor, status, or phase; retrieve trial details (eligibility, locations, outcomes).

## When to Use

- Find active or completed clinical trials for a condition, drug name, or sponsor
- Pull eligibility criteria, primary/secondary outcomes, intervention details
- Cross-check whether a specific NCT ID exists and is currently recruiting
- Build a sponsor portfolio by listing all their trials at a given phase

Prefer `web_search` for general "is drug X being studied" framings; this skill targets the wire-data fit (status, eligibility, outcomes).

## Prerequisites

- Python 3.10+ (stdlib only — uses `urllib`, `argparse`, `json`)
- Outbound HTTPS to `clinicaltrials.gov` (port 443)
- No API key, no signup

## How to Run

Commands run via the bundled helper script. `${HERMES_SKILL_DIR}` is substituted at scan time by the skill loader; copy-paste the resolved path or run from inside the skill directory.

```bash
# Search by condition (paginated, sorted by last update)
python3 "${HERMES_SKILL_DIR}/scripts/clinical_trials.py" search "COVID-19" --limit 5

# Search by drug/intervention
python3 "${HERMES_SKILL_DIR}/scripts/clinical_trials.py" drug "Paxlovid" --status RECRUITING

# Get full trial detail (eligibility, outcomes, locations)
python3 "${HERMES_SKILL_DIR}/scripts/clinical_trials.py" detail NCT05373043

# Search by sponsor with phase filter
python3 "${HERMES_SKILL_DIR}/scripts/clinical_trials.py" sponsor "Pfizer" --phase 3 --limit 10

# Advanced: status + phase combined
python3 "${HERMES_SKILL_DIR}/scripts/clinical_trials.py" search "lung cancer" \
    --status ACTIVE --phase 2
```

## Quick Reference

| Command | Positional | Key flags | Returns |
|---------|-----------|-----------|---------|
| `search` | `<condition>` | `--status`, `--phase`, `--limit` | `{total, results: [...]}` |
| `drug` | `<drug_name>` | `--status`, `--limit` | `{total, results: [...]}` |
| `detail` | `<NCT_ID>` | — | full trial dict |
| `sponsor` | `<sponsor_name>` | `--phase`, `--status`, `--limit` | `{total, results: [...]}` |

`--status` choices: `ACTIVE`, `RECRUITING`, `COMPLETED`, `TERMINATED` (raw API values; the script maps `ACTIVE` → `ACTIVE_NOT_RECRUITING` on the wire).

`--phase` choices: `1`, `2`, `3`, `4`.

Default `--limit`: `10`. Hard ceiling: `100` (ClinicalTrials.gov page size).

## Procedure

1. Identify which query shape fits the user request: condition (general), drug (intervention), sponsor (organization), or detail (NCT ID).
2. Run the matching command; capture the JSON output.
3. For `detail`, surface eligibility criteria, primary outcomes, and the location list — these are the three blocks most often quoted in research summaries.
4. For searches, report total count first, then the truncated `results` array; the script returns at most `--limit` items but `total` reflects the full match count.

## Pitfalls

- The `ACTIVE` status is mapped to `ACTIVE_NOT_RECRUITING` on the wire — the API does not advertise a literal `ACTIVE` value, despite the SKILL.md choice text. If a user complains about an "ACTIVE" filter returning nothing, switch to `RECRUITING` or `ALL`.
- `enrollment` may be `null` (the API only reports counts when the trial has finished enrolling); treat `null` as "unknown", not zero.
- `locations` is empty when the trial has no contact sites published — common for terminated/withdrawn trials.
- The ClinicalTrials.gov API v2 rate-limit is generous but not infinite; avoid running more than ~10 search commands in a 60s window per IP.
- `phase` is a list in the API, not a single string; the script joins entries with `, ` for readability but the source may be `[PHASE2, PHASE3]`.

## Verification

Run the bundled tests (no network required — all HTTP is mocked):

```bash
scripts/run_tests.sh tests/skills/test_clinical_trials_skill.py -q
```

Spot-check a real call before quoting results to the user:

```bash
python3 "${HERMES_SKILL_DIR}/scripts/clinical_trials.py" search "hypertension" --limit 3 | head -40
```

The output should be JSON with `total` and `results[].nct_id`, `title`, `status`, `phase`, `sponsor`.
