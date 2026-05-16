---
name: osint-investigation
description: Public-records OSINT investigation framework — FEC campaign finance, SEC EDGAR filings, USAspending contracts, Senate lobbying, OFAC sanctions, ICIJ offshore leaks. Entity resolution across sources, cross-link analysis, timing correlation, evidence chains. Python stdlib only.
version: 0.1.0
platforms: [linux, macos, windows]
author: Hermes Agent (adapted from ShinMegamiBoson/OpenPlanter, MIT)
metadata:
  hermes:
    tags: [osint, investigation, public-records, campaign-finance, sec, fec, sanctions, due-diligence, journalism]
    category: research
    related_skills: [domain-intel, arxiv]
---

# OSINT Investigation — Public Records Cross-Reference

Investigative framework for public-records OSINT: campaign finance, government
contracts, corporate filings, lobbying, sanctions, offshore leaks. Resolve entities
across heterogeneous sources, build cross-links with explicit confidence, run
statistical timing tests, and produce structured evidence chains.

**Python stdlib only.** Zero install. Works on Linux, macOS, Windows. No API keys
needed for any Phase-1 source.

Adapted from the MIT-licensed ShinMegamiBoson/OpenPlanter project. Boston/Massachusetts
specifics from the original have been generalized.

## When to use this skill

Use when the user asks for:

- "follow the money" — campaign donations → contract awards, lobbying → legislation
- corporate due diligence — who controls company X, who do they donate to
- sanctions screening — is entity X on OFAC SDN, ICIJ offshore leaks
- pay-to-play investigation — donors who became vendors
- multi-source entity resolution where naming varies (LLC suffixes, abbreviations)
- evidence-chain construction with explicit confidence levels

Do NOT use this skill for:

- general web research → `web_search` / `web_extract`
- domain/infrastructure OSINT → `domain-intel` skill
- academic literature → `arxiv` skill
- social-media profile discovery → `sherlock` skill (optional)

## Workflow

The agent runs scripts via the `terminal` tool. `SKILL_DIR` is the directory
holding this SKILL.md.

### 1. Identify which sources apply

Read the data-source wiki entries to plan the investigation:

```
ls SKILL_DIR/references/sources/
cat SKILL_DIR/references/sources/fec.md
cat SKILL_DIR/references/sources/sec-edgar.md
cat SKILL_DIR/references/sources/usaspending.md
cat SKILL_DIR/references/sources/senate-ld.md
cat SKILL_DIR/references/sources/ofac-sdn.md
cat SKILL_DIR/references/sources/icij-offshore.md
```

Each entry follows a 9-section template: summary, access, schema, coverage,
cross-reference keys, data quality, acquisition, legal, references.

The **cross-reference potential** section maps join keys between sources — read
those first to pick the right pair.

### 2. Acquire data

Each source has a stdlib-only fetch script in `SKILL_DIR/scripts/`:

```bash
# FEC individual contributions (federal campaign finance).
# `--contributor` filters by donor name. Use uppercase 'LAST, FIRST'.
python3 SKILL_DIR/scripts/fetch_fec.py --contributor "SMITH, JOHN" --state NY --cycle 2024 \
    --out data/fec_donations.csv

# SEC EDGAR filings (corporate disclosures)
python3 SKILL_DIR/scripts/fetch_sec_edgar.py --cik 0000320193 \
    --types 10-K,10-Q --out data/edgar_filings.csv

# USAspending federal contracts
python3 SKILL_DIR/scripts/fetch_usaspending.py --recipient "EXAMPLE CORP" \
    --fy 2024 --out data/contracts.csv

# Senate LD-1 / LD-2 lobbying disclosures
python3 SKILL_DIR/scripts/fetch_senate_ld.py --client "EXAMPLE CORP" \
    --year 2024 --out data/lobbying.csv

# OFAC SDN sanctions list (full snapshot)
python3 SKILL_DIR/scripts/fetch_ofac_sdn.py --out data/ofac_sdn.csv

# ICIJ Offshore Leaks — downloads ~70 MB bulk CSV on first use,
# then searches it locally. Cached for 30 days under
# $HERMES_OSINT_CACHE/icij/ (default: ~/.cache/hermes-osint/icij/).
python3 SKILL_DIR/scripts/fetch_icij_offshore.py --entity "EXAMPLE CORP" \
    --out data/icij.csv
```

All outputs are normalized CSV with a header row. Re-run scripts idempotently.

When a private individual won't be in a source (e.g. SEC EDGAR for a non-public-
company person, USAspending for someone who isn't a federal contractor, Senate
LDA for someone who isn't a lobbying client), the script returns 0 rows with a
clear warning rather than silently writing an empty CSV. EDGAR specifically
flags when the company-name resolver matched an individual Form 3/4/5 filer
rather than a corporate registrant.

Rate-limit notes are in each source's wiki entry. Default fetchers sleep
politely between paginated requests. **DEMO_KEY rate limits exhaust quickly** —
real investigations should set the matching env var (`FEC_API_KEY`,
`SENATE_LDA_TOKEN`, etc.). All scripts surface 429 responses immediately with
the upstream's quota message so the user knows to slow down or supply a key.

### 3. Resolve entities across sources

Normalize names and find matches between two CSV files:

```bash
# Match donors (FEC) against contract recipients (USAspending)
python3 SKILL_DIR/scripts/entity_resolution.py \
    --left  data/fec_donations.csv  --left-name-col  contributor_name \
    --right data/contracts.csv      --right-name-col recipient_name \
    --out data/cross_links.csv
```

Three matching tiers with explicit confidence:

| Tier | Method | Confidence |
|------|--------|------------|
| `exact` | Normalized strings equal after suffix/punctuation strip | high |
| `fuzzy` | Sorted-token equality (word-bag match) | medium |
| `token_overlap` | ≥60% token overlap, ≥2 shared tokens, tokens ≥4 chars | low |

Output `cross_links.csv` columns: `match_type, confidence, left_name,
right_name, left_normalized, right_normalized, left_row, right_row`.

### 4. Statistical timing correlation (optional)

Test whether donations cluster suspiciously near contract awards using a
permutation test:

```bash
python3 SKILL_DIR/scripts/timing_analysis.py \
    --donations data/fec_donations.csv --donation-date-col date \
        --donation-amount-col amount --donation-donor-col contributor_name \
        --donation-recipient-col candidate_name \
    --contracts data/contracts.csv --contract-date-col award_date \
        --contract-vendor-col recipient_name \
    --cross-links data/cross_links.csv \
    --permutations 1000 \
    --out data/timing.json
```

Null hypothesis: donation timing is independent of contract dates. One-tailed
p-value = fraction of permutations with mean nearest-award distance ≤ observed.
Minimum 3 donations per (donor, vendor) pair to run the test.

### 5. Build the findings JSON (evidence chain)

```bash
python3 SKILL_DIR/scripts/build_findings.py \
    --cross-links data/cross_links.csv \
    --timing data/timing.json \
    --out data/findings.json
```

Every finding has `id, title, severity, confidence, summary, evidence[], sources[]`.
Each evidence item points back to a specific row in a source CSV. The user (or a
follow-up agent) can verify every claim against its source.

## Confidence and evidence discipline

This is the load-bearing rule of the skill. Tell the user:

- Every claim must trace to a record. No naked assertions.
- Confidence tier travels with the claim. `match_type=fuzzy` is "probable",
  not "confirmed."
- Entity resolution produces candidates, NOT conclusions. A `fuzzy` match
  between "ACME LLC" and "Acme Holdings Group" is a lead, not a fact.
- Statistical significance ≠ wrongdoing. p < 0.05 means the timing pattern
  is unlikely under the null. It does not establish corruption.
- All data sources here are public records. They may still contain
  inaccuracies, stale info, or redactions (GDPR, sealed records).

## Adding a new data source

Use the template:

```bash
cp SKILL_DIR/templates/source-template.md \
    SKILL_DIR/references/sources/<your-source>.md
```

Fill in all 9 sections. Write a `fetch_<source>.py` script in `scripts/` that
uses stdlib only and writes a normalized CSV. Update the source list in the
"When to use" section above.

## Tools and their limits

- `entity_resolution.py` does NOT use external fuzzy libraries (no rapidfuzz,
  no jellyfish). Token-bag matching is the upper bound here. If you need
  Levenshtein, transliteration, or phonetic matching, pip-install separately.
- `timing_analysis.py` uses Python's `random` for permutations. For
  reproducibility, pass `--seed N`.
- `fetch_*.py` scripts use `urllib.request` and respect `Retry-After`. Heavy
  bulk usage may still violate ToS — read each source's legal section first.

## Legal note

All Phase-1 sources are public records. Bulk acquisition is permitted under
their respective access terms (FOIA, public records law, ICIJ explicit
publication, OFAC public data). However:

- Some sources rate-limit aggressively. Respect their headers.
- Some redact registrant info (GDPR on WHOIS, sealed filings).
- Cross-referencing public records to identify private individuals can have
  ethical implications. The skill produces evidence chains, not accusations.
