---
name: osint-investigation
description: Public-records OSINT investigation framework — SEC EDGAR filings, USAspending contracts, Senate lobbying, OFAC sanctions, ICIJ offshore leaks, NYC property records (ACRIS), OpenCorporates registries, CourtListener court records, Wayback Machine archives, Wikipedia + Wikidata, GDELT news monitoring. Entity resolution across sources, cross-link analysis, timing correlation, evidence chains. Python stdlib only.
version: 0.1.0
platforms: [linux, macos, windows]
author: Hermes Agent (adapted from ShinMegamiBoson/OpenPlanter, MIT)
metadata:
  hermes:
    tags: [osint, investigation, public-records, sec, sanctions, corporate-registry, property, courts, due-diligence, journalism]
    category: research
    related_skills: [domain-intel, arxiv]
---

# OSINT Investigation — Public Records Cross-Reference

Investigative framework for public-records OSINT: government contracts,
corporate filings, lobbying, sanctions, offshore leaks, property records,
court records, web archives, knowledge bases, and global news. Resolve
entities across heterogeneous sources, build cross-links with explicit
confidence, run statistical timing tests, and produce structured evidence
chains.

**Python stdlib only.** Zero install. Works on Linux, macOS, Windows. Most
sources work with no API key (OpenCorporates has an optional free token
that raises rate limits).

Adapted from the MIT-licensed ShinMegamiBoson/OpenPlanter project; expanded
to cover identity / property / litigation / archives / news sources that
the original didn't address.

## When to use this skill

Use when the user asks for:

- "follow the money" — government contracts, lobbying → legislation, sanctions
- corporate due diligence — who controls company X, where are they
  incorporated, who serves on their boards, what filings have they made
- sanctions screening — is entity X on OFAC SDN, ICIJ offshore leaks
- pay-to-play investigation — contractors with offshore ties, lobbying
  clients winning awards
- property ownership — find recorded deeds/mortgages by name or address
  (NYC; for other counties point users at the relevant recorder)
- litigation history — find federal + state court opinions and PACER dockets
- multi-source entity resolution where naming varies (LLC suffixes, abbreviations)
- evidence-chain construction with explicit confidence levels
- "what's been said about X" — international news (GDELT) + Wikipedia
  narrative + Wayback Machine to recover dead URLs

Do NOT use this skill for:

- general web research → `web_search` / `web_extract`
- domain/infrastructure OSINT → `domain-intel` skill
- academic literature → `arxiv` skill
- social-media profile discovery → `sherlock` skill (optional)
- US **federal** campaign finance — FEC is intentionally NOT covered here
  (the API is unreliable for ad-hoc contributor-name queries on the free
  DEMO_KEY tier). For federal donations, point users at
  https://www.fec.gov/data/ directly.

## Workflow

The agent runs scripts via the `terminal` tool. `SKILL_DIR` is the directory
holding this SKILL.md.

### 1. Identify which sources apply

Read the data-source wiki entries to plan the investigation:

```
ls SKILL_DIR/references/sources/

# Federal financial / regulatory
cat SKILL_DIR/references/sources/sec-edgar.md       # corporate filings
cat SKILL_DIR/references/sources/usaspending.md     # federal contracts
cat SKILL_DIR/references/sources/senate-ld.md       # lobbying
cat SKILL_DIR/references/sources/ofac-sdn.md        # sanctions
cat SKILL_DIR/references/sources/icij-offshore.md   # offshore leaks

# Identity / property / litigation / archives / news
cat SKILL_DIR/references/sources/nyc-acris.md       # NYC property records
cat SKILL_DIR/references/sources/opencorporates.md  # global corporate registry
cat SKILL_DIR/references/sources/courtlistener.md   # court records (federal + state)
cat SKILL_DIR/references/sources/wayback.md         # Wayback Machine archives
cat SKILL_DIR/references/sources/wikipedia.md       # Wikipedia + Wikidata
cat SKILL_DIR/references/sources/gdelt.md           # global news monitoring
```

Each entry follows a 9-section template: summary, access, schema, coverage,
cross-reference keys, data quality, acquisition, legal, references.

The **cross-reference potential** section maps join keys between sources — read
those first to pick the right pair.

### 2. Acquire data

Each source has a stdlib-only fetch script in `SKILL_DIR/scripts/`:

**Federal financial / regulatory**

```bash
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

**Identity / property / litigation / archives / news**

```bash
# NYC property records (deeds, mortgages, liens) — ACRIS via Socrata
python3 SKILL_DIR/scripts/fetch_nyc_acris.py --name "SMITH, JOHN" \
    --out data/acris.csv
python3 SKILL_DIR/scripts/fetch_nyc_acris.py --address "571 HUDSON" \
    --out data/acris_addr.csv

# OpenCorporates — 130+ jurisdiction corporate registry
# (free token required; set OPENCORPORATES_API_TOKEN or pass --token)
python3 SKILL_DIR/scripts/fetch_opencorporates.py --query "Example Corp" \
    --jurisdiction us_ny --out data/opencorporates.csv

# CourtListener — federal + state court opinions, PACER dockets
python3 SKILL_DIR/scripts/fetch_courtlistener.py --query "Smith v. Example Corp" \
    --type opinions --out data/courts.csv

# Wayback Machine — historical web captures
python3 SKILL_DIR/scripts/fetch_wayback.py --url "example.com" \
    --match host --collapse digest --out data/wayback.csv

# Wikipedia + Wikidata — narrative bio + structured facts
# Set HERMES_OSINT_UA=your-app/1.0 (your@email) to identify yourself
python3 SKILL_DIR/scripts/fetch_wikipedia.py --query "Bill Gates" \
    --out data/wp.csv

# GDELT — global news in 100+ languages, ~2015→present
python3 SKILL_DIR/scripts/fetch_gdelt.py --query '"Example Corp"' \
    --timespan 1y --out data/gdelt.csv
```

All outputs are normalized CSV with a header row. Re-run scripts idempotently.

When a private individual won't be in a source (e.g. SEC EDGAR for a non-public-
company person, USAspending for someone who isn't a federal contractor, Senate
LDA for someone who isn't a lobbying client), the script returns 0 rows with a
clear warning rather than silently writing an empty CSV. EDGAR specifically
flags when the company-name resolver matched an individual Form 3/4/5 filer
rather than a corporate registrant.

Rate-limit notes are in each source's wiki entry. Default fetchers sleep
politely between paginated requests. **API keys raise rate limits** for
sources that support them (`SEC_USER_AGENT`, `SENATE_LDA_TOKEN`,
`OPENCORPORATES_API_TOKEN`, `COURTLISTENER_TOKEN`). All scripts surface
429 responses immediately with the upstream's quota message so the user
knows to slow down or supply a key.

### 3. Resolve entities across sources

Normalize names and find matches between two CSV files:

```bash
# Match lobbying clients (Senate LDA) against contract recipients (USAspending)
python3 SKILL_DIR/scripts/entity_resolution.py \
    --left  data/lobbying.csv   --left-name-col  client_name \
    --right data/contracts.csv  --right-name-col recipient_name \
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

Test whether two time series cluster suspiciously close together — e.g.
lobbying filings near contract awards — using a permutation test:

```bash
python3 SKILL_DIR/scripts/timing_analysis.py \
    --donations data/lobbying.csv --donation-date-col filing_date \
        --donation-amount-col income --donation-donor-col client_name \
        --donation-recipient-col registrant_name \
    --contracts data/contracts.csv --contract-date-col award_date \
        --contract-vendor-col recipient_name \
    --cross-links data/cross_links.csv \
    --permutations 1000 \
    --out data/timing.json
```

The script's column flags are intentionally generic — the original tool was
written for donations vs awards, but it works for any (event, payee) time
series joined through cross-links. Null hypothesis: event timing is
independent of award dates. One-tailed p-value = fraction of permutations
with mean nearest-award distance ≤ observed. Minimum 3 events per (payer,
vendor) pair to run the test.

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
