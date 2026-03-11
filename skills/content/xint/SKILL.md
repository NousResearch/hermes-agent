---
name: xint
description: Analyze X/Twitter accounts and posts using xint or xint-rs. Use for evidence-backed audience, engagement, and topic intelligence before writing.
version: 1.0.0
author: 0xNyk + Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [x, twitter, analytics, research, content-strategy, growth]
    related_skills: [duckduckgo-search, writing-plans]
---

# xint / xint-rs (X Intelligence)

Use this skill to extract factual engagement intelligence from X (Twitter) before writing content.

Primary use cases:
- Account-level performance reconnaissance
- Topic/niche signal discovery
- Viral post pattern mining
- Input generation for article and launchpack workflows

## Tool Choice

Prefer `xint-rs` when available:
- Faster (single Rust binary)
- Good for repeated sweeps and larger sample windows

Fallback to `xint` (Python) when:
- You need parity with existing Python scripts
- Rust binary is unavailable

## Install

### Option A: xint-rs (preferred)

```bash
# Clone and build
cd /tmp
git clone https://github.com/0xNyk/xint-rs.git
cd xint-rs
cargo build --release

# Optional: install globally
install -m 0755 target/release/xint-rs /usr/local/bin/xint-rs
```

Verify:
```bash
xint-rs --help
```

### Option B: xint (Python)

```bash
cd /tmp
git clone https://github.com/0xNyk/xint.git
cd xint
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Verify:
```bash
xint --help
```

## Standard Workflow (Hermes)

1) Define the research target
- account handle(s)
- date window
- post-count/sample window
- success metric (impressions, likes, replies, bookmarks proxy, etc.)

2) Collect raw intelligence with xint/xint-rs
- run account/post analysis
- save raw output to a timestamped artifact file

3) Normalize into structured output
- parse into JSON/CSV table: post_url, timestamp, format, topic, hooks, metrics

4) Derive evidence-backed findings
- top-performing themes
- hook patterns
- format distribution (thread, single post, media-rich)
- recommended angles with explicit metric evidence

5) Feed downstream systems
- provide findings to writing workflows (e.g., x-article-factory)
- preserve source links for traceability

## Example Commands

Note: exact flags may vary by version. Always check `--help` first.

```bash
# Inspect help to confirm available commands/flags
xint-rs --help
xint --help

# Example pattern: analyze an account window
# (replace with current syntax from --help)
xint-rs analyze --handle 0xNyk --limit 200 --since 30d --format json > /tmp/xint-0xNyk.json

# Python variant (replace flags to match installed version)
xint analyze --handle 0xNyk --limit 200 --since 30d --format json > /tmp/xint-0xNyk.json
```

## Hermes Integration Pattern

When invoked for content research, produce this output contract:

1. Data quality check
- command/version used
- sample size
- date range
- any missing/partial fields

2. Top findings (ranked)
- each finding must cite at least 1 post URL and metric values

3. Actionable recommendations
- hooks to reuse
- topic angles to test
- format and posting cadence suggestions

4. Machine-usable artifact
- JSON or CSV path saved locally
- compact markdown summary for human review

## Evidence Gate (Required)

Do NOT output unsupported claims.

Every strategic claim must include:
- source post/account URL
- metric evidence
- timeframe context

If evidence is insufficient, explicitly say:
- "insufficient evidence"
- what is missing
- what additional pull is needed

## Pitfalls

- X data freshness and API constraints can skew recency interpretation.
- Small sample windows overfit to outliers.
- Viral posts are not always transferable across niches.
- Keep raw artifacts so conclusions remain auditable.

## Suggested Artifact Naming

```text
research/xint/<handle>/<YYYY-MM-DD>/raw.json
research/xint/<handle>/<YYYY-MM-DD>/normalized.csv
research/xint/<handle>/<YYYY-MM-DD>/findings.md
```
