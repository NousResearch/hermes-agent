---
name: real-sports-delivery
description: Orchestrate the Real Sports extractor with Hermes — run extraction, validate ai-ready.json freshness, generate delivery-ready picks, and post results to Discord. Treats the extractor repo as the source of truth and keeps Hermes focused on orchestration.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [RealSports, Extraction, Discord, Cron, Orchestration, Automation]
    related_skills: [github-pr-workflow, hermes-agent]
prerequisites:
  commands: [hermes, npm, node, python3, git]
---

# Real Sports Delivery Orchestration

Use this skill when you want Hermes to orchestrate the `real-extractor` repo rather than re-implement the scraper. The extraction repo remains the source of truth for authentication, extraction, normalization, and `ai-ready.json` generation.

## What this skill does

1. Runs or reuses the extractor output in a configured `real-extractor` checkout.
2. Verifies that `data/normalized/<date>/ai-ready.json` exists and is fresh.
3. Invokes Hermes with a focused prompt that tells it to read the extracted JSON, generate delivery-ready picks, and post them through the configured Discord/messaging tools.
4. Supports dry runs so you can validate the workflow without sending anything.
5. Gives you a clean cron-friendly entrypoint for scheduled runs.

## Default configuration

Set the extractor checkout path with one of:

- `REALSPORTS_EXTRACTOR_DIR`
- `--extractor-dir /absolute/path/to/real-extractor`

If neither is provided, the helper script will fail fast and tell you what to set.

## Quick start

```bash
# Validate the workflow without posting
python3 "$SKILL_DIR/scripts/run_real_sports_delivery.py" \
  --extractor-dir ~/src/real-extractor \
  --dry-run

# Run extraction and hand off to Hermes for delivery
python3 "$SKILL_DIR/scripts/run_real_sports_delivery.py" \
  --extractor-dir ~/src/real-extractor
```

## Recommended workflow

### 1) Keep the extractor repo separate

The extractor repo should continue to own:

- browser auth
- site scraping
- normalization
- `picks.json`
- `ai-ready.json`

Do not duplicate that logic in Hermes.

### 2) Let Hermes own orchestration

Hermes should:

- decide whether to re-run extraction
- validate freshness and target date
- read `ai-ready.json`
- generate the delivery summary
- post to Discord when the messaging tool is available
- report failures clearly

### 3) Schedule it with cron

A cron job can attach this skill and call the helper script directly, or ask Hermes to execute the workflow as a scheduled task.

Example pattern:

```bash
hermes cron create "0 8 * * *" \
  "Run the Real Sports delivery workflow from the configured extractor checkout." \
  --skill real-sports-delivery \
  --deliver discord
```

## Pitfalls

- **No `HASS_TOKEN` equivalent here** — this workflow depends on the extractor auth file in the extractor repo and whatever Discord/messaging setup Hermes already has.
- **Freshness matters** — if the `ai-ready.json` timestamp is stale, rerun extraction first.
- **Do not hardcode repo paths** — use `REALSPORTS_EXTRACTOR_DIR` or the script flag.
- **Keep delivery logic out of the extractor repo** — the goal is to keep that repository lean.

## Verification

A good run should confirm:

- the extractor repo exists and has `.auth/realsports-auth.json`
- `npm run extract` succeeds when requested
- `ai-ready.json` exists for the expected target date
- Hermes can read the JSON and produce a structured delivery result
- Discord posting succeeds or fails with a specific, actionable error

## Notes

This skill is intentionally Hermes-side only. It does not replace the extractor. It wraps the extractor so Hermes can manage scheduling, delivery, retries, and future improvements.
