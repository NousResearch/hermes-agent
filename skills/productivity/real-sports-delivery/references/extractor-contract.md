# Real Sports extractor contract

This skill treats the extractor repo as a black box and relies on a few stable files and behaviors.

## Required files

- `.auth/realsports-auth.json` — extractor auth captured by `npm run auth`
- `data/normalized/<date>/ai-ready.json` — delivery input for Hermes
- `data/normalized/<date>/picks.json` — richer debug artifact for humans

## ai-ready.json fields

The delivery workflow only relies on a small contract:

- `targetDate` — the site day the file belongs to
- `extractedAt` — ISO-8601 timestamp used for freshness checks
- `posts` — the list Hermes should read and turn into a delivery summary

## Fallback compatibility

For local testing, the helper script also accepts a root-level `ai-ready.json` if the normalized directory is absent. When that fallback is used, the helper still validates `targetDate` and `extractedAt` before invoking Hermes.

## Operational notes

- The extractor repo is responsible for auth and extraction.
- Hermes is responsible for orchestration, delivery, and retries.
- The helper script does not try to infer picks from raw site data; it only hands off the extracted JSON to Hermes.
