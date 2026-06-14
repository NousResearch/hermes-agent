# X Poster Skill

Draft and publish posts to X/Twitter with strict confirmation gates.

## When to Use

- Drafting short Japanese/English posts for a personal secretary workflow
- Publishing only after explicit user confirmation

## Prerequisites

- `X_API_KEY`, `X_API_SECRET`, `X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET` in `~/.hermes/.env`
- Optional: `X_DRY_RUN=1` forces publish path to stay offline

## How to Run

```bash
py -3 skills/productivity/x-poster/scripts/x_post.py draft --text "hello"
py -3 skills/productivity/x-poster/scripts/x_post.py publish --text "hello" --confirmed
```

## Safety

- `draft` is auto-allowed (local generation only)
- `publish` always requires `--confirmed` unless `--dry-run`
