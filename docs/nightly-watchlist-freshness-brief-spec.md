# Nightly Watchlist Freshness Brief Spec

## Goal

Give Joe a local-first way to keep investor / AI / developer watchlists from silently going stale. The helper reads a human-maintained Markdown watchlist and an optional JSON state file, then reports which accounts are overdue for review, which are due soon, and which entries are missing tracking state.

## Non-goals

- No network requests.
- No platform login.
- No liking, replying, following, posting, messaging, or other human interaction.
- No mutation of the watchlist or state file.

## Inputs

### Watchlist Markdown

Default: `~/.hermes/memories/INVESTOR_SOCIAL_WATCHLIST.md`.

The parser extracts bullet entries that contain a URL, for example:

```markdown
## Threads accounts
- james93.lin — https://www.threads.com/@james93.lin
- market.sherlock - https://www.threads.com/@market.sherlock
```

It must ignore instructional sections that do not contain URLs.

### State JSON

Optional file containing either:

```json
{
  "items": {
    "https://www.threads.com/@james93.lin": {
      "last_checked": "2026-06-01",
      "cadence_days": 14,
      "priority": "high",
      "notes": "portfolio framework"
    }
  }
}
```

or a flat URL-keyed mapping:

```json
{
  "https://www.threads.com/@james93.lin": {"last_checked": "2026-06-01"}
}
```

Defaults: `cadence_days = 14`, `priority = normal`.

## Output contract

Markdown by default:

- `Overdue` entries: `today - last_checked > cadence_days`
- `Due soon` entries: due within `--soon-days`
- `Missing tracking state` entries: watchlist URLs with no state entry or no valid `last_checked`
- If there are no overdue, due-soon, or missing-state entries, output exactly `[SILENT]`

`--json` emits a machine-readable payload with `overdue`, `due_soon`, `missing_state`, and `summary`.

## CLI

```bash
python scripts/watchlist_freshness_brief.py \
  --watchlist ~/.hermes/memories/INVESTOR_SOCIAL_WATCHLIST.md \
  --state ~/.hermes/state/watchlist_freshness.json \
  --today 2026-06-19 \
  --soon-days 3
```

## Verification plan

- Unit tests for parser behavior, including instructional Markdown sections.
- Unit tests for overdue / due-soon / missing-state classification.
- Unit tests for exact `[SILENT]` behavior when nothing needs attention.
- Smoke check against Joe's local investor watchlist in read-only mode.
