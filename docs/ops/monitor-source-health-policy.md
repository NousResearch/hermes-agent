# Monitor Source-Health Fallback Policy

Monitor jobs must distinguish verified inspection from weak discovery. A fallback source can keep a monitor useful, but it must not let Hermes report unverified candidates as facts.

## Canonical states

Every monitor that uses primary/fallback sources should emit a `HERMES_MONITOR_STATUS:{...}` JSON line containing these fields:

- `source_access_ok`: true when at least one source was reached and parsed; false when no source was accessible.
- `fallback_used`: true when a fallback source was accessed after a primary source was attempted.
- `confidence`: `high`, `medium`, `low`, or `none`, based on the strongest accessible source.
- `delivery_allowed`: true only when at least one accessible non-discovery source is verified/single-source and medium-or-higher confidence.
- `delivery_suppressed`: true when the run found candidates but policy blocks delivery.
- `suppression_reason`: one of `source_access_failed`, `discovery_only_source`, `unverified_source`, `low_confidence_source`, or `no_sources_attempted`.
- `failure_reason`: concrete extraction/access failure text when access failed or delivery was suppressed.
- `sources`: per-source evidence records with `name`, `access_ok`, `verification`, `confidence`, `fallback`, and optional counts/notes.

## Delivery policy

Delivery is allowed only when there is at least one accessible source with:

1. `verification` = `verified` or `single_source`; and
2. `confidence` = `medium` or `high`; and
3. the source is not `discovery_only`.

Delivery is suppressed when:

- no source was accessible;
- all accessible sources are discovery-only;
- all accessible sources are unverified;
- the best accessible source is low-confidence.

This is intentionally conservative: web search, Firecrawl snippets, Reddit scraping, and other brittle fallbacks may discover leads, but they should not trigger user-facing monitor alerts unless the monitor verifies the item against a stronger source.

## Implementation

Use `cron.source_health` from monitor scripts:

```python
from cron.source_health import SourceObservation, monitor_status_line

observations = [
    SourceObservation(
        name="reddit-json",
        access_ok=False,
        verification="verified",
        confidence="high",
        failure_reason="HTTP 403",
    ),
    SourceObservation(
        name="web-search-fallback",
        access_ok=True,
        verification="discovery_only",
        confidence="low",
        fallback=True,
        items_seen=4,
    ),
]

print(monitor_status_line(observations))
```

The emitted status line is parsed by cron delivery. If `delivery_allowed` is false or `delivery_suppressed` is true, the scheduler forces the successful run's user-facing delivery to `[SILENT]` while still saving the full output artifact for audit. Monitor scripts should still prefer returning `[SILENT]` themselves when they know the source is weak; the scheduler-side gate is the backstop that prevents discovery-only or unverified fallback candidates from becoming alerts.
