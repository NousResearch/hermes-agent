---
name: source-manager
description: >-
  Maintain the source network a design/technology scouting agent monitors — RSS
  feeds, YouTube channels, Reddit communities, GitHub repos, newsletters,
  changelogs, and competitor sites. Use this whenever the user wants to add,
  remove, prioritize, or evaluate a monitoring source; asks "what should I be
  watching for X"; reports a source has gone stale, sponsored, or low-value; or
  wants to audit the health of an existing watch list. Trigger proactively
  whenever a scouting or trend-research workflow references sources, feeds, or
  "what to follow" rather than waiting for the user to say "source manager."
---

# Source Manager

Read [../../references/operating-model.md](../../references/operating-model.md) first for the shared signal categories, priority scale, and storage conventions this skill's output feeds into.

## Objective
Maintain a focused source network for design, technology, inspiration, and market signals.

## Inputs
- source URL
- platform
- account or publication name
- domain
- topics
- source quality
- desired monitoring frequency
- access method
- notes

## Source Types
Support:

- RSS or Atom feeds
- YouTube channels
- YouTube searches
- Reddit communities
- GitHub repositories
- GitHub searches
- newsletters
- company changelogs
- product launch sites
- design publications
- architecture publications
- social accounts
- forums
- research publications
- conference pages
- product pages
- competitor sites

## Source Record
Store sources using:

```yaml
name:
type:
url:
topics:
priority:
quality:
frequency:
access_method:
last_checked:
last_success:
failure_count:
status:
notes:
```

## Priority Levels
- `critical`: check daily
- `high`: check several times per week
- `normal`: check weekly
- `low`: check monthly
- `event_driven`: check only when triggered

## Access Preference
Use the least fragile method available:

1. official API
2. RSS or Atom
3. public JSON endpoint
4. email or newsletter ingestion
5. search engine index
6. structured extraction
7. third-party scraper
8. browser automation

## Source Evaluation
Rate every source from 0–5 for:

- originality
- technical depth
- visual quality
- timeliness
- relevance
- consistency
- signal-to-noise ratio

Remove or downgrade sources that:
- repeat other publications
- post mostly sponsored material
- produce high volume but low value
- frequently publish unverifiable claims
- rarely generate actionable signals

## Monitoring Logic
When checking a source:
1. fetch only unseen items
2. deduplicate by URL, title, and semantic similarity
3. ignore obvious reposts
4. extract text, metadata, and visuals
5. assign provisional categories
6. send useful items to the relevant skill (typically `signal-scout`)
7. update source health

## Failure Handling
If a source fails:
- retry once
- try an alternate access method
- record the failure
- do not repeatedly hammer the source
- mark as degraded after three failures
- mark as inactive after ten failures
- alert only when a critical source becomes unavailable

## Output
Return:
- new items discovered
- failed sources
- sources to downgrade
- sources to add
- sources with unusual activity
