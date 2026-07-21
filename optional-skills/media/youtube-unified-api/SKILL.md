---
name: youtube-unified-api
description: "Call YouTube APIs through MyBrandMetrics Discovery."
version: 1.0.0
author: Clawbus; Hermes Agent
license: MIT
platforms: [linux, macos, windows]
required_environment_variables:
  - name: MYBRANDMETRICS_API_KEY
    description: MyBrandMetrics API key for YouTube Discovery requests
metadata:
  hermes:
    tags: [YouTube, Analytics, Reporting, Media, MyBrandMetrics]
    homepage: https://www.clawbus.com/skills/youtube-unified-api
---

# YouTube Unified API Skill

Use this skill to call YouTube Data API v3, YouTube Analytics API v2, and
YouTube Reporting API v1 through the MyBrandMetrics Discovery API. It provides
curl request patterns, endpoint references, schemas, OAuth scope notes, and raw
discovery snapshots for verification or tool generation.

## When to Use

- The user needs to call YouTube Data, Analytics, or Reporting APIs with a
  MyBrandMetrics API key.
- The user asks for exact YouTube endpoint paths, query parameters, request
  bodies, response schemas, uploads, downloads, or OAuth scopes.
- The user wants curl examples for YouTube API calls routed through
  `https://api.mybrandmetrics.com/discovery`.
- The user asks to build endpoint or tool schemas from YouTube discovery data.

## Prerequisites

- `MYBRANDMETRICS_API_KEY` for `X-API-Key` authentication.
- A connected YouTube account in MyBrandMetrics with scopes suitable for the
  requested API.
- Network access to `https://api.mybrandmetrics.com`.

Use `MYBRANDMETRICS_API_BASE_URL` only when the user explicitly needs a
non-default API base URL. Otherwise default to `https://api.mybrandmetrics.com`.

## References

Load references progressively:

- `references/index.md` — reference map and service coverage.
- `references/curl.md` — authentication, account selection, request bodies,
  uploads, downloads, and error handling.
- `references/mybrandmetrics-api.md` — Discovery API routing, account selectors,
  and scope behavior.
- `references/services/youtube-data-v3.md` — YouTube Data API v3 methods and
  schemas.
- `references/services/youtube-analytics-v2.md` — YouTube Analytics API v2
  methods and schemas.
- `references/services/youtube-reporting-v1.md` — YouTube Reporting API v1
  methods and schemas.
- `references/catalog.json` — machine-readable endpoint catalog.
- `references/discovery_cache/*.json` — raw discovery documents for verification
  or regeneration.

## How to Run

Use the `terminal` tool with curl:

```bash
curl -sS "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/channels" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  --url-query "part=snippet,contentDetails,statistics" \
  --url-query "mine=true"
```

Service names:

| Service | Google API | Example route |
| --- | --- | --- |
| `youtube` | YouTube Data API v3 | `/discovery/youtube/channels` |
| `youtubeAnalytics` | YouTube Analytics API v2 | `/discovery/youtubeAnalytics/reports` |
| `youtubeReporting` | YouTube Reporting API v1 | `/discovery/youtubeReporting/jobs` |

## Quick Reference

List the authenticated channel:

```bash
curl -sS "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/channels" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  --url-query "part=snippet,contentDetails,statistics" \
  --url-query "mine=true"
```

Fetch analytics report:

```bash
curl -sS "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtubeAnalytics/reports" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  --url-query "ids=channel==MINE" \
  --url-query "startDate=2026-01-01" \
  --url-query "endDate=2026-01-31" \
  --url-query "metrics=views,estimatedMinutesWatched"
```

Create a playlist with JSON body:

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/playlists" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  --json "@body.json" \
  --url-query "part=snippet,status"
```

Create `body.json` with the `write_file` tool before running the command:

```json
{
  "snippet": {
    "title": "Uploads to Review"
  },
  "status": {
    "privacyStatus": "private"
  }
}
```

Download a reporting media resource:

```bash
curl -sS -L "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtubeReporting/media/${RESOURCE_NAME}" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  --url-query "alt=media" \
  -o report.bin
```

## Procedure

1. Read `references/index.md` to choose the relevant service reference.
2. Read `references/curl.md` for request construction and account selection.
3. Read `references/mybrandmetrics-api.md` when scopes, routing, or account
   selectors matter.
4. Read exactly one service reference under `references/services/` for the
   requested endpoint family.
5. If `/internal/token/access` or a discovery response returns multiple
   accounts, show only `display_name`, `email`, and `oauth_connection_id`; ask
   the user which account to use, then include `oauth_connection_id=<id>`.
6. Use `references/catalog.json` for programmatic endpoint lookup or generated
   tool schemas.
7. Use `references/discovery_cache/*.json` only to verify raw discovery fields
   or regenerate derived references.

## Maintenance

Refresh raw discovery snapshots when Google discovery documents change:

```bash
python3 "${HERMES_HOME:-$HOME/.hermes}/skills/media/youtube-unified-api/scripts/sync_discovery.py"
```

The script fetches official discovery documents and keeps existing cache files
when a network fetch fails.

## Pitfalls

- Do not print or store `access_token` values returned by account selection.
- Prefer `X-API-Key`; the backend may accept `X-API_KEY`, but generated examples
  use `X-API-Key`.
- If curl lacks `--json`, use `-H "Content-Type: application/json"` with
  `--data-binary @body.json`.
- If the API returns `409` with `reauth_required`, the connected YouTube account
  needs new scopes or refresh-token reauthorization.
- Replace path placeholders such as `${VIDEO_ID}` and `${RESOURCE_NAME}` before
  calling endpoints.

## Verification

```bash
curl -sS "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/channels" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  --url-query "part=snippet" \
  --url-query "mine=true"
```
