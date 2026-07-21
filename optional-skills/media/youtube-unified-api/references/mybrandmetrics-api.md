# MyBrandMetrics API

## Purpose

Use the MyBrandMetrics Discovery API when the user provides a MyBrandMetrics API key and wants to call YouTube Data, Analytics, or Reporting endpoints.

## Base URL and Header

```bash
export MYBRANDMETRICS_API_BASE_URL="https://api.mybrandmetrics.com"
export MYBRANDMETRICS_API_KEY="YOUR_API_KEY"
```

Send:

```bash
-H "X-API-Key: $MYBRANDMETRICS_API_KEY"
```

The backend also accepts `X-API_KEY`, but prefer `X-API-Key`.

## Route Mapping

Route:

```text
/discovery/{service}/{path}
```

Services:

| Service | Google API | Common prefix omitted in generated examples |
| --- | --- | --- |
| `youtube` | YouTube Data API v3 | `youtube/v3/` |
| `youtubeAnalytics` | YouTube Analytics API v2 | `v2/` |
| `youtubeReporting` | YouTube Reporting API v1 | `v1/` |

Examples:

```text
/discovery/youtube/channels
/discovery/youtube/videos
/discovery/youtubeAnalytics/reports
/discovery/youtubeReporting/jobs
```

The backend can also match the full discovery path when needed, but generated examples use the shorter path.

## Account Selection

Before calling a discovery endpoint, check connected YouTube accounts:

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/internal/token/access" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Content-Type: application/json" \
  --data '{"source_key":"youtube"}'
```

Response shape:

```json
{
  "source_key": "youtube",
  "token_type": "Bearer",
  "access_token": "...",
  "expires_in": 3600,
  "scope": "...",
  "tokens": [
    {
      "source_key": "youtube",
      "token_type": "Bearer",
      "access_token": "...",
      "expires_in": 3600,
      "scope": "...",
      "oauth_connection_id": "42",
      "google_sub": "...",
      "email": "a@example.com",
      "display_name": "Account A"
    }
  ]
}
```

If `tokens` contains more than one entry, ask the user which account to use. Present only `display_name`, `email`, and `oauth_connection_id`; do not display or use `access_token`. Use the selected `oauth_connection_id` in the discovery request:

```bash
curl -H "X-API-Key: YOUR_API_KEY" "https://api.mybrandmetrics.com/discovery/youtube/channels?part=snippet,contentDetails,statistics&mine=true&oauth_connection_id=42"
```

Discovery requests without an account selector may return:

```json
{
  "error": "account_selection_required",
  "source_key": "youtube",
  "connections": [
    {
      "oauth_connection_id": "42",
      "email": "a@example.com",
      "display_name": "Account A",
      "label": "Account A (a@example.com)"
    }
  ]
}
```

Supported selector query parameters:

```text
oauth_connection_id
google_sub
email
connection_id
account_id
channel_id
```

## Scope Behavior

The target user must have a connected YouTube account with a suitable scope:

- Data API: any of `youtube.readonly`, `youtube`, `youtube.force-ssl`, `youtube.upload`.
- Analytics API: any of `yt-analytics.readonly`, `yt-analytics-monetary.readonly`.
- Reporting API: any of `yt-analytics.readonly`, `yt-analytics-monetary.readonly`.

If scopes or refresh tokens are missing, the API returns `409` with `error=reauth_required`.

## Headers

Use `X-API-Key` for discovery calls.
