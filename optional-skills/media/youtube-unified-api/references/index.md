# YouTube Unified API References

This reference set is generated from the YouTube discovery documents and is intended for curl-based calls through the MyBrandMetrics Discovery API.

## Load Order

1. Read `curl.md` for MyBrandMetrics API key authentication, request construction, uploads, downloads, and error handling.
2. Read `mybrandmetrics-api.md` when using the MyBrandMetrics API key or selecting a connected YouTube account.
3. Read the relevant service file under `services/` for endpoint-specific parameters and curl examples.
4. Use `catalog.json` for programmatic lookup or to build tool schemas.
5. Use `discovery_cache/*.json` only when the generated references need verification against raw discovery data.

## Source Documents

- `youtube`: [YouTube Data API v3](services/youtube-data-v3.md), 83 methods, 199 schemas, source `https://youtube.googleapis.com/$discovery/rest?version=v3`
- `youtubeAnalytics`: [YouTube Analytics API v2](services/youtube-analytics-v2.md), 8 methods, 12 schemas, source `https://youtubeanalytics.googleapis.com/$discovery/rest?version=v2`
- `youtubeReporting`: [YouTube Reporting API v1](services/youtube-reporting-v1.md), 8 methods, 18 schemas, source `https://youtubereporting.googleapis.com/$discovery/rest?version=v1`

## Coverage

- Total methods: 99
- Methods with media upload variants: 8
- Methods supporting media download: 2

## Files

- `curl.md`: shared MyBrandMetrics curl workflow.
- `mybrandmetrics-api.md`: MyBrandMetrics `/discovery` routing and `X-API-Key` notes.
- `services/youtube-data-v3.md`: YouTube Data API v3 methods and schemas.
- `services/youtube-analytics-v2.md`: YouTube Analytics API v2 methods and schemas.
- `services/youtube-reporting-v1.md`: YouTube Reporting API v1 methods and schemas.
- `catalog.json`: compact machine-readable endpoint catalog.
- `discovery_cache/*.json`: raw discovery snapshots.
