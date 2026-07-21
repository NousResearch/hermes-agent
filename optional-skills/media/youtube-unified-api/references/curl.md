# curl Usage

## Authentication

Use a MyBrandMetrics API key in the `X-API-Key` header.

```bash
export MYBRANDMETRICS_API_KEY="YOUR_API_KEY"
export MYBRANDMETRICS_API_BASE_URL="https://api.mybrandmetrics.com"
```

Simple example:

```bash
curl -H "X-API-Key: YOUR_API_KEY" "https://api.mybrandmetrics.com/discovery/youtube/channels?part=snippet,contentDetails,statistics&mine=true"
```

Environment variable example:

```bash
curl -H "X-API-Key: $MYBRANDMETRICS_API_KEY" "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/channels?part=snippet,contentDetails,statistics&mine=true"
```

## Discovery API

Route shape:

```text
/discovery/{service}/{path}
```

Supported service names:

- `youtube`
- `youtubeAnalytics`
- `youtubeReporting`

Examples:

```bash
curl -H "X-API-Key: YOUR_API_KEY" "https://api.mybrandmetrics.com/discovery/youtube/channels?part=snippet,contentDetails,statistics&mine=true"
```

```bash
curl -sS "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtubeAnalytics/reports" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  --url-query "ids=channel==MINE" \
  --url-query "startDate=2026-01-01" \
  --url-query "endDate=2026-01-31" \
  --url-query "metrics=views,estimatedMinutesWatched"
```

## Account Selection

Before calling a discovery endpoint, check which YouTube accounts are available for the API key:

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/internal/token/access" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Content-Type: application/json" \
  --data '{"source_key":"youtube"}'
```

If the response has `tokens` with more than one entry, ask the user which account to use. Show only `display_name`, `email`, and `oauth_connection_id`; do not display or use `access_token`. Then add `oauth_connection_id` to the discovery request:

```bash
curl -H "X-API-Key: YOUR_API_KEY" "https://api.mybrandmetrics.com/discovery/youtube/channels?part=snippet,contentDetails,statistics&mine=true&oauth_connection_id=42"
```

Discovery requests without an account selector may return `409` with `error=account_selection_required` and a `connections` list when multiple accounts are available.

Supported account selector query parameters:

- `oauth_connection_id`
- `google_sub`
- `email`
- `connection_id`
- `account_id`
- `channel_id`

## Query Parameters

Prefer `--url-query` for query strings. It appends parameters to the URL without changing the request body, so it works for GET, POST, PATCH, PUT, and DELETE:

```bash
curl -sS "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/channels" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  --url-query "part=snippet,statistics" \
  --url-query "mine=true"
```

The method references list each parameter's `location`. Path parameters appear inside the URL path. Query parameters should be passed with `--url-query`.

## JSON Bodies

For methods with a request schema, place the JSON request body in `body.json` and call:

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/playlists" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  --json "@body.json" \
  --url-query "part=snippet,status"
```

If the local curl version does not support `--json`, use:

```bash
-H "Content-Type: application/json" --data-binary "@body.json"
```

## Uploads

Methods with `mediaUpload` include a second upload curl example. Set:

```bash
export MEDIA_FILE="video.mp4"
export MIME_TYPE="video/mp4"
```

For multipart uploads, use `-F` and explicitly define the `metadata` content type:

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/upload/youtube/v3/videos?uploadType=multipart&part=snippet,status" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -F "metadata=@metadata.json;type=application/json" \
  -F "media=@$MEDIA_FILE;type=$MIME_TYPE"
```

For resumable uploads, use the upload URL and add:

```bash
-H "X-Upload-Content-Type: $MIME_TYPE"
-H "X-Upload-Content-Length: $(wc -c < "$MEDIA_FILE")"
```

Then upload bytes to the resumable upload session URL when the API returns one.

## Downloads

Methods marked as supporting media download may require `alt=media`. Save binary output with `-o`:

```bash
curl -sS -L "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtubeReporting/media/${RESOURCE_NAME}" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  --url-query "alt=media" \
  -o report.bin
```

Replace path placeholders such as `${VIDEO_ID}` before calling.

## Error Handling

Use `-fS` when scripts should fail on non-2xx responses, or capture both body and status:

```bash
curl -sS -w "\n%{http_code}\n" -o response.json "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/channels" ...
```

Errors generally return a JSON object with an `error` field. If the connected YouTube account needs reauthorization, the API returns `409` with `error=reauth_required`.
