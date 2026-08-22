# Crawl4AI Docker Bind Pitfall

## The Problem

The `unclecode/crawl4ai:latest` entrypoint (`/app/entrypoint.sh`) conditionally
binds gunicorn based on the presence of `CRAWL4AI_API_TOKEN`:

```bash
if [[ -n "${CRAWL4AI_API_TOKEN:-}" || "${CRAWL4AI_JWT_ENABLED:-false}" == "true" ]]; then
    GUNICORN_BIND="${GUNICORN_BIND:-[::]:${PORT}}"
else
    GUNICORN_BIND="127.0.0.1:${PORT}"   # unreachable from host!
fi
```

When `CRAWL4AI_API_TOKEN` is not set, gunicorn binds to `127.0.0.1:11235`
inside the container. Docker port mapping only forwards to `0.0.0.0`, so
traffic from the host is silently dropped.

**Logs to look for:** `"binding loopback only (127.0.0.1:11235)"`

## The Fix

### 1. Set CRAWL4AI_API_TOKEN in the container env

In `docker-compose.yml`:

```yaml
services:
  crawl4ai:
    image: unclecode/crawl4ai:latest
    ports:
      - "127.0.0.1:11235:11235"    # localhost-only host exposure
    environment:
      - CRAWL4AI_API_TOKEN=${CRAWL4AI_API_TOKEN}
    env_file:
      - .env
```

### 2. Add Bearer token to the HTTP client

No `crawl4ai` Python dependency needed. Add an optional `api_token` param
to the HTTP client and send it as `Authorization: Bearer *** on every
request:

```python
class Crawl4AIClient:
    def __init__(self, base_url, *, api_token=None, ...):
        self._api_token = api_token

    def _post(self, path, payload):
        headers = {}
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"
```

### 3. Token resolution order

CLI arg > env var > None (no auth fallback):

```python
api_token = args.crawl4ai_token or os.environ.get("CRAWL4AI_API_TOKEN") or None
```

## Verification

After starting the container with the token set, check the logs:

```
docker logs lah-discovery-crawl4ai | grep -E "binding|loopback|Listening"
```

Expected: `"Listening at: http://[::]:11235"` (dual-stack, reachable).

If you see `"binding loopback only"`, the token is missing.

## No-secret logging

The API token is never printed. Only status codes, response shapes, and
timings are captured. The `Authorization` header is sent by the client
but never logged by the application layer.