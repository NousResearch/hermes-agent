# Remote Browser Hermes Plugin

This plugin exposes Remote Browser as a Hermes cloud browser provider named
`remote_browser`.

It creates a Remote Browser session, waits until the CDP endpoint is ready,
returns that endpoint to Hermes, and terminates the session during cleanup.

## Enable

```yaml
browser:
  cloud_provider: remote_browser
```

Store the API key in `~/.hermes/.env`:

```bash
REMOTE_BROWSER_API_KEY=rb_...
```

## Optional Config

Non-secret settings belong in `~/.hermes/config.yaml`:

```yaml
browser:
  cloud_provider: remote_browser
  remote_browser:
    base_url: https://brapi.remote-browser.dev
    create_path: /dashboard/remote-browsers
    status_path_template: /dashboard/remote-browsers/{session_id}/status
    terminate_path_template: /dashboard/remote-browsers/{session_id}/terminate
    timeout_minutes: 5
    poll_timeout_seconds: 120
    poll_interval_seconds: 2
    ready_grace_seconds: 10
    resolution: 2560x1440
    region: auto
    recording_enabled: true
    recording_retention_days: 7
    launch_arguments:
      - --disable-dev-shm-usage
    profile_id:
    profile_name:
    proxy_type:
    proxy_url:
```

The provider still accepts the older `REMOTE_BROWSER_*` environment variables
for compatibility with existing installs, but new configuration should use
`browser.remote_browser`.

## API Contract

By default the plugin calls:

```http
POST /dashboard/remote-browsers
Authorization: Bearer <REMOTE_BROWSER_API_KEY>
X-API-Key: <REMOTE_BROWSER_API_KEY>
Content-Type: application/json
```

Accepted response fields:

```json
{
  "id": "rb_xxx",
  "displayId": "rb_xxx",
  "cdpUrl": "wss://brapi.remote-browser.dev/cdp/rb_xxx?token=...",
  "connectUrl": "wss://..."
}
```

The plugin stores `id` or `displayId` as Hermes' `bb_session_id`, returns
`cdpUrl` or `connectUrl` as `cdp_url`, and injects the API key into the CDP URL
as `/api-key/<key>` for the Remote Browser CDP gateway.
