# gotify-notify

Send high-value, actionable notifications from Hermes Agent to a
[Gotify](https://gotify.net/) server. Messages are rendered as **markdown**
in the Gotify client by default.

## Setup

### 1. Create a Gotify Application

In your Gotify web UI: **Apps** → **Create Application**. Copy the
**Application Token**.

### 2. Configure environment variables

Add the following to `~/.hermes/.env`:

```bash
GOTIFY_URL=http://gotify.local           # Your Gotify server URL
GOTIFY_APP_TOKEN=your_application_token   # Token from step 1
# Optional: override content type (default: text/markdown)
# GOTIFY_CONTENT_TYPE=text/plain
```

### 3. Enable the plugin

The plugin auto-loads when `GOTIFY_URL` and `GOTIFY_APP_TOKEN` are set.
To explicitly enable it in `config.yaml`:

```yaml
plugins:
  enabled:
    - gotify-notify
```

### 4. Restart Hermes

```bash
hermes gateway restart   # gateway
# or exit and relaunch the CLI
```

## Usage

The agent calls `gotify_send` automatically when appropriate — completed
long-running tasks, failures, security findings, approval requests, etc.

You can also trigger it manually in a session:

> Send a gotify notification with title "Test" and message "Hello **world**"

### Markdown rendering

Messages include the `extras.client::display.contentType` field set to
`text/markdown` (Gotify's [markdown display
extension](https://gotify.net/docs/msgextras#clientdisplay)). This means
**bold**, *italic*, `code`, [links](https://gotify.net), lists, and
headings render natively in the Gotify client.

To disable markdown rendering globally, set:

```bash
GOTIFY_CONTENT_TYPE=text/plain
```

## Configuration reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOTIFY_URL` | Yes | — | Gotify server base URL |
| `GOTIFY_APP_TOKEN` | Yes | — | Application token from Gotify |
| `GOTIFY_CONTENT_TYPE` | No | `text/markdown` | Content-Type for message rendering |

## Tool schema

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `message` | string | Yes | — | Notification body (markdown, max 4000 chars) |
| `title` | string | No | `Hermes` | Notification title (max 120 chars) |
| `priority` | integer | No | `5` | Gotify priority (1–10; 2=info, 5=important, 8=urgent, 10=critical) |
