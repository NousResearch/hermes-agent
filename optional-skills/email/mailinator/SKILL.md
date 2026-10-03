---
name: mailinator
description: "Fetch and inspect emails from public Mailinator inboxes."
version: 1.0.0
author: Sam Lipton
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [email, mailinator, inbox, api, testing]
    homepage: https://www.mailinator.com/
---

# Mailinator - Free, disposable email

## Requirements

- None. No API key required. Agent needs access to 'curl' or similar
- [Optional] Alternatively, install the Mailinator CLI for mcp access - npm install -g mailinator-cli

## Quick Start

```bash
# List emails in a public inbox
curl -s "https://www.mailinator.com/api/v2/domains/public/inboxes/{inbox}"

# Get a specific email's content (with raw body)
curl -s "https://www.mailinator.com/api/v2/domains/public/inboxes/{inbox}/messages/{message_id}?format=raw"

# Get a specific email's HTML content
curl -s "https://www.mailinator.com/api/v2/domains/public/inboxes/{inbox}/messages/{message_id}/html"
```

## API Endpoints

### List Inbox Messages

```
GET https://www.mailinator.com/api/v2/domains/public/inboxes/{inbox}
```

Example:
```bash
curl -s "https://www.mailinator.com/api/v2/domains/public/inboxes/joe" | python3 -m json.tool
```

Response includes:
- `msgs`: Array of email metadata (`id`, `from`, `fromfull`, `origfrom`, `subject`, `to`, `time`, `seconds_ago`, `ip`)
- `domain`: The domain used (public)
- `to`: The inbox name

> **Endpoint status:** The `domains/public/...` path above is verified working
> (HTTP 200 with a live `msgs` array), but Mailinator's official API overview
> (https://www.mailinator.com/mailinator-api/) documents the token-authenticated
> **private**-inbox workflow rather than this public path. Treat the public
> endpoint as a convenient, verified public interface and the token/private
> workflow (see "Public vs Private Domains" below) as the authoritative,
> upstream-documented contract for anything beyond throwaway public inboxes.

### Get Email Content

```
GET https://www.mailinator.com/api/v2/domains/public/inboxes/{inbox}/messages/{message_id}?format=raw
```

Example:
```bash
curl -s "https://www.mailinator.com/api/v2/domains/public/inboxes/joe/messages/{message_id}?format=raw" | python3 -m json.tool
```

Response includes:
- `parts`: Array of email parts (text/plain, text/html)
- `headers`: Full email headers
- `subject`, `from`, `to`, `id`, `time`

## Public vs Private Domains

- **Public domains** (e.g., mailinator.com): No authentication required. Verified
  working against `https://www.mailinator.com/api/v2/domains/public/...`.
- **Private domains**: Require an API token. Per Mailinator's API overview, you
  (1) sign up for a plan with a private domain and (2) generate your unique API
  token in the dashboard. Private endpoints are documented under the
  `api.mailinator.com` host, e.g.
  `GET https://api.mailinator.com/api/v2/domains/private/inboxes/{inbox}` and
  message/SMS fetch + HTTP POST inject variants. See the token-authenticated
  examples in "Advanced Examples" below.

## HTTP API vs MCP

- **HTTP API v2**: Recommended for polling, stateless, no session management needed
- **MCP WebSocket/SSE**: Only for real-time streaming; requires persistent connection

For detailed comparison, see `references/mcp-vs-http-api.md`.

## Use Cases

- Receiving email in workflows (e.g. signup for some service)
- Testing email functionality in applications
- Verifying email delivery and content
- Extracting verification codes from emails
- Monitoring email-based notifications

## Notes

- Public inboxes are shared - emails may be deleted by other users
- **HTTP API v2 is recommended** - Stateless polling works reliably; see `references/mcp-vs-http-api.md` for details
- MCP WebSocket/SSE requires persistent connections - not suitable for one-off HTTP requests
- There is no signup, login, or API auth required to access Public inboxes and emails
- You do not need to "create" email addresses. All possible email addresses @mailinator.com already exist. Use anything you like.

## Advanced Examples

### Private Domain Access (with API Token)

```bash
# For private domains, use the api.mailinator.com host and pass your API token.
# The token is documented as a bearer credential; some tooling also accepts it
# as a ?token= query parameter.
curl -s "https://api.mailinator.com/api/v2/domains/{your-domain}/inboxes/{inbox}" \
  -H "Authorization: Bearer ***"
```

### Error Handling

```bash
# Check for empty inbox
response=$(curl -s "https://www.mailinator.com/api/v2/domains/public/inboxes/nonexistent")
if echo "$response" | grep -q '"msgs":\[\]'; then
  echo "Inbox is empty"
elif echo "$response" | grep -q '"error"'; then
  echo "Error: Inbox not found or invalid"
fi

# Handle rate limiting (429 response)
if curl -s -o /dev/null -w "%{http_code}" "https://www.mailinator.com/api/v2/domains/public/inboxes/joe" | grep -q "429"; then
  echo "Rate limit exceeded - wait before retrying"
fi
```

### Integration with Hermes Agent

```python
# Add this to your Hermes skill or agent tool.
# TWO independent safety layers, applied per dynamic component:
#   1. urllib.parse.quote(...) percent-encodes each value so slashes, query
#      params, spaces, '#', '&', etc. cannot alter the URL's structure
#      (path traversal / injected query string). safe="" encodes everything.
#   2. shell_quote(...) then quotes the fully-built command so nothing can
#      escape into the shell that runs curl.
# Encode the variables INDIVIDUALLY before inserting them; never quote the
# whole assembled URL as a substitute for encoding its parts.
from urllib.parse import quote
from hermes_tools import terminal, shell_quote, json_parse

BASE = "https://www.mailinator.com/api/v2/domains/public/inboxes"

def fetch_inbox(inbox_name: str) -> dict:
    """Fetch emails from a Mailinator inbox."""
    inbox = quote(inbox_name, safe="")
    url = f"{BASE}/{inbox}"
    cmd = f"curl -s {shell_quote(url)}"
    result = terminal(command=cmd)
    if result["exit_code"] == 0:
        return json_parse(result["output"])
    raise Exception(f"Failed to fetch inbox: {result['output']}")

def fetch_email(inbox_name: str, message_id: str) -> dict:
    """Fetch a specific email's content."""
    inbox = quote(inbox_name, safe="")
    msg_id = quote(message_id, safe="")
    # format=raw is a literal query param WE control, appended after the
    # encoded path so an encoded component cannot inject its own query string.
    url = f"{BASE}/{inbox}/messages/{msg_id}?format=raw"
    cmd = f"curl -s {shell_quote(url)}"
    result = terminal(command=cmd)
    if result["exit_code"] == 0:
        return json_parse(result["output"])
    raise Exception(f"Failed to fetch email: {result['output']}")
```

### Alert When New Email Arrives (Cross-Platform Hermes Cronjob)

Use Hermes' built-in `cronjob` tool rather than an OS-specific `crontab`/bash
script. This works identically on Linux, macOS, and Windows (no `crontab`,
no `/tmp`, no shell-specific syntax), which keeps this skill honest about its
declared `platforms: [linux, macos, windows]`.

Ask Hermes to create a recurring job whose prompt does the polling and
comparison in-agent. For example:

> "Every 5 minutes, fetch the Mailinator inbox 'joe' via
> `https://www.mailinator.com/api/v2/domains/public/inboxes/joe`, count the
> messages, and if the count is higher than the previous run, alert me with how
> many new messages arrived. Remember the last count between runs."

The agent persists the previous count in its own state across runs, so there is
no temp-file bookkeeping. Delivery/notification is handled by the cronjob's
`deliver` target (a connected messaging platform), not by a hand-rolled
notification block.

If you specifically need a raw scripted poller on a Unix-like host, the logic is
simply: GET the inbox, `len(msgs)`, compare to the previously stored count, and
alert on an increase — but prefer the Hermes cronjob above for portability.

## References

- [Mailinator API Docs](https://www.mailinator.com/mailinator-api/)
- [MCP Endpoint](https://www.mailinator.com/mcp) (requires initialization)
- [Public Inbox Demo](https://www.mailinator.com/v4/public/inboxes/show?inbox=joe)
- [MCP vs HTTP API Comparison](references/mcp-vs-http-api.md) - Detailed comparison and when to use each
