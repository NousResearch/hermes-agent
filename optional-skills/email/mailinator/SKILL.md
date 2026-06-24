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
- `msgs`: Array of email metadata (id, from, subject, time, seconds_ago)
- `domain`: The domain used (public)
- `to`: The inbox name

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

- **Public domains** (e.g., mailinator.com, gmail.com): No authentication required
- **Private domains**: Require an API token (contact Mailinator for access)

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
# For private domains, include your API token
curl -s "https://www.mailinator.com/api/v2/domains/{your-domain}/inboxes/{inbox}" \
  -H "Authorization: Bearer YOUR_API_TOKEN"
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
# Add this to your Hermes skill or agent tool
from hermes_tools import terminal

def fetch_inbox(inbox_name: str) -> dict:
    """Fetch emails from a Mailinator inbox."""
    cmd = f'curl -s "https://www.mailinator.com/api/v2/domains/public/inboxes/{inbox_name}"'
    result = terminal(command=cmd)
    if result["exit_code"] == 0:
        return json_parse(result["output"])
    raise Exception(f"Failed to fetch inbox: {result['output']}")

def fetch_email(inbox_name: str, message_id: str) -> dict:
    """Fetch a specific email's content."""
    cmd = f'curl -s "https://www.mailinator.com/api/v2/domains/public/inboxes/{inbox_name}/messages/{message_id}?format=raw"'
    result = terminal(command=cmd)
    if result["exit_code"] == 0:
        return json_parse(result["output"])
    raise Exception(f"Failed to fetch email: {result['output']}")
```

### Alert When New Email Arrives (Cron Job)

```bash
# Create a cron job to check for new emails every 5 minutes
# Add to your crontab: */5 * * * * /path/to/mailinator-alert.sh

#!/bin/bash
# mailinator-alert.sh

INBOX="joe"
LAST_CHECK_FILE="/tmp/mailinator_${INBOX}_last_check"

# Get current message count
CURRENT_COUNT=$(curl -s "https://www.mailinator.com/api/v2/domains/public/inboxes/${INBOX}" | \
  python3 -c "import sys,json; print(len(json.load(sys.stdin).get('msgs',[])))" 2>/dev/null)

# Get previous count
if [ -f "$LAST_CHECK_FILE" ]; then
  PREVIOUS_COUNT=$(cat "$LAST_CHECK_FILE")
else
  PREVIOUS_COUNT=0
fi

# Save current count for next check
echo "$CURRENT_COUNT" > "$LAST_CHECK_FILE"

# Alert if new messages arrived
if [ "$CURRENT_COUNT" -gt "$PREVIOUS_COUNT" ]; then
  NEW_MSGS=$((CURRENT_COUNT - PREVIOUS_COUNT))
  echo "New email alert: ${NEW_MSGS} new message(s) in ${INBOX} inbox"
  # Add your notification logic here (Slack, email, etc.)
fi
```

## References

- [Mailinator API Docs](https://www.mailinator.com/mailinator-api/)
- [MCP Endpoint](https://www.mailinator.com/mcp) (requires initialization)
- [Public Inbox Demo](https://www.mailinator.com/v4/public/inboxes/show?inbox=joe)
- [MCP vs HTTP API Comparison](references/mcp-vs-http-api.md) - Detailed comparison and when to use each
