# Reading Emails with Himalaya

This reference covers reading, viewing, and managing email messages using the Himalaya CLI.

## Basic Message Reading

Read an email by its ID (shows plain text):

```bash
himalaya message read 42
```

Read with full headers:

```bash
himalaya message read 42 --headers
```

## Full MIME Export

Export the raw MIME source of a message:

```bash
himalaya message export 42 --full
```

Export to a file:

```bash
himalaya message export 42 --full > /tmp/email-42.eml
```

The MIME export is useful for:
- Inspecting email source for debugging deliverability
- Saving emails as `.eml` files for archival
- Extracting raw HTML content or attachments

## Reading HTML Emails

Himalaya renders the plain-text part by default. To read HTML emails:

1. Export the raw MIME source:
   ```bash
   himalaya message export 42 --full > /tmp/email.eml
   ```

2. Extract the HTML part with a MIME parser (e.g., `munpack`, `ripmime`, or Python).

Alternatively, view the email in a browser by saving the HTML part to a file:

```bash
himalaya message export 42 --full | python3 -c "
import sys, email
msg = email.message_from_bytes(sys.stdin.buffer.read())
for part in msg.walk():
    if part.get_content_type() == 'text/html':
        sys.stdout.buffer.write(part.get_payload(decode=True))
        break
" > /tmp/email.html && open /tmp/email.html
```

## Handling Forwarded Emails

Forwarded emails appear as a MIME message with `Content-Type: message/rfc822` attachment. Read the forwarded content:

### Using himalaya export

```bash
himalaya message export 42 --full | python3 -c "
import sys, email
msg = email.message_from_bytes(sys.stdin.buffer.read())
for part in msg.walk():
    if part.get_content_type() == 'message/rfc822':
        fwd = part.get_payload()[0]
        print(f'From: {fwd[\"From\"]}')
        print(f'Subject: {fwd[\"Subject\"]}')
        print(f'Date: {fwd[\"Date\"]}')
        print()
        print(fwd.get_payload(decode=True).decode('utf-8', errors='replace'))
" 
```

### Using himalaya read with Python

```bash
EMAIL_BODY=$(himalaya message read 42 2>/dev/null)
echo "$EMAIL_BODY" | python3 -c "
import sys
lines = sys.stdin.readlines()
# Find forwarded message boundary
in_fwd = False
for line in lines:
    if line.startswith('--- Forwarded message ---'):
        in_fwd = True
    if in_fwd:
        print(line, end='')
"
```

## Extracting Verification Codes from Forwarded Emails

A common use case is extracting one-time passwords or verification codes from forwarded emails:

```bash
himalaya message export 42 --full | python3 -c "
import sys, email, re
msg = email.message_from_bytes(sys.stdin.buffer.read())
body = ''
for part in msg.walk():
    if part.get_content_type() in ('text/plain', 'message/rfc822'):
        if part.get_content_type() == 'message/rfc822':
            fwd = part.get_payload()[0]
            for sub in fwd.walk():
                if sub.get_content_type() == 'text/plain':
                    body = sub.get_payload(decode=True).decode('utf-8', errors='replace')
        else:
            body = part.get_payload(decode=True).decode('utf-8', errors='replace')

# Common verification code patterns
patterns = [
    r'(?:verification|confirmation|activation|one-time|OTP|security)\s*(?:code|number)?[:\s]*([A-Z0-9]{4,8})',
    r'(?:code|pin|token)[:\s]*(\d{4,8})',
    r'(\d{4,8})\s*(?:is your|is the).*?(?:code|OTP|verification)',
]
for p in patterns:
    m = re.search(p, body, re.IGNORECASE)
    if m:
        print(m.group(1))
        break
"
```

## Message Flags

List flags on a message:

```bash
himalaya flag list 42
```

Add a flag (e.g., mark as seen, flagged, or answered):

```bash
himalaya flag add 42 --flag seen
himalaya flag add 42 --flag flagged
himalaya flag add 42 --flag answered
```

Remove a flag:

```bash
himalaya flag remove 42 --flag seen
```

Common flags: `seen`, `answered`, `flagged`, `deleted`, `draft`, `recent`.

## Pagination

List emails with pagination:

```bash
himalaya envelope list --page 1 --page-size 20
himalaya envelope list --page 2 --page-size 20
```

Combine with search:

```bash
himalaya envelope list from example.com --page 1 --page-size 10
```

Get total count for pagination:

```bash
himalaya envelope list --output json | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Total: {len(d)}')"
```

## Output Formats

Most read/list commands support `--output` for structured output:

```bash
himalaya message read 42 --output json
himalaya envelope list --output json
```

JSON output is useful for programmatic processing of email content, fields, and attachments.
