# gog Gmail OAuth: token storage and keychain patterns

When `gog` (v0.11.0+) authorizes a Gmail/Google Workspace account, it stores OAuth 2.0 credentials and tokens as follows:

## Storage locations

```
~/Library/Application Support/gogcli/
  credentials.json   # OAuth client_id + client_secret (one per client)
  keyring/           # keyring backend state (often empty on macOS)
  tokens/            # token cache (often empty; tokens live in system keychain)
```

## Keychain entries (macOS)

All tokens are stored in the login keychain under service name `gogcli`:

```
security find-generic-password -s "gogcli"
```

Account names follow the pattern `token:<client>:<email>`:
- `token:default:user@domain.com` — default client
- `token:imap:user@domain.com` — if using a named client

## Token format

The keychain value is a JSON object:
```json
{
  "refresh_token": "1//03C...",
  "services": ["gmail", "calendar", "contacts", "drive", "docs", "sheets"],
  "scopes": [
    "email",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.settings.basic",
    "https://www.googleapis.com/auth/gmail.settings.sharing",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid"
  ],
  "created_at": "2026-03-05T11:04:00.905617Z"
}
```

## Reading refresh token from keychain

```bash
security find-generic-password -s "gogcli" -a "token:default:user@domain.com" -w | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['refresh_token'])"
```

## Scope constraints

`gog`'s default Gmail authorization uses the REST API scope (`gmail.modify`), **not** the IMAP/SMTP scope (`https://mail.google.com/`). This means:

- `gog` can read/search Gmail via REST API (works perfectly)
- The refresh token **cannot** be used for IMAP XOAUTH2 authentication
- Google's IMAP server will return `{"status":"400","schemes":"Bearer","scope":"https://mail.google.com/"}` — a scope mismatch, not a credential failure
- To use the same client ID for IMAP, the user must re-authorize with `https://mail.google.com/` scope via a browser OAuth flow

## Using gog as a himalaya fallback

When himalaya can't authenticate (password blocked by Google, no OAuth2 build, scope mismatch), `gog` provides a working fallback for reading and searching email:

```bash
# Recent inbox
gog -a user@domain.com gmail search "newer_than:7d" --max 20 --json

# By subject
gog -a user@domain.com gmail search "subject:meeting" --max 10 --json

# By sender
gog -a user@domain.com gmail search "from:boss@company.com" --max 10 --json

# Unread only
gog -a user@domain.com gmail search "is:unread" --max 20 --json
```

The `--json` flag gives structured output with thread IDs, dates, senders, subjects, labels (including UNREAD), and message counts.
