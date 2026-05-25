# Browser OAuth flow for himalaya IMAP scope

When a himalaya account using Google IMAP needs an OAuth2 refresh token with the `https://mail.google.com/` scope, and no such token exists yet, the user must complete a one-time browser OAuth authorization.

## When this is needed

- Password auth is blocked by Google ("Application-specific password required")
- Existing OAuth tokens have REST API scope only (e.g., from `gog`)
- Building himalaya from source with `+oauth2` feature

## Flow overview

1. Start a local HTTP server on a random high port to catch the OAuth redirect
2. Open the user's browser to Google's OAuth consent screen with `https://mail.google.com/` scope
3. User approves (Google auto-selects the correct account if `login_hint` is set)
4. Browser redirects to local server with `code` parameter
5. Exchange authorization code for access + refresh tokens
6. Store refresh token in system keychain for himalaya to use

## Automated script pattern

```python
#!/usr/bin/env python3
"""OAuth for Google IMAP scope. Run once per account."""
import http.server, json, subprocess, sys, threading, time
import urllib.parse, urllib.request, webbrowser

CLIENT_ID = "YOUR_CLIENT_ID"
CLIENT_SECRET="YOUR...= 60072
REDIRECT = f"http://127.0.0.1:{PORT}/oauth2/callback"
SCOPE = "https://mail.google.com/"
EMAIL = "user@domain.com"
ACCT = f"token:imap:{EMAIL}"

code = None
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        global code
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        if 'code' in params:
            code = params['code'][0]
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<h1>Done! Close this window.</h1>')
            threading.Thread(target=self.server.shutdown, daemon=True).start()
    def log_message(self, *a): pass

server = http.server.HTTPServer(('127.0.0.1', PORT), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()

auth_url = "https://accounts.google.com/o/oauth2/auth?" + urllib.parse.urlencode({
    'client_id': CLIENT_ID, 'redirect_uri': REDIRECT,
    'response_type': 'code', 'scope': SCOPE,
    'access_type': 'offline', 'prompt': 'consent',
    'login_hint': EMAIL, 'include_granted_scopes': 'true',
})

print(f"Opening browser for {EMAIL}...", flush=True)
webbrowser.open(auth_url)

for _ in range(120):
    if code: break
    time.sleep(0.5)
server.shutdown()

if not code:
    print("ERROR: Timed out", flush=True)
    sys.exit(1)

# Exchange code for tokens
data = urllib.parse.urlencode({
    'code': code, 'client_id': CLIENT_ID,
    'client_secret': CLIENT_SECRET, 'redirect_uri': REDIRECT,
    'grant_type': 'authorization_code',
}).encode()

req = urllib.request.Request(
    'https://oauth2.googleapis.com/token', data=data, method='POST',
    headers={'Content-Type': 'application/x-www-form-urlencoded'}
)
resp = json.loads(urllib.request.urlopen(req, timeout=10).read())

rt = resp.get('refresh_token', 'MISSING')
print(f"Got refresh token: {rt[:20]}...", flush=True)

# Store in keychain
subprocess.run(['security', 'delete-generic-password',
    '-a', ACCT, '-s', 'himalaya-oauth'], capture_output=True)
subprocess.run(['security', 'add-generic-password',
    '-a', ACCT, '-s', 'himalaya-oauth',
    '-w', json.dumps(resp), '-U'])
print("Saved to keychain!", flush=True)
```

## Key points

- **`include_granted_scopes=true`** ensures existing scopes aren't lost
- **`prompt=consent`** forces a refresh token (without it, Google may not issue one if previously authorized)
- **`login_hint`** pre-fills the account picker so the user just clicks "Allow"
- The local server listens on `127.0.0.1` only — never accessible from the network
- The redirect URI must exactly match what's registered in the Google Cloud Console
- Store the full token response (not just the refresh token) — access token is useful for immediate use

## After the flow

Update himalaya's config.toml to reference the keychain entry:

```toml
backend.auth.refresh-token = { cmd = "/path/to/get-refresh-token.sh" }
backend.auth.access-token = { cmd = "/path/to/get-access-token.sh" }
```

Where `get-refresh-token.sh` reads from keychain:
```bash
#!/bin/bash
security find-generic-password -s "himalaya-oauth" -a "token:imap:user@domain.com" -w | \
  python3 -c 'import sys,json; print(json.load(sys.stdin)["refresh_token"])'
```

And `get-access-token.sh` exchanges refresh for access:
```bash
#!/bin/bash
CLIENT_ID="..."
CLIENT_SECRET=*** find-generic-password -s "himalaya-oauth" -a "token:imap:user@domain.com" -w | python3 -c 'import sys,json; print(json.load(sys.stdin)["refresh_token"])')

curl -s --max-time 10 \
  -d "client_id=${CLIENT_ID}" \
  -d "client_secret=${CLIENT_SECRET}" \
  -d "refresh_token=${REFRESH_TOKEN}" \
  -d "grant_type=refresh_token" \
  "https://oauth2.googleapis.com/token" | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])'
```

## Why not `gog auth add`?

`gog auth add` uses the Gmail REST API scope (`gmail.modify`), not the IMAP scope (`https://mail.google.com/`). Running `gog auth add --services gmail` will NOT give you an IMAP-compatible token. You must use `https://mail.google.com/` as the scope, which requires either a custom OAuth flow (above) or a Google Cloud Console project with the IMAP scope enabled.
