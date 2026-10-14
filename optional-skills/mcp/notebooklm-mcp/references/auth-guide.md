# NotebookLM — authentication deep dive

> **Security notice**: NotebookLM (Gemini Notebook) has **no official API**. The only supported authentication path is to extract Google browser cookies and cache them locally. A `cookies.txt` file or `auth.json` cache is equivalent to a Google session. Treat them as secrets. Never commit, share, or expose them in logs, screenshots, or chat messages.
>
> **All extraction and import steps in this document are performed by the user**, on a machine and browser they control. The agent never extracts, reads, parses, or forwards cookie values — it runs `nlm` commands as black-box operations only after explicit user consent.

NotebookLM (Gemini Notebook) has **no official API**. Authentication is done by extracting **Google browser cookies** from a logged-in session and caching them. The CLI/MCP refreshes CSRF tokens and session IDs automatically from those cookies.

## Where tokens are stored

```
~/.notebooklm-mcp-cli/profiles/<profile>/auth.json
```

Default profile: `default`. Each profile is an isolated Google account.

## Cookie file format (manual mode)

The file should contain the raw cookie string from Chrome DevTools:

```
SID=abc123...; HSID=xyz789...; SSID=...; APISID=...; SAPISID=...; __Secure-1PSID=...; __Secure-3PSID=...; ...
```

Rules:
- Lines starting with `#` are treated as comments and ignored.
- The file can contain the cookie string on one or multiple lines.
- A template `cookies.txt` is included in the upstream repository.

## How the user extracts cookies manually

> User-performed steps — the agent must not execute them. The user follows this checklist on their own machine, then hands the resulting `cookies.txt` to the import step and deletes it right after.

1. Open Chrome and go to **https://notebooklm.google.com**
2. Make sure you are logged in to your Google account.
3. Press **F12** (or **Cmd+Option+I** on Mac) to open DevTools.
4. Click the **Network** tab.
5. In the filter box, type: `batchexecute`
6. Click on any notebook to trigger a request.
7. Click on a `batchexecute` request in the list.
8. In the right panel, scroll to **Request Headers**.
9. Find the line starting with `cookie:`.
10. Right-click the cookie **value** and select **Copy value**.
11. Paste into a text file and save as `cookies.txt`.

Then on the server:

```bash
nlm login --manual --file cookies.txt
# or interactive:
nlm login --manual
```

## OpenClaw CDP provider

When an OpenClaw-managed browser is already running and exposing a Chrome DevTools Protocol endpoint, `nlm` can read cookies from it without launching a second browser:

```bash
nlm login --provider openclaw --cdp-url http://127.0.0.1:18800
```

- Uses `suppress_origin=True` for websocket CDP commands to support managed endpoints that reject the default Origin header.
- The browser must already be logged in to Google / NotebookLM.
- Increase the DevTools timeout if the endpoint is slow:

```bash
nlm login --provider openclaw --cdp-url http://127.0.0.1:18800 --devtools-timeout 15
```

## Auto mode (desktop with browser)

```bash
nlm login              # launches a dedicated browser profile, you log in, cookies extracted
```

Supported browsers (auto-detected, in order): Chrome, Arc, Brave, Edge, Chromium, Firefox, Vivaldi, Opera.

Prefer a specific browser:

```bash
nlm config set auth.browser chromium
```

How it works:
1. The first available supported browser is detected (or your preferred browser if configured).
2. A dedicated browser profile is created for authentication.
3. The browser launches with the appropriate automation backend (CDP for Chromium-family; direct cookie DB read for Firefox).
4. You log in to NotebookLM via the browser.
5. Cookies are extracted and cached; CSRF/session fields are refreshed automatically when needed.
6. The browser is closed automatically.

## Multi-profile auth

```bash
nlm login --profile work
nlm login --profile personal
nlm login switch work
nlm login profile list
nlm login profile delete personal
nlm login profile rename personal pro
```

The MCP server always uses the **active default profile**. Switching the default profile instantaneously switches the MCP server's Google account:

```bash
nlm config set auth.default_profile work
```

## Auth lifecycle

| Component | Duration | Refresh |
|-----------|----------|---------|
| Cookies | ~2-4 weeks | Auto-refresh via headless browser (if profile saved) |
| CSRF token | minutes | Auto-refreshed on every request failure |
| Session ID | session | Embedded in cookies |

## Understanding `auth_status`

`nlm login --check` reports one of:

| Status | Meaning |
|--------|---------|
| `ok` | Cookies valid, last check succeeded |
| `stale` | Cookies present but not verified recently — re-run `nlm login` to confirm |
| `unverified` | Cookies present but never checked — run `nlm login --check` |
| `failed` | Cookies invalid or expired — re-login |

## Copying auth.json to a headless server

If you authenticated on a desktop and want the same session on a server:

```bash
# On the desktop:
scp ~/.notebooklm-mcp-cli/profiles/default/auth.json \
    user@server:~/.notebooklm-mcp-cli/profiles/default/auth.json

# On the server:
nlm login --check    # confirm the copied cookies still work
```

## Troubleshooting auth

### `ClientAuthenticationError` / `network_error`

Cookies are present but the session is dead or the network is blocked. Re-extract cookies:

```bash
nlm login --manual --file cookies.txt   # manual
nlm login                               # auto (desktop)
nlm login --provider openclaw --cdp-url http://127.0.0.1:18800  # CDP
```

### Cookie replay vs browser-bound auth

Some Google endpoints require a browser-bound session (not just cookie replay). Diagnose with:

```bash
nlm doctor auth-replay
```

If cookie replay fails, you need auto mode with a real browser (or OpenClaw CDP).

### Browser not found (headless)

```bash
nlm doctor
# Browser: not found
# → A supported browser is required for authentication
# Headless auth: not available (no saved profile)
```

Fix: use manual file mode or OpenClaw CDP. See the headless flow diagram in `SKILL.md`.

## Security hardening

### File permissions

Set restrictive permissions on the cookie cache immediately after creation:

```bash
chmod 600 ~/.notebooklm-mcp-cli/profiles/default/auth.json
find ~/.notebooklm-mcp-cli/profiles -type f -name "*.json" -exec chmod 600 {} \;
```

On first setup, also create the directory with a restrictive umask:

```bash
install -d -m 700 ~/.notebooklm-mcp-cli
```

If the environment supports it, prefer encrypting `auth.json` at rest (`gpg --symmetric`, `ansible-vault`, or the platform's keychain).

### Redact before logging

Before sharing `nlm` output or MCP tool results, redact cookies and bearer values:

```bash
nlm doctor 2>&1 | sed -E 's/(SID|HSID|SSID|APISID|SAPISID|__Secure-[0-9]PSID)[^=]*=[^; ]*/\1=REDACTED/g'
```

### CDP endpoint trust

Only use OpenClaw CDP URLs that the user explicitly configured and controls. Validate the endpoint with `openclaw config get browser.profiles` before running `nlm login --provider openclaw`.

### Audit the actual implementation

When using this skill with real data, confirm:

1. `auth.json` is never written to version control.
2. `auth.json` and any `cookies.txt` are stored with `0600` permissions or encrypted at rest.
3. Cookies are not printed to stdout/stderr by the upstream `nlm` binary.
4. Temporary `cookies.txt` files are deleted immediately after `nlm login --manual --file` succeeds.
5. The OpenClaw CDP IP/port is explicitly allowlisted and controlled by the user.
6. The `notebooklm-mcp-cli` package is installed from [PyPI](https://pypi.org/project/notebooklm-mcp-cli/) with a pinned version, not an untrusted source.
7. The user has permission to use the Google account in this way (no shared or borrowed sessions).
8. The MCP server only talks to `notebooklm.google.com` and `*.google.com` domains.
