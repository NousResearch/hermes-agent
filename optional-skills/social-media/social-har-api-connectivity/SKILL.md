---
name: social-har-api-connectivity
description: "Capture a social login session via CDP (authorized only)."
version: 1.1.0
author: Joerg Peetz (JPeetz), Hermes Agent
license: MIT
platforms: [linux, macos]
prerequisites:
  commands: ["node", "npm", "Google Chrome or Chromium"]
  packages: ["websockets"]
metadata:
  hermes:
    tags: [social, har, api, connectivity, reverse-engineering, chrome, cdp]
    homepage: https://agentskills.io
    requires_toolsets: [terminal, files]
---

# Social HAR API Connectivity

Connect an agent to a social platform's API by driving Chrome via CDP, capturing
the login flow as the user authenticates, extracting the session tokens, and
building a reusable client. **Authorized use only** — it captures a session the
user themselves logs into; do not use on accounts you do not own.

**This is an interactive workflow: the agent orchestrates, the user authenticates.**

> Tier: **optional** (`optional-skills/social-media/`). Capturing authenticated
> sessions can touch platform Terms of Service — install it only if you need it,
> and use it only on logins you are entitled to perform.

## When to Use
- A platform you need to post/read has no official API access and you hold the
  login for it (undocumented endpoints).
- You need a session token for an endpoint that has no API tier.
- Not for: platforms with a working official API — prefer that (see table).

## Prerequisites
- Chrome (or Chromium) started with remote debugging, e.g.:
  `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --remote-debugging-port=9222`
- Python 3.8+ with `websockets` (`pip install websockets`).
- The user is present to complete login + MFA/CAPTCHA.

## How to Run
Via the `terminal` tool, e.g.:
```
terminal(command="python3 scripts/chrome_capture_client.py --platform x --out /tmp/sess --port 9222 --timeout 120")
```

## Quick Reference
| Platform | Login URL | Prefer official API? |
|---|---|---|
| Bluesky | https://bsky.app/login | Yes — App Password + AT Protocol |
| Mastodon | `[instance]/auth/sign_in` | Yes — bearer token |
| X/Twitter | https://x.com/login | For endpoints not in API tier |
| LinkedIn | https://www.linkedin.com/login | Yes — OAuth |
| Instagram | https://www.instagram.com/accounts/login/ | Yes — Meta Graph API |
| TikTok | https://www.tiktok.com/login | Anti-bot heavy — may be fragile |
| Reddit | https://www.reddit.com/login | Yes — OAuth script app |

## Procedure
1. Agent prompts: "Which social platform do you want to connect?" and confirms the
   user has the login. **Only proceed on logins the user is entitled to.**
2. Agent starts Chrome with CDP (`--remote-debugging-port=9222`), visible mode.
3. Agent runs `chrome_capture_client.py --platform <platform> --out <tmp>`, which
   navigates to the login URL and begins watching network traffic.
4. Agent tells the user the login page is open; user enters credentials and
   handles MFA/CAPTCHA in the Chrome window.
5. Agent waits (timeout), then the script extracts session cookies + auth tokens
   and writes them to `<out>/session.json` with `chmod 600`. **Completion
   criterion:** the file exists and mode is 600.
6. Agent confirms connection and builds a reusable posting client from the session
   material. **Completion criterion:** a client can make an authenticated call.

## Pitfalls
- **MFA is expected** — the visible Chrome window is for the user to complete 2FA.
- **Session tokens expire** (hours to days) — re-capture when stale.
- **Tokens stay local** — never hardcoded, never committed (chmod 600).
- **TikTok anti-bot is aggressive** — capture may fail or sessions expire fast.
- `websockets` must be installed. If absent, error is explicit ("no module").

## Verification
- The captured `session.json` contains at least one request carrying an auth
  header (Authorization, Cookie, or x-* token).
- An authenticated call to the platform succeeds using the extracted token.
- `ls -l <out>/session.json` shows `-rw-------`.
