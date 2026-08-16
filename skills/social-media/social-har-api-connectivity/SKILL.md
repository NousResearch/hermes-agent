---
name: social-har-api-connectivity
description: "Connect a social platform's API by driving Chrome to capture the login + network traffic — the user logs in, the agent captures the session (authorized use only)."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
prerequisites:
  commands: ["node", "npm"]
  packages: ["websockets"]
metadata:
  hermes:
    tags: [social, har, api, connectivity, reverse-engineering, chrome, cdp, posting]
    homepage: https://agentskills.io
---

# Social HAR API Connectivity

Connect an agent to any social platform's API by driving Chrome via CDP,
capturing the login flow as the user authenticates, extracting the session
tokens, and building a reusable client.

**This is an interactive workflow: the agent orchestrates, the user authenticates.**

## Workflow
1. The agent prompts the user: "Which social platform do you want to connect?"
2. The agent starts Chrome with CDP (visible mode) and navigates to the login page.
3. The agent tells the user the login page is open — the user enters credentials
   and handles any MFA/CAPTCHA in the browser window.
4. The agent monitors network traffic via CDP and detects the post-login redirect.
5. The agent stops capture, extracts session cookies + auth tokens + API host, and
   saves the data to a temp directory (chmod 600).
6. The agent confirms connection and builds a reusable posting client.

## Included tool
`scripts/chrome_capture_client.py` — a Python script that automates the Chrome
CDP capture. The agent runs this for the user.

## Supported platforms (login URLs)
| Platform | Login URL | Prefer official API? |
|---|---|---|
| Bluesky | https://bsky.app/login | Yes — App Password + AT Protocol |
| Mastodon | [instance]/auth/sign_in | Yes — bearer token |
| X/Twitter | https://x.com/login | For endpoints not in API tier |
| LinkedIn | https://www.linkedin.com/login | Yes — OAuth |
| Instagram | https://www.instagram.com/accounts/login/ | Yes — Meta Graph API |
| Facebook | https://www.facebook.com/login | Yes — Meta Graph API |
| TikTok | https://www.tiktok.com/login | Anti-bot heavy — may be fragile |
| Reddit | https://www.reddit.com/login | Yes — OAuth script app |
| Pinterest | https://www.pinterest.com/login | Yes — official API if approved |
| Threads | https://www.threads.net/login | Capture for undocumented endpoints |
| YouTube | https://accounts.google.com/ | Yes — YouTube Data API |

## Pitfalls
- MFA is expected — the visible Chrome window is for the user to complete 2FA.
- Session tokens expire (hours to days) — re-capture when stale.
- Tokens stay local — never hardcoded, never committed.
- TikTok anti-bot is aggressive — capture may fail or sessions may expire fast.
- The `websockets` Python package must be installed (`pip install websockets`).