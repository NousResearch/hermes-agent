---
name: social-har-api-connectivity
description: "Capture browser sessions to drive social APIs (authorized use only)."
version: 1.0.0
author: "Joerg Peetz (@JPeetz) + Hermes Agent"
license: MIT
platforms: [macos, linux]
type: optional
source: https://github.com/JPeetz/hermes-agent/tree/main/optional-skills/social-media/social-har-api-connectivity
metadata.hermes:
  tags: [social, api, chrome, har, session]
  category: social-media
requires_toolsets:
  - web
  - browser
  - file
---

# Social HAR API Connectivity

Connect your Hermes Agent to **any social platform's API** by capturing a real browser login session — no official API key needed.

## How it works

1. The agent starts Chrome in visible mode
2. You log in normally (handle passwords, MFA, CAPTCHA yourself)
3. The agent captures all network traffic during the login flow
4. Session tokens, cookies, and API endpoints are extracted into a reusable client
5. The agent can now post/read on that platform through the captured session

## Why use this

Existing skills (`xurl`, `linkedin-posting`) cover platforms with documented APIs. This skill covers the gap: platforms where no agent-friendly API wrapper exists, or where the official API doesn't expose the endpoints you need.

## Platforms

Works with any platform that has a web login. Best results:
- **Bluesky** — prefer App Password with official API
- **TikTok** — anti-bot detection is aggressive, sessions expire fast
- **Mastodon / ActivityPub** — good, standard API after capture
- **Discord** — works, token lifespan varies
- **Reddit** — works, but official OAuth is simpler
- **Instagram / Facebook** — limited, sessions are short-lived

## Security & Ethics

- **Authorized accounts only.** Only capture sessions for accounts you own or have explicit permission to automate.
- **Session hygiene.** Captured tokens are stored locally. The skill never exfiltrates credentials.
- **Chrome visible mode.** You see exactly what the agent is doing — no hidden automation.
- **No password capture.** The agent captures HTTPS traffic headers, not keystrokes or form data.

## Files

- `scripts/chrome_capture_client.py` — CDP-based HAR capture tool
- `SKILL.md` — this file

## Requirements

- Google Chrome or Chromium installed
- Hermes Agent with browser toolset enabled
- Network capture requires Chrome DevTools Protocol access