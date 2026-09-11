---
name: headed-browser-inspect
description: Use when a headed Playwright/Camoufox page is held for owner view and the agent must read or evaluate that same tab without opening a second page or copying cookies.
version: 0.1.0
author: TotalLag, Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [browser, playwright, camoufox, inspect, takeover]
    related_skills: []
---

# Headed browser inspect (POC)

A second Playwright `connect()` cannot see pages owned by another client (`contexts` is empty). Opening a temp page reloads the site and looks like a refresh.

This skill keeps one hold client on the visible tab and talks to it over a unix socket. Same URL does not `goto`.

## When to Use

- Owner is watching a headed Camoufox/Firefox page (Xvfb / noVNC).
- Agent needs DOM text or `page.evaluate`, not VNC clicking.
- User noticed extra refreshes from a second tab.

Do not use for Chromium CDP (`connect_over_cdp` is Chromium-only). Do not use this to scrape private messages.

## Prerequisites

- Playwright client version matching the Camoufox/Firefox server.
- A browser server websocket, default `ws://127.0.0.1:9377/camoufox`.
- System Python if that is what launched the server.

## How to Run

`SKILL_DIR` is this skill directory.

```bash
# hold the visible tab + inspect socket (long-running)
/usr/bin/python3 "$SKILL_DIR/scripts/hold_page.py"

# same tab, no navigation
/usr/bin/python3 "$SKILL_DIR/scripts/browser_inspect.py" eval --js '() => document.title'

# navigates only if the URL changed
/usr/bin/python3 "$SKILL_DIR/scripts/browser_inspect.py" eval 'https://example.com/' --js '() => location.href'
```

Env (optional): `HERMES_BROWSER_WS`, `HERMES_INSPECT_SOCK`, `HERMES_HOLD_URL`.

## Procedure

1. Start or reuse the headed browser server on localhost.
2. Run `hold_page.py` so the display has a live page and `inspect.sock` exists.
3. Prefer `eval` with no URL when the owner is already on the page.
4. Pass a URL only when you must change pages. Check `navigated` in the JSON.
5. Present public-page talk as executive highlights (see `references/executive-highlights.md`). Do not paste raw comment dumps.

## Privacy

See `references/privacy.md`. This POC does not read cookie databases and does not include live session data.

## Common Pitfalls

1. Importing a file named `inspect.py` shadows Python's stdlib `inspect`. This skill uses `browser_inspect.py`.
2. A second `connect()` still cannot list hold-owned pages. Use the socket.
3. Copying cookies into a temp page causes reloads and is a privacy leak. Do not add that path.
4. Default hold URL is `https://example.com/`. Do not commit real group or post URLs.

## Verification Checklist

- [ ] `python3 "$SKILL_DIR/scripts/test_inspect.py"` passes (no browser required)
- [ ] Hold process prints `HOLDING` and `inspect serve`
- [ ] `eval` with the current URL returns `"navigated": false`
- [ ] No cookies, profile paths, or real social IDs in the skill tree
