# Hermes Browser Bridge

Automated browser control through your authenticated Chrome session.

## Overview

Hermes Browser Bridge connects Hermes Agent to a user's already-authenticated browser via MCP (Model Context Protocol). This enables natural, human-like browser automation using the actual login session.

**Full Documentation:** https://github.com/fajarkurnia0388/hermes-browser-bridge

## Quick Start

1. Hermes will automatically install dependencies
2. Load the Chrome extension:
   - Open `chrome://extensions/`
   - Enable Developer mode
   - Click _Load unpacked_
   - Select the `extension/` folder
3. Bridge auto-connects at `ws://127.0.0.1:8787`

## Tools Available

- **Snapshots:** `browser_snapshot`, `browser_screenshot`
- **Interactions:** `browser_click`, `browser_type`, `browser_press_key`
- **Navigation:** `browser_navigate`, `browser_tabs`, `browser_switch_tab`
- **Advanced:** `browser_find_element`, `browser_execute_script`, `browser_wait_for_selector`, `browser_scroll`, `browser_wait`

## Key Features

✨ **Real Authenticated Session** — Uses your actual logged-in browser, not a simulator

✨ **Human Fingerprint Preserved** — Automation feels natural and indistinguishable from manual activity

✨ **WSL2 + Windows Support** — Run agent on Linux, control Chrome on Windows

✨ **Secure & Local** — Bridge listens only to `127.0.0.1` (localhost)

## Security

- No external network exposure (localhost only)
- Per-operation debugger attach/detach
- Screenshots transmitted via WebSocket, not persisted
- For production: add token authentication

## Requirements

- Node.js 18+
- Google Chrome / Chromium / Edge 116+

## Troubleshooting

See full documentation: https://github.com/fajarkurnia0388/hermes-browser-bridge

---

**Maintained by:** [Fajar Kurnia](https://github.com/fajarkurnia0388)  
**License:** MIT
