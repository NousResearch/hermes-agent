---
name: chrome-cdp
description: Browser automation via Chrome DevTools Protocol. Launch Chrome, navigate, click, type, screenshot, evaluate JS, manage cookies. Works on Linux, Mac, Windows, and WSL2. Config-based profile management.
tags: [chrome, cdp, browser, automation, scraping]
---

# Chrome CDP - Browser Automation Skill

Full browser automation via Chrome DevTools Protocol. No external services, no Selenium, no Playwright — just raw CDP over WebSocket.

## First-Time Setup

Run the interactive setup wizard:

```bash
python3 scripts/cdp_connector.py setup
```

This will:
1. Detect Chrome on your system
2. Let you choose a Chrome profile:
   - **Option 1**: Use existing Chrome profile (enter path)
   - **Option 2**: Create fresh profile for CDP (recommended)
   - **Option 3**: Delete existing CDP profile and start fresh
3. Configure host/port (auto-detects WSL2 and Windows host IP)
4. Save everything to `config.json`

### Config File

Settings are stored in `scripts/config.json` alongside the connector. You can also edit this file directly:

```json
{
  "default_host": "127.0.0.1",
  "default_port": 9222,
  "chrome_profile": "/path/to/chrome/profile",
  "auto_launch": false,
  "headless": false
}
```

View current config: `python3 scripts/cdp_connector.py config`

Any command flag (`--host`, `--port`, `--profile`) overrides the config value for that run.

## Quick Start (after setup)

```bash
# Launch Chrome
python3 scripts/cdp_connector.py launch

# Navigate
python3 scripts/cdp_connector.py navigate https://example.com

# Screenshot
python3 scripts/cdp_connector.py screenshot /tmp/page.png
```

## Platform Setup

### Linux (native)
Works out of the box. Launches `google-chrome` or `chromium`.

### Mac
Works out of the box. Launches Google Chrome from Applications.

### Windows (native)
Works out of the box. Uses `chrome.exe` from Program Files.

### WSL2 (Windows Subsystem for Linux)
WSL2 cannot reach Windows localhost directly due to network isolation. The setup wizard auto-detects this and offers two options:

**Option A:** Launch Chrome inside WSL (if installed):
```bash
python3 scripts/cdp_connector.py launch --in-wsl
```

**Option B:** Use Windows Chrome via host IP (set during setup):
- The setup wizard finds your Windows host IP automatically
- You launch Chrome on Windows side manually or via PowerShell
- The connector talks to Chrome via the host IP

## Commands

### setup
Interactive setup wizard for first-time configuration.
```bash
python3 scripts/cdp_connector.py setup
python3 scripts/cdp_connector.py setup --non-interactive  # Use defaults
```

### config
Show current configuration.
```bash
python3 scripts/cdp_connector.py config
```

### launch
Start Chrome with remote debugging.
```bash
python3 scripts/cdp_connector.py launch [--host HOST] [--port PORT] [--profile PATH] [--in-wsl] [--headless]
```
- `--profile`: Override config profile for this session
- `--in-wsl`: Use WSL-friendly flags
- `--headless`: Run without GUI (server use)

### status
Check if Chrome CDP is reachable.
```bash
python3 scripts/cdp_connector.py status [--host HOST] [--port PORT]
```

### tabs
List open tabs.
```bash
python3 scripts/cdp_connector.py tabs
```

### navigate
Open URL in a tab (uses first tab if none specified).
```bash
python3 scripts/cdp_connector.py navigate <url> [--tab TAB_ID] [--wait 5]
```
- `--wait`: Seconds to wait for page load (default: 5)

### open
Open URL in a new tab.
```bash
python3 scripts/cdp_connector.py open <url>
```

### screenshot
Capture page screenshot.
```bash
python3 scripts/cdp_connector.py screenshot <output_path> [--tab TAB_ID] [--full] [--wait N]
```
- `--full`: Capture full page (not just viewport)

### click
Click an element by CSS selector.
```bash
python3 scripts/cdp_connector.py click <selector> [--tab TAB_ID]
```

### type
Type text into an element.
```bash
python3 scripts/cdp_connector.py type <selector> <text> [--tab TAB_ID] [--clear]
```
- `--clear`: Clear field before typing

### press
Press a keyboard key.
```bash
python3 scripts/cdp_connector.py press <key> [--tab TAB_ID]
```
Keys: Enter, Tab, Escape, Backspace, ArrowUp, ArrowDown, ArrowLeft, ArrowRight, etc.

### eval
Evaluate JavaScript and return result.
```bash
python3 scripts/cdp_connector.py eval <js_code> [--tab TAB_ID]
```

### cookies
Get all cookies from the browser.
```bash
python3 scripts/cdp_connector.py cookies
```

### set-cookie
Set a cookie.
```bash
python3 scripts/cdp_connector.py set-cookie <name> <value> [--domain .example.com] [--path /]
```

### close-tab
Close a specific tab.
```bash
python3 scripts/cdp_connector.py close-tab <tab_id>
```

## Typical Workflow

```bash
# Full example: login to a site, take screenshot
python3 scripts/cdp_connector.py launch
python3 scripts/cdp_connector.py navigate https://example.com/login
python3 scripts/cdp_connector.py type 'input[name="email"]' 'user@example.com'
python3 scripts/cdp_connector.py type 'input[name="password"]' 'secret123'
python3 scripts/cdp_connector.py click 'button[type="submit"]'
python3 scripts/cdp_connector.py screenshot /tmp/logged_in.png --wait 5
```

## Profile Management

The skill supports three profile modes via `setup`:

1. **Use existing profile** — point to your real Chrome profile. You keep all your logins, cookies, extensions. Best for automating sites where you're already logged in.

2. **Create fresh profile** — isolated profile for CDP only. Your real Chrome stays untouched. Best for testing, scraping, or when you don't want to risk your main profile.

3. **Delete and recreate** — removes the existing CDP profile and creates a clean one. Best for resetting state.

## Dependencies

- Python 3.8+
- `websockets` library (install via pip)
- Chrome/Chromium installed on the machine

## Pitfalls

- **Port conflicts**: If port 9222 is in use, Chrome launch will fail. Kill existing Chrome or use a different port.
- **Tab IDs change**: After navigation, tab IDs may change. Get fresh ones via `tabs` command.
- **Headless mode**: Add `--headless` flag to launch for server use (no GUI needed).
- **WSL2 networking**: Cannot reach Windows localhost directly. Use the default gateway IP from `ip route show`.
- **Chrome accumulation**: Chrome spawns many processes. Kill all Chrome before fresh launch if things get weird.
- **Selector syntax**: Uses CSS selectors. For XPath, use `eval` with `document.evaluate()`.
- **Anti-automation detection**: Some sites detect CDP. Add `--disable-blink-features=AutomationControlled` in launch flags to reduce detection.
- **Profile lock**: Chrome locks its profile directory. If you get "profile in use" errors, close all Chrome instances first.

## Verification

1. `python3 scripts/cdp_connector.py status` should return `{"connected": true}`
2. `python3 scripts/cdp_connector.py tabs` should return a JSON array of tabs
3. `python3 scripts/cdp_connector.py navigate https://example.com` should return success
4. `python3 scripts/cdp_connector.py screenshot /tmp/test.png` should create an image file
