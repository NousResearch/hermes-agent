---
name: browser-automation
description: "Set up local browser automation for AI agents on WSL/Linux — agent-browser + Lightpanda engine, zero GUI deps"
tags: [hermes, agent-browser, lightpanda, wsl, browser-automation, local-chromium]
---

# Browser Automation Setup (WSL/Linux)

## When to Use

Use when setting up browser automation for an AI agent (Hermes, Claude Code, Codex CLI) in a **WSL or headless Linux environment** — no desktop GUI, no X11, no sudo for system deps. Covers local Chrome/Chromium-based and Lightpanda engine setups.

## Problem

Chrome/Chromium requires a dozen shared libraries (`libnspr4`, `libnss3`, `libgtk-3-0`, etc.) that aren't pre-installed on minimal WSL or headless Linux. Installing them requires `sudo` (often blocked), and even then Chrome auto-launch can fail silently. Lightpanda avoids all of this.

## Quick Start (Lightpanda — Recommended for WSL/Linux)

```bash
# 1. Install agent-browser CLI
npm install -g agent-browser

# 2. Download Lightpanda headless browser (Zig, 124MB, no deps)
curl -L -o /tmp/lightpanda \
  https://github.com/lightpanda-io/browser/releases/download/nightly/lightpanda-x86_64-linux
chmod a+x /tmp/lightpanda
sudo mv /tmp/lightpanda /usr/local/bin/lightpanda

# 3. Configure agent-browser to use Lightpanda by default
mkdir -p ~/.agent-browser
cat > ~/.agent-browser/config.json << 'EOF'
{
  "engine": "lightpanda",
  "executablePath": "/usr/local/bin/lightpanda"
}
EOF

# 4. (For Hermes) Set engine in config.yaml
hermes config set browser.engine lightpanda
```

## Verification

```bash
# Test basic navigation + snapshot
agent-browser open https://example.com
agent-browser snapshot -i
# Expected: heading "Example Domain" [ref=e1], link "Learn more" [ref=e2]
```

## Hermes Integration

Once agent-browser + Lightpanda are installed:
- Browser toolset should auto-detect agent-browser
- Tools: `browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_screenshot`, etc.
- Interactive elements get ref IDs (`@e1`, `@e2`) from accessibility tree snapshots
- Hermes uses these refs to click/type/fill

### Starting browser in session
- No `/browser connect` needed — Hermes auto-launches agent-browser as a sidecar
- Use browser tools directly in your prompts

## Lightpanda vs Chrome

| Factor | Chrome | Lightpanda |
|--------|--------|------------|
| Binary size | ~177 MB | ~124 MB |
| System deps | 15+ libs (GTK, NSS, CUPS, X11, ALSA) | None — pure Zig binary |
| sudo required | Yes (apt install) | No (just curl + mv) |
| Headless | Yes | Yes (purpose-built) |
| Screenshots | Yes | No (accessibility tree only) |
| Speed | Standard | 1.3-5.8x faster navigation |
| Use case | Full visual interaction | Text-based automation, scraping |

## Chrome Setup (Alternative — If Lightpanda Is Not Suitable)

If you need screenshots or full visual rendering, install Chrome with dependencies:

```bash
# Install system deps (requires sudo)
sudo apt-get update
sudo apt-get install -y libnspr4 libnss3 libatk-bridge2.0-0t64 \
  libgtk-3-0t64 libgbm1 libxkbcommon0 libxshmfence1 \
  libcups2t64 libdrm2 libasound2t64

# Download and install Chrome via agent-browser
agent-browser install
agent-browser install --with-deps
```

Then optionally pin executable path:
```bash
echo 'export AGENT_BROWSER_EXECUTABLE_PATH=~/.agent-browser/browsers/chrome-*/chrome' >> ~/.bashrc
```

## Pitfalls

- **Chrome exit code 127**: Missing shared libraries. Run `agent-browser install --with-deps` (requires sudo) or switch to Lightpanda.
- **Permission denied on download**: Always `chmod a+x` after curl.
- **agent-browser not found in PATH**: `npm root -g` tells you where globals land. Symlink from a PATH dir or add to PATH.
- **Daemon already running**: If you change engine/executable-path, run `agent-browser close --all` first, then retry with new flags.
- **Hermes still using Chrome**: Check `hermes config` for `browser.engine`. If missing or `auto`, set to `lightpanda` explicitly.

## References

- `references/hermes-browser-setup.md` — Full Hermes-specific browser configuration details
- `references/wsl-chrome-deps.md` — Chrome dependency troubleshooting for WSL
