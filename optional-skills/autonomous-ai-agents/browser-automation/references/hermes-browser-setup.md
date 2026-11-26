# Hermes Browser Setup — WSL Configuration Details

Captured from session 2026-05-27 (Hermes Agent v0.14 "Foundation Release").

## Config Changes Made

### config.yaml

```
browser:
  engine: lightpanda          # avoid Chrome GUI deps on WSL
  inactivity_timeout: 120
  command_timeout: 30
  record_sessions: false
  allow_private_urls: false
  auto_local_for_private_urls: true
```

Set via: `hermes config set browser.engine lightpanda`

### agent-browser config (~/.agent-browser/config.json)

```json
{
  "engine": "lightpanda",
  "executablePath": "/usr/local/bin/lightpanda"
}
```

## Agent-Browser Binary Location

When installed via `npm install -g agent-browser`, binary lands at:

```
/home/<username>/.hermes/node/lib/node_modules/agent-browser/bin/agent-browser-linux-x64
```

The npm global bin dir is `/home/<username>/.hermes/node/bin` (not the usual `/usr/local/bin` or `~/.local/bin`).

Symlink from `~/.local/bin/agent-browser` if that dir is on PATH.

## Hermes Browser Toolset

Already enabled ✓. Check with:

```
hermes tools list | grep browser
```

Expected: `✓ enabled  browser  🌐 Browser Automation`

## Session Commands Used

```bash
# Install agent-browser
npm install -g agent-browser

# Find binary
npm root -g
ls /home/<username>/.hermes/node/lib/node_modules/agent-browser/bin/

# Download Lightpanda
curl -L -o /tmp/lightpanda \
  https://github.com/lightpanda-io/browser/releases/download/nightly/lightpanda-x86_64-linux
chmod a+x /tmp/lightpanda
sudo mv /tmp/lightpanda /usr/local/bin/lightpanda

# Test
agent-browser --engine lightpanda --executable-path /usr/local/bin/lightpanda \
  open https://news.ycombinator.com

# Snapshot with compact output
agent-browser snapshot -i -c

# Close daemon (needed before changing engine)
agent-browser close --all
```

## Available Browser Tools (in Hermes session)

| Tool | Purpose |
|------|---------|
| `browser_navigate` | Go to URL (call first) |
| `browser_snapshot` | Get accessibility tree with ref IDs |
| `browser_click` | Click element by ref |
| `browser_type` | Type into textbox |
| `browser_screenshot` | Take screenshot (Chrome only — not Lightpanda) |