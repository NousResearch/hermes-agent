---
name: cloakbrowser
description: Route Hermes browser tools through CloakBrowser, a stealth Chromium with C++ source-level fingerprint patches. Use when browser_navigate or browser_snapshot are getting captcha-walled, blocked by Cloudflare/Turnstile, or served bot-detection pages instead of real content.
version: 1.0.0
author: Shashwat Gokhe
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Browser, Stealth, Anti-Bot, Cloudflare, Fingerprint, Chromium, CDP]
    related_skills: [scrapling, browser-harness-driving]
    homepage: https://github.com/CloakHQ/CloakBrowser
    requires_toolsets: [browser]
prerequisites:
  commands: [python3]
required_environment_variables: []
---

# CloakBrowser

[CloakBrowser](https://github.com/CloakHQ/CloakBrowser) is a patched Chromium binary with 49 source-level C++ fingerprint patches (canvas, WebGL, audio, fonts, GPU, screen, WebRTC, network timing, `navigator.webdriver`, CDP input behavior). Unlike `playwright-stealth` or `undetected-chromedriver`, it does not inject JavaScript or tweak runtime flags, so detection sites see a real browser because it is a real browser.

This skill wires Hermes' built-in browser tools (`browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_vision`) through that binary by launching it with `--remote-debugging-port=9222` and pointing `browser.cdp_url` at the resulting CDP endpoint. Once configured, every Hermes browser tool call routes through CloakBrowser automatically.

## When to Use

- `browser_navigate` or `web_extract` returns a Cloudflare challenge page, "Verify you are human" wall, or reCAPTCHA prompt instead of the real content
- A target site behaves differently for Hermes than it does in your real browser (different layout, missing data, redirected to a login wall)
- You need to scrape a Cloudflare/Akamai/PerimeterX/DataDome-protected site
- You want a stealth backend without paying for Browserbase / Browser Use cloud

Note: this only affects Hermes' built-in `browser_*` tools (the ones registered in `tools/browser_tool.py`). External MCP browser servers do not honor `browser.cdp_url` and must be configured separately.

## Install

CloakBrowser is a pip package that downloads its own patched Chromium binary on first run.

```bash
pip install cloakbrowser
python -m cloakbrowser install   # downloads ~206 MB stealth Chromium to ~/.cloakbrowser/
python -m cloakbrowser info      # verify binary path and version
```

The pip package only exposes the binary plus a `launch()` helper. The `cloakserve` CDP multiplexer described in the upstream README ships only inside the Docker image (`cloakhq/cloakbrowser`); the helper script below launches the same binary directly with `--remote-debugging-port`, which is what we want anyway.

If you have Docker available and prefer it, the official one-liner works too:

```bash
docker run -d --name cloak -p 127.0.0.1:9222:9222 cloakhq/cloakbrowser cloakserve
```

Either way, the only thing Hermes needs is a reachable CDP endpoint on `127.0.0.1:9222`.

## Run as a long-lived CDP endpoint

A helper script and a systemd-user unit are shipped with this skill under `scripts/`. They auto-detect the newest installed Chromium build under `~/.cloakbrowser/chromium-*` and expose CDP on `127.0.0.1:9222` with a persistent profile.

Linux (systemd-user):

```bash
SKILL_DIR=~/.hermes/skills/software-development/cloakbrowser
install -m 0755 "$SKILL_DIR/scripts/cloakserve-hermes.sh" ~/.local/bin/cloakserve-hermes
mkdir -p ~/.config/systemd/user
install -m 0644 "$SKILL_DIR/scripts/cloakbrowser.service" ~/.config/systemd/user/cloakbrowser.service
systemctl --user daemon-reload
systemctl --user enable --now cloakbrowser.service
```

macOS or Linux without systemd:

```bash
~/.hermes/skills/software-development/cloakbrowser/scripts/cloakserve-hermes.sh &
disown
```

Windows (PowerShell):

```powershell
python -c "import cloakbrowser, glob, os, subprocess; b=sorted(glob.glob(os.path.expanduser('~/.cloakbrowser/chromium-*/chrome*')))[-1]; subprocess.Popen([b, '--remote-debugging-port=9222', '--remote-debugging-address=127.0.0.1', '--user-data-dir='+os.path.expanduser('~/.cloakbrowser/profile'), '--headless=new', '--no-first-run', '--no-default-browser-check', '--disable-dev-shm-usage'])"
```

Verify the endpoint is up:

```bash
curl -s http://127.0.0.1:9222/json/version
```

The response should advertise `Chrome/146.x.x.x` (no `HeadlessChrome` in the UA).

## Wire Hermes to it

```bash
hermes config set browser.cdp_url http://127.0.0.1:9222
```

Hermes' `tools/browser_tool.py` resolves the CDP target in this order:

1. `BROWSER_CDP_URL` environment variable (live override from `/browser connect`)
2. `browser.cdp_url` in `~/.hermes/config.yaml` (persistent)
3. Per-task cloud provider session (Browserbase / Browser Use / Firecrawl)

With the config key set, every Hermes browser tool call attaches to CloakBrowser. Successful snapshots include `"stealth_features": ["cdp_override"]`.

## Verify it works

Run inside the Hermes venv so the actual tool path is exercised:

```bash
cd $(hermes config show 2>/dev/null | awk '/Install:/{print $2}')
source venv/bin/activate
python -c "
from tools.browser_tool import _get_cdp_override, browser_navigate
import json
print('CDP:', _get_cdp_override())
r = browser_navigate('https://bot.sannysoft.com', task_id='cloak_smoke')
d = json.loads(r) if isinstance(r, str) else r
print('stealth_features:', d.get('stealth_features'))
print(d.get('snapshot', '')[:600])
"
```

Expected output:

- CDP resolves to a `ws://127.0.0.1:9222/devtools/browser/<uuid>` URL
- `stealth_features` contains `"cdp_override"`
- The bot.sannysoft.com snapshot shows `WebDriver (New): missing (passed)`, `WebDriver Advanced: passed`, and a real `Chrome/146.0.0.0` UA, not `HeadlessChrome`

Other useful smoke targets:

- `https://browserscan.net/bot-detection`
- `https://demo.fingerprint.com/playground`
- `https://abrahamjuliot.github.io/creepjs/`

## Common pitfalls

- **`SingletonLock: File exists` after a crash.** The previous Chromium left lock files in the profile. Fix: `rm -f ~/.cloakbrowser/profile/Singleton{Lock,Cookie,Socket}` and restart the service.
- **`cloakserve` command not found.** It is shipped only by the Docker image entrypoint, not by the pip package. Use the helper script under `scripts/` or `docker run cloakhq/cloakbrowser cloakserve`. `python -m cloakbrowser` only manages the binary (`install`, `info`, `update`, `clear-cache`).
- **External MCP browser tools ignore `browser.cdp_url`.** Stealth applies only to Hermes' in-process browser tools registered in `tools/browser_tool.py`. If your session is using a separate MCP browser server, configure that server's CDP target separately or disable it.
- **`BROWSER_CDP_URL` env var wins over `browser.cdp_url`.** If `/browser connect` was used earlier in the session, it sets `BROWSER_CDP_URL` and takes precedence. Unset it (`unset BROWSER_CDP_URL`) to fall back to the config value.
- **Camofox backend takes precedence too.** If `CAMOFOX_URL` is set, Hermes routes to Camofox unless `BROWSER_CDP_URL` overrides it. Pick one stealth backend per session.
- **Headed mode.** The helper script defaults to `--headless=new`. Pass `--headless=false` (or `--headed`) on the command line to watch the browser; needs a display server (Xorg / Wayland / WSLg / macOS / Windows desktop).
- **Proxies.** CloakBrowser supports HTTP and SOCKS5 natively. Append `--proxy-server=socks5://user:pass@host:port` to the helper script's invocation or to the systemd unit's `ExecStart`.
- **Updates.** `python -m cloakbrowser update` pulls a newer binary; the helper script always picks the newest `chromium-*` directory, so a service restart is enough after an upgrade.
- **WSL2.** Works out of the box in headless mode. For headed mode, you need WSLg (Windows 11) or an X server on the Windows side.
- **Disk usage.** The Chromium binary is about 206 MB and lives at `~/.cloakbrowser/chromium-<version>/`. `python -m cloakbrowser clear-cache` removes old versions after an update.

## Files installed by this skill

| Path | Purpose |
|------|---------|
| `~/.cloakbrowser/chromium-<version>/` | Patched Chromium binary (managed by `python -m cloakbrowser`) |
| `~/.cloakbrowser/profile/` | Persistent browser profile (cookies, localStorage) |
| `~/.local/bin/cloakserve-hermes` | Helper script that launches the binary with CDP |
| `~/.config/systemd/user/cloakbrowser.service` | Optional systemd-user unit for auto-start (Linux) |
| `~/.hermes/config.yaml` (`browser.cdp_url`) | Tells Hermes where to attach |

## References

- Upstream repo: https://github.com/CloakHQ/CloakBrowser
- Hermes browser tool source: `tools/browser_tool.py` (`_get_cdp_override`, `_resolve_cdp_override`)
- Hermes config key: `browser.cdp_url` (see `hermes_cli/config.py`)
- Related: `scrapling` skill for HTTP-only stealth, `browser-harness-driving` skill for complex web app automation
