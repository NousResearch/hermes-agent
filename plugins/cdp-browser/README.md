# cdp-browser — Composed One-Pass CDP Driver for Hermes Agent

**Drive any Chrome-family browser (Brave, Chrome, Edge, Chromium) with one-pass
composed actions, parallel tab spaces, and token-cheap semantic snapshots — as
native Hermes tools.**

```
cdp_list    → list open tabs (targetId | title | url)
cdp_run     → run a multi-step browser script in ONE websocket connection
cdp_spaces  → run N named tabs concurrently, each with its own script
```

Built for LLM agents that need fast, cheap, reliable browser automation. The
design borrows three ideas from [citrolabs/ego-lite](https://github.com/citrolabs/ego-lite)
and implements them directly over the Chrome DevTools Protocol — no separate
browser, no proprietary runtime, no API key.

---

## Why it improves browser use

### 1. Code-base, not CLI-base (up to 2.5× faster workflows)

Classic browser automation makes the agent loop: *snapshot → decide → click →
snapshot → decide → type…* — a fresh round-trip through the LLM for every
single action.

`cdp_run` flips that. You write the **whole task as one JSON steps script**
and the driver executes every step in a single websocket connection:

```json
[
  {"op": "open_tab", "url": "https://example.com"},
  {"op": "snapshot", "max": 15},
  {"op": "eval", "expr": "document.title"},
  {"op": "capture", "out": "C:/tmp/shot.png"},
  {"op": "close"}
]
```

One invocation. One result payload. The agent composes; the browser executes.
No intermediate reasoning round-trips — exactly the pattern ego-lite uses to
finish complex tasks faster with far fewer tokens.

### 2. Spaces — parallel isolated contexts

One browser, many tabs, all driven at once. `cdp_spaces` runs N named tabs
concurrently, each with its own steps script — ideal for scraping N sites,
filling N forms, or generating N videos in parallel:

```json
{
  "spaces": [
    {"name": "a", "tab": "new", "url": "https://example.com",  "steps": [{"op": "snapshot"}]},
    {"name": "b", "tab": "new", "url": "https://httpbin.org/html", "steps": [{"op": "snapshot"}]}
  ]
}
```

Each space is fully isolated — they share the browser process but never each
other's tabs or state. Your own tabs stay untouched.

### 3. Strong, cheap semantic snapshots (~7× fewer tokens)

Instead of dumping a giant accessibility tree (100–700 elements per capture),
`snapshot` returns a **compact semantic view**: interactive elements with
role, visible text, bounding box, and a stable CSS selector — plus a short
body-text digest.

Measured on a real Gemini page: **26 semantic elements** vs **~190 AX refs**.
That's roughly a **7× reduction in tokens per snapshot**, which is the
dominant cost in most browser-automation sessions.

```json
{"role": "button", "txt": "Close sidebar", "sel": "button", "x": 238, "y": 8}
```

Selectors prefer `id` and `data-test-id` attributes when present, so they
survive re-renders better than positional coordinates.

---

## How to add to Hermes (2 ways)

### Option A — plugin directory (recommended, no CLI)

```bash
# 1. Clone into your Hermes plugins dir
#    (profile-aware: use $HERMES_HOME if set, else ~/.hermes)
git clone https://github.com/sahilthakur456111-stack/cdp-browser ~/.hermes/plugins/cdp-browser

# 2. Install the one Python dependency
pip install websocket-client

# 3. Enable it
hermes plugins enable cdp-browser
```

Restart / start your next Hermes session. The tools appear automatically.

### Option B — `hermes plugins install` (from the repo)

```bash
hermes plugins install sahilthakur456111-stack/cdp-browser
hermes plugins enable cdp-browser
pip install websocket-client
```

### Prerequisite: launch your browser with a CDP port

The driver talks to a browser that exposes the Chrome DevTools Protocol. Launch
any Chrome-family browser with a remote debugging port (avoid 9222 — Edge
often owns it):

```bash
# Brave example:
"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe" \
  --remote-debugging-port=9333
```

Then verify: `curl http://127.0.0.1:9333/json/version`

---

## Usage

### List tabs

```
cdp_list(port=9333)
```

Returns one line per page tab: `targetId | title | url`. Use the targetId as
`tab` in `cdp_run`.

### Run a steps script

```
cdp_run(steps='[{"op":"snapshot","max":15},{"op":"eval","expr":"document.title"}]',
        tab='auto', port=9333)
```

- `tab` accepts: `auto` (first page tab), `gemini` (first gemini.google.com
  tab), `new` (open a fresh tab), or an exact targetId from `cdp_list`.

### Parallel spaces

```
cdp_spaces(spaces='{"spaces":[...]}', port=9333)
```

Returns per-space step results keyed by space name.

## Steps reference

| op | params | description |
|---|---|---|
| `open_tab` | `url` | create + attach a new tab (force-navigates if url given) |
| `navigate` | `url` | navigate the attached tab |
| `snapshot` | `max` | compact semantic snapshot (interactive els + text digest) |
| `eval` | `expr`, `ret` | `Runtime.evaluate` (ret=returnByValue, default true) |
| `focus` | `sel` | focus element by CSS selector |
| `type` | `text` | `Input.insertText` — fires native input events, no clipboard |
| `click` | `sel` | `el.click()` by CSS selector |
| `click_coord` | `x`, `y` | `Input.dispatchMouseEvent` press+release |
| `upload` | `sel`, `file` | `DOM.setFileInputFiles` — attach a file, no native dialog (retries 8×) |
| `wait` | `ms` | sleep |
| `capture` | `out` | `Page.captureScreenshot` to PNG |
| `close` | — | close the attached tab |
| `echo` | `msg` | debug marker |

---

## Security

The plugin was audited before release (2026-08-14). Key properties:

- **No shell execution** — subprocess args are passed as a list, never
  `shell=True`. No command injection surface.
- **No credentials** — no API keys, no stored secrets, no network access at
  import time.
- **Trusted-agent model** — the `eval` op executes arbitrary JavaScript in the
  attached tab (that's the point of browser automation). Only enable this
  plugin for agents you trust to drive your browser.
- **CDP ports are unauthenticated** — the remote-debugging port allows any
  local process to control the browser. Bind to `127.0.0.1` and never expose
  the port to a network.
- **Timeouts + truncation** — every driver call is bounded (120s) and error
  output is truncated before reaching the agent.

## License

MIT — see [LICENSE](LICENSE).

## Architecture

```
plugin.yaml     manifest (name, version, provides_tools)
schemas.py      LLM-facing tool schemas (descriptions + parameter specs)
tools.py        handlers — shell out to the bundled driver
driver/cdp_browser.py   the composed CDP driver (ops protocol, websocket client)
```

The driver is a standalone, dependency-light Python script (`websocket-client`
only) — usable from the terminal without Hermes:

```bash
python driver/cdp_browser.py list --port 9333
python driver/cdp_browser.py run steps.json --tab auto --port 9333
python driver/cdp_browser.py spaces spaces.json --port 9333
```

## Verified

Gauntlet-verified end-to-end (5/5) on 2026-08-14: one-pass composition, tab
lifecycle, parallel spaces, live semantic snapshots, focus/type/upload/cleanup
on a real Gemini composer — zero quota spent on read-only verification.

## Related

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) — the agent
  framework this plugin extends
- [ego-lite](https://github.com/citrolabs/ego-lite) — source of the three
  design ideas (code-base, spaces, strong snapshots)
