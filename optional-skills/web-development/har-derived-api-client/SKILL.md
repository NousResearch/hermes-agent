---
name: har-derived-api-client
description: Record a site's XHR into a HAR, derive an HTTP client.
version: 0.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Browser, HAR, API, Reverse-Engineering, Playwright]
    category: web-development
---

# HAR-Derived API Client

Drive a website once with a real browser while recording its network traffic
to a HAR file, then distill that HAR into the site's private JSON API so you
can call it directly with plain HTTP — often cheaper and faster than
browser-controlling the page on every request. Credit: trick by Jared Longster,
popularized by Dax (thdxr). This captures and replays; it does NOT bypass
auth, solve CAPTCHAs, or defeat bot-detection — if the site needs a logged-in
session, provide equivalent credentials at runtime from an approved secret
source; never paste secrets from a HAR into generated code or command history.

The scripts are stdlib-plus-Playwright: capture needs Playwright, derivation
is pure stdlib, replay needs only `requests`/`httpx` (or `curl`).

Covers **every Hermes browser pathway**: the default local `browser_navigate`
backend, plus the cloud/remote backends (Browserbase, Browser-Use, Firecrawl)
and any `/browser connect` CDP endpoint. There are two capture scripts — one
for a browser you launch, one for a browser you attach to over CDP — because
HAR recording works differently in each case (see How to Run).

## When to Use

- "Build a CLI/client for <website>" — derive its API instead of scripting clicks.
- "This site has no public API but the page clearly fetches JSON."
- You're about to loop `browser_navigate` for the same query repeatedly — stop and derive the endpoint once.
- Reverse-engineering an autocomplete, search, feed, or checkout XHR.
- You captured a session on a cloud backend (Browserbase / Browser-Use / Firecrawl) or via `/browser connect` and want the API without re-renting the browser.

## Prerequisites

- Playwright + a browser binary (capture step only):
  - `pip install playwright` then `playwright install chromium`
  - (If a system Playwright already has browsers under `~/.cache/ms-playwright`, reuse it.)
- `requests` or `httpx` for the replay step (stdlib `urllib` also works).
- No API keys are needed by the skill itself. Treat captured keys, tokens, and
  cookies as secrets; `har_to_client.py` reports their presence and redacts
  credential headers plus common query/JSON/form credential fields. Capture
  scripts restrict newly written HAR files to the current OS user where POSIX
  permissions are available.
- For the CDP path (`har_capture_cdp.py`): a reachable CDP endpoint. On Hermes,
  run `/browser connect` to print the active endpoint, or read `BROWSER_CDP_URL`
  / `browser.cdp_url` in config. Cloud backends expose it as `cdpUrl`/`connectUrl`.

## How to Run

Scripts under this skill's `scripts/`, invoked through the `terminal` tool.
**Pick the capturer by pathway** — this is the part that trips people up:

| Browser pathway | How Hermes reaches it | Capturer |
|---|---|---|
| Local `browser_navigate` (default, agent-browser/Playwright) | launched locally | `har_capture.py` |
| Camofox (`CAMOFOX_URL` set) | local REST/CDP | `har_capture_cdp.py` if it exposes CDP, else drive it yourself |
| Browserbase / Browser-Use / Firecrawl (cloud) | **CDP** (`cdpUrl`) | `har_capture_cdp.py` |
| `/browser connect <url>` / `BROWSER_CDP_URL` | **CDP** | `har_capture_cdp.py` |

Rule of thumb: **if Hermes *launched* the browser, use `har_capture.py`; if it
*connected to* one over CDP, use `har_capture_cdp.py`.** `har_capture.py` uses
Playwright's `record_har_path`, which only works on a locally-owned context.
`har_capture_cdp.py` attaches with `connect_over_cdp()` and assembles the HAR
from `page.on("request"/"response")` events, because `record_har_path` is
unavailable on a connected browser.

Then, for either path:

- `har_to_client.py` — filters the HAR to XHR/fetch/JSON, groups by endpoint, and prints params, headers, bodies, and replay hints (User-Agent / cookie / auth).

Resolve paths against this skill's directory. Canonical loop:

```bash
# 1a. Capture, LOCAL browser (Hermes launched it)
python3 scripts/har_capture.py "https://SITE/" out.har \
  --action "fill:input[name=search]:my query" --action "sleep:3" --wait 2

# 1b. Capture, CDP browser (cloud backend or /browser connect)
#     get the endpoint from /browser connect or BROWSER_CDP_URL
python3 scripts/har_capture_cdp.py "ws://HOST/devtools/browser/..." out.har \
  --goto "https://SITE/" --action "fill:input[name=search]:my query" \
  --action "sleep:3" --wait 2

# 2. Derive — read the endpoints out of the HAR
python3 scripts/har_to_client.py out.har --host SITE --max-body 400

# 3. Replay — write a tiny client from the printed endpoint (see Procedure)
```

## Quick Reference

```
har_capture.py <url> <out.har> [--wait S] [--headed] [--action SPEC ...]
  action SPEC:  fill:SELECTOR:TEXT | press:SELECTOR:KEY | click:SELECTOR
                goto:URL | sleep:SECONDS      (run in order after page load)
  use when Hermes LAUNCHED the browser (local browser_navigate default)

har_capture_cdp.py <cdp_url> <out.har> [--goto URL] [--wait S] [--action SPEC ...]
  same action SPEC; attaches to an existing CDP browser and does NOT close it
  use for cloud backends (Browserbase/Browser-Use/Firecrawl) & /browser connect

har_to_client.py <in.har> [--host SUBSTR] [--include-static] [--max-body N]
  default: keeps only XHR/fetch/JSON; --host narrows to one domain
  prints per endpoint: redacted query params, non-boring req headers, req body sample,
                       response status/content-type + body sample
  prints "### Replay hints": the browser User-Agent, cookie/auth presence
```

## Procedure

0. **Pick the capturer by pathway** (see How to Run table). Launched-locally → `har_capture.py`; reached over CDP → `har_capture_cdp.py`. On Hermes, `/browser connect` tells you the CDP endpoint when a cloud/remote backend is active.
1. **Find the interaction.** Open the site with `browser_navigate` (or `--headed` capture) to see which selector to type into / click, and confirm a JSON XHR fires in devtools/network.
2. **Capture the HAR** via the `terminal` tool. Order `--action` to reach the request: `fill` the box, then `sleep` long enough for the debounced XHR, and always leave `--wait` at the end so late responses flush. Both capturers embed response bodies, so the derived client sees real payload shapes.
3. **Derive** with `har_to_client.py --host <domain>`. Read off: the method, the URL/path template (numeric/UUID segments collapse to `{id}`), query params, request-body JSON, and the `### Replay hints` block.
4. **Write the client.** Recreate the request exactly — same scheme, method, path, query params, and body. Send the non-secret headers the site actually needs, including the **User-Agent** from the replay hints. If hints report cookies or an auth/token header, obtain an equivalent current value from an approved runtime secret source; never paste the HAR value into source code or a shell command.
5. **Test browserless.** Run the client with the `terminal` tool and confirm it returns the same data the browser saw. This is the payoff: no browser in the loop.
6. **(Optional) Wrap as a CLI** — a small `argparse` script over the derived call, e.g. `search.py "frank herbert"`.

Worked example (Wikipedia search-title, derived + replayed live):

```python
import requests
r = requests.get(
    "https://en.wikipedia.org/w/rest.php/v1/search/title",
    params={"q": "frank herbert", "limit": 5},
    headers={"accept": "application/json",
             "User-Agent": "Mozilla/5.0 ... Chrome/131 Safari/537.36"},  # from HAR
    timeout=15,
)
for p in r.json()["pages"]:
    print(p["title"], "-", p.get("description"))
```

## Pitfalls

- **Default library User-Agent gets 403.** Many sites (Wikipedia, Cloudflare-fronted APIs) reject `python-requests/x.y`. Always send the browser UA from the replay hints. This is the #1 reason a derived client fails when the browser succeeded.
- **A failed `--action` aborts before the HAR flushes** — you get no file. If capture errors on a selector, the run produced nothing; fix the selector (use `--headed` to watch) and rerun. Don't debug a missing HAR.
- **Server-rendered pages have no XHR** to derive — `har_to_client.py` prints "No API-looking entries". The data came in the HTML; scrape it or find the interaction that does fetch JSON.
- **Debounced/typeahead XHRs need a real pause.** Add `--action "sleep:3"` after `fill`; typing alone won't have fired the request when the HAR closes.
- **Auth/session endpoints** need equivalent current `Cookie`/`Authorization` values, and those expire. The derivation output redacts credential headers and common structured credential fields, but arbitrary unstructured bodies can still contain sensitive data. Load approved values at runtime from a secret manager, protected environment, or other project-approved source. HARs contain live secrets — restrict access and delete `out.har` after deriving.
- **`record_har_content="embed"` makes big HARs.** Use `--max-body` to cap what's printed; the file itself can be large for media-heavy pages.
- **Endpoints shift.** Sites change private APIs without notice. Re-run the capture→derive loop when a client breaks rather than patching URLs by hand.
- **Wrong capturer = empty/no HAR.** `har_capture.py` on a cloud/CDP backend records nothing (it launches its own local browser instead of the one you meant). `har_capture_cdp.py` needs the endpoint; on Hermes get it from `/browser connect` or `BROWSER_CDP_URL`. Match the capturer to the pathway (How to Run table).
- **Headless-Chrome UA is a weak tell.** Local/agent-browser capture yields a `HeadlessChrome/...` User-Agent; some sites sniff the "Headless" token. Cloud backends (Browserbase/Browser-Use) send a real desktop-Chrome UA, so a client derived from a cloud capture replays more reliably. If a headless-derived client 403s where the browser didn't, swap the "Headless" UA for a normal Chrome UA string before assuming the endpoint changed.
- **CDP capture doesn't close the browser.** `har_capture_cdp.py` attaches to a browser it doesn't own and leaves it running — correct for cloud/remote sessions Hermes manages. Don't add a close; let the owning backend tear it down.

## Verification

End-to-end proof against a live site with no API key:

```bash
python3 scripts/har_capture.py "https://en.wikipedia.org/wiki/Main_Page" /tmp/wiki.har \
  --action "fill:input[name=search]:dune messiah" --action "sleep:3" --wait 2
python3 scripts/har_to_client.py /tmp/wiki.har --host wikipedia.org --max-body 200
```

Expect the derivation to print `GET https://en.wikipedia.org/w/rest.php/v1/search/title`
with `q` and `limit` params and a JSON `pages` response — then replay it with the
Procedure snippet and confirm matching titles come back over plain HTTP.
