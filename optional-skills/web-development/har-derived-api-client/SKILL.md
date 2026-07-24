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
can call it directly with plain HTTP — far cheaper and faster than
browser-controlling the page on every request. Credit: trick by Jared Longster,
popularized by Dax (thdxr). This captures and replays; it does NOT bypass
auth, solve CAPTCHAs, or defeat bot-detection — if the site needs a logged-in
session, you carry its headers/cookies forward, you don't forge them.

The two scripts are stdlib-plus-Playwright: capture needs Playwright,
derivation is pure stdlib, replay needs only `requests`/`httpx` (or `curl`).

## When to Use

- "Build a CLI/client for <website>" — derive its API instead of scripting clicks.
- "This site has no public API but the page clearly fetches JSON."
- You're about to loop `browser_navigate` for the same query repeatedly — stop and derive the endpoint once.
- Reverse-engineering an autocomplete, search, feed, or checkout XHR.

## Prerequisites

- Playwright + a browser binary (capture step only):
  - `pip install playwright` then `playwright install chromium`
  - (If a system Playwright already has browsers under `~/.cache/ms-playwright`, reuse it.)
- `requests` or `httpx` for the replay step (stdlib `urllib` also works).
- No API keys. Any keys/tokens the client needs are the ones the HAR captured.

## How to Run

Two scripts under this skill's `scripts/`, both invoked through the `terminal` tool:

1. `har_capture.py` — launches Chromium, runs your scripted interactions, writes a HAR with request+response bodies embedded.
2. `har_to_client.py` — filters the HAR to XHR/fetch/JSON, groups by endpoint, and prints params, headers, bodies, and replay hints (User-Agent / cookie / auth).

Resolve paths against this skill's directory. Canonical loop:

```bash
# 1. Capture — trigger the interaction whose network call you want
python3 scripts/har_capture.py "https://SITE/" out.har \
  --action "fill:input[name=search]:my query" --action "sleep:3" --wait 2

# 2. Derive — read the endpoints out of the HAR
python3 scripts/har_to_client.py out.har --host SITE --max-body 400

# 3. Replay — write a tiny client from the printed endpoint (see Procedure)
```

## Quick Reference

```
har_capture.py <url> <out.har> [--wait S] [--headed] [--action SPEC ...]
  action SPEC:  fill:SELECTOR:TEXT | press:SELECTOR:KEY | click:SELECTOR
                goto:URL | sleep:SECONDS      (run in order after page load)

har_to_client.py <in.har> [--host SUBSTR] [--include-static] [--max-body N]
  default: keeps only XHR/fetch/JSON; --host narrows to one domain
  prints per endpoint: query params, non-boring req headers, req body sample,
                       response status/content-type + body sample
  prints "### Replay hints": the browser User-Agent, cookie/auth presence
```

## Procedure

1. **Find the interaction.** Open the site with `browser_navigate` (or `--headed` capture) to see which selector to type into / click, and confirm a JSON XHR fires in devtools/network.
2. **Capture the HAR** via the `terminal` tool. Order `--action` to reach the request: `fill` the box, then `sleep` long enough for the debounced XHR, and always leave `--wait` at the end so late responses flush. `har_capture.py` records response bodies (`record_har_content="embed"`), so the derived client sees real payload shapes.
3. **Derive** with `har_to_client.py --host <domain>`. Read off: the method, the URL/path template (numeric/UUID segments collapse to `{id}`), query params, request-body JSON, and the `### Replay hints` block.
4. **Write the client.** Recreate the request exactly — same method, path, query params, body. Send the headers the site actually needs: at minimum copy the **User-Agent** from the replay hints. If hints report cookies or an auth/token header, resend those too.
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
- **Auth/session endpoints** need the captured `Cookie`/`Authorization` header, and those expire. The derived client is only as durable as the credential; re-capture when it 401s. HARs contain live secrets — treat `out.har` as sensitive and delete it after deriving.
- **`record_har_content="embed"` makes big HARs.** Use `--max-body` to cap what's printed; the file itself can be large for media-heavy pages.
- **Endpoints shift.** Sites change private APIs without notice. Re-run the capture→derive loop when a client breaks rather than patching URLs by hand.

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
