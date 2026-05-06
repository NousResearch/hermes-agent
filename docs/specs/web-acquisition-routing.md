# Web Acquisition Routing and Scrapling Pilot

Status: Phase 3 internal adapter implemented without global registration
Owner: Hermes Agent web tooling / research skills
Scope: documentation, optional-skill positioning, isolated runtime pilot scripts, static pilot verification, and an internal adapter only

## Decision

Scrapling is an optional `difficult_web_extract` fallback candidate. It is not a default Hermes web backend, not a replacement for `web_extract`, not a replacement for browser automation, and not a global MCP/tool.

The default acquisition ladder remains:

1. `web_search` for discovery.
2. `web_extract` for ordinary URL-to-markdown extraction.
3. Browser tools for interaction, screenshots, login flows, visual inspection, or full JavaScript workflows.
4. Scrapling only when a difficult extraction needs selector-driven, batch-oriented, or session-aware fallback behavior and the caller can provide a narrow target.

## Non-goals

Phase 0 explicitly does not:

- install Scrapling or browser dependencies;
- add Scrapling to the Hermes main virtualenv;
- modify `toolsets.py` or `_HERMES_CORE_TOOLS`;
- change the public `web_extract` schema;
- register a new top-level tool in `tools/registry.py`;
- add a new wake route before the route catalog supports it;
- promise universal anti-bot, CAPTCHA, paywall, or login bypass.

## When Scrapling may be considered

Use Scrapling as a pilot/fallback only when at least one condition is true:

- `web_extract` returns empty, boilerplate, or the wrong main content;
- the task needs CSS, XPath, text, or regex selection against a known page structure;
- the target is a batch of homogeneous pages such as announcements, products, listings, or jobs;
- a lightweight session/cookie flow is needed but a full browser agent would be too heavy;
- the page is lightly blocked and a controlled fallback is worth trying, without treating stealth mode as guaranteed bypass.

## When not to use Scrapling

Do not use Scrapling for:

- ordinary search or broad web reconnaissance;
- ordinary article summarization where `web_extract` works;
- visual verification, screenshots, page interaction, or complex browser state;
- authenticated/private data, paywalled content, explicit no-scrape contexts, or clear Terms-of-Service conflict;
- strong CAPTCHA, enterprise WAF, or adversarial anti-bot systems;
- default Hermes tool exposure.

## Runtime isolation

If a later phase installs Scrapling, it must use an isolated runtime, not the Hermes main environment.

Recommended runtime:

```text
~/.hermes/runtimes/scrapling/
```

Recommended Python on Hank's current machine:

```text
/Users/zhaopufan/.local/bin/python3.11
```

Do not install into:

```text
~/.hermes/hermes-agent/venv
```

Reason: the current Hermes venv uses Python 3.14, while Scrapling currently declares `requires-python >=3.10` and official classifiers through Python 3.13. Its browser/fingerprint stack can also pull Playwright/Patchright/browserforge dependencies that should not contaminate the core agent runtime.

## Fetcher selection policy

| Mode | Scrapling class | Use when | Default |
| --- | --- | --- | --- |
| Static | `Fetcher` / `FetcherSession` | Static HTML, selector extraction, fast homogeneous pages | Preferred first attempt |
| Dynamic | `DynamicFetcher` / dynamic sessions | JavaScript-rendered content, `wait_selector`, `network_idle` | Explicit only |
| Stealth | `StealthyFetcher` / stealth sessions | Light blocking after static/dynamic fail | Explicit fallback only |

Stealth mode must stay opt-in. It is a fallback, not a capability promise.

## Receipt contract

Future adapters or scripts should emit a structured JSON receipt:

```json
{
  "backend": "scrapling",
  "mode": "static|dynamic|stealth",
  "url": "https://example.com/page",
  "selector": ".article",
  "content": "...",
  "elapsed_ms": 1234,
  "fallback_reason": "web_extract_empty|selector_required|batch_homogeneous|light_block",
  "errors": []
}
```

Receipts must preserve enough evidence to audit what was fetched, how it was selected, which fallback was used, and why failures happened.

## Future implementation phases

### Phase 1: optional skill documentation

- Reposition `optional-skills/research/scrapling/SKILL.md` as a difficult extraction fallback skill.
- Remove broad claims around generic crawling and Cloudflare bypass.
- Keep setup commands isolated and explicit.
- Document receipt schema and fallback rules.

### Phase 2: isolated runtime pilot

Implemented pilot files:

```text
optional-skills/research/scrapling/requirements.txt
optional-skills/research/scrapling/scripts/setup_runtime.py
optional-skills/research/scrapling/scripts/scrapling_extract.py
tests/test_scrapling_optional_runtime.py
```

Acceptance criteria:

- runtime setup is repeatable through `setup_runtime.py`;
- `setup_runtime.py --dry-run` prints a JSON setup plan;
- missing Scrapling dependency returns a structured JSON error from `scrapling_extract.py`;
- no Scrapling dependency appears in Hermes main venv;
- browser assets/downloads install only when `--install-browsers` is explicitly requested.

Phase 2 pilot evidence captured on 2026-05-06:

- `pytest tests/test_scrapling_optional_runtime.py -q` passes with `7 passed`.
- `setup_runtime.py --dry-run` emits a parseable JSON setup plan and does not include `scrapling install` unless `--install-browsers` is requested.
- `setup_runtime.py` created/reused `/Users/zhaopufan/.hermes/runtimes/scrapling` through `/Users/zhaopufan/.local/bin/python3.11` with `browser_install_requested=false`.
- Hermes main runtime remains clean: `importlib.util.find_spec("scrapling") is None` in the main Hermes Python 3.14 venv.
- The isolated runtime can perform a static public-page selector extraction against `https://example.com` with selector `h1`, returning a structured Scrapling JSON receipt with no errors.
- Browser-backed dynamic/stealth assets remain out of scope for this validation pass; do not run `scrapling install` until a specific dynamic/stealth pilot case justifies it.

### Phase 3: internal adapter, no global registration

Implemented files:

```text
tools/web_acquire.py
tests/tools/test_web_acquire.py
```

The adapter calls the isolated runtime and normalizes receipts, but it does not call `registry.register(...)` and is not part of any public toolset.

Phase 3 adapter evidence captured on 2026-05-06:

- `pytest tests/tools/test_web_acquire.py tests/test_scrapling_optional_runtime.py -q` passes with `16 passed` after safety hardening.
- `tools/web_acquire.py` contains no `registry.register(...)` call.
- `difficult_web_extract("https://example.com", selector="h1", mode="static")` calls the isolated Scrapling runtime and returns a structured receipt containing `<h1>Example Domain</h1>` with `errors=[]` when URL safety allows the locally proxied public target.
- The adapter checks `tools.url_safety.is_safe_url()` before invoking Scrapling and returns `URLSafetyBlocked` without spawning subprocesses for unsafe targets.
- The adapter checks `tools.website_policy.check_website_access()` before invoking Scrapling and returns `WebsitePolicyBlocked` without spawning subprocesses for policy-blocked hosts.
- Hermes main runtime remains clean: `pip show scrapling` in the main Hermes venv reports package not found.

The adapter already reuses the existing safety layers:

```text
tools/url_safety.py
tools/website_policy.py
```

### Phase 4: route-linked fallback, only after pilot proof

Implemented route gate:

```text
difficult_web_extract
```

The route is task-named and resolves to `/bg`. It wakes the optional Scrapling skill only when installed or provided through wake metadata, appends `terminal`, `file`, `web`, and `skills`, and deliberately does not add the browser toolset.

Trigger policy:

- requires an extraction context such as `web_extract`, `extract`, `scrape`, `抽取`, or `抓取`;
- also requires a control/fallback signal such as selector/XPath/regex, batch homogeneous pages, `web_extract` empty/wrong content, session/cookie, or light anti-bot fallback;
- ordinary article summarization and browser-only tasks must not trigger it.

Do not name the route `scrapling`; routes should describe work, not vendor/library choices.

Do not add Scrapling to the Hermes main virtualenv, global MCP, public tool registry, or default toolsets.

## Kill gates

Pause or remove the pilot if any of these happen:

1. isolated Python 3.11 runtime setup is unstable;
2. Scrapling leaks dependencies into the Hermes main environment;
3. browser or Patchright install cost is too high;
4. fetches hang or ignore timeouts;
5. difficult-page results are not materially better than Firecrawl/Tavily/browser;
6. stealth mode creates compliance or risk ambiguity;
7. route integration creates noisy or wrong wakeups;
8. receipts are not auditable.

## Upgrade gates

Only consider raising Scrapling above B/pilot if all are true:

1. three to five real difficult extraction cases succeed;
2. `web_extract` clearly fails where Scrapling succeeds;
3. batch homogeneous extraction is faster or cleaner than browser automation;
4. receipt output is stable and auditable;
5. runtime isolation is proven;
6. any future route is first-class and tested;
7. stealth remains off by default;
8. docs continue to avoid universal bypass claims.
