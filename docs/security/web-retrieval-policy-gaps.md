# Web retrieval policy and reliability gaps

This note records gaps observed in the web search and extraction path. It is
an audit record, not an implementation proposal; each item needs an explicit
policy decision and tests before behavior changes.

## Confirmed gaps

1. The website blocklist is not enforced uniformly across extraction
   providers. `plugins/web/firecrawl/provider.py` calls
   `check_website_access`, but the Tavily, Exa, and Parallel extract providers
   do not. Changing `web.extract_backend` can therefore change whether the
   same configured policy is applied.
2. `web_search_tool` returns provider results without checking result URLs
   against the website blocklist. Search can expose a blocked domain even when
   a subsequent Firecrawl extraction would reject it.
3. There is no shared `robots.txt` preflight in the search/extraction path.
   Provider-specific behavior, if any, is not represented in Hermes' result.
4. The path has no shared per-domain rate limiter, bounded retry/backoff
   contract, `Retry-After` handling, or retrieval-response cache. Provider SDK
   behavior is therefore inconsistent and opaque to callers.
5. `_DEFAULT_WEBSITE_BLOCKLIST` always supplies `enabled: False`, making
   `policy.get("enabled", True)` in `load_website_blocklist` misleading: the
   fallback `True` is unreachable after the merge.
6. `check_website_access` returns `None` both when a URL is allowed and when
   policy evaluation fails open (for example malformed default configuration
   or a missing YAML dependency). Callers cannot distinguish permission from
   an unevaluated policy.

## Provenance gap

`web_extract_tool` trims provider results to URL, title, content, error, and an
optional `blocked_by_policy` marker. HTTP status, final redirect URL,
retrieval timestamp, content digest, provider metadata, and policy-evaluation
state do not survive the boundary. A downstream agent cannot make an auditable
claim about what was retrieved from those fields alone.

## Guardrails for follow-up work

- Apply one policy preflight before provider dispatch and repeat it for final
  redirect URLs.
- Preserve a typed distinction among allowed, blocked, and policy-not-evaluated.
- Decide and document robots behavior before adding any challenge-bypass or
  evasion-capable acquisition backend.
- Bound retries and browser escalation per source and per workflow.
- Return enough provenance for callers to independently validate evidence,
  without returning secrets or provider-private metadata.
