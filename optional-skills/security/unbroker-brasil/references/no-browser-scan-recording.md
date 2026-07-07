# No-browser scan recording pattern

Use this when an unbroker batch/brief explicitly forbids browser use or browser tools are unavailable, but you still need to drain a scan batch and write the ledger accurately.

## Pattern

1. For each broker, run cheap read-only probes only:
   - `web_search` with `site:<broker-domain> "Full Name"` plus city/state or stable aliases.
   - Lightweight HTTP fetches of the broker home/search/opt-out pages when they are public.
2. Treat empty search-engine results as **inconclusive**, not `not_found`.
3. Record `not_found` only when the broker's own accessible search/opt-out flow returns an empty result set or a public result page clearly lacks a subject record.
4. Record `blocked` when the only remaining verification path requires one of:
   - browser/JS app execution with no server-side result set,
   - HTTP 403 / Cloudflare / DataDome / similar anti-bot gate,
   - JS/progress/recaptcha-style public-records funnels that echo the query but expose no verifiable result cards,
   - face-image upload or reverse-image search,
   - signup, credit card, payment, or account creation,
   - CAPTCHA/verification flow that cannot be attempted under the brief's constraints.
5. Do not record `not_found` from constructed-profile 404s or guessed slugs. A 404 usually means the URL pattern is wrong; in no-browser mode it is `blocked`/inconclusive unless the broker's own accessible search/opt-out flow returned an actual empty result set.
6. Evidence should include the exact probe type, queries, URLs checked, status/result summary, and the blocker. Example evidence shape:

```json
{
  "scanned_via": "web_search+lightweight_http",
  "queries": ["site:example.com subject name + city"],
  "checked_urls": ["https://example.com/opt-out"],
  "web_search_results": 0,
  "http_status": "403 Forbidden",
  "blocker": "Search/opt-out page blocked lightweight HTTP; browser was prohibited, so verification could not be completed."
}
```

## Common outcomes

- Reverse image brokers: if presence can only be checked by uploading a face/photo, record `blocked` unless the brief specifically authorizes and provides the image path.
- JS-only opt-out apps: a 200 response that is only an application shell is not evidence of absence; record `blocked` if no non-browser API/result set is visible.
- JS/progress public-records funnels (for example pages that show "your search is for X" with a progress bar/recaptcha assets) are query echoes, not result cards; record `blocked` unless a real result card corroborates the subject.
- HTTP 403 on the broker home/search page is `blocked`; include the status and that no usable indexed profile was found.
- Plain HTML search forms that accept POST and return an explicit broker-side empty message can support `not_found` (for example a white-pages search response saying "There is no match in our free White Pages database"). Capture the exact empty-result phrase in evidence.
- Payment-gated searches: do not sign up or provide payment; record `blocked` with `payment-gated search` as the blocker.

This preserves the state-machine invariant: **absence requires broker-side evidence; inability to verify is `blocked`, not `not_found`.**