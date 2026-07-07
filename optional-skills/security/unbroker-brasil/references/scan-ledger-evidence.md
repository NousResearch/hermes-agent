# Scan ledger evidence patterns

Use this when recording read-only discovery outcomes from batch scans, especially anti-bot-heavy people-search sites.

## Anti-bot blocked scans

When a broker's search/results page returns a real anti-bot wall (Cloudflare, DataDome, managed challenge) and no stealth/operator browser is available in the current run, record `blocked` rather than guessing `not_found`.

Recommended ledger shape:

```bash
$PDD record <subject> <broker> blocked --evidence '{
  "scanned_via":"terminal_http|web_extract|browser|scrapling",
  "search_urls":["https://..."],
  "result":"blocked_by_cloudflare|blocked_by_antibot",
  "http_status":403,
  "notes":"Search/results page reached an anti-bot wall; no listing verification possible without stealth/operator browser."
}'
```

Do **not** record `not_found` from an anti-bot wall, a 403, a device-check page, or a constructed URL returning 404.

## Recording `not_found`

Only record `not_found` after an actual search path returns an empty or non-matching result set. For people-search sites, combine at least two independent signals when possible:

- an on-site or canonical search/result page that renders real results, and
- an exact `site:`/web-search probe or exact-name URL probe.

Evidence should include:

```json
{
  "scanned_via": "terminal_http|web_extract|browser|operator_browser",
  "search_urls": ["..."],
  "result": "not_found",
  "notes": "Describe the rendered result set and why it does not match subject anchors: surname, city/state, known address, phone, relatives, or aliases."
}
```

If broad results render but all are namesakes, say so explicitly in `notes`; include the distinguishing anchors that ruled them out.

## Verify after recording

After every batch ledger write, read back both per-case and aggregate state:

```bash
$PDD show <subject> <broker>
$PDD status <subject>
```

This catches JSON/evidence mistakes immediately and confirms the batch count changed as expected.
