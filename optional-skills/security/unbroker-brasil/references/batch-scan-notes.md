# Batch scan notes

Use this reference when a user asks to execute a specific unbroker scan batch or a delegated brief, especially when the task is read-only discovery and the deliverable is ledger updates.

## Operator pattern

1. From the skill directory, set `PDD="python3 scripts/pdd.py"` mentally or in shell.
2. Inspect the batch with:
   - `$PDD plan <subject> --batch` for reduce/grouping context.
   - `$PDD plan <subject>` when you need the per-broker `search_vectors`, method, and disclosure fields.
3. For each broker in the brief, execute discovery only unless the brief explicitly says to opt out.
4. Record every broker with `$PDD record <subject> <broker> <state> --found <true|false> --evidence '<JSON>'`.
5. Verify the write with `$PDD show <subject> <broker>` before reporting completion.

## State-selection notes from field runs

- Use `blocked`, not `not_found`, when verification cannot be completed because the broker requires account login, membership/trial signup, image upload, government ID/selfie, hard CAPTCHA, or a browser/session that is not currently available.
- In browser-disallowed scan batches, an empty `site:`/broad web-search result plus blocked/inconclusive direct fetches is **not** enough for `not_found`; record `blocked` with the attempted queries/URLs and the exact gate instead. This preserves the case for a later browser/stealth/operator pass.
- Use `not_found` only after a reachable search path returns results that can be disambiguated against the dossier/search vectors, or after a broker-owned opt-out/search flow returns no match.
- Record `listing_urls: []` for blocked/not-found scans unless a real candidate listing URL was inspected. Do not invent listing URLs from search templates.
- Keep evidence concise and structured: include `scanned_via`, `queries` or `search_queries`, `attempted_urls` when relevant, `listing_urls`, `result`, and a short `reason`.
- For people-search pages with namesake results, mention the disambiguation basis in `reason` (e.g. wrong city/state/country/relatives/phone/address).

## Useful site quirks

- FreePeopleDirectory home search constructs name URLs client-side as `/name/<First>-<Last>` with title-cased tokens and optional `/STATE/City` suffix for U.S. locations. A reachable exact-name page may still list U.S. namesakes; verify match basis before marking `found`.
- FaceCheck is reverse-image based. Name/phone/address probes alone cannot verify a listing; without an authorized subject image and an allowed upload/removal flow, record `blocked` rather than guessing.
- Historical-record/account sites such as Archives/Classmates/FamilySearch often require login, membership, or account-backed search to verify identity matches. If the brief is discovery-only and no account/browser is available, record `blocked` with the account/session gate as the reason.
