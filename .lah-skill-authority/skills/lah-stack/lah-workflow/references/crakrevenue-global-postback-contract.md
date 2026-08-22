# CrakRevenue Global S2S Postback Contract (LAH canonical)

Established: LAH_CRAKREVENUE_GLOBAL_POSTBACK_CANONICAL_AUDIT_V1 (2026-08-14, READ_ONLY audit, interrupted by operator before Phase 5 tests / final verdict).
Authority = runtime code, NOT prior assumptions. Candidate URLs in old docs are CANDIDATES, never authority.

## Endpoint + auth

- Route: `/lah-postback/affiliate` (GET or POST; else 405). Implemented in lah-core `wp-content/plugins/lah-core-router/includes/class-lah-conversion-endpoint.php` (`LAH_Conversion_Endpoint`). Registered on `init`; path compare via `LAH_Utils::current_request_path()`.
- Handler flow: normalize (`LAH_Utils::normalize_affiliate_conversion_request`) → auth (`validate_auth`) → `LAH_Conversion_Store::store_or_get` (dedup) → push to LAHB (`LAH_S2S_Push::push_conversion`) ONLY when created=true → `LAH_Reconciliation::reconcile` → `LAH_Exoclick_Postback::maybe_send_for_conversion` (outbound S2S).
- Auth HYBRID (class-lah-conversion-endpoint.php `validate_auth`):
  1. `X-LAH-SECRET` header — preferred; if present but wrong → 403 immediately, no fallback.
  2. Legacy `?secret=` query/body param — backwards compat, TODO removal (R4 risk: secret in logs/referrers).
  3. None → 401 `missing_secret`.
  - Validation: `LAH_Utils::validate_affiliate_postback_secret($network, $secret)` = `hash_equals` against WP option `lah_affiliate_postback_secrets[<network>]` (e.g. `['crakrevenue' => '<secret>']`). Never print the real secret; represent as `<POSTBACK_SECRET>` / `<REDACTED_POSTBACK_SECRET>`.

## Parameter semantics (class-lah-utils.php `normalize_affiliate_conversion_request`)

| URL param | Aliases accepted | Internal destination | Notes |
|---|---|---|---|
| network | — | network | REQUIRED; sanitize_key, max 80 |
| transaction_id | txid, trans_id | transaction_id | max 160; provider-side dedup identity |
| affiliate_click_id | aff_sub, subid, sub_id, click_id, clickid | affiliate_click_id | max 512. `aff_sub` IS the primary click identity (LAH's own click id round-trips here) |
| external_click_token | external_click_token, conversions_tracking (+ alias `ct` via normalize_common_ids) | external_click_token | ExoClick Conversion API token only |
| status | conversion_status, event | status | default `'conversion'` when empty. LAH-side constant, NOT proof of sale |
| payout | amount, revenue | payout | numeric → round(x,4); non-numeric → null (no error, NEVER 0) |
| currency | — | currency | max 8 |
| offer_id | — | offer_id | max 120; stored + transported to LAHB |
| goal_id | — | goal_id | max 120; stored + transported; CAN carry provider goal identity (e.g. CrakRevenue Goal.id 37170 "SOI") |
| campaign_id | aff_sub3, campaignid, campaign | campaign_id | max 64. CONFIRMED: aff_sub3 = campaign_id (comment: "Canonical CrakRevenue SubID contract (lah-runtime.js)") |
| zone_id | aff_sub2, zoneid, zone | zone_id | max 64. CONFIRMED: aff_sub2 = zone_id |
| site_id | siteid, site | site_id | max 64 |
| source | — | source | max 64, verbatim; `is_test=1` when source == `synthetic_e2e` or prefix `synthetic_e2e_` (deliberately NOT broad prefix — `synthetic_e2evil` must NOT match) |
| (implicit) | is_test | is_test | explicit `is_test=1` also preserved |

Identity requirement (400 `missing_identity` if none): at least one of transaction_id | affiliate_click_id | external_click_token.
Internal `event_type` is HARDCODED `'conversion'` at normalize — real classification happens at LAHB push (below).

## Event vs paid conversion — CRITICAL GATE

- `LAH_S2S_Push::build_sale_event` (class-lah-s2s-push.php): `payout > 0` → `event_type='sale'`, `event_semantics='PAID'`; `payout == 0` OR absent → `event_type='conversion'`, `event_semantics='UNVERIFIED'`; null NEVER coerced to 0.0.
- Outcome semantics contract (lah-workflow `references/affiliate-outcome-semantics-contract.md`), rule 3: `status=conversion` is a LAH-side constant, not proof of sale.
- LAHB: `projectCanonicalSaleEvent` requires `event_type==='sale'` AND `import_source==='lah_php_postback'` (main also checks `event_semantics==='PAID'`). UNVERIFIED rows are NOT projected into `affiliate_conversions` (the revenue ledger); zone contract paid_conversions/downstream_events reads only affiliate_conversions.
- Real-world proof (T03, 2026-08-13): the account's ONLY conversion in window was an APPROVED SOI signup (Goal 37170 "SOI", payout $0.00, cpa_flat) — a downstream intent signal, NOT a paid PPS sale. Provider sends it through the same callback. Payout is the only discriminator LAH has today.

## Dedup

- `LAH_Utils::conversion_dedupe_key`: `tx:{network}:{transaction_id}` (max 191) when BOTH network+transaction_id present; else `fb:` + md5 of {network, affiliate_click_id, external_click_token, offer_id, goal_id}.
- `LAH_Conversion_Store::store_or_get`: find_by_dedupe_key first; duplicate → `created=false` → NO LAHB push, response `duplicate:true` (HTTP 200).
- LAHB side: `INSERT OR IGNORE` on event_id (composite fallback import_source|timestamp|type|click_id|...|transaction_id|goal_id).

## Attribution (LAHB `src/money/reconciliation.js`)

Priority:
1. **direct_postback_dimensions (Mode B)** — conversion carries campaign_id+zone_id from the postback itself; NO prior LAH event needed. Sentinels `unknown-campaign`/`unknown-zone` never count. THIS is why the account-level GLOBAL postback matters: LAH dynamically selects offers, so per-offer postbacks are insufficient.
2. click_event (Mode A) — conversion.click_id → events.click_id (first-event anchor).
3. external_token_event — conversion.external_click_token → events.external_click_token.

PHP-side reconciliation (lah-core `class-lah-reconciliation.php`) matches affiliate_click_id / external_click_token against `wp2h_lah_clicks` (LAH_Click_Store); >1 candidate → `blocked/ambiguous_*`.

## Outbound → provider → postback → LAH round-trip map

Outbound injection (lah-core `wp-content/plugins/lah-landing-engine/assets/js/lah-runtime.js`, redirect builder ~L1443):
- clickId → `click_id`, `clickid`, `aff_sub`, `subPublisher`, `clicktag`
- zoneId → `zone_id`, `zoneid`, `aff_sub2`, `zone`
- campaignId → `campaign_id`, `campaignid`, `campaign`, `aff_sub3`
- URL macro replacement `{click_id}` / `{clickid}` also supported (encodeURIComponent).

Round trips:
- zone_id → aff_sub2 → `{aff_sub2}` → zone_id → attribution.zone_id (live-proven: CrakRevenue Stat.affiliate_info2 = zone, e.g. 5858478 / 5688510)
- campaign_id → aff_sub3 → `{aff_sub3}` → campaign_id → attribution.campaign_id (live-proven: Stat.affiliate_info3 filter, e.g. 8539232 == campaign)
- click id → aff_sub → `{aff_sub}` → affiliate_click_id → reconciliation match (or external_click_token via `ct=` ExoClick S2S)

## CONTRACT TENSION (documented 2026-08-14, NOT resolved)

- OpenClaw certified builder (`openclaw-runtime/lah-openclaw-mvp/src/services/crakrevenue-tracking-url-builder.js` + `canonical-tracking-context.js`): offer 10138 has exactly ONE free SubID slot, network param name `aff_sub1`; `aff_sub5` = PROHIBITED_RESERVED_BY_NETWORK (refused with error); SubID1 value pattern `[A-Za-z0-9_.-]`, max 64.
- Production landing runtime + LIVE campaign (t.vlmai-5.com/406295/7709) use multi-dim `aff_sub`/`aff_sub2`/`aff_sub3`/`aff_sub5` — extra slots network-injected (aff_sub5=SF_… smartlink funnel).
- The PHP postback does NOT read `aff_sub1` (not in the pick list) — only aff_sub/aff_sub2/aff_sub3/click_id/clickid etc. If a tracking URL uses aff_sub1 only, the `{aff_sub}` macro (SubID1) is what returns it — needs provider confirmation.
- Postback macros `{aff_sub}`, `{aff_sub2}`, `{aff_sub3}`: CONFIRMED usable (live report evidence, affiliate_info1..3 surface). `{transaction_id}`, `{payout}`, `{offer_id}`, `{goal_id}`, `{source}`, `{currency}`, `{offer_name}`, `{clickid}`: official macro list NOT in local repos → UNCONFIRMED; ask provider support (global postback context especially).
- Operator-accessible evidence path without API creds: CrakRevenue Statistics → Custom Parameters Report, dimension "Sub ID 2" (per aff_sub2).

## Multi-worktree authority pitfall (apply to ANY lah-brain/lah-core audit)

- `lah-brain-wt-deployed` (24d833c, branch fix/dual-funnel-observability-v1) was STALE: still had `normalizeEventType: conversion→sale`, `payout: toNumber(payout||revenue)`, NO event_semantics.
- `origin/main` (d9aff83) contains the outcome fix (PR #214 merge of `affiliate-outcome-synthetic-postback-v1`, 37245d5) AND the dual-funnel work.
- NEVER trust a worktree's file content by name ("wt-deployed" ≠ fresh). Resolve authority: `git merge-base --is-ancestor <fix_sha> origin/main` → YES means fix is in the deploy line. Check `git -C <repo> rev-parse origin/main` / `origin/HEAD` before quoting semantics.

## Redaction rule (operator-enforced 2026-08-14)

Files that may embed REAL postback secrets (openclaw `data/execution-receipts.json`, `data/cloe-governed-action-packets.json`, campaign-ledger.json, any URL with `secret=`) must be read ONLY through a redaction filter (`re.sub(r'(secret["\'=:\s]*)[A-Za-z0-9_\-\.]{6,}', r'\1<REDACTED_POSTBACK_SECRET>', s)`), never grepped/dumped raw. The operator BLOCKED a raw receipts grep and a subsequent read command mid-audit — stop and present status rather than pushing through after a denial.
