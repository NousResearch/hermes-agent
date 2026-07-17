# API Surfaces

Use this reference before interpreting data or proposing fixes.

## Admin GraphQL Surfaces Used In V1

Official references verified against Shopify Admin API `2026-04`:

- `shopLocales`
- `availableLocales`
- `shopLocaleEnable`
- `shopLocaleUpdate`
- `markets`
- `marketUpdate`
- `marketWebPresenceUpdate`
- `translatableResources`
- `translationsRegister`
- `deliveryProfiles`
- `shop`

## Locale Readiness

Use:

- `shopLocales` for enabled and published locale state
- `translatableResources` with `translations(locale: "...")` for full coverage counts

V1 coverage resource types:

- `PRODUCT`
- `COLLECTION`
- `PAGE`
- `BLOG`
- `ARTICLE`
- `SHOP`
- `SHOP_POLICY`

Why this list:

- It covers the core store copy most merchants care about.
- It is reviewable.
- It avoids inflating the score with technical resource types that merchants never read directly.

Coverage math:

- `eligible`: text fields counted after skipping `handle` and non-text payloads
- `current`: translation exists and `outdated` is `false`
- `outdated`: translation exists and `outdated` is `true`
- `missing`: no translation found

Readiness score:

```text
current / eligible
```

Gap score:

```text
(missing + outdated) / eligible
```

Use complete `translatableResources` pagination; do not sample. This is a field-based Shopify API score, not a translation-app coverage score. If a merchant compares the figures, explain the denominator: this skill counts current text fields / eligible text fields, while an app may count items or use a snapshot-based denominator.

## Query Transport And Rate Limits

The Admin GraphQL API is rate-limited by calculated query cost, not by a fixed request count. The helper therefore:

- batches up to three locale translation selections into one standard query;
- reads Shopify's returned throttle status when available and waits only when capacity is low;
- retries throttling and transient network failures with exponential backoff and jitter;
- uses Shopify CLI `store bulk execute` automatically when auditing four or more locales, and allows `--transport bulk` for very large single-locale audits;
- streams JSONL records directly into coverage counters rather than loading a whole bulk export into memory;
- runs at most three bulk jobs concurrently for large audits and checkpoints each completed resource type beside the requested audit output.

Bulk queries are read-only and use the same Shopify CLI stored OAuth authentication. Shopify currently permits up to five concurrent bulk queries per shop; this helper deliberately caps itself at three so it remains responsive alongside merchant workflows. A bulk query is not automatically resubmitted after a CLI watch failure: it may already be running, so first inspect it with `shopify store bulk status --store <store>.myshopify.com`. If an audit stops after one or more resource types finish, repeat the same command with `--resume` and the same `--output` file. Use `--transport standard` only to troubleshoot an unsupported bulk query; it is intentionally slower for large audits.

## Markets

Use `markets` to read:

- basic market state
- regions
- currency settings
- web presence
- default and alternate locales

Use `marketUpdate` for:

- enabling local currencies through `currencySettings`

Use `marketWebPresenceUpdate` for:

- changing the market main language
- adding a market language

V1 does not auto-create a market web presence because the subfolder or domain choice needs human review.

## Shipping Coverage

Use `deliveryProfiles` and each profile's location groups and zone country coverage.

Interpretation rule:

- If a market country does not appear in any shipping zone, call it a shipping gap.
- If shipping zones exist but the mapping is ambiguous, mark it as `check by hand`.

This is a shipping profile audit. It does not fully represent every fulfillment app or backend flow.

## Storefront Checks

Use storefront crawling for:

- `hreflang` alternate links
- canonical link
- policy-page visibility

Use admin data for:

- policy content existence
- market and locale state

## Automatic Redirection

Shopify Help Center documents automatic redirection in `Online Store > Preferences`, but V1 does not rely on a documented Admin GraphQL field for this setting.

Rule:

- do not audit or score this setting
- do not claim a precise API-backed on/off value unless Shopify documents a stable field for it
- mention it only as a manual reminder in the final report

## Selector Detection

Do not audit or score selector visibility in this skill.

Reason:

- selector visibility is theme-dependent
- selectors may be rendered dynamically
- static HTML alone is not reliable enough for a strong pass/fail result

Mention selector checks only as a low-priority manual reminder in the report.

## Safe Mutations In V1

Allowed after approval:

- `shopLocaleEnable`
- `shopLocaleUpdate`
- `marketUpdate`

Not auto-applied in V1:

- `marketCreate`
- `marketRegionsCreate`
- `marketWebPresenceCreate`
- `marketWebPresenceUpdate`
- `translationsRegister`

Those can still be suggested in the report as next steps.
