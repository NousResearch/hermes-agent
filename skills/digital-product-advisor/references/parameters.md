# Parameter Model

The advisor should treat currency, location, and brand scope as first-class parameters.

## Canonical Request Schema

```json
{
  "product_category": "bluetooth_earbuds",
  "region": null,
  "city": null,
  "currency": null,
  "display_currency": null,
  "budget": {
    "min": null,
    "max": null,
    "currency": null
  },
  "brand_scope": {
    "mode": "open_market",
    "include": [],
    "exclude": []
  },
  "marketplaces": [],
  "purchase_channel": ["online"],
  "use_case": [],
  "phone_ecosystem": null,
  "must_have": [],
  "nice_to_have": [],
  "deal_breakers": [],
  "purchase_timeline": null,
  "link_policy": {
    "include_product_links": true,
    "preferred_link_types": ["official", "trusted_marketplace"],
    "allow_search_links_when_direct_unavailable": true
  }
}
```

## Product Link Rules

- Include product links in reports by default when the user is making a buying decision.
- Prefer official product pages and trusted marketplace links from the user's region.
- For China requests, useful link types include official brand pages, JD/Tmall flagship-store product pages, or clearly labeled JD/Tmall search links.
- Never fabricate exact product URLs. If a direct listing cannot be verified, use a search link and label it as `search link`.
- Avoid affiliate links unless the user explicitly asks for them and disclose that they are affiliate links.

## Brand Scope Modes

- `open_market`: compare all relevant brands.
- `include_only`: only compare listed brands.
- `exclude`: compare broadly but exclude listed brands.
- `brand_first`: user wants one brand; compare models within that brand.
- `brand_vs_brand`: user asks to compare a small set of brands.

## Currency Rules

- If region is China and currency is missing, infer CNY.
- If user says RMB or 元, infer CNY.
- If user says USD, EUR, GBP, JPY, etc., use that display currency.
- If source prices use different currencies, label them clearly.
- Avoid live exchange conversion unless needed; if converting, mention the rate date/source.

## Location Rules

- Region/country is enough for MVP.
- City is optional and mainly affects offline stores, shipping, and service availability.
- Marketplace hints can imply region: JD/Tmall/Pinduoduo imply China; Best Buy implies US/Canada; Amazon depends on domain.

## Clarification Limit

Ask no more than 3 questions before proceeding. Prefer assumptions when obvious, but state them.
