# Audit Rules

Use this reference before writing findings, scores, or fix plans.

## Severity Buckets

Use only these user-facing buckets:

- `Strengths`
- `Priority fixes`
- `Improve next`
- `Manual review`

Keep internal severity simple:

- `high`
- `medium`
- `manual`

## Locale Findings

Raise `high` when:

- a market language is not published
- a market main language is not enabled
- readiness is below `70%`

Raise `medium` when:

- readiness is from `70%` to `89%`
- more than `10%` of fields are stale
- a published language is not attached to any market

Raise `manual` when:

- the locale exists but there is not enough evidence to decide which market should own it

## Market Findings

Do not mark a market wrong only because it serves multiple countries or multiple languages.

Raise `medium` instead when:

- the market groups many countries and the language setup looks thin
- the market has no dedicated web presence
- the market uses one main language even though several large language groups are present

Raise `high` when:

- the market main language is unpublished
- the market references a language that is not enabled

## Shipping Findings

Raise `high` when:

- a market country has no shipping-zone coverage

Raise `manual` when:

- shipping profiles exist but mapping to the market is unclear

## Storefront Findings

Raise `high` when:

- no `hreflang` links are found on a localized store
- no canonical link is found

Raise `medium` when:

- policy links are missing from common store pages

Raise `manual` when:

- a theme-side international UX item needs human review

## Strategy Suggestions

Keep market expansion suggestions in plain language:

- show the current language count
- show the remaining language slots out of `20`
- suggest new markets only after considering what the store actually sells
- allow replacing weak markets with stronger ones, but treat that as a recommendation, not a fix

Before writing any category or country suggestion:

- identify the store type from `shop.name`, `shop.description`, collection names, visible product names, and at least one product-detail page when possible
- use at least `3` external references if the advice claims category-country opportunity
- prefer industry associations, government trade reports, or category research over generic blogs
- if a claim is seasonal or region-specific, say whether it is `evidence` or `inference`
- avoid default advice like `just add more languages` unless it is connected to a specific country demand pattern

The business suggestion block should answer:

- what this store mainly sells
- who is most likely to buy it
- which countries look strongest for this category
- why those countries fit this category
- what the merchant should do in Markets, languages, pricing, or policies because of that fit

Use this wording style:

- `You already cover 8 languages. Shopify allows up to 20 published extra languages. That leaves room for 12 more.`
- `This market is not broken, but it is carrying too many countries for one setup.`
- `People can reach the store, but they are not getting a clearly local experience yet.`
- `This market is not broken. It just has not been localized deeply yet.`
- `There is no obvious failure here, but there is still room to improve.`
- `As a manual follow-up, check whether the storefront has a country/language switcher and whether Shopify automatic redirection is enabled in admin.`
- `Because this store sells pet travel products, country prioritization should consider driving habits, small-home urban living, and demand for safety and portability instead of pet population alone.`

## Customer-Facing Report Rules

- Use short, plain-language sections for overall view, strengths, priority fixes, improve next, languages, markets, shipping, storefront, international growth ideas, and approval-only fixes.
- Render in the conversation language. `--lang auto` follows the machine locale; pass `zh-CN` or `en` when it differs from the conversation.
- Every issue explains what is wrong, why it matters, and what to do next.
- Prefer plain language over `webPresence`, `alternateLocales`, `translatableResource`, or `digest`. Explain URL strategy as shared root domain, subfolder, or separate domain/subdomain.
- Treat storefront selector visibility and automatic redirection as manual follow-ups, not precise automated findings.
- Tie international recommendations to the actual category, product use case, and buying context. Separate evidence from inference whenever external research informs the recommendation.

## Fix Plan Rules

The fix bundle must be short and reviewable.

Each action needs:

- `title`
- `summary`
- `risk`
- `mutation`
- `variables`
- `execute_supported`

If the action changes URL structure, countries inside a market, or translation content:

- do not include it in the executable bundle
- place it under guided next steps instead
