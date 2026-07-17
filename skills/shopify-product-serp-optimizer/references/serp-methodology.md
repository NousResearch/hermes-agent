# Product SERP Methodology

Use this reference before scanning, batching, scoring, reporting, or rewriting product SERP content fields.

## Core Frame

Optimize this chain:

```text
Search intent -> product evidence -> SERP promise -> safe Shopify fields -> distribution opportunity
```

Do not start from character counts. Start from whether the product page is the right target for a query and whether the page can prove the promise made in search results.

## SEO Category Boundary

This skill is product SERP optimization.

It includes:

- Single-product read-only HTML audit generation when the user names one explicit product.
- Read-only product scanning and five-product opportunity batches.
- Query intent and query class mapping for product pages.
- Product title, product description, SEO title, and meta description scoring.
- Evidence-backed candidates for Shopify `title`, `descriptionHtml`, `seo.title`, and `seo.description`.
- Image alt text assessment and direct in-skill optimization for product media.
- HTML reporting for product SERP, content, and distribution opportunities.
- Product-led content and community opportunity guidance.

It excludes:

- Technical SEO: canonical, indexability, performance, Core Web Vitals, rendered HTML, browser console, and theme source.
- Structured data repair: Product, Offer, Review, AggregateRating, Breadcrumb, FAQ, and ProductGroup JSON-LD.
- Search Console opportunity mapping.
- Merchant Center feed optimization.
- Automated backlink building or external posting.
- Redirects, translations, handles, theme code, reviews, ratings, prices, variants, tags, vendor, and collections.

## Opportunity Scoring

Use scoring to decide where to start, not to replace judgment.

Add priority when:

- Product is `ACTIVE` and has `onlineStoreUrl`.
- SEO title is missing, templated, vague, duplicated, or not aligned to product evidence.
- Meta description is missing, too generic, unsupported, or fails to address buyer intent.
- Product description has concrete evidence that the current title, description, or metadata does not use.
- Product description is thin but the product category supports clear micro-intents.
- Product media alt text is missing, repeated, overlong, or generic.
- The product is a natural product-page result, not a collection, guide, or comparison result.

Subtract or exclude when:

- Product is archived, unpublished, or lacks storefront visibility.
- The page needs handle, redirect, translation, schema, theme, review, rating, price, variant, tag, collection, vendor, or product copy edits.
- The product evidence is too thin to support claims.
- The available opportunity is mainly technical, informational, or collection-level.

Batch interpretation:

- Batch 1: highest-confidence, safest product SERP content opportunities.
- Batch 2: product SERP opportunities that also need content support.
- Batch 3+: lower certainty, thinner evidence, or opportunities needing merchant/external proof first.

## Query Class Taxonomy

Classify candidate queries before writing product SERP copy.

| Query class | Definition | Product page fit | Better target when not product |
| --- | --- | --- | --- |
| Product type | Names the exact product category, such as "carry-on laptop backpack" | High when the product is a strong match | Collection if broad or many products match |
| Attribute | Adds color, material, feature, size, or capacity, such as "water-resistant carry-on laptop backpack" | High when evidence exists | Collection/filter page if many products share it |
| Use case | Describes the situation, such as "laptop backpack for business trips" | Medium-high if page explains the use case | Blog/guide if mostly educational |
| Audience | Names the buyer or user, such as "for remote workers" | High when fit is supported by specs or copy | Collection if many products serve the same audience |
| Problem/solution | Names a specific pain, such as "backpack with padded laptop compartment" | Medium-high if the product directly solves it | Guide if it needs education or comparison |
| Brand | Includes brand name | High for owned products | Brand homepage or collection if query is broad |
| Model/SKU | Specific model, variant, SKU, or product name | High | Product page |
| Size/fit | Fit-specific query, such as "for 16-inch laptop" | High only with dimensions/compatibility evidence | Guide or size chart when evidence is incomplete |
| Material | Material-specific query, such as "nylon travel backpack" | High only when material is explicit | Collection/filter page |
| Occasion/season | Seasonal or occasion modifier | Medium | Collection or campaign page |
| Commercial investigation | "best", "review", "vs", "comparison", "alternative" | Usually low for a single product page | Blog, collection, comparison page |
| Informational | "how to", "what is", "guide", "tips" | Low | Blog/guide |
| Navigational | Store, brand, or known URL intent | Depends | Homepage, brand page, or product page |
| Transactional | "buy", "shop", "near me" | High when product is purchasable | Collection if query is broad |

If the target query is not product-page intent, do not force product-page copy changes. Recommend collection, blog, technical, structured-data, or search-opportunity work instead.

## Micro-Intent Expansion Ladder

Never stop at a broad theme. Expand from macro to micro until the target is concrete enough to support a product page, article, comparison page, or community post.

Use these layers:

| Layer | Question | Example |
| --- | --- | --- |
| Object | What is the product? | carry-on laptop backpack |
| Attribute | What visible or documented feature matters? | water-resistant carry-on laptop backpack |
| Use case | Where or when is it used? | carry-on laptop backpack for business trips |
| Constraint | What limitation shapes the search? | carry-on laptop backpack under airplane seat |
| Audience | Who is it for? | carry-on laptop backpack for remote workers |
| Problem | What specific pain does it solve? | carry-on laptop backpack with padded 16-inch device compartment |
| Comparison | What alternative is being evaluated? | carry-on laptop backpack vs rolling laptop bag |
| Purchase stage | What is the shopper trying to do now? | buy carry-on laptop backpack for 16-inch laptop |

Expansion rule:

```text
Broad theme -> object -> attribute -> use case -> constraint -> audience/problem -> comparison or purchase stage
```

Example:

```text
travel backpack
carry-on laptop backpack
carry-on laptop backpack for 16-inch laptop
carry-on laptop backpack under airplane seat
water-resistant carry-on laptop backpack for business trips
carry-on laptop backpack vs rolling laptop bag
```

Use the broad theme only as a starting point. The final recommendation must be specific enough to evaluate evidence and choose the right page type.

## Evidence Ledger

Separate supported facts from unsupported claims.

Supported evidence can come from:

- Product title and SEO fields.
- Product description and description HTML.
- Product type, vendor, tags, options, variants, SKU, price, and availability.
- Collections and product URL.
- Product media alt text and filenames.
- Verified review/rating data when available.
- Merchant-provided target query or positioning.
- GSC query evidence when the user provides it.

**In read-only (public) mode:** SEO fields come from HTML `<title>` and `<meta>` tags only (cannot distinguish custom vs fallback). Collection data is unavailable. Product status is inferred from `published_at`. Product URL is constructed from domain + handle.

Unsupported or high-risk claims include:

- "Best", "safest", "guaranteed", "certified", "doctor recommended", "eco-friendly", "non-toxic", "waterproof", "crash tested", "free shipping", and warranty or return claims unless directly evidenced.
- Ratings, review counts, certifications, and awards unless verified.
- Compatibility, size fit, material, performance, or compliance claims not present in product data.

## Live Intent Evidence Gate

Do not generate Enhanced snippets suggestions from memory alone.

Every FAQ direction, comparison direction, how-to direction, details/spec direction, and feature highlight must have all three evidence layers:

- merchant evidence from Shopify product data
- live Google search-intent evidence gathered during the current run
- live Amazon ecommerce user-intent evidence gathered during the current run

Use live Google evidence such as:

- autocomplete
- related searches
- People Also Ask
- wording patterns from current product, review, comparison, and how-to results

Use live Amazon evidence such as:

- autocomplete
- result-title wording
- bullet-point wording
- Compare With Similar Items patterns when available
- recurring Q&A themes
- recurring review themes

If one of those evidence layers is missing, do not guess. Mark the item as blocked by missing evidence.

FAQ note:

- As of April 30, 2026, Google FAQ rich results are generally limited to health or government sites.
- Therefore, FAQ suggestions in this skill are content and snippet-support recommendations, not a promise of FAQ rich-result eligibility.

## Shopify SEO Fallback Rules

Shopify SEO fields can be empty while the storefront still has effective search metadata.

Use these rules before auditing:

| Shopify field | If null or empty | Audit behavior |
| --- | --- | --- |
| `seo.title` | Falls back to `product.title` | Score the product title as the current effective SEO title. |
| `seo.description` | Falls back to the first 155 characters of `product.description` | Score that fallback text as the current effective meta description. |

Do not write "missing SEO title" or "missing meta description" in the report just because the API field is null. Say that Shopify is using the default product title or product description fallback, then judge whether the fallback is good enough.

**In read-only (public) mode:** The fallback rules above cannot be resolved because the HTML always shows the effective value. Treat SEO title and meta description values as `"public_html_effective"` — do not claim they are custom SEO fields or Shopify fallbacks. Score the content itself (is it weak? generic? misaligned?) rather than the source.

## SEO Title Rubric

Score out of 100.

| Criterion | Points | Rule |
| --- | ---: | --- |
| Product-page intent fit | 20 | Title targets a query class suitable for the product page. |
| Specific product identity | 20 | Names the product type clearly enough for shoppers and search engines. |
| Evidence-backed qualifier | 20 | Includes only supported attributes, audience, use case, brand, model, or fit. |
| Uniqueness | 15 | Avoids duplicate boilerplate across products. |
| Readability | 15 | Reads naturally and avoids awkward keyword stacking. |
| Risk control | 10 | Avoids unsupported superlatives, compliance claims, and URL/handle pressure. |

Use character length only as a soft warning. A shorter title that misses intent is not better than a slightly longer title that is clear and accurate.

Good title patterns:

```text
[Product type] for [audience/use case] | [Brand]
[Attribute] [product type] for [specific use] | [Brand]
[Model/product line] [product type] with [supported qualifier] | [Brand]
```

Avoid:

- Keyword lists.
- Repeating the same brand boilerplate on every product.
- Claims that are not visible in the product page evidence.
- Collection-level head terms when the product is only one option.
- Broad topics that should become guides or collections.

## Meta Description Rubric

Score out of 100.

| Criterion | Points | Rule |
| --- | ---: | --- |
| Accurate page summary | 20 | Describes the actual product page, not a category or guide. |
| Search intent alignment | 20 | Speaks to the buyer problem or use case behind the target query. |
| Evidence-backed benefits | 25 | Uses only facts supported by product evidence. |
| Differentiation | 15 | Adds a meaningful reason to click beyond the product name. |
| Natural language | 10 | Reads like a useful snippet, not a keyword list. |
| Risk control | 10 | Avoids unsupported superlatives, shipping promises, review claims, and compliance language. |

Good meta pattern:

```text
[Product type] for [audience/use case], with [supported feature 1] and [supported feature 2]. Ideal for [specific scenario].
```

Meta descriptions are not guaranteed to be shown in Google results. Treat them as a SERP promise and page summary, not a ranking loophole.

## Product Description Gap Matrix

Use this as both an audit matrix and a rewrite boundary. Description changes are allowed only when they stay evidence-backed, improve product-page intent fit, and are approved together with the rest of the product bundle.

| Buyer question | Evidence to look for | Recommendation when missing |
| --- | --- | --- |
| What is it? | Clear product type and use case | Add a concise opening sentence. |
| Who is it for? | Audience, size, fit, use case | Add fit guidance or size/compatibility notes. |
| Why this one? | Specific material, function, design, proof | Add supported differentiators, not generic quality claims. |
| Will it fit/work? | Dimensions, variant options, compatibility | Add size chart, compatibility note, or "not sure" guidance. |
| How do I use it? | Installation, care, setup, operation | Add use/care section or FAQ. |
| Can I trust it? | Reviews, warranty, returns, delivery, brand proof | Add verified trust signals or policy links. |
| What else should I compare? | Related products, collection links, guides | Suggest internal links with descriptive anchors. |

If product evidence is weak, recommend content improvements instead of writing aggressive copy.

## Product Title And Description Rewrite Rules

When rewriting `title` or `descriptionHtml`:

- Preserve the existing handle. Do not change the URL slug as part of title optimization.
- Prefer clearer product identity, audience, use case, and supported qualifiers over keyword repetition.
- Keep the opening description content useful to buyers first, then search snippets second.
- Do not inject unsupported claims, certifications, shipping promises, review claims, or legal/compliance language.
- Treat title, description, SEO title, meta description, and alt text as one review bundle. Show all recommended fields together before asking for approval.

## Image Alt Text Boundary

Use alt text for image understanding and accessibility, not keyword stuffing.

Good alt pattern:

```text
[Visible object] + [specific attribute] + [context/use]
```

Use this skill's built-in product-image alt text workflow when:

- product-media alt text is missing, repetitive, generic, or overlong
- the active model can inspect product images directly
- a context-only fallback is needed because direct image inspection failed

Vision is optional for the overall SERP skill. Without a vision model, the skill may still audit and optimize product titles, descriptions, SEO fields, and context-only Alt Text candidates at lower confidence. It must never claim image understanding or mark a candidate as `source: "vision"` without current-run pixel evidence.

Do not claim image understanding unless the image workflow actually inspected pixels during the current run.

Read `alt-text-rules.md` before generating or reviewing alt text candidates.

## Enhanced Snippets Module

Generate one Enhanced snippets module per product with exactly these five sections:

- FAQ directions
- comparison directions and comparison axes
- how-to directions
- details and specs directions
- feature highlight directions

Hard rules:

- Do not guess.
- Do not create any item without merchant evidence plus live Google plus live Amazon evidence.
- If evidence is incomplete, report the gap instead of filling it with generic ecommerce assumptions.
- Prefer narrow, high-intent directions over broad content buckets.
- Keep the module tied to product-page, comparison-page, or support-content reality. Do not imply that every direction belongs on the product page itself.

Required framing by section:

- FAQ directions:
  - surface the highest-friction buyer questions
  - answer only with facts the merchant can support now
  - do not promise FAQ rich-result eligibility
- Comparison directions:
  - name the real comparison target or comparison class
  - specify the concrete axes users actually compare, such as dimensions, fit, material, speed, compatibility, quantity, warranty, maintenance, or setup complexity
  - do not invent benchmarks or superiority claims
- How-to directions:
  - focus on real user tasks before purchase, during setup, or during usage
  - tie each direction to a visible or documented product capability
- Details and specs directions:
  - surface technical attributes buyers repeatedly need, such as dimensions, materials, capacity, compatibility, wattage, ingredients, included parts, or care instructions
  - mark absent data as a merchant-content gap instead of filling it in
- Feature highlight directions:
  - list only substantiated features
  - connect each feature to buyer value
  - do not convert a mere adjective into a feature without evidence

## Blog And Article Opportunity Method

Blog topics are not random content ideas. They must map to a search intent that is too broad or too educational for the product page, then link back to the product with a natural anchor.

Use these clusters:

| Cluster | Fit | Required narrowing |
| --- | --- | --- |
| Best | Commercial investigation | Add audience, constraint, price band, use case, or compatibility. |
| Comparison/alternative | Shopper compares substitutes | Name the exact alternatives and the decision criteria. |
| Informational | Buyer needs education | Narrow by use case, constraint, setup, care, or buying stage. |
| Problem/solution | Buyer has a concrete pain | Name the pain, the situation, and what evidence is needed. |
| Compatibility | Buyer needs fit confidence | Name the object, size, device, room, model, material, or environment. |

Every topic must include:

- Target intent.
- Why this product supports the topic.
- Suggested title.
- Target reader.
- Internal link anchor.
- Risk or evidence needed.

Bad topic:

```text
How to choose a backpack
```

Better topic path:

```text
How to choose a carry-on laptop backpack for a 16-inch laptop
How to pack a carry-on laptop backpack for a two-day business trip
Carry-on laptop backpack vs rolling laptop bag for short flights
Best carry-on laptop backpack for under-seat storage
```

## Distribution And Community Opportunity Method

This is distribution and off-page opportunity guidance, not automated backlink building.

For each product, suggest:

- Community types instead of invented exact opportunities when no live web search was run.
- Reddit search operators, such as `site:reddit.com "carry-on laptop backpack under airplane seat"`.
- Blogger search operators, such as `"carry-on laptop backpack" "review"` or `"business travel packing" "gear list"`.
- Facebook group search operators, such as `"business travel packing" "laptop backpack"`.
- Post angle: ask a specific, useful question or share a real experience before mentioning a product.
- Reply angle: solve the user's problem first, then mention a guide or product only if relevant.
- Outreach angle: explain why the product fits the creator's audience and offer useful evidence, photos, or a sample policy.

Boundaries:

- Do not fabricate backlinks.
- Do not recommend spam comments, mass posting, or hidden link schemes.
- Do not encourage review manipulation.
- Do not promise rankings.
- Do not automate external posting.
- Mark external research as "not live-verified" unless a separate web search was actually run.

## HTML Report Structure

The final report should be a single-file HTML artifact with:

- Beginner-friendly summary page with store, total products, audited products, average SEO score, estimated improvement percentage, and key takeaways.
- One independent `section` per product.
- SERP score.
- Current product title, product description, SEO title, and meta description.
- Recommended product title, product description, SEO title, and meta description.
- Evidence ledger.
- Micro-intent expansion ladder.
- Content gap and buyer objection matrix.
- Enhanced snippets suggestions with the five required sections.
- Blog/article opportunity map.
- Community, Reddit, blogger, and Facebook group direction.

Layout constraints:

- Use bento-style cards without nesting cards inside cards.
- Use `grid-template-columns: repeat(auto-fit, minmax(260px, 1fr))` or similar.
- Use `minmax(0, 1fr)` where long text might appear.
- Use `overflow-wrap:anywhere` on content blocks.
- Do not use one-line ellipsis for titles or meta descriptions.
- Keep each product page printable with `break-after: page`.
- Use a light editorial / boutique audit aesthetic, not generic dashboard decoration.
- Use emoji as section markers only; text labels must carry the meaning.
- Do not show a long technical boundary section on the summary page.
- Do not show a final protected-fields or "do-not-touch" card in each product page. Keep protected-field rules in the skill behavior, not in the merchant-facing report.
- Localize all static report labels to the user's language when `language`, `locale`, or `userLanguage` is known. The agent should also provide product analysis text in that same language.
- Include a built-in zero-dependency `Export as PDF` button that calls the browser's native print flow and hides itself in print mode.
- Generate every report from `assets/report-template.html`; do not hand-write or substitute a custom HTML report when the template is available.
- Before considering the report complete, verify the generated HTML contains:
  - `class="export-button"`
  - `onclick="window.print()"`
  - `@media print`
- If any marker is missing, regenerate the report before delivery.

## Stop Conditions

Recommend no product SERP copy change when:

- Current SEO title and meta description score 80+ and no query evidence suggests a problem.
- The target query belongs to collection, blog, comparison, technical, or structured-data intent.
- The product page lacks evidence for the proposed claim.
- The only improvement is cosmetic wordsmithing.

Do not generate Enhanced snippets items when:

- live Google evidence was not gathered
- live Amazon evidence was not gathered
- the item would require guessing missing product specs or unsupported buyer claims

## Delivery Rule

When this skill is triggered:

- Treat the HTML audit report as the primary deliverable.
- Generate the report in the same turn by default.
- Do not stop after giving chat-only title/meta suggestions.
- Do not ask whether the user wants the report unless file creation is actually blocked.
- If Shopify access is unavailable:
  - If a product URL is available, use **read-only (public storefront) mode** to extract full product data from public JSON + HTML scraping per `references/public-data-extraction.md`.
  - If only a product title is available, generate a provisional read-only report from merchant-visible product wording plus live Google and Amazon evidence, and label the evidence limits clearly.
- For broader requests, generate the report for the current analyzed batch rather than waiting for all future batches.

Stop completely before writes when:

- A change would require handle, redirect, theme, translation, schema, review, rating, price, variant, tag, vendor, or collection edits beyond the approved SERP content bundle.
- The user asks for full-store processing in one pass.
- Product availability or status makes storefront targeting unclear.

## References

- Google title link best practices: https://developers.google.com/search/docs/appearance/title-link
- Google snippets and meta descriptions: https://developers.google.com/search/docs/appearance/snippet
- Google image SEO: https://developers.google.com/search/docs/appearance/google-images
- Google link best practices: https://developers.google.com/search/docs/crawling-indexing/links-crawlable
- Google helpful content guidance: https://developers.google.com/search/docs/fundamentals/creating-helpful-content
- Google structured data policies: https://developers.google.com/search/docs/appearance/structured-data/sd-policies
- Shopify SEO overview: https://help.shopify.com/en/manual/promoting-marketing/seo
- Shopify product SEO field API: https://shopify.dev/docs/api/admin-graphql/latest/input-objects/SEOInput
- Shopify product update API: https://shopify.dev/docs/api/admin-graphql/latest/mutations/productUpdate
- Shopify file update API: https://shopify.dev/docs/api/admin-graphql/latest/mutations/fileUpdate
- Ahrefs ecommerce product page SEO: https://ahrefs.com/blog/ecommerce-product-page-seo/
- Semrush ecommerce keyword research: https://www.semrush.com/blog/ecommerce-keyword-research/
- Moz keyword research: https://moz.com/beginners-guide-to-seo/keyword-research
- Yoast product page SEO: https://yoast.com/product-page-seo/
- Backlinko ecommerce SEO: https://backlinko.com/ecommerce-seo
- Baymard product descriptions: https://baymard.com/blog/product-descriptions
