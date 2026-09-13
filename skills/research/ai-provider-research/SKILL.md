---
name: ai-provider-research
description: "Use when comparing AI providers, prices, or modalities."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [AI, LLM, providers, pricing, modalities, vision, EU, GDPR, API]
    related_skills: [research-intelligence-workflows, model-ops-workflows]
---

# AI Provider Research

## Overview

Use this class-level skill when a user wants a current comparison of AI inference providers or gateway routes: price, model availability, EU hosting/residency, privacy, image or other modality support, API compatibility, and operational or contractual modes. Produce a decision-ready comparison grounded in live provider/gateway metadata, not a generic model description.

A provider route is not the same thing as the model laboratory. Keep separate:

- **Model lab:** the organization that created the model.
- **Provider/host:** the service actually serving inference.
- **Gateway:** the service through which the user sends requests and that may add markup, routing, caching, or failover.

For Discord/mobile delivery, use compact bullets and provider cards; avoid wide Markdown tables.

## Research Workflow

1. **Define the comparison unit.** Record the exact model ID, provider route, gateway (if any), currency, and the meaning of “European” (inference/hosting region, not merely company headquarters).
2. **Honor source-scope restrictions literally.** If the request says “official only,” use only first-party provider, cloud, product, pricing, documentation, legal, and status sources as evidence. Search engines may discover URLs, but do not cite search-result pages, aggregators, benchmarks, reseller pages, or third-party price lists. If an official source does not publish a requested fact, report it as **not publicly documented**, rather than filling the gap with a third-party estimate.
3. **Query live structured sources first.** Prefer the official provider model catalog/API and the gateway’s live `/v1/models` endpoint. Use official model pages and pricing/docs to explain fields or fill in operational details. Use search only to discover URLs, never as the final evidence when a direct source is available.
4. **Capture source metadata.** Record retrieval timestamp (with timezone), exact URL/API endpoint, final redirected URL, page/document update date when available, model ID, provider, geolocation, pricing units, effective-date language, and relevant capability fields. Treat model pages, JSON-LD, embedded pricing JSON, and API records as potentially different snapshots and call out material discrepancies. For Requesty, preserve the full route ID—including provider namespace, API namespace, and `@region` suffix—and query the live `/v1/models` catalog before relying on a rendered model page.
5. **Extract dynamic pricing pages, not just rendered text.** Azure/Microsoft pricing pages may render `$-` in the visible HTML while embedding actual regional values in `data-amount` JSON attributes. Parse the pricing table rows, deployment scope, context length, token-meter labels, and region key from the page source; preserve the exact official URL and retrieval date. Check both the standard/realtime column and any priority/batch column, and distinguish `N/A` from an omitted field.
6. **Normalize prices.** Put input, output, cached-input, and—when published—cache-write prices on a common basis of USD/EUR per 1M tokens. Separately show direct-provider/base rates and gateway customer rates. Apply gateway markup only when the gateway states it, and label BYOK pricing separately. Use a tool for all arithmetic and show one or two representative workloads (for example 100k input + 10k output). For cloud deployment comparisons, keep Global, Data Zone, and Regional/Standard rows separate; calculate any Data Zone uplift only from same-model, same-context rows and label it as an observed price difference, not a contractual surcharge unless the vendor says so.
7. **Compare routes, not just brands.** The same model may have global, Singapore, US, and EU routes with different prices, retention, or capabilities. A cheaper route is not an equivalent substitute if the user needs EU residency.
8. **Extract modalities from explicit fields.** Check `supports_vision` or `image_input.supported` for image input, plus PDF, video, audio, image generation, tool calling, reasoning, caching, structured output, and computer-use fields where available. Absence means “not verified,” not automatically “unsupported.” Prefer model-specific metadata over generic gateway feature pages.
9. **Check operating modes and terms.** Distinguish serverless per-token realtime, flex/low-priority processing, async or batch windows, dedicated GPU-hour deployments, enterprise commitments, minimum spend, SLA, and support. Mark customer-specific pricing as undisclosed rather than estimating it.
10. **Check privacy and residency claims carefully.** Separate a live route geolocation field from a vendor’s general infrastructure statement, and separate both from the provider’s legal domicile. Record retention, training use, DPA/GDPR statements, and whether the evidence applies to the exact route.
11. **Handle future effective dates conservatively.** A live page retrieved before a requested future date is a current snapshot, not proof of the price that will apply on that date. Look for an official effective date, announcement, or dated pricing notice. If none exists, provide the current snapshot and explicitly mark the future-date price as unconfirmed; do not forecast or backdate it.
12. **State uncertainty and verify high-risk capabilities.** If a generic docs page names only some vision/video models while the live catalog says another model supports vision, report the exact evidence and recommend a small image smoke test before production. Do not silently promote “not listed” into “not supported.”

## Requesty Live-Catalog Recipe

For a Requesty-only pricing or regionality audit, use this evidence order:

1. Fetch `https://router.requesty.ai/v1/models` without authentication and filter exact model IDs; record HTTP status, retrieval timestamp, `id`, `pricing`, `context_window`, capabilities, and `geolocation`.
2. Fetch the matching `www.requesty.ai/models/...` page. Treat its “Provider prices” section and JSON-LD offers as the rendered price snapshot; record the page’s stated provider-price update date.
3. Fetch `https://www.requesty.ai/pricing` for the gateway billing rule. Keep upstream/provider rates separate from the customer price: Requesty may state a PAYG markup and a different BYOK rule.
4. Fetch the relevant `docs.requesty.ai` API and EU-routing pages to interpret endpoint scope and residency. An EU Requesty router location is not proof of EU model inference; require the route’s own `geolocation` and region suffix.

Interpret Requesty’s common token fields using the model page labels: `input_price` and `output_price` are per-token equivalents of Input/Output per 1M; `cached_price` is cache-read input; `caching_price` is cache-write input when present. A `pricing` element with `prompt_tokens_threshold > 0` is a long-context tier. If a route has only a threshold-0 element, report “no separate long-context tier exposed” rather than inventing one.

When a page says “No markup on provider prices” but its detailed pricing disclosure says PAYG adds a percentage, quote both and flag the wording conflict. Do not silently treat the provider-rate display as the final PAYG bill. See `references/requesty-live-catalog-evidence.md` for the reusable field map and evidence template.

## Direct Provider vs Gateway

Always report both when relevant:

- **Direct provider:** exact base URL/model ID, direct price, processing modes, and direct privacy/residency statements.
- **Gateway route:** gateway model ID, route geolocation, gateway markup or fees, BYOK treatment, failover/routing behavior, and gateway-specific capability metadata.

A gateway model page may quote upstream provider rates and then add a PAYG markup. Do not compare that marked-up rate to a direct-provider rate without making the layers explicit. “EU data residency” at the gateway does not automatically prove that every upstream route is EU-hosted; use the route’s own geolocation/hosting evidence.

## Modality and Image Checklist

For image questions, report separately:

- text input/output and API shape (`chat`, `responses`, or Anthropic Messages)
- image input: URL and/or Base64 support if documented
- image understanding/vision: explicit model flag
- video, PDF, and audio: explicit support only; otherwise “not verified”
- image generation: normally a separate capability and not implied by vision
- tool calling, reasoning, JSON schema, prompt caching, web search, and computer use

Generic gateway support for image URLs or Base64 does not prove that the selected model accepts images. Conversely, a model-level vision flag is stronger evidence than a generic feature page. For production, send a minimal redacted image request and record status/error behavior without exposing credentials or private image data.

## Video-Generation Pricing and Quality Comparisons

- **Quality recommendation:** Prefer Seedance 2.5 when higher output quality is the priority.

For video models, keep token-unit pricing, per-second examples, and subscription-credit economics as separate layers. Normalize only after fixing resolution, duration, aspect ratio, input modality, plan term, and promotion. Treat vendor quality claims as capability evidence—not as a quality score—and label internal/vendor benchmarks separately from independent evaluations. For a reusable Seedance evidence snapshot and source map, see `references/seedance-video-comparison.md`.

When a vendor's rendered page and pricing/API documentation disagree on capabilities such as maximum resolution, report the discrepancy and verify against the live account/API rather than choosing one silently. A targeted model upgrade should be described as workflow-specific unless an apples-to-apples benchmark supports a universal quality claim.

## Price Comparison Format

Use a compact mobile-friendly structure:

- **Route:** direct/gateway, region, exact model ID
- **Price:** input / output / cached input per 1M tokens
- **Gateway fee:** markup, per-request fee, or BYOK rule
- **Example:** normalized cost for a stated workload
- **Trade-off:** EU location, latency, retention, capability, or support difference

State that prices are live snapshots and may change. Do not convert currencies without a live exchange-rate source; if no conversion is needed, keep the source currency.

## Operating and Contractual Modes

Use these labels consistently:

- **Serverless/realtime:** shared capacity, per-token billing, interactive requests.
- **Flex:** lower priority or best-effort processing, usually discounted.
- **Async/batch:** background work with a stated completion window; pricing may be customer-specific.
- **Dedicated:** reserved/private capacity, commonly billed per GPU-hour with a commitment.
- **Enterprise:** volume discounts, SSO/RBAC, audit/compliance support, custom SLA, or custom hosting.

Do not infer that a model’s chat API supports video, PDF, or background jobs merely because the platform offers those features for other models.

## Evidence and Citation Rules

- Cite the exact model page and live API endpoint used.
- Keep direct-provider and gateway evidence visibly separate.
- Record retrieval date/time for current prices and availability.
- Quote vendor privacy/compliance claims as vendor claims unless independently verified.
- Explain conflicts between a generic documentation page and an exact live model record instead of hiding them.
- If no public direct price exists, say so and point to the account/contract/API source that controls the price.

A worked Requesty/Sference snapshot and source notes are in `references/requesty-sference-glm-5-3-flash.md`. Azure/Microsoft dynamic pricing and official-only region-research notes are in `references/azure-official-pricing-research.md`.

## Common Pitfalls

- Treating a provider’s company location as proof of EU inference hosting.
- Treating a gateway’s upstream rate as the final customer price.
- Comparing per-million-token prices with different cached-token or currency definitions.
- Claiming video/PDF/audio support because a gateway supports the modality in general.
- Treating `vision=true` as image generation.
- Using a search snippet instead of the live provider model object.
- Reporting a current price without retrieval date or route/model ID.
- Converting “not documented” into “not supported.”
- Using a wide comparison table in Discord when compact cards would be clearer.

## Verification Checklist

- [ ] Exact model ID and provider route captured.
- [ ] Live provider/gateway API queried.
- [ ] Geolocation/hosting evidence separated from company domicile.
- [ ] Input, output, and cached prices normalized with a tool.
- [ ] Gateway markup/BYOK treatment separated from direct rates.
- [ ] Image support checked at model level; other modalities marked explicitly.
- [ ] Processing/contract modes and undisclosed prices labeled accurately.
- [ ] Privacy, retention, training, and compliance claims scoped to the route.
- [ ] Retrieval date and direct source URLs included.
