# AI Visibility Stack Implementation Plan

Site: `{{domain}}`
Date: `{{date}}`
Mode: Audit-only / Draft implementation / Live implementation

## 1. Baseline

- Canonical host:
- CMS/stack:
- SEO plugin/control plane:
- Sitemap(s):
- Robots.txt:
- llms.txt:
- llms-full.txt:
- Schema source(s):
- Cache/CDN:
- GSC/Bing access:
- AI visibility monitoring available:

## 2. Codex SEO / GEO audit artifacts

- `/seo audit` artifact:
- `/seo geo` artifact:
- GEO Optimizer score:
- Top technical blockers:
- Top schema blockers:
- Top content/citability blockers:

## 3. WordPress implementation decision

Choose one:

- Plugin: GEO-AI-Woo after backup/conflict approval.
- MU plugin: custom llms/robots/header/schema additions.
- Cloudflare Worker: edge-hosted llms.txt + crawler rules + redirects.
- No live change: implementation spec only.

Conflict gates:

- Yoast/RankMath duplicate schema risk:
- Robots/crawler conflict:
- Existing llms endpoint conflict:
- Cache conflict:
- WooCommerce/product schema conflict:

## 4. Files/endpoints to ship

- `/llms.txt`:
- `/llms-full.txt`:
- `robots.txt` AI crawler policy:
- HTTP Link header:
- Organization/WebSite/WebPage/Article/Product schema:
- Priority URL list:

## 5. Monitoring prompts

Brand/entity prompts:

1.
2.
3.

Commercial prompts:

1.
2.
3.

Problem/answer prompts:

1.
2.
3.

Track per engine: ChatGPT, Perplexity, Claude, Gemini, Google AI Overviews where available.

## 6. Verification checklist

- [ ] Public `/llms.txt` 200 and correct content-type.
- [ ] Public `/llms-full.txt` 200 or intentionally disabled.
- [ ] Robots allows citation/search bots per policy.
- [ ] Schema parses and matches visible content.
- [ ] SEO plugin canonical/robots untouched unless deliberately changed.
- [ ] Cache-busted public URLs verified.
- [ ] GSC sitemap resubmitted if public architecture changed.
- [ ] AI visibility monitoring baseline snapshot captured.

## 7. T+ monitoring

- T+14: crawl/indexability/AI-crawler logs/prompt snapshots.
- T+45: GSC impressions/clicks/queries and citation deltas.
- T+90: content/prioritization refresh and competitor-citation comparison.
