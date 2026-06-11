# Codex SEO / GEO / AEO Hermes Reference

Use this reference to turn Codex-style SEO commands into repeatable Hermes execution.

## Command routing

| User command | Primary engine | Hermes follow-up |
|---|---|---|
| `/seo audit <url>` | Codex SEO audit + GEO Optimizer audit | public QA, canonical/indexability checks, artifact summary |
| `/seo geo <url>` | GEO Optimizer + Codex SEO GEO workflow | `llms.txt`, robots, schema, AI crawler implementation plan |
| `/seo content <url>` | Codex SEO content workflow | E-E-A-T, helpfulness, answer extraction, source/citation improvements |
| `/seo schema <url>` | Codex SEO schema workflow | visible-content support check and JSON-LD validation |
| `/seo cluster <seed>` | Codex SEO cluster workflow | topical map, intent ownership, cannibalization actions |
| `/seo performance <url>` | Codex SEO performance workflow | CWV/PageSpeed budget and rendered QA |
| `/seo ecommerce <url>` | Codex SEO ecommerce workflow | affiliate/WooCommerce/product schema compliance |

## Evidence-first workflow

1. Save all raw outputs to an artifact directory.
2. Separate findings into `confirmed`, `likely`, and `needs API credential`.
3. Do not claim ranking/visibility movement without before/after data.
4. For WordPress, implement only after backup + conflict gates.
5. Verify public changed URLs after purge: HTTP status, canonical, robots, H1, schema, mobile rendering where relevant.

## WordPress AI visibility checklist

- `/llms.txt` is public, readable, curated, and canonical-only.
- `/llms-full.txt` is either useful and canonical-only or intentionally absent.
- `robots.txt` distinguishes citation/search bots from training bots when the business policy requires it.
- Organization/WebSite/Article/Product schema matches visible content.
- Affiliate/commercial content has disclosure and no fake ratings/prices/testing claims.
- Yoast/RankMath/AIOSEO remains the canonical source of SEO metadata unless deliberately migrated.
- CDN/plugin cache is purged and cache-busted URLs are verified.

## AI visibility monitoring fields

Store snapshots as JSONL with:

```json
{
  "timestamp": "2026-01-01T00:00:00Z",
  "engine": "perplexity|chatgpt|claude|gemini|aio",
  "prompt": "best tool for ...",
  "answer": "...",
  "citations": ["https://example.com/page"],
  "competitors": ["competitor.com"],
  "brand_mentioned": true,
  "notes": "entity confusion / hallucination / source lost"
}
```

## GEO research patterns to apply safely

- Add source-backed statistics only when current and verifiable.
- Add concise quotations only when accurately attributed.
- Improve fluency and extractability; do not keyword-stuff.
- Use technical terms/entities where they improve precision.
- Add visible proof blocks for reviews/commercial claims.

## Risk controls

Remove or rewrite:

- hidden prompt-injection text
- comments telling AI systems to cite the site
- unsupported “tested” claims
- fake ratings, prices, or reviews
- raw schema/script artifacts in article bodies
- stale years, broken affiliate links, and irrelevant images
