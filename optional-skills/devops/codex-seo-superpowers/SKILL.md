---
name: codex-seo-superpowers
description: "Codex-first SEO/GEO/AEO/AI-visibility workflows for Hermes: audit sites, generate llms.txt plans, check schema/CWV/content/topical clusters, and produce artifact-first WordPress implementation plans."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [seo, geo, aeo, ai-visibility, codex-seo, wordpress, llms.txt, schema, gsc, pagespeed, dataforseo, firecrawl]
    related_skills: [hermes-agent]
---

# Codex SEO Superpowers

Use this skill when the user asks Hermes to become stronger at SEO, GEO, AEO, AI-search visibility, answer-engine optimization, `llms.txt`, schema, topical clustering, Core Web Vitals, WordPress SEO implementation, or Codex-style `/seo ...` workflows.

This skill is an integration layer for public SEO/GEO/AEO toolkits. It does **not** claim rankings or AI citations without measured evidence.

## Source stack

Recommended public stack:

1. `AgriciDaniel/codex-seo` — main Codex-first SEO/GEO/AEO audit and strategy suite.
2. `Auriti-Labs/geo-optimizer-skill` — focused GEO audit/fix engine for AI crawler access, `llms.txt`, schema, and citability.
3. `madeburo/GEO-AI-Woo` — WordPress/WooCommerce implementation candidate for `llms.txt`, AI crawler rules, AI metadata, and product data.
4. `elmohq/elmo` — AI visibility tracking architecture for prompt/citation snapshots and answer drift.
5. `GEO-optim/GEO` — research reference for Generative Engine Optimization methods.

## Install engines locally

Keep third-party repos outside the Hermes source tree:

```bash
mkdir -p ~/.hermes/vendor

if [ ! -d ~/.hermes/vendor/codex-seo/.git ]; then
  git clone --depth 1 https://github.com/AgriciDaniel/codex-seo ~/.hermes/vendor/codex-seo
fi
cd ~/.hermes/vendor/codex-seo
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-core.txt beautifulsoup4 lxml requests

if [ ! -d ~/.hermes/vendor/geo-optimizer-skill/.git ]; then
  git clone --depth 1 https://github.com/Auriti-Labs/geo-optimizer-skill ~/.hermes/vendor/geo-optimizer-skill
fi
cd ~/.hermes/vendor/geo-optimizer-skill
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[rich,config]'
```

## Command mapping

Use the wrapper script in this skill for artifact-first runs:

```bash
python3 ~/.hermes/skills/devops/codex-seo-superpowers/scripts/hermes_codex_seo_runner.py \
  /seo audit https://example.com \
  --out ~/.hermes/organic-growth-os/sites/example.com/codex-seo-audit
```

Supported prompt commands:

- `/seo audit <url>` — full SEO/GEO/AEO evidence bundle.
- `/seo geo <url>` — AI crawler, `llms.txt`, schema, and citability readiness.
- `/seo content <url>` — E-E-A-T, helpfulness, answer extraction, and AI citation readiness.
- `/seo schema <url>` — JSON-LD/schema detection and recommendations.
- `/seo cluster <keyword-or-url>` — topical architecture and cannibalization map.
- `/seo performance <url>` — CWV/PageSpeed-oriented performance checks.
- `/seo ecommerce <url>` — product, affiliate, and WooCommerce SEO checks.

## WordPress implementation gate

Treat `madeburo/GEO-AI-Woo` as an implementation candidate, not an automatic install.

Before a live WordPress plugin install or custom `/llms.txt` implementation:

1. Back up plugin/theme/MU-plugin state.
2. Detect active SEO/schema/crawler controls: Yoast, RankMath, AIOSEO, custom MU plugins, Cloudflare Workers, robots filters, sitemap plugins.
3. Choose implementation mode:
   - Plugin mode: install GEO-AI-Woo only after explicit approval and conflict checks.
   - MU-plugin mode: custom WordPress endpoint/header/schema additions.
   - Worker mode: edge-hosted `/llms.txt`, crawler rules, redirects, and headers.
   - Plan-only mode: produce files/settings without live changes.
4. Purge caches.
5. Verify public surfaces:
   - `/llms.txt` status/content.
   - `/llms-full.txt` status/content or intentional absence.
   - `robots.txt` policy for citation/search bots.
   - JSON-LD parses and matches visible content.
   - canonical/robots/indexability remains correct.

## AI visibility monitoring model

Use Elmo-style monitoring when the user wants ongoing proof:

- Track brand/entity prompts, commercial prompts, and problem/answer prompts.
- Capture answer text, citations, competitors, source URLs, timestamp, engine, and prompt variant.
- Compare snapshots over time.
- Report only measured deltas: citation gained/lost, competitor changed, answer drift, entity confusion, hallucinated facts.

## Output contract

For serious runs, save artifacts to disk and report paths. Include:

- canonical URL and target intent
- crawl/indexability status
- technical SEO and CWV/PageSpeed status when available
- schema/metadata findings
- `llms.txt`, `robots.txt`, and AI crawler access findings
- content helpfulness, E-E-A-T, and AEO answer-block readiness
- GEO/citability score and top fixes
- topical architecture/cannibalization flags
- WordPress implementation plan
- monitoring plan: T+14/T+45/T+90 plus prompt/citation tracking

## Safety rules

- Never fabricate DataForSEO, GSC, PageSpeed, Firecrawl, rankings, or AI-citation results.
- Do not install WordPress plugins without explicit approval.
- Do not replace Yoast/RankMath/AIOSEO as the SEO control plane unless explicitly requested.
- Do not commit API keys, credentials, raw crawl secrets, or customer private data.
- For YMYL pages, recommend factual review, medical/legal/financial source support, or noindex/merge when quality is insufficient.
