---
name: ai-red-team-intel-brief
description: Systematic research and compilation of AI red teaming, security, and alignment research into threat-intel-style briefings. Covers jailbreaks, CVEs, adversarial ML, alignment failures, new tools, and lab safety publications.
version: 1.0.0
---

# AI Red Team Intelligence Brief

Produce a threat-intel-style briefing on AI red teaming, security, and alignment research. Output reads like a peer-written threat intel report — technical depth, no dumbing down, exploitable-now assessments, and defender action items.

## Trigger Conditions
- User requests "red team brief", "AI security intel", "adversarial ML update"
- Scheduled cron job for daily/weekly AI red team intel

## Research Sources (Priority Order)

### 1. arXiv cs.CR + cs.AI (Primary)
Navigate to `https://arxiv.org/list/cs.CR/current` and `https://arxiv.org/list/cs.AI/current`. Scan for: jailbreak, backdoor, adversarial, extraction, poisoning, alignment, red team, prompt injection, safety, manipulation, exploit, vulnerability, model inversion, membership inference.

### 2. Lab Safety Blogs
- **Anthropic Research**: `https://www.anthropic.com/research`
- **Anthropic Red Team**: `https://red.anthropic.com/` — frontier red team evaluations
- **Google DeepMind**: `https://deepmind.google/blog/` — "Responsibility & Safety" tag
- **OpenAI Blog**: `https://openai.com/blog` — may be blocked by bot detection; try but don't block on failure

### 3. GitHub New/Updated Repos
Search for: `LLM jailbreak red team`, `prompt injection tool`, `AI red team framework`. Filter for new repos (< 7 days), active commits, growing stars.

### 4. Web Search (if available)
- `"LLM jailbreak 2026"` — new attack techniques
- `"prompt injection new technique 2026"` — specific attack vector
- `"alignment failure AI safety 2026"` — alignment research
- `"AI security CVE vulnerability 2026"` — disclosed vulns
- `"adversarial attacks LLM backdoor data poisoning arxiv 2026"` — arXiv catch-all

### 5. Hacker News Fallback
Navigate to `https://news.ycombinator.com/`. Filter for AI/LLM/security/vulnerability/jailbreak/red team stories.

## Paper Triage Framework

### Tier 1 — CRITICAL (Include with full analysis)
- Novel jailbreak/prompt injection technique working on production models
- Formal result breaking a widely-assumed security property
- New attack class not previously documented
- Frontier lab publishing red team evaluation results with exploit chains
- CVE with AI-specific root cause

### Tier 2 — HIGH (Include with moderate analysis)
- New benchmark/framework for evaluating red team capabilities
- Alignment research with direct red team implications
- Agentic security architecture studies
- New open-source red teaming tool with actual adoption

### Tier 3 — NOTABLE (Brief mention)
- Manipulation/social engineering evaluation frameworks
- Interpretability work relevant to safety failures
- Federated learning attacks with model extraction implications
- Industry policy/framework publications

### Exclude
- Pure theory without empirical results
- Survey/position papers without new techniques
- Non-AI security papers without ML component
- Marketing or hype without technical substance

## Output Format

Deliver as a single markdown brief with this structure:

### Header
- Date (ISO 8601)
- Coverage window (last 24h / last 7d / custom range)
- Sources hit (arXiv, lab blogs, GitHub, web, HN — mark which were reachable)
- Item counts by tier (e.g. 2 CRITICAL · 4 HIGH · 6 NOTABLE)

### Tier 1 — CRITICAL
For each item:
- Title — paper / post / repo name
- Source — arXiv ID, blog URL, or repo URL
- TL;DR — 2 sentences, technical, no hype
- Attack class — what this is (e.g. "indirect prompt injection via tool output", "membership inference on RAG corpus")
- Affected systems — which production models / harnesses / agent frameworks are exposed
- Exploitable now? — Yes / Partial (lab conditions only) / No (theoretical) — with one-line justification
- Defender action — concrete mitigation or detection guidance
- Hands-on flag — [REPLICATE] if worth reproducing for portfolio/skill building

### Tier 2 — HIGH
For each item:
- Title + Source
- TL;DR — 1-2 sentences
- Why it matters — relevance to red team work
- Defender action — if applicable

### Tier 3 — NOTABLE
For each item:
- Title + Source
- One-line summary

### Footer
- Sources unreachable this cycle — list any that errored (402, blocked, rate-limited)
- Next cycle focus — anything flagged for deeper follow-up

### Length budget
- Daily brief: hard cap 15 items total across all tiers
- Weekly brief: hard cap 25 items
- If exceeded, drop lowest-scoring NOTABLE items first and add N items trimmed note

## Writing Style
- **Threat intel tone**: direct, technical, no hype
- **Technical depth**: enough detail to actually use the information
- **Exploitable-now assessment**: honest about theoretical vs. works today
- **Actionable guidance**: every Tier 1 item gets specific defender/builder recommendations
- **Hands-on flags**: mark items worth replicating for portfolio/skill building

## Fallback Protocol

### Billing Exhaustion (402 errors)
All `web_search` calls will fail — pivot to browser navigation immediately. arXiv listing pages, lab blogs, GitHub search, and HN all work via browser without web search.

### Browser/CDP Failures
Use `web_extract` on specific URLs and `web_search` with targeted queries. For arXiv: `site:arxiv.org` with date-range filters to catch last-24h papers without loading listing pages.

### Terminal-Only Mode (no web_search, no browser)
- **HN Algolia API**: `https://hn.algolia.com/api/v1/search` — free JSON, no auth. Use `tags=story&numericFilters=created_at_i>{cutoff_timestamp}` for recent stories.
- **GitHub API**: `https://api.github.com` — unauthenticated repo search (60/hr). Search: `LLM+jailbreak+red+team`, `prompt+injection+tool`.
- **arXiv API**: `https://export.arxiv.org/api/query` — Atom XML. Query: `cat:cs.CR+AND+abs:LLM+OR+abs:jailbreak`. May rate-limit; skip cycle rather than retrying.

## Quality Checks
- Every paper citation verified against the arXiv abstract (not just search snippets)
- Lab blog claims sourced to the actual post, not secondary coverage
- GitHub repos must have recent activity (no abandoned projects)
- Exploitable-now assessments must be honest: don't inflate risk, don't downplay it

## Pitfalls
- arXiv listing pages are large — always jump to end for newest papers
- Some arXiv IDs are cross-lists from other categories — check the primary subject
- GitHub repos with high stars may be jailbreak catalogues, not evaluation frameworks
- Anthropic and DeepMind blogs may file safety posts under non-obvious categories
- Don't confuse model capability evaluations (MMLU) with safety/red team evaluations
- For production incidents, triangulate across at least two independent sources

## Related Skills
- `ai-intelligence-brief` — general AI industry trends
- `ai-red-teaming-research` — research workflow for AI security experiments
- `ai-paper-deep-dive` — deep single-paper mastery
