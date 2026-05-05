     1|---
     2|name: ai-red-team-intel-brief
     3|description: Systematic research and compilation of AI red teaming, security, and alignment research into threat-intel-style briefings. Covers jailbreaks, CVEs, adversarial ML, alignment failures, new tools, and lab safety publications.
     4|version: 1.0.0
     5|---
     6|
     7|# AI Red Team Intelligence Brief
     8|
     9|Produce a threat-intel-style briefing on AI red teaming, security, and alignment research. Output reads like a peer-written threat intel report — technical depth, no dumbing down, exploitable-now assessments, and defender action items.
    10|
    11|## Trigger Conditions
    12|- User requests "red team brief", "AI security intel", "adversarial ML update"
    13|- Scheduled cron job for daily/weekly AI red team intel
    14|
    15|## Research Sources (Priority Order)
    16|
    17|### 1. arXiv cs.CR + cs.AI (Primary)
    18|Navigate to `https://arxiv.org/list/cs.CR/current` and `https://arxiv.org/list/cs.AI/current`. Scan for: jailbreak, backdoor, adversarial, extraction, poisoning, alignment, red team, prompt injection, safety, manipulation, exploit, vulnerability, model inversion, membership inference.
    19|
    20|### 2. Lab Safety Blogs
    21|- **Anthropic Research**: `https://www.anthropic.com/research`
    22|- **Anthropic Red Team**: `https://red.anthropic.com/` — frontier red team evaluations
    23|- **Google DeepMind**: `https://deepmind.google/blog/` — "Responsibility & Safety" tag
    24|- **OpenAI Blog**: `https://openai.com/blog` — may be blocked by bot detection; try but don't block on failure
    25|
    26|### 3. GitHub New/Updated Repos
    27|Search for: `LLM jailbreak red team`, `prompt injection tool`, `AI red team framework`. Filter for new repos (< 7 days), active commits, growing stars.
    28|
    29|### 4. Web Search (if available)
    30|- `"LLM jailbreak 2026"` — new attack techniques
    31|- `"prompt injection new technique 2026"` — specific attack vector
    32|- `"alignment failure AI safety 2026"` — alignment research
    33|- `"AI security CVE vulnerability 2026"` — disclosed vulns
    34|- `"adversarial attacks LLM backdoor data poisoning arxiv 2026"` — arXiv catch-all
    35|
    36|### 5. Hacker News Fallback
    37|Navigate to `https://news.ycombinator.com/`. Filter for AI/LLM/security/vulnerability/jailbreak/red team stories.
    38|
    39|## Paper Triage Framework
    40|
    41|### Tier 1 — CRITICAL (Include with full analysis)
    42|- Novel jailbreak/prompt injection technique working on production models
    43|- Formal result breaking a widely-assumed security property
    44|- New attack class not previously documented
    45|- Frontier lab publishing red team evaluation results with exploit chains
    46|- CVE with AI-specific root cause
    47|
    48|### Tier 2 — HIGH (Include with moderate analysis)
    49|- New benchmark/framework for evaluating red team capabilities
    50|- Alignment research with direct red team implications
    51|- Agentic security architecture studies
    52|- New open-source red teaming tool with actual adoption
    53|
    54|### Tier 3 — NOTABLE (Brief mention)
    55|- Manipulation/social engineering evaluation frameworks
    56|- Interpretability work relevant to safety failures
    57|- Federated learning attacks with model extraction implications
    58|- Industry policy/framework publications
    59|
    60|### Exclude
    61|- Pure theory without empirical results
    62|- Survey/position papers without new techniques
    63|- Non-AI security papers without ML component
    64|- Marketing or hype without technical substance
    65|
    66|## Output Format
    67|
    68|```
    69|# AI Red Team & Security Intelligence Brief
    70|**DATE | Collection Window**
    71|
    72|## TIER 1 — CRITICAL: Action Required Now
    73|### N. [Title]
    74|**Source:** URL | **Date:** | **Authors:**
    75|**What it is:** 2-3 sentence description
    76|**Technical core:** Key mechanisms, formal results, or exploit details
    77|**Why it matters:** Impact assessment — who is affected, what assumptions are broken
    78|**Exploitable right now?** Yes/No/Conditional — with specifics
    79|**Defender/builder action items:** 2-3 concrete steps
    80|
    81|## TIER 2 — HIGH: Significant Developments
    82|[Same format, moderate depth]
    83|
    84|## TIER 3 — NOTABLE: Track and Evaluate
    85|[Same format, brief]
    86|
    87|## CROSS-CUTTING ANALYSIS
    88|- Synthesis of themes across items
    89|- What to replicate/study hands-on
    90|```
    91|
    92|## Writing Style
    93|- **Threat intel tone**: direct, technical, no hype
    94|- **Technical depth**: enough detail to actually use the information
    95|- **Exploitable-now assessment**: honest about theoretical vs. works today
    96|- **Actionable guidance**: every Tier 1 item gets specific defender/builder recommendations
    97|- **Hands-on flags**: mark items worth replicating for portfolio/skill building
    98|
    99|## Fallback Protocol
   100|
   101|### Billing Exhaustion (402 errors)
   102|All `web_search` calls will fail — pivot to browser navigation immediately. arXiv listing pages, lab blogs, GitHub search, and HN all work via browser without web search.
   103|
   104|### Browser/CDP Failures
   105|Use `web_extract` on specific URLs and `web_search` with targeted queries. For arXiv: `site:arxiv.org` with date-range filters to catch last-24h papers without loading listing pages.
   106|
   107|### Terminal-Only Mode (no web_search, no browser)
   108|- **HN Algolia API**: `https://hn.algolia.com/api/v1/search` — free JSON, no auth. Use `tags=story&numericFilters=created_at_i>{cutoff_timestamp}` for recent stories.
   109|- **GitHub API**: `https://api.github.com` — unauthenticated repo search (60/hr). Search: `LLM+jailbreak+red+team`, `prompt+injection+tool`.
   110|- **arXiv API**: `https://export.arxiv.org/api/query` — Atom XML. Query: `cat:cs.CR+AND+abs:LLM+OR+abs:jailbreak`. May rate-limit; skip cycle rather than retrying.
   111|
   112|## Quality Checks
   113|- Every paper citation verified against the arXiv abstract (not just search snippets)
   114|- Lab blog claims sourced to the actual post, not secondary coverage
   115|- GitHub repos must have recent activity (no abandoned projects)
   116|- Exploitable-now assessments must be honest: don't inflate risk, don't downplay it
   117|
   118|## Pitfalls
   119|- arXiv listing pages are large — always jump to end for newest papers
   120|- Some arXiv IDs are cross-lists from other categories — check the primary subject
   121|- GitHub repos with high stars may be jailbreak catalogues, not evaluation frameworks
   122|- Anthropic and DeepMind blogs may file safety posts under non-obvious categories
   123|- Don't confuse model capability evaluations (MMLU) with safety/red team evaluations
   124|- For production incidents, triangulate across at least two independent sources
   125|
   126|## Related Skills
   127|- `ai-intelligence-brief` — general AI industry trends
   128|- `ai-red-teaming-research` — research workflow for AI security experiments
   129|- `ai-paper-deep-dive` — deep single-paper mastery
   130|