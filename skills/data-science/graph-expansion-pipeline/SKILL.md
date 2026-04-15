---
name: graph-expansion-pipeline
description: >
  Graph Expansion Pipeline: multi-phase enrichment workflow for batches of
  people/contacts. Runs Scout OSINT research → Weave upsert → Sift deep dive →
  Weave synthesis (relationship mapping) → Google Doc report. Includes graceful
  degradation when web search tools are unavailable (credit limits, API errors).
  Use for expanding the social graph with new contacts, onboarding contact lists,
  or periodic enrichment runs.
metadata:
  author: Indigo Karasu
  version: "1.0.0"
---

# Graph Expansion Pipeline

Multi-phase enrichment workflow for batches of people/contacts. Produces a Google Doc report and updates Weave with new persons and relationships.

## When to use

- Expand the social graph with a batch of new contacts
- Onboard a contact list from an event, conference, or introduction
- Run periodic enrichment on existing Weave entries
- Generate a report on a group of people for meeting prep or outreach

## Pipeline Phases

1. **Scout OSINT Research** — Web search for each target (name, email, company). Collects profiles, publications, and public information.
2. **Weave Upsert** — Insert or update Person nodes in Weave with discovered data.
3. **Sift Deep Dive** — Extract detailed information from search results and existing Weave data.
4. **Weave Synthesis** — Map relationships between targets (shared organizations, locations, existing connections).
5. **Report Compilation** — Generate a Google Doc with findings and save the link.

## Parallelization Pattern

**CRITICAL (April 2026 finding): `delegate_task` subagents CANNOT use MCP tools** (including `mcp_tavily_tavily_search`, `mcp_tavily_tavily_extract`, etc.). This means the Scout/Sift research phases — which depend on Tavily for web search — must be run at the top level, not delegated to subagents. Subagent research calls silently fail (return empty results) because they lack MCP tool access.

**Recommended approach:**
- **Scout research phase:** Run at the top level. Use `mcp_tavily_tavily_search` directly for web research (up to 5 queries per call, batch targets together). Use `terminal` for GitHub/Crossref API queries (these work fine in subagents if needed).
- **Sift deep dive phase:** Also run at the top level for Tavily-dependent research.
- **Weave upsert + synthesis phases:** Can be delegated to subagents (terminal-only, no MCP dependency).
- **Report generation:** Run at top level (may need Google Docs API access).

```python
# Phase 1: Scout OSINT — MUST run at top level (needs MCP/Tavily)
# Batch Tavily searches: research 3-5 targets per search query
from hermes_tools import execute_code

# Use mcp_tavily_tavily_search directly for web research
# Use terminal/curl for GitHub API, Crossref API
```

**Fact nodes must use MERGE, not CREATE, on re-runs** — If a Fact with the same `id` already exists from a previous run, `CREATE` throws "Found duplicated primary key value" and fails silently. Always use `MERGE` on the fact `id` field:

```python
# WRONG — fails on re-runs with duplicate primary key error
CREATE (fact:Fact {id: 'fact-shemtov-smol', ...})

# CORRECT — MERGE on id, SET other properties
MERGE (fact:Fact {id: 'fact-shemtov-smol'})
ON CREATE SET fact.predicate = 'published', fact.value = '...', ...
ON MATCH SET fact.value = '...', fact.record_time = '{record_time}'
```

**DBLP is a primary source, not just a fallback** — In April 2026 runs, DBLP returned verified publications for 6/10 targets, making it the highest-yield source (higher than Google News RSS, which mostly returned empty search URLs). Prioritize DBLP queries alongside (not after) Google News for academic/tech targets.

**DBLP publication cross-referencing** — When a target appears on a multi-author paper (e.g., Open X-Embodiment with 260+ authors), download the paper's HTML from arxiv and parse `citation_author` meta tags. Then cross-reference ALL expansion targets against the full author list to find hidden co-authorship connections.

**Edge deduplication priority order** — When the same pair has multiple `Knows` edges with different `rel_type` values, keep only the most specific. Priority from highest to lowest: `coauthor` (1) > `colleague` (2) > `former_colleague` (3) > `shared_field` (4) > `colleague_of` (5). Delete all lower-priority edges. **BUT: coauthor edges must only exist between genuinely confirmed co-authors — verified via arXiv/semantic scholar author lists. UX/design targets should NOT have coauthor edges on ML/robotics papers.**

**Career transition edge type management (April 2026 finding)** — When the Sift deep dive discovers a target has changed employers (e.g., moved from Google to Sportradar, or from Google to Toyota Research Institute), ALL `colleague` edges between that target and current employees of the former employer must be downgraded to `former_colleague`. This also affects cross-org targets: if laith Ulaby is now at TRI (not Google), his edges to current Googlers are `former_colleague`, not `colleague`. Conversely, if Ankita Akerkar is still at Google, her edges to other current Googlers should be `colleague`, not `former_colleague`. Always verify current employer from the Sift deep dive before finalizing edge types.

**Single-pass employer-based edge correction (preferred pattern — MANDATORY)** — Build a `current_employers` dict mapping each target ID to their verified current org (use plain org names like `'Google'`, `'Sportradar'`, `'Toyota Research Institute'`, `'Meta'` — NOT `'Sportradar (formerly Google)'` which makes org comparison logic fragile), then iterate ALL target pairs in a single pass. **Do NOT do ad-hoc fixes for specific edges** — ad-hoc fixes create cascading errors that require 4+ correction rounds (observed April 2026). Instead, compute the expected edge type for every single pair, then apply ALL corrections at once, then verify against expected math:

**Post-correction verification (mandatory after EVERY edge correction pass):** After applying corrections, compute the expected edge count math and compare:

```
N = number of targets
Google_count = number of current Googlers
Sportradar_count = number at Sportradar (former Googlers)
TRI_count = number at TRI (former Googlers)
Meta_count = number at Meta
Coauthor_pairs = set of confirmed co-author pairs (tuples of sorted IDs)

Expected edges:
  coauthor = len(Coauthor_pairs) * 2  (bidirectional)
  colleague = C(Google_count, 2) * 2 - len(Coauthor_pairs_in_Google) * 2  (Google pairs minus coauthors)
  former_colleague = (Google_count * Sportradar_count + Google_count * TRI_count) * 2
  shared_field = remaining edges (Meta cross-org, Sportradar-TRI, etc.)
```

If the math doesn't match, DO NOT proceed to the next phase. Re-audit edges until counts match expected values. The April 2026 run required 5 correction passes before counts aligned (90 total: 6 coauthor, 36 colleague, 28 former_colleague, 20 shared_field).

```python
# Build employer map from enrichments (with explicit overrides for known transitions)
current_employers = {}
for tid, enrich in enrichments.items():
    org = enrich.get('org', 'Unknown')
    email = enrich.get('email', '')
    # Override email-domain inference for known transitions
    if 'behzadi' in enrich.get('name', '').lower():
        org = 'Sportradar'  # NOT 'Sportradar (formerly Google)' — keep org names clean for comparison
    if 'ulaby' in enrich.get('name', '').lower():
        org = 'Toyota Research Institute'
    # @googlemail.com is equivalent to @google.com for org inference
    if 'googlemail.com' in email and org == 'Unknown':
        org = 'Google'
    current_employers[tid] = org

# Define coauthor pairs (takes priority over employer-based types)
coauthor_pairs = {
    (id_yury, id_hadar),   # Open X-Embodiment
    (id_yury, id_peter),   # Open X-Embodiment
    (id_hadar, id_peter),  # Open X-Embodiment
}

def get_edge_type(id1, id2, org1, org2):
    """Determine correct edge type + context based on current employers."""
    # Check co-authorship first (highest priority)
    pair = tuple(sorted([id1, id2]))
    if pair in coauthor_pairs:
        return 'coauthor', 'Co-authors on Open X-Embodiment paper'
    # Same organization
    if org1 == org2 == 'Google':
        return 'colleague', 'Same organization: Google'
    # Former colleagues (one person left Google)
    if {org1, org2} == {'Google', 'Sportradar'}:
        return 'former_colleague', 'Behshad Behzadi formerly at Google, now at Sportradar'
    if {org1, org2} == {'Google', 'Toyota Research Institute'}:
        return 'former_colleague', 'laith Ulaby formerly at Google, now at Toyota Research Institute'
    # Cross-organization
    if 'Meta' in {org1, org2}:
        return 'shared_field', 'Cross-organization: UX/AI research field'
    if 'Sportradar' in {org1, org2} or 'Toyota Research Institute' in {org1, org2}:
        return 'shared_field', 'Cross-organization: AI/UX industry'
    return 'shared_field', 'Cross-organization: UX/AI research field'

# Single pass: correct all edges based on employer pairs
for i, id1 in enumerate(target_ids):
    for id2 in target_ids[i+1:]:
        correct_type, correct_ctx = get_edge_type(id1, id2, current_employers[id1], current_employers[id2])
        for from_id, to_id in [(id1, id2), (id2, id1)]:
            conn.execute("MATCH (p1:Person {id: $from_id})-[r:Knows]->(p2:Person {id: $to_id}) SET r.rel_type = $rt, r.context = $ctx, r.record_time = $time",
                         {'from_id': from_id, 'to_id': to_id, 'rt': correct_type, 'ctx': correct_ctx, 'time': record_time})
```

**Key pitfall: email-domain org inference fails for career transitions.** `behshad@google.com` → org inference says "Google" but person is at Sportradar. `marcpaulina@googlemail.com` → org inference says "Unknown" but person is at Google (googlemail.com = consumer Gmail alias used by Googlers). Always cross-check email-domain inference against Sift deep dive findings before classifying employers.

**Employer map org names must be clean for set comparison (April 2026 finding).** When building the `current_employers` dict for edge type classification, use plain org names like `'Google'`, `'Sportradar'`, `'Toyota Research Institute'` — NOT compound names like `'Sportradar (formerly Google)'`. Compound names break `org1 == org2` checks (e.g., `'Google' == 'Sportradar (formerly Google)'` is always False, so employer pairs involving Behshad never match the `'Google' == 'Google'` condition and get misclassified as `shared_field` instead of `former_colleague`). Use a separate hardcoded override map for career transitions and check with `set` membership (`if {org1, org2} == {'Google', 'Sportradar'}`) rather than string prefix matching.

**Coauthor pairs must be checked BEFORE employer-based edge types (April 2026 finding).** Without an explicit priority check, the employer-based logic classifies coauthor pairs (e.g., Yury Pinsky ↔ Hadar Shemtov, both at Google) as `colleague` instead of `coauthor`. Since `coauthor` has higher semantic value than `colleague`, always check coauthor pairs first in the edge type resolution function, then fall through to employer-based logic.

**Hunter.io quota exhaustion timing** — In cron runs, Hunter.io can exhaust on the verification/test call (email-verifier endpoint) before any domain-search or email-finder calls. The pre-flight check should use the cheapest endpoint (domain-search with `limit=1`) and if it returns 403, skip all Hunter calls immediately.

**Hunter.io 1010 Cloudflare error on email-finder (April 2026 finding)** — Hunter.io's email-finder endpoint now returns HTTP 403 with error code 1010 ("error code: 1010") even when domain-search works fine. This is a Cloudflare/WAF block, not a quota error. It's distinct from the quota-exhaustion 403 (which returns a JSON `errors` array with `authentication_failed` or rate-limit details). When email-finder returns 1010, do NOT retry — fall back to domain-search for org identification and use cached data for position/verification details. Domain-search (`domain-search?domain=X&limit=1`) remains functional even when email-finder is blocked.

**When delegate_task IS useful:**
- Weave database upserts (terminal-only Cypher operations)
- File I/O operations (writing intermediate results)
- Batch terminal commands (GitHub API, Crossref API)

**When delegate_task is NOT useful:**
- Any phase that needs `mcp_tavily_*` tools
- Any phase that needs `web_search`, `web_extract` (blocked by credits)
- Browser automation (subagents lack browser context)

**Coauthor edge accuracy for UX/design targets (April 2026 finding):** When the Sift deep dive disambiguates a target from "Research Scientist" to "UX Designer" or "Interaction Designer" (common when DBLP returns inflated publication counts for common surnames), ALL `coauthor` edges between that target and confirmed researchers must be downgraded to `shared_field` or `colleague`. UX designers are rarely co-authors on ML/robotics papers — the DBLP "co-authorship" is a disambiguation artifact. Only confirmed researchers on the same paper (verified via arXiv author lists or Semantic Scholar) should retain `coauthor` edges. For the April 2026 batch, this applied to: Ankita Akerkar (UX Designer), Marc Paulina (UX Lead), Gustavo Moura (Interaction Designer), and laith Ulaby (UX Research Director). Only Yury Pinsky, Hadar Shemtov, and Peter Oh retained legitimate `coauthor` edges for the Open X-Embodiment paper.

**IMPORTANT: The coauthor pair set must be verified before every synthesis run.** Marc Paulina and Gustavo Moura were incorrectly included as co-authors in one intermediate pass, creating wrong `coauthor` edges. The correct coauthor pairs for April 2026 are ONLY: (Yury, Hadar), (Yury, Peter), (Hadar, Peter). Always check against arXiv author lists or Semantic Scholar before confirming co-authorship.

**Coauthor pair tuples must be SORTED to match `get_edge_type()` lookup (April 2026 finding):** The `get_edge_type()` function sorts the two IDs before checking membership in `coauthor_pairs` (`pair = tuple(sorted([id1, id2]))`), but if the `coauthor_pairs` set was defined with IDs in their natural/UUID order (not sorted), the lookup silently fails. This caused Hadar Shemtov ↔ Peter Oh to be classified as `colleague` instead of `coauthor` in one run. **Fix:** Always define `coauthor_pairs` with sorted tuples:

```python
# WRONG — unsorted tuples, lookup fails when get_edge_type sorts them
coauthor_pairs = {
    (hadar_id, peter_id),  # hadar_id starts with 'a9de...', peter_id starts with 'a1fe...'
    # sorted() would produce ('a1fe...', 'a9de...') which doesn't match
}

# CORRECT — always use sorted tuples
coauthor_pairs = set()
for id1, id2 in [(yury_id, hadar_id), (yury_id, peter_id), (hadar_id, peter_id)]:
    coauthor_pairs.add(tuple(sorted([id1, id2])))
```

After edge creation, always verify: if coauthor edges < expected count, this is the likely cause.

**`delegate_task` with `toolsets: ['web']` for Sift deep dives (April 2026 finding):**

While `delegate_task` subagents cannot use MCP tools (including Tavily), they CAN use the built-in `web_search` tool when the `'web'` toolset is enabled. This is the correct approach for the Sift deep dive phase — delegate web research on individual targets to subagents that can use `web_search` in parallel:

```python
# CORRECT — delegate Sift deep dive with web toolset
from hermes_tools import delegate_task

results = delegate_task(
    goal="Research Yury Pinsky's current role at Google (Gemini/Bard), publications, and notable achievements. Focus on: Gemini product leadership, Imagen 3 paper, any recent 2024-2025 updates.",
    toolsets=['web'],
    context="Yury Pinsky is Director of Product Management at Google, leading Gemini/Bard extensions. Email: ypinsky@google.com"
)
```

This approach works because `web_search` is a built-in Hermes tool, not an MCP tool. Each subagent can independently search for specific targets, and results are automatically synthesized.

**Observed performance (April 2026 run, 10 targets):**
- Top-level Tavily parallel search: ~3s per query, ~30s total for all targets
- GitHub/Crossref API via terminal: ~25s for 10 targets
- delegate_task with web toolset for Sift: ~67s for all 10 targets (parallel subagent research)
- Total Scout phase: ~60s at top level (vs ~120s failed via delegate_task)

## Input Format

JSONL file with one target per line:

```jsonl
{"name": "Onny Chatterjee", "email": "onnychatterjee@meta.com", "id": "17dbb3f4-344a-41e5-8e54-e768f965e4a8"}
{"name": "Ankita Akerkar", "email": "aakerkar@google.com", "id": "6b3053ba-ec9c-513a-91c1-9483a60cb819"}
```

Required fields: `name`, `email`, `id` (UUID for Weave deduplication).

## Graceful Degradation Pattern

Web search tools may be unavailable due to credit limits, API errors, CAPTCHA blocking, or missing credentials. The pipeline continues using base data (name, email, domain inference):

```python
# Attempt web research
try:
    search_result = web_search(query=query, limit=3)
    if search_result and search_result.get('data', {}).get('web'):
        # Process results
        confidence = 'high' if len(results) >= 2 else 'med'
    else:
        # Empty results - continue with base data
        confidence = 'low'
except Exception as e:
    # Tool unavailable - continue with base data
    confidence = 'low'
    print(f"Web search unavailable: {e}")

# Proceed with upsert regardless
# Don't fail the entire pipeline due to missing enrichment
```

## Cached Data as Primary Execution Path (April 2026 Finding)

In practice, cron-run expansion pipelines should use **cached Hunter.io-enriched data** from previous interactive runs rather than attempting fresh API calls:

- `HUNTER_API_KEY` is typically set in interactive sessions but NOT available to cron jobs
- `web_search` tool is blocked by credit exhaustion (Firecrawl API limits)
- Browser tools are CAPTCHA-blocked on Google Search/Scholar/LinkedIn

**Recommended pattern for cron runs:**

```python
expansion_dir = Path('/root/.hermes/commons/data/ocas-expansion')

# Load cached research from previous interactive run
if (expansion_dir / 'complete_dossier.json').exists():
    with open(expansion_dir / 'complete_dossier.json') as f:
        dossier = json.load(f)
    targets = dossier.get('targets', [])
    print(f"Using cached enriched data: {len(targets)} profiles")
    # Proceed with Weave upsert and synthesis using cached data
else:
    # Fall back to base data (name, email, domain inference only)
    print("No cached data available. Proceeding with base data only.")
```

This allows meaningful pipeline execution (Weave upsert, relationship mapping, report generation) even with zero live API calls. The cached data from April 2026 runs includes Hunter.io verification scores, position data, LinkedIn URLs, and patent search results.

**Hunter.io as Primary OSINT Source (When web_search Blocked)**

Hunter.io provides reliable email verification and domain enrichment without requiring web search credits. Use when `web_search` returns "Payment Required". **However, as of April 2026, Hunter.io can also fail with 403 Forbidden** (monthly quota exhausted or Cloudflare 1010 WAF block), not just 429 rate-limit. When Hunter returns 403 on `email-finder`, proceed immediately to cached data and domain inference — do not retry. **Critical timing note (April 2026): The `email-finder` endpoint exhausts quota before `domain-search`. In cron runs, the very first `email-finder` call returns 403, while `domain-search?limit=1` still returns 200. This makes `domain-search` an unreliable quota detector — if `email-finder` returns 403, treat the entire Hunter quota as exhausted even if `domain-search` appears to work.**

```python
import requests

# HUNTER_API_KEY is NOT available in os.environ during cron jobs.
# Must manually load from .env file:
from pathlib import Path
env_file = Path.home() / '.hermes' / '.env'
with open(env_file) as f:
    for line in f:
        if line.startswith('HUNTER_API_KEY='):
            HUNTER_API_KEY = line.strip().split('=', 1)[1]
            break

for target in targets:
    name = target['name']
    email = target['email']
    domain = email.split('@')[-1] if '@' in email else None
    
    findings = []
    
    if domain and domain not in ['gmail.com']:
        # Domain search - find all emails at company
        url = f"https://api.hunter.io/v2/domain-search?domain={domain}&api_key={HUNTER_API_KEY}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            domain_data = response.json()
            # Extract organization name, email pattern, colleagues
        
        # Email finder - verify specific person's email
        first_name = name.split()[0]
        last_name = name.split()[-1] if len(name.split()) > 1 else ''
        url = f"https://api.hunter.io/v2/email-finder?domain={domain}&first_name={first_name}&last_name={last_name}&api_key={HUNTER_API_KEY}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            finder_data = response.json()
            if 'data' in finder_data and finder_data['data'].get('email'):
                score = finder_data['data'].get('score', 0)
                confidence = 'high' if score >= 90 else 'med' if score >= 50 else 'low'
                findings.append({
                    'type': 'email_finder',
                    'email': finder_data['data']['email'],
                    'score': score,
                    'verification': finder_data['data'].get('verification', {})
                })
```

**Hunter.io free tier:** 50 requests/month (sufficient for quarterly expansion runs on ~10-15 targets). **Pre-flight pitfall (April 2026):** Domain-search with `limit=1` returns HTTP 200 even when monthly quota is exhausted, making it an unreliable quota detector. The email-finder endpoint returns 403 immediately when quota is gone. If domain-search succeeds but email-finder returns 403, treat the entire Hunter quota as exhausted and fall back to cached data/domain inference — do not attempt further Hunter calls.

**Infrastructure constraints observed (April 2026):**
- **Hunter.io: PRIMARY OSINT SOURCE** — API key configured and working reliably. Use for domain search (org identification) and email-finder (position, LinkedIn, verification score). Free tier: 50 requests/month. Confidence scoring: score ≥90 = high (0.9), 50-89 = med (0.6), <50 = low (0.3). **NOTE:** HUNTER_API_KEY is NOT available in `os.environ` during cron jobs — must manually load from `~/.hermes/.env`. Additionally, Hunter.io can return **403 Forbidden** (not just 429 rate-limit) when monthly quota is exhausted; treat 403 same as exhausted quota and fall back to cached data/domain inference immediately.
- Firecrawl API: Credit exhaustion common — web_extract and web_search fail with "Payment Required: Insufficient credits"
- **`web_search` tool: BLOCKED by credits/provider limits** — returns "Payment Required: Insufficient credits". Also SearXNG backend may return 502 Bad Gateway, making the tool completely unavailable even with credits.
- **SearXNG (behind web_search): Can return 502 Bad Gateway** — When the SearXNG service is down, `web_search` fails with `SearXNG failed: Server error '502 Bad Gateway'`. This is distinct from credit exhaustion. When both SearXNG (502) and Tavily (432) fail, the only remaining search options are direct API calls (GitHub, DBLP, Semantic Scholar, Crossref, arXiv).
- DuckDuckGo via curl: CAPTCHA-blocked for HTML/lite variants, Google News works but may be empty
- LinkedIn: Login wall blocks scraping without authentication
- **Browser tools (browser_navigate, etc.): CAPTCHA/rate-limit blocked** on Google Search, Google Scholar, LinkedIn. Not reliable for OSINT when search APIs are exhausted.
- LadybugDB Python module: Available as `real_ladybug` in hermes-agent venv (not system Python). **IMPORTANT:** `lb.Database(path, read_only=True)` throws `RuntimeError: Cannot create an empty database under READ ONLY mode` even when the database file exists. Always use `lb.Database(path)` without `read_only=True` for both reads and writes. The `read_only=True` flag prevents database initialization and should not be used.
- Google Docs API: Requires user OAuth token at `/root/.hermes/google_token.json` (NOT service account). Token successfully created docs even without explicit `documents` scope in April 2026 run. **If OAuth flow is not completed (no `google_token.json` exists), skip Google Docs and save locally.**
- Brave/Exa: Not configured in environment
- **Tavily MCP: FULLY EXHAUSTED AS OF APRIL 2026** — `mcp_tavily_tavily_search` and `mcp_tavily_tavily_extract` both return HTTP 432 ("exceeds your plan's set usage limit") for ALL calls, including `basic` depth. This is not a temporary rate limit — it's a monthly plan quota exhaustion that persists across runs. `web_search` (Firecrawl-backed) similarly returns "Payment Required: Insufficient credits." **Assume zero web search capability from Tavily/Firecrawl for the remainder of each month and plan around pure API-based research.** When Tavily returns 432, do not retry — immediately switch to the API-only fallback stack (Semantic Scholar, DBLP, GitHub API, Google Developer profile probing, Google News RSS, direct curl URL checks).
- **Tavily MCP in cron jobs:** Even less reliable than interactive runs due to month-to-month quota carryover. **Do not plan any pipeline phase around Tavily availability.** Build the entire research phase around API-only sources that have no monthly quota (GitHub, DBLP, Semantic Scholar with pacing, Google Developer profile probing, Google News RSS).
**Google Drive storage quota exceeded** — Service accounts may hit quota limits. Fallback to local markdown report when quota exceeded. The OCIGCP service account can READ existing docs but CANNOT create new ones or write to existing ones (403 "caller does not have permission"). When `documents.create` fails with 403, fall back immediately — don't retry.

**LadybugDB Fact node schema (April 2026 cron finding)** — Fact nodes use `id`, `predicate`, `value` properties (NOT `content`). Attempting to MERGE on a non-existent `content` property throws `"Binder exception: Cannot find property content for f."`. Always use `id` as the unique key for Fact MERGE operations:

```python
# WRONG — throws BinderException because 'content' property doesn't exist
MERGE (f:Fact {content: '...'})

# CORRECT — MERGE on 'id' (the actual unique key), SET other properties
MERGE (f:Fact {id: 'fact-shemtov-phd'})
SET f.predicate = 'has_education',
    f.value = 'PhD Computational Linguistics, Stanford',
    f.confidence = 0.9,
    f.source_type = 'research',
    f.source_ref = 'expansion_pipeline',
    f.record_time = '{now}'
```

The confirmed Fact schema: `id`, `predicate`, `value`, `confidence`, `source_type`, `source_ref`, `record_time`. Do NOT use `content`, `description`, or `name` — these properties do not exist on Fact nodes.

**delegate_task with web toolset for Scout research (April 2026 cron finding)** — Using `delegate_task` with `toolsets=['web', 'terminal']` for Scout OSINT research is effective, since web_search is a built-in Hermes tool (not MCP). Each subagent can independently search via web_search and use terminal for API queries (curl/GitHub/Semantic Scholar). Observed performance: ~30s total for 3 targets in parallel, ~60s total for all 10 targets in 4 parallel batches. This approach avoids the MCP tool unavailability issue noted in the original skill documentation.
- **arXiv API via curl:** May return empty results from cron/sandbox environments (IP blocking or network restrictions). Write a Python script using `urllib.request` rather than piping curl to `python3 -c` — the pipe approach fails silently in sandbox environments.
- **arXiv API rate-limiting:** Returns HTTP 429 "Rate exceeded" after ~10 rapid requests. Add `time.sleep(3)` between requests or batch queries carefully.
- **Semantic Scholar API rate-limiting:** Returns HTTP 429 after ~5-6 rapid requests. Add 3-second delays between queries. Use `fields=` parameter to get detailed info in one call rather than multiple calls per author.
- **`delegate_task` subagents CANNOT use MCP tools** — confirmed again in April 2026. Subagents calling `mcp_tavily_tavily_search` get empty/error results. Must run Tavily-dependent phases at top level.
- **Google developer profile pages are JS-rendered SPAs** — `curl` returns only the shell HTML with `noindex` meta tag. HTTP 200 confirms the profile exists, but you cannot extract profile content from the HTML. The probe is only useful for confirming Google employment.
- **`execute_code` f-string quoting limitation** — Complex shell commands with nested quotes (pipelines, python -c with JSON parsing) cause SyntaxError in execute_code. Always write complex commands as script files to `/tmp/` and run `python3 /tmp/script.py` instead. Specifically, f-strings with nested dictionary access like `f"MATCH (p:Person {{id: '{u[\"id\"]}'}})\"` fail because Python's f-string parser gets confused by the alternating quote types. **PREFERRED FIX:** Use LadybugDB's parameterized query interface (`conn.execute(cypher, params_dict)`) which eliminates all quoting issues for MATCH/SET operations. For CREATE operations where parameterized queries may not work, use string concatenation: `"MERGE (p:Person {id: '" + u["id"] + "'})"`.

**Working patterns confirmed (April 2026 runs):**

- **Google Developer profile probing** — Probing `https://developers.google.com/profile/u/{email_username}` for `@google.com` targets confirms Google employment (returns 200 for existing profiles, 404 otherwise). Highly effective for short-handle Google employees (e.g., `hadar`, `rux`, `behshad`, `poh`, `moura`).
- **DBLP API is more reliable than Semantic Scholar for Google/Meta researchers** — `https://dblp.org/search/publ/api?q=author:{name}&format=json&h=5` returns actual publications with venues and years. Semantic Scholar frequently returns 0-paper placeholder entries (empty author stubs that look like matches but have `paperCount: 0, citationCount: 0`). Treat Semantic Scholar results with `paperCount: 0` as non-matches — these are API artifacts, not real profiles.
- **Contextual Google News RSS searches** — Adding organization/context keywords (e.g., `"Behshad Behzadi" Sportradar CTO AI`) yields results that plain name searches miss. Use `'query': '%22{Name}+{Surname}%22+{Context}'` format.
- **Hunter.io API works reliably** when `HUNTER_API_KEY` is set — provides email verification, position data, LinkedIn URLs without web search credits
- Google Docs creation via user OAuth token works reliably with simple `insertText` operations (but cron jobs may not have token access)
- Sequential index calculation for multi-line inserts works when `current_index` is updated after each insert
- **Google Patents remained accessible throughout** — key source for high-value findings on Google researchers. Patent numbers (US9666187, US11016638, US20250272122) provided concrete evidence of technical contributions when other sources failed.
- **Browser tools do NOT bypass CAPTCHA** — browser_navigate to Google Search/Scholar/LinkedIn all blocked. Use Hunter.io instead.
- **GitHub API direct queries work** — `curl -s "https://api.github.com/users/{username}"` returns profile data without auth for public profiles
- **GitHub username pattern matching** — try `firstlast`, `first-last`, `flast`, `email_username` patterns to find profiles. **CRITICAL: GitHub usernames do not always match real names.** E.g., `rux` belongs to Russ Anderson (company: Lilli, location: London), NOT Russell Matsuo (Google). Always cross-reference with `name`, `bio`, `company` fields before claiming a match. A matching username alone is low-confidence (0.3).
- **Local report fallback** — When Google Drive quota exceeded, save markdown to `/root/.hermes/commons/data/ocas-expansion/report_YYYY-MM-DD.md` and write status to `last_run_report.txt`

**Google News RSS pattern (confirmed April 2026, MEDIUM yield for cron runs):**

```python
import urllib.request
import re

# Free OSINT source when Tavily is exhausted
# NOTE: returns mostly search-echo titles for unknown/niche targets
name = "Behshad Behzadi"
query = f'%22{name.replace(" ", "+")}+Google%22'  # Add context keywords for better results
url = f"https://news.google.com/rss/search?q={query}"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=15) as resp:
    xml = resp.read().decode('utf-8', errors='replace')

# Parse titles
titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', xml)
if not titles:
    titles = re.findall(r'<title>(.*?)</title>', xml)

# Filter out the search-title noise (VERY common — Google News RSS returns
# the search echo title like '"Name Surname" - Google News' as the first result)
search_title = f'"{name}" - Google News'
real_titles = [t for t in titles if t not in ['Google News'] and t != search_title]

# Parse dates
dates = re.findall(r'<pubDate>(.*?)</pubDate>', xml)

# Parse links
links = re.findall(r'<link>(.*?)</link>', xml)
links = [l for l in links if l.startswith('http')][:5]
```

**Yield characteristics (April 2026 observation):** For well-known executives/public figures (e.g., Behshad Behzadi at Sportradar), returns genuine article titles. For most Google/Meta researchers, returns only the search-echo title with zero useful content. Adding context keywords (organization name, role) improves yield. Use `urllib.request` instead of `curl` for reliability in sandbox environments. No API key needed. Works from cron.

**High-yield Tavily search techniques (confirmed April 2026):**

- **`include_domains` filtering** — Using `include_domains: ["linkedin.com","en.wikipedia.org","scholar.google.com","patents.google.com"]` in `mcp_tavily_tavily_search` produces targeted, high-confidence results. LinkedIn results from Tavily include current title, company, education, and activity even without login access. Example: searching `"Behshad Behzadi Google executive role position"` with LinkedIn domain returned full professional profile (CPO/CTO/Chief AI Officer at Sportradar, co-founder of Google Assistant/Lens, Ecole Polytechnique education).
- **Patent inventor queries** — Searching `inventor:"Hadar Shemtov" Google patents` via Tavily returns confirmed patent records with assignee history, providing strong identity verification and career timeline. Useful for establishing technical contributions and prior affiliations (e.g., Hadar Shemtov's PARC patent predating Google employment).
- **Research blog acknowledgments** — Google Research blog posts (research.google/blog/) frequently acknowledge contributors by name. Searching for a target's name alongside Google product names surfaces their involvement in specific projects (e.g., "Hadar Shemtov Google Translate" confirmed involvement in CVSS corpus and Translatotron via blog acknowledgments).
- **`search_depth: "advanced"`** — Consistently returns richer results with longer content snippets than `basic` for OSINT purposes.

**DBLP name disambiguation pitfall (April 2026 finding):**

DBLP's `author:{name}` search aggregates publications from ALL researchers matching that name query, not just the target person. This causes wildly inflated publication counts for common surnames:

- "Peter Oh" → 721 pubs, but most are from different "Oh" researchers (Sehong Oh, Jihoon Oh, etc.). Only Open X-Embodiment and the CHIL paper are plausibly attributable to our target. The 2025 "QuICSeedR" publication is from a different Peter Oh in bioinformatics.
- "Gustavo Moura" → 121 pubs, includes many distinct "Gustavo Moura" researchers across different fields.
- "Marc Paulina" → 77 pubs, dominated by "Beatriz Paulina Garcia Salgado" (not our target). Only "PepCompass" (2025) is plausibly ours.

**For the Peter Oh case specifically**: When filtering recent publications by venue relevance to a Google Robotics researcher, discard bioinformatics venues (Bioinform., J. Computational Biology) and medical venues (CHIL conference workshops). Keep only ML/robotics venues (ICRA, CoRR with robotics keywords).

For rare names (Hadar Shemtov, Behshad Behzadi), DBLP returns accurate results. For common surnames, treat the total publication count and venue list with low confidence. Cross-check venue relevance against the target's known domain (e.g., a Google Robotics researcher should have robotics/ML venues, not bioinformatics or Portuguese education venues).

**Mitigation strategies:**
1. Filter publications by venue relevance to the target's known domain before counting.
2. Cross-reference Semantic Scholar `paperCount` — if DBLP returns 721 but Semantic Scholar shows a much lower count, the DBLP aggregation is likely disambiguated.
3. Flag any target with >50 DBLP publications as "likely disambiguation issue" and manually review the top 5 publications for domain match.
4. For `@google.com` targets, Google Developer profile confirmation is a more reliable identity signal than DBLP publication count.

**GitHub identity verification pitfall (April 2026 finding):**

GitHub's `/search/users?q=FirstName+LastName+in:name` endpoint frequently returns ZERO results even for well-known tech employees (tested against 7 Google/Meta researchers with public profiles). Direct username lookups (`/users/{username}`) work reliably for known usernames but the search endpoint is unreliable for OSINT discovery. Always prefer direct username probing over search.

**GitHub API commit search by author-email returns zero results for most Google/Meta employees** (April 2026 cron run). The endpoint `api.github.com/search/commits?q=author-email:{email}` found zero commits for all 10 targets. This is expected — Google and Meta employees typically use internal version control (Piper, etc.), not public GitHub. Don't waste time on commit searches for `@google.com` or `@meta.com` targets; prefer username probing and Google Developer profile checking instead.

**GitHub username pattern matching frequently returns profiles that are NOT the target person. Common failure modes:**
- `"pinsky"` → profile with name "stuart" (different person entirely)
- **`"rux"` → profile with name "Russ Anderson" at "Lilli" — completely different person from "Russell Matsuo" at Google. This is a textbook false positive.**
- **`"moura"` → profile with name "Lilian Moura" at "Zé Delivery" — not Gustavo Moura at Google.**
- **`"behzadi"` → profile with name "Mehdi Behzadi" — not Behshad Behzadi.**
- Empty/inactive accounts with 0 repos, no name, no bio, no company — these cannot confirm identity
- Profile matches must be verified by comparing `name`, `bio`, and `company` fields against known target attributes
- Only treat a GitHub match as confirmed if at least one identifying field (name, bio, company, location) corroborates the target's identity
- Profiles with 0 repos, 0 followers, and no identifying information should be treated as unconfirmed even if the username matches the target's name pattern
- **False positive scoring:** GitHub match with corroborating `name` field = med confidence (0.6). GitHub match with only username similarity = low confidence (0.3). GitHub match with contradictory `name` field = discard (not the same person).

**Self-relationship prevention (April 2026 finding):**

When creating `Knows` edges in loops (especially `former_colleague` loops where one person appears in both source and target lists), always check `id1 != id2` before creating the edge. Without this guard, LadybugDB will happily create a self-relationship (`Person → Same Person`) which is never semantically valid. Detection query:

```python
# Find and delete self-relationships
self_rels = list(conn.execute("MATCH (p:Person)-[r:Knows]->(p) RETURN p.name, r.rel_type"))
for name, rel_type in self_rels:
    print(f"  Self-relationship found: {name} --[{rel_type}]--> {name}")
conn.execute("MATCH (p:Person)-[r:Knows]->(p) DELETE r")
```

Prevention pattern — always filter the target list to exclude the source:

```python
# WRONG — creates self-relationship
for gid in google_ids:  # includes behshad_id if he's in the Google list
    cypher = f"MERGE (p1)-[r:Knows]->(p2) WHERE p1.id = '{behshad_id}' AND p2.id = '{gid}'"

# CORRECT — exclude self from target list
for gid in google_ids:
    if gid == behshad_id:
        continue  # no self-relationships
```

**Cached data `sources` field format inconsistency (April 2026 finding):**

The `sources` field in `scout_findings.json` and `scout_research_final.json` can be either a `dict` (keyed by source name, e.g., `{'dblp': {...}, 'google_news': {...}}`) or a `list` (from older run formats, e.g., `['name', 'google_dev_profile']`). Always check the type before accessing: use `isinstance(sources, dict)` and fall back gracefully. Code that assumes dict format (`sources.get('dblp', {})`) will crash with `TypeError: list indices must be integers or slices, not str` on older-format data. Defensive pattern:

```python
sources = data.get('sources', {})
if isinstance(sources, list):
    sources = {k: True for k in sources}  # Convert list to dict
# Now safe to use sources.get('dblp', {})
```

**`__import__()` fails in execute_code heredocs (April 2026 finding):**

Never use `__import__('module')` inline in f-strings or heredoc scripts within `execute_code`. The Python runtime inside execute_code does not resolve `__import__('timezone')` correctly — it throws `ModuleNotFoundError: No module named 'timezone'` even though `from datetime import timezone` works fine. Always import modules at the top of the script, not inline:

```python
# WRONG — crashes with ModuleNotFoundError
cypher = f"""... record_time = '{__import__("datetime").datetime.now(__import__("timezone").timezone.utc).isoformat()}'"""

# CORRECT — import at script top
from datetime import datetime, timezone
record_time = datetime.now(timezone.utc).isoformat()
cypher = f"""... record_time = '{record_time}'"""
```

**Spurious edge type validation (April 2026 finding):**

Previous pipeline runs (especially LLM-driven synthesis) can hallucinate invalid `rel_type` values on `Knows` edges — e.g., `wife_of`, `mother_of`, `father_of`, `SPOUSE`, `son_of`, `husband_of`. These are semantically wrong and contaminate the graph. Always run a **validation pass** before synthesis that deletes any `Knows.rel_type` not in the valid set:

```python
valid_types = {'colleague', 'former_colleague', 'coauthor', 'shared_field', 'friend', 'family', 'acquaintance'}

# Find and delete spurious edges between expansion targets
for id1 in target_ids:
    for id2 in target_ids:
        if id1 == id2:
            continue
        rels = list(conn.execute(
            "MATCH (p1:Person {id: $id1})-[r:Knows]->(p2:Person {id: $id2}) RETURN p1.name, p2.name, r.rel_type",
            {'id1': id1, 'id2': id2}
        ))
        for row in rels:
            n1, n2, rt = row
            if rt not in valid_types:
                conn.execute(
                    "MATCH (p1:Person {id: $id1})-[r:Knows]->(p2:Person {id: $id2}) WHERE r.rel_type = $rt DELETE r",
                    {'id1': id1, 'id2': id2, 'rt': rt}
                )
                print(f"  Deleted spurious: {n1} --[{rt}]--> {n2}")
```

Run this cleanup **before and after** the synthesis phase to catch both pre-existing and newly-created spurious edges.

**`findings` field type inconsistency (April 2026 finding):**

The `findings` field in cached `scout_findings.json` and `complete_dossier.json` can be either a `dict` (structured data) or a `str` (plain text), depending on which pipeline version created the entry. Always check the type before accessing:

```python
raw_findings = profile.get('findings', '')
if isinstance(raw_findings, dict):
    findings_summary = json.dumps(raw_findings, default=str)[:300]
elif isinstance(raw_findings, str):
    findings_summary = raw_findings[:300]
else:
    findings_summary = str(raw_findings)[:300]
```

**f-string with JSON in Cypher queries fails in execute_code (April 2026 finding):**

Using `json.dumps()` inside f-string Cypher queries within `execute_code` heredocs causes `SyntaxError: '(' was never closed`. The triple-quote f-string parser cannot handle nested brackets from JSON. Solutions:
1. **Use parameterized queries** (preferred): `conn.execute(cypher, {'ids': target_ids})`
2. **Write script to `/tmp/`** and run with `python3 /tmp/script.py` instead of inline heredocs
3. **Use string concatenation** instead of f-strings for dynamic list injection

```python
# WRONG — causes SyntaxError in execute_code
dup_check = list(conn.execute(f"""
    MATCH (p1:Person)-[r:Knows]->(p2:Person)
    WHERE p1.id IN {json.dumps(target_ids)}
    RETURN p1.name, p2.name
"""))

# CORRECT — parameterized query
dup_check = list(conn.execute(
    "MATCH (p1:Person)-[r:Knows]->(p2:Person) WHERE p1.id IN $ids RETURN p1.name, p2.name",
    {'ids': target_ids}
))
```

**Delete-then-recreate edge pattern is fragile (April 2026 finding):**

When deduplicating edges, deleting ALL edges between a pair and then recreating the best one is risky — if recreation fails (e.g., due to a separate bug like the `__import__` issue above), you lose all edges for that pair. 31 pairs lost all connections during a dedup pass before the bug was caught. Prefer MATCH+SET upgrades:

```python
# RISKY — if recreation fails, pair has zero edges
conn.execute("MATCH (p1)-[r:Knows]->(p2) WHERE p1.id = X AND p2.id = Y DELETE r")
conn.execute("CREATE (p1)-[r:Knows {...}]->(p2)")  # if this fails, data lost

# SAFER — find best edge, then delete only the worse ones
best_edges = list(conn.execute("MATCH ... RETURN r.rel_type ORDER BY priority"))
worst_rels = [r for r in all_edges if r != best]
# Only delete inferior edges, keeping the best one intact
for edge in worst_rels:
    conn.execute("MATCH ... DELETE r")
# Then upgrade the surviving best edge
```

**GitHub API in sandbox environments (April 2026 finding):**

Piping `curl` output to `python3 -c "..."` fails silently in sandbox/execute_code environments — the pipe appears to work but produces empty output. Instead:
1. Write a Python script to `/tmp/gh_search.py` using `urllib.request` (no external deps needed)
2. Use `from hermes_tools import terminal` to run `python3 /tmp/gh_search.py`
3. Save results to `/tmp/github_results.json` and read back with `read_file`

This pattern is more reliable than inline shell pipes for any API that returns JSON.

**Pre-flight API check (run before committing to Phase 1):**

Before spending time on the full Scout research phase, test whether paid APIs still have quota. This saves 30-60 seconds on cron runs where quotas are already exhausted:

```python
import urllib.request, json

def check_api_quotas():
    """Return dict of API availability. Run before Phase 1."""
    quotas = {"hunter": False, "tavily": False}
    
    # Hunter: single domain-search call (cheapest endpoint)
    env_file = Path.home() / '.hermes' / '.env'
    hunter_key = ""
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith('HUNTER_API_KEY='):
                    hunter_key = line.strip().split('=', 1)[1]
    if hunter_key:
        try:
            url = f"https://api.hunter.io/v2/domain-search?domain=google.com&api_key={hunter_key}&limit=1"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                quotas["hunter"] = True
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                quotas["hunter"] = False  # Quota exhausted
            pass
    
    # Tavily: will be caught on first search call via HTTP 432
    # No cheap pre-flight; catch the error inline
    quotas["tavily"] = "unknown"  # Tested on first call
    
    return quotas
```

If `quotas["hunter"]` is False, skip all Hunter.io calls and use cached `scout_findings.json` domain_search data for org identification instead (the domain_search data from previous runs is often still valid even when email-finder was blocked by 403).

**Multi-author paper cross-referencing (April 2026 finding):**

When a target is confirmed on a large multi-author paper (e.g., Imagen 3 with 262 authors), download the paper's HTML from arxiv and parse the full author list with `citation_author` meta tags. Then cross-reference ALL expansion targets against the author list to find hidden connections:

```python
# Download paper HTML and extract all authors
import re
r = terminal(command=f'curl -sL "https://arxiv.org/abs/{arxiv_id}" -o /tmp/paper.html', timeout=15)

# Parse citation_author meta tags from HTML
script = '''
import re
with open("/tmp/paper.html") as f:
    content = f.read()
authors = re.findall(r'citation_author.*?content="(.*?)"', content)
target_surnames = {"Pinsky", "Behzadi", "Moura", "Shemtov", "Matsuo", "Akerkar"}
for a in authors:
    for t in target_surnames:
        if t in a:
            print(f"MATCH: {a}")
'''
with open("/tmp/crossref.py", "w") as f:
    f.write(script)
result = terminal(command="python3 /tmp/crossref.py", timeout=10)
```

This technique confirmed Yury Pinsky on Imagen 3 (position 257 of 262) and would catch any other targets who are co-authors.

**Fallback strategy when primary tools fail:**
1. Use Google News RSS with contextual keywords (pattern above) — best free source
2. Use DBLP API for publication verification — more reliable than Semantic Scholar
3. Use Google Developer profile probing for `@google.com` targets — confirms employment
4. Use email domain for company inference (`@google.com` → Google)
5. Use network positioning (who else is in the expansion queue)
6. Cross-reference targets against multi-author paper author lists (pattern above)
7. Proceed with Weave upsert using base identity data
8. Generate HTML report for manual Google Drive upload if Docs API unavailable
9. Use cached data from previous runs when available (check `/root/.hermes/commons/data/ocas-expansion/`)

**Cron-run degradation strategy (April 2026 finding):**

Cron jobs face stricter constraints than interactive runs: HUNTER_API_KEY often unset, Tavily quota already consumed, no browser access, arXiv API may be network-blocked. When running as a cron job:

1. **Check cached data first** — load `complete_dossier.json` and `scout_findings.json` from previous interactive runs. These contain verified data that is still valid.
2. **Budget Tavily calls** — max 2 per cron run, catching HTTP 432 immediately. Do NOT attempt more if quota is exhausted.
3. **Fall back to direct APIs** — GitHub API (`curl`/`urllib`), Google Patents (via curl), Crossref — these have no monthly quota limits.
4. **Domain inference as baseline** — `@google.com` → Google, `@meta.com` → Meta. This is sufficient for Weave upsert with `confidence: 0.3` when no enrichment is available.
5. **Accept shallow data** — A Weave upsert with just name, email, and inferred org is better than no upsert. Position/occupation details can be enriched in future interactive runs.
6. **Skip Google Docs on cron failure** — Save local markdown report and write path to `last_run_report.txt`. OAuth tokens may be expired or unavailable in cron context.

**`real_ladybug` API import (April 2026 cron finding):**

The correct import is `import real_ladybug as lb` (NOT `from real_ladybug import LadybugDB`). The API pattern is:

```python
import real_ladybug as lb

db = lb.Database('/root/.hermes/commons/db/ocas-weave/weave.lbug')
conn = lb.Connection(db)

result = conn.execute("MATCH (p:Person) RETURN count(p) AS cnt")
rows = result.get_all()  # Returns list of lists
cols = result.get_column_names()  # Returns list of column name strings

# Close when done
conn.close()
db.close()
```

Key API details:
- `result.get_all()` returns a list of lists (not dicts). Access by column index: `rows[0][0]`
- `result.get_column_names()` returns column names as strings
- `rows_as_dict()` returns a QueryResult object, NOT a Python dict — do not use it as data
- Use `MATCH ... RETURN p.name, p.occupation` (property selectors) to get column-based results, which map to column names via `get_column_names()`
- Property filtering `{prop: value}` in MATCH patterns is NOT supported — use WHERE clauses instead
- `CREATE` for relationships cannot bind nodes with bare `{{id: $val}}` when MATCH could match multiple labels — use explicit labels like `(a:Person {id: $id})`
- The word `description` can trigger parser errors in CREATE statements — use f-string interpolation with single-quoted string values
- `type(r)` function doesn't exist in LadybugDB — use `r.rel_type` instead
- `confidence` property must be FLOAT, not STRING

**Spurious relationship cleanup pattern (April 2026 cron finding):**

Previous pipeline runs (especially LLM-driven synthesis) can hallucinate invalid `rel_type` values on `Knows` edges — e.g., `wife_of`, `mother_of`, `father_of`, `SPOUSE`, `son_of`, `husband_of`. These are semantically wrong and contaminate the graph. The cleanup query uses WHERE with `OR` to find all family/romantic edge types, then deletes them:

```python
# Find all spurious family/romantic edges
spurious = list(conn.execute("""
    MATCH (a:Person)-[r:Knows]->(b:Person) 
    WHERE r.rel_type = 'husband_of' OR r.rel_type = 'wife_of' OR r.rel_type = 'SPOUSE'
    OR r.rel_type = 'father_of' OR r.rel_type = 'son_of' OR r.rel_type = 'mother_of'
    OR r.rel_type = 'daughter_of'
    RETURN a.name, b.name, r.rel_type, r.context
"""))

# Delete each one (handling None context values)
for row in spurious:
    from_name = row[0].replace("'", "")
    to_name = row[1].replace("'", "")
    rel_type = row[2]
    context = (row[3] or '').replace("'", "").replace('"', '')
    
    conn.execute(f"MATCH (a:Person)-[r:Knows]->(b:Person) WHERE a.name = '{from_name}' AND b.name = '{to_name}' AND r.rel_type = '{rel_type}' AND r.context = '{context}' DELETE r RETURN count(r)")
```

**Incremental refresh strategy (April 2026 cron finding):**

When the same targets are re-queued (e.g., overnight expansion runs on the same 10 contacts), the pipeline should operate as an incremental refresh rather than a full re-run. Key strategies:

1. **Check previous completion data first** — Load `pipeline_completion_{date}.json` and compare target IDs. If all targets match, this is a refresh run.
2. **Load cached Scout data** — Previous `phase1_scout_*.json` files contain DBLP, Semantic Scholar, GitHub, and Google Dev Profile data that doesn't change daily. Reuse it.
3. **Re-query rate-limited sources only** — Semantic Scholar (429 rate limits in previous run) and Google News (time-sensitive) benefit from fresh queries.
4. **Focus Sift deep dive on gap-filling** — The Sift deep dive should target areas where previous data was thin or potentially wrong (disambiguation issues, missing occupations).
5. **Pre-existing data preservation** — Before overwriting Person nodes, check existing Weave values and keep higher-confidence data. Don't overwrite a verified `occupation` with `Unknown`.

**Sift deep dive disambiguation corrections (April 2026 finding):**

The Sift deep dive phase is critical for catching disambiguation errors that API-only Scout research misses. In the April 2026 cron run, two major disambiguation errors were caught:

1. **Marc Paulina** — DBLP showed 79 publications in computational biology/ML, but web research revealed the person at Google is actually a UX Designer for Wear OS. The ML publications belong to a different researcher with a similar name.

2. **Gustavo Moura** — DBLP showed 121 publications and Semantic Scholar showed 19 papers with 203 citations, but these belong to "Gustavo L. C. Moura" (a chemistry/computational researcher). The actual person at Google is a UX/Design professional with an engineering background.

The deep dive caught these errors by checking LinkedIn and Google Developer profiles, which showed UX/design roles rather than research positions. This validates the importance of the Sift phase — DBLP and Semantic Scholar data alone is insufficient for disambiguation.

**Pattern:** When DBLP shows >50 publications for a single name, treat it as a disambiguation flag. Cross-reference with the person's actual role (which Sift web research can confirm) before attributing publications. UX designers rarely have 50+ DBLP publications in ML/biology.

**DuckDuckGo via curl pattern (works when web_search is blocked, but unreliable — CAPTCHA likely):**

```python
from hermes_tools import terminal
import re

query = '"Behshad Behzadi" CTO Sportradar Google'
search_url = f'https://html.duckduckgo.com/html/?q={query.replace(" ", "+")}'
result = terminal(command=f'curl -s -L -A "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36" "{search_url}" | head -300', timeout=15)
html = result.get('output', '')

# Parse snippets
snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html)
titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html)
# Note: DDG HTML often returns CAPTCHA page instead of results. Prefer Google News RSS.
```

**Hunter.io email-verifier endpoint behavior for Google addresses (April 2026 cron finding):**

| Email format | Hunter score | Hunter result | Actual validity |
|---|---|---|---|
| `aakerkar@google.com` (long handle) | 100 | deliverable | ✓ Valid |
| `ypinsky@google.com` (short handle) | 88 | deliverable | ✓ Valid |
| `moura@google.com` (short handle) | 88 | deliverable | ✓ Valid |
| `hadar@google.com` (short handle) | 89 | deliverable | ✓ Valid |
| `rux@google.com` (short handle) | 91 | deliverable | ✓ Valid |
| `behshad@google.com` (short handle) | 0 | undeliverable | ✗ False negative |
| `poh@google.com` (short handle) | 0 | undeliverable | ✗ False negative |
| `onnychatterjee@meta.com` (long handle) | 73 | risky | ⚠ Valid but risky |
| `laith.ulaby@gmail.com` (personal) | 88 | deliverable | ✓ Valid |
| `marcpaulina@googlemail.com` (alias) | 91 | deliverable | ✓ Valid |

Key insight: 2 of 7 short `@google.com` handles returned `score=0/undeliverable` from Hunter.io. These are false negatives — confirmed valid via Google Developer profile probing (HTTP 200). The `score=0` result appears when Google's SMTP rejects Hunter's verification probe. Always cross-reference `undeliverable` results for `@google.com` addresses with Google Developer profile checks before concluding the address is invalid.

**Hunter.io domain search + email-finder for current employer identification (April 2026 cron finding):**

Behshad Behzadi's `@google.com` address returned `undeliverable` from Hunter.io, but a separate `email-finder` query at his current employer `sportradar.com` returned `b.behzadi@sportradar.com` with score=99 and position=CTO. This is the highest-yield Hunter.io technique for people who have changed employers: after email verification fails for the original domain, run a domain-search or email-finder on the suspected new employer domain. This technique confirmed career transition information without any web search.

**Site accessibility matrix (April 2026 observations):**

| Site | Access Status | Notes |
|------|---------------|-------|
| DuckDuckGo HTML | ❌ CAPTCHA | Requires interactive solving |
| DuckDuckGo Lite | ❌ CAPTCHA | Same challenge |
| Google Search | ❌ CAPTCHA | "Unusual traffic detected" |
| Google Scholar | ❌ Login wall | Redirects to sign-in |
| Google News | ✓ Works | Returns results, may be empty |
| **Google News RSS** | **✓ HIGH YIELD** | **`https://news.google.com/rss/search?q=%22Name+Surname%22` — best free source for person OSINT. Returns article titles, dates, and source names. Parse with regex for `<title>` and `<pubDate>`. Works reliably from sandbox/cron environments.** |
| Bing | ❌ CAPTCHA | "Solve the challenge below" |
| Crunchbase | ❌ Cloudflare block | Ray ID logged |
| LinkedIn | ❌ Login wall | All profiles require auth |
| GitHub | ✓ Works | No results for private individuals |
| UX Magazine | ✓ Works | Good for UX research profiles |
| Wikipedia | ✓ Works | Via API or direct URL |
| Jina AI Reader | ⚠️ Mixed | `r.jina.ai/<url>` works for some sites, returns mostly nav/chrome for JS-heavy sites like Sportradar. Best for article-heavy sites like Slator. |
| Google Patents HTML | ❌ Empty results | Direct HTML scraping returns no patent data. Previous runs via Tavily worked; direct curl does not. |
| Semantic Scholar API | ⚠️ Returns EMPTY author stubs | Returns 0-paper/0-citation placeholder entries for many names — these are API artifacts, not real profiles. Treat `paperCount: 0` results as non-matches. Prefer DBLP for publication verification. |
| **DBLP API** | **✓ HIGH YIELD** | **`https://dblp.org/search/publ/api?q=author:{name}&format=json&h=5` — returns actual publications with venues, years, and author lists. More reliable than Semantic Scholar for tech/CS researchers. Works from cron/sandbox environments.** |
| **Google Developer Profile** | **✓ HIGH YIELD (Google employees only)** | **`https://developers.google.com/profile/u/{email_username}` — returns 200 if profile exists, 404 if not. Confirms Google employment for `@google.com` targets. Works from cron/sandbox environments via `urllib`.** |
| Sportradar.com | ❌ Blocks Jina | Returns generic homepage/region selector instead of article content. Not extractable via Jina Reader. |
| Blog.google | ⚠️ URL rot | Specific post slugs 404 over time. Blog restructuring removes old URLs. Try Google Blog search instead. |

**Cached data reuse pattern:**

```python
import json
from pathlib import Path

expansion_dir = Path('/root/.hermes/commons/data/ocas-expansion')

# Load previous research results
if (expansion_dir / 'scout_research_final.json').exists():
    with open(expansion_dir / 'scout_research_final.json') as f:
        cached_scout = json.load(f)
    print(f"Using cached Scout data: {len(cached_scout)} profiles")

# Load previous Weave synthesis
if (expansion_dir / 'weave_synthesis.json').exists():
    with open(expansion_dir / 'weave_synthesis.json') as f:
        cached_weave = json.load(f)
    print(f"Using cached relationships: {cached_weave.get('relationships_created', 0)}")

# Compile report from cached data when fresh searches fail
# This allows meaningful output even with zero API calls
```

**HTML Report Fallback Pattern (when Google Docs OAuth unavailable):**

```python
html_report = f'''<!DOCTYPE html>
<html>
<head>
    <title>Graph Expansion Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; }}
        h1 {{ color: #1a73e8; border-bottom: 2px solid #1a73e8; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; }}
        th {{ background-color: #1a73e8; color: white; }}
    </style>
</head>
<body>
    <h1>Graph Expansion Pipeline Report</h1>
    <p><strong>Run ID:</strong> {run_id}</p>
    <p><strong>Targets:</strong> {len(targets)}</p>
    [Continue with full report content...]
</body>
</html>
'''

html_file = f'/root/.hermes/commons/data/ocas-expansion/{run_id}-report.html'
with open(html_file, 'w') as f:
    f.write(html_report)

# User can upload to Google Drive and open with Google Docs
```

## Weave Schema (Person Nodes)

Discovered through inspection — do not assume properties. Query existing nodes first:

```python
import real_ladybug as lb
from pathlib import Path

DB_PATH = Path("/root/.hermes/commons/db/ocas-weave/weave.lbug")
db = lb.Database(str(DB_PATH), read_only=True)
conn = lb.Connection(db)

# Inspect existing Person schema
sample = list(conn.execute("MATCH (p:Person) RETURN p LIMIT 1"))
print(f"Properties: {sample[0][0].keys()}")

conn.close()
db.close()
```

**Confirmed Person properties:**
- `id` (string, primary key)
- `name` (string, full name)
- `name_given` (string, first name)
- `name_family` (string, last name)
- `email` (string)
- `phone` (string, optional)
- `location_city` (string, optional)
- `location_country` (string, optional)
- `occupation` (string, optional) — **USE THIS for job title, NOT `position`**
- `org` (string, organization/company) — **USE THIS for company, NOT `company`**
- `notes` (string, optional)
- `source_type` (string: imported/inferred/direct/user-stated)
- `source_ref` (string, source file or reference)
- `confidence` (float, 0.0–1.0)
- `record_time` (ISO 8601 timestamp)
- `event_time` (ISO 8601, optional)
- `valid_from` / `valid_until` (ISO 8601, optional)
- `google_resource_name` (string, for Google Contacts sync)
- `clay_id` (string, optional)

**Do NOT use:** `domain`, `last_updated`, `created_at`, `updated_at`, `position`, `company` — these do not exist. Use `org` for company and `occupation` for position/title.

### Critical: Confidence Property Type

The `confidence` property is stored as **FLOAT**, not STRING. Do not use string values like `'med'` or `'low'`:

```python
# WRONG - causes "Expression med has data type STRING but expected DOUBLE"
SET p.confidence = 'med'

# CORRECT - use numeric values
SET p.confidence = 0.6  # medium confidence
SET p.confidence = 0.3  # low confidence
SET p.confidence = 0.9  # high confidence

# Or map from string labels
confidence_map = {'high': 0.9, 'med': 0.6, 'low': 0.3}
confidence_num = confidence_map.get(confidence_str, 0.5)
```

### Critical: Duplicate Knows Edge Prevention (April 2026 finding)

LadybugDB's `MERGE` on `Knows` edges matches on **all specified properties** in the pattern. This means `MERGE (a)-[r:Knows {rel_type: 'colleague'}]->(b)` creates a **separate edge** from `MERGE (a)-[r:Knows {rel_type: 'former_colleague'}]->(b)` or `MERGE (a)-[r:Knows]->(b)` (no rel_type). The result is **duplicate/triplicate edges** between the same pair of people.

**Observed:** 118 internal relationships between 10 targets — many pairs had 2 or 3 parallel `Knows` edges with different `rel_type` values (e.g., both `None` and `colleague`, or `colleague`, `former_colleague`, and `None`).

**Fix: Always include `rel_type` in MERGE patterns and clean up duplicates before synthesis:**

```python
# CORRECT — always specify rel_type in the MERGE pattern
MERGE (p1)-[r:Knows {rel_type: 'colleague'}]->(p2)
ON CREATE SET r.confidence = 0.7, r.record_time = '{record_time}'
ON MATCH SET r.confidence = 0.7, r.record_time = '{record_time}'

# Cleanup query — delete edges with NULL rel_type (duplicate of typed edge)
# Run this BEFORE creating new relationships
conn.execute('''
    MATCH (p1:Person)-[r:Knows]->(p2:Person)
    WHERE r.rel_type IS NULL
    DELETE r
''')

# Also delete edges where the same pair has multiple edges of different rel_types
# Keep only the most specific one (colleague > None)
conn.execute('''
    MATCH (p1:Person)-[r1:Knows]->(p2:Person),
          (p1)-[r2:Knows]->(p2)
    WHERE r1.rel_type = 'colleague' AND r2.rel_type IS NULL
    DELETE r2
''')
```

**Notes field dedup pattern (April 2026 finding):**

Multiple pipeline runs accumulate duplicate "Flags:" entries in Person `notes` fields. Simple `'; '.join(set(parts))` fails because partial overlaps (e.g., "Flags: ux_specialist" vs "Flags: ux_specialist, ml_specialist") are not exact duplicates. The correct pattern:

1. Split notes on `'; '`
2. For each part, check if it starts with `'Flags: '` or `'flags: '`
3. If it does, extract the flag values into a flat `set()` across all parts
4. For non-flag parts, deduplicate by normalized (lowercased, stripped) content
5. Rebuild: `; Flags: {merged_sorted_flags}; {unique_non_flag_parts}`

Apply this pattern during Phase 2 upsert when merging run data with existing Weave notes.

**Pre-synthesis dedup check:** Before Phase 4 (Weave Synthesis), run:
```python
# Check for duplicate edges between same pair
result = list(conn.execute('''
    MATCH (p1:Person)-[r:Knows]->(p2:Person)
    WHERE p1.id < p2.id
    RETURN p1.name, p2.name, count(r) AS edge_count
    ORDER BY edge_count DESC
    LIMIT 20
'''))
for row in result:
    if row[2] > 1:
        print(f"WARNING: {row[0]} -> {row[1]} has {row[2]} edges (expected 1)")
```

**Exact-duplicate edge dedup (April 2026 finding):** LadybugDB can create two or more edges with the *identical* `rel_type` between the same pair (e.g., two separate `colleague` edges). The priority-based dedup pattern (keep `coauthor`, delete `colleague`) doesn't handle this because both edges have the same type. Fix:

```python
# Detect exact duplicates: same pair, same rel_type, count > 1
duplicates = list(conn.execute('''
    MATCH (p1:Person)-[r:Knows]->(p2:Person)
    WHERE p1.id IN [target_ids]
    AND p2.id IN [target_ids]
    RETURN p1.id, p1.name, p2.id, p2.name, r.rel_type, count(r) AS cnt
    ORDER BY p1.name, p2.name, r.rel_type
'''))

for row in duplicates:
    p1_id, p1_name, p2_id, p2_name, rel_type, cnt = row
    if cnt > 1:
        # Delete ALL edges of this type between this pair, then recreate one
        conn.execute(f'''
            MATCH (p1:Person {{id: '{p1_id}'}})-[r:Knows]->(p2:Person {{id: '{p2_id}'}})
            WHERE r.rel_type = '{rel_type}'
            DELETE r
        ''')
        conn.execute(f'''
            MATCH (p1:Person {{id: '{p1_id]'}}), (p2:Person {{id: '{p2_id}'}})
            CREATE (p1)-[r:Knows {{rel_type: '{rel_type}', confidence: 0.7,
                context: 'Same organization', source_ref: 'expansion_pipeline_synthesis',
                record_time: '{record_time}'}}]->(p2)
        ''')
```

Run this AFTER the priority-based dedup pass, which only removes *different*-type duplicate edges.

**MERGE ON MATCH behavior:** Use `ON MATCH SET` to update existing edges rather than creating new ones. Without `ON MATCH`, re-running the same MERGE creates no new edge but also doesn't update properties.

**MERGE on Knows with different rel_type throws duplicate primary key (April 2026 finding)** — When a `Knows {rel_type: 'colleague'}` edge already exists between two people, attempting `MERGE (p1)-[r:Knows {rel_type: 'coauthor'}]->(p2)` throws `"Found duplicated primary key value"` because LadybugDB's primary key includes the direction (from → to) irrespective of rel_type. You cannot have two outgoing Knows edges from the same person to the same person, regardless of rel_type differences. **Fix:** Instead of MERGE for edge type upgrades (colleague → coauthor), use `SET` on the existing edge to change its properties:

```python
# WRONG — throws "Found duplicated primary key value" if any Knows edge exists already
MERGE (p1:Person {id: 'X'})-[r:Knows {rel_type: 'coauthor'}]->(p2:Person {id: 'Y'})

# CORRECT — find existing edge and upgrade it
MATCH (p1:Person {id: 'X'})-[r:Knows]->(p2:Person {id: 'Y'})
WHERE r.rel_type = 'colleague'
SET r.rel_type = 'coauthor', r.context = '...', r.confidence = 0.8
```

This is LadybugDB-specific behavior (standard Neo4j would allow multiple edges with different properties). All edge upgrades should use MATCH+SET rather than MERGE.

### Pre-existing data preservation (April 2026 finding)

When targets already exist in Weave (common for re-runs and periodic enrichment), the pipeline must **check current Person node values before overwriting**. Specifically:

1. Query each target's current `occupation` and `confidence` from Weave BEFORE the upsert phase
2. If Weave has a non-empty `occupation` and the OSINT run discovered nothing (or found a less specific value), **keep the Weave value** and boost confidence to at least 0.6 (reflecting that it was previously verified)
3. If Weave has a higher `confidence` than OSINT would set, **keep the higher value** — OSINT missing a role doesn't mean the role is wrong
4. Never overwrite `org` with `Unknown` if Weave already has a known org (e.g., laith Ulaby had `Google` in Weave but gmail.com domain inference yielded `Unknown`)

**Pattern:**
```python
# Before upsert, check existing data
existing = list(conn.execute(f"MATCH (p:Person {{id: '{tid}'}}) RETURN p.occupation, p.confidence, p.org"))
if existing:
    current_occ = existing[0][0] or ''
    current_conf = existing[0][1] or 0.0
    current_org = existing[0][2] or ''
    
    # Preserve verified occupation
    if current_occ and not osint_occupation:
        osint_occupation = current_occ
    
    # Preserve higher confidence
    if current_conf > osint_confidence:
        osint_confidence = current_conf
    
    # Don't overwrite known org with "Unknown"
    if current_org and current_org != 'Unknown' and osint_org == 'Unknown':
        osint_org = current_org
```

**Observed:** Onny Chatterjee, Ankita Akerkar, and laith Ulaby all had specific verified occupations in Weave (Pathfinding UX Researcher, UX/Product Designer, Director of UX Research) that the OSINT run failed to discover. The initial upsert would have left their occupations empty and confidence at 0.3, downgrading what Weave already knew.

**`@googlemail.com` is a Gmail alias used by Googlers (April 2026 finding):** Some Google employees use `@googlemail.com` addresses instead of `@google.com`. Domain inference treats these as "Unknown" since it's technically a consumer Gmail domain, but the person is at Google. Marc Paulina (`marcpaulina@googlemail.com`) is an EMEA UX Lead at Google. Always treat `@googlemail.com` the same as `@google.com` for employer inference.

**Hunter.io email verification of short-handle @google.com addresses (April 2026 finding):**

Hunter.io's email-verifier endpoint returns `score=0` and `result=undeliverable` for short-handle Google email addresses (e.g., `behshad@google.com`, `poh@google.com`). This is a false negative — these addresses are valid Google employee emails confirmed by Google Developer profile probes (returning HTTP 200). Longer handles verify correctly (e.g., `aakerkar@google.com` → score=100, `ypinsky@google.com` → score=88). The likely cause is Google's SMTP server rejecting verification probes for short aliases. When Hunter.io email verification returns `undeliverable` for an `@google.com` address, cross-reference with Google Developer profile probing (`developers.google.com/profile/u/{handle}`) before concluding the email is invalid. If the dev profile returns 200, the email is valid despite the Hunter score.

**Hunter.io confidence scoring (April 2026 finding):**

Hunter.io email-finder returns a `score` field (0-100). Map to Weave confidence:

| Hunter Score | Weave Confidence | Meaning |
|--------------|------------------|---------|
| ≥90 | 0.9 (high) | Email verified with high certainty |
| 50-89 | 0.6 (med) | Email likely valid |
| <50 | 0.3 (low) | Uncertain verification |

**April 2026 run results:**
- Ankita Akerkar: score=98 → high (also returned position: "UX Designer")
- Russell Matsuo: score=96 → high
- Peter Oh: score=99 → high
- Onny Chatterjee: score=48 → low

Position data is returned when available via `data['data']['position']`.

## Weave Schema (Fact Nodes)

Fact nodes store provenance-backed claims about people. Unlike Person notes (free text), Facts are structured and queryable.

**Confirmed Fact properties:**
- `id` (string, unique identifier — use descriptive prefixes like `fact-shemtov-gemini`)
- `predicate` (string: published/research_area/team_member/career_transition/works_at/etc.)
- `value` (string, the claim content)
- `confidence` (float, 0.0–1.0)
- `source_type` (string: research/inferred/imported/direct)
- `source_ref` (string, source URL or reference)
- `record_time` (ISO 8601 timestamp)

**Do NOT use:** `fact_id` — the property is `id`, not `fact_id`.

**Creation pattern:**

```python
# Create Fact AND link to Person in one query
create = """
CREATE (fact:Fact {
    id: $id,
    predicate: $predicate,
    value: $value,
    confidence: $confidence,
    source_type: $source_type,
    source_ref: $source_ref,
    record_time: $record_time
})
WITH fact
MATCH (p:Person {id: $person_id})
CREATE (p)-[:HasFact]->(fact)
RETURN p.name AS name, fact.id AS fact_id
"""
result = conn.execute(create, params)
```

**Useful predicates for expansion pipeline:**
- `published` — paper/book/patent the person authored
- `research_area` — field or topic they work in
- `team_member` — team or project they belong to
- `career_transition` — significant career change (with explanation in value)

## Weave Schema (Knows Relationships)

```python
# Inspect existing Knows relationships
rels = list(conn.execute("MATCH (p1:Person)-[r:Knows]->(p2:Person) RETURN r LIMIT 1"))
print(f"Relationship properties: {rels[0][0].keys() if rels else 'No relationships'}")
```

**Confirmed Knows properties:**
- `rel_type` (string: colleague/friend/family/etc.)
- `since` (ISO 8601 timestamp)
- `context` (string, how they know each other)
- `source_ref` (string, source of relationship data)
- `confidence` (float, 0.0–1.0)
- `record_time` (ISO 8601 timestamp)

**Do NOT use:** `source` — use `source_ref` instead.

## Upsert Pattern

```python
import real_labybug as lb
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path("/root/.hermes/commons/db/ocas-weave/weave.lbug")
db = lb.Database(str(DB_PATH), read_only=False)
conn = lb.Connection(db)

for target in targets:
    person_id = target['id']
    name = target['name'].strip()
    email = target['email']
    record_time = datetime.now(timezone.utc).isoformat()
    
    # Parse name
    name_parts = name.split()
    name_given = name_parts[0] if name_parts else name
    name_family = ' '.join(name_parts[1:]) if len(name_parts) > 1 else ''
    
    # Extract org from email domain
    domain = email.split('@')[1] if '@' in email else ''
    
    # Escape single quotes
    safe_name = name.replace("'", "''")
    
    cypher = f"""
    MERGE (p:Person {{id: '{person_id}'}})
    SET p.name = '{safe_name}',
        p.name_given = '{name_given}',
        p.name_family = '{name_family}',
        p.email = '{email}',
        p.org = '{domain}',
        p.source_type = 'imported',
        p.source_ref = 'expansion_queue.jsonl',
        p.confidence = 0.3,
        p.record_time = '{record_time}'
    RETURN p.id, p.name
    """
    
    result = list(conn.execute(cypher))
    if result:
        print(f"✓ Upserted: {name}")

conn.close()
db.close()
```

## Relationship Creation Pattern

```python
# Create colleague relationships for people at same organization
record_time = datetime.now(timezone.utc).isoformat()

for i, id1 in enumerate(org_member_ids):
    for id2 in org_member_ids[i+1:]:
        rel_cypher = f"""
        MATCH (p1:Person {{id: '{id1}'}})
        MATCH (p2:Person {{id: '{id2}'}})
        MERGE (p1)-[r:Knows {{rel_type: 'colleague'}}]->(p2)
        ON CREATE SET r.since = '{record_time}',
                      r.context = 'Same organization: {org_name}',
                      r.source_ref = 'expansion_pipeline_synthesis',
                      r.confidence = 0.7,
                      r.record_time = '{record_time}'
        RETURN p1.name, p2.name
        """
        result = list(conn.execute(rel_cypher))
        if result:
            print(f"  ✓ {name1} --[colleague]--> {name2}")
```

## Google Drive Upload Patterns

### Option A: rclone (Recommended — works reliably)

```python
from hermes_tools import terminal
import json

# Upload markdown report to Google Drive
report_file = "/root/.hermes/commons/data/ocas-expansion/expansion_report.md"
terminal(f"rclone copy {report_file} GDrive:/hermes-reports/")

# Get file ID for Google Doc link
result = terminal(f"rclone lsjson GDrive:/hermes-reports/expansion_report.md")
file_info = json.loads(result['output'])
file_id = file_info[0].get('ID', '')
drive_link = f"https://docs.google.com/document/d/{file_id}/edit"

# Save link
with open("/root/.hermes/commons/data/ocas-expansion/last_run_report.txt", "w") as f:
    f.write(drive_link)
```

**Advantages:** No OAuth token management, works with existing rclone GDrive remote, handles file ID extraction automatically.

### Option B: Google Docs API (User OAuth token)

Use user OAuth token (NOT service account) — service accounts fail with Docs API:

```python
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# Load credentials (user OAuth, NOT service account)
with open('/root/.hermes/google_token.json') as f:
    token_data = json.load(f)

creds = Credentials(
    token=token_data.get('token'),
    refresh_token=token_data.get('refresh_token'),
    token_uri='https://oauth2.googleapis.com/token',
    client_id=token_data['client_id'],
    client_secret=token_data['client_secret'],
    scopes=token_data.get('scopes', []),
)

if not creds.valid and creds.refresh_token:
    creds.refresh(Request())

docs_service = build('docs', 'v1', credentials=creds)
drive_service = build('drive', 'v3', credentials=creds)

# Create document
doc_title = f"Graph Expansion Pipeline Report - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
doc = docs_service.documents().create(body={'title': doc_title}).execute()
doc_id = doc['documentId']

# Build content as list of lines (simpler than complex styling)
content_lines = [
    doc_title,
    "",
    f"Run Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
    "",
    "EXECUTIVE SUMMARY",
    "[Summary text...]",
    "",
    "TARGET ROSTER",
    "[Target list...]",
]

# Build batch update requests - simple sequential insertText
requests = []
current_index = 1  # Start after document body start

for line in content_lines:
    text = line + '\n'
    requests.append({
        'insertText': {
            'location': {'index': current_index},
            'text': text
        }
    })
    current_index += len(text)

docs_service.documents().batchUpdate(documentId=doc_id, body={'requests': requests}).execute()

# Get shareable link
drive_file = drive_service.files().get(fileId=doc_id, fields='webViewLink').execute()
shareable_link = drive_file.get('webViewLink')
```

**Important:**
- Do NOT use `namedStyleType` for styling — field does not exist in Docs API v1, causes `HttpError 400`
- Do NOT use complex text styling — stick to plain `insertText` operations
- Sequential `insertText` with calculated indices works fine when you update `current_index` after each insert
- If a run fails partway, reuse the `doc_id` from the failed run — document persists even if batchUpdate fails
- Save the doc_id and link to `last_run_report.txt` for recovery
- **Scope validation required** — token must include `https://www.googleapis.com/auth/documents` scope

**Working pattern confirmed (April 2026):**
```python
# Load report content from markdown file
with open('/root/.hermes/commons/data/ocas-expansion/run_report.md', 'r') as f:
    report_md = f.read()

# Create document
title = f"Graph Expansion Pipeline Report — 2026-04-09"
doc = docs_service.documents().create(body={'title': title}).execute()
doc_id = doc['documentId']

# Insert content at index 1 (start of fresh document body)
# NOTE: Do NOT use body_start or body_start - 1 — it throws
# "insertion index must be inside the bounds of an existing paragraph"
# on some documents. Index 1 on a fresh document always works.
requests = [{
    'insertText': {
        'location': {'index': 1},
        'text': report_md
    }
}]

# Execute batch update
docs_service.documents().batchUpdate(documentId=doc_id, body={'requests': requests}).execute()

# Get shareable link
doc_url = f"https://docs.google.com/document/d/{doc_id}/edit"

# Save link
with open('/root/.hermes/commons/data/ocas-expansion/last_run_report.txt', 'w') as f:
    f.write(doc_url)
```

This pattern successfully created a 4000+ character report with proper line breaks preserved.

**April 2026 Run Notes:**
- Token at `/root/.hermes/google_token.json` was valid and not expired
- Docs API access worked reliably with user OAuth (NOT service account)
- Sequential `insertText` with index calculation worked without errors
- Report successfully created: https://docs.google.com/document/d/1aQP_qIyY8BQKweiFAsOv_LhTdF8pa01e6Ymp6V5m6Wk

**Scope Error Handling (April 2026 finding):**
`drive.file` scope is sufficient for creating new Google Docs (confirmed). Do NOT fall back just because `documents` scope is absent. Only fall back on `RefreshError` or genuine auth failures:

```python
from google.auth.exceptions import RefreshError

try:
    doc = docs_service.documents().create(body={'title': title}).execute()
    # ... proceed with normal flow
except RefreshError as e:
    error_msg = str(e)
    # Only fall back on genuine auth failure, NOT missing 'documents' scope
    # drive.file is sufficient for creating new docs
    report_path = '/root/.hermes/commons/data/ocas-expansion/graph_expansion_report.md'
    with open(report_path, 'w') as f:
        f.write(markdown_report)
    with open('/root/.hermes/commons/data/ocas-expansion/last_run_report.txt', 'w') as f:
        f.write(f'file://{report_path}')
    print(f'Report saved locally (Google auth failed): {report_path}')
```

**Scope validation before run:**
`drive.file` scope is sufficient for creating new Google Docs (confirmed April 2026). The explicit `documents` scope is NOT required. Only fall back to local if the token is completely invalid or refresh fails.

```python
with open('/root/.hermes/google_token.json') as f:
    token_data = json.load(f)
scopes = token_data.get('scopes', [])
has_drive_scope = any('drive' in s for s in scopes)
if not has_drive_scope:
    print("WARNING: Token lacks any Drive scope. Report will be saved locally.")
    use_google_docs = False
# drive.file is sufficient — do NOT require explicit documents scope
```

## Output Files

- `/root/.hermes/commons/data/ocas-expansion/scout_findings.json` — Scout research results (confidence-scored)
- `/root/.hermes/commons/data/ocas-expansion/sift_findings.json` — Deep dive findings
- `/root/.hermes/commons/data/ocas-expansion/complete_dossier.json` — Merged Scout + Sift data
- `/root/.hermes/commons/data/ocas-expansion/final_weave_synthesis.json` — Synthesis with classification flags
- `/root/.hermes/commons/data/ocas-expansion/graph_expansion_report.md` — Full markdown report
- `/root/.hermes/commons/data/ocas-expansion/last_run_report.txt` — Google Doc link or local path

## Synthesis Flags Pattern

Assign classification flags to entities based on discovered attributes for quick filtering:

```python
synthesis_flags = []

if "patent" in findings.lower():
    synthesis_flags.append("patent_holder")
if "phd" in findings.lower():
    synthesis_flags.append("advanced_degree")
if "gemini" in findings.lower():
    synthesis_flags.append("frontier_model_contributor")
if "infrastructure" in findings.lower() or "HPC" in findings:
    synthesis_flags.append("infra_specialist")
if "robotics" in findings.lower() or "UAV" in findings:
    synthesis_flags.append("robotics_researcher")
if "NLP" in findings or "query" in findings.lower():
    synthesis_flags.append("nlp_specialist")

# Store with entity record
entity["synthesis_flags"] = synthesis_flags
```

**High-value flag combinations (April 2026 findings):**
- `patent_holder + nlp_specialist` → Technical leadership candidate
- `advanced_degree + frontier_model_contributor` → Research scientist
- `infra_specialist` → Platform/engineering hub node
- `robotics_researcher` → Specialized domain expert

**Confidence scoring guidance:**
- `high` (0.9): Verified identity with multiple corroborating sources (e.g., Hadar Shemtov — arXiv + ACL Anthology + email match)
- `med` (0.6): Strong candidate match with some uncertainty (e.g., Gustavo Moura — GitHub profile matches domain/location but no direct email verification)
- `low` (0.3): Email domain inference only, no public footprint (8 of 10 targets in April 2026 run)

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Hunter.io returns 403 Forbidden | Monthly quota exhausted or account issue. Do not retry. Fall back to cached data and domain inference. |
| `os.environ.get('HUNTER_API_KEY')` returns empty in cron job | Cron environment doesn't load `.env`. Manually load from `~/.hermes/.env` by parsing the file. |
| Web search returns "Payment Required: Insufficient credits" | Firecrawl credits exhausted. Continue with base data (name, email, domain inference). Do not retry. |
| LadybugDB "Cannot find property X" | Properties are dynamic — query existing nodes first: `MATCH (p:Person) RETURN p LIMIT 1`. Only use properties that exist on existing nodes, or use `ON CREATE SET` for new properties. |
| Google Docs batch insert fails with index error | Use single `insertText` at index 1 with all content. Do not calculate indices or use multiple insert operations. |
| Google Docs service account auth fails | Service account credentials don't work for Docs API. Use user OAuth token at `/root/.hermes/google_token.json` instead. |
| Weave upsert succeeds but verification shows old data | LadybugDB may have lock contention. Close connection and reopen for verification. |
| Semantic Scholar returns matches with 0 papers | These are empty author stubs (API artifacts), not real profiles. Filter out any result where `paperCount == 0`. Use DBLP instead for publication verification. |
**Google Docs API unavailable** — Fall back to local markdown report. Save to `/root/.hermes/commons/data/ocas-expansion/graph_expansion_report.md` and write the path to `last_run_report.txt`.

**Google Docs creation may fail with `ACCESS_TOKEN_SCOPE_INSUFFICIENT` (April 2026 cron finding)** — Even when the token at `/root/.hermes/google_token.json` exists, `documents.create()` can return HTTP 403 with `ACCESS_TOKEN_SCOPE_INSUFFICIENT`. This happens when the token's scopes don't include `https://www.googleapis.com/auth/documents`. The `drive.file` scope alone is NOT always sufficient — it depends on how the token was created. When this error occurs, fall back immediately to `rclone copy` (which uploads the markdown file to Google Drive) or local file saves.

**rclone upload as reliable fallback for report generation (April 2026 cron finding)** — When Google Docs API fails with 403, `rclone copy {report_file} GDrive:/hermes-reports/` successfully uploads the markdown file. Then `rclone lsjson` retrieves the file ID for the shareable link. This is more reliable than the Docs API for cron jobs where OAuth scopes may be insufficient. Use this pattern:

```python
from hermes_tools import terminal

# Upload to Google Drive via rclone
result = terminal(command=f"rclone copy {report_path} GDrive:/hermes-reports/ --verbose", timeout=30)

# Get file ID for shareable link
result2 = terminal(command="rclone lsjson GDrive:/hermes-reports/graph_expansion_report_YYYY-MM-DD.md", timeout=15)
file_info = json.loads(result2['output'])
file_id = file_info[0]['ID']
report_link = f"https://drive.google.com/file/d/{file_id}/view"

# Save link
with open(expansion_dir / 'last_run_report.txt', 'w') as f:
    f.write(report_link)
```

**rclone is available in the sandbox environment** and the `GDrive:` remote is pre-configured.

**rclone upload as reliable fallback (April 2026 cron finding)** — When Google Docs API fails, `rclone copy {report_file} GDrive:/hermes-reports/` successfully uploads the markdown report. The GDrive remote must be configured (`rclone listremotes` shows `GDrive:`). This is more reliable than the Docs API for cron jobs where OAuth scopes may be insufficient. Report link becomes `file:///{local_path}` rather than a Google Doc URL.
| LadybugDB query results are node/edge objects | `list(conn.execute(...))` returns tuples of Cypher objects, not flat dicts. Access node properties with `dict(row[0])` or use direct property access in the RETURN clause (e.g., `RETURN p.name, p.occupation` instead of `RETURN p`). |
| Hunter.io email-finder returns 403 error code 1010 | Cloudflare/WAF block, not quota exhaustion. Domain-search still works. Fall back to domain-search for org identification and cached data for position details. |
| Duplicate Knows edges between same pair | LadybugDB MERGE on `Knows {rel_type: X}` creates separate edges per `rel_type` value. Always include `rel_type` in MERGE patterns. Run dedup cleanup before synthesis (delete `NULL` rel_type edges when a typed edge exists between same pair). See "Duplicate Knows Edge Prevention" section. |
| LadybugDB `type(r)` function doesn't exist | LadybugDB does not support the Cypher `type()` function for relationship types. `RETURN type(r)` throws "Catalog exception: function TYPE does not exist." Use `r.rel_type` instead — it returns the relationship type stored as a property. All queries that need the edge type should reference `r.rel_type` directly. |
| LadybugDB Fact MERGE on wrong property | Fact nodes use `id` as unique key, NOT `content`. `MERGE (f:Fact {content: '...'})` throws `"Binder exception: Cannot find property content for f."`. Always use `MERGE (f:Fact {id: 'fact-...'})` with `SET f.predicate = '...', f.value = '...'`. |
| LadybugDB `real_ladybug` API | Import as `import real_ladybug`. Use `real_ladybug.Database(path)` and `real_ladybug.Connection(db)`. Not `from real_ladybug import connect` or `real_labybug` (typo). **Do NOT use `read_only=True`** — `Database(path, read_only=True)` throws `RuntimeError: Cannot create an empty database under READ ONLY mode` even for existing databases. Always use `Database(path)` without `read_only`. |
| Google Docs service account read-only | OCIGCP service account can read existing docs (`documents.get`) but cannot create (`documents.create` → 403) or write (`batchUpdate` → 403). Use user OAuth token at `/root/.hermes/google_token.json` for write access, or fall back to local markdown. |
| Google Docs `ACCESS_TOKEN_SCOPE_INSUFFICIENT` | Token at `/root/.hermes/google_token.json` may lack `documents` scope even if `drive.file` is present. Fall back to `rclone copy {report_file} GDrive:/hermes-reports/` for upload, or save locally and write `file://` path to `last_run_report.txt`. |
| LadybugDB `read_only=True` RuntimeError | `lb.Database(path, read_only=True)` throws `RuntimeError: Cannot create an empty database under READ ONLY mode` even for existing databases. Always use `lb.Database(path)` without `read_only`. |
| SearXNG 502 Bad Gateway | When the SearXNG service underlying `web_search` is down, all search queries fail with `SearXNG failed: Server error '502 Bad Gateway'`. This is separate from credit exhaustion (Payment Required). Fall back to direct API calls (GitHub, DBLP, Semantic Scholar, Crossref, arXiv, Google News RSS). |

## LadybugDB Direct Access Pattern (Confirmed Working)

```python
from real_ladybug import Database, Connection
from pathlib import Path

DB_PATH = Path("/root/.hermes/commons/db/ocas-weave/weave.lbug")

# Initialize database and connection
db = Database(str(DB_PATH))
db.init_database()
conn = Connection(db)

# Execute Cypher
result = conn.execute("MATCH (p:Person) RETURN count(p) AS total")
print(f"Total Person nodes: {list(result)}")

# Always close connections
conn.close()
db.close()
```

**Critical: Property Names and Types**

- `confidence` is **FLOAT** (0.0–1.0), NOT STRING — using `'med'` or `'low'` causes "Expression med has data type STRING but expected DOUBLE"
- Use `org` for company/organization, NOT `company` or `organization`
- Use `occupation` for job title, NOT `position`
- Use `source_ref` for source reference, NOT `source`

**Confirmed Person properties (April 2026):** `id`, `name`, `name_given`, `name_family`, `email`, `phone`, `location_city`, `location_country`, `occupation`, `org`, `notes`, `source_type`, `source_ref`, `confidence`, `record_time`, `event_time`, `valid_from`, `valid_until`, `google_resource_name`, `clay_id`

**String Escaping in Cypher:**

```python
# Single quotes must be escaped by doubling
name_escaped = name.replace("'", "''")

# For notes/summary fields, strip quotes entirely to avoid parser issues
summary_clean = summary.replace("'","").replace('"', '')
```

**Parameterized queries (PREFERRED for MATCH/SET):** `conn.execute()` accepts a dict as the second argument for parameterized queries. This avoids f-string quoting issues entirely and is the safer pattern for reads and updates:

```python
# PREFERRED — parameterized (no f-string, no escaping)
conn.execute(
    "MATCH (p:Person {id: $id}) SET p.occupation = $occ, p.confidence = $conf RETURN p.name",
    {'id': target_id, 'occ': occupation, 'conf': confidence_num}
)

# PREFERRED — parameterized edge check
conn.execute(
    "MATCH (p1:Person {id: $id1})-[r:Knows]->(p2:Person {id: $id2}) RETURN r.rel_type",
    {'id1': m1_id, 'id2': m2_id}
)

# PREFERRED — parameterized edge update
conn.execute(
    "MATCH (p1:Person {id: $id1})-[r:Knows]->(p2:Person {id: $id2}) SET r.rel_type = $rt, r.context = $ctx, r.record_time = $time",
    {'id1': id1, 'id2': id2, 'rt': 'colleague', 'ctx': context, 'time': record_time}
)

# WARNING: CREATE with parameterized relationship properties may not work.
# For CREATE operations, f-strings are still needed but be careful with quoting.
# Use the parameterized approach for all MATCH/RETURN/SET operations.
```

This pattern was validated in April 2026 cron runs and eliminated the `id1]` vs `id1}` typo class of f-string errors entirely.

**NoneType dict.get() pitfall (April 2026):**

When a section of findings is explicitly set to `None` (e.g., Hunter.io returns 403 and the function returns `None`), `dict.get('hunter', {}).get('score', 0)` **crashes with AttributeError**. This is because `dict.get()` returns the default only if the key is *missing* — not if the key exists with value `None`. So `findings.get('hunter', {})` returns `None`, and then `.get('score', 0)` fails on NoneType.

```python
# WRONG — crashes when findings['hunter'] is None (not missing)
if findings.get('hunter', {}).get('score', 0) >= 90:
    confidence = 'high'

# CORRECT — use conditional attribute access or explicit None check
hunter_data = findings.get('hunter') or {}
if hunter_data.get('score', 0) >= 90:
    confidence = 'high'

# ALTERNATIVE — check for None explicitly
if findings.get('hunter') is not None and findings['hunter'].get('score', 0) >= 90:
    confidence = 'high'
```

This pattern applies anywhere a nested dict access chains through a value that may be explicitly `None` rather than an empty dict.

**Upsert Pattern:**

```python
from datetime import datetime, timezone

record_time = datetime.now(timezone.utc).isoformat()
confidence_map = {'high': 0.9, 'med': 0.6, 'low': 0.3}
confidence_num = confidence_map.get(confidence_str, 0.5)

cypher = f"""
MERGE (p:Person {{id: '{person_id}'}})
SET p.name = '{name_escaped}',
    p.email = '{email}',
    p.org = '{org}',
    p.source_type = 'imported',
    p.source_ref = 'expansion_pipeline',
    p.confidence = {confidence_num},
    p.record_time = '{record_time}'
"""
result = conn.execute(cypher)
```

## Google Docs Fallback Pattern

When service account auth fails (returns `id_token` but no `access_token`), fall back to local markdown:

```python
# Attempt Google Docs creation
try:
    doc = docs_service.documents().create(body={'title': title}).execute()
    # ... proceed with normal flow
except google.auth.exceptions.RefreshError as e:
    if 'No access token in response' in str(e):
        # Fall back to local markdown
        report_path = '/root/.hermes/commons/data/ocas-expansion/graph_expansion_report.md'
        with open(report_path, 'w') as f:
            f.write(markdown_report)
        with open('/root/.hermes/commons/data/ocas-expansion/last_run_report.txt', 'w') as f:
            f.write(f'file://{report_path}')
        print(f'Report saved locally: {report_path}')
```

## When NOT to use

- Single contact onboarding (use `weave.upsert.person` directly)
- Research without graph updates (use Scout or Sift alone)
- Image-to-action processing (use Look)
