---
name: graph-expansion-pipeline-execution
description: >
  Graph Expansion Pipeline execution guide — practical lessons from running
  the multi-phase enrichment workflow (Scout → Weave → Sift → Synthesis).
  Covers fallback strategies when web_search fails, bot detection workarounds,
  and output delivery options.
metadata:
  author: Indigo Karasu
  version: "1.0.0"
---

# Graph Expansion Pipeline Execution

This skill documents practical lessons from executing the Graph Expansion Pipeline for batch enrichment of contact targets.

## Pipeline Phases

1. **Scout OSINT** — Person-focused OSINT research with provenance-backed briefs
2. **Weave Upsert** — Stage findings for social graph insertion
3. **Sift Deep Dive** — Topic research for publications, talks, technical contributions
4. **Final Synthesis** — Consolidate findings with confidence scores
5. **Report Generation** — Compile markdown report and optionally upload to Google Docs

## Key Findings from Production Runs

### 1. Search Tool Priority

**Priority order for research queries:**

1. **Direct API calls via `execute_code`/`terminal`** — Most reliable for academic/technical targets. Semantic Scholar API, DBLP API, GitHub API, and arXiv API all work without CAPTCHA or rate limits from cloud environments. Use `curl` or Python `requests` for:
   - Semantic Scholar: `api.semanticscholar.org/graph/v1/author/search?query=NAME&fields=...` — returns author profiles, papers, h-index, citations, affiliations
   - DBLP: `dblp.org/search/publ/api?q=NAME&format=json` — publication records; also `dblp.org/search/author/api?q=NAME&format=json` for author disambiguation
   - GitHub: `api.github.com/search/users?q=NAME`, `api.github.com/search/commits?q=author-email:EMAIL`
   - arXiv: `export.arxiv.org/api/query?search_query=...`
2. **`web_search` (native SearXNG-based)** — Works reliably from the parent agent via the `searchx` skill. Returns structured web results without CAPTCHA or rate limits. **Note:** The built-in `web_search` tool (non-SearXNG) is credit-blocked and should not be used; the `searchx` skill routes through local SearXNG which works. Use for general web context, person/company lookups.
3. **Jina Reader (`r.jina.ai/{url}`)** — Free URL content extraction fallback when `web_extract` credits are exhausted (Firecrawl "Payment Required" errors). Free tier: 200 requests/day. Use via `curl -s -H 'Accept: text/plain' 'https://r.jina.ai/{url}'`. Returns clean Markdown. Works for Substack, Medium, most blog posts. Fails on YouTube (429 rate limit) and heavily JS-gated pages.
4. **`mcp_tavily_tavily_search`** (search_depth: "advanced") — Fourth choice. Returns structured results with full text. **Caveat: Has monthly rate limits (432 error).** When rate-limited, fall back to direct API calls or SearXNG web_search.
5. **`browser_navigate`** — Fallback for specific platform pages only. Triggers CAPTCHA on Google/Bing/DuckDuckGo within 3-5 queries. Use ONLY for direct URL navigation to known pages, never for general web search.
6. **`web_extract` (Firecrawl)** — BROKEN/credit-blocked. Returns "Payment Required: Failed to scrape. Insufficient credits." Do not use for batch pipeline work.

**Key lessons:**
- Subagents using browser_navigate for search consistently fail (CAPTCHA on every major engine).
- When both Tavily AND browser_navigate fail (rate limits + CAPTCHA), direct API calls via `execute_code` are the reliable fallback. This is the CAPTCHA cascade fallback documented in Scout skill.
- Semantic Scholar is exceptionally good for disambiguating common names because it returns paper titles, venues, and co-authors that let you confirm identity matches.

**Direct API endpoints (use via `execute_code` with `curl` or Python `requests`):**

| API | Endpoint | Best For | Rate Limit |
|-----|----------|----------|------------|
| Semantic Scholar | `api.semanticscholar.org/graph/v1/author/search?query=NAME&limit=5&fields=name,paperCount,citationCount,hIndex,homepage,affiliations,papers.title,papers.year,papers.citationCount` | Academic/research targets, h-index, paper lists | 1 req/sec unauthenticated |
| Semantic Scholar (paper) | `api.semanticscholar.org/graph/v1/paper/search?query=TITLE&limit=3&fields=title,year,citationCount,authors.name,authors.authorId` | Finding a known paper's author list | 1 req/sec |
| Semantic Scholar (author ID) | `api.semanticscholar.org/graph/v1/author/{ID}?fields=name,paperCount,citationCount,hIndex,affiliations,papers.title,papers.year` | Full profile lookup once you have an ID | 1 req/sec |
| DBLP (publications) | `dblp.org/search/publ/api?q=NAME&format=json&h=10` | Publication records by name | None (fair use) |
| DBLP (authors) | `dblp.org/search/author/api?q=NAME&format=json&h=5` | Author disambiguation (matches "B. Behzadi" etc.) | None (fair use) |
| GitHub Users | `api.github.com/search/users?q=NAME+in:name&per_page=5` then `api.github.com/users/{login}` for full profile | Developer identity, repos, company | 10 req/min unauthenticated |
| GitHub Commits | `api.github.com/search/commits?q=author-email:EMAIL&per_page=5` | Email-to-GitHub identity resolution | 10 req/min |
| arXiv | `export.arxiv.org/api/query?search_query=au:NAME_AND_ti:KEYWORD&max_results=3` | Academic preprints | None (fair use) |

**Profile URL probing pitfalls:**
- **Google Developer Profiles** (`developers.google.com/profile/u/{handle}`) — Returns HTTP 200 for ALL handles regardless of whether the profile exists. The page is a React SPA that shows a generic "My Profile" template. Only useful if the target's name appears in the HTML content (rare). Do NOT assume a 200 means the profile exists.
- **Google Research** (`research.google/people/{slug}/`) — Returns 200 with a "not found" message for most slugs. Not reliable for confirming employment.
- **LinkedIn** — Requires authentication. Do not attempt automated scraping.

**GitHub Search Expansion:** When `type=users` returns 0 results, expand to:
- `type=commits` — Most reliable for developer identity
- `type=issues` — Finds mentions in discussions
- `type=pullrequests` — Shows code review activity

### 1b. Confidence Scoring Heuristics

| Signal | Confidence Level |
|--------|------------------|
| 50+ GitHub commits with verified email | High (≥0.8) |
| 10-50 commits or PRs | Medium (0.5-0.7) |
| 1-10 commits, unverified | Low (0.3-0.4) |
| 0 public activity, corporate email only | Low (0.3) |
| Search too noisy (common name) | Unknown |

**Email domain reliability:**
- Corporate emails (@google.com, @meta.com) confirm affiliation but NOT role
- Personal emails (gmail.com, googlemail.com) with activity = stronger identity signal
- ~80% of targets with corporate emails had zero public GitHub activity in production run

**Email handle heuristics (Google):**
- Short handles (3-4 chars like `rux@`, `poh@`) → High seniority/early employee
- Standard handles → Normal employee
- `@googlemail.com` → Consumer Gmail, not corporate

### 2. Bot Detection Patterns

**Most Aggressive:**
- Google Search — Triggers "unusual traffic" within 3-5 queries
- Bing — CAPTCHA after ~5 queries
- LinkedIn — Login wall + bot detection
- DuckDuckGo — Rate limiting

**Most Reliable:**
- arXiv — No bot detection, clean HTML
- Google Patents — No bot detection, structured data
- GitHub — Minimal bot detection for search pages

**Strategy:** When general search engines block you, pivot to specialized repositories:
- Academic targets → arXiv, ACL Anthology, PubMed, DBLP
- Technical targets → GitHub, GitLab, StackOverflow
- Corporate targets → Google Patents, company blogs

### 3. Parallel Delegation Strategy

**Subagent OSINT delegation works well when using the native `web_search` tool (SearXNG-based) via the `searchx` skill.** Unlike browser-based search or Tavily, SearXNG does not trigger CAPTCHA and produces reliable results. Use `delegate_task` with `toolsets=["web"]` for parallel person research:

```python
delegate_task(tasks=[
    {"goal": "OSINT research on Target A...", "toolsets": ["web"]},
    {"goal": "OSINT research on Target B...", "toolsets": ["web"]},
    {"goal": "OSINT research on Target C...", "toolsets": ["web"]},
])
```

**When subagent search fails (CAPTCHA or rate limits):** Fall back to direct API calls via `execute_code` with Semantic Scholar, DBLP, GitHub, and arXiv APIs. See Pitfall 6 and Search Tool Priority above.

**For Tavily-dependent searches:** Prefer `mcp_tavily_tavily_search` (search_depth: "advanced") from the parent agent, not subagents, because Tavily rate limits are shared across all calls.

For non-search batch work (file processing, data synthesis), use `delegate_task` with `tasks` array:

```python
delegate_task(tasks=[
    {"goal": "Process Target A data into format X...", "context": "..."},
    {"goal": "Process Target B data into format X...", "context": "..."},
    {"goal": "Process Target C data into format X...", "context": "..."},
])
```

**Benefits of delegation:**
- 3x parallelism reduces total runtime
- Each subagent has isolated context and terminal
- Failures don't block other targets

**Limits:** Max 3 tasks per batch. For 10 targets, run 4 batches (3+3+3+1).

**Low-visibility target expectation:** ~30-40% of corporate email targets (especially @google.com with short handles) will have minimal or zero public footprint. Mark these as "identity unresolved" with confidence 0.3-0.4 rather than spending excess iterations. Key examples: Gustavo Moura, Peter Oh, Marc Paulina produced almost no public results despite valid corporate emails.

### 4. Confidence Scoring

Assign confidence based on evidence quality:

| Confidence | Criteria |
|------------|----------|
| **HIGH** | Multiple corroborating sources (arXiv + patents + talks) |
| **MEDIUM** | Single strong source (confirmed publication or role) |
| **LOW** | Email domain only, no public presence |

**Email Handle Heuristics (Google):**
- Short handles (3-4 chars like `rux@`, `poh@`) → High seniority/early employee
- Standard handles → Normal employee
- `@googlemail.com` → Consumer Gmail, not corporate

### Pitfall 5: Google Docs API — OAuth Profile Selection

**Prerequisite:** The default Hermes profile (`/root/.hermes/google_token.json`) typically has Gmail-only scopes (`gmail.readonly`, `gmail.modify`). The Docs and Drive scopes are on the **Indigo profile** (`/root/.hermes-indigo/google_token.json`).

**Check token scopes before creating docs:**
```python
import json
with open('/root/.hermes/google_token.json') as f:
    token_data = json.load(f)
scopes = token_data.get('scopes', [])
has_docs = 'documents' in str(scopes).lower()
has_drive = 'drive' in str(scopes).lower()
```

**If default profile lacks Docs scope:**
1. Try the Indigo profile (`/root/.hermes-indigo/google_token.json`)
2. Use `HERMES_HOME=/root/.hermes-indigo` environment variable
3. Build the docs service with the Indigo credentials

**Pattern for creating Google Docs with the Indigo profile:**
```python
import os
os.environ['HERMES_HOME'] = '/root/.hermes-indigo'

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

with open('/root/.hermes-indigo/google_token.json') as f:
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

service = build('docs', 'v1', credentials=creds)
doc = service.documents().create(body={"title": "Report Title"}).execute()
doc_id = doc['documentId']
```

**Common Error — Insufficient Scopes (403):**
```
HttpError 403: "Request had insufficient authentication scopes."
```
This means the token used doesn't have the `documents` scope. Switch to the Indigo profile token.

**Fallback:** If no profile has Docs scope, save the report as a local file:
```
/root/.hermes/commons/data/ocas-expansion/last_run_report.txt
```

**Prerequisite:** Google Workspace OAuth must be configured OR service account must have Drive quota.

**Check:**
```bash
ls -la ~/.hermes/google_oauth.json
ls -la ~/.hermes/credentials/*.json  # Service account
```

**Common Error — Service Account Quota:**
```
HttpError 403: "Service Accounts do not have storage quota.
Leverage shared drives, or use OAuth delegation instead."
```

**If service account has no quota:**
1. Service accounts have isolated Drive with zero default quota
2. Options: (a) Use user OAuth, (b) Configure Shared Drive, (c) Manual upload
3. Docs API may not be enabled for service account — fallback to .txt file

**If OAuth not configured:**
1. Run `hermes setup`
2. Configure Google Workspace OAuth with drive scope
3. Re-run report upload

**Fallback:** Save markdown report locally:
```\n/root/.hermes/commons/data/ocas-expansion/last_run_report.txt
```

**Manual Google Doc creation:**
1. Copy report contents
2. Create new Google Doc at https://docs.google.com
3. Paste and share
4. Save link to `google_doc_link.txt`

## Common Pitfalls

### Pitfall 1: Repeated web_search After Failure

**Wrong:** Keep calling `web_search` after "Payment Required" errors.

**Right:** Immediately pivot to `browser_navigate` with direct URLs.

### Pitfall 2: LinkedIn Expectations

**Wrong:** Expect to extract LinkedIn profile data without authentication.

**Right:** LinkedIn requires login. Use alternative signals (arXiv, patents, GitHub) for identity resolution.

### Pitfall 3: Name Ambiguity

**Wrong:** Assume all mentions of "Peter Oh" refer to the same person.

**Right:** Require corroborating evidence (email, company, field) before merging identities. Document uncertainty.

### Pitfall 4: Weave Upsert Without Staging

**Wrong:** Direct Weave upserts during pipeline execution.

**Right:** Stage upserts to JSON first, then process via Weave skill commands.

### Pitfall 5: Name Disambiguation Failure

**Wrong:** Assume the first search result for a name matches the target person.

**Right:** Require corroborating evidence (email domain, company, field, specific role) before merging identities. Common-name targets are especially dangerous:
- "Russell Matsuo" → Staff UX Designer at Google (The Org), NOT the RT-X robotics author (that's "Russell Mendonca" in the author list). DBLP returns 4 results including an "Akitaka Matsuo" (legal dialogue systems) and a "Russ Tedrake + Yoky Matsuoka" co-edit that is NOT the same person.
- "Gustavo Moura" → Semantic Scholar returns 10+ different people: "Gustavo L. C. Moura" (chemistry, 19 papers, 203 cites), "G. C. Moura" (different field), etc. The DBLP stream learning results point to "Gustavo E. A. P. A. Batista" — the surname is different! The `moura@google.com` person CANNOT be confirmed as any of these without additional evidence. Mark as "identity unresolved."
- "Peter Oh" → At least 4 different people across dConstruct Robotics, Iterable, Charlie Health, and Google. Semantic Scholar returns "Peter Oh" papers about corporate finance (CEO Networks) and holographic imaging — completely different fields. The `poh@google.com` person is likely a different Peter Oh than any found in academic databases.
- "Marc Paulina" → DBLP returns 78 results, but almost all are "Beatriz Paulina Garcia Salgado" (hyperspectral image processing), not "Marc Paulina" from Google. The PepCompass paper is by different authors entirely.
- "Laith Ulaby" → Semantic Scholar finds a "Laith Ulaby" with 3 papers about sea music in the Arab Gulf (ethnomusicology), not UX research. This is likely a different person than the UX Director.

**Semantic Scholar disambiguation strategy:**
1. Search by exact name first: `author/search?query=Full+Name`
2. SS often returns multiple author entries for the same name (e.g., "Behshad Behzadi" has 3 entries: computational biology h-index 8, device orientation h-index 0, and German localization h-index 0). Each entry is a different person — check paper topics against known context (company, domain) before merging.
3. If results are ambiguous, search by name + paper title keyword: `author/search?query=Name+Keyword`
4. Check paper titles and co-authors against known context (company, domain)
5. Use author ID lookups once you've confirmed the right person
6. If no confident match exists, mark as "identity unresolved" and document all candidate matches

**Rule:** If a search returns results for a person with the same name but different company/field, do NOT merge. Mark as "identity unresolved" and note the ambiguity.
**Rule:** If a search returns results for a person with the same name but different company/field, do NOT merge. Mark as "identity unresolved" and note the ambiguity.

### Pitfall 6: Subagent OSINT Delegation

**Wrong:** Delegating OSINT research to subagents expecting them to navigate search engines.

**Right:** Run `mcp_tavily_tavily_search` queries from the parent agent directly when Tavily is available. Subagents hit CAPTCHA walls on Google/Bing/DuckDuckGo and waste their iteration budget. When Tavily is also rate-limited (432 error), fall back to direct API calls via `execute_code` with `curl`: Semantic Scholar, DBLP, GitHub, and arXiv APIs all work reliably from cloud environments. Reserve subagent delegation for truly independent work (e.g., processing different output formats, parallel file writes), not for search queries.

### Pitfall 7: Tavily Rate Limiting

**Wrong:** Assuming Tavily always works and can handle an entire batch of 10+ research targets.

**Right:** Tavily has monthly rate limits. When you hit a 432 error ("exceeds your plan's set usage limit"), immediately switch to direct API calls. The CAPTCHA cascade fallback from the Scout skill is: (1) GitHub API, (2) direct profile probing, (3) platform-specific APIs (Semantic Scholar, DBLP, arXiv), (4) report findings with confidence levels.

### Pitfall 8: Semantic Scholar Name Disambiguation

**Wrong:** Trusting the first Semantic Scholar author result for a common name like "Peter Oh" or "Gustavo Moura".

**Right:** Semantic Scholar returns multiple authors for common names. Always:
1. Check paper titles and venues against known context (is this really about robotics? stream learning? etc.)
2. Look at h-index and citation counts as disambiguation signals
3. Use the `fields=papers.title,papers.year,papers.citationCount` parameter to see actual papers
4. If paper topics don't match the target's known domain, that's a DIFFERENT person
5. For extremely common names, search by paper title + name to find the specific author ID

### Pitfall 8: Batch API Scripting vs. Individual Tool Calls

**Wrong:** Making 30+ individual `mcp_tavily_tavily_search` or `terminal` calls for 10 targets one at a time.

**Right:** Write a single Python script with all API calls (Semantic Scholar, DBLP, GitHub, arXiv) and run it via `execute_code`. This consolidates 40+ HTTP requests into 2-3 script executions with built-in rate-limit delays. Reserve individual tool calls for follow-up probes on specific targets. The script pattern:
1. Load targets from JSONL
2. Loop through targets calling each API with `requests.get()` and `time.sleep()` for rate limits
3. Save consolidated results to a JSON file
4. Do a second pass for deep dives on targets with initial hits
5. Merge all findings into a single consolidated JSON for downstream pipeline phases

### Pitfall 9: Tavily Rate Limit Alternative Research

**Wrong:** Stopping research entirely when Tavily returns 432 (plan limit exceeded).

**Right:** Follow the CAPTCHA cascade from Scout skill, but with a specific batch pattern:
- First, run a single `execute_code` script that calls Semantic Scholar API, DBLP API, GitHub API, and arXiv API for ALL targets in sequence
- Use `time.sleep(1.0-1.2)` between Semantic Scholar calls (1 req/sec rate limit)
- Use `time.sleep(0.3-0.5)` between GitHub/DBLP/arXiv calls (more lenient)
- For targets with hits, run a second deep-dive script with specific author ID lookups and paper searches
- Only use `mcp_tavily_tavily_search` for final verification or when APIs produce no results

### Pitfall 9: LadybugDB Query Result Format

**Wrong:** Assuming LadybugDB (real_ladybug) returns dicts for all query types.

**Right:** The return format depends on your Cypher RETURN clause AND the real_ladybug version:
- **`RETURN p`** (whole node): Format varies by version. In **real_ladybug 0.15.3**, `RETURN p` returns **dicts** but with keys mapped to `None` values (the node properties are stored differently internally). **Use column selectors instead** — they always work. In older versions, `RETURN p` may return dicts with populated values.
- **`RETURN p.id, p.name, ...`** (column selectors): rows are **lists**, NOT dicts. This is the **reliable format**. Use `r.get_column_names()` to map column names to indices. Example: `cols = r.get_column_names(); name_idx = cols.index('p.name'); rows[0][name_idx]`.
- **`r.get_all()`** returns a Python list of rows.
- **`r.rows_as_dict()`** returns a QueryResult object, NOT a Python dict — do not iterate or access it like a dict.
- **Key error pattern**: Using `row.get('name')` on a list row from column selectors raises `TypeError: list indices must be integers or slices, not str`. Always match your access pattern to the return format.
- **Best practice**: Always use `RETURN p.field1, p.field2, ...` column selectors for read-back verification. `RETURN p` is unreliable for property access.
- The `RETURN p` format is easiest for property access but includes internal fields (`_ID`, `_LABEL`) that you need to filter out.

### Pitfall 10: LadybugDB Cypher MATCH for Relationships

**Wrong:** Using a comma between two node patterns in a single MATCH clause for relationship creation:
```cypher
MATCH (p:Person {id: 'X'}, (f:Fact {id: 'Y'}) MERGE (p)-[:HasFact]->(f)
-- Parser error: Invalid input at comma
```

**Right:** Use separate MATCH clauses for each node, then CREATE the relationship:
```cypher
MATCH (p:Person {id: 'X'}) MATCH (f:Fact {id: 'Y'}) CREATE (p)-[:HasFact]->(f)
```

**Also critical:** When writing batch upsert scripts in Python, double-check f-string interpolation in Cypher queries. A typo like `{person_id]'` instead of `{person_id}'` inside an f-string will cause a `SyntaxError` that can be hard to spot in a large script. Always test scripts with a single target first before batch execution.

The LadybugDB parser does not support comma-separated node patterns in a single MATCH — each `MATCH` introduces one pattern.

### Pitfall 11: Weave Batch Upsert Script Pattern

**Wrong:** Using `python3 -c "..."` with complex inline Python for LadybugDB batch upserts. Multi-line f-strings with Cypher queries break due to quote escaping issues.

**Right:** Write the upsert script to a file first with `write_file`, then execute via `terminal`:
```python
write_file(path="/tmp/weave_upsert.py", content=upsert_script)
terminal(command="python3 /tmp/weave_upsert.py")
```
The script should use `real_ladybug` directly, MERGE on Person `id` for upserts, and always read back after write to confirm success. Batch all 10 targets in a single script execution to avoid DB lock contention.

### Pitfall 12: LadybugDB Schema Property Mismatch

**Wrong:** Setting properties like `linkedin`, `dev_profile_url`, `education`, or `dblp_total_publications` on Person nodes in MERGE/SET statements. LadybugDB throws `Binder exception: Cannot find property X for p` for any property not defined in the DDL.

**Right:** The Person table schema has EXACTLY these columns: `id, name, name_given, name_family, email, phone, location_city, location_country, occupation, org, notes, google_resource_name, clay_id, source_type, source_ref, confidence, event_time, record_time, valid_from, valid_until`. Nothing else. Data that doesn't fit these columns must go in:
- **`notes`** field — For semi-structured text like LinkedIn URLs, education, dev profile links, key facts. Keep under 500 chars. **Do NOT use `''` escaping** — strip single quotes entirely (see Pitfall 15).
- **Fact nodes** — For structured claims with provenance (`predicate`, `value`, `confidence`, `source_type`, `source_ref`). Use `CREATE (f:Fact {...})` then `MATCH (p:Person {id: X}) MATCH (f:Fact {id: Y}) CREATE (p)-[:HasFact]->(f)` as two separate statements (see Pitfall 10).
- **Preference nodes** — For likes/dislikes with `category`, `value`, `valence`.

**Before writing any MERGE/SET on Person**, check the schema in `references/schemas.md` or run `CALL show_tables() RETURN *` followed by property inspection. Always MERGE only on known columns.

**Pattern for batch upsert scripts:** Write enrichment data fields that don't have Person columns into a `notes` string (stripped of single quotes, truncated to 500 chars), and add structured claims as Fact nodes linked via HasFact edges. Example:
```python
def safe_str(s, max_len=500):
    """Strip single quotes aggressively for LadybugDB Cypher compatibility."""
    s = str(s).replace("'", "").replace('"', '').replace('`', '')[:max_len]
    return s

extra_parts = []
if linkedin:
    extra_parts.append("LinkedIn: " + linkedin_url)
if dev_profile_url:
    extra_parts.append("Dev Profile: " + dev_profile_url)
if education:
    extra_parts.append("Education: " + education)
notes = safe_str('; '.join(extra_parts)[:500])
```

### Pitfall 13: read_file JSON Parsing

**Wrong:** Using `read_file` for JSON files and trying to parse the line-numbered output directly.

**Right:** `read_file` returns content with line-number prefixes (e.g., `     1|{"key": ...}`) that breaks `json.loads()`. Instead, write a small Python script to a file, then execute it with `terminal`:
```python
# Write a script that reads the file directly
write_file(path="/tmp/inspect.py", content="""
import json
with open("/path/to/data.json") as f:
    d = json.load(f)
print(d["key"])
""")
terminal(command="python3 /tmp/inspect.py")
```

## Output Structure

```
/root/.hermes/commons/data/ocas-expansion/
├── {run_id}.json              # Tracking file
├── scout_findings.json        # Phase 1 results
├── weave_upserts.json         # Phase 2 staged data
├── final_findings.json        # Consolidated results
└── last_run_report.txt        # Human-readable report
```

### Pitfall 14: Correcting Prior Assumptions During Deep Dive

**Wrong:** Treating Phase 1 (Scout) findings as final and not updating Weave when Phase 3 (Sift) discovers corrections.

**Right:** The Sift deep dive often finds corrections to Phase 1 findings (e.g., a person moved companies, a role was wrong, or a name was misspelled like ` laith Ulaby` with a leading space). When this happens:
1. Immediately update the Person node in Weave with corrected data (`org`, `occupation`, `notes`)
2. Add a `career_correction` or `identity_correction` Fact node with provenance pointing to the Sift source
3. Log the correction in the Phase 3 output file for the final report

**Real example from production:** Phase 1 listed "Laith Ulaby" at "Toyota Research Institute" but Sift deep dive found he's actually "Director of Insights at Webflow since Feb 2022" — the TRI data was stale. The pipeline corrected the Weave node and added a `career_correction` Fact.

### Pitfall 15: Fact Node Value Escaping for Apostrophes

**Wrong:** Using Python f-strings with values containing apostrophes (like `AI's`) in Cypher CREATE statements. The `''` escaping inside f-strings causes `SyntaxError: f-string expression part cannot include a backslash`.

**Partially wrong:** Even with string concatenation, the SQL-style `''` double-single-quote escaping **does NOT work reliably in LadybugDB Cypher**. The parser treats `''text''` as a string termination followed by unparseable tokens, causing errors like `Parser exception: mismatched input ''Meta Pathfinding UX'' expecting {<EOF>, ';', SP}`.

**Right:** Strip all single quotes from values entirely before inserting into Cypher statements:
```python
def safe_str(s, max_len=500):
    """Strip single quotes aggressively for LadybugDB Cypher compatibility."""
    s = str(s).replace("'", "").replace('"', '').replace('`', '')[:max_len]
    return s

# Then use string concatenation:
value_safe = safe_str(value)[:300]
q = "CREATE (f:Fact {id: '" + fact_id + "', predicate: '" + predicate + "', value: '" + value_safe + "', ...})"
```

This is the **only** reliably working approach in production. The `''` escaping method (from SQL convention) fails in LadybugDB's Cypher parser. Stripping quotes loses some fidelity (e.g., "Meta Pathfinding UX" becomes "Meta Pathfinding UX" — usually acceptable) but guarantees parse-free execution.

### Pitfall 16: web_extract Credit Exhaustion

**Wrong:** Relying on `web_extract` (Firecrawl) for batch URL content extraction during pipeline runs. It returns "Payment Required: Failed to scrape. Insufficient credits" after the monthly quota is exhausted.

**Right:** When `web_extract` fails, use **Jina Reader** as a free fallback. Jina Reader fetches any URL and returns clean Markdown via a simple HTTP request:
```bash
curl -s -H 'Accept: text/plain' 'https://r.jina.ai/{url}' | head -300
```
Or in Python:
```python
import requests
r = requests.get(f"https://r.jina.ai/{url}", headers={"Accept": "text/plain"}, timeout=30)
content = r.text  # Returns Markdown with Title, URL Source, and content
```

**Jina Reader specs:** Free tier allows 200 requests/day. Works for: Substack, Medium, most blogs, news sites, documentation. Fails for: YouTube (429 rate limit), heavily JS-gated SPAs, LinkedIn (auth wall).

**Pipeline pattern:** When `web_extract` returns credit errors, immediately switch all URL extraction to Jina Reader for the remainder of the run. Do not retry `web_extract` — the credit limit doesn't reset during a session.

### Pitfall 17: web_search SearXNG Works from Parent Agent

**Wrong:** Assuming `web_search` is always credit-blocked and unusable (old finding from built-in web_search tool).

**Right:** The SearXNG-based search (via the `searchx` skill) works reliably from the parent agent. It returns structured web results with titles, URLs, and descriptions without CAPTCHA issues. **Key distinction:** The **built-in** `web_search` tool (Firecrawl-backed) IS credit-blocked. The **searchx skill** (local SearXNG) is NOT. When the skill instructs "use web_search", interpret it as using the searchx skill, not the built-in tool.

**Best use:** Person/company/product lookups where you need titles and URLs but not full page content. Pair with Jina Reader for extracting content from the discovered URLs.

## When to Use

- Batch enrichment of contact lists
- Due diligence on potential hires/partners
- Social graph expansion with provenance tracking
- Research on low-visibility technical leaders

## When Not to Use

- Single-target research (use Scout directly)
- Illegal data collection
- Targets with known private/security concerns

## Related Skills

- `ocas-scout` — OSINT research methodology
- `ocas-sift` — Web research and synthesis
- `ocas-weave` — Social graph management
- `graph-expansion-pipeline` — Pipeline orchestration
