---
name: research-fallback-pivot
description: Strategy for performing deep research when standard web_search is credit-limited and major search engines (Google/Scholar) trigger bot detection.
---

# Research Fallback Pivot

When performing deep dives on individuals (researchers, executives, engineers) and encountering `web_search` credit limits or Google CAPTCHAs, pivot from "General Search" to "Direct Registry Navigation."

## Trigger Conditions
- `web_search` returns "Insufficient credits".
- `browser_navigate` to Google/Scholar returns a "detected unusual traffic" or CAPTCHA page.
- High-value targets who likely have profiles in structured academic or legal databases.

## Step-by-Step Workflow

### Tier 0: Check credit availability first
Before anything else, try one `mcp_tavily_tavily_search` call. If it returns a 432 "usage limit" error, immediately skip to Tier 2 (API-only). Do not waste additional calls on credit-limited services.

### Tier 1: Web search (if credits available)
- Use `mcp_tavily_tavily_search` or `web_search` for broad queries.
- If credits run out mid-research, pivot immediately to Tier 2.

### Tier 2: API-only collection (proven fallback from cloud environments)

When all web search services are credit-limited AND major search engines (Google, Bing) return CAPTCHA pages from cloud IPs, use this specific API stack in order:

1. **GitHub API** (no auth required, curl-based):
   - User search: `curl -s "https://api.github.com/search/users?q=FULLNAME"`
   - Commit search by email: `curl -s "https://api.github.com/search/commits?q=author-email:EMAIL"`
   - User profile: `curl -s "https://api.github.com/users/LOGIN"` (returns name, company, bio, location, blog)
   - Search Users, Issues, Commits, and Organizations. Email-based commit search is especially precise for corporate addresses.

2. **Semantic Scholar API** (free, 100 requests/5min):
   - Author search: `curl -s "https://api.semanticscholar.org/graph/v1/author/search?query=FULLNAME&limit=5"`
   - Author details: `curl -s "https://api.semanticscholar.org/graph/v1/author/AUTHOR_ID?fields=name,affiliations,paperCount,citationCount,hIndex,papers.title,papers.year,papers.venue,papers.citationCount,papers.authors,papers.externalIds"`
   - Paper search: `curl -s "https://api.semanticscholar.org/graph/v1/paper/search?query=TOPIC&limit=3&fields=title,authors,year"`
   - **Rate limit: 429 errors are common.** Space requests 3-5 seconds apart. Batch by getting all needed data per author in one call.

3. **ORCID Public API** (free, JSON):
   - Search: `curl -s -H "Accept: application/json" "https://pub.orcid.org/v3.0/search/?q=FIRSTNAME+LASTNAME&rows=3"`
   - Profile: `curl -s -H "Accept: application/json" "https://pub.orcid.org/v3.0/{ORCID_ID}/person"`
   - Note: Name matching is loose — results often include non-target people with similar names. Verify by checking given-name/family-name fields.

4. **arXiv API** (free, XML):
   - Search: `curl -s "http://export.arxiv.org/api/query?search_query=au:LASTNAME+FIRSTNAME&max_results=5"`
   - Returns author names in `<name>` tags and paper titles. Limited to arXiv-indexed papers only.

5. **Direct profile URL probing** (curl with status code check):
   - `curl -s -o /dev/null -w "%{http_code}" -L --max-time 10 URL` — 200 means profile exists, 404 means not found
   - **LinkedIn:** `https://www.linkedin.com/in/slug/` — returns 200 for valid slugs but content is auth-walled. 999 status = exists but bot-blocked.
   - **Google Developer profiles:** `https://developers.google.com/profile/u/USERNAME` — **useless**, returns 200 for ALL usernames with identical generic content. Do not rely on these.
   - **Google Research:** `https://research.google/people/SLUG/` — returns 404 for most researchers. URL format is not predictable. Not reliable for probing.
   - **DuckDuckGo HTML:** Also returns CAPTCHA/empty results from cloud environments. Do not waste time on `html.duckduckgo.com/html/`.

6. **Curl-based page extraction** for known URLs:
   - When DDG HTML search (initial step) returns results with specific URLs, fetch them directly with `curl -s -L -H "User-Agent: Mozilla/5.0"` and parse with regex.
   - Niche profile sites (conference bios, company leadership pages, industry press) often work well — Women in Tech Summit, Gambling Insider, SwissCognitive, etc.

7. **DBLP** (Computer Science bibliography):
   - Search: `curl -s "https://dblp.org/search/author/api?q=NAME&format=json&h=3"`
   - Publication search: `curl -s "https://dblp.org/search/publ/api?q=KEYWORD&format=json&h=5"`
   - Note: Name matching is loose; returns multiple candidates. Verify each manually.

### Tier 3: Domain-specific pivots

- **Google employees:** Search for their name associated with known Google projects (Gemini, Imagen, Assistant, Lens) on Semantic Scholar and arXiv. Get paper author lists to confirm association.
- **Meta employees:** Search GitHub for `@meta.com` email commits.
- **Patents:** Use `https://patents.google.com/` directly for inventor search.

### Synthesis
Aggregate findings from structured sources to build the profile. Mark confidence levels: "high" for confirmed (direct source match), "med" for inferred (email domain + project association), "low" for unconfirmed (name-only match without verification).

### Tier 2.5: SearXNG local instance (if available)

If a SearXNG instance is running locally (e.g., on localhost:8889), it can serve as an unlimited, credit-free search engine with aggregated results from multiple backends (Google, Brave, DuckDuckGo, Startpage, etc.).

**How to query SearXNG via browser:**
1. Navigate to `http://localhost:8889/search?q=YOUR+QUERY&format=json`
2. Parse the JSON response via `browser_console` (`document.body.innerText`)
3. Results are in the `results` array with `url`, `title`, `content`, `engine`, and `score` fields
4. Fetch full pages by navigating to result URLs with `browser_navigate`, then extract text with `browser_console` or `browser_snapshot`

**Advantages over API-only collection:**
- Returns aggregated results from multiple search engines simultaneously
- No API keys or credits needed
- Supports all query types (not just person/entity lookup)
- Direct URL extraction makes page fetching easy

**Limitations:**
- Requires a running SearXNG instance
- Some engines may be blocked or return access denied from the server's IP
- Result quality depends on which engines SearXNG can reach
- No specialized academic/database APIs, just general web search

**Example workflow:**
```
1. browser_navigate → http://localhost:8889/search?q=QUERY&format=json
2. browser_console → document.body.innerText (parse JSON for top results)
3. browser_navigate → https://result-url.example.com/page
4. browser_console → document.querySelector('article,main').innerText
5. Repeat for additional results or secondary searches
```

## Pitfalls & Tips
- **CAPTCHA cascade:** From cloud environments, ALL major search engines (Google, Bing, DDG HTML) will block headless browsers simultaneously. If two engines block, assume all will — stop browser-based search immediately and switch to Tier 2 API-only collection.
- **Semantic Scholar rate limits:** 429 errors are common when making rapid sequential requests. Batch author data into single calls (include all fields in one request). Wait 3-5 seconds between requests if hitting limits.
- **LinkedIn auth walls:** LinkedIn returns HTTP 200 for valid profile slugs but shows a login wall instead of content. Status 999 means the profile exists but is bot-blocked. LinkedIn profile titles visible in the `<title>` tag can sometimes confirm name and current company.
- **Google Developer profiles are generic:** `developers.google.com/profile/u/{username}` returns 200 for every username with identical generic content. This is NOT a valid way to confirm a person's Google employment.
- **GitHub user search is ambiguous:** Common names (Peter Oh, Gustavo Moura) return many results. Always cross-reference with commit email search (`author-email:EMAIL`) for disambiguation.
- **ORCID name matching is loose:** Search results frequently include non-target people with similar names. Always verify by checking the `given-names` and `family-names` fields in the profile response.
- **Verify Identity:** In large collaborations (e.g., Imagen, Gemini author lists with 50+ people), verify the specific role or authorship order to ensure the target isn't just a peripheral contributor. Use Semantic Scholar's `papers.authors` field to confirm exact position.
- **DBLP JSON API format:** The `.json` suffix on DBLP author pages doesn't work. Use the search API (`dblp.org/search/author/api`) instead.
- **Browser tool CAPTCHA:** Even `browser_navigate` to Google Search triggers CAPTCHA from cloud. Do not attempt browser-based general web search from cloud environments.
