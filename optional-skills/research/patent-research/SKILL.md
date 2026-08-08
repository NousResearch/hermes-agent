---
name: patent-research
description: Search and analyze patents via free APIs. Google Patents XHR API (no key, covers US/EP/WO/JP/CN/KR), assignee/inventor tracking, CPC/IPC classification, patent landscape analysis, prior art search, citation chain analysis.
platforms: [linux, macos, windows]
---

# Patent Research

Search and analyze patents across global databases using free public APIs. **No API keys required.**

## Helper script

This skill includes `scripts/patent_search.py` — a complete CLI tool for all patent research operations.

```bash
# Basic keyword search
python3 SKILL_DIR/scripts/patent_search.py search "blockchain consensus"

# Search by assignee (company)
python3 SKILL_DIR/scripts/patent_search.py assignee "Apple Inc."

# Search by CPC classification
python3 SKILL_DIR/scripts/patent_search.py cpc G06N20/00

# Get patent details (abstract, assignee, inventors)
python3 SKILL_DIR/scripts/patent_search.py detail US11074495B2

# Citation chain analysis
python3 SKILL_DIR/scripts/patent_search.py citations US11074495B2

# Patent landscape analysis
python3 SKILL_DIR/scripts/patent_search.py landscape "quantum computing" --companies "IBM,Google,Microsoft"

# Advanced search with filters
python3 SKILL_DIR/scripts/patent_search.py search "neural network" --since 2023 --status GRANTED --num 20
```

`SKILL_DIR` is the directory containing this SKILL.md file. All output is structured JSON.

## Commands

| Command | What it does | Example |
|---------|-------------|---------|
| `search` | Keyword search with filters | `search "AI" --since 2023 --status GRANTED` |
| `assignee` | Search by company name | `assignee "Apple Inc."` |
| `inventor` | Filter search by inventor | `search "camera" --inventor "John Smith"` |
| `cpc` | Search by CPC classification | `cpc G06N20/00` |
| `detail` | Fetch full patent details | `detail US11074495B2` |
| `citations` | Extract forward/backward citations | `citations US11074495B2` |
| `landscape` | Competitive landscape analysis | `landscape "blockchain" --companies "IBM,Intel"` |

## Primary: Google Patents XHR API

Free, comprehensive, no auth. Covers US, EP, WO, JP, CN, KR, DE, FR, GB + 100 more offices.

### Query syntax

| Pattern | Finds |
|---------|-------|
| `blockchain AND consensus` | Both terms |
| `"machine learning"` | Exact phrase |
| `assignee:"Apple Inc."` | By company |
| `inventor:"John Doe"` | By inventor |
| `cpc:G06N20/00` | By CPC classification |
| `ipc:G06N` | By IPC classification |
| `status:GRANTED` | By legal status |
| `language:ENGLISH` | By language |
| `priority:2020` | By priority date |
| `office:US` | By patent office |

### Search parameters

| Param | Description |
|-------|-------------|
| `q` | Search query (Google Patents syntax) |
| `num` | Results per page (max 100) |
| `language` | `ENGLISH`, `JAPANESE`, `CHINESE` |
| `date` | Range: `YYYYMMDD000000/YYYYMMDD000000` |
| `status` | `GRANTED`, `PENDING`, `EXPIRED` |
| `office` | `US`, `EP`, `WO`, `JP`, `CN`, `KR` |
| `type` | `patent` or `application` |
| `page` | Page number (0-indexed) |

### Raw API call

```bash
# Search: recent blockchain patents
curl -sL "https://patents.google.com/xhr/query?url=q%3Dblockchain%2Bconsensus%26num%3D5" \
  -H "User-Agent: Mozilla/5.0"
```

### Get patent details (HTML)

```bash
curl -sL "https://patents.google.com/patent/US11074495B2/en" -H "User-Agent: Mozilla/5.0"
```

Extractables: `<meta name="DC.description">` (abstract), `div.claims` (claims text), `tr.assignee`, `tr.inventor`, classification codes (CPC/IPC).

### Response format

```json
{
  "results": {
    "total_num_results": 125048,
    "total_num_pages": 100,
    "cluster": [{
      "result": [{
        "id": "patent/US11074495B2/en",
        "rank": 0,
        "patent": {
          "title": "System and method for...",
          "snippet": "...artificial intelligence..."
        }
      }]
    }]
  }
}
```

## Patent landscape analysis

When asked for patent landscape / competitive analysis:

1. **Assignee analysis** — search each major company, compare counts
2. **CPC clustering** — group results by CPC class to identify technology clusters
3. **Citation analysis** — forward citations reveal influence, backward reveal prior art
4. **Trend analysis** — compare year-over-year filing counts
5. **Geographic analysis** — compare filings across US/EP/WO/JP/CN offices

## Citation chain analysis

```bash
# Extract patent IDs from the citation section
python3 SKILL_DIR/scripts/patent_search.py citations US11074495B2
```

## Pitfalls

- **User-Agent required** — Google Patents XHR API rejects requests without `Mozilla/5.0` header
- **Rate limits** — undocumented; generous but don't spam. Stay under 10 req/min
- **URL encoding** — the `url` query parameter must itself be URL-encoded (double encoding)
- **Patent number format** varies by office: `US12345678B2`, `EP1234567A1`, `WO2024123456A1`
- **Legal status** — Google's classification may differ from official registers; verify for critical searches
- **API changes** — undocumented internal API; may break without notice

## Platform compatibility

Pure Python stdlib (`urllib.request`, `json`, `html.parser`). Works identically on Linux, macOS, and Windows with no dependencies.

- Google Patents API uses HTTPS (port 443) — works behind most firewalls
- Patent detail pages are HTML — parsed with stdlib `HTMLParser`

## When to use this vs other tools

| Task | Better tool | Why |
|------|-------------|-----|
| "Prior art for my idea" | **This skill** | Patent databases > general web search |
| "What does company X work on?" | **This skill** | Patents reveal R&D focus before products |
| "Find all patents in this technology area" | **This skill** | CPC classification is precise |
| "Scientific papers about this topic" | `arxiv` skill | Academic papers, not patents |
| "General research about a company" | `web_search` | News, products, press releases |
| "Who invented this technology?" | **This skill** | Inventor tracking via patent assignments |

## Workflow: Prior Art Search

1. Identify key concepts and synonyms
2. Search by keyword with date range
3. Review results — note patent IDs and CPC classifications
4. Fetch details for top relevant patents
5. Use CPC codes to broaden/narrow the search
6. Check forward/backward citations for each key patent

## Workflow: Competitive Intelligence

1. Identify target companies in the technology space
2. Search each as `assignee:"Company Name"` sorted by date
3. Classify results by CPC to identify technology focus shifts
4. Note prolific inventors — track their movement between companies
5. Compare filing rates year-over-year