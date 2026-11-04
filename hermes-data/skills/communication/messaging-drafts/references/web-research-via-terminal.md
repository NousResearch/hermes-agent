# Web Research via Terminal (Wikipedia & structured sources)

When `web_search` / `web_extract` / browser tools are unavailable or slow, use
curl against structured APIs directly from the terminal. The Wikipedia REST API
is particularly useful — it returns clean text with no HTML, no API key
required, and works with a simple curl call.

## Wikipedia REST API (recommended — cleanest output)

### Summary endpoint (best for quick answers)

```
GET https://en.wikipedia.org/api/rest_v1/page/summary/{Page_Title}
```

Returns: `extract` field with 2–3 paragraph plain-text summary, plus metadata
(title, description, thumbnail, page URL).

```bash
curl -s "https://en.wikipedia.org/api/rest_v1/page/summary/National_High_Speed_Rail_Corporation_Limited" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('extract',''))"
```

### Full page text via MediaWiki API (for long-form content)

```
GET https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext&titles={Page_Title}&format=json
```

Returns: full page text in the `extract` field under the page ID.

```bash
curl -s "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext&titles=List_of_high-speed_railway_lines_in_India&format=json" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); pages=d['query']['pages']; [print(p.get('extract','')) for p in pages.values()]"
```

### Search Wikipedia for relevant pages

```
GET https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json
```

Returns: `search[]` with titles, snippets (with `<span class="searchmatch">` highlights).

```bash
curl -s "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=%22Hyderabad%20Bengaluru%22%20high-speed%20rail&format=json" \
  | python3 -c "
import sys,json
d = json.load(sys.stdin)
for r in d.get('query',{}).get('search',[]):
    print(r['title'])
    snippet = r.get('snippet','')[:250].replace('<span class=\"searchmatch\">','*').replace('</span>','*')
    print('  ' + snippet)
"
```

## URL encoding for page titles

Wikipedia page titles use underscores for spaces and percent-encoding for
special characters. Common patterns:

| Character | Encoded form | Example |
|-----------|-------------|---------|
| space | `_` | `Hyderabad–Bengaluru_high-speed_rail_corridor` |
| – (en-dash) | `%E2%80%93` | `Hyderabad%E2%80%93Bengaluru` |
| / | `%2F` | rare in page titles |
| & | `%26` | rare in page titles |

**Shortcut**: use Python's `urllib.parse.quote()` to encode, then replace
`%20` with `_`:

```python
import urllib.parse
title = "Hyderabad–Bengaluru high-speed rail corridor"
encoded = urllib.parse.quote(title).replace('%20', '_')
# → Hyderabad%E2%80%93Bengaluru_high-speed_rail_corridor
url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
```

## When to use this over web_search / web_extract

| Situation | Recommended tool |
|-----------|-----------------|
| Need a factual summary of a known entity | Wikipedia REST API (this ref) |
| Searching for pages on a topic | Wikipedia search (this ref) + then summary |
| Need current news / recent events | web_search (Google) |
| Need content behind JS-rendered pages | browser tools |
| Need a structured data dump (tables, lists) | MediaWiki API full extract (this ref) |
| URL is known, page is plain-text | curl + web_extract both fine |

## NDR-specific domain research pattern

Voice messages frequently bundle **research + WhatsApp follow-up** in one
request — e.g. "research the high-speed rail body and then send a message to
Bhaskar about it." The sequence is:

1. Identify the subject to research
2. If it's a known entity (organization, person, project), try Wikipedia
   first — fastest and most structured
3. If Wikipedia doesn't have it, try web_search (if configured) or fall back
   to the subject's own website via curl/browser
4. Compile findings into a structured summary for the user
5. Only then proceed to the WhatsApp/message workflow (handled by the main
   skill)
