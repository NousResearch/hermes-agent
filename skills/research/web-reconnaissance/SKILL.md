---
name: web-reconnaissance
description: "Web reconnaissance and API reverse engineering for discovering public APIs, endpoints, and architecture of web applications."
version: 1.0.0
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [recon, api-discovery, web-analysis, reverse-engineering, nextjs, ssr, csr]
---

# Web Reconnaissance & API Reverse Engineering

Systematic discovery of public APIs, endpoints, and architecture from web applications without authentication.

## When to Use

- User asks to "find APIs" for a website
- Need to understand the backend architecture of a web app
- Looking for undocumented endpoints or data sources
- Wanting to identify external services/CDNs used by an application

## Core Approach

1. **Static Analysis First** (always fastest, zero requests)
2. **Targeted Probing** (intelligent endpoint guessing)
3. **Service Discovery** (identify external dependencies)
4. **Dynamic Analysis** (browser network tab, if available)

## Static Analysis

### 1. Inspect HTML Source

```bash
curl -s 'https://target.com/' > /tmp/page.html
```

Look for:
- **Embedded JSON data** in `<script>` tags (common in SSR/RSC)
- **CDN URLs** (xesque.target.com, cdn.target.com)
- **Video/platform references** (nivo, bunny, cloudfront)
- **API endpoint hints** in fetch/axios calls

```bash
# Extract JSON blobs from script tags
grep -oP '__NEXT_DATA__.*?script>' /tmp/page.html | sed 's/<[^>]*>//g' | jq .

# Find CDN/service references
grep -oE 'https://[a-z0-9.-]+\.(target|cdn|cloudflare|bunny|nivo)\.com' /tmp/page.html
```

### 2. Check HTTP Headers

```bash
curl -sI 'https://target.com/'
```

Key indicators:
- `Server: Vercel` / `Server: Next.js` → Next.js App Router (likely RSC)
- `x-nextjs-data` → Next.js API routes may exist
- `x-powered-by` → framework identification
- `cache-control` patterns → static vs dynamic

### 3. Analyze JavaScript Chunks

Download and inspect webpack/bundled chunks:

```bash
# Extract chunk URLs from HTML
grep -oP '_next/static/chunks/[a-f0-9-]+\.js' /tmp/page.html | sort -u

# Download a chunk (example)
curl -s 'https://target.com/_next/static/chunks/main-app-abc123.js' | grep -E '(fetch|axios|api|endpoint|graphql)' | head -20
```

## Targeted Probing

### Common API Patterns

```bash
# REST endpoints
for endpoint in /api/v1/contents /api/v1/courses /api/v1/journeys /api/catalog /api/content; do
  status=$(curl -s -o /dev/null -w "%{http_code}" "https://target.com$endpoint")
  echo "$status - $endpoint"
done

# GraphQL
curl -s -o /dev/null -w "%{http_code}" -X POST 'https://target.com/graphql'

# Next.js API routes
curl -s -o /dev/null -w "%{http_code}" 'https://target.com/api/search'
curl -s -o /dev/null -w "%{http_code}" 'https://target.com/api/catalog'
```

### HTTP Method Variation

```bash
# Try different methods
for method in GET POST OPTIONS HEAD; do
  status=$(curl -s -X $method -o /dev/null -w "%{http_code}" 'https://target.com/api/content')
  echo "$method - $status"
done
```

## Service Discovery

### Identify External Services

Patterns found in HTML/JSON:

```
xesque.rocketseat.dev    # Custom CDN for assets
vz-dc851587-83d.b-cdn.net # Bunny CDN
api.stripe.com           # Payment
analytics.google.com     # Tracking
```

**Why this matters**: External services often have public APIs that reveal data indirectly.

### CDN Pattern Matching

```bash
# Extract all external domains
curl -s 'https://target.com/' | grep -oE 'https://[^"'\'' ]+\.(com|io|dev|net)' | sort -u
```

## Modern Stack Patterns

### Next.js App Router (RSC)

**Indicators**:
- No client-side XHR/Fetch calls in Network tab
- JSON data embedded in `__NEXT_DATA__` or `self.__next_f.push`
- Fast initial page load, no "loading..." states

**Behavior**:
- APIs are **internal** and called server-side
- No public REST/GraphQL for anonymous users
- Data serialized into HTML at build/render time

**What this means**:
- Static analysis = your best bet
- Dynamic probing = mostly 404s
- Look for authenticated endpoints (requires login)

### Traditional SSR/SPA

**Indicators**:
- XHR/Fetch calls visible in Network tab
- `fetch()` in source code
- Loading states/spinners

**Behavior**:
- Public APIs likely exist
- Dynamic probing more effective
- GraphQL introspection may work

## Sitemap & Robots.txt Parsing

Before brute-forcing endpoints, check for exposed route maps:

```bash
# Check sitemap
curl -s 'https://target.com/sitemap.xml' | grep -oP '<loc>\K[^<]+' | sort -u

# Check sitemap index (multiple sitemaps)
curl -s 'https://target.com/sitemap_index.xml' | grep -oP '<sitemap>\s*<loc>\K[^<]+'

# Check robots.txt (may reveal hidden endpoints)
curl -s 'https://target.com/robots.txt'
```

**What this reveals**:
- Public route patterns (`/jornada/{slug}`, `/catalogo/*`)
- Hidden/disallowed routes (robots.txt Disallow directives)
- Content structure and taxonomies

## Systematic Brute Force (Python)

For large-scale endpoint testing:

```python
import subprocess

def test_endpoints(base_url, endpoints):
    results = []
    for endpoint in endpoints:
        url = f'{base_url}{endpoint}'
        for method in ['GET', 'POST', 'OPTIONS']:
            status = subprocess.run(
                ['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', '-X', method, url],
                capture_output=True,
                text=True
            ).stdout.strip()
            if status not in ['404', '000']:
                results.append({
                    'endpoint': endpoint,
                    'method': method,
                    'status': status
                })
    return results

endpoints = [
    '/api/v1/contents', '/api/v1/courses', '/api/v1/journeys',
    '/api/catalog', '/api/content', '/graphql',
    '/me', '/auth/login', '/auth/session'
]
results = test_endpoints('https://target.com', endpoints)
```

**Key patterns to test**:
- Versioned: `/api/v1/*`, `/api/v2/*`
- Unversioned: `/api/*`, `/*`
- GraphQL: `/graphql`, `/api/graphql`
- Auth: `/auth/*`, `/login`, `/logout`, `/me`, `/session`

## Subdomain Enumeration

Test common API/Service subdomains:

```bash
for subdomain in api api-v1 api-v2 platform platform-api graphql ws websocket cdn content; do
  status=$(curl -s -o /dev/null -w "%{http_code}" "https://${subdomain}.target.com/")
  echo "$status ${subdomain}.target.com"
done
```

**Patterns**:
- `api.domain.com` - Main API
- `platform.domain.com` - Platform-specific API
- `graphql.domain.com` - GraphQL endpoint
- `cdn.domain.com` - Static assets
- `vz-*.b-cdn.net` - Bunny CDN video thumbnails

## Route Pattern Recognition

Extract route patterns from sitemap/HTML:

```bash
# Extract dynamic route patterns from sitemap
curl -s 'https://target.com/sitemap.xml' | grep -oP '<loc>\K[^<]+' | sed 's|/[^/]*$|/{slug}|' | sort -u

# Example output from Rocketseat:
# /jornada/{slug}
# /jornada/{slug}/visao-geral
# /jornada/{slug}/conteudos
# /jornada/{slug}/sala/{node-slug}
```

**Why this matters**: Reveals taxonomy and potential authenticated routes even if public 404s.

## Browser Analysis (When Available)

If browser tools work:

1. Open DevTools → Network tab
2. Filter by XHR/Fetch
3. Reload page → watch all API calls
4. Copy cURL from any promising request
5. Test without auth tokens first

## Pitfalls & Gotchas

### ❌ Don't Do

- Assume APIs exist just because it's a SPA
- Skip static analysis and jump to brute force
- Claim "browser tools broken" as a permanent constraint
- Report 404s as "no API exists" (could be auth-only)

### ✅ Do

- Start with `curl -sI` and HTML inspection
- Look for embedded JSON data (Next.js RSC pattern)
- Test both HTTP/2 and HTTP/1.1 (some services only accept one)
- Try OPTIONS/HEAD before GET (sometimes reveals hidden routes)
- Document what you found AND what you ruled out

### Common Failure Patterns

1. **Browser timeouts/DevToolsActivePort** → Fall back to static analysis, don't abort
2. **CORS blocking client-side calls** → Use server-side `curl` instead
3. **404s on all /api/* routes** → May be SSR/RSC, check embedded data
4. **Blank responses** → May require auth or specific headers

## Workflow Summary

```
1. curl -sI <url>                    → Check headers, identify stack
2. curl -s <url> | grep -i 'api'     → Look for endpoint hints
3. Extract __NEXT_DATA__ / JSON blobs → Parse embedded data
4. Probe /api/*, /graphql, /v1/*     → Targeted endpoint testing
5. Identify CDN/external services    → Map external dependencies
6. Document findings                → What worked, what didn't, why
```

## Report Template

Use `SKILL_DIR/references/report-template.md` for documenting investigation findings in a structured format.