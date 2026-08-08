# Native Fetch Provider

Local HTTP fetch + readability extract provider for Hermes Agent.  
**No API key required.** Register as `web.extract_backend: native`.

## Usage

```yaml
# config.yaml
web:
  search_backend: ddgs       # or searxng, brave-free, etc.
  extract_backend: native    # local HTTP extraction, no API key

plugins:
  enabled:
    - web/native
```

### Zero-config

Explicit config is optional. On an install with **no web API keys and nothing
configured**, backend selection now auto-resolves to the free stack:

- **search** → `ddgs` (when the `ddgs` package is importable)
- **extract** → `native` (when this plugin is installed)

A deliberately configured paid backend always wins — the free providers are
only chosen as the fallback when nothing else can serve the capability.

Install dependencies:

```bash
pip install "hermes-agent[native-fetch]"
# or manually:
uv pip install "readability-lxml>=0.8.4,<0.10" "html2text>=2025.4.15,<2026"
```

## How It Works

1. Receives a URL to extract
2. Fetches the page via `httpx` (HTTP GET)
3. Extracts main content via Mozilla `readability-lxml`
4. Converts to clean Markdown via `html2text`
5. Returns structured result to the agent

Extract-only — pair with any search provider (`ddgs`, `searxng`, etc.).

## Configuration

All behavioral settings live under `web.native` in `config.yaml` (defaults
shown).

```yaml
web:
  extract_backend: native
  native:
    timeout: 30                  # HTTP request timeout (seconds)
    max_redirects: 5             # Max redirects to follow (each hop SSRF-checked)
    max_response_bytes: 2000000  # Max raw response body size (bytes, 2 MB)
    max_chars: 50000             # Default characters returned per URL
    max_chars_cap: 200000        # Absolute hard cap on returned chars
    cache_ttl: 900               # In-memory cache TTL (seconds, 15 min)
    readability: true            # Enable readability-lxml main-content extraction
    use_trusted_proxy: false     # Route through HTTP_PROXY/HTTPS_PROXY when set
    user_agent: ""               # Override User-Agent; empty → provider default
```

## Dependencies

- `readability-lxml` — Mozilla Readability for main-content extraction
- `html2text` — HTML to Markdown conversion
- `httpx` — Async HTTP client (core Hermes dependency)

## Security

- SSRF protection via `async_is_safe_url` — blocks requests to private/internal networks
- Redirects are followed manually and **every hop is re-validated** with
  `async_is_safe_url` before the next request, so a public URL cannot redirect
  into a private/internal address
- URL secrets detection (`_PREFIX_RE`) runs in `web_extract_tool` before dispatch
  — blocks URLs containing API keys/tokens
- No JavaScript execution — static HTML only