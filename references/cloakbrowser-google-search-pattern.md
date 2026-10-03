# CloakBrowser Google Search Pattern

## What
CloakBrowser provides a C++-patched Chromium that evades Google's bot detection and renders search results directly.

## When to Use
- When Brave Search API key is invalidated
- When DuckDuckGo returns empty results
- For Google search fallback/verification

## Tradeoffs
- Slower than APIs (Chromium boot overhead)
- Reliable as a verification or fallback

## Reference
See swarm-observability/references/cloakbrowser-google-search-pattern.md for detailed usage.