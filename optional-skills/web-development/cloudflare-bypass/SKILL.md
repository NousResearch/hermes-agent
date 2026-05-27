---
name: cloudflare-bypass
description: Bypass Cloudflare with TLS fingerprint impersonation (curl_cffi). Get clean markdown from protected pages. Zero cost, zero API keys.
version: 2.0.0
author: akifcankilic
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [cloudflare, bypass, web-scraping, markdown, curl_cffi, scraper]
    related_skills: [scrapling, web-extract]
prerequisites:
  commands: [python3]
  pip: [curl_cffi, html2text]
---

# Cloudflare Bypass

**TL;DR:** `pip install curl_cffi html2text` + `python3 cfget.py <url>` = bypass Cloudflare, get clean markdown. $0, zero API keys, zero browser deps.

## What it does

Bypasses Cloudflare's "Just a moment..." / "Checking your browser" protection. Works where curl, wget, and headless Chromium fail. **curl_cffi** impersonates real Chrome at the TLS fingerprint level to get through.

## How it works

```
curl_cffi → TLS fingerprint impersonation (Chrome 124) → Cloudflare bypass → HTML
    ↓
html2text → markdown conversion → clean output
```

> **Why curl_cffi?** cloudscraper only handles basic CF challenges. curl_cffi impersonates a real browser at the TLS layer — it works on tougher sites (Eksi Sozluk entry pages, 9gag, medium).

## Installation

```bash
pip install curl_cffi html2text
```

If you hit PEP 668:
```bash
pip install --break-system-packages curl_cffi html2text
```

## Usage

```bash
python3 cfget.py https://example.com            # one-shot, markdown output
python3 cfget.py https://example.com | head -20 # pipe
python3 cfget.py https://example.com > page.md  # save to file
```

### Auto fallback

If `curl_cffi` is not installed, falls back to `cloudscraper`. If `html2text` is missing, strips HTML tags and outputs plain text. Only requires Python 3.

## Limitations

- **Client-side rendered SPAs** won't work (React/Vue/Angular — no JS execution)
- **Multi-layer protection** (sahibinden.com: Cloudflare + custom bot detection) needs residential proxy
- **No rate limiting** — be polite or your IP gets banned

## Tested sites

| Site | Status | Notes |
|------|--------|-------|
| eksisozluk.com | ✅ | Homepage, debe, entry, topic pages |
| 9gag.com | ✅ | Homepage and subpages |
| webtekno.com | ✅ | Tech news |
| donanimhaber.com | ✅ | Tech news |
| cnnturk.com | ✅ | News site |
| fanatik.com.tr | ✅ | Sports news |
| trendyol.com | ✅ | E-commerce |
| roblox.com | ✅ | Gaming platform |
| steampowered.com | ✅ | Game store |
| medium.com | ✅ | Blog platform |
| fandom.com | ✅ | Wiki platform |
| itch.io | ✅ | Game marketplace |
| sahibinden.com | ❌ | Multi-layer CF + custom detection, proxies needed |

## cfget.py

Single file, ~60 lines, three-tier fallback:

```python
# 1. curl_cffi (primary) — TLS fingerprint impersonation
# 2. cloudscraper (fallback) — basic CF challenge solver
# 3. html2text (primary) — HTML → markdown
#    regex (fallback) — strip tags, plain text
```

---

*Built with vibe coding. Open a PR, file an issue, contribute.*
