---
name: web-unblocker
description: Use when a website blocks you with Cloudflare, captcha, 403, or region-locked content. 跨平台反爬：绕过Cloudflare/验证码/403/地区锁。全球销售商及大陆供货商采购实测验证。Cross-platform via curl_cffi, works on Linux/macOS/Windows.
version: 1.0.0
author: andorexu
license: MIT
metadata:
  hermes:
    tags: [web, anti-crawl, cloudflare, curl-cffi, scraping, stealth, unblocker]
    related_skills: [curl-cffi-proxy]
---

# Web Unblocker — 跨平台反爬 / Cloudflare绕过

## Overview / 概述

Bypass Cloudflare, captcha walls, 403 blocks, and region locks using `curl_cffi` — a Python library that impersonates real browsers at the TLS level. No Windows dependency. Works on any OS with Python 3.8+.

使用 `curl_cffi` 绕过 Cloudflare、验证码、403 封禁和地区锁。跨平台，只需 `pip install curl_cffi` 一个依赖。

**Sibling skill:** `curl-cffi-proxy` — Windows-specific version with Astrill proxy support. Use that if you're on Windows with Astrill.

## When to Use / 触发场景

- Website returns Cloudflare challenge / "Checking your browser" / captcha / 验证码
- 403 Forbidden with normal `curl` or `urllib` / 被网站封禁
- Region-locked content (Chinese sites blocking non-CN IPs, or vice versa) / 地区锁
- JavaScript-rendered pages where `curl` gets empty body
- ImportYeti, Amazon, Google, LinkedIn when blocked
- User says "反爬" / "unblock" / "绕过" / "被挡了" / "打不开" / "Cloudflare"

**Don't use for:** normal pages that work fine with `curl` (overkill), large-scale scraping (rate limiting still applies), sites requiring login with OAuth flows.

## Quick Start

```python
from curl_cffi import requests

# Basic impersonation (Chrome 120 on Windows)
resp = requests.get("https://example.com", impersonate="chrome120")
print(resp.status_code, len(resp.text))
```

## Impersonation Profiles

| Profile | Browser | OS | Best for |
|---------|---------|----|----------|
| `chrome120` | Chrome 120 | Win/Mac | General purpose |
| `chrome110` | Chrome 110 | Win/Mac | Older sites |
| `safari15_5` | Safari 15.5 | Mac | Apple-friendly sites |
| `edge101` | Edge 101 | Windows | Microsoft sites |
| `firefox102` | Firefox 102 | Win/Mac | Privacy-focused sites |

Rotation strategy: if `chrome120` fails → try `chrome110` → then `safari15_5`.

## Installation

```bash
pip install curl_cffi
```

One dependency. That's it.

## Usage Patterns

### Pattern 1: Simple GET with retry

```python
from curl_cffi import requests
import time

def fetch(url, max_retries=3):
    profiles = ["chrome120", "chrome110", "safari15_5"]
    for attempt in range(max_retries):
        profile = profiles[attempt % len(profiles)]
        try:
            resp = requests.get(url, impersonate=profile, timeout=15)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 403:
                time.sleep(2)  # Back off on blocks
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None

html = fetch("https://blocked-site.com")
```

### Pattern 2: With proxy (for region-locked content)

```python
from curl_cffi import requests

proxies = {"http": "http://proxy:port", "https": "http://proxy:port"}
resp = requests.get(
    "https://region-locked-site.com",
    impersonate="chrome120",
    proxies=proxies,
    timeout=15
)
```

### Pattern 3: POST with JSON body

```python
from curl_cffi import requests

resp = requests.post(
    "https://api.example.com/search",
    json={"query": "stuff"},
    impersonate="chrome120",
    headers={"Content-Type": "application/json"}
)
data = resp.json()
```

### Pattern 4: Anti-crawl one-liner (terminal)

```bash
python3 -c "
from curl_cffi import requests
r = requests.get('$URL', impersonate='chrome120')
print(r.text[:3000])
"
```

## Regional Routing

| Source Region | Target | Strategy |
|--------------|--------|----------|
| China mainland | Chinese sites (1688, Baidu, Sogou) | Direct, no proxy needed |
| China mainland | Global sites (Google, LinkedIn) | Proxy required (Astrill/Clash) |
| Outside China | Chinese sites | Proxy to CN IP may help |
| Anywhere | Cloudflare-protected | curl_cffi impersonation first |

## Common Pitfalls

1. **Forgetting timeouts.** Always set `timeout=15` — blocked connections can hang forever.
2. **No backoff on 403.** If you get 403, wait 2-5 seconds before retrying. Rapid retries trigger permanent bans.
3. **Using the same profile every time.** Rotate between chrome120/chrome110/safari15_5 on failures.
4. **Sending no headers.** Add at minimum `User-Agent` (curl_cffi adds this automatically with impersonate).
5. **Assuming it bypasses everything.** Cloudflare Enterprise with JS challenges may still block. Fall back to browser tools for those.
6. **Confusing with the sibling skill.** `curl-cffi-proxy` is Windows-only with Astrill. This skill is the cross-platform version without OS-specific proxy wiring.

## When curl_cffi Fails

If impersonation doesn't work after 3 profile rotations with backoff, the site likely uses:
- Cloudflare Turnstile (interactive JS challenge)
- reCAPTCHA v2/v3
- Browser fingerprinting beyond TLS

Fallback: use `browser_navigate` to load the page in a real browser, then `browser_snapshot` to read content.

## Verification Checklist

- [ ] `pip install curl_cffi` succeeds
- [ ] `from curl_cffi import requests` works in Python
- [ ] Tested with `impersonate="chrome120"` on a known-good URL
- [ ] Proxy configured if needed for region-locked targets
- [ ] Timeouts set on all requests
- [ ] Profile rotation implemented for retries


## Author / 作者

- **GitHub:** [github.com/andorexu](https://github.com/andorexu)
- **Company / 公司:** 百赛联（深圳）科技有限公司
- **Email:** andore@sina.com

