---
name: web-unblocker
description: "Bypass Cloudflare, captcha, 403, region-locks. 跨平台反爬虫工具集。"
version: 1.0.0
author: andorexu
license: MIT
metadata:
  hermes:
    tags: [web, anti-crawl, cloudflare, curl-cffi, scraping, stealth, unblocker]
    related_skills: []
---

# Web Unblocker — 跨平台反爬 / Cloudflare绕过

## Overview / 概述

Bypass Cloudflare, captcha walls, 403 blocks, and region locks using `curl_cffi` — a Python library that impersonates real browsers at the TLS level. No Windows dependency. Works on any OS with Python 3.8+.

使用 `curl_cffi` 绕过 Cloudflare、验证码、403 封禁和地区锁。跨平台，只需 `pip install curl_cffi` 一个依赖。

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
6. **Assuming curl_cffi requires special proxy setup.** curl_cffi works directly — `pip install curl_cffi` is all you need. It impersonates browsers at the TLS level without OS-specific proxy wiring.
7. **Astrill VPN cannot be automated.** No CLI, no UI Automation elements (custom rendering), no tray icon detection. When user says "开梯子" and Astrill is the proxy → must ask user to manually connect. Always check `ProxyEnable` registry first — if already 1, proxy is on.
8. **Giving up on proxy too early.** User preference: "try everything before asking." Check ProxyEnable → search for proxy process → try CLI → try GUI automation → only then ask user.
9. **Bot browser detection (Apollo.io pattern).** Some sites (Apollo.io, Cloudflare-protected) detect Playwright/browser automation and block JS execution. Symptoms: curl gets HTTP 200, but `browser_navigate` shows empty page with loading spinner that never resolves. The page HTML loads but JavaScript never renders. **Diagnosis:** Check `browser_console` for `document.body.innerHTML.length` > 0 but `document.body.innerText` = "" and `document.querySelectorAll('input').length` = 0 after 5-8 seconds. **No workaround** — these sites require manual login or cookie injection from user's authenticated browser.
10. **Login redirect as diagnostic signal.** Some platforms redirect login flow to reveal account status. Example: 裁判文书网 (wenshu.court.gov.cn) — if clicking login button redirects to Alipay binding page ("是否绑定您的在线服务账号？"), the account is NOT registered/activated on that platform. Registered accounts proceed directly to dashboard. This redirect pattern is a diagnostic, not a failure — it tells you the account needs registration first.
11. **Don't assume proxy/ladder is the issue.** User correction (2026-06-12): "不要动不动就拿梯子关了说事" — don't keep blaming the proxy being off. When something fails, check actual error messages and root causes first. Proxy status is just one of many possible issues. Jumping to "梯子关了" as the explanation without verification frustrates the user and wastes time. Always diagnose the actual error before suggesting proxy as the cause.
12. **Don't mix browser + winremote for the same login flow.** User correction (2026-06-12): "不能混用，只能选择一个方案" — browser tools use independent headless Chromium with separate session/cookies, while winremote controls the user's desktop Edge with local cache/login state. Mixing them creates session confusion. Pick ONE approach and go all the way. Either: (a) pure browser — independent session, OCR captcha, manual login each time; or (b) pure winremote — user's Edge session, handles captcha via screenshot, but can't click JS-rendered elements.
13. **裁判文书网 login OAuth flow.** The login form is inside a cross-origin iframe at `account.court.gov.cn`. Two-step barrier: first a WAF text captcha (`/waf_text_captcha`), then the actual login form with phone/password/captcha. **Technique — navigate directly to the iframe OAuth URL** (extracted from `document.querySelector('iframe').src`) to bypass the WAF page and reach the login form directly. The captcha on the login form is embedded as a base64 data URL in the DOM at `document.querySelectorAll('img')[1].src` (100×30px, 4 characters, deformed font). **Pitfall:** Both RapidOCR and MiniCPM-V Q4 have poor accuracy on these captcha images — expect 3+ retry cycles. See `references/wenshu-login-flow.md` for full transcript.

## Loop Detection — When to Stop Retrying

**Rule:** If the same action produces the same result 3+ times, STOP and diagnose. Do not retry the same approach.

**Pattern recognition:**
- Click button → page changes to state A
- Click different element → page returns to original state
- Click button again → page changes to state A again
- **This is a loop.** The button click triggers an unavoidable redirect.

**What to do instead:**
1. Check if the redirect reveals diagnostic info (see pitfall #10)
2. Try a completely different approach (different URL, API endpoint, cookie injection)
3. If no alternative exists, report the blocker to user with diagnosis

**Anti-pattern:** Clicking the same element or repeating the same sequence expecting different results. User feedback: "你要总这么循环就停下吧，别折腾了" — stop looping, don't waste cycles.

## When curl_cffi Fails

If impersonation doesn't work after 3 profile rotations with backoff, the site likely uses:
- Cloudflare Turnstile (interactive JS challenge)
- reCAPTCHA v2/v3
- Browser fingerprinting beyond TLS

Fallback: use `browser_navigate` to load the page in a real browser, then `browser_snapshot` to read content.

---

## Captcha Defense Layers（验证码三层防御架构）

When browser navigation hits a captcha wall (slider, image selection, text OCR), use this **free, layered pipeline**. Do NOT reach for paid captcha services (2captcha/Capsolver) unless the user explicitly approves the cost.

### Architecture

```
① Stealth Layer: Playwright + playwright-stealth
   → Avoid triggering captcha in the first place
   → Injects navigator.webdriver=false, realistic plugins/languages
   ↓ captcha still triggered
② OCR Layer: RapidOCR + ddddocr (complementary)
   → RapidOCR: clean fonts, fast (3ms), on Windows via winremote
   → ddddocr: distorted/overlapping/noisy text, Chinese-optimized
   ↓ slider / image-selection / rotate-puzzle (beyond OCR)
③ Multimodal Layer: Qwen3.5-3B-Omni via Ollama (installing, pending completion)
   → Screenshot → vision LLM → "gap at x=260" → script drags
   → Hardware: Intel Arc 140V, 16GB shared VRAM — can run 3B-8B models
   → Model path: D:\Ollama\models (when installed)
```

### Installed Tools

| Layer | Tool | Location | Version |
|-------|------|----------|---------|
| Stealth | Playwright + playwright-stealth | WSL Hermes venv | playwright 1.60, stealth 2.0.3 |
| OCR-a | RapidOCR | Windows Python312 (winremote) | 3.x onnx |
| OCR-b | ddddocr | Windows Python312 | 1.6.1 |
| Multimodal | Qwen3.5-3B-Omni via Ollama | D:\Ollama (installing) | Pending |

Installation verified 2026-06-10. Chromium 148.0 headless shell confirmed working.

### Test Results (2026-06-10)

- **1688/Ali NC Slider**: browser JS event simulation FAILS (requires real mouse trajectory). Stealth layer may reduce trigger rate; full bypass needs multimodal or paid service.
- **Playwright launch**: ✅ Chromium 148 headless OK from WSL.
- **ddddocr model load**: ✅ instantiated successfully.

### Decision Record

- 2026-06-10: Adopted 3-layer free architecture. Rejected paid 2captcha/Capsolver for now. Rejected self-built OpenCV slider solver (high maintenance per-site).
- 2026-06-10: Chose Qwen3.5-3B-Omni over Qwen-VL-2B/MiniCPM-V 2.6 for multimodal layer. Omni has voice+video understanding, only 1GB more than VL-2B. User's Intel Arc 140V has 16GB shared VRAM — easily handles 3GB model.
- 2026-06-10: Ollama installation in progress via BitsTransfer (~200KB/s, ETA 1-2 hours). Will install to D:\Ollama with models at D:\Ollama\models.



## Verification Checklist

When a platform requires login and captcha blocks automated access, use **cookie injection** from the user's authenticated browser session.

### When to Use

- Platform has captcha that blocks automated login (1688/Taobao, etc.)
- User is already logged in on their local browser
- Need to maintain authenticated session for API calls

### Cookie Export Workflow

1. User opens platform in Edge/Chrome (already logged in)
2. Press `F12` → Console tab
3. Type: `copy(document.cookie)` → press Enter
4. Cookie string is now in clipboard → paste to agent

**Note:** `document.cookie` only returns non-HttpOnly cookies. For full auth (especially Taobao/1688), also need HttpOnly cookies from Edge Application panel:
- F12 → Application → Cookies → select domain
- Copy all cookies including HttpOnly ones

### ⚠️ CRITICAL: IP-Binding Pitfall

**Chinese platform cookies are IP-bound.** They only work from the same IP that generated them.

| Environment | Cookie Works? |
|-------------|:---:|
| Same Windows machine (curl_cffi) | ✅ YES |
| WSL (different IP) | ❌ NO |
| Browserbase/remote browser | ❌ NO |
| Different computer | ❌ NO |

**Verified 2026-06-10:** 1688 workbench login confirmed via Windows curl_cffi + exported cookies + same IP. WSL attempts all failed (redirected to login).

### Cookie Usage Pattern

```python
# On Windows Python312 (same IP as browser)
import curl_cffi.requests as requests

cookie_str = "_tb_token_=xxx; __cn_logon__=true; cookie1=yyy; ..."  # from user

cookie_dict = {}
for item in cookie_str.split("; "):
    if "=" in item:
        k, v = item.split("=", 1)
        cookie_dict[k.strip()] = v.strip()

resp = requests.get(
    "https://work.1688.com/",
    cookies=cookie_dict,
    impersonate="chrome120",
    timeout=15
)
# Check: resp.url should NOT contain "login"
```

### Cookie Storage

- Store in `E:\百龙资料\v3数据\<platform>_cookies.txt`
- Record in `accounts.xlsx` → 平台账号 Sheet → update status to "已登录(Cookie)"
- Note expiry dates — most Taobao cookies expire in 7 days
- **Auto-renewal:** Regular use keeps cookies alive; if expired, user re-exports

### 1688 Specific (Verified Working)

- Cookie file: `E:\百龙资料\v3数据\1688_cookies.txt`
- Key cookies: `__cn_logon__=true`, `__cn_logon_id__=andore`, `cookie1`, `cookie2`, `_tb_token_`
- Expiry: Most cookies 2026-06-17 (7 days), some 2027 (1 year+)
- Search pages are JS-rendered → curl_cffi gets empty shell; need Playwright for product lists

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

