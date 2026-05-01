---
name: proxy-tester
description: Test a list of proxies for liveness, latency, and type classification (datacenter/residential/mobile). No account creation — purely network diagnostics for your own proxy pool.
version: 1.0.0
author: Hermes Agent (built for Suhaas)
license: MIT
prerequisites:
  tools: [terminal]
  python_packages: ["requests>=2.0", "urllib3"]
metadata:
  hermes:
    tags: [security, networking, proxies, diagnostics, defensive]
    related_skills: [referral-fraud-detector, hermes-agent]
---

# Proxy Tester

Diagnostic tool for your own proxy pool. Checks liveness, latency, and classifies proxy type (datacenter vs residential vs mobile) using IP-intel heuristics. No account creation, no site interaction — just connect-and-measure.

## When to use this skill

- You have a list of SOCKS5/HTTP proxies and want to know which are alive
- Measuring proxy performance (latency, success rate) before using them for any task
- Auditing your own proxy provider's claims (e.g., "residential" vs actual IP ranges)
- Building a healthy proxy rotation for legitimate crawling/scraping of your own sites
- Pre-deployment validation before running a large batch job through a proxy

**NOT for:** Using proxies to create accounts, bypass geo-blocks, or automate actions on third-party sites. This is purely diagnostic.

## Supported protocols

- HTTP/HTTPS proxies
- SOCKS4/SOCKS5 proxies
- Format: `protocol://host:port` (e.g., `socks5://123.45.67.89:1080`)
- Optional auth: `protocol://user:pass@host:port`

## What it tests

| Check | What it means |
|-------|----------------|
| **Connect** | TCP handshake succeeds within timeout |
| **HTTPS CONNECT** | For HTTP proxies: can tunnel TLS? (required for modern sites) |
| **Latency** | Round-trip time for a tiny HEAD request through proxy |
| **IP reveal** | What external IP the proxy presents (for geolocation/type checks) |
| **Type classification** | Heuristics: ASN / IP range → datacenter vs residential vs mobile |
| **Success rate** | Out of N attempts, how many complete without error |

## Usage

### Quick-start: from colon-separated credentials

Many proxy providers export credentials as `host:port:user:pass`. Convert first:

```bash
# Convert and pipe directly into proxy-audit
convert_credentials.py --scheme http < proxies_raw.txt | proxy-audit --output ~/proxy-report --iterations 3

# Or save converted file first
convert_credentials.py --scheme http < proxies_raw.txt > proxies_uri.txt
proxy-audit proxies_uri.txt --output ~/proxy-report
```

### Standard modes

Call the `proxy-audit` script (installed to `~/.hermes/scripts/`):

```bash
# Mode 1 — from file (one proxy URI per line)
proxy-audit ~/my-proxies.txt --output ~/proxy-report

# Mode 2 — interactive paste (no file argument)
proxy-audit --output ~/proxy-report
# Paste your proxy list, one per line. Press Ctrl+D when done.
```

### Flags

```
--output DIR, -o DIR   Output directory (default: ./proxy-report)
--concurrency N, -c N  Parallel workers (default: 10, max: 50)
--timeout SEC, -t SEC  Connection timeout in seconds (default: 10)
--probe-url URL        URL to fetch through proxy
                       (default: https://httpbin.org/ip)
--iterations N, -n N   How many times to test each proxy (default: 3)
--no-ip-intel          Skip reverse-DNS IP-type classification (faster)
```

**Why `httpbin.org/ip`?** It returns JSON `{"origin": "x.x.x.x"}` — perfect for confirming the proxy's exit IP. You can override with your own endpoint (e.g., your site's `/api/ip`) if you prefer.

## Output

```
proxy-report/
├── summary.md          # Markdown table: proxy, status, latency(avg), ip, type, success_rate
├── details.jsonl       # Line-delimited JSON of every probe attempt
├── healthy.json        # Subset: proxies that passed all iterations
├── dead.json           # Subset: proxies that never connected
└── by_type/
    ├── datacenter.json
    ├── residential.json
    └── mobile.json     # Grouped by inferred IP class
```

**`summary.md` example:**
```markdown
# Proxy Test Report — 2026-05-01

| Proxy | Status | Avg Latency | Exit IP | Type | Success |
|-------|--------|-------------|---------|------|---------|
| socks5://123.45.67.89:1080 | ✅ alive | 124ms | 123.45.67.89 | datacenter | 3/3 |
| http://user:pass@…:8080 | ❌ auth_failed | — | — | — | 0/3 |
| socks5://2001:db8::1:1080 | ⚠ timeout | — | — | — | 0/3 |

**Healthy:** 1/3 (33%)  
**Average latency:** 124ms  
**Types:** 1 datacenter
```

## Classification heuristics (IP-intel)

When `--no-ip-intel` is **not** set, the skill uses reverse DNS heuristics:

- **Reverse DNS lookup** — checks PTR record for provider keywords (`cloud`, `aws`, `digitalocean` → datacenter; `verizon`, `t-mobile` → mobile)
- **Private/reserved ranges** — RFC1918 and other reserved blocks flagged as `private`
- **Carrier keyword lists** — 20+ datacenter and mobile patterns bundled
- **No external API keys** — everything happens locally

**Accuracy:** Pattern-based, not perfect. Some residential proxies exit via carrier-grade NAT that looks residential. Some cloud VMs have neutral hostnames and will be classified `unknown`. Use as a signal, not a verdict.

**Caveats:** Heuristic only. Some residential proxies exit through datacenter NATs and will be misclassified.

## Pitfalls

- **Stale proxies** — many will be dead; always test before use
- **Rate-limited proxies** — some allow only N requests/min; this test uses few requests but real workloads may still trigger limits
- **GeoIP mismatch** — IP intelligence is approximate; ASN != guaranteed type
- **IPv6** support depends on your system's IPv6 stack; many proxies are IPv4-only
- **Auth failures** — wrong username/password shows as `407 Proxy Authentication Required`
- **Credential URL construction bug (pre-2026-05-01)** — Older versions built credentialed proxy URLs without the scheme (`http://user:pass@host:port` → `user:pass@host:port`), causing `requests` to raise `InvalidProxyURL: malformed and could be missing the host`. Fixed by ensuring `scheme://` prefix is included when credentials are present. Verify your installed `proxy_tester.py` has `f"{scheme}://{username}:{password}@{host}:{port}"` in `build_session()`.

## Related

- `referral-fraud-detector` — once you have healthy proxies, audit your referral logs for IP-based abuse patterns
- `hermes-agent` — general purpose agent that can run this skill as part of larger workflows
- `cron` — schedule periodic proxy health checks

## Example: Automated proxy health monitoring

```bash
# Check your proxy pool every 6 hours and alert on failures
hermes cron add \
  --name "proxy-health" \
  "0 */6 * * *" \
  "hermes skill proxy-tester test ~/proxies.txt --output ~/proxy-logs/\n\nIf healthy count < 3, send alert: hermes gateway send --channel #alerts 'Proxy pool degraded'"
```

## Verification

After updating the skill or when debugging proxy failures, run the included integrity checker:

```bash
# From the skill directory or via Hermes
~/.hermes/skills/devops/proxy-tester/scripts/verify_fix.py
# or
hermes skill proxy-tester verify
```

This confirms the credential URL scheme fix is present in `proxy_tester.py`.

## Reference

**`scripts/convert_credentials.py`** — Convert `host:port:user:pass` lines to proper `scheme://user:pass@host:port` URIs. Useful when your proxy provider exports credentials in colon-delimited format. See its docstring for usage.
