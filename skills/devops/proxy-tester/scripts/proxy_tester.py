#!/usr/bin/env python3
"""Proxy Tester — diagnose liveness, latency, and IP-type classification.

No account creation. No third-party site interaction beyond a single IP-reveal
request through the proxy. All traffic goes to https://httpbin.org/ip by default
(you can override with --probe-url to your own endpoint).
"""

import argparse
import html
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from requests.auth import HTTPProxyAuth
from urllib3.util import parse_url

# ---------------------------------------------------------------------------
# IP / ASN intelligence (baked — update manually via script occasionally)
# ---------------------------------------------------------------------------

# Known datacenter ASNs (partial list — expand as needed)
DATACENTER_ASNS = {
    "AS13335",  # Cloudflare
    "AS16509",  # Amazon AWS
    "AS15169",  # Google
    "AS8075",   # Microsoft
    "AS20473",  # OVH
    "AS24940",  # Hetzner
    "AS14061",  # DigitalOcean
    "AS393406", # Google Cloud
    "AS14618",  # IBM
    "AS54825",  # Packet (Equinix)
}

# Mobile carrier patterns (substring in org/ASN description)
MOBILE_PATTERNS = {"verizon", "t-mobile", "o2", "vodafone", "telefónica", "orange", "megafon"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_proxy(uri: str) -> Dict:
    """Parse proxy URL into components. Raises ValueError on malformed."""
    parsed = parse_url(uri)
    if not parsed.scheme or not parsed.host or not parsed.port:
        raise ValueError(f"Invalid proxy URI: {uri}")

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https", "socks4", "socks5"):
        raise ValueError(f"Unsupported protocol: {scheme}")

    username, password = (None, None)
    if parsed.auth:
        # Split ONCE after the username so passwords containing ':' survive
        # (e.g. host:port:user:p@ss:w0rd -> user='user', password='p@ss:w0rd').
        # Matches convert_credentials.py, which preserves colon-bearing passwords.
        username, _, password = parsed.auth.partition(":")
        password = password or None

    return {
        "uri": uri,
        "scheme": scheme,
        "host": parsed.host,
        "port": parsed.port,
        "username": username,
        "password": password,
    }


def build_session(proxy_cfg: Dict) -> requests.Session:
    """Create a requests.Session configured for this proxy."""
    session = requests.Session()
    proxy_url = f"{proxy_cfg['scheme']}://{proxy_cfg['host']}:{proxy_cfg['port']}"
    if proxy_cfg['username']:
        # Inject creds into proxy URL or use auth hook
        proxy_url = f"{proxy_cfg['scheme']}://{proxy_cfg['username']}:{proxy_cfg['password'] or ''}@{proxy_cfg['host']}:{proxy_cfg['port']}"
        session.proxies = {"http": proxy_url, "https": proxy_url}
    else:
        session.proxies = {"http": proxy_url, "https": proxy_url}
    session.trust_env = False  # ignore local env proxy vars
    return session


def probe_proxy(proxy_cfg: Dict, probe_url: str, timeout: int) -> Dict:
    """Connect through proxy, fetch probe_url, return result dict."""
    session = build_session(proxy_cfg)
    result = {
        "proxy": proxy_cfg["uri"],
        "timestamp": time.time(),
        "success": False,
        "latency_ms": None,
        "exit_ip": None,
        "error": None,
        "error_type": None,
    }

    try:
        start = time.time()
        resp = session.get(probe_url, timeout=timeout)
        elapsed = (time.time() - start) * 1000
        result["latency_ms"] = round(elapsed, 1)

        if resp.status_code == 200:
            result["success"] = True
            # Parse JSON: expect {"origin": "x.x.x.x"}
            try:
                data = resp.json()
                result["exit_ip"] = data.get("origin", "").strip()
                result["response_snippet"] = None
            except Exception:
                # Non-JSON response (e.g. HTML) — clear exit_ip, store a clean snippet
                result["exit_ip"] = None
                import re
                text_only = re.sub(r"<[^>]+>", "", resp.text).strip()
                snippet = text_only[:80].replace("\n", " ") or "non-JSON response"
                result["response_snippet"] = snippet
        else:
            result["error"] = f"HTTP {resp.status_code}"
            result["error_type"] = "http_error"

    except requests.exceptions.ProxyError as e:
        result["error"] = str(e)
        result["error_type"] = "proxy_error"
    except requests.exceptions.ConnectTimeout:
        result["error"] = "connect timeout"
        result["error_type"] = "timeout"
    except requests.exceptions.ReadTimeout:
        result["error"] = "read timeout"
        result["error_type"] = "timeout"
    except Exception as e:
        result["error"] = str(e)
        result["error_type"] = "unknown"

    return result


def classify_ip(ip: str) -> str:
    """Heuristic classification of an IP address type using reverse DNS.
    Returns: 'datacenter', 'residential', 'mobile', or 'unknown'."""
    import socket
    import re

    if not ip:
        return "unknown"

    # Private/reserved ranges
    import ipaddress
    try:
        ip_obj = ipaddress.ip_address(ip)
        if ip_obj.is_private or ip_obj.is_reserved:
            return "private"
    except Exception:
        pass

    # Reverse DNS lookup
    try:
        hostname = socket.gethostbyaddr(ip)[0].lower()
    except Exception:
        hostname = ""

    # Datacenter keywords in reverse DNS
    dc_keywords = [
        "cloud", "compute", "aws", "azure", "gcp", "googlecloud",
        "amazonaws", "digitalocean", "vultr", "linode", "hetzner",
        "ovh", "ibm", "oracle", "scaleway", "packet", "equinix",
        "rackspace", "servercore", "nodebility", "hostinger",
    ]
    # Mobile carrier keywords
    mobile_keywords = [
        "verizon", "att", "t-mobile", "sprint", "o2", "vodafone",
        "telefonica", "orange", "telekom", "mtn", "claro", "vivo",
    ]

    for kw in dc_keywords:
        if kw in hostname:
            return "datacenter"
    for kw in mobile_keywords:
        if kw in hostname:
            return "mobile"

    # Fallback: if no strong signals, likely residential / unknown
    return "residential" if hostname else "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_proxies(path: Path) -> List[str]:
    lines = path.read_text().splitlines()
    proxies = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
    return proxies


def main():
    ap = argparse.ArgumentParser(description="Proxy Tester — liveness, latency, IP-type classifier")
    ap.add_argument("file", nargs="?", type=Path, help="Text file with one proxy URI per line (omit for interactive paste mode)")
    ap.add_argument("--output", "-o", type=Path, default=Path("./proxy-report"),
                    help="Output directory (default: ./proxy-report)")
    ap.add_argument("--concurrency", "-c", type=int, default=10,
                    help="Parallel workers (default: 10, max: 50)")
    ap.add_argument("--timeout", "-t", type=int, default=10,
                    help="Connection timeout in seconds (default: 10)")
    ap.add_argument("--probe-url", type=str, default="https://httpbin.org/ip",
                    help="URL to fetch through proxy (default: https://httpbin.org/ip)")
    ap.add_argument("--iterations", "-n", type=int, default=3,
                    help="How many times to test each proxy (default: 3)")
    ap.add_argument("--no-ip-intel", action="store_true",
                    help="Skip reverse-DNS IP-type classification")
    ap.add_argument("--report", choices=["md", "html"], default="md",
                    help="Output format: 'md' (default markdown table) or 'html' (branded dark/cyan report)")
    args = ap.parse_args()

    # Load proxies — either from file or interactive stdin
    if args.file:
        proxies = load_proxies(args.file)
        print(f"\n🔍 Proxy Tester\n   Input file: {args.file}\n")
    else:
        print("\n📋 Paste your proxy list (one per line). When done, press Ctrl+D (Unix) or Ctrl+Z (Windows):\n")
        lines = sys.stdin.read().splitlines()
        proxies = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
        print(f"✓ Loaded {len(proxies)} proxies from stdin.\n")
        args.output = args.output  # keep as-is

    if not proxies:
        print("❌ No proxies found. Provide a file or paste a list.")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"\n🔍 Proxy Tester\n   Input: {args.file} ({len(proxies)} proxies) · concurrency {args.concurrency}\n")

    concurrency = max(1, min(args.concurrency, 50))  # clamp per --help max

    # ── Bounded concurrent probing ───────────────────────────────────────────
    def test_one(uri: str) -> dict:
        """Probe a single proxy across N iterations and return its aggregated result."""
        try:
            proxy_cfg = parse_proxy(uri)
        except Exception as e:
            return {
                "proxy": uri,
                "success_rate": 0.0,
                "avg_latency_ms": None,
                "exit_ip": None,
                "ip_type": "unknown",
                "error": str(e),
                "iterations": [],
            }

        iter_results = []
        for i in range(args.iterations):
            iter_results.append(probe_proxy(proxy_cfg, args.probe_url, args.timeout))
            if i < args.iterations - 1:
                time.sleep(0.2)  # brief stagger between iterations

        successes = [r for r in iter_results if r["success"]]
        success_rate = len(successes) / args.iterations * 100
        avg_latency = (sum(r["latency_ms"] for r in successes) / len(successes)) if successes else None
        exit_ip = successes[0]["exit_ip"] if successes else None

        response_snippet = None
        for s in successes:
            snippet = s.get("response_snippet")
            if snippet:
                response_snippet = snippet
                break

        ip_type = "unknown"
        if not args.no_ip_intel and exit_ip:
            ip_type = classify_ip(exit_ip)

        return {
            "proxy": uri,
            "success_rate": round(success_rate, 1),
            "avg_latency_ms": round(avg_latency, 1) if avg_latency else None,
            "exit_ip": exit_ip,
            "response_snippet": response_snippet,
            "ip_type": ip_type,
            "iterations": iter_results,
        }

    all_results = []
    done = 0
    total = len(proxies)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_to_uri = {pool.submit(test_one, uri): uri for uri in proxies}
        for fut in as_completed(future_to_uri):
            uri = future_to_uri[fut]
            done += 1
            res = fut.result()
            all_results.append(res)
            # Preserve original proxy order for stable report output
            print(f"[{done:03d}/{total}] {uri}", flush=True)

    # Restore input order so summary/report match the file order
    order = {uri: i for i, uri in enumerate(proxies)}
    all_results.sort(key=lambda r: order.get(r["proxy"], 0))

    # Per-proxy human-readable summary line
    for r in all_results:
        uri = r["proxy"]
        sr = r["success_rate"]
        if sr == 100:
            print(f"✅ {uri} — alive — {r['avg_latency_ms']:.0f}ms — {r.get('exit_ip') or r.get('response_snippet') or '—'}")
        elif sr > 0:
            print(f"⚠ {uri} — flaky — {sr:.0f}% — {r['avg_latency_ms']:.0f}ms")
        else:
            err = (r.get("iterations") or [{}])[0].get("error", "unknown error")
            print(f"❌ {uri} — dead — {err}")

    # ── Write outputs ────────────────────────────────────────────────────────
    import json

    healthy = [r for r in all_results if r["success_rate"] == 100]
    dead = [r for r in all_results if r["success_rate"] == 0]
    avg_lat = sum(r["avg_latency_ms"] for r in healthy) / len(healthy) if healthy else 0
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    total = len(all_results)

    if args.report == "html":
        summary = args.output / "report.html"
        type_counts = {}
        for r in healthy:
            t = r.get("ip_type") or "unknown"
            type_counts[t] = type_counts.get(t, 0) + 1

        rows_html = ""
        for r in all_results:
            status_icon = "✅" if r["success_rate"] == 100 else ("⚠" if r["success_rate"] > 0 else "❌")
            latency = f"{r['avg_latency_ms']:.0f}ms" if r["avg_latency_ms"] else "—"
            ip = r.get("exit_ip") or "—"
            ip_short = ip[:15] + "…" if len(ip) > 15 else ip
            ip_type = r.get("ip_type") or "unknown"
            proxy_disp = html.escape(str(r["proxy"]))
            # Show response snippet for non-JSON probes (e.g. site returns HTML)
            if r.get("response_snippet") and ip == "—":
                # Escape: a probe endpoint could return markup that executes on open.
                snippet_esc = html.escape(r["response_snippet"])
                ip_display = f'<em style="color:var(--text-muted);font-size:0.75rem">{snippet_esc}</em>'
            else:
                ip_display = f'<code>{html.escape(ip_short)}</code>'
            row_class = "row-healthy" if r["success_rate"] == 100 else ("row-flaky" if r["success_rate"] > 0 else "row-dead")
            rows_html += f"""<tr class="{row_class}">
                <td class="proxy-cell">{proxy_disp}</td>
                <td>{status_icon}</td>
                <td>{latency}</td>
                <td>{ip_display}</td>
                <td><span class="tag tag-{html.escape(ip_type)}">{html.escape(ip_type)}</span></td>
                <td>{r["success_rate"]:.0f}%</td>
            </tr>"""

        type_badges = "".join(
            f'<span class="type-chip">{html.escape(t)}: {c}</span>' for t, c in sorted(type_counts.items())
        ) if type_counts else '<span class="type-chip">none classified</span>'

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Proxy Test Report — {html.escape(timestamp)}</title>
<style>
  :root {{
    --bg: #0d1117;
    --surface: #161b22;
    --border: #21262d;
    --text: #e6edf3;
    --text-muted: #7d8590;
    --cyan: #39c5cf;
    --cyan-dim: rgba(57,197,207,0.15);
    --green: #3fb950;
    --green-dim: rgba(63,185,80,0.15);
    --amber: #d29922;
    --amber-dim: rgba(210,153,34,0.15);
    --red: #f85149;
    --red-dim: rgba(248,81,73,0.15);
    --radius: 8px;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 2rem; }}
  .container {{ max-width: 1000px; margin: 0 auto; }}
  header {{ margin-bottom: 2rem; }}
  h1 {{ font-size: 1.5rem; font-weight: 600; color: var(--cyan); letter-spacing: -0.02em; }}
  .meta {{ color: var(--text-muted); font-size: 0.85rem; margin-top: 0.25rem; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
  .stat-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 1rem; }}
  .stat-label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); margin-bottom: 0.4rem; }}
  .stat-value {{ font-size: 1.6rem; font-weight: 700; color: var(--cyan); }}
  .stat-value.green {{ color: var(--green); }}
  .stat-value.amber {{ color: var(--amber); }}
  .type-chips {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.75rem; }}
  .type-chip {{ background: var(--cyan-dim); color: var(--cyan); font-size: 0.75rem; padding: 0.25rem 0.6rem; border-radius: 20px; border: 1px solid rgba(57,197,207,0.3); }}
  table {{ width: 100%; border-collapse: collapse; background: var(--surface); border-radius: var(--radius); overflow: hidden; border: 1px solid var(--border); }}
  th {{ text-align: left; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); padding: 0.75rem 1rem; border-bottom: 1px solid var(--border); background: rgba(255,255,255,0.02); }}
  td {{ padding: 0.7rem 1rem; font-size: 0.875rem; border-bottom: 1px solid var(--border); vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  .row-healthy {{ background: var(--green-dim); }}
  .row-flaky {{ background: var(--amber-dim); }}
  .row-dead {{ background: var(--red-dim); opacity: 0.7; }}
  .proxy-cell {{ font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.8rem; color: var(--text-muted); max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  code {{ font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.8rem; color: var(--cyan); background: var(--cyan-dim); padding: 0.1rem 0.4rem; border-radius: 4px; }}
  .tag {{ font-size: 0.7rem; padding: 0.2rem 0.5rem; border-radius: 4px; text-transform: uppercase; letter-spacing: 0.05em; }}
  .tag-datacenter {{ background: rgba(138,109,255,0.15); color: #a78bfa; }}
  .tag-residential {{ background: rgba(63,185,80,0.15); color: #56d364; }}
  .tag-mobile {{ background: rgba(210,153,34,0.15); color: #e3b341; }}
  .tag-unknown {{ background: rgba(125,133,144,0.15); color: var(--text-muted); }}
  footer {{ margin-top: 2rem; text-align: center; color: var(--text-muted); font-size: 0.75rem; }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>PROXY TEST REPORT</h1>
    <p class="meta">{html.escape(timestamp)} · {total} proxies · {args.iterations} iteration(s) · probe: {html.escape(args.probe_url)}</p>
  </header>

  <div class="stats">
    <div class="stat-card">
      <div class="stat-label">Total Tested</div>
      <div class="stat-value">{total}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Healthy</div>
      <div class="stat-value green">{len(healthy)}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Dead</div>
      <div class="stat-value {'red' if dead else 'green'}">{len(dead)}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Avg Latency</div>
      <div class="stat-value">{avg_lat:.0f}ms</div>
      <div class="type-chips">{type_badges}</div>
    </div>
  </div>

  <table>
    <thead>
      <tr>
        <th>Proxy</th>
        <th>Status</th>
        <th>Latency</th>
        <th>Exit IP</th>
        <th>Type</th>
        <th>Success</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>

  <footer>Generated by Hermes proxy-tester skill · {html.escape(args.probe_url)}</footer>
</div>
</body>
</html>"""
        with open(summary, "w") as f:
            f.write(html_content)
        print(f"\n✓ HTML report → {summary}")

    else:
        # Markdown (default)
        summary = args.output / "summary.md"
        with open(summary, "w") as f:
            f.write(f"# Proxy Test Report — {timestamp}\n\n")
            f.write("| Proxy | Status | Avg Latency | Exit IP | Type | Success |\n")
            f.write("|-------|--------|-------------|---------|------|--------|\n")
            for r in all_results:
                status = "✅" if r["success_rate"] == 100 else ("⚠" if r["success_rate"] > 0 else "❌")
                latency = f"{r['avg_latency_ms']:.0f}ms" if r["avg_latency_ms"] else "—"
                ip = r.get("exit_ip") or "—"
                if r.get("response_snippet") and not r.get("exit_ip"):
                    ip = f'"{r["response_snippet"]}"'
                f.write(f"| `{r['proxy']}` | {status} | {latency} | `{ip}` | {r.get('ip_type','unknown')} | {r['success_rate']:.0f}% |\n")
            f.write(f"\n**Healthy:** {len(healthy)}/{total} ({len(healthy)/total*100:.0f}%)\n")
            if healthy:
                f.write(f"**Average latency (healthy):** {avg_lat:.0f} ms\n")
        print(f"\n✓ Summary → {summary}")

    # Full details JSONL
    details = args.output / "details.jsonl"
    with open(details, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"✓ Details → {details}")

    # Healthy subset
    (args.output / "healthy.json").write_text(json.dumps(healthy, indent=2))
    (args.output / "dead.json").write_text(json.dumps(dead, indent=2))
    print(f"✓ Healthy: {len(healthy)}, Dead: {len(dead)}")

    # Group by type
    by_type = {}
    for r in healthy:
        t = r.get("ip_type") or "unknown"
        by_type.setdefault(t, []).append(r)
    (args.output / "by_type").mkdir(exist_ok=True)
    for t, items in by_type.items():
        (args.output / "by_type" / f"{t}.json").write_text(json.dumps(items, indent=2))
    print(f"✓ Grouped by type → by_type/\n")


if __name__ == "__main__":
    main()
