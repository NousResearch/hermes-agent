#!/usr/bin/env python3
"""Proxy Tester — diagnose liveness, latency, and IP-type classification.

No account creation. No third-party site interaction beyond a single IP-reveal
request through the proxy. All traffic goes to https://httpbin.org/ip by default
(you can override with --probe-url to your own endpoint).
"""

import argparse
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

    return {
        "uri": uri,
        "scheme": scheme,
        "host": parsed.host,
        "port": parsed.port,
        "username": parsed.auth.split(":")[0] if parsed.auth and ":" in parsed.auth else parsed.auth,
        "password": parsed.auth.split(":")[1] if parsed.auth and ":" in parsed.auth else None,
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
            except Exception:
                result["exit_ip"] = resp.text.strip()[:100]
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

    print(f"\n🔍 Proxy Tester\n   Input: {args.file} ({len(proxies)} proxies)\n")
    all_results = []

    # ── Iterate over proxies ──────────────────────────────────────────────────
    for idx, uri in enumerate(proxies, 1):
        print(f"[{idx:03d}/{len(proxies)}] Testing {uri} … ", end="", flush=True)
        try:
            proxy_cfg = parse_proxy(uri)
        except Exception as e:
            print(f"❌ malformed: {e}")
            all_results.append({
                "proxy": uri,
                "success_rate": 0.0,
                "avg_latency_ms": None,
                "exit_ip": None,
                "ip_type": "unknown",
                "error": str(e),
                "iterations": [],
            })
            continue

        # Run N iterations
        iter_results = []
        for i in range(args.iterations):
            res = probe_proxy(proxy_cfg, args.probe_url, args.timeout)
            iter_results.append(res)
            if i < args.iterations - 1:
                time.sleep(0.2)  # brief stagger between iterations

        # Aggregate
        successes = [r for r in iter_results if r["success"]]
        success_rate = len(successes) / args.iterations * 100
        avg_latency = (sum(r["latency_ms"] for r in successes) / len(successes)) if successes else None
        exit_ip = successes[0]["exit_ip"] if successes else None

        # Classification (if enabled & we have an IP)
        ip_type = "unknown"
        if not args.no_ip_intel and exit_ip:
            ip_type = classify_ip(exit_ip)  # placeholder — expand with real ASN DB

        # Summary print
        if success_rate == 100:
            print(f"✅ alive — {avg_latency:.0f}ms — {exit_ip} — {ip_type}")
        elif success_rate > 0:
            print(f"⚠ flaky — {success_rate:.0f}% — {avg_latency:.0f}ms — {exit_ip}")
        else:
            print(f"❌ dead — {iter_results[0]['error']}")

        # Store aggregated result
        agg = {
            "proxy": uri,
            "success_rate": round(success_rate, 1),
            "avg_latency_ms": round(avg_latency, 1) if avg_latency else None,
            "exit_ip": exit_ip,
            "ip_type": ip_type,
            "iterations": iter_results,
        }
        all_results.append(agg)

    # ── Write outputs ────────────────────────────────────────────────────────
    import json

    # Summary markdown
    summary = args.output / "summary.md"
    with open(summary, "w") as f:
        f.write(f"# Proxy Test Report — {time.strftime('%Y-%m-%d')}\n\n")
        f.write("| Proxy | Status | Avg Latency | Exit IP | Type | Success |\n")
        f.write("|-------|--------|-------------|---------|------|---------|\n")
        for r in all_results:
            status = "✅" if r["success_rate"] == 100 else ("⚠" if r["success_rate"] > 0 else "❌")
            latency = f"{r['avg_latency_ms']:.0f}ms" if r["avg_latency_ms"] else "—"
            ip = r["exit_ip"] or "—"
            f.write(f"| `{r['proxy']}` | {status} | {latency} | {ip} | {r['ip_type']} | {r['success_rate']:.0f}% |\n")

        healthy = [r for r in all_results if r["success_rate"] == 100]
        f.write(f"\n**Healthy:** {len(healthy)}/{len(all_results)} ({len(healthy)/len(all_results)*100:.0f}%)\n")
        if healthy:
            avg_lat = sum(r["avg_latency_ms"] for r in healthy) / len(healthy)
            f.write(f"**Average latency (healthy):** {avg_lat:.0f} ms\n")

    print(f"\n✓ Summary → {summary}")

    # Full details JSONL
    details = args.output / "details.jsonl"
    with open(details, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"✓ Details → {details}")

    # Healthy subset
    healthy = [r for r in all_results if r["success_rate"] == 100]
    (args.output / "healthy.json").write_text(json.dumps(healthy, indent=2))
    dead = [r for r in all_results if r["success_rate"] == 0]
    (args.output / "dead.json").write_text(json.dumps(dead, indent=2))
    print(f"✓ Healthy: {len(healthy)}, Dead: {len(dead)}")

    # Group by type
    by_type = {}
    for r in healthy:
        t = r["ip_type"] or "unknown"
        by_type.setdefault(t, []).append(r)
    (args.output / "by_type").mkdir(exist_ok=True)
    for t, items in by_type.items():
        (args.output / "by_type" / f"{t}.json").write_text(json.dumps(items, indent=2))
    print(f"✓ Grouped by type → by_type/\n")


if __name__ == "__main__":
    main()
