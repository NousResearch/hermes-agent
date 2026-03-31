#!/usr/bin/env python3
"""
Domain Intelligence — Passive OSINT via Python stdlib.

Usage:
    python domain_intel.py subdomains example.com
    python domain_intel.py ssl example.com
    python domain_intel.py whois example.com
    python domain_intel.py dns example.com
    python domain_intel.py available example.com
    python domain_intel.py bulk example.com github.com google.com --checks ssl,dns

All output is structured JSON. No dependencies beyond Python stdlib.
Works on Linux, macOS, and Windows.
"""

import json
import re
import socket
import ssl
import sys
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone


# ─── Subdomain Discovery (crt.sh) ──────────────────────────────────────────

def subdomains(domain, include_expired=False, limit=200):
    """Find subdomains via Certificate Transparency logs."""
    url = f"https://crt.sh/?q=%25.{urllib.parse.quote(domain)}&output=json"
    req = urllib.request.Request(url, headers={
        "User-Agent": "domain-intel-skill/1.0", "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=15) as r:
        entries = json.loads(r.read().decode())

    seen, results = set(), []
    now = datetime.now(timezone.utc)
    for e in entries:
        not_after = e.get("not_after", "")
        if not include_expired and not_after:
            try:
                dt = datetime.strptime(not_after[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                if dt <= now:
                    continue
            except ValueError:
                pass
        for name in e.get("name_value", "").splitlines():
            name = name.strip().lower()
            if name and name not in seen:
                seen.add(name)
                results.append({
                    "subdomain": name,
                    "issuer": e.get("issuer_name", ""),
                    "not_after": not_after,
                })

    results.sort(key=lambda r: (r["subdomain"].startswith("*"), r["subdomain"]))
    return {"domain": domain, "count": min(len(results), limit), "subdomains": results[:limit]}


# ─── SSL Certificate Inspection ────────────────────────────────────────────

def check_ssl(host, port=443, timeout=10):
    """Inspect the TLS certificate of a host."""
    def flat(rdns):
        r = {}
        for rdn in rdns:
            for item in rdn:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    r[item[0]] = item[1]
        return r

    def parse_date(s):
        for fmt in ("%b %d %H:%M:%S %Y %Z", "%b  %d %H:%M:%S %Y %Z"):
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        return None

    warning = None
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as s:
                cert, cipher, proto = s.getpeercert(), s.cipher(), s.version()
    except ssl.SSLCertVerificationError as e:
        warning = str(e)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as s:
                cert, cipher, proto = s.getpeercert(), s.cipher(), s.version()

    not_after = parse_date(cert.get("notAfter", ""))
    now = datetime.now(timezone.utc)
    days = (not_after - now).days if not_after else None
    is_expired = days is not None and days < 0

    if is_expired:
        status = f"EXPIRED ({abs(days)} days ago)"
    elif days is not None and days <= 14:
        status = f"CRITICAL — {days} day(s) left"
    elif days is not None and days <= 30:
        status = f"WARNING — {days} day(s) left"
    else:
        status = f"OK — {days} day(s) remaining" if days is not None else "unknown"

    return {
        "host": host, "port": port,
        "subject": flat(cert.get("subject", [])),
        "issuer": flat(cert.get("issuer", [])),
        "subject_alt_names": [f"{t}:{v}" for t, v in cert.get("subjectAltName", [])],
        "not_before": parse_date(cert.get("notBefore", "")).isoformat() if parse_date(cert.get("notBefore", "")) else "",
        "not_after": not_after.isoformat() if not_after else "",
        "days_remaining": days, "is_expired": is_expired, "expiry_status": status,
        "tls_version": proto,
        "cipher_suite": cipher[0] if cipher else None,
        "serial_number": cert.get("serialNumber", ""),
        "verification_warning": warning,
    }


# ─── WHOIS Lookup ──────────────────────────────────────────────────────────

WHOIS_SERVERS = {
    "com": "whois.verisign-grs.com", "net": "whois.verisign-grs.com",
    "org": "whois.pir.org", "io": "whois.nic.io", "co": "whois.nic.co",
    "ai": "whois.nic.ai", "dev": "whois.nic.google", "app": "whois.nic.google",
    "tech": "whois.nic.tech", "shop": "whois.nic.shop", "store": "whois.nic.store",
    "online": "whois.nic.online", "site": "whois.nic.site", "cloud": "whois.nic.cloud",
    "digital": "whois.nic.digital", "media": "whois.nic.media", "blog": "whois.nic.blog",
    "info": "whois.afilias.net", "biz": "whois.biz", "me": "whois.nic.me",
    "tv": "whois.nic.tv", "cc": "whois.nic.cc", "ws": "whois.website.ws",
    "uk": "whois.nic.uk", "co.uk": "whois.nic.uk", "de": "whois.denic.de",
    "nl": "whois.domain-registry.nl", "fr": "whois.nic.fr", "it": "whois.nic.it",
    "es": "whois.nic.es", "pl": "whois.dns.pl", "ru": "whois.tcinet.ru",
    "se": "whois.iis.se", "no": "whois.norid.no", "fi": "whois.fi",
    "ch": "whois.nic.ch", "at": "whois.nic.at", "be": "whois.dns.be",
    "cz": "whois.nic.cz", "br": "whois.registro.br", "ca": "whois.cira.ca",
    "mx": "whois.mx", "au": "whois.auda.org.au", "jp": "whois.jprs.jp",
    "cn": "whois.cnnic.cn", "in": "whois.inregistry.net", "kr": "whois.kr",
    "sg": "whois.sgnic.sg", "hk": "whois.hkirc.hk", "tr": "whois.nic.tr",
    "ae": "whois.aeda.net.ae", "za": "whois.registry.net.za",
    "space": "whois.nic.space", "zone": "whois.nic.zone", "ninja": "whois.nic.ninja",
    "guru": "whois.nic.guru", "rocks": "whois.nic.rocks", "live": "whois.nic.live",
    "game": "whois.nic.game", "games": "whois.nic.games",
}


def whois_lookup(domain):
    """Query WHOIS servers for domain registration info."""
    parts = domain.split(".")
    server = WHOIS_SERVERS.get(".".join(parts[-2:])) or WHOIS_SERVERS.get(parts[-1])
    if not server:
        return {"error": f"No WHOIS server for .{parts[-1]}"}

    try:
        with socket.create_connection((server, 43), timeout=10) as s:
            s.sendall((domain + "\r\n").encode())
            chunks = []
            while True:
                c = s.recv(4096)
                if not c:
                    break
                chunks.append(c)
            raw = b"".join(chunks).decode("utf-8", errors="replace")
    except Exception as e:
        return {"error": str(e)}

    patterns = {
        "registrar": r"(?:Registrar|registrar):\s*(.+)",
        "creation_date": r"(?:Creation Date|Created|created):\s*(.+)",
        "expiration_date": r"(?:Registry Expiry Date|Expiration Date|Expiry Date):\s*(.+)",
        "updated_date": r"(?:Updated Date|Last Modified):\s*(.+)",
        "name_servers": r"(?:Name Server|nserver):\s*(.+)",
        "status": r"(?:Domain Status|status):\s*(.+)",
        "dnssec": r"DNSSEC:\s*(.+)",
    }
    result = {"domain": domain, "whois_server": server}
    for key, pat in patterns.items():
        matches = re.findall(pat, raw, re.IGNORECASE)
        if matches:
            if key in ("name_servers", "status"):
                result[key] = list(dict.fromkeys(m.strip().lower() for m in matches))
            else:
                result[key] = matches[0].strip()

    for field in ("creation_date", "expiration_date", "updated_date"):
        if field in result:
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(result[field][:19], fmt).replace(tzinfo=timezone.utc)
                    result[field] = dt.isoformat()
                    if field == "expiration_date":
                        days = (dt - datetime.now(timezone.utc)).days
                        result["expiration_days_remaining"] = days
                        result["is_expired"] = days < 0
                    break
                except ValueError:
                    pass
    return result


# ─── DNS Records ───────────────────────────────────────────────────────────

def dns_records(domain, types=None):
    """Resolve DNS records using system DNS + Google DoH."""
    if not types:
        types = ["A", "AAAA", "MX", "NS", "TXT", "CNAME"]
    records = {}

    for qtype in types:
        if qtype == "A":
            try:
                records["A"] = list(dict.fromkeys(
                    i[4][0] for i in socket.getaddrinfo(domain, None, socket.AF_INET)
                ))
            except Exception:
                records["A"] = []
        elif qtype == "AAAA":
            try:
                records["AAAA"] = list(dict.fromkeys(
                    i[4][0] for i in socket.getaddrinfo(domain, None, socket.AF_INET6)
                ))
            except Exception:
                records["AAAA"] = []
        else:
            url = f"https://dns.google/resolve?name={urllib.parse.quote(domain)}&type={qtype}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "domain-intel-skill/1.0"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    data = json.loads(r.read())
                records[qtype] = [
                    a.get("data", "").strip().rstrip(".")
                    for a in data.get("Answer", []) if a.get("data")
                ]
            except Exception:
                records[qtype] = []

    return {"domain": domain, "records": records}


# ─── Domain Availability Check ─────────────────────────────────────────────

def check_available(domain):
    """Check domain availability using passive signals (DNS + WHOIS + SSL)."""
    signals = {}

    # DNS
    try:
        a = [i[4][0] for i in socket.getaddrinfo(domain, None, socket.AF_INET)]
    except Exception:
        a = []

    try:
        ns_url = f"https://dns.google/resolve?name={urllib.parse.quote(domain)}&type=NS"
        req = urllib.request.Request(ns_url, headers={"User-Agent": "domain-intel-skill/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            ns = [x.get("data", "") for x in json.loads(r.read()).get("Answer", [])]
    except Exception:
        ns = []

    signals["dns_a"] = a
    signals["dns_ns"] = ns
    dns_exists = bool(a or ns)

    # SSL
    ssl_up = False
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with socket.create_connection((domain, 443), timeout=3) as s:
            with ctx.wrap_socket(s, server_hostname=domain):
                ssl_up = True
    except Exception:
        pass
    signals["ssl_reachable"] = ssl_up

    # WHOIS (quick check)
    tld = domain.rsplit(".", 1)[-1]
    server = WHOIS_SERVERS.get(tld)
    whois_avail = None
    whois_note = ""
    if server:
        try:
            with socket.create_connection((server, 43), timeout=10) as s:
                s.sendall((domain + "\r\n").encode())
                raw = b""
                while True:
                    c = s.recv(4096)
                    if not c:
                        break
                    raw += c
                raw = raw.decode("utf-8", errors="replace").lower()
            if any(p in raw for p in ["no match", "not found", "no data found", "status: free"]):
                whois_avail = True
                whois_note = "WHOIS: not found"
            elif "registrar:" in raw or "creation date:" in raw:
                whois_avail = False
                whois_note = "WHOIS: registered"
            else:
                whois_note = "WHOIS: inconclusive"
        except Exception as e:
            whois_note = f"WHOIS error: {e}"

    signals["whois_available"] = whois_avail
    signals["whois_note"] = whois_note

    if not dns_exists and whois_avail is True:
        verdict, conf = "LIKELY AVAILABLE", "high"
    elif dns_exists or whois_avail is False or ssl_up:
        verdict, conf = "REGISTERED / IN USE", "high"
    elif not dns_exists and whois_avail is None:
        verdict, conf = "POSSIBLY AVAILABLE", "medium"
    else:
        verdict, conf = "UNCERTAIN", "low"

    return {"domain": domain, "verdict": verdict, "confidence": conf, "signals": signals}


# ─── Bulk Analysis ─────────────────────────────────────────────────────────

COMMAND_MAP = {
    "subdomains": subdomains,
    "ssl": check_ssl,
    "whois": whois_lookup,
    "dns": dns_records,
    "available": check_available,
}


def bulk_check(domains, checks=None, max_workers=5):
    """Run multiple checks across multiple domains in parallel."""
    if not checks:
        checks = ["ssl", "whois", "dns"]

    def run_one(d):
        entry = {"domain": d}
        for check in checks:
            fn = COMMAND_MAP.get(check)
            if fn:
                try:
                    entry[check] = fn(d)
                except Exception as e:
                    entry[check] = {"error": str(e)}
        return entry

    results = []
    with ThreadPoolExecutor(max_workers=min(max_workers, 10)) as ex:
        futures = {ex.submit(run_one, d): d for d in domains[:20]}
        for f in as_completed(futures):
            results.append(f.result())

    return {"total": len(results), "checks": checks, "results": results}


# ─── CLI Entry Point ───────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()
    args = sys.argv[2:]

    if command == "bulk":
        # Parse --checks flag
        checks = None
        domains = []
        i = 0
        while i < len(args):
            if args[i] == "--checks" and i + 1 < len(args):
                checks = [c.strip() for c in args[i + 1].split(",")]
                i += 2
            else:
                domains.append(args[i])
                i += 1
        result = bulk_check(domains, checks)
    elif command in COMMAND_MAP:
        result = COMMAND_MAP[command](args[0])
    else:
        print(f"Unknown command: {command}")
        print(f"Available: {', '.join(COMMAND_MAP.keys())}, bulk")
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
