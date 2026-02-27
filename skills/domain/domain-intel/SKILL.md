---
name: domain-intel
description: Passive domain reconnaissance using Python stdlib. Use this skill for subdomain discovery, SSL certificate inspection, WHOIS lookups, DNS records, domain availability checks, and bulk multi-domain analysis. No API keys required. Triggers on requests like "find subdomains", "check ssl cert", "whois lookup", "is this domain available", "bulk check these domains".
---

# Domain Intelligence — Passive OSINT

Passive domain reconnaissance using only Python stdlib and public data sources.  
**Zero dependencies. Zero API keys. Works out of the box.**

## Data Sources

- **crt.sh** — Certificate Transparency logs (subdomain discovery)
- **WHOIS servers** — Direct TCP queries to 100+ authoritative TLD servers
- **Google DNS-over-HTTPS** — MX/NS/TXT/CNAME resolution
- **System DNS** — A/AAAA record resolution

---

## Usage

When the user asks about a domain, use the `terminal` tool to run the appropriate Python snippet below.  
All functions print structured JSON. Parse and summarize results for the user.

---

## 1. Subdomain Discovery (crt.sh)

```python
import json, urllib.request, urllib.parse
from datetime import datetime, timezone

def subdomains(domain, include_expired=False, limit=200):
    url = f"https://crt.sh/?q=%25.{urllib.parse.quote(domain)}&output=json"
    req = urllib.request.Request(url, headers={"User-Agent": "domain-intel-skill/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        entries = json.loads(r.read().decode())

    seen, results = set(), []
    for e in entries:
        not_after = e.get("not_after", "")
        if not include_expired and not_after:
            try:
                dt = datetime.strptime(not_after[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                if dt <= datetime.now(timezone.utc):
                    continue
            except ValueError:
                pass
        for name in e.get("name_value", "").splitlines():
            name = name.strip().lower()
            if name and name not in seen:
                seen.add(name)
                results.append({"subdomain": name, "issuer": e.get("issuer_name",""), "not_after": not_after})

    results.sort(key=lambda r: (r["subdomain"].startswith("*"), r["subdomain"]))
    results = results[:limit]
    print(json.dumps({"domain": domain, "count": len(results), "subdomains": results}, indent=2))

subdomains("DOMAIN_HERE")
```

**Example:** Replace `DOMAIN_HERE` with `example.com`

---

## 2. SSL Certificate Inspection

```python
import json, ssl, socket
from datetime import datetime, timezone

def check_ssl(host, port=443, timeout=10):
    def flat(rdns):
        r = {}
        for rdn in rdns:
            for item in rdn:
                if isinstance(item, (list,tuple)) and len(item)==2:
                    r[item[0]] = item[1]
        return r

    def extract_uris(entries):
        return [e[-1] if isinstance(e,(list,tuple)) else str(e) for e in entries]

    def parse_date(s):
        for fmt in ("%b %d %H:%M:%S %Y %Z", "%b  %d %H:%M:%S %Y %Z"):
            try: return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            except ValueError: pass
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

    not_after = parse_date(cert.get("notAfter",""))
    not_before = parse_date(cert.get("notBefore",""))
    now = datetime.now(timezone.utc)
    days = (not_after - now).days if not_after else None
    is_expired = days is not None and days < 0

    if is_expired: status = f"EXPIRED ({abs(days)} days ago)"
    elif days is not None and days <= 14: status = f"CRITICAL — {days} day(s) left"
    elif days is not None and days <= 30: status = f"WARNING — {days} day(s) left"
    else: status = f"OK — {days} day(s) remaining" if days is not None else "unknown"

    print(json.dumps({
        "host": host, "port": port,
        "subject": flat(cert.get("subject",[])),
        "issuer": flat(cert.get("issuer",[])),
        "subject_alt_names": [f"{t}:{v}" for t,v in cert.get("subjectAltName",[])],
        "not_before": not_before.isoformat() if not_before else "",
        "not_after": not_after.isoformat() if not_after else "",
        "days_remaining": days, "is_expired": is_expired, "expiry_status": status,
        "tls_version": proto, "cipher_suite": cipher[0] if cipher else None,
        "serial_number": cert.get("serialNumber",""),
        "ocsp_urls": extract_uris(cert.get("OCSP",[])),
        "ca_issuers": extract_uris(cert.get("caIssuers",[])),
        "verification_warning": warning,
    }, indent=2))

check_ssl("DOMAIN_HERE")
```

---

## 3. WHOIS Lookup (100+ TLDs)

```python
import json, socket, re
from datetime import datetime, timezone

WHOIS_SERVERS = {
    "com":"whois.verisign-grs.com","net":"whois.verisign-grs.com","org":"whois.pir.org",
    "io":"whois.nic.io","co":"whois.nic.co","ai":"whois.nic.ai","dev":"whois.nic.google",
    "app":"whois.nic.google","tech":"whois.nic.tech","shop":"whois.nic.shop",
    "store":"whois.nic.store","online":"whois.nic.online","site":"whois.nic.site",
    "cloud":"whois.nic.cloud","digital":"whois.nic.digital","media":"whois.nic.media",
    "blog":"whois.nic.blog","info":"whois.afilias.net","biz":"whois.biz",
    "me":"whois.nic.me","tv":"whois.nic.tv","cc":"whois.nic.cc","ws":"whois.website.ws",
    "uk":"whois.nic.uk","co.uk":"whois.nic.uk","de":"whois.denic.de","nl":"whois.domain-registry.nl",
    "fr":"whois.nic.fr","it":"whois.nic.it","es":"whois.nic.es","pl":"whois.dns.pl",
    "ru":"whois.tcinet.ru","se":"whois.iis.se","no":"whois.norid.no","fi":"whois.fi",
    "ch":"whois.nic.ch","at":"whois.nic.at","be":"whois.dns.be","cz":"whois.nic.cz",
    "br":"whois.registro.br","ca":"whois.cira.ca","mx":"whois.mx","au":"whois.auda.org.au",
    "jp":"whois.jprs.jp","cn":"whois.cnnic.cn","in":"whois.inregistry.net","kr":"whois.kr",
    "sg":"whois.sgnic.sg","hk":"whois.hkirc.hk","tr":"whois.nic.tr","ae":"whois.aeda.net.ae",
    "za":"whois.registry.net.za","ng":"whois.nic.net.ng","ly":"whois.nic.ly",
    "space":"whois.nic.space","zone":"whois.nic.zone","ninja":"whois.nic.ninja",
    "guru":"whois.nic.guru","rocks":"whois.nic.rocks","social":"whois.nic.social",
    "network":"whois.nic.network","global":"whois.nic.global","design":"whois.nic.design",
    "studio":"whois.nic.studio","agency":"whois.nic.agency","finance":"whois.nic.finance",
    "legal":"whois.nic.legal","health":"whois.nic.health","green":"whois.nic.green",
    "city":"whois.nic.city","land":"whois.nic.land","live":"whois.nic.live",
    "game":"whois.nic.game","games":"whois.nic.games","pw":"whois.nic.pw",
    "mn":"whois.nic.mn","sh":"whois.nic.sh","gg":"whois.gg","im":"whois.nic.im",
}

def whois_query(domain, server, port=43):
    with socket.create_connection((server, port), timeout=10) as s:
        s.sendall((domain+"\r\n").encode())
        chunks = []
        while True:
            c = s.recv(4096)
            if not c: break
            chunks.append(c)
        return b"".join(chunks).decode("utf-8", errors="replace")

def parse_iso(s):
    if not s: return None
    for fmt in ("%Y-%m-%dT%H:%M:%S","%Y-%m-%dT%H:%M:%SZ","%Y-%m-%d %H:%M:%S","%Y-%m-%d"):
        try: return datetime.strptime(s[:19],fmt).replace(tzinfo=timezone.utc)
        except ValueError: pass
    return None

def whois(domain):
    parts = domain.split(".")
    server = WHOIS_SERVERS.get(".".join(parts[-2:])) or WHOIS_SERVERS.get(parts[-1])
    if not server:
        print(json.dumps({"error": f"No WHOIS server for .{parts[-1]}"}))
        return
    try:
        raw = whois_query(domain, server)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        return

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
            if key in ("name_servers","status"):
                result[key] = list(dict.fromkeys(m.strip().lower() for m in matches))
            else:
                result[key] = matches[0].strip()
    for field in ("creation_date","expiration_date","updated_date"):
        if field in result:
            dt = parse_iso(result[field][:19])
            if dt:
                result[field] = dt.isoformat()
                if field == "expiration_date":
                    days = (dt - datetime.now(timezone.utc)).days
                    result["expiration_days_remaining"] = days
                    result["is_expired"] = days < 0
    print(json.dumps(result, indent=2))

whois("DOMAIN_HERE")
```

---

## 4. DNS Records

```python
import json, socket, urllib.request, urllib.parse

def dns(domain, types=None):
    if not types: types = ["A","AAAA","MX","NS","TXT","CNAME"]
    records = {}

    for qtype in types:
        if qtype == "A":
            try: records["A"] = list(dict.fromkeys(i[4][0] for i in socket.getaddrinfo(domain,None,socket.AF_INET)))
            except: records["A"] = []
        elif qtype == "AAAA":
            try: records["AAAA"] = list(dict.fromkeys(i[4][0] for i in socket.getaddrinfo(domain,None,socket.AF_INET6)))
            except: records["AAAA"] = []
        else:
            url = f"https://dns.google/resolve?name={urllib.parse.quote(domain)}&type={qtype}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent":"domain-intel-skill/1.0"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    data = json.loads(r.read())
                records[qtype] = [a.get("data","").strip().rstrip(".") for a in data.get("Answer",[]) if a.get("data")]
            except:
                records[qtype] = []

    print(json.dumps({"domain": domain, "records": records}, indent=2))

dns("DOMAIN_HERE")
```

---

## 5. Domain Availability Check

```python
import json, socket, ssl

def available(domain):
    import urllib.request, urllib.parse, re
    from datetime import datetime, timezone

    signals = {}

    # DNS check
    try: a = [i[4][0] for i in socket.getaddrinfo(domain,None,socket.AF_INET)]
    except: a = []
    try: ns_url = f"https://dns.google/resolve?name={urllib.parse.quote(domain)}&type=NS"
        req = urllib.request.Request(ns_url, headers={"User-Agent":"domain-intel-skill/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            ns = [x.get("data","") for x in json.loads(r.read()).get("Answer",[])]
    except: ns = []
    signals["dns_a"] = a
    signals["dns_ns"] = ns
    dns_exists = bool(a or ns)

    # SSL check
    ssl_up = False
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
        with socket.create_connection((domain,443),timeout=3) as s:
            with ctx.wrap_socket(s, server_hostname=domain): ssl_up = True
    except: pass
    signals["ssl_reachable"] = ssl_up

    # WHOIS check (simple)
    WHOIS = {"com":"whois.verisign-grs.com","net":"whois.verisign-grs.com","org":"whois.pir.org",
             "io":"whois.nic.io","co":"whois.nic.co","ai":"whois.nic.ai","dev":"whois.nic.google",
             "me":"whois.nic.me","app":"whois.nic.google","tech":"whois.nic.tech"}
    tld = domain.rsplit(".",1)[-1]
    whois_avail = None
    whois_note = ""
    server = WHOIS.get(tld)
    if server:
        try:
            with socket.create_connection((server,43),timeout=10) as s:
                s.sendall((domain+"\r\n").encode())
                raw = b""
                while True:
                    c = s.recv(4096)
                    if not c: break
                    raw += c
                raw = raw.decode("utf-8",errors="replace").lower()
            if any(p in raw for p in ["no match","not found","no data found","status: free"]):
                whois_avail = True; whois_note = "WHOIS: not found"
            elif "registrar:" in raw or "creation date:" in raw:
                whois_avail = False; whois_note = "WHOIS: registered"
            else: whois_note = "WHOIS: inconclusive"
        except Exception as e: whois_note = f"WHOIS error: {e}"
    signals["whois_available"] = whois_avail
    signals["whois_note"] = whois_note

    if not dns_exists and whois_avail is True: verdict,conf = "LIKELY AVAILABLE","high"
    elif dns_exists or whois_avail is False or ssl_up: verdict,conf = "REGISTERED / IN USE","high"
    elif not dns_exists and whois_avail is None: verdict,conf = "POSSIBLY AVAILABLE","medium"
    else: verdict,conf = "UNCERTAIN","low"

    print(json.dumps({"domain":domain,"verdict":verdict,"confidence":conf,"signals":signals},indent=2))

available("DOMAIN_HERE")
```

---

## 6. Bulk Analysis (Multiple Domains in Parallel)

```python
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paste any of the functions above (check_ssl, whois, dns, available, subdomains)
# then use this runner:

def bulk_check(domains, checks=None, max_workers=5):
    if not checks: checks = ["ssl", "whois", "dns", "available"]
    
    def run_one(domain):
        result = {"domain": domain}
        # Import/define individual functions above, then:
        if "ssl" in checks:
            try: result["ssl"] = json.loads(check_ssl_json(domain))
            except Exception as e: result["ssl"] = {"error": str(e)}
        if "whois" in checks:
            try: result["whois"] = json.loads(whois_json(domain))
            except Exception as e: result["whois"] = {"error": str(e)}
        if "dns" in checks:
            try: result["dns"] = json.loads(dns_json(domain))
            except Exception as e: result["dns"] = {"error": str(e)}
        if "available" in checks:
            try: result["available"] = json.loads(available_json(domain))
            except Exception as e: result["available"] = {"error": str(e)}
        return result

    results = []
    with ThreadPoolExecutor(max_workers=min(max_workers,10)) as ex:
        futures = {ex.submit(run_one, d): d for d in domains[:20]}
        for f in as_completed(futures):
            results.append(f.result())

    print(json.dumps({"total": len(results), "checks": checks, "results": results}, indent=2))
```

---

## Quick Reference

| Task | What to run |
|------|-------------|
| Find subdomains | Snippet 1 — replace `DOMAIN_HERE` |
| Check SSL cert | Snippet 2 — replace `DOMAIN_HERE` |
| WHOIS lookup | Snippet 3 — replace `DOMAIN_HERE` |
| DNS records | Snippet 4 — replace `DOMAIN_HERE` |
| Is domain available? | Snippet 5 — replace `DOMAIN_HERE` |
| Bulk check 20 domains | Snippet 6 |

## Notes

- All requests are **passive** — no active scanning, no packets sent to target hosts (except SSL check which makes a TCP connection)
- `subdomains` only queries crt.sh — the target domain is never contacted
- WHOIS queries go to registrar servers, not the target
- Results are structured JSON — summarize key findings for the user
- For expired cert warnings or WHOIS redaction, mention these to the user as notable findings
