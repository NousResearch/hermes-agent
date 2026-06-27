---
name: domain-intel
description: Passive domain reconnaissance using Python stdlib. Use this skill for subdomain discovery, SSL certificate inspection, WHOIS lookups, DNS records, domain availability checks, and bulk multi-domain analysis. No API keys required. Triggers on requests like "find subdomains", "check ssl cert", "whois lookup", "is this domain available", "bulk check these domains".
license: MIT
---

Passive domain intelligence using only Python stdlib and public data sources.
Zero dependencies. Zero API keys. Works out of the box.

## Capabilities

- Subdomain discovery via crt.sh certificate transparency logs
- Live SSL/TLS certificate inspection (expiry, cipher, SANs, TLS version)
- WHOIS lookup — supports 100+ TLDs via direct TCP queries
- DNS records: A, AAAA, MX, NS, TXT, CNAME
- Domain availability check (DNS + WHOIS + SSL signals)
- Bulk multi-domain analysis in parallel (up to 20 domains)

## Data Sources

- crt.sh — Certificate Transparency logs
- WHOIS servers — Direct TCP to 100+ authoritative TLD servers  
- Google DNS-over-HTTPS — MX/NS/TXT/CNAME resolution
- System DNS — A/AAAA records
