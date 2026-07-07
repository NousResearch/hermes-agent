#!/usr/bin/env python3
"""Generate Brazil-focused discovery queries for unbroker-brasil.

Accepts either flags or a JSON object on stdin with keys:
full_name, aliases, emails, phones, locations.
"""
from __future__ import annotations

import argparse
import json
import sys
from urllib.parse import quote_plus

PRIORITY_DOMAINS = [
    "jusbrasil.com.br",
    "escavador.com",
    "telelistas.net",
    "reclameaqui.com.br",
    "cnpj.biz",
    "casadosdados.com.br",
    "empresascnpj.com",
]


def as_list(v):
    if not v:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [str(v).strip()]


def quoted(s: str) -> str:
    return '"' + s.replace('"', '') + '"'


def build_queries(full_name: str, aliases: list[str], emails: list[str], phones: list[str], locations: list[str]) -> list[dict]:
    queries: list[dict] = []
    names = [full_name] + aliases
    names = [n for n in names if n]

    for name in names:
        queries.append({"lane": "identity", "query": quoted(name)})
        for loc in locations:
            queries.append({"lane": "identity_location", "query": f"{quoted(name)} {quoted(loc)}"})
        for domain in PRIORITY_DOMAINS:
            queries.append({"lane": "site_priority", "domain": domain, "query": f"{quoted(name)} site:{domain}"})

    for email in emails:
        queries.append({"lane": "email", "query": quoted(email)})
        for domain in PRIORITY_DOMAINS:
            queries.append({"lane": "email_site_priority", "domain": domain, "query": f"{quoted(email)} site:{domain}"})

    for phone in phones:
        variants = {phone, phone.replace(" ", ""), phone.replace("-", ""), phone.replace("(", "").replace(")", "")}
        for v in sorted(x for x in variants if x):
            queries.append({"lane": "phone", "query": quoted(v)})
            for loc in locations:
                queries.append({"lane": "phone_location", "query": f"{quoted(v)} {quoted(loc)}"})

    # Deduplicate while preserving order.
    seen = set()
    out = []
    for item in queries:
        key = item["query"]
        if key in seen:
            continue
        seen.add(key)
        item["search_url"] = "https://www.google.com/search?q=" + quote_plus(item["query"])
        out.append(item)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-name", default="")
    ap.add_argument("--alias", action="append", default=[])
    ap.add_argument("--email", action="append", default=[])
    ap.add_argument("--phone", action="append", default=[])
    ap.add_argument("--location", action="append", default=[])
    ap.add_argument("--json", action="store_true", help="read subject JSON from stdin")
    args = ap.parse_args()

    data = {}
    if args.json:
        data = json.load(sys.stdin)

    full_name = args.full_name or data.get("full_name", "") or data.get("name", "")
    aliases = args.alias + as_list(data.get("aliases"))
    emails = args.email + as_list(data.get("emails"))
    phones = args.phone + as_list(data.get("phones"))
    locations = args.location + as_list(data.get("locations")) + as_list(data.get("prior_locations"))

    result = build_queries(full_name, aliases, emails, phones, locations)
    print(json.dumps({"count": len(result), "queries": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
