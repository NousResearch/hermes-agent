#!/usr/bin/env python3
"""Probe which backend Nous Portal uses for a model, and its quantization on OpenRouter.

Usage:
  python3 nous_backend.py <model-id> [--crossref]

Examples:
  python3 nous_backend.py deepseek/deepseek-v4-flash
  python3 nous_backend.py moonshotai/kimi-k2.5 --crossref

Reads the Nous Portal OAuth token from ~/.hermes/auth.json (refreshed token
preferred path: if the stored access_token is expired this script prints a
warning and exits; run `hermes login --provider nous` to refresh first).

Makes ONE tiny chat request (max_tokens=5) through the Portal, reads the
`provider` field from the response (the backend that served it), then — with
--crossref — looks up that backend's quantization level for the model via the
OpenRouter model page (public, no key needed).
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

AUTH = Path.home() / ".hermes" / "auth.json"
PORTAL_BASE = "https://inference-api.nousresearch.com/v1"


def load_token() -> str | None:
    try:
        data = json.loads(AUTH.read_text())
        entry = data.get("providers", {}).get("nous", {})
        tok = entry.get("access_token")
        expires = entry.get("expires_at", "")
        return tok, expires
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR reading {AUTH}: {exc}", file=sys.stderr)
        return None, None


def portal_provider(model: str, token: str) -> str | None:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
    }).encode()
    req = urllib.request.Request(
        f"{PORTAL_BASE}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "hermes-cli/quantcheck",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        d = json.load(urllib.request.urlopen(req, timeout=60))
        return d.get("provider")
    except urllib.error.HTTPError as e:
        print(f"Portal request failed: HTTP {e.code}: {e.read()[:200]!r}", file=sys.stderr)
        return None


# Map Portal response provider display names -> OpenRouter endpoint slug prefixes.
# OpenRouter slugs look like "novita/fp8", "streamlake/fp8", "cloudflare", etc.
def slug_candidates(provider_name: str) -> list[str]:
    n = provider_name.lower().replace(" ", "").replace("-", "")
    aliases = {
        "novita": ["novita"],
        "streamlake": ["streamlake"],
        "moonshotai": ["moonshotai"],
        "arceeai": ["arcee"],
        "alibaba": ["alibaba"],
        "baidu": ["baidu"],
        "cloudflare": ["cloudflare"],
        "deepinfra": ["deepinfra"],
        "digitalocean": ["digitalocean"],
        "akashml": ["akashml"],
        "chutes": ["chutes"],
        "coreweave": ["coreweave"],
        "gmicloud": ["gmicloud"],
        "mancer": ["mancer"],
        "parasail": ["parasail"],
        "siliconflow": ["siliconflow"],
        "venice": ["venice"],
        "phala": ["phala"],
        "atlascloud": ["atlas-cloud", "atlascloud"],
        "fireworks": ["fireworks"],
        "together": ["together"],
        "azure": ["azure"],
        "amazonbedrock": ["amazon-bedrock"],
    }
    return aliases.get(n, [n])


def openrouter_quant(model: str, provider_name: str) -> tuple[str | None, list[tuple[str, str]]]:
    base = model.split(":", 1)[0]
    url = f"https://openrouter.ai/{base}"
    req = urllib.request.Request(url, headers={"Accept": "text/html", "User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")

    rows: set[tuple[str, str]] = set()
    for m in re.finditer(r'provider_slug\\":\\"([^"\\]+)\\"', html):
        tail = html[m.end() : m.end() + 400]
        qm = re.search(r'quantization\\":\\"([^"\\]+)\\"', tail)
        if qm:
            rows.add((m.group(1), qm.group(1)))

    cands = slug_candidates(provider_name)
    for slug, q in sorted(rows):
        slug_norm = slug.lower().split("/")[0].replace("-", "")
        if any(slug_norm == c.replace("-", "") or slug_norm.startswith(c.replace("-", "")) for c in cands):
            return q, sorted(rows)
    return None, sorted(rows)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: nous_backend.py <model-id> [--crossref]", file=sys.stderr)
        sys.exit(2)
    model = sys.argv[1]
    crossref = "--crossref" in sys.argv

    tok, expires = load_token()
    if not tok:
        sys.exit(1)

    print(f"model: {model}")
    print(f"portal token expires: {expires}")
    backend = portal_provider(model, tok)
    if backend is None:
        print("no provider field returned (likely first-party serving or an error).")
        sys.exit(0)
    print(f"nous backend: {backend}")

    if crossref:
        q, all_rows = openrouter_quant(model, backend)
        if not all_rows:
            print("openrouter cross-reference: no endpoint data found for this model.")
        else:
            print("openrouter endpoints for this model:")
            for slug, level in all_rows:
                mark = "  <-- nous backend" if slug.lower().startswith(backend.lower().replace(" ", "")[:6]) else ""
                print(f"  {slug:30s} {level}{mark}")
            if q:
                print(f"CROSSREF RESULT: {backend} serves this model at '{q}'")
                if q in ("fp16", "bf16"):
                    print("verdict: full precision.")
                elif q in ("fp8", "mxfp8"):
                    print("verdict: fp8 — check whether the model is QAT/natively fp8 (parity) or a downgraded copy.")
                else:
                    print(f"verdict: '{q}' — reduced precision vs native weights.")


if __name__ == "__main__":
    main()
