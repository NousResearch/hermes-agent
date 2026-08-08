#!/usr/bin/env python3
"""
price.py -- read-only on-chain token facts via the public DexScreener API.

No API key or credentials. GET requests only. Documented public service:
https://docs.dexscreener.com/  (api.dexscreener.com)

Usage:
  python price.py <token_address>             # facts for one token (highest-liquidity pair)
  python price.py --search "<name or symbol>" # find same-name tokens (brand-squatting check)

Prints JSON to stdout. Designed to never crash: on any failure it prints
{"error": "..."} so the calling agent can record "not found" instead of breaking.
"""
import json
import argparse
import urllib.request
import urllib.parse
import urllib.error

DEX_TOKENS = "https://api.dexscreener.com/latest/dex/tokens/"
DEX_SEARCH = "https://api.dexscreener.com/latest/dex/search?q="
TIMEOUT = 15
UA = "crypto-alpha-research-skill/0.1 (+https://github.com/catfrommarss/hermes-skills)"


def _get(url):
    req = urllib.request.Request(
        url, headers={"User-Agent": UA, "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summary(pair):
    liquidity = pair.get("liquidity") or {}
    base = pair.get("baseToken") or {}
    return {
        "chain": pair.get("chainId"),
        "dex": pair.get("dexId"),
        "name": base.get("name"),
        "symbol": base.get("symbol"),
        "address": base.get("address"),
        "priceUsd": _num(pair.get("priceUsd")),
        "fdv": _num(pair.get("fdv")),
        "marketCap": _num(pair.get("marketCap")),
        "liquidityUsd": _num(liquidity.get("usd")),
        "volume24h": _num((pair.get("volume") or {}).get("h24")),
        "priceChange": pair.get("priceChange") or {},
        "pairAddress": pair.get("pairAddress"),
        "url": pair.get("url"),
    }


def _by_liquidity(pairs):
    return sorted(
        pairs, key=lambda p: (p.get("liquidity") or {}).get("usd") or 0, reverse=True
    )


def token_facts(address):
    data = _get(DEX_TOKENS + urllib.parse.quote(address))
    pairs = data.get("pairs") or []
    if not pairs:
        return {
            "address": address,
            "found": False,
            "note": "no DEX pairs found (unlisted, wrong chain, or not a token)",
        }
    ranked = _by_liquidity(pairs)
    best = _summary(ranked[0])
    best["found"] = True
    best["pairsCount"] = len(pairs)
    best["otherPairs"] = [_summary(p) for p in ranked[1:4]]
    best["source"] = "dexscreener"
    return best


def search_name(query):
    data = _get(DEX_SEARCH + urllib.parse.quote(query))
    pairs = data.get("pairs") or []
    seen = set()
    candidates = []
    for pair in _by_liquidity(pairs):
        addr = (pair.get("baseToken") or {}).get("address")
        if not addr or addr in seen:
            continue
        seen.add(addr)
        candidates.append(_summary(pair))
        if len(candidates) >= 15:
            break
    return {
        "query": query,
        "candidatesCount": len(candidates),
        "note": (
            "Multiple distinct addresses with the same name = brand-squatting "
            "risk. Confirm the official contract via the project's own channels."
        ),
        "candidates": candidates,
        "source": "dexscreener",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Read-only token facts via the public DexScreener API (no key)."
    )
    parser.add_argument("address", nargs="?", help="token contract / mint address")
    parser.add_argument(
        "--search", metavar="NAME", help="find tokens by name/symbol (brand-squat check)"
    )
    args = parser.parse_args()
    try:
        if args.search:
            result = search_name(args.search)
        elif args.address:
            result = token_facts(args.address)
        else:
            result = {"error": "provide a token address, or --search <name>"}
    except urllib.error.URLError as exc:
        result = {"error": "network error: %s" % exc}
    except (ValueError, OSError) as exc:
        result = {"error": "request failed: %s" % exc}
    except Exception as exc:  # never crash the agent's tool call
        result = {"error": "unexpected: %s" % exc}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
