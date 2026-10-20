#!/usr/bin/env python3
"""
Bitcoin read-only client for Hermes.
Query addresses, transactions, blocks, mempool, fees, network stats, price,
and large unconfirmed transactions (whale watch).
Standard library only (urllib, json, argparse). No API key required.
"""
import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

MEMPOOL_API = "https://mempool.space/api"
FALLBACK_APIS = [
    "https://mempool.space/api",
    "https://blockstream.info/api",
]
COINGECKO_API = "https://api.coingecko.com/api/v3"

# Simple in-process price cache to reduce CoinGecko rate-limit hits.
_PRICE_CACHE = {"data": None, "currency": None, "ts": 0.0}
_PRICE_CACHE_TTL = 60.0  # seconds


def fetch_json_with_fallback(path_or_url, timeout=30, retries=2):
    """Fetch JSON from the primary mempool.space API, falling back to other
    public explorers if the primary returns 5xx or network errors. The argument
    can be a full URL or a path starting with /.
    """
    if path_or_url.startswith("http"):
        candidates = [path_or_url]
    else:
        candidates = [base + path_or_url for base in FALLBACK_APIS]

    last_error = None
    for url in candidates:
        req = urllib.request.Request(url, headers={
            "User-Agent": "HermesBitcoinSkill/1.0",
            "Accept": "application/json",
        })
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")[:200]
                last_error = RuntimeError(f"HTTP {e.code}: {body}")
                if e.code == 429 and attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                # For 4xx client errors, do not retry the same URL but try next candidate.
                if 400 <= e.code < 500 and e.code != 429:
                    break
            except urllib.error.URLError as e:
                last_error = RuntimeError(f"Network error: {e.reason}")
                break
    raise last_error


def fetch_json(url, timeout=30, retries=2):
    """Fetch JSON from url with retries on 429, raise on failure."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "HermesBitcoinSkill/1.0",
        "Accept": "application/json",
    })
    last_error = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")[:200]
            last_error = RuntimeError(f"HTTP {e.code}: {body}")
            if e.code == 429 and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise last_error from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error: {e.reason}") from e
    raise last_error


def fetch_text(url, timeout=30, retries=2):
    """Fetch plain text from url with retries on 429."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "HermesBitcoinSkill/1.0",
        "Accept": "text/plain",
    })
    last_error = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8").strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")[:200]
            last_error = RuntimeError(f"HTTP {e.code}: {body}")
            if e.code == 429 and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise last_error from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error: {e.reason}") from e
    raise last_error


def get_btc_price(currency="usd", use_cache=True):
    """Fetch BTC price from CoinGecko with short-lived in-process cache."""
    if not use_cache:
        return {}
    currency = currency.lower()
    now = time.time()
    if use_cache and _PRICE_CACHE["data"] is not None and _PRICE_CACHE["currency"] == currency and (now - _PRICE_CACHE["ts"]) < _PRICE_CACHE_TTL:
        return _PRICE_CACHE["data"]
    url = f"{COINGECKO_API}/simple/price?ids=bitcoin&vs_currencies={currency}&include_24hr_change=true"
    data = fetch_json(url).get("bitcoin", {})
    _PRICE_CACHE["data"] = data
    _PRICE_CACHE["currency"] = currency
    _PRICE_CACHE["ts"] = now
    return data


def fmt_btc(sats):
    return f"{sats / 1e8:.8f} BTC"


def fmt_fiat(value, price_btc=None, currency="usd"):
    if price_btc is None:
        return None
    currency = currency.lower()
    amount = value * price_btc / 1e8
    if currency == "usd":
        return f"${amount:,.2f}"
    if currency == "eur":
        return f"€{amount:,.2f}"
    return f"{currency.upper()} {amount:,.2f}"


def fmt_time_unix(ts):
    try:
        dt = datetime.utcfromtimestamp(ts).replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(ts)


def fmt_time(iso_str_or_ts):
    if iso_str_or_ts is None:
        return None
    if isinstance(iso_str_or_ts, (int, float)):
        return fmt_time_unix(iso_str_or_ts)
    try:
        dt = datetime.fromisoformat(iso_str_or_ts.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return iso_str_or_ts


def format_hashrate(hs):
    """Format hashrate in H/s to human readable SI unit."""
    if hs is None or hs == 0:
        return None
    units = ["H/s", "kH/s", "MH/s", "GH/s", "TH/s", "PH/s", "EH/s", "ZH/s"]
    exp = min(int(math.log10(hs) / 3), len(units) - 1)
    val = hs / (10 ** (exp * 3))
    return f"{val:.3f} {units[exp]}"


def block_subsidy(height):
    """Calculate coinbase subsidy (sats) for a given block height."""
    halvings = height // 210_000
    if halvings >= 64:
        return 0
    return int(50 * 1e8) >> halvings


def net_value_for_address(tx, address):
    """Net value (sats) moving to/from address in this transaction."""
    received = 0
    spent = 0
    for inp in tx.get("vin", []):
        prevout = inp.get("prevout") or {}
        if prevout.get("scriptpubkey_address") == address:
            spent += prevout.get("value", 0)
    for out in tx.get("vout", []):
        if out.get("scriptpubkey_address") == address:
            received += out.get("value", 0)
    return received - spent


def fee_rate_for_tx(tx):
    fee = tx.get("fee", 0)
    vsize = tx.get("vsize")
    if not vsize and tx.get("weight"):
        vsize = math.ceil(tx.get("weight") / 4)
    if not vsize:
        return None
    return round(fee / vsize, 2)


def cmd_address(args):
    addr = args.address
    data = fetch_json_with_fallback(f"/address/{addr}")
    chain_stats = data.get("chain_stats", {})
    mempool_stats = data.get("mempool_stats", {})
    price = get_btc_price(args.currency, use_cache=not args.no_fiat)
    price_btc = price.get(args.currency)

    funded = chain_stats.get("funded_txo_sum", 0)
    spent = chain_stats.get("spent_txo_sum", 0)
    balance = funded - spent
    mempool_funded = mempool_stats.get("funded_txo_sum", 0)
    mempool_spent = mempool_stats.get("spent_txo_sum", 0)
    unconfirmed_balance = mempool_funded - mempool_spent

    result = {
        "address": addr,
        "balance_sats": balance,
        "balance_btc": balance / 1e8,
        "balance_fiat": fmt_fiat(balance, price_btc, args.currency) if not args.no_fiat else None,
        "unconfirmed_balance_sats": unconfirmed_balance,
        "unconfirmed_balance_btc": unconfirmed_balance / 1e8,
        "tx_count_confirmed": chain_stats.get("tx_count", 0),
        "tx_count_unconfirmed": mempool_stats.get("tx_count", 0),
        "fiat_currency": args.currency.upper() if not args.no_fiat else None,
        "btc_price": price_btc if not args.no_fiat else None,
        "btc_24h_change_percent": price.get(f"{args.currency}_24h_change") if not args.no_fiat else None,
    }
    print(json.dumps(result, indent=2))


def cmd_txs(args):
    """List recent confirmed transactions for an address."""
    addr = args.address
    limit = max(1, args.limit)
    txs = fetch_json_with_fallback(f"/address/{addr}/txs")
    price = get_btc_price(args.currency, use_cache=not args.no_fiat)
    price_btc = price.get(args.currency)

    results = []
    for tx in txs[:limit]:
        status = tx.get("status") or {}
        net_value = net_value_for_address(tx, addr)
        results.append({
            "txid": tx.get("txid"),
            "confirmed": status.get("confirmed", False),
            "block_height": status.get("block_height"),
            "block_time": fmt_time(status.get("block_time")),
            "fee_sats": tx.get("fee"),
            "fee_rate_sat_vb": fee_rate_for_tx(tx),
            "net_value_sats": net_value,
            "net_value_btc": net_value / 1e8,
            "net_value_fiat": fmt_fiat(net_value, price_btc, args.currency) if not args.no_fiat else None,
            "is_coinbase": any(i.get("is_coinbase") for i in tx.get("vin", [])),
        })

    if getattr(args, "format", "json").lower() in ("csv", "tsv"):
        print(format_csv(results, args.format, args.currency, price_btc))
        return

    output = {
        "address": addr,
        "count": len(results),
        "fiat_currency": args.currency.upper() if not args.no_fiat else None,
        "btc_price": price_btc if not args.no_fiat else None,
        "btc_24h_change_percent": price.get(f"{args.currency}_24h_change") if not args.no_fiat else None,
        "transactions": results,
        "note": "mempool.space returns the latest 50 confirmed transactions by default. Use pagination or a block explorer for deeper history.",
    }
    print(json.dumps(output, indent=2))


def cmd_utxo(args):
    """List spendable UTXOs for an address."""
    addr = args.address
    utxos = fetch_json_with_fallback(f"/address/{addr}/utxo")
    price = get_btc_price(args.currency, use_cache=not args.no_fiat)
    price_btc = price.get(args.currency)

    results = []
    total = 0
    for u in utxos:
        value = u.get("value", 0)
        status = u.get("status") or {}
        total += value
        results.append({
            "txid": u.get("txid"),
            "vout": u.get("vout"),
            "value_sats": value,
            "value_btc": value / 1e8,
            "value_fiat": fmt_fiat(value, price_btc, args.currency) if not args.no_fiat else None,
            "confirmed": status.get("confirmed", False),
            "block_height": status.get("block_height"),
            "block_time": fmt_time(status.get("block_time")),
        })

    if getattr(args, "format", "json").lower() in ("csv", "tsv"):
        print(format_csv(results, args.format, args.currency, price_btc))
        return

    output = {
        "address": addr,
        "utxo_count": len(results),
        "total_value_sats": total,
        "total_value_btc": total / 1e8,
        "total_value_fiat": fmt_fiat(total, price_btc, args.currency) if not args.no_fiat else None,
        "fiat_currency": args.currency.upper() if not args.no_fiat else None,
        "btc_price": price_btc if not args.no_fiat else None,
        "utxos": results,
    }
    print(json.dumps(output, indent=2))


def cmd_report(args):
    """Generate a combined report for an address or transaction."""
    query = args.query
    is_txid = len(query) == 64 and all(c in "0123456789abcdefABCDEF" for c in query)
    price = get_btc_price(args.currency, use_cache=not args.no_fiat)
    price_btc = price.get(args.currency)
    result = {
        "query": query,
        "type": "transaction" if is_txid else "address",
        "btc_price": price_btc if not args.no_fiat else None,
        "fiat_currency": args.currency.upper() if not args.no_fiat else None,
    }
    if is_txid:
        tx_data = fetch_json_with_fallback(f"/tx/{query}")
        status = tx_data.get("status") or {}
        inputs = tx_data.get("vin", [])
        outputs = tx_data.get("vout", [])
        input_total = sum((i.get("prevout") or {}).get("value", 0) for i in inputs)
        output_total = sum(o.get("value", 0) for o in outputs)
        is_coinbase = any(i.get("is_coinbase") for i in inputs)
        fee = 0 if is_coinbase else input_total - output_total
        vsize = tx_data.get("vsize")
        if not vsize and tx_data.get("weight"):
            vsize = math.ceil(tx_data.get("weight") / 4)
        if not vsize:
            vsize = 1
        result["transaction"] = {
            "txid": query,
            "confirmed": status.get("confirmed", False),
            "block_height": status.get("block_height"),
            "block_time": fmt_time(status.get("block_time")),
            "fee_sats": fee,
            "fee_rate_sat_vb": round(fee / vsize, 2),
            "input_count": len(inputs),
            "output_count": len(outputs),
            "input_total_sats": input_total,
            "output_total_sats": output_total,
            "is_coinbase": is_coinbase,
        }
    else:
        addr_data = fetch_json_with_fallback(f"/address/{query}")
        chain_stats = addr_data.get("chain_stats", {})
        mempool_stats = addr_data.get("mempool_stats", {})
        funded = chain_stats.get("funded_txo_sum", 0)
        spent = chain_stats.get("spent_txo_sum", 0)
        balance = funded - spent
        mempool_funded = mempool_stats.get("funded_txo_sum", 0)
        mempool_spent = mempool_stats.get("spent_txo_sum", 0)
        unconfirmed_balance = mempool_funded - mempool_spent
        txs = fetch_json_with_fallback(f"/address/{query}/txs")[:5]
        recent = []
        for tx in txs:
            status = tx.get("status") or {}
            recent.append({
                "txid": tx.get("txid"),
                "block_height": status.get("block_height"),
                "block_time": fmt_time(status.get("block_time")),
                "net_value_sats": net_value_for_address(tx, query),
                "fee_rate_sat_vb": fee_rate_for_tx(tx),
            })
        result["address"] = {
            "balance_sats": balance,
            "balance_btc": balance / 1e8,
            "balance_fiat": fmt_fiat(balance, price_btc, args.currency) if not args.no_fiat else None,
            "unconfirmed_balance_sats": unconfirmed_balance,
            "tx_count_confirmed": chain_stats.get("tx_count", 0),
            "recent_transactions": recent,
        }
    print(json.dumps(result, indent=2))


def cmd_compare(args):
    """Compare balance and activity across multiple addresses."""
    price = get_btc_price(args.currency, use_cache=not args.no_fiat)
    price_btc = price.get(args.currency)
    addresses = args.addresses
    results = []
    total_balance = 0
    total_tx_count = 0
    for addr in addresses:
        data = fetch_json_with_fallback(f"/address/{addr}")
        chain_stats = data.get("chain_stats", {})
        mempool_stats = data.get("mempool_stats", {})
        balance = chain_stats.get("funded_txo_sum", 0) - chain_stats.get("spent_txo_sum", 0)
        total_balance += balance
        tx_count = chain_stats.get("tx_count", 0)
        total_tx_count += tx_count
        results.append({
            "address": addr,
            "balance_sats": balance,
            "balance_btc": balance / 1e8,
            "balance_fiat": fmt_fiat(balance, price_btc, args.currency) if not args.no_fiat else None,
            "tx_count": tx_count,
            "unconfirmed_tx_count": mempool_stats.get("tx_count", 0),
        })

    if getattr(args, "format", "json").lower() in ("csv", "tsv"):
        print(format_csv(results, args.format, args.currency, price_btc))
        return

    output = {
        "addresses_compared": len(addresses),
        "total_balance_sats": total_balance,
        "total_balance_btc": total_balance / 1e8,
        "total_balance_fiat": fmt_fiat(total_balance, price_btc, args.currency) if not args.no_fiat else None,
        "total_tx_count": total_tx_count,
        "fiat_currency": args.currency.upper() if not args.no_fiat else None,
        "btc_price": price_btc if not args.no_fiat else None,
        "results": results,
    }
    print(json.dumps(output, indent=2))


def cmd_fee_history(args):
    """Historical average fee rates by block."""
    period = args.period
    if period not in ("1w", "1m", "3m", "6m", "1y"):
        raise RuntimeError("Period must be one of: 1w, 1m, 3m, 6m, 1y")
    data = fetch_json(f"{MEMPOOL_API}/v1/mining/blocks/fee-rates/{period}")
    rows = []
    for row in data:
        rows.append({
            "avg_height": row.get("avgHeight"),
            "time": fmt_time(row.get("timestamp")),
            "min_sat_vb": row.get("avgFee_0"),
            "p10_sat_vb": row.get("avgFee_10"),
            "median_sat_vb": row.get("avgFee_50"),
            "p90_sat_vb": row.get("avgFee_90"),
            "max_sat_vb": row.get("avgFee_100"),
        })

    if getattr(args, "format", "json").lower() in ("csv", "tsv"):
        print(format_csv(rows, args.format))
        return

    summary = {
        "period": period,
        "blocks": len(rows),
        "median_min": round(sum(r["min_sat_vb"] for r in rows) / len(rows), 2) if rows else None,
        "median_p50": round(sum(r["median_sat_vb"] for r in rows) / len(rows), 2) if rows else None,
        "median_p90": round(sum(r["p90_sat_vb"] for r in rows) / len(rows), 2) if rows else None,
        "max_observed": max((r["max_sat_vb"] for r in rows), default=None),
        "fee_history_sample": rows[-7:],
    }
    print(json.dumps(summary, indent=2))


def cmd_tx(args):
    txid = args.txid
    data = fetch_json_with_fallback(f"/tx/{txid}")
    status = data.get("status") or {}
    price = get_btc_price(args.currency, use_cache=not args.no_fiat)
    price_btc = price.get(args.currency)

    inputs = data.get("vin", [])
    outputs = data.get("vout", [])
    input_total = sum((i.get("prevout") or {}).get("value", 0) for i in inputs)
    output_total = sum(o.get("value", 0) for o in outputs)
    is_coinbase = any(i.get("is_coinbase") for i in inputs)
    fee = 0 if is_coinbase else input_total - output_total
    vsize = data.get("vsize")
    if not vsize and data.get("weight"):
        vsize = math.ceil(data.get("weight") / 4)
    if not vsize:
        vsize = 1

    result = {
        "txid": txid,
        "confirmed": status.get("confirmed", False),
        "block_height": status.get("block_height"),
        "block_hash": status.get("block_hash"),
        "block_time": fmt_time(status.get("block_time")),
        "fee_sats": fee,
        "fee_btc": fee / 1e8,
        "fee_fiat": fmt_fiat(fee, price_btc, args.currency) if not args.no_fiat else None,
        "fee_rate_sat_vb": round(fee / vsize, 2) if vsize else None,
        "size": data.get("size"),
        "vsize": vsize,
        "version": data.get("version"),
        "locktime": data.get("locktime"),
        "input_count": len(inputs),
        "output_count": len(outputs),
        "input_total_sats": input_total,
        "output_total_sats": output_total,
        "is_coinbase": is_coinbase,
        "fiat_currency": args.currency.upper() if not args.no_fiat else None,
        "btc_price": price_btc if not args.no_fiat else None,
    }
    if args.verbose:
        result["inputs"] = inputs
        result["outputs"] = outputs
    print(json.dumps(result, indent=2))


def cmd_block(args):
    query = args.block
    if query.isdigit():
        block_hash = fetch_text(f"{MEMPOOL_API}/block-height/{query}")
        height = int(query)
    else:
        block_hash = query
        status = fetch_json(f"{MEMPOOL_API}/block/{block_hash}/status")
        height = status.get("height")
    data = fetch_json(f"{MEMPOOL_API}/block/{block_hash}")

    subsidy = block_subsidy(height)
    result = {
        "id": data.get("id"),
        "height": height,
        "version": data.get("version"),
        "timestamp": fmt_time(data.get("timestamp")),
        "tx_count": data.get("tx_count"),
        "size": data.get("size"),
        "weight": data.get("weight"),
        "merkle_root": data.get("merkle_root"),
        "previousblockhash": data.get("previousblockhash"),
        "mediantime": fmt_time(data.get("mediantime")),
        "nonce": data.get("nonce"),
        "bits": data.get("bits"),
        "difficulty": data.get("difficulty"),
        "subsidy_sats": subsidy,
        "subsidy_btc": subsidy / 1e8,
        "fees_sats": None,
        "fees_btc": None,
        "reward_sats": None,
        "reward_btc": None,
        "note": "Reward and fees are no longer returned by mempool.space /block endpoint. Subsidy is calculated from height.",
    }
    print(json.dumps(result, indent=2))


def cmd_mempool(args):
    url = f"{MEMPOOL_API}/mempool"
    data = fetch_json(url)
    result = {
        "count": data.get("count"),
        "vsize": data.get("vsize"),
        "vsize_mb": round(data.get("vsize", 0) / 1e6, 3) if data.get("vsize") else None,
        "total_fee_sats": data.get("total_fee"),
        "total_fee_btc": data.get("total_fee", 0) / 1e8,
        "fee_histogram": data.get("fee_histogram", [])[:10],
    }
    print(json.dumps(result, indent=2))


def cmd_fees(args):
    url = f"{MEMPOOL_API}/v1/fees/recommended"
    data = fetch_json(url)
    print(json.dumps(data, indent=2))


def cmd_stats(args):
    tip_height = int(fetch_text(f"{MEMPOOL_API}/blocks/tip/height"))
    tip_hash = fetch_text(f"{MEMPOOL_API}/block-height/{tip_height}")
    block = fetch_json(f"{MEMPOOL_API}/block/{tip_hash}")
    hashrate_data = fetch_json(f"{MEMPOOL_API}/v1/mining/hashrate/1m")
    diff_adj = fetch_json(f"{MEMPOOL_API}/v1/mining/difficulty-adjustments/1m")
    price = get_btc_price(args.currency, use_cache=not args.no_fiat)
    price_btc = price.get(args.currency)

    current_hashrate = hashrate_data.get("currentHashrate")
    avg_7d = None
    hashrates = hashrate_data.get("hashrates", [])
    if hashrates:
        window = hashrates[-7:]
        avg_7d = sum(h.get("avgHashrate", 0) for h in window) / len(window)

    next_retarget = None
    if diff_adj and isinstance(diff_adj, list) and len(diff_adj) > 0:
        # mempool.space returns difficulty-adjustment rows newest-first, but
        # defensively select the row with the maximum timestamp to avoid
        # depending on ordering.
        latest = max(
            (row for row in diff_adj if isinstance(row, list) and len(row) >= 4),
            key=lambda row: row[0] if isinstance(row[0], (int, float)) else 0,
            default=None,
        )
        if latest is not None:
            next_retarget = {
                "estimated_change_percent": latest[3],
                "last_adjustment_height": latest[1],
                "last_adjustment_timestamp": fmt_time(latest[0]),
                "current_difficulty": hashrate_data.get("currentDifficulty") or block.get("difficulty"),
            }

    result = {
        "tip_height": tip_height,
        "tip_hash": tip_hash,
        "tip_time": fmt_time(block.get("timestamp")),
        "difficulty": block.get("difficulty"),
        "current_hashrate_hs": current_hashrate,
        "current_hashrate_formatted": format_hashrate(current_hashrate),
        "avg_hashrate_7d_hs": avg_7d,
        "avg_hashrate_7d_formatted": format_hashrate(avg_7d),
        "next_retarget": next_retarget,
        "btc_price": price_btc if not args.no_fiat else None,
        "fiat_currency": args.currency.upper() if not args.no_fiat else None,
        "btc_24h_change_percent": price.get(f"{args.currency}_24h_change") if not args.no_fiat else None,
    }
    print(json.dumps(result, indent=2))


def cmd_price(args):
    currencies = args.currencies.split(",")
    vs = ",".join(currencies)
    url = f"{COINGECKO_API}/simple/price?ids=bitcoin&vs_currencies={vs}&include_24hr_change=true&include_market_cap=true&include_24hr_vol=true"
    data = fetch_json(url)
    print(json.dumps(data.get("bitcoin", {}), indent=2))


def cmd_whale(args):
    """Report large unconfirmed transactions from the mempool."""
    threshold_btc = args.threshold
    threshold_sats = int(threshold_btc * 1e8)
    data = fetch_json(f"{MEMPOOL_API}/mempool/recent")
    price = get_btc_price(args.currency, use_cache=not args.no_fiat)
    price_btc = price.get(args.currency)

    whales = [t for t in data if t.get("value", 0) >= threshold_sats]
    whales.sort(key=lambda x: x.get("value", 0), reverse=True)

    results = []
    for w in whales[:args.limit]:
        value = w.get("value", 0)
        fee = w.get("fee", 0)
        vsize = w.get("vsize", 1)
        results.append({
            "txid": w.get("txid"),
            "value_sats": value,
            "value_btc": value / 1e8,
            "value_fiat": fmt_fiat(value, price_btc, args.currency) if not args.no_fiat else None,
            "fee_sats": fee,
            "fee_rate_sat_vb": round(fee / vsize, 2) if vsize else None,
            "vsize": vsize,
        })

    output = {
        "threshold_btc": threshold_btc,
        "count": len(results),
        "fiat_currency": args.currency.upper() if (price_btc and not args.no_fiat) else None,
        "btc_price": price_btc if not args.no_fiat else None,
        "transactions": results,
        "note": "Whale watch scans the last ~10 recently-arrived mempool transactions. It is not exhaustive.",
    }
    print(json.dumps(output, indent=2))


def cmd_verify(args):
    query = args.query
    if len(query) == 64 and all(c in "0123456789abcdefABCDEF" for c in query):
        try:
            fetch_json_with_fallback(f"/tx/{query}")
            print(json.dumps({"type": "transaction", "identifier": query, "valid": True}, indent=2))
            return
        except RuntimeError as e:
            print(json.dumps({"type": "transaction", "identifier": query, "valid": False, "error": str(e)}, indent=2))
            return
    elif (query.startswith("1") or query.startswith("3") or query.startswith("bc1")) and 25 <= len(query) <= 90:
        try:
            fetch_json_with_fallback(f"/address/{query}")
            print(json.dumps({"type": "address", "identifier": query, "valid": True}, indent=2))
            return
        except RuntimeError as e:
            print(json.dumps({"type": "address", "identifier": query, "valid": False, "error": str(e)}, indent=2))
            return
    else:
        print(json.dumps({"type": "unknown", "identifier": query, "valid": False, "error": "Not a recognized Bitcoin txid or address format"}, indent=2))


CHECKLISTS = {
    "general": [
        "Cite mempool.space and CoinGecko as primary sources.",
        "Distinguish confirmed data from unconfirmed mempool data.",
        "Do not infer ownership of an address without public attribution.",
        "Avoid price predictions and trading advice.",
        "Cross-check high-stakes claims against a second block explorer.",
    ],
    "address": [
        "Confirm the address exists on-chain before reporting its balance.",
        "Report confirmed balance separately from unconfirmed balance.",
        "Do not assume the address owner based on balance alone.",
        "Include the timestamp or block height of the latest activity if relevant.",
        "Note that privacy tools (CoinJoin, mixers) can obscure address meaning.",
    ],
    "txs": [
        "Verify every txid independently before using it in a story.",
        "Report net value relative to the address, not just absolute transaction size.",
        "Distinguish incoming (positive net value) from outgoing (negative net value).",
        "Do not assume a transaction proves payment to a specific entity.",
    ],
    "utxo": [
        "Confirm each UTXO exists and is confirmed before reporting it as spendable.",
        "Remember that UTXO value alone does not reveal address ownership.",
        "Large numbers of small UTXOs may indicate dust or exchange wallet behavior.",
    ],
    "compare": [
        "Do not imply ownership links between addresses based only on balance patterns.",
        "Report aggregate totals with the correct fiat conversion and timestamp.",
        "Note that the same entity may control multiple addresses, but this skill does not prove it.",
    ],
    "report": [
        "Verify the query type (address vs transaction) before interpreting the report.",
        "Cross-check any headline numbers with the raw txid or address in a block explorer.",
        "Include the timestamp of the report because balances and mempool data change quickly.",
    ],
    "tx": [
        "Verify the txid exists and is confirmed before describing it as final.",
        "Report block height, timestamp, and fee rate.",
        "Distinguish coinbase transactions from normal transactions.",
        "Do not infer the purpose of a transaction from inputs/outputs alone.",
        "For large transfers, verify the receiving address is not an exchange hot wallet unless proven.",
    ],
    "fees": [
        "Report the exact fee rate tier (fastest/halfHour/hour/economy/minimum).",
        "Note that mempool congestion changes quickly.",
        "Compare current fees with a recent average to give context.",
        "Mention whether data is from mempool.space recommended fees endpoint.",
    ],
    "mining": [
        "Report difficulty, hashrate, and retarget estimate with units (EH/s, %).",
        "Use 7-day average hashrate rather than instantaneous current hashrate for trend context.",
        "Calculate subsidy correctly from block height (halvings every 210,000 blocks).",
        "Do not confuse subsidy with total block reward (subsidy + fees).",
    ],
    "price": [
        "Quote the fiat currency and the timestamp of the price check.",
        "Include 24h change and note the source (CoinGecko).",
        "Avoid implying causation between on-chain events and short-term price moves.",
    ],
}


def cmd_editorial(args):
    topic = args.topic
    items = CHECKLISTS.get(topic, CHECKLISTS["general"])
    print(json.dumps({"topic": topic, "checklist": items}, indent=2))


def format_csv(rows, fmt, currency="usd", price_btc=None):
    """Convert list of dicts to CSV or TSV."""
    if not rows:
        return ""
    delimiter = "\t" if fmt.lower() == "tsv" else ","
    headers = list(rows[0].keys())
    lines = [delimiter.join(headers)]
    for row in rows:
        values = []
        for h in headers:
            v = row.get(h)
            if v is None:
                values.append("")
            elif isinstance(v, (list, dict)):
                values.append(json.dumps(v))
            else:
                s = str(v)
                if delimiter == "," and (("," in s) or ("\n" in s) or ('"' in s)):
                    s = '"' + s.replace('"', '""') + '"'
                values.append(s)
        lines.append(delimiter.join(values))
    return "\n".join(lines)


def _add_format_arg(parser):
    parser.add_argument("--format", default="json", choices=["json", "csv", "tsv"], help="Output format (default: json)")


def _add_fiat_args(parser):
    parser.add_argument("--currency", default="usd", help="Fiat currency for price conversion (default: usd)")
    parser.add_argument("--no-fiat", action="store_true", help="Skip fiat price conversion (avoids CoinGecko call)")


def main():
    parser = argparse.ArgumentParser(
        prog="bitcoin_client.py",
        description="Read-only Bitcoin research client for Hermes",
    )
    parser.add_argument("--currency", default="usd", help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)

    p_addr = sub.add_parser("address", help="Address balance and stats")
    p_addr.add_argument("address")
    _add_fiat_args(p_addr)
    p_addr.set_defaults(func=cmd_address)

    p_txs = sub.add_parser("txs", help="Recent transactions for an address")
    p_txs.add_argument("address")
    p_txs.add_argument("--limit", type=int, default=10, help="Number of transactions to return (default: 10)")
    _add_fiat_args(p_txs)
    _add_format_arg(p_txs)
    p_txs.set_defaults(func=cmd_txs)

    p_utxo = sub.add_parser("utxo", help="Spendable UTXOs for an address")
    p_utxo.add_argument("address")
    _add_fiat_args(p_utxo)
    _add_format_arg(p_utxo)
    p_utxo.set_defaults(func=cmd_utxo)

    p_report = sub.add_parser("report", help="Combined report for an address or txid")
    p_report.add_argument("query")
    _add_fiat_args(p_report)
    p_report.set_defaults(func=cmd_report)

    p_compare = sub.add_parser("compare", help="Compare multiple addresses")
    p_compare.add_argument("addresses", nargs="+")
    _add_fiat_args(p_compare)
    _add_format_arg(p_compare)
    p_compare.set_defaults(func=cmd_compare)

    p_feehist = sub.add_parser("fee-history", help="Historical average fee rates")
    p_feehist.add_argument("--period", default="1m", choices=["1w", "1m", "3m", "6m", "1y"], help="Period (default: 1m)")
    _add_format_arg(p_feehist)
    p_feehist.set_defaults(func=cmd_fee_history)

    p_tx = sub.add_parser("tx", help="Transaction details")
    p_tx.add_argument("txid")
    p_tx.add_argument("--verbose", action="store_true", help="Include full inputs/outputs")
    _add_fiat_args(p_tx)
    p_tx.set_defaults(func=cmd_tx)

    p_block = sub.add_parser("block", help="Block by height or hash")
    p_block.add_argument("block")
    p_block.set_defaults(func=cmd_block)

    p_mempool = sub.add_parser("mempool", help="Mempool summary")
    p_mempool.set_defaults(func=cmd_mempool)

    p_fees = sub.add_parser("fees", help="Recommended fee rates")
    p_fees.set_defaults(func=cmd_fees)

    p_stats = sub.add_parser("stats", help="Network stats and BTC price")
    _add_fiat_args(p_stats)
    p_stats.set_defaults(func=cmd_stats)

    p_price = sub.add_parser("price", help="BTC price in one or more currencies")
    p_price.add_argument("--currencies", default="usd,eur", help="Comma-separated currencies")
    p_price.set_defaults(func=cmd_price)

    p_whale = sub.add_parser("whale", help="Large unconfirmed transactions in the mempool")
    p_whale.add_argument("--threshold", type=float, default=1.0, help="Minimum transaction value in BTC (default: 1.0)")
    p_whale.add_argument("--limit", type=int, default=10, help="Maximum results to return (default: 10)")
    _add_fiat_args(p_whale)
    p_whale.set_defaults(func=cmd_whale)

    p_verify = sub.add_parser("verify", help="Verify existence of address or txid")
    p_verify.add_argument("query")
    p_verify.set_defaults(func=cmd_verify)

    p_editorial = sub.add_parser("editorial", help="Print editorial fact-check checklist")
    p_editorial.add_argument("topic", nargs="?", default="general", choices=["general", "address", "txs", "tx", "utxo", "fees", "mining", "price", "compare", "report"], help="Tailor checklist to topic")
    p_editorial.set_defaults(func=cmd_editorial)

    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
