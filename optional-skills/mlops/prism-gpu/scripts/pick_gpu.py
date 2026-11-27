#!/usr/bin/env python3
"""Choose a GPU from Prism's live offer list against a VRAM floor and a price ceiling.

Usage:
    pick_gpu.py [--min-vram-gb N] [--max-usdg-per-hour F] [--duration N]
                [--min-trust open|isolated|attested|confidential]
                [--include-staker-only] [--offers FILE] [--json]

Reads capacity, ranks it, and prints the choice with the reason it won and the
deposit the window would cost. Offers come from the public HTTP endpoint, so
this script reads them without a wallet and reserves nothing. Stdlib only.

Nodes publish VRAM in MiB and report a little under the number on the card: a
24 GB card publishes 24564 MiB, a 32 GB card 32607 MiB. ``--min-vram-gb`` takes
the label and converts at 1000 MiB per GB, which keeps that headroom, matches
the SDK's own ``min_vram_mib`` default of 16000, and means asking for 24 GB
selects a 24 GB card instead of rejecting it. Every figure printed is MiB.

Exit codes, so a caller can branch without parsing the output:

    0   an offer matched; the choice is on stdout
    1   capacity is online but nothing met the constraints
    2   wrong arguments
    3   capacity could not be read
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

OFFERS_URL = "https://api.prismnetwork.tech/v1/offers"
MICROS = 1_000_000
# Deliberately decimal. See the module docstring: nodes report under nominal.
MIB_PER_GB = 1000

# Ascending. An offer satisfies a floor when its own class sits at or above it.
TRUST_CLASSES = ("open", "isolated", "attested", "confidential")

MATCHED = 0
NO_MATCH = 1
UNREACHABLE = 3


def fetch(url, timeout=15):
    request = urllib.request.Request(url, headers={"accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def load(path, timeout=15):
    """Every published offer. The endpoint can filter by trust class, but then a
    constraint that matched nothing comes back as an empty list and the report
    cannot say what each machine failed on. Filtering happens here instead."""
    if path:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    return fetch(OFFERS_URL, timeout)


def rate_micros(offer):
    """Base units per second, or None when the offer publishes no usable rate.
    Zero would sort to the front of a cheapest-first ranking."""
    try:
        rate = int(offer.get("rate_per_second"))
    except (TypeError, ValueError):
        return None
    return rate if rate > 0 else None


def rate_per_hour(offer):
    return (rate_micros(offer) or 0) * 3600 / MICROS


def vram_mib(offer):
    gpu = offer.get("gpu")
    try:
        return int(gpu.get("vram_mib", 0)) if isinstance(gpu, dict) else 0
    except (TypeError, ValueError):
        return 0


def floor_mib(min_vram_gb):
    return int(min_vram_gb * MIB_PER_GB)


def rejections(offer, min_vram_mib, max_micros_per_second, min_trust, include_staker_only):
    """Every reason at once, so the report can name what each machine failed on."""
    reasons = []
    if not offer.get("node_id"):
        reasons.append("no node id to fund")
    if not offer.get("online", True):
        reasons.append("offline")
    have = vram_mib(offer)
    if have < min_vram_mib:
        reasons.append(f"{have} MiB VRAM under the {min_vram_mib} MiB floor")
    rate = rate_micros(offer)
    if rate is None:
        reasons.append("no published rate")
    elif max_micros_per_second is not None and rate > max_micros_per_second:
        reasons.append(f"{rate * 3600 / MICROS:.4f} USDG/hr over the ceiling")
    offered = offer.get("trust_class", "open")
    if offered not in TRUST_CLASSES:
        reasons.append(f"unrecognised trust class {offered!r}")
    elif TRUST_CLASSES.index(offered) < TRUST_CLASSES.index(min_trust):
        reasons.append(f"trust class {offered} below {min_trust}")
    if offer.get("staker_only") and not include_staker_only:
        reasons.append("reserved for stakers")
    return reasons


def score(offer):
    try:
        return int(offer.get("benchmark_score", 0))
    except (TypeError, ValueError):
        return 0


def rank(offer):
    return (
        rate_micros(offer) or 0,
        -score(offer),
        -vram_mib(offer),
    )


def choose(offers, min_vram_mib, max_micros_per_second, min_trust, include_staker_only):
    eligible, rejected = [], []
    for offer in offers:
        if not isinstance(offer, dict):
            continue
        reasons = rejections(offer, min_vram_mib, max_micros_per_second, min_trust, include_staker_only)
        (rejected if reasons else eligible).append((offer, reasons))
    eligible.sort(key=lambda pair: rank(pair[0]))
    return [offer for offer, _ in eligible], rejected


def describe(offer):
    gpu = offer.get("gpu") if isinstance(offer.get("gpu"), dict) else {}
    return (f"{gpu.get('model', 'GPU')} · {vram_mib(offer)} MiB · "
            f"{rate_per_hour(offer):.4f} USDG/hr · trust {offer.get('trust_class', 'open')}")


def why(winner, runners_up):
    if not runners_up:
        return "the only offer that met every constraint"
    matched = len(runners_up) + 1
    next_up = runners_up[0]
    cheaper = rate_per_hour(next_up) - rate_per_hour(winner)
    if cheaper > 0:
        return f"cheapest of {matched} matching offers, {cheaper:.4f} USDG/hr under the next one"
    if score(winner) > score(next_up):
        return f"tied on price across {matched} offers, highest benchmark score"
    if vram_mib(winner) > vram_mib(next_up):
        return f"tied on price and score across {matched} offers, largest card"
    # Only offers whose whole rank tuple matches are tied. Counting the matching
    # field instead once reported cards of two different sizes as identical.
    tied = sum(1 for offer in runners_up if rank(offer) == rank(winner))
    return (f"tied with {tied} other offer(s) of {matched} on price, score and card size; "
            "taken in list order")


def report(winner, runners_up, rejected, duration):
    rate = rate_micros(winner)
    deposit = rate * duration
    lines = [
        f"chosen       {describe(winner)}",
        f"node         {winner.get('node_id')}",
        f"why          {why(winner, runners_up)}",
        f"deposit      {deposit / MICROS:.6f} USDG for a {duration}s window "
        f"({rate} base units per second)",
        "             The whole window is deposited up front and the release is what",
        "             stops the meter.",
    ]
    if winner.get("staker_only"):
        lines.append("caution      this node only rents to bonded stakers, so funding it is "
                     "refused unless this wallet is one")
    lines.append(f"next         lease_run_release.py \"<command>\" --duration {duration} "
                 f"--node {winner.get('node_id')} --rate-per-second {rate}")
    if runners_up:
        lines.append("runners-up:")
        lines.extend(f"  {describe(offer)}" for offer in runners_up[:4])
    if rejected:
        lines.append("passed over:")
        lines.extend(f"  {describe(offer)}: {'; '.join(reasons)}" for offer, reasons in rejected[:6])
    return "\n".join(lines)


def as_json(winner, runners_up, rejected, duration, min_vram_mib):
    return json.dumps({
        "chosen": winner,
        "reason": why(winner, runners_up),
        "duration_seconds": duration,
        "min_vram_mib": min_vram_mib,
        "deposit_base_units": rate_micros(winner) * duration,
        "rate_usdg_per_hour": round(rate_per_hour(winner), 6),
        "runners_up": runners_up,
        "passed_over": [{"offer": offer, "reasons": reasons} for offer, reasons in rejected],
    }, indent=2)


def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-vram-gb", type=float, default=16.0,
                        help="VRAM floor as the card is labelled; 24 means 24000 MiB")
    parser.add_argument("--max-usdg-per-hour", type=float, default=None)
    parser.add_argument("--duration", type=int, default=600,
                        help="lease window in seconds, used for the deposit figure")
    parser.add_argument("--min-trust", choices=TRUST_CLASSES, default="open")
    parser.add_argument("--include-staker-only", action="store_true",
                        help="keep offers only bonded stakers can rent")
    parser.add_argument("--offers", default=None,
                        help="read a saved offer list instead of the live one")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.min_vram_gb <= 0 or args.duration <= 0:
        parser.error("--min-vram-gb and --duration must be positive")
    if args.max_usdg_per_hour is not None and args.max_usdg_per_hour <= 0:
        parser.error("--max-usdg-per-hour must be positive")
    return args


def main(argv=None):
    args = parse(sys.argv[1:] if argv is None else argv)
    try:
        offers = load(args.offers)
    except (OSError, urllib.error.URLError, ValueError) as e:
        print(f"capacity could not be read: {e}", file=sys.stderr)
        return UNREACHABLE
    if not isinstance(offers, list):
        print("capacity answered in an unexpected shape; try again shortly", file=sys.stderr)
        return UNREACHABLE

    ceiling = None
    if args.max_usdg_per_hour is not None:
        ceiling = int(args.max_usdg_per_hour * MICROS / 3600)
    wanted = floor_mib(args.min_vram_gb)
    eligible, rejected = choose(offers, wanted, ceiling, args.min_trust, args.include_staker_only)
    if not offers:
        print("no capacity is published right now. Supply is a handful of machines and it "
              "changes; try again shortly.", file=sys.stderr)
        return NO_MATCH
    if not eligible:
        print(f"no offer met the constraints; {len(offers)} were published", file=sys.stderr)
        for offer, reasons in rejected[:8]:
            print(f"  {describe(offer)}: {'; '.join(reasons)}", file=sys.stderr)
        return NO_MATCH

    winner, runners_up = eligible[0], eligible[1:]
    print(as_json(winner, runners_up, rejected, args.duration, wanted) if args.json
          else report(winner, runners_up, rejected, args.duration))
    return MATCHED


if __name__ == "__main__":
    sys.exit(main())
