#!/usr/bin/env python3
"""Report the spend limits in force and say whether a planned lease fits inside them.

Usage:
    budget_check.py [--duration N] [--rate-per-second N] [--max-usdg F] [--json]

Needs no wallet and moves no money. Run it before committing to a long window:
the whole window is deposited up front, so the deposit is what the day is
charged against, not the smaller amount settlement ends up taking.

``--rate-per-second`` takes the ``rate_per_second`` field of the offer chosen by
``pick_gpu.py``. With both it and ``--duration``, the report says whether the
deposit clears the per-lease cap and today's remainder.

Exit codes, so a caller can branch without parsing the output:

    0   the limits are readable, and any planned deposit fits
    1   the planned deposit does not fit
    2   wrong arguments
    3   the limits or the ledger are unusable, so nothing may spend
"""

import argparse
import json
import sys

if sys.version_info < (3, 10):
    print("prismnetwork needs Python 3.10 or newer; this interpreter is "
          f"{sys.version_info.major}.{sys.version_info.minor}. Run these scripts with a "
          "3.10 or newer python3.", file=sys.stderr)
    raise SystemExit(2)

from prismnetwork import BudgetError, SpendLedger, call_ceiling, read_budget

MICROS = 1_000_000

FITS = 0
DOES_NOT_FIT = 1
UNUSABLE = 3


def usdg(micros):
    return f"{int(micros) / MICROS:.6f} USDG"


def verdict(deposit, ceiling, remaining):
    """Every limit the deposit breaks, in the order a caller can act on them."""
    problems = []
    if deposit > ceiling:
        problems.append(f"deposit {usdg(deposit)} is over the {usdg(ceiling)} per-lease cap; "
                        "shorten the window, pick a cheaper offer, or raise PRISM_MAX_USDG")
    if remaining is not None and deposit > remaining:
        problems.append(f"deposit {usdg(deposit)} is over the {usdg(remaining)} left of today; "
                        "wait for the rolling 24 hours to clear or raise PRISM_DAILY_BUDGET_USDG")
    return problems


def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--duration", type=int, default=None, help="planned lease window in seconds")
    parser.add_argument("--rate-per-second", type=int, default=None,
                        help="the chosen offer's rate_per_second, in USDG base units")
    parser.add_argument("--max-usdg", type=float, default=None,
                        help="lower the per-lease ceiling for this plan; it cannot raise it")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.rate_per_second is not None and args.rate_per_second <= 0:
        parser.error("--rate-per-second must be positive")
    if (args.duration is None) != (args.rate_per_second is None):
        parser.error("--duration and --rate-per-second go together; pass both or neither")
    if args.max_usdg is not None and args.max_usdg <= 0:
        # Caught here rather than by call_ceiling, whose failure means the
        # environment's limits are broken and sends the caller to inspect a
        # ledger that is fine.
        parser.error("--max-usdg must be positive; it lowers the per-lease ceiling")
    return args


def main(argv=None):
    args = parse(sys.argv[1:] if argv is None else argv)
    try:
        budget = read_budget()
        ledger = SpendLedger(budget.ledger_path, budget.daily_micros, budget.max_per_call_micros)
        ceiling = call_ceiling(args.max_usdg, budget.max_per_call_micros)
        status = ledger.status()
        remaining = ledger.remaining()
    except BudgetError as e:
        print(f"the spend limits are unusable, so nothing may spend: {e}", file=sys.stderr)
        return UNUSABLE

    if args.max_usdg is not None and int(args.max_usdg * MICROS) > ceiling:
        print(f"requested    {args.max_usdg} USDG, clamped to {usdg(ceiling)} by PRISM_MAX_USDG")

    deposit = None
    problems = []
    if args.duration is not None:
        deposit = args.rate_per_second * args.duration
        problems = verdict(deposit, ceiling, remaining)

    if args.json:
        print(json.dumps({
            **status,
            "ceiling_for_this_plan": usdg(ceiling),
            "planned_deposit": usdg(deposit) if deposit is not None else None,
            "fits": not problems if deposit is not None else None,
            "problems": problems,
        }, indent=2))
        return DOES_NOT_FIT if problems else FITS

    print(f"daily budget       {status['daily_budget']}")
    print(f"spent last 24h     {status['spent_last_24h']}")
    print(f"remaining today    {status['remaining_today']}")
    print(f"max per lease      {status['max_per_call']}")
    print(f"ceiling used here  {usdg(ceiling)}")
    print(f"ledger             {status['ledger']}")
    charges = status["charges_last_24h"]
    print(f"charges last 24h   {'none' if not charges else len(charges)}")
    for charge in charges:
        reference = f" ({charge['reference']})" if charge.get("reference") else ""
        print(f"  {charge['at']} {charge.get('tool', 'prism')} {charge['amount']}{reference}")

    if deposit is None:
        return FITS
    print(f"planned deposit    {usdg(deposit)} for {args.duration}s at "
          f"{args.rate_per_second} base units per second")
    if problems:
        for problem in problems:
            print(f"BLOCKED {problem}", file=sys.stderr)
        return DOES_NOT_FIT
    print("verdict            fits inside both limits")
    return FITS


if __name__ == "__main__":
    sys.exit(main())
