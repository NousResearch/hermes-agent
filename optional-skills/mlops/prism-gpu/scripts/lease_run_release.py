#!/usr/bin/env python3
"""Rent a GPU, run one command on it, release it, and print what it cost to find out.

Usage:
    lease_run_release.py "nvidia-smi" [--duration 600] [--min-vram-gb 24]
                         [--node 0x... --rate-per-second N]
                         [--image DIGEST_PINNED] [--max-usdg F]
                         [--min-trust open|isolated|attested|confidential]
                         [--timeout 120] [--dry-run]

The release sits in a ``finally`` block, because the deposit covers the whole
window and only the release stops the meter. A command that crashes still gives
the wallet its unused seconds back; an abandoned lease bills to the end of its
window. When the release itself cannot be reached, the output names the lease
and the ``scripts/release_lease.py`` call that stops it.

One machine is priced before anything is signed. Pass ``--node`` and
``--rate-per-second`` from ``pick_gpu.py`` to fund the exact offer that was
priced, or let this script read capacity and pick on the same rules. Either way
the deposit reserved in the ledger, the deposit shown, and the ``max_deposit``
the escrow is allowed to pull are one figure: ``rate_per_second × duration``.
The per-lease ceiling only lowers it.

Reads ``PRISM_AGENT_KEY`` from the environment: a funded wallet on Robinhood
Chain (id 4663), holding USDG for the deposit and native ETH for gas. The spend
ledger at ``~/.prism/spend.json`` is written before the money moves.

Exit codes, so a caller can branch without parsing the output:

    0   the lease ran and was released, or --dry-run priced it without spending
    1   the run failed and nothing was spent
    2   wrong arguments, no capacity, or a deposit the spend limits refuse
        (nothing has been signed on any of these; --dry-run reaches them all)
    3   money moved and the lease is still open; the output says how to release it
    4   the lease was funded and released, but the command did not run
"""

import argparse
import json
import os
import signal
import sys

if sys.version_info < (3, 10):
    print("prismnetwork needs Python 3.10 or newer; this interpreter is "
          f"{sys.version_info.major}.{sys.version_info.minor}. Run these scripts with a "
          "3.10 or newer python3.", file=sys.stderr)
    raise SystemExit(2)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pick_gpu
from release_lease import NoWallet, agent_from_env

from prismnetwork import (
    DEFAULT_IMAGE,
    BudgetError,
    PrismError,
    SpendLedger,
    call_ceiling,
    read_budget,
    record_spend,
)

MICROS = 1_000_000
RECEIPTS = "https://api.prismnetwork.tech/proof/index.json"
# The SDK's own cap for a batch command. The SSH path does not enforce it, but
# the argument list gives out around the same size.
COMMAND_LIMIT = 8192

RELEASED = 0
UNSPENT_FAILURE = 1
USAGE = 2
MONEY_AT_RISK = 3
FUNDED_NOT_RUN = 4


def usdg(micros):
    return f"{int(micros) / MICROS:.6f} USDG"


def deposited(lease, planned):
    """What the escrow pulled, from the funding receipt where the log was
    readable and from the plan where it was not. The ledger settles against this
    rather than the quote, which is the other side's document.

    A figure above the plan means the log was misread; ``max_deposit`` is
    enforced on the way in, so the plan is the safer of the two."""
    held = getattr(lease, "deposit_micros", None)
    if isinstance(held, int) and not isinstance(held, bool) and 0 < held <= planned:
        return held
    return planned


class NoCapacity(Exception):
    """Nothing published matches the constraints, so there is nothing to price."""


def price(args):
    """Name one machine and its rate before anything is signed.

    A rate is what makes the deposit knowable, and the deposit is what the
    ledger and the escrow ceiling are both set from. Taking the rate from an
    offer and then leaving the node open would let funding land on a dearer
    machine than the one that was checked."""
    if args.node:
        return args.node, args.rate_per_second, f"node {args.node} at the rate given"
    offers = pick_gpu.load(None)
    if not isinstance(offers, list):
        raise NoCapacity("capacity answered in an unexpected shape; try again shortly")
    eligible, rejected = pick_gpu.choose(
        offers, pick_gpu.floor_mib(args.min_vram_gb), None, args.min_trust, False
    )
    if not eligible:
        missed = "\n".join(f"  {pick_gpu.describe(offer)}: {'; '.join(reasons)}"
                           for offer, reasons in rejected[:8])
        raise NoCapacity(
            f"no offer met the constraints; {len(offers)} were published\n{missed}"
        )
    winner = eligible[0]
    # Anything without a node id or a positive rate was rejected above, so both
    # of these are present.
    return winner["node_id"], pick_gpu.rate_micros(winner), pick_gpu.describe(winner)


def recovery(lease_id, funding_hash):
    """What to run to stop a meter this process could not reach.

    Every line is a shell command, because the caller is holding a terminal and
    a running bill. A failure inside ``lease()`` can leave
    a funded machine that this script never got an id for, so the instructions
    have to work from the funding transaction alone."""
    lines = ["The escrow holds the deposit and the meter is running until the lease is",
             "released. Stop it through the `terminal` tool, from the skill directory:"]
    if lease_id is not None:
        lines.append(f"  python3 scripts/release_lease.py --lease {lease_id}")
        lines.append(f"  python3 scripts/verify_receipt.py --lease {lease_id}")
    else:
        lines.append("  python3 scripts/release_lease.py --list")
        if funding_hash:
            lines.append(f"  # the lease funded by {funding_hash} is the one to release")
        lines.append("  python3 scripts/release_lease.py --lease <id from that list>")
    return "\n".join(lines)


def explain(error, released=False):
    """A PrismError says which side of the wire it happened on. ``broadcast`` is
    False before the funding transaction is signed, the transaction hash after,
    and None where nothing established which. ``released`` overrides all three:
    once the lease is closed the meter is stopped whatever the failure was, and
    telling the caller to go hunting for an open lease sends it after a lease
    that no longer exists."""
    sent = getattr(error, "broadcast", None)
    body = getattr(error, "body", None)
    detail = f": {json.dumps(body)}" if body else ""
    lease_id = body.get("lease_id") if isinstance(body, dict) else None
    if released:
        return (FUNDED_NOT_RUN,
                f"{fault(error)} on a lease that was funded and then released{detail}\n"
                "The meter is stopped. Settlement charges the seconds served and returns "
                "the rest, so the cost is the provisioning, not the window.")
    if isinstance(sent, str) and sent:
        return (MONEY_AT_RISK,
                f"{fault(error)} after the deposit was broadcast in {sent}{detail}\n"
                + recovery(lease_id, sent))
    if sent is False:
        return UNSPENT_FAILURE, f"{fault(error)} before anything was signed{detail}"
    return (MONEY_AT_RISK,
            f"{fault(error)}, and nothing established whether the deposit was sent{detail}\n"
            "Treat it as spent until the wallet's leases say otherwise.\n"
            + recovery(lease_id, None))


def fault(error):
    """A Prism error carries a code. Anything from below the SDK carries a type."""
    return getattr(error, "code", None) or type(error).__name__


def stop_meter(agent, lease):
    """Release the lease. Returns the failure that prevented it, or None.

    Catches everything, because a release that fails on a reset connection is
    still a lease that is billing, and an exception escaping here would skip the
    report that says so. Returning the failure rather than raising keeps it
    distinguishable from whatever the command itself did."""
    try:
        agent.end_lease(lease)
        print(f"released     lease {lease.lease_id}; settlement charges the seconds served")
        return None
    except Exception as e:
        print(f"RELEASE FAILED {fault(e)}: the wallet is still paying for lease "
              f"{lease.lease_id}.", file=sys.stderr)
        return e


def run_lease(agent, ledger, args, node, planned):
    def fund():
        lease = agent.lease(
            image=args.image,
            duration_seconds=args.duration,
            min_vram_mib=pick_gpu.floor_mib(args.min_vram_gb),
            preferred_node_id=node,
            max_deposit=planned,
            min_trust_class=args.min_trust,
        )
        return {
            "value": lease,
            "settled_micros": deposited(lease, planned),
            "reference": lease.funding_hash,
        }

    try:
        lease = record_spend(ledger, "prism_lease_run_release", planned, fund)
    except BaseException:
        # Funding blocks for minutes waiting on the receipt and then on access.
        # An interrupt in that window can land after the deposit is broadcast.
        print("interrupted while funding; a deposit may already be in escrow.\n"
              + recovery(None, None), file=sys.stderr)
        raise
    failed = None
    # Everything from here down is inside the try: the meter is running, and a
    # print to a closed stdout is enough to skip the release if it sits outside.
    try:
        print(f"lease        {lease.lease_id} funded in {lease.funding_hash}")
        print(f"deposit      {usdg(deposited(lease, planned))} for a {args.duration}s window")
        print(f"access       {lease.access.get('ssh_host')}:{lease.access.get('ssh_port')}")
        result = agent.run(lease, args.command, timeout=args.timeout)
        print(f"exit         {result.get('code')}")
        # Both streams, always. The box is destroyed a few lines below and
        # nothing on it survives, so a diagnostic dropped here is a diagnostic
        # that costs another deposit to reproduce.
        for stream in ("stdout", "stderr"):
            text = (result.get(stream) or "").rstrip()
            if text:
                print(f"--- {stream} ---")
                print(text)
    except Exception as e:
        # Held rather than raised: the release below decides what this costs.
        failed = e
    finally:
        unreleased = stop_meter(agent, lease)

    if unreleased is not None:
        print(f"the lease was funded but not released: {fault(unreleased)}\n"
              + recovery(lease.lease_id, lease.funding_hash), file=sys.stderr)
        return MONEY_AT_RISK
    print(f"receipt      python3 scripts/verify_receipt.py --lease {lease.lease_id}")
    print(f"             (published to {RECEIPTS} once settlement lands)")
    if failed is not None:
        code, message = explain(failed, released=True)
        print(f"the command did not run: {message}", file=sys.stderr)
        return code
    return RELEASED


def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", help="shell command to run on the GPU")
    parser.add_argument("--duration", type=int, default=600, help="lease window in seconds")
    parser.add_argument("--min-vram-gb", type=float, default=16.0,
                        help="VRAM floor as the card is labelled; 24 means 24000 MiB")
    parser.add_argument("--node", default=None,
                        help="node_id from pick_gpu.py; funds that offer and no other")
    parser.add_argument("--rate-per-second", type=int, default=None,
                        help="that offer's rate_per_second, in USDG base units")
    parser.add_argument("--image", default=DEFAULT_IMAGE,
                        help="digest-pinned image; the control plane matches on the digest")
    parser.add_argument("--max-usdg", type=float, default=None,
                        help="lower the per-call ceiling for this lease; it cannot raise it")
    parser.add_argument("--min-trust", default="open",
                        choices=("open", "isolated", "attested", "confidential"))
    parser.add_argument("--timeout", type=int, default=120, help="seconds to wait for the command")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the ceiling and the plan, spend nothing")
    args = parser.parse_args(argv)
    if args.duration <= 0 or args.timeout <= 0 or args.min_vram_gb <= 0:
        parser.error("--duration, --timeout and --min-vram-gb must be positive")
    if not args.command.strip():
        parser.error("command must not be empty")
    if (args.node is None) != (args.rate_per_second is None):
        parser.error("--node and --rate-per-second go together; a node without its rate "
                     "leaves the deposit unknown before it is signed")
    if args.rate_per_second is not None and args.rate_per_second <= 0:
        parser.error("--rate-per-second must be positive")
    if args.max_usdg is not None and args.max_usdg <= 0:
        parser.error("--max-usdg must be positive; it lowers the per-call ceiling")
    if args.timeout >= args.duration:
        parser.error(f"--timeout {args.timeout} leaves no room inside a {args.duration}s "
                     "window. Provisioning and the SSH connect budget run to about four "
                     "minutes before the command starts, so the window has to be longer "
                     "than the command is allowed to take.")
    if "@sha256:" not in args.image:
        parser.error(f"--image {args.image} carries no digest. The control plane matches on "
                     "the digest, so a tag is refused after the deposit is planned. Pin it "
                     "with @sha256:<64 hex>.")
    if len(args.command.encode("utf-8")) > COMMAND_LIMIT:
        parser.error(f"the command is over {COMMAND_LIMIT} bytes and would not survive the "
                     "argument list. Fetch the payload on the box instead of inlining it.")
    return args


def terminate(*_):
    """SIGTERM's default disposition kills the interpreter without unwinding, so
    the release in the finally never runs and the window bills out. Turning it
    into SystemExit unwinds. SIGKILL stays unrecoverable, which is what
    release_lease.py --list is for."""
    sys.exit(MONEY_AT_RISK)


def main(argv=None):
    args = parse(sys.argv[1:] if argv is None else argv)
    try:
        budget = read_budget()
        ledger = SpendLedger(budget.ledger_path, budget.daily_micros, budget.max_per_call_micros)
        ceiling = call_ceiling(args.max_usdg, budget.max_per_call_micros)
    except BudgetError as e:
        print(f"the spend limits are unusable, so nothing may spend: {e}", file=sys.stderr)
        return USAGE

    if args.max_usdg is not None and int(args.max_usdg * MICROS) > ceiling:
        print(f"requested    {args.max_usdg} USDG, clamped to {usdg(ceiling)} by PRISM_MAX_USDG")
    print(f"ceiling      {usdg(ceiling)} for this lease")
    try:
        left = ledger.remaining()
    except BudgetError as e:
        print(f"the spend ledger is unreadable, so nothing may spend: {e}", file=sys.stderr)
        return USAGE
    print("remaining    unlimited (daily ceiling removed)" if left is None
          else f"remaining    {usdg(left)} of today's budget")

    try:
        node, rate, described = price(args)
    except NoCapacity as e:
        print(f"nothing to lease: {e}", file=sys.stderr)
        return USAGE
    except (OSError, ValueError) as e:
        print(f"capacity could not be read, so no deposit can be planned: {e}", file=sys.stderr)
        return USAGE

    planned = rate * args.duration
    print(f"machine      {described}")
    print(f"node         {node}")
    print(f"deposit      {usdg(planned)} planned for {args.duration}s at {rate} base units/s")

    # Checked here rather than left to the ledger, so the refusal names the
    # limit and the fix instead of surfacing as a mid-flight budget error.
    if planned > ceiling:
        print(f"deposit {usdg(planned)} is over the {usdg(ceiling)} per-lease cap; shorten the "
              "window, pick a cheaper offer, or raise PRISM_MAX_USDG", file=sys.stderr)
        return USAGE
    if left is not None and planned > left:
        print(f"deposit {usdg(planned)} is over the {usdg(left)} left of today; wait for the "
              "rolling 24 hours to clear or raise PRISM_DAILY_BUDGET_USDG", file=sys.stderr)
        return USAGE

    if args.dry_run:
        print(f"dry run      would lease {pick_gpu.floor_mib(args.min_vram_gb)} MiB or more for "
              f"{args.duration}s on {args.image} and run: {args.command}")
        return RELEASED

    signal.signal(signal.SIGTERM, terminate)

    try:
        agent = agent_from_env()
    except NoWallet as e:
        print(str(e), file=sys.stderr)
        return USAGE
    except ValueError as e:
        print(f"PRISM_AGENT_KEY is not usable: {e}", file=sys.stderr)
        return USAGE

    try:
        return run_lease(agent, ledger, args, node, planned)
    except PrismError as e:
        code, message = explain(e)
        print(f"the lease did not complete: {message}", file=sys.stderr)
        return code
    except BudgetError as e:
        print(f"refused by the spend limits: {e}", file=sys.stderr)
        return USAGE


if __name__ == "__main__":
    sys.exit(main())
