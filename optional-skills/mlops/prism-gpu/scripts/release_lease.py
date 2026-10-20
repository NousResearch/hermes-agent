#!/usr/bin/env python3
"""Release a lease by id, or list the leases this wallet is still paying for.

Usage:
    release_lease.py --lease <id>
    release_lease.py --list [--json]

The whole window is deposited when a lease is funded and the release is what
stops the meter, so a lease outlives the process that opened it and so does its
bill. This is the way back: ``--list`` names every lease the wallet holds and
which of them are still billing, ``--lease`` releases one by id. Settlement
then charges the seconds between access opening and the release and returns the
rest.

``scripts/lease_run_release.py`` releases in a ``finally`` block and only points
here when that release could not be reached, or when the failure happened
before it had a lease object at all.

Reads ``PRISM_AGENT_KEY``: the same wallet that funded the lease. No other
wallet can release it.

Exit codes, so a caller can branch without parsing the output:

    0   released, or the list printed
    1   the release was refused; the wallet is still paying for that lease
    2   wrong arguments, or no usable wallet
"""

import argparse
import json
import os
import sys

if sys.version_info < (3, 10):
    print("prismnetwork needs Python 3.10 or newer; this interpreter is "
          f"{sys.version_info.major}.{sys.version_info.minor}. Run these scripts with a "
          "3.10 or newer python3.", file=sys.stderr)
    raise SystemExit(2)

from prismnetwork import DEFAULT_ESCROW, PrismAgent, PrismError

RELEASED = 0
REFUSED = 1
USAGE = 2

# The states the control plane treats as finished. Anything else is a machine
# the wallet is still being charged for.
SETTLED_STATES = ("closing", "settlement_pending", "finalized", "refunded", "failed")

SUMMARY_FIELDS = ("state", "node_id", "duration_seconds", "started_at", "ends_at",
                  "deposit_base_units", "image")


class NoWallet(Exception):
    """Raised before anything reaches the network."""


def agent_from_env():
    key = (os.environ.get("PRISM_AGENT_KEY") or "").strip()
    if not key:
        raise NoWallet(
            "PRISM_AGENT_KEY is not set. It holds the private key of a wallet funded with "
            "USDG and a little native ETH on Robinhood Chain (id 4663). Every call that "
            "touches a lease signs with it, and only the wallet that funded a lease can "
            "release it. Reading capacity with pick_gpu.py needs no wallet."
        )
    return PrismAgent(
        key,
        os.environ.get("PRISM_ESCROW", DEFAULT_ESCROW),
        api_base=os.environ.get("PRISM_API_BASE", "https://prismnetwork.tech"),
        rpc_url=os.environ.get("PRISM_RPC_URL", "https://rpc.mainnet.chain.robinhood.com"),
    )


def billing(record):
    """Whether this record is a machine the wallet is still paying for.

    An entry that is not an object counts as billing. The safe reading of an
    unrecognisable record is that the meter may still be running."""
    if not isinstance(record, dict):
        return True
    return str(record.get("state", "")) not in SETTLED_STATES


def summarise(record):
    """One line per lease. The control plane may add fields, so unknown ones are
    counted rather than dropped silently."""
    if not isinstance(record, dict):
        return f"  ? BILLING unreadable entry: {record!r}"
    known = [f"{name}={record[name]}" for name in SUMMARY_FIELDS if record.get(name) is not None]
    extra = [name for name in record
             if name not in SUMMARY_FIELDS and name != "lease_id" and record.get(name) is not None]
    tail = f" (+{len(extra)} more fields; --json prints them)" if extra else ""
    marker = "BILLING" if billing(record) else "settled"
    return f"  {record.get('lease_id')} {marker} {' '.join(known)}{tail}"


def show(records, as_json):
    if not isinstance(records, list):
        # An empty list means no leases. Anything else means the call did not
        # answer the question, and printing "no leases" for it would be a false
        # all-clear on a wallet that may well be paying for one.
        print(f"the lease list came back as {type(records).__name__}, not a list, so "
              "what this wallet is paying for is unknown. Check the escrow address in "
              "PRISM_ESCROW and retry.", file=sys.stderr)
        return REFUSED
    if as_json:
        print(json.dumps(records, indent=2))
        return RELEASED
    if not records:
        print("no leases on this escrow for this wallet")
        return RELEASED
    open_now = [record for record in records if billing(record)]
    print(f"{len(records)} lease(s), {len(open_now)} still billing")
    for record in records:
        print(summarise(record))
    if open_now:
        ids = " ".join(str(record.get("lease_id")) for record in open_now
                       if isinstance(record, dict))
        print(f"release each of these: {ids}")
        first = next((r.get("lease_id") for r in open_now if isinstance(r, dict)), None)
        if first is not None:
            print(f"  python3 scripts/release_lease.py --lease {first}")
    return RELEASED


def release(agent, lease_id):
    result = agent.release(lease_id)
    print(f"released     lease {lease_id}; the meter is stopped")
    print("settlement   charges the seconds between access opening and this release, "
          "and returns the rest")
    if isinstance(result, dict) and result:
        print(f"control plane {json.dumps(result)}")
    print(f"receipt      python3 scripts/verify_receipt.py --lease {lease_id}")
    return RELEASED


def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--lease", default=None, help="id of the lease to release")
    what.add_argument("--list", action="store_true", dest="list_leases",
                      help="every lease this wallet holds on this escrow")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.lease is not None and not str(args.lease).strip():
        parser.error("--lease takes the lease id the failure named")
    return args


def main(argv=None):
    args = parse(sys.argv[1:] if argv is None else argv)
    try:
        agent = agent_from_env()
    except NoWallet as e:
        print(str(e), file=sys.stderr)
        return USAGE
    except ValueError as e:
        print(f"PRISM_AGENT_KEY is not usable: {e}", file=sys.stderr)
        return USAGE

    try:
        if args.list_leases:
            return show(agent.leases(), args.json)
        return release(agent, args.lease)
    except PrismError as e:
        detail = f": {json.dumps(e.body)}" if getattr(e, "body", None) else ""
        if args.list_leases:
            print(f"the lease list could not be read: {e.code}{detail}", file=sys.stderr)
            return USAGE
        if e.code in ("lease_not_found", "not_found"):
            print(f"no lease {args.lease} for this wallet on this escrow: {e.code}{detail}\n"
                  "Ids restart at 1 with each escrow deployment, and only the wallet that "
                  "funded a lease can release it. `--list` shows what this wallet holds; "
                  "PRISM_ESCROW picks the deployment.", file=sys.stderr)
            return REFUSED
        print(f"the release was refused: {e.code}{detail}\n"
              f"The wallet is still paying for lease {args.lease} until the window ends. "
              "Retry it.", file=sys.stderr)
        return REFUSED
    except Exception as e:
        # A traceback here would exit 1 by luck rather than by decision. The
        # meaning is the same either way: nothing was released.
        what = "the lease list could not be read" if args.list_leases else "the release failed"
        print(f"{what}: {type(e).__name__}: {e}", file=sys.stderr)
        return REFUSED


if __name__ == "__main__":
    sys.exit(main())
