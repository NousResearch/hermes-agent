#!/usr/bin/env python3
"""Check a published Prism settlement receipt offline, then say what to match on chain.

Usage:
    verify_receipt.py <receipt_id | receipt url | path/to/receipt.json>
    verify_receipt.py --lease <lease_id> [--escrow 0x...]
    verify_receipt.py --self-test

Recomputes ``receipt_hash`` from the canonical payload and reconciles charged,
refunded and provider-paid amounts. Everything here is stdlib: no wallet, no
key, and no dependency on the caller having run the lease. The canonical form
is field-declaration order with absent fields dropped, so a re-serialisation
that sorts keys produces a different digest and a false negative.

Lease ids are numbered per escrow deployment and restart at 1 with each one, so
the same id names a different run on each escrow the feed has published. Lookup
by lease id is therefore scoped to one escrow, ``PRISM_ESCROW`` or the live
default, and refuses to guess when the scope still leaves more than one.

Exit codes, so a caller can branch without parsing the output:

    0   verified and the run completed cleanly
    1   a check failed; this receipt is not evidence
    2   wrong arguments, or no receipt for that lease on that escrow
    3   verified, but the run did not complete cleanly (see failure_class)
    4   the feed could not be reached, so nothing was checked
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from collections import OrderedDict

FEED = "https://api.prismnetwork.tech/proof/receipts"
INDEX = "https://api.prismnetwork.tech/proof/index.json"
EXPLORER = "https://robinhoodchain.blockscout.com/tx"
# The escrow currently taking deposits. Retired deployments stay in the feed
# with their own lease numbering, which is why this scopes the lookup.
DEFAULT_ESCROW = "0xfD4228eEEfC49e4b76A0CD40af9fdd546220B2FD"

OK = 0
FAILED = 1
USAGE = 2
NOT_CLEAN = 3
UNREACHABLE = 4

# Declaration order. Sorting these produces a different digest.
RECEIPT_FIELDS = (
    "receipt_id", "lease_id", "node_id_hash", "gpu_model", "runtime_seconds",
    "charged_base_units", "refunded_base_units", "provider_paid_base_units",
    "failure_class", "outcome", "trust_class", "attestation",
    "credited_seconds", "repro",
)
REPRO_FIELDS = (
    "executor", "token_hash", "spec_hash", "image_digest", "command_hash",
    "result_hash", "stdout_hash", "stderr_hash", "report_hash", "exit_code",
    "expected_exit_code", "succeeded", "truncated",
)
EMPTY_STREAM = hashlib.sha256(b"").hexdigest()

FAILURE_MEANING = {
    "interrupted": "the machine stopped answering before the paid window ended",
    "provisioning_timeout": "the machine never booted",
}


def canonical(receipt):
    payload = OrderedDict()
    for field in RECEIPT_FIELDS:
        # The only optional field that survives as an explicit null. The rest
        # disappear, which is what keeps older receipts byte-identical.
        if field == "failure_class":
            payload[field] = receipt.get("failure_class")
        elif field == "repro" and "repro" in receipt:
            payload[field] = OrderedDict(
                (name, receipt["repro"][name])
                for name in REPRO_FIELDS
                if name in receipt["repro"]
            )
        elif field in receipt:
            payload[field] = receipt[field]
    return json.dumps(payload, separators=(",", ":")).encode()


def get(url, timeout=30):
    request = urllib.request.Request(url, headers={"accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def is_address(value):
    text = str(value).strip()
    return (text.lower().startswith("0x") and len(text) == 42
            and all(c in "0123456789abcdefABCDEF" for c in text[2:]))


def escrow_in_force(explicit=None):
    """The escrow to scope a lookup to. Whitespace is stripped before the
    fallbacks, so PRISM_ESCROW=" " falls through to the default rather than
    matching every entry that omits the field."""
    named = next((c for c in (explicit, os.environ.get("PRISM_ESCROW")) if c and c.strip()),
                 DEFAULT_ESCROW)
    return named.strip().lower()


def resolve(index, lease_id, escrow):
    """Pick the one receipt for ``lease_id`` that ``escrow`` settled.

    Ids repeat across escrow deployments, so matching on the id alone hands
    back whichever entry the feed happened to list first, which can be a
    stranger's run of a different length on a retired contract. Anything short
    of exactly one match on the named escrow raises instead of guessing."""
    same_id = [r for r in index.get("receipts", []) if str(r.get("lease_id")) == str(lease_id)]
    if not same_id:
        raise LookupError(
            f"no published receipt for lease {lease_id} on any escrow. Settlement lands a "
            "little after the release; a lease still open has no receipt yet."
        )
    matches = [r for r in same_id
               if str(r.get("escrow_address", "")).strip().lower() == escrow]
    if not matches:
        elsewhere = sorted({str(r.get("escrow_address")) for r in same_id})
        raise LookupError(
            f"lease {lease_id} has {len(same_id)} published receipt(s), none settled by the "
            f"escrow at {escrow}. The id is in use on: {', '.join(elsewhere)}. Those are other "
            "leases that happen to share the number. Set PRISM_ESCROW or pass --escrow to name "
            "the deployment the lease was funded on."
        )
    ids = sorted({str(r["receipt_id"]) for r in matches if r.get("receipt_id")})
    if not ids:
        raise LookupError(
            f"the feed lists lease {lease_id} on escrow {escrow} without a receipt id, so "
            "there is nothing to fetch. Pass the receipt id."
        )
    if len(ids) > 1:
        raise LookupError(
            f"lease {lease_id} on escrow {escrow} resolves to {len(ids)} receipts "
            f"({', '.join(ids)}), so the id does not identify a run. Pass the receipt id."
        )
    return ids[0]


def load(argument):
    if argument.startswith(("http://", "https://")):
        return get(argument)
    if os.path.exists(argument):
        with open(argument, encoding="utf-8") as handle:
            return json.load(handle)
    if argument.endswith(".json") or os.sep in argument:
        # It was written as a path. Sending it to the feed turns a typo into a
        # 404 and blames the network for a file that is simply not there.
        raise FileNotFoundError(f"no such file: {argument}")
    return get(f"{FEED}/{argument}.json")


# What report() reads without a default. A document short of these is not a
# receipt, and the feed is a network source, so this is checked before printing.
REQUIRED = ("receipt_hash", "receipt_id", "charged_base_units",
            "refunded_base_units", "provider_paid_base_units")


def missing_fields(receipt):
    if not isinstance(receipt, dict):
        return ["the shape of a receipt"]
    return [name for name in REQUIRED if receipt.get(name) is None]


def report(receipt):
    """No I/O beyond stdout, so the exit code is the whole verdict."""
    computed = hashlib.sha256(canonical(receipt)).hexdigest()
    published = receipt.get("receipt_hash")
    charged = int(receipt["charged_base_units"])
    refunded = int(receipt["refunded_base_units"])
    paid = int(receipt["provider_paid_base_units"])
    runtime = receipt.get("runtime_seconds")
    # Seconds held and not charged for. Read as the metered figure it understates
    # the run and the arithmetic stops reconciling.
    credited = receipt.get("credited_seconds")
    failure = receipt.get("failure_class")

    failures = []
    if computed != published:
        failures.append(f"receipt_hash mismatch: computed {computed}, published {published}")
    if paid > charged:
        failures.append(f"provider paid {paid} of {charged} charged")
    if receipt.get("outcome") == "disputed":
        failures.append("receipt is disputed and is not final proof")

    print(f"receipt      {receipt['receipt_id']}")
    print(f"escrow       {receipt.get('escrow_address')} lease {receipt.get('chain_lease_id')}")
    print(f"gpu          {receipt.get('gpu_model')}, trust class {receipt.get('trust_class', 'unstated')}")
    print(f"outcome      {receipt.get('outcome')} ({failure or 'ran clean'})")
    print(f"metered      {runtime}s, charged {charged} base units ({charged / 1e6:.6f} USDG)")
    if credited is not None:
        print(f"credited     {credited}s held and NOT charged")
    print(f"deposit      {charged + refunded} base units, {refunded} refunded, {paid} to the provider")
    if "repro" in receipt:
        repro = receipt["repro"]
        print(f"image        {repro.get('image_digest')}")
        print(f"executor     {repro.get('executor')}, exit {repro.get('exit_code')}"
              f" (expected {repro.get('expected_exit_code')})")
        for stream in ("stdout_hash", "stderr_hash"):
            value = repro.get(stream)
            note = " (empty)" if value == EMPTY_STREAM else ""
            print(f"{stream:<12} {value}{note}")
    print(f"receipt_hash {computed} {'ok' if computed == published else 'MISMATCH'}")
    print(f"settlement   {EXPLORER}/{receipt.get('transaction_hash')}")

    if failures:
        print()
        for line in failures:
            print(f"FAIL {line}")
        return FAILED

    print()
    print("Offline checks pass. Still to confirm on Robinhood Chain (id 4663):")
    if receipt.get("outcome") == "refunded":
        # LeaseRefunded's third field is reasonHash, so a refund's receipt hash
        # is self-consistency evidence rather than a value the chain committed.
        print(f"  the escrow at {receipt.get('escrow_address')} emitted one LeaseRefunded")
        print(f"  for leaseId {receipt.get('chain_lease_id')} refunding {refunded},")
        print(f"  carrying the canonical reason hash for {failure}.")
        print("  The receipt hash above is NOT committed by a refund event.")
    else:
        print(f"  the escrow at {receipt.get('escrow_address')} emitted one LeaseFinalized")
        print(f"  for leaseId {receipt.get('chain_lease_id')} with receiptHash 0x{computed},")
        print(f"  charged {charged}, providerPaid {paid}, refunded {refunded},")
        print("  and fee + providerPaid == charged.")

    if failure:
        print()
        print(f"CAUTION this run did not complete cleanly: {failure}"
              f" ({FAILURE_MEANING.get(failure, 'reason not recognised by this checker')}).")
        if credited is not None:
            print(f"        {credited}s of the window were held and never billed.")
        print("        Cite it only with the failure named alongside the claim.")
        return NOT_CLEAN
    return OK


# Two receipts exactly as the feed published them, one clean and one cut short.
# Pinned because the classification below is the part that is easy to get
# backwards: credited_seconds is uncharged time, so a checker that meters it
# reports 152s on a lease that billed 192.
PINNED = {
    "clean": {
        "receipt_id": "8f3e0c1d-391c-8510-9f77-ebc574905ffe",
        "lease_id": "34",
        "node_id_hash": "0xde1fbfc73bdf94761ae16a7f6c98716ea22ece3ce23c1b67d4a7053cb5605dfc",
        "gpu_model": "RTX 6000Ada",
        "runtime_seconds": 28,
        "charged_base_units": 6216,
        "refunded_base_units": 126984,
        "provider_paid_base_units": 5595,
        "failure_class": None,
        "outcome": "finalized",
        "trust_class": "open",
        "escrow_address": "0xfd4228eeefc49e4b76a0cd40af9fdd546220b2fd",
        "chain_lease_id": "34",
        "receipt_hash": "2970ee269291020bf03a2d665f6e68accb3a77048dfc0902d0c1d1b5a9a2dbd3",
        "transaction_hash": "0x1de4eba627f8ffc6c0ce3f628b648c5f13d9f2815843d1fc6979a9b731309d88",
    },
    "interrupted": {
        "receipt_id": "916ea93f-e7fd-8c59-b20d-80c70019d280",
        "lease_id": "143",
        "node_id_hash": "0x8126aa45f5829caa2ded18ab38fbe41954f67c5675d70d38278eb6f446077688",
        "gpu_model": "RTX 5880Ada",
        "runtime_seconds": 192,
        "charged_base_units": 42624,
        "refunded_base_units": 157176,
        "provider_paid_base_units": 38362,
        "failure_class": "interrupted",
        "outcome": "finalized",
        "trust_class": "open",
        "credited_seconds": 152,
        "escrow_address": "0x62c042265991bea17b07229322a01850974626da",
        "chain_lease_id": "143",
        "receipt_hash": "93517fa85353029e27db4d1c50721bff751e6c8a894b0e50da37574a61e4ecf5",
        "transaction_hash": "0xb8de1c478fab41fb3dc4daf6ff4ac9b8c30d3774282a8c91c5aef176d6cd66b5",
    },
}

# Three published index entries that all carry lease_id 34, one per escrow the
# feed has ever settled. Pinned because this collision is the whole reason
# lookup by lease id is scoped: matching on the number alone returns whichever
# of these the feed lists first.
COLLIDING_INDEX = {
    "receipts": [
        {"lease_id": "34", "receipt_id": "8f3e0c1d-391c-8510-9f77-ebc574905ffe",
         "escrow_address": "0xfd4228eeefc49e4b76a0cd40af9fdd546220b2fd",
         "runtime_seconds": 28, "charged_base_units": 6216},
        {"lease_id": "34", "receipt_id": "397e9593-7cea-8d9a-9ee3-238ab4b63bb6",
         "escrow_address": "0x62c042265991bea17b07229322a01850974626da",
         "runtime_seconds": 1800, "charged_base_units": 399600},
        {"lease_id": "34", "receipt_id": "b6d76bdf-de6e-8a5f-8972-85065406120f",
         "escrow_address": "0x71df0ef3bc81022cb3bec0b1a05f52f12bafcded",
         "runtime_seconds": 120, "charged_base_units": 26640},
    ]
}


def self_test():
    """Check the pinned receipts without touching the network.

    Written with explicit comparisons rather than assert, because `python3 -O`
    strips assert and this is the check SKILL.md points at to back the billing
    claims. A self-test that passes with its checks removed proves nothing."""
    failures = []
    for name, receipt in PINNED.items():
        computed = hashlib.sha256(canonical(receipt)).hexdigest()
        if computed != receipt["receipt_hash"]:
            failures.append(f"{name}: canonicalisation disagrees with the published hash")
        print(f"--- {name} ---")
        code = report(receipt)
        expected = OK if name == "clean" else NOT_CLEAN
        if code != expected:
            failures.append(f"{name}: exit {code}, expected {expected}")
        print()

    interrupted = PINNED["interrupted"]
    rate = interrupted["charged_base_units"] / interrupted["runtime_seconds"]
    if rate != 222:
        failures.append(
            "the charge divides by runtime_seconds at the quoted rate, so a checker "
            "that meters credited_seconds is reporting the wrong number"
        )

    print("--- tampered, expected to be rejected ---")
    if report(dict(PINNED["clean"], charged_base_units=1)) != FAILED:
        failures.append("an edited receipt has to fail")

    live = escrow_in_force(DEFAULT_ESCROW)
    try:
        resolved = resolve(COLLIDING_INDEX, 34, live)
    except LookupError as e:
        resolved = f"LookupError: {e}"
    if resolved != PINNED["clean"]["receipt_id"]:
        failures.append(
            "lease 34 on the live escrow is the 28-second run, not one of the two "
            f"leases numbered 34 on the retired escrows; got {resolved}"
        )
    try:
        resolve(COLLIDING_INDEX, 34, "0x" + "11" * 20)
        failures.append("a lease id on an escrow that never settled it must not resolve")
    except LookupError:
        pass

    print()
    if failures:
        for line in failures:
            print(f"SELF-TEST FAILED: {line}", file=sys.stderr)
        return FAILED
    print(f"{len(PINNED)} pinned receipts verified, tampering rejected, "
          "colliding lease ids scoped to one escrow")
    return OK


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("receipt", nargs="?", help="receipt id, receipt URL, or a local .json file")
    parser.add_argument("--lease", default=None,
                        help="look the receipt up by lease id, scoped to one escrow")
    parser.add_argument("--escrow", default=None,
                        help="escrow the lease was funded on; defaults to PRISM_ESCROW, "
                             "then to the live escrow")
    parser.add_argument("--self-test", action="store_true",
                        help="verify the pinned receipts offline; no network")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.self_test:
        if args.receipt or args.lease or args.escrow:
            parser.error("--self-test runs offline against the pinned receipts and takes "
                         "nothing else")
        return self_test()
    if bool(args.receipt) == bool(args.lease):
        parser.error("pass a receipt id, or --lease, or --self-test")
    if args.lease is not None and not str(args.lease).strip().isdigit():
        parser.error(f"--lease takes the lease number, not {args.lease!r}")
    if args.escrow is not None and not is_address(args.escrow):
        parser.error(f"--escrow takes a 20-byte address, not {args.escrow!r}")

    try:
        target = (resolve(get(INDEX), args.lease, escrow_in_force(args.escrow))
                  if args.lease else args.receipt)
        receipt = load(target)
    except LookupError as e:
        print(str(e), file=sys.stderr)
        return USAGE
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return USAGE
    except (urllib.error.URLError, OSError) as e:
        print(f"the feed could not be reached: {e}", file=sys.stderr)
        return UNREACHABLE
    except ValueError as e:
        print(f"the receipt did not parse as JSON: {e}", file=sys.stderr)
        return USAGE
    missing = missing_fields(receipt)
    if missing:
        print(f"that is not a settlement receipt; it has no {', '.join(missing)}",
              file=sys.stderr)
        return USAGE
    return report(receipt)


if __name__ == "__main__":
    sys.exit(main())
