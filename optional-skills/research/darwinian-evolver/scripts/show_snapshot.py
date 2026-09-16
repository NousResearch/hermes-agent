"""
show_snapshot.py — Dump the population from a darwinian-evolver snapshot pickle.

Usage:
    python show_snapshot.py PATH/TO/iteration_N.pkl [--field prompt_template]

The script is intentionally Organism-agnostic: it walks `org.__dict__` and prints
all str fields. By default it shows `prompt_template` if present; pass --field to
target a different attribute (e.g. `regex_pattern`, `sql_query`, `code_block`).
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import pickle
import sys
from pathlib import Path


HERMES_SNAPSHOT_SECRET = "HERMES_SNAPSHOT_SECRET"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", type=Path)
    ap.add_argument(
        "--field",
        default=None,
        help="Organism attribute to display. Defaults to the first str field found.",
    )
    ap.add_argument("--top", type=int, default=None, help="Show only top N by score.")
    ap.add_argument(
        "--i-trust-this-file",
        action="store_true",
        help=(
            "Required acknowledgement that the snapshot is from a trusted source. "
            "pickle.loads executes arbitrary code embedded in the file (RCE) and "
            "must NEVER be run on snapshots received from untrusted parties."
        ),
    )
    args = ap.parse_args()

    if not args.snapshot.exists():
        sys.exit(f"snapshot not found: {args.snapshot}")

    if not args.i_trust_this_file:
        sys.exit(
            "refusing to unpickle: pickle.loads is equivalent to executing arbitrary "
            "code from the snapshot file. Only proceed if you created/control this "
            "file, then re-run with --i-trust-this-file.\n"
            f"  file: {args.snapshot}"
        )

    # Verify HMAC integrity before unpickling. The HMAC must live in a sidecar
    # file (e.g. `iteration_N.pkl.hmac`) so we can verify the raw bytes WITHOUT
    # unpickling them first. pickle.loads() executes arbitrary code, so the
    # integrity check MUST happen on the file bytes, not on values extracted
    # post-unpickle.
    secret = os.environ.get(HERMES_SNAPSHOT_SECRET)
    if secret:
        hmac_path = args.snapshot.with_suffix(args.snapshot.suffix + ".hmac")
        if not hmac_path.exists():
            sys.exit(
                f"HMAC sidecar not found: {hmac_path}. The snapshot must be "
                "accompanied by an HMAC file written by the producer."
            )
        raw_bytes = args.snapshot.read_bytes()
        expected_hmac = hmac_path.read_text().strip()
        computed = hmac.new(
            secret.encode(), raw_bytes, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(computed, expected_hmac):
            sys.exit(
                f"HMAC verification failed for {args.snapshot} — "
                "snapshot may have been tampered with or corrupted."
            )
    else:
        print(
            f"WARNING: HERMES_SNAPSHOT_SECRET is not set — skipping HMAC verification "
            f"for {args.snapshot}. Set the environment variable to enable integrity checks.",
            file=sys.stderr,
        )

    print(
        f"WARNING: unpickling {args.snapshot} — this executes code embedded in the "
        "file. Only safe for snapshots you produced yourself.",
        file=sys.stderr,
    )

    # The outer pickle wraps a dict; the inner pickle contains the actual organism
    # objects, which must be importable under their original dotted path. If you
    # ran a custom driver, make sure its module is on sys.path before calling this.
    outer = pickle.loads(args.snapshot.read_bytes())  # noqa: S301 — gated by --i-trust-this-file
    if not isinstance(outer, dict) or "population_snapshot" not in outer:
        sys.exit("not a darwinian-evolver snapshot (no population_snapshot key)")
    inner = pickle.loads(outer["population_snapshot"])  # noqa: S301 — gated by --i-trust-this-file
    pairs = inner["organisms"]  # list of (Organism, EvaluationResult)

    print(f"# organisms: {len(pairs)}\n")
    ranked = sorted(pairs, key=lambda p: getattr(p[1], "score", 0) or 0, reverse=True)
    if args.top:
        ranked = ranked[: args.top]

    for i, (org, res) in enumerate(ranked):
        score = getattr(res, "score", float("nan"))
        print(f"=== rank {i} score={score:.3f} ===")
        # pick field
        field = args.field
        if field is None:
            for k, v in vars(org).items():
                if isinstance(v, str) and not k.startswith("_") and k not in {"id",}:
                    field = k
                    break
        val = getattr(org, field, None) if field else None
        if val is None:
            print(f"  (no string field; org fields: {list(vars(org).keys())})")
        else:
            print(f"  {field} ({len(val)} chars):")
            for ln in val.splitlines()[:30]:
                print(f"    {ln}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
