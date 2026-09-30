#!/usr/bin/env python3
"""userscan entry point.

  python run.py                    # default: T1, core+extended, 4 s budget
  python run.py --max-tier T2 --deep --allow-vss --budget-ms 8000
  python run.py --out profile.json
  python run.py --list             # show registered probes and exit
  python run.py --all-users        # add an aggregate-only pass per login account
  python run.py --os linux         # select probes as if on another OS (testing)
  python run.py --home /tmp/h --localappdata ... --appdata ... --hermes-home ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from userscan.runner import run, scan_accounts  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser("userscan")
    ap.add_argument("--max-tier", choices=["T0", "T1", "T2", "T3"], default="T1",
                    help="privacy cap: values above this tier are stripped from the output")
    ap.add_argument("--budget-ms", type=int, default=4000)
    ap.add_argument("--allow-vss", action="store_true", help="allow esentutl /vss copy (slow, admin)")
    ap.add_argument("--deep", action="store_true", help="run deep-tier probes too (SRUM, profile walks)")
    ap.add_argument("--only", action="append", help="probe id or family (repeatable)")
    ap.add_argument("--skip-family", action="append")
    ap.add_argument("--out", help="write JSON to this file instead of stdout")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--os", dest="os_name", choices=["detect", "windows", "darwin", "linux"], default="detect",
                    help="probe selection OS; default detects the real one")
    ap.add_argument("--all-users", action="store_true",
                    help="also run user-level L1 detectors once per login account; aggregate output only")
    ap.add_argument("--include-operator", action="store_true",
                    help="with --all-users: keep operator lab accounts (hn-e2e, ns960, ...)")
    ap.add_argument("--home", help="override l0 home (also HOME/USERPROFILE for the run)")
    ap.add_argument("--localappdata", help="override l0 localappdata / LOCALAPPDATA")
    ap.add_argument("--appdata", help="override l0 appdata / APPDATA")
    ap.add_argument("--hermes-home", help="override l0 hermes_home / HERMES_HOME ('' = unset)")
    args = ap.parse_args(argv)

    # import OS probe modules so their registrations land
    import userscan.specs  # noqa: F401

    if args.list:
        from userscan.registry import REGISTRY, INSIGHTS
        for pid in sorted(REGISTRY):
            p = REGISTRY[pid]
            print(f"{pid:44} {p.level} {p.tier} {p.collect:8} gate={p.gate or '-':28} {p.family}")
        for iid in sorted(INSIGHTS):
            print(f"L3 {iid:41} {INSIGHTS[iid].inputs}")
        return 0

    only = list(args.only) if args.only else None
    os_override = None if args.os_name == "detect" else args.os_name
    overrides = {k: getattr(args, k) for k in ("home", "localappdata", "appdata", "hermes_home")
                 if getattr(args, k) is not None}
    # The account pass runs first: its read guard is process-wide and must not overlap run()'s threads.
    accounts = scan_accounts(os_override=os_override, include_operator=args.include_operator) if args.all_users else None
    out = run(max_tier=args.max_tier, budget_ms=args.budget_ms, allow_vss=args.allow_vss,
              only=only, skip_family=list(args.skip_family or []), include_deep=args.deep,
              os_override=os_override, overrides=overrides or None)
    if accounts is not None:
        out["all_users"] = accounts
    text = json.dumps(out, indent=2, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        extra = f"  accounts={len(accounts['accounts'])}" if accounts is not None else ""
        print(f"wrote {args.out}  wall_ms={out['run']['wall_ms']}  facts={len(out['facts'])}  insights={len(out['insights'])}{extra}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
