#!/usr/bin/env python3
"""harness_state.py — on-disk state for the coding-harness execution loop.

Single source of truth for a long-horizon coding task: the goal/done-definition,
the ordered list of increments (each a falsifiable hypothesis), and a running log.
Survives context compaction — re-read with `status` whenever you lose the thread.

Stdlib only. Cross-platform. The agent drives this via the `terminal` tool.

Usage:
    harness_state.py init "<done-definition>" [--force]
    harness_state.py add-increment "<summary>" [--predict TEXT] [--risk TEXT]
    harness_state.py record-verification <change_id> <pass|fail|partial>
                         [--note TEXT] [--verdict keep|revert|partial]
    harness_state.py status
    harness_state.py --self-test

State file defaults to ./.hermes/coding-harness/state.json
(override with --state PATH or env CODING_HARNESS_STATE).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_VERSION = "1.0"
DEFAULT_REL_PATH = Path(".hermes") / "coding-harness" / "state.json"
DEFAULT_VERDICT = {"pass": "keep", "fail": "revert", "partial": "partial"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _state_path(args: argparse.Namespace) -> Path:
    override = getattr(args, "state", None) or os.environ.get("CODING_HARNESS_STATE")
    return Path(override) if override else DEFAULT_REL_PATH


def _load(path: Path) -> dict:
    if not path.exists():
        sys.exit(
            f"error: no state at {path}. Run `init` first "
            f"(or pass --state / set CODING_HARNESS_STATE)."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"error: cannot read state at {path}: {exc}")


def _save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: temp file in the same dir, then replace.
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _next_change_id(state: dict) -> str:
    return f"ch_{len(state['increments']) + 1:03d}"


def _find(state: dict, change_id: str) -> dict:
    for inc in state["increments"]:
        if inc["change_id"] == change_id:
            return inc
    sys.exit(f"error: no increment {change_id!r} in state.")


# --- commands -------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> None:
    path = _state_path(args)
    if path.exists() and not args.force:
        sys.exit(f"error: state already exists at {path}. Use --force to overwrite.")
    state = {
        "manifest_version": MANIFEST_VERSION,
        "goal": args.goal,
        "created_at": _now(),
        "increments": [],
        "log": [f"{_now()} init: goal set"],
    }
    _save(path, state)
    print(f"initialized {path}\ngoal: {args.goal}")


def cmd_add_increment(args: argparse.Namespace) -> None:
    path = _state_path(args)
    state = _load(path)
    change_id = _next_change_id(state)
    state["increments"].append(
        {
            "change_id": change_id,
            "summary": args.summary,
            "predicted_impact": {
                "expected": args.predict or "",
                "at_risk": args.risk or "",
            },
            "verification": {"status": "pending", "note": "", "verdict": ""},
        }
    )
    state["log"].append(f"{_now()} add-increment {change_id}: {args.summary}")
    _save(path, state)
    print(change_id)


def cmd_record_verification(args: argparse.Namespace) -> None:
    path = _state_path(args)
    state = _load(path)
    inc = _find(state, args.change_id)
    verdict = args.verdict or DEFAULT_VERDICT[args.status]
    inc["verification"] = {
        "status": args.status,
        "note": args.note or "",
        "verdict": verdict,
    }
    state["log"].append(
        f"{_now()} verify {args.change_id}: {args.status} ({verdict})"
    )
    _save(path, state)
    print(f"{args.change_id}: {args.status} -> {verdict}")


def _render(state: dict) -> str:
    lines = []
    lines.append(f"GOAL: {state['goal']}")
    lines.append(f"created: {state.get('created_at', '?')}")
    incs = state["increments"]
    done = sum(1 for i in incs if i["verification"]["verdict"] == "keep")
    lines.append(f"increments: {len(incs)} ({done} kept)")
    lines.append("")
    for inc in incs:
        v = inc["verification"]
        status = v["status"] or "pending"
        verdict = f" -> {v['verdict']}" if v["verdict"] else ""
        lines.append(f"[{inc['change_id']}] {status}{verdict}  {inc['summary']}")
        pi = inc["predicted_impact"]
        if pi.get("expected"):
            lines.append(f"    expect: {pi['expected']}")
        if pi.get("at_risk"):
            lines.append(f"    risk:   {pi['at_risk']}")
        if v.get("note"):
            lines.append(f"    proof:  {v['note']}")
    pending = [i["change_id"] for i in incs if i["verification"]["verdict"] != "keep"]
    lines.append("")
    if pending:
        lines.append(f"NEXT: resume at {pending[0]} (not yet kept)")
    else:
        lines.append("NEXT: all increments kept -> run FINAL VERIFY (full suite)")
    return "\n".join(lines)


def cmd_status(args: argparse.Namespace) -> None:
    print(_render(_load(_state_path(args))))


def cmd_self_test(_args: argparse.Namespace) -> None:
    """Run a full init -> add -> verify -> status cycle in a temp dir."""
    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp) / "state.json"
        ns = argparse.Namespace

        cmd_init(ns(state=str(sp), goal="self-test goal", force=False))
        assert sp.exists(), "init did not write state"

        cmd_add_increment(
            ns(state=str(sp), summary="first increment", predict="x passes", risk="y breaks")
        )
        state = json.loads(sp.read_text())
        assert state["increments"][0]["change_id"] == "ch_001", "bad change_id"

        cmd_record_verification(
            ns(state=str(sp), change_id="ch_001", status="pass", note="ran tests", verdict=None)
        )
        state = json.loads(sp.read_text())
        v = state["increments"][0]["verification"]
        assert v["status"] == "pass" and v["verdict"] == "keep", "verdict not derived"
        assert len(state["log"]) == 3, f"expected 3 log lines, got {len(state['log'])}"

        out = _render(state)
        assert "GOAL: self-test goal" in out
        assert "[ch_001] pass -> keep" in out
        assert "all increments kept" in out

        # round-trip: reload is byte-stable
        reloaded = json.loads(sp.read_text())
        assert reloaded == state, "state not stable across reload"

    print("self-test OK")


# --- arg parsing ----------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true", help="run an internal sanity cycle and exit")
    p.add_argument("--state", help="path to state file (default ./.hermes/coding-harness/state.json)")
    sub = p.add_subparsers(dest="command")

    pi = sub.add_parser("init", help="create state with a done-definition")
    pi.add_argument("goal")
    pi.add_argument("--force", action="store_true", help="overwrite existing state")
    pi.set_defaults(func=cmd_init)

    pa = sub.add_parser("add-increment", help="append an increment (a falsifiable hypothesis)")
    pa.add_argument("summary")
    pa.add_argument("--predict", help="expected impact (success signal)")
    pa.add_argument("--risk", help="at-risk regression to watch")
    pa.set_defaults(func=cmd_add_increment)

    pr = sub.add_parser("record-verification", help="record the external verification result")
    pr.add_argument("change_id")
    pr.add_argument("status", choices=["pass", "fail", "partial"])
    pr.add_argument("--note", help="actual command + outcome (the external proof)")
    pr.add_argument("--verdict", choices=["keep", "revert", "partial"], help="override derived verdict")
    pr.set_defaults(func=cmd_record_verification)

    ps = sub.add_parser("status", help="print goal, increments, and next step")
    ps.set_defaults(func=cmd_status)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        cmd_self_test(args)
        return
    if not getattr(args, "command", None):
        parser.print_help()
        sys.exit(2)
    args.func(args)


if __name__ == "__main__":
    main()
