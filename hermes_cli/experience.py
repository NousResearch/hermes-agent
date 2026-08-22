"""CLI subcommand: `hermes experience <subcommand>`.

Thin shell around ``ExperienceStoreMixin`` (hermes_state_experience.py) and the
scoring/rendering helpers in ``agent/experience.py``. Level 2 experience
learning silently adds context to prompts and silently accumulates rows; this
is how a human sees what it holds, why it retrieved something, and how to make
it forget.

``why`` is the one that earns its keep: it renders the exact block a given
prompt would receive, with the score behind each row. Without it the feature is
a black box — you can see that it changed an answer but not what it fed in.

This module intentionally has no side effects at import time — main.py wires
the argparse subparsers on demand.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional

OUTCOMES = ("success", "partial", "failure", "interrupted")


def _fmt_age(ts: Optional[float]) -> str:
    """Relative age, matching `hermes curator`'s rendering."""
    if not ts:
        return "never"
    secs = int(max(0.0, time.time() - float(ts)))
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def _truncate(text: Any, width: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= width else s[: width - 1] + "…"


def _open_db():
    from hermes_state import SessionDB

    return SessionDB()


def _current_workspace() -> str:
    """The scoping key for the directory the CLI was invoked from."""
    try:
        from agent.coding_context import project_facts_for

        facts = project_facts_for(None)
        if facts and facts.get("root"):
            return str(facts["root"])
    except Exception:
        pass
    try:
        import os

        return os.getcwd()
    except Exception:
        return ""


# ── stats ───────────────────────────────────────────────────────────────


def _cmd_stats(args) -> int:
    db = _open_db()
    try:
        stats = db.experience_stats()
    finally:
        db.close()

    if getattr(args, "json", False):
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 0

    total = stats.get("total", 0)
    if not total:
        print("experience: nothing recorded yet")
        print("  (turns that use no tools and do not fail are never stored)")
        return 0

    print(f"experience: {total} recorded, {stats.get('live', 0)} live")
    print()
    print("  outcomes")
    for name in OUTCOMES:
        print(f"    {name:12s} {stats.get(name, 0)}")
    print()
    print("  evidence")
    print(f"    tests passed {stats.get('verified_pass', 0)}")
    print(f"    tests failed {stats.get('verified_fail', 0)}")
    print(f"    unverified   {stats.get('unverified', 0)}")
    print()
    print("  signals")
    print(f"    recovered    {stats.get('recovered', 0)}  (failed, then found a working path)")
    print(f"    corrected    {stats.get('corrected', 0)}  (you pushed back afterwards)")
    print(f"    observations {stats.get('observations', 0)}  (total times these tasks were seen)")
    print(f"    confidence   {stats.get('avg_confidence', 0.0):.2f} avg")
    return 0


# ── list ────────────────────────────────────────────────────────────────


def _cmd_list(args) -> int:
    db = _open_db()
    try:
        rows = db.export_experiences()
    finally:
        db.close()

    if getattr(args, "workspace", None):
        want = str(args.workspace)
        if want == ".":
            want = _current_workspace()
        rows = [r for r in rows if str(r.get("workspace") or "") == want]
    if getattr(args, "outcome", None):
        rows = [r for r in rows if r.get("outcome") == args.outcome]
    if not getattr(args, "all", False):
        rows = [r for r in rows if not r.get("superseded")]

    limit = int(getattr(args, "limit", 20) or 20)
    rows = rows[:limit]

    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0

    if not rows:
        print("experience: no rows match")
        return 0

    print(f"  {'id':10s}  {'outcome':11s}  {'conf':>5s}  {'seen':>4s}  "
          f"{'evidence':10s}  {'age':>8s}  task")
    for r in rows:
        flag = " *" if r.get("superseded") else ""
        print(
            f"  {str(r.get('id', ''))[:8]:10s}  "
            f"{str(r.get('outcome', '')):11s}  "
            f"{float(r.get('confidence') or 0):5.2f}  "
            f"{int(r.get('observations') or 0):4d}  "
            f"{_truncate(r.get('verification') or '-', 10):10s}  "
            f"{_fmt_age(r.get('updated_at')):>8s}  "
            f"{_truncate(r.get('task'), 52)}{flag}"
        )
    if getattr(args, "all", False):
        print("\n  * superseded — corrected by you, no longer retrieved")
    return 0


# ── show ────────────────────────────────────────────────────────────────


def _resolve_id(db, prefix: str) -> Optional[Dict[str, Any]]:
    """Look up by full id, else by unique short prefix.

    Listings print 8-character ids, so a prefix has to work or every other
    command would need copy-paste of a 32-char hex string.
    """
    exact = db.get_experience(prefix)
    if exact:
        return exact
    matches = [r for r in db.export_experiences()
               if str(r.get("id", "")).startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"experience: '{prefix}' is ambiguous ({len(matches)} matches)",
              file=sys.stderr)
    return None


def _cmd_show(args) -> int:
    db = _open_db()
    try:
        row = _resolve_id(db, str(args.id))
    finally:
        db.close()

    if not row:
        print(f"experience: no row matching '{args.id}'", file=sys.stderr)
        return 1

    if getattr(args, "json", False):
        print(json.dumps(row, indent=2, ensure_ascii=False))
        return 0

    print(f"id           {row.get('id')}")
    print(f"task         {row.get('task')}")
    print(f"outcome      {row.get('outcome')}")
    print(f"verification {row.get('verification') or '-'}")
    print(f"confidence   {float(row.get('confidence') or 0):.4f}")
    print(f"observations {row.get('observations')}  "
          f"(success {row.get('success_count')}, failure {row.get('failure_count')})")
    print(f"corrections  {row.get('correction_count')}")
    print(f"superseded   {'yes' if row.get('superseded') else 'no'}")
    print(f"workspace    {row.get('workspace')}")
    print(f"updated      {_fmt_age(row.get('updated_at'))}")
    for label, key in (
        ("strategy", "strategy"),
        ("failure", "failure_reason"),
        ("recovery", "recovery"),
        ("correction", "user_correction"),
        ("exit reason", "exit_reason"),
    ):
        val = row.get(key)
        if val:
            print(f"{label:12s} {val}")
    tools = row.get("tools")
    if tools:
        print(f"{'tools':12s} {', '.join(tools) if isinstance(tools, list) else tools}")
    metrics = row.get("metrics")
    if metrics:
        print(f"{'metrics':12s} {json.dumps(metrics, ensure_ascii=False)}")
    return 0


# ── why ─────────────────────────────────────────────────────────────────


def _cmd_why(args) -> int:
    """Show exactly what a prompt would retrieve, and why.

    Runs the real scoring path — same candidate fetch, same ranking, same
    renderer the turn prologue uses — so the output is what the model would
    actually be handed, not an approximation of it.
    """
    from agent.experience import format_experience_block, rank_rows, tokenize
    from agent.experience_runtime import experience_config

    query = str(args.query)
    cfg = experience_config()
    workspace = str(getattr(args, "workspace", "") or "") or _current_workspace()

    db = _open_db()
    try:
        candidates = db.fetch_experience_candidates(
            workspace=workspace, max_age_days=float(cfg["max_age_days"])
        )
    finally:
        db.close()

    top = rank_rows(
        candidates,
        query,
        limit=int(cfg["max_results"]),
        min_score=float(cfg["min_score"]),
        max_age_days=float(cfg["max_age_days"]),
    )
    block = format_experience_block(top, max_chars=int(cfg["max_context_chars"]))

    if getattr(args, "json", False):
        print(json.dumps({
            "query": query,
            "workspace": workspace,
            "enabled": cfg["enabled"],
            "retrieval_enabled": cfg["retrieval_enabled"],
            "query_tokens": tokenize(query),
            "candidates": len(candidates),
            "matched": [
                {"id": r.get("id"), "score": round(r.get("_score", 0.0), 4),
                 "task": r.get("task"), "outcome": r.get("outcome")}
                for r in top
            ],
            "block": block,
        }, indent=2, ensure_ascii=False))
        return 0

    if not cfg["enabled"]:
        print("experience: DISABLED (experience.enabled=false) — nothing would be injected")
    elif not cfg["retrieval_enabled"]:
        print("experience: retrieval off (experience.retrieval_enabled=false) — "
              "outcomes are still recorded, nothing is injected")

    print(f"workspace   {workspace or '(none)'}")
    print(f"query terms {', '.join(tokenize(query)) or '(none — nothing to match on)'}")
    print(f"candidates  {len(candidates)} live rows considered")
    print(f"floor       {cfg['min_score']} (rows scoring below are dropped)")
    print()

    if not top:
        print("no match — this prompt would get NO injected context")
        return 0

    print(f"  {'score':>5s}  {'id':10s}  task")
    for r in top:
        print(f"  {r.get('_score', 0.0):5.3f}  {str(r.get('id', ''))[:8]:10s}  "
              f"{_truncate(r.get('task'), 60)}")
    print()
    print("would be injected into the API copy of the user message:")
    print()
    print(block)
    return 0


# ── forget / prune ──────────────────────────────────────────────────────


def _confirm(prompt: str, assume_yes: bool) -> bool:
    """Ask before deleting, unless --yes was passed.

    Non-interactive stdin (a pipe, CI) refuses rather than silently deleting:
    a scripted caller that meant it can say so with --yes.
    """
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print("experience: refusing to delete without --yes "
              "(stdin is not a terminal)", file=sys.stderr)
        return False
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def _cmd_forget(args) -> int:
    db = _open_db()
    try:
        if getattr(args, "all", False):
            total = db.experience_stats().get("total", 0)
            if not total:
                print("experience: nothing to forget")
                return 0
            if not _confirm(f"Delete all {total} experiences?",
                            getattr(args, "yes", False)):
                print("aborted")
                return 1
            removed = db.clear_experiences()
            print(f"experience: forgot {removed} rows")
            return 0

        if not getattr(args, "id", None):
            print("experience: give an id, or --all", file=sys.stderr)
            return 1

        row = _resolve_id(db, str(args.id))
        if not row:
            print(f"experience: no row matching '{args.id}'", file=sys.stderr)
            return 1
        if not _confirm(f"Forget {str(row['id'])[:8]} — {_truncate(row.get('task'), 60)}?",
                        getattr(args, "yes", False)):
            print("aborted")
            return 1
        removed = db.delete_experience(str(row["id"]))
        print(f"experience: forgot {str(row['id'])[:8]}" if removed
              else "experience: nothing deleted")
        return 0 if removed else 1
    finally:
        db.close()


def _cmd_prune(args) -> int:
    db = _open_db()
    try:
        before = db.experience_stats().get("total", 0)
        removed = db.prune_experiences(
            max_rows=int(getattr(args, "max_rows", 2000) or 2000),
            max_age_days=float(getattr(args, "max_age_days", 365) or 365),
        )
    finally:
        db.close()
    print(f"experience: pruned {removed} of {before} rows")
    return 0


# ── wiring ──────────────────────────────────────────────────────────────


def register_cli(parent: argparse.ArgumentParser) -> None:
    """Attach `experience` subcommands to *parent*.

    main.py calls this with the ArgumentParser returned by
    ``subparsers.add_parser("experience", ...)``.
    """
    parent.set_defaults(func=lambda a: (parent.print_help(), 0)[1])
    subs = parent.add_subparsers(dest="experience_command")

    p_stats = subs.add_parser("stats", help="Summary of what has been learned")
    p_stats.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    p_stats.set_defaults(func=_cmd_stats)

    p_list = subs.add_parser("list", aliases=["ls"], help="List stored experiences")
    p_list.add_argument("--workspace", metavar="PATH",
                        help="Only this project root; '.' resolves the current one")
    p_list.add_argument("--outcome", choices=OUTCOMES, help="Only this outcome")
    p_list.add_argument("--limit", type=int, default=20, help="Rows to show (default 20)")
    p_list.add_argument("--all", action="store_true",
                        help="Include superseded rows (hidden by default)")
    p_list.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    p_list.set_defaults(func=_cmd_list)

    p_show = subs.add_parser("show", help="Everything stored for one experience")
    p_show.add_argument("id", help="Full id or a unique prefix (list prints 8 chars)")
    p_show.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    p_show.set_defaults(func=_cmd_show)

    p_why = subs.add_parser(
        "why",
        help="Show what a given prompt would retrieve, and the score behind each row",
    )
    p_why.add_argument("query", help="The prompt to test retrieval against")
    p_why.add_argument("--workspace", metavar="PATH",
                       help="Score as if run from this project root (default: current)")
    p_why.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    p_why.set_defaults(func=_cmd_why)

    p_forget = subs.add_parser("forget", help="Delete one experience, or all of them")
    p_forget.add_argument("id", nargs="?", help="Full id or a unique prefix")
    p_forget.add_argument("--all", action="store_true", help="Delete every experience")
    p_forget.add_argument("--yes", "-y", action="store_true", help="Skip the confirmation")
    p_forget.set_defaults(func=_cmd_forget)

    p_prune = subs.add_parser("prune", help="Drop expired and surplus rows")
    p_prune.add_argument("--max-rows", type=int, default=2000, dest="max_rows",
                         help="Keep at most this many rows (default 2000)")
    p_prune.add_argument("--max-age-days", type=float, default=365, dest="max_age_days",
                         help="Drop rows older than this (default 365)")
    p_prune.set_defaults(func=_cmd_prune)
