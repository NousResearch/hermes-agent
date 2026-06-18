"""``hermes skill-stats`` subcommand — surface usage_report() telemetry.

Uses ``usage_report()`` from ``tools/skill_usage.py`` and reuses the
``_fmt_ts()`` helper from ``hermes_cli.curator`` for human-readable timestamps.

Table format mirrors the curator status output style.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _fmt_ts(ts: Optional[str]) -> str:
    """Convert an ISO timestamp to a human-readable relative time string.

    Copied from ``hermes_cli.curator._fmt_ts`` so the skill-stats command
    works without importing the curator module (which pulls in the full
    curator state machine).
    """
    if not ts:
        return "never"
    try:
        dt = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return str(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - dt
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def _cmd_skill_stats(args) -> int:
    """Display a table of all skills with their usage telemetry."""
    from tools.skill_usage import usage_report

    rows: List[Dict[str, Any]] = usage_report()

    if not rows:
        print("skill-stats: no skills found")
        return 0

    # --- summary header -------------------------------------------------------
    total = len(rows)
    by_prov: Dict[str, int] = {}
    for r in rows:
        by_prov[r.get("provenance", "agent")] = by_prov.get(r.get("provenance", "agent"), 0) + 1

    prov_parts = ", ".join(f"{v} {k}" for k, v in sorted(by_prov.items()))
    print(f"skill-stats: {total} skill(s) on disk  [{prov_parts}]")

    total_uses = sum(r.get("use_count", 0) for r in rows)
    total_views = sum(r.get("view_count", 0) for r in rows)
    total_patches = sum(r.get("patch_count", 0) for r in rows)
    print(
        f"             total: use={total_uses}  view={total_views}  patch={total_patches}"
    )

    # --- table ----------------------------------------------------------------
    # Header
    print()
    print(
        f"  {'NAME':<38} {'PROVENANCE':<10} {'USE':>5} {'VIEW':>5} "
        f"{'PATCH':>5} {'STATE':<9} {'LAST ACTIVITY':<12}"
    )
    print(
        f"  {'-'*38} {'-'*10} {'-'*5} {'-'*5} {'-'*5} "
        f"{'-'*9} {'-'*12}"
    )

    # Sort by name for stable output
    for r in sorted(rows, key=lambda x: x.get("name", "")):
        name = r.get("name", "?")
        prov = r.get("provenance", "?")
        use = r.get("use_count", 0)
        view = r.get("view_count", 0)
        patch = r.get("patch_count", 0)
        state = r.get("state", "active")
        last = _fmt_ts(r.get("last_activity_at"))

        print(
            f"  {name:<38} {prov:<10} {use:>5} {view:>5} "
            f"{patch:>5} {state:<9} {last:<12}"
        )

    return 0


# ---------------------------------------------------------------------------
# argparse wiring (called from hermes_cli.main)
# ---------------------------------------------------------------------------

def build_skill_stats_parser(subparsers, *, cmd_skills_stats: Any = None) -> None:
    """Attach the ``skill-stats`` subcommand to ``subparsers``.

    ``cmd_skills_stats`` is an optional callable handler. When omitted the
    module-level ``_cmd_skill_stats`` is used directly.
    """
    handler = cmd_skills_stats or _cmd_skill_stats

    skill_stats_parser = subparsers.add_parser(
        "skill-stats",
        help="Show per-skill usage telemetry (use/view/patch counts and last activity)",
        description=(
            "Print a table of all installed skills with their usage telemetry "
            "from ``~/.hermes/skills/.usage.json``. Covers every skill on disk "
            "regardless of provenance (agent-created, bundled, or hub-installed). "
            "Use this to answer \"how often is this skill used?\" independent "
            "of whether the curator manages it."
        ),
    )
    skill_stats_parser.set_defaults(func=handler)


def cli_main(argv=None) -> int:
    """Standalone entry point for ``python -m hermes_cli.subcommands.skill_stats``."""
    parser = argparse.ArgumentParser(prog="hermes skill-stats")
    build_skill_stats_parser(parser)
    args = parser.parse_args(argv)
    fn = getattr(args, "func", None)
    if fn is None:
        parser.print_help()
        return 0
    return int(fn(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(cli_main())
