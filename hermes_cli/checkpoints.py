"""`hermes checkpoints` CLI subcommand.

None of these require the agent to be running. Safe to call any time.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Optional

from hermes_cli.sizefmt import format_bytes as _fmt_bytes


def _fmt_age(ts: Any) -> str:
    try:
        age = time.time() - float(ts)
    except (TypeError, ValueError):
        return "—"
    if age < 0:
        return "now"
    for bound, div, unit in ((60, 1, "s"), (3600, 60, "m"), (86400, 3600, "h")):
        if age < bound:
            return f"{int(age / div)}{unit} ago"
    return f"{int(age / 86400)}d ago"


def cmd_status(args: argparse.Namespace) -> int:
    from tools.checkpoint_manager import store_status

    info = store_status()
    base = info["base"]
    print(f"Checkpoint base: {base}")
    print(f"Total size:      {_fmt_bytes(info['total_size_bytes'])}")
    print(f"  store/         {_fmt_bytes(info['store_size_bytes'])}")
    print(f"  legacy-*       {_fmt_bytes(info['legacy_size_bytes'])}")
    print(f"Projects:        {info['project_count']}")

    projects = sorted(info["projects"], key=lambda p: (p.get("last_touch") or 0), reverse=True)
    if projects:
        print()
        print(f"  {'WORKDIR':<60}  {'COMMITS':>7}  {'LAST TOUCH':>12}  STATE")
        for p in projects[: getattr(args, "limit", None) or 20]:
            wd = p.get("workdir") or "(unknown)"
            if len(wd) > 60:
                wd = "…" + wd[-59:]
            state = "live" if p.get("exists") else "orphan"
            print(f"  {wd:<60}  {p.get('commits', 0):>7}  {_fmt_age(p.get('last_touch')):>12}  {state}")

    legacy = info.get("legacy_archives", [])
    if legacy:
        print()
        print(f"Legacy archives ({len(legacy)}):")
        _print_archives(sorted(legacy, key=lambda a: a.get("mtime", 0), reverse=True))
        print()
        print("Clear with: hermes checkpoints clear-legacy")
    return 0


def _print_archives(archives) -> None:
    for arch in archives:
        print(f"  {arch['name']:<40}  {_fmt_bytes(arch['size_bytes']):>10}")


def cmd_prune(args: argparse.Namespace) -> int:
    from tools.checkpoint_manager import prune_checkpoints, store_status

    delete_orphans = not args.keep_orphans

    # Restricts orphan deletion to exactly the identities shown in the confirmation preview
    # (v2 project hashes / pre-v2 shadow repo paths). `None` = no restriction (--force: no
    # preview to bind to).
    orphan_allowlist: Optional[set] = None

    if delete_orphans and not args.force:
        info = store_status()
        orphans = [p for p in info.get("projects", []) if not p.get("exists")]
        pre_v2_orphans = [p for p in info.get("pre_v2_projects", []) if not p.get("exists")]
        if orphans or pre_v2_orphans:
            print(f"This will permanently delete {len(orphans) + len(pre_v2_orphans)} "
                  "orphan checkpoint project(s) whose workdir is not currently reachable:")
            print()
            for p in orphans:
                print(f"  {p.get('workdir') or '(unknown)'}  ({p.get('commits', 0)} commit(s))")
            for p in pre_v2_orphans:
                print(f"  {p.get('workdir') or '(unknown)'}  (pre-v2 shadow repo)")
            print()
            print("A workdir can be unreachable because the project was deleted,")
            print("or because an external volume / network share / VPN is down.")
            print("Pass --keep-orphans to prune stale entries only.")
            if not _confirm("Delete these orphan projects?"):
                print("Aborted.")
                return 1
        # Bind deletion to exactly what was displayed/confirmed: a project that goes orphan
        # only *after* the preview (workdir vanishes while waiting on input()) must not be
        # swept up. Set unconditionally for every non-force run — an EMPTY preview binds an
        # EMPTY allowlist, so it can never authorize orphans found by the later rescan.
        orphan_allowlist = {p["hash"] for p in orphans}
        orphan_allowlist.update(p["path"] for p in pre_v2_orphans)

    print("Pruning checkpoint store…")
    print(f"  retention_days:    {args.retention_days}")
    print(f"  delete_orphans:    {delete_orphans}")
    print(f"  max_total_size_mb: {args.max_size_mb}")
    print()

    result = prune_checkpoints(
        retention_days=args.retention_days,
        delete_orphans=delete_orphans,
        max_total_size_mb=args.max_size_mb,
        orphan_allowlist=orphan_allowlist)
    print(f"Scanned:         {result['scanned']}")
    print(f"Deleted orphan:  {result['deleted_orphan']}")
    print(f"Deleted stale:   {result['deleted_stale']}")
    print(f"Errors:          {result['errors']}")
    print(f"Bytes reclaimed: {_fmt_bytes(result['bytes_freed'])}")
    return 0


def _confirm(prompt: str) -> bool:
    try:
        return input(f"{prompt} [y/N]: ").strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def _confirmed(args: argparse.Namespace, prompt: str) -> bool:
    """``--force`` or an interactive yes; prints ``Aborted.`` otherwise."""
    if args.force or _confirm(prompt):
        return True
    print("Aborted.")
    return False


def cmd_clear(args: argparse.Namespace) -> int:
    from tools.checkpoint_manager import CHECKPOINT_BASE, clear_all, store_status

    info = store_status()
    if info["total_size_bytes"] == 0 and not Path(CHECKPOINT_BASE).exists():
        print("Nothing to clear — checkpoint base does not exist.")
        return 0

    print(f"This will delete the ENTIRE checkpoint base at {info['base']}")
    print(f"  size:        {_fmt_bytes(info['total_size_bytes'])}")
    print(f"  projects:    {info['project_count']}")
    print(f"  legacy dirs: {len(info.get('legacy_archives', []))}")
    print()
    print("All /rollback history for every working directory will be lost.")
    if not _confirmed(args, "Proceed?"):
        return 1

    result = clear_all()
    if result["deleted"]:
        print(f"Cleared. Reclaimed {_fmt_bytes(result['bytes_freed'])}.")
        return 0
    print("Could not clear checkpoint base (see logs).")
    return 2


def cmd_clear_legacy(args: argparse.Namespace) -> int:
    from tools.checkpoint_manager import clear_legacy, store_status

    info = store_status()
    legacy = info.get("legacy_archives", [])
    if not legacy:
        print("No legacy archives to clear.")
        return 0

    total = sum(a.get("size_bytes", 0) for a in legacy)
    print(f"Found {len(legacy)} legacy archive(s), total {_fmt_bytes(total)}:")
    _print_archives(legacy)
    print()
    print("Legacy archives hold pre-v2 per-project shadow repos, moved aside")
    print("during the single-store migration. Delete when you're confident")
    print("you don't need the old /rollback history.")
    if not _confirmed(args, "Delete all legacy archives?"):
        return 1

    result = clear_legacy()
    print(f"Deleted {result['deleted']} archive(s), reclaimed {_fmt_bytes(result['bytes_freed'])}.")
    return 0


def register_cli(parser: argparse.ArgumentParser) -> None:
    """Wire subcommands onto the ``hermes checkpoints`` parser."""
    parser.set_defaults(func=cmd_status)  # bare `hermes checkpoints` → status
    subs = parser.add_subparsers(dest="checkpoints_command", metavar="COMMAND")

    p_status = subs.add_parser("status", help="Show total size, project count, and per-project breakdown")
    p_status.add_argument("--limit", type=int, default=20, help="Max projects to list (default 20)")
    p_status.set_defaults(func=cmd_status)

    p_list = subs.add_parser("list", help="Alias for 'status'")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.set_defaults(func=cmd_status)

    p_prune = subs.add_parser("prune", help="Delete orphan/stale checkpoints and GC the store")
    p_prune.add_argument("--retention-days", type=int, default=7,
                         help="Drop projects whose last_touch is older than N days (default 7)")
    p_prune.add_argument("--max-size-mb", type=int, default=500,
                         help="After orphan/stale prune, drop oldest commits "
                              "per project until total size <= this (default 500)")
    p_prune.add_argument("--keep-orphans", action="store_true",
                         help="Skip deleting projects whose workdir no longer exists")
    p_prune.add_argument("-f", "--force", action="store_true",
                         help="Skip the orphan-deletion confirmation prompt")
    p_prune.set_defaults(func=cmd_prune)

    for name, help_text, func in (
        ("clear", "Delete the entire checkpoint base (all /rollback history)", cmd_clear),
        ("clear-legacy", "Delete only the legacy-<ts>/ archives from v1 migration", cmd_clear_legacy),
    ):
        p_clear = subs.add_parser(name, help=help_text)
        p_clear.add_argument("-f", "--force", action="store_true", help="Skip confirmation prompt")
        p_clear.set_defaults(func=func)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from datetime import datetime  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'cmd_list': ('hermes_cli.plugins_cmd', 'cmd_list'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
