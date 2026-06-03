#!/usr/bin/env python3
"""Caduceus 1-click installer — add Caduceus to an existing Hermes install.

Caduceus is a native deep-planning mode + dynamic multi-agent workflow engine
("the Loom") + per-task Auto Router for Hermes. This script overlays the
Caduceus files from THIS checkout onto a target `hermes-agent` install, backing
up every file it touches so the change is fully reversible.

It is **safe**: it backs up before writing, writes a restore manifest, never
deletes anything it didn't add, leaves Caduceus OFF by default (opt-in via
`/caduceus on`), and is idempotent. Pure standard library — nothing to pip.

Usage:
    python3 install_caduceus.py [TARGET]        # install onto TARGET (auto-detect if omitted)
    python3 install_caduceus.py --dry-run       # show what would change, do nothing
    python3 install_caduceus.py --with-desktop  # also rebuild the Electron desktop (needs node/npm)
    python3 install_caduceus.py --uninstall     # restore the most recent backup
    python3 install_caduceus.py --list-targets  # show detected Hermes installs

The CLI/TUI backend works immediately after install + a Hermes restart. The
desktop UI (status-bar toggle + Orchestration Theater) only changes after a
desktop rebuild (`--with-desktop`, or rebuild manually).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys

# Hermes version this Caduceus build targets. Overlaying onto a substantially
# different version can clobber that version's changes to the shared core files
# (cli.py, run_agent.py, ...). We warn (not block) since every write is backed up.
BUILT_FOR_VERSION = "0.15.1"
BASE_COMMIT = "b34ee8074"

HERE = os.path.dirname(os.path.abspath(__file__))

# The Caduceus change set (relative paths). New modules + modified core files +
# desktop + docs + tests. Generated from `git diff --name-only <base>..caduceus`.
MANIFEST = [
    # New, isolated modules (the bulk of the feature)
    "agent/caduceus.py",
    "agent/auto_router.py",
    "agent/workflow/__init__.py",
    "agent/workflow/budget.py",
    "agent/workflow/dsl.py",
    "agent/workflow/engine.py",
    "agent/workflow/events.py",
    "agent/workflow/journal.py",
    "agent/workflow/reliability.py",
    "agent/workflow/runner.py",
    "agent/workflow/sandbox.py",
    "agent/workflow/scheduler.py",
    "agent/workflow/structured.py",
    "tools/workflow_tool.py",
    # Modified core (additive, guarded — see docs/caduceus/PR_DESCRIPTION.md)
    "agent/agent_init.py",
    "agent/agent_runtime_helpers.py",
    "agent/conversation_loop.py",
    "agent/system_prompt.py",
    "agent/tool_executor.py",
    "cli.py",
    "gateway/run.py",
    "hermes_cli/commands.py",
    "hermes_cli/config.py",
    "model_tools.py",
    "run_agent.py",
    "toolsets.py",
    "tools/delegate_tool.py",
    "tui_gateway/server.py",
    # Desktop (renderer source; needs a rebuild to take effect)
    "apps/desktop/src/app/session/hooks/use-message-stream.ts",
    "apps/desktop/src/app/settings/constants.ts",
    "apps/desktop/src/app/shell/app-shell.tsx",
    "apps/desktop/src/app/shell/hooks/use-statusbar-items.tsx",
    "apps/desktop/src/components/workflow/AgentCard.tsx",
    "apps/desktop/src/components/workflow/TheaterPanels.tsx",
    "apps/desktop/src/components/workflow/WorkflowTheater.tsx",
    "apps/desktop/src/components/workflow/theater-format.ts",
    "apps/desktop/src/store/caduceus.ts",
    "apps/desktop/src/store/workflow.ts",
    # Docs + tests (harmless; nice to ship with the feature)
    "docs/caduceus/README.md",
    "docs/caduceus/INSTALL.md",
    "docs/caduceus/PR_DESCRIPTION.md",
    "docs/caduceus/USER_GUIDE.md",
    "docs/caduceus/DESIGN.md",
    "docs/caduceus/IMPLEMENTATION.md",
    "docs/caduceus/PARITY.md",
    "docs/caduceus/evidence/PLANNING_LOOP_CONTRACT.md",
    "docs/caduceus/eval/parity_eval.py",
    "docs/caduceus/eval/auto_router_selftest.py",
    "docs/caduceus/eval/ab_compare.py",
    "tests/caduceus/__init__.py",
    "tests/caduceus/test_caduceus_state.py",
    "tests/caduceus/test_auto_router.py",
    "tests/caduceus/test_route_worker_model.py",
    "tests/workflow/__init__.py",
    "tests/workflow/test_loom_offline.py",
]

# Files that exist in stock Hermes (modified, not new). Used only for messaging.
_MODIFIED_CORE = {
    "agent/agent_init.py", "agent/agent_runtime_helpers.py", "agent/conversation_loop.py",
    "agent/system_prompt.py", "agent/tool_executor.py", "cli.py", "gateway/run.py",
    "hermes_cli/commands.py", "hermes_cli/config.py", "model_tools.py", "run_agent.py",
    "toolsets.py", "tools/delegate_tool.py", "tui_gateway/server.py",
    "apps/desktop/src/app/session/hooks/use-message-stream.ts",
    "apps/desktop/src/app/settings/constants.ts", "apps/desktop/src/app/shell/app-shell.tsx",
    "apps/desktop/src/app/shell/hooks/use-statusbar-items.tsx",
    "apps/desktop/src/store/workflow.ts",
}

BACKUP_ROOT = ".caduceus-backups"
RESTORE_MANIFEST = "restore-manifest.json"


# ---------------------------------------------------------------------------
# Colors (no-op when not a TTY)
# ---------------------------------------------------------------------------
def _c(code: str) -> str:
    return code if sys.stdout.isatty() else ""


BOLD, DIM, GRN, YEL, RED, CYN, RST = (
    _c("\033[1m"), _c("\033[2m"), _c("\033[32m"), _c("\033[33m"),
    _c("\033[31m"), _c("\033[36m"), _c("\033[0m"),
)


def info(msg): print(f"{CYN}•{RST} {msg}")
def ok(msg): print(f"{GRN}✓{RST} {msg}")
def warn(msg): print(f"{YEL}!{RST} {msg}")
def err(msg): print(f"{RED}✗{RST} {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Target detection / validation
# ---------------------------------------------------------------------------
def candidate_targets() -> list[str]:
    home = os.path.expanduser("~")
    cands = [
        os.environ.get("HERMES_AGENT_DIR", ""),
        os.path.join(home, ".hermes", "hermes-agent"),
        os.path.join(home, ".local", "share", "hermes", "hermes-agent"),
        "/opt/hermes/hermes-agent",
    ]
    # Windows desktop install (also reachable from WSL via /mnt/c)
    local = os.environ.get("LOCALAPPDATA")
    if local:
        cands.append(os.path.join(local, "hermes", "hermes-agent"))
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "User"
    cands.append(f"/mnt/c/Users/{user}/AppData/Local/hermes/hermes-agent")
    # pip-installed package location
    try:
        import importlib.util
        spec = importlib.util.find_spec("run_agent")
        if spec and spec.origin:
            cands.append(os.path.dirname(spec.origin))
    except Exception:
        pass
    seen, out = set(), []
    for c in cands:
        if c and c not in seen and is_hermes_install(c):
            seen.add(c)
            out.append(c)
    return out


def is_hermes_install(path: str) -> bool:
    return all(os.path.exists(os.path.join(path, f)) for f in ("run_agent.py", "cli.py", "toolsets.py"))


def read_version(path: str) -> str | None:
    # Best-effort: pyproject.toml [project] version, else a VERSION file.
    pp = os.path.join(path, "pyproject.toml")
    if os.path.exists(pp):
        try:
            with open(pp, encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s.startswith("version") and "=" in s:
                        return s.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    for vf in ("VERSION", "version.txt"):
        p = os.path.join(path, vf)
        if os.path.exists(p):
            try:
                return open(p, encoding="utf-8").read().strip()
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------
def _ts() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def do_install(target: str, dry_run: bool, force: bool) -> int:
    if HERE == os.path.abspath(target):
        err("Source and target are the same directory. Run the installer from a "
            "Caduceus checkout against a *separate* Hermes install.")
        return 2
    if not is_hermes_install(target):
        err(f"Not a Hermes install (missing run_agent.py/cli.py/toolsets.py): {target}")
        return 2

    # Version guard (warn, don't block — every write is backed up).
    tv = read_version(target)
    if tv and tv != BUILT_FOR_VERSION and not force:
        warn(f"Target Hermes version is {tv}; this Caduceus build targets "
             f"{BUILT_FOR_VERSION} (base {BASE_COMMIT}).")
        warn("Overlaying onto a different version may clobber that version's "
             "edits to shared core files. Re-run with --force to proceed anyway "
             "(everything is backed up and reversible with --uninstall).")
        return 3

    backup_dir = os.path.join(target, BACKUP_ROOT, _ts())
    restore = {"created": _ts(), "built_for": BUILT_FOR_VERSION, "entries": []}

    info(f"Target:  {target}")
    info(f"Source:  {HERE}")
    info(f"Backup:  {backup_dir}" + (" (dry-run, not created)" if dry_run else ""))
    print()

    missing_src = [p for p in MANIFEST if not os.path.exists(os.path.join(HERE, p))]
    if missing_src:
        err(f"Source checkout is missing {len(missing_src)} Caduceus file(s); "
            f"is this a complete Caduceus checkout? e.g. {missing_src[0]}")
        return 2

    n_new = n_mod = 0
    for rel in MANIFEST:
        src = os.path.join(HERE, rel)
        dst = os.path.join(target, rel)
        existed = os.path.exists(dst)
        action = "update" if existed else "add"
        if existed:
            n_mod += 1
        else:
            n_new += 1
        tag = f"{YEL}{action}{RST}" if existed else f"{GRN}{action}{RST}"
        print(f"  {tag:>14}  {rel}")
        if dry_run:
            continue
        # Back up the original (if any) before overwriting.
        if existed:
            bpath = os.path.join(backup_dir, rel)
            os.makedirs(os.path.dirname(bpath), exist_ok=True)
            shutil.copy2(dst, bpath)
        restore["entries"].append({"path": rel, "existed": existed})
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    if dry_run:
        print()
        info(f"Dry run: would add {n_new} new file(s), update {n_mod} existing file(s). Nothing written.")
        return 0

    os.makedirs(backup_dir, exist_ok=True)
    with open(os.path.join(backup_dir, RESTORE_MANIFEST), "w", encoding="utf-8") as f:
        json.dump(restore, f, indent=2)

    print()
    ok(f"Installed Caduceus: {n_new} new file(s), {n_mod} updated (originals backed up).")
    return 0


def do_uninstall(target: str) -> int:
    root = os.path.join(target, BACKUP_ROOT)
    if not os.path.isdir(root):
        err(f"No Caduceus backups found under {root}; nothing to uninstall.")
        return 2
    backups = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if not backups:
        err("No backup snapshots found.")
        return 2
    latest = os.path.join(root, backups[-1])
    mf = os.path.join(latest, RESTORE_MANIFEST)
    if not os.path.exists(mf):
        err(f"Backup {latest} has no {RESTORE_MANIFEST}; cannot safely restore.")
        return 2
    restore = json.load(open(mf, encoding="utf-8"))
    info(f"Restoring from {latest}")
    restored = removed = 0
    for e in restore["entries"]:
        rel, existed = e["path"], e["existed"]
        dst = os.path.join(target, rel)
        if existed:
            bpath = os.path.join(latest, rel)
            if os.path.exists(bpath):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(bpath, dst)
                restored += 1
        else:
            # Caduceus-added file — remove it.
            if os.path.exists(dst):
                os.remove(dst)
                removed += 1
    ok(f"Uninstalled Caduceus: restored {restored} original file(s), removed {removed} added file(s).")
    info("Restart Hermes (and rebuild the desktop if you rebuilt it for Caduceus).")
    return 0


def rebuild_desktop(target: str) -> int:
    desktop = os.path.join(target, "apps", "desktop")
    if not os.path.isdir(desktop):
        warn("No apps/desktop in target; skipping desktop rebuild (backend-only install).")
        return 0
    npm = shutil.which("npm")
    if not npm:
        warn("npm not found on PATH; skipping desktop rebuild. The CLI/TUI works "
             "now; rebuild the desktop manually to get the status-bar toggle + Theater.")
        return 0
    info("Rebuilding the Electron desktop (npm run build)…")
    try:
        subprocess.run([npm, "run", "build"], cwd=desktop, check=True)
    except subprocess.CalledProcessError as e:
        err(f"Desktop build failed ({e}). The backend still works; see "
            "docs/caduceus/IMPLEMENTATION.md to finish the desktop manually.")
        return 1
    ok("Desktop rebuilt. (If you run the packaged app, repack app.asar — see IMPLEMENTATION.md.)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Caduceus 1-click installer for Hermes")
    ap.add_argument("target", nargs="?", help="path to the hermes-agent install (auto-detected if omitted)")
    ap.add_argument("--dry-run", action="store_true", help="show changes, write nothing")
    ap.add_argument("--with-desktop", action="store_true", help="also rebuild the Electron desktop")
    ap.add_argument("--uninstall", action="store_true", help="restore the most recent backup")
    ap.add_argument("--list-targets", action="store_true", help="list detected Hermes installs and exit")
    ap.add_argument("--force", action="store_true", help="proceed despite a version mismatch")
    args = ap.parse_args()

    print(f"{BOLD}⚕ Caduceus installer{RST}  {DIM}(built for Hermes {BUILT_FOR_VERSION}){RST}\n")

    detected = candidate_targets()
    if args.list_targets:
        if detected:
            info("Detected Hermes install(s):")
            for d in detected:
                print(f"    {d}   {DIM}(v{read_version(d) or '?'}){RST}")
        else:
            warn("No Hermes install auto-detected. Pass the path explicitly.")
        return 0

    target = args.target or (detected[0] if detected else None)
    if not target:
        err("No Hermes install found. Pass the path: python3 install_caduceus.py /path/to/hermes-agent")
        if detected:
            info("Detected: " + ", ".join(detected))
        return 2
    target = os.path.abspath(os.path.expanduser(target))

    if args.uninstall:
        return do_uninstall(target)

    rc = do_install(target, args.dry_run, args.force)
    if rc != 0 or args.dry_run:
        return rc

    if args.with_desktop:
        rebuild_desktop(target)
    else:
        info("Desktop UI unchanged (backend-only). Re-run with --with-desktop to "
             "rebuild the status-bar toggle + Orchestration Theater.")

    print()
    ok("Done. Next steps:")
    print(f"    1. Restart Hermes.")
    print(f"    2. Run {BOLD}/caduceus on{RST} to enable deep-planning mode.")
    print(f"    3. Say \"workflow\" on a big task to fan out on the Loom.")
    print(f"    {DIM}Uninstall anytime: python3 install_caduceus.py --uninstall{RST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
