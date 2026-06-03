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
    python3 install_caduceus.py --repack-only   # repack a prebuilt renderer into app.asar (no node)
    python3 install_caduceus.py --verify        # health-check an existing install and exit
    python3 install_caduceus.py --uninstall     # restore the most recent backup
    python3 install_caduceus.py --list-targets  # show detected Hermes installs

The CLI/TUI backend works immediately after install + a Hermes restart. The
desktop UI (status-bar toggle + Orchestration Theater) only changes after a
desktop rebuild (`--with-desktop`, or rebuild manually). The asar repack itself
is stdlib-only (no node/npx); only the renderer build step needs node/npm.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import struct
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
    "agent/local_manager.py",
    "agent/workflow/__init__.py",
    "agent/workflow/budget.py",
    "agent/workflow/dsl.py",
    "agent/workflow/engine.py",
    "agent/workflow/events.py",
    "agent/workflow/journal.py",
    "agent/workflow/local_gate.py",
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
    "apps/desktop/src/app/shell/caduceus-menu-panel.tsx",
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
    "docs/caduceus/RELEASE_NOTES.md",
    "docs/caduceus/PR_DESCRIPTION.md",
    "docs/caduceus/USER_GUIDE.md",
    "docs/caduceus/DESIGN.md",
    "docs/caduceus/IMPLEMENTATION.md",
    "docs/caduceus/LOCAL.md",
    "docs/caduceus/PARITY.md",
    "docs/caduceus/evidence/PLANNING_LOOP_CONTRACT.md",
    "docs/caduceus/eval/parity_eval.py",
    "docs/caduceus/eval/auto_router_selftest.py",
    "docs/caduceus/eval/ab_compare.py",
    "tests/caduceus/__init__.py",
    "tests/caduceus/test_caduceus_state.py",
    "tests/caduceus/test_auto_router.py",
    "tests/caduceus/test_route_worker_model.py",
    "tests/caduceus/test_local_mode.py",
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


def _valid_backups(target: str) -> list[str]:
    """Backup snapshot names (sorted oldest→newest) that have a restore manifest."""
    root = os.path.join(target, BACKUP_ROOT)
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root)
                  if os.path.exists(os.path.join(root, d, RESTORE_MANIFEST)))


def _files_differ(a: str, b: str) -> bool:
    try:
        with open(a, "rb") as fa, open(b, "rb") as fb:
            return fa.read() != fb.read()
    except OSError:
        return True


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

    missing_src = [p for p in MANIFEST if not os.path.exists(os.path.join(HERE, p))]
    if missing_src:
        err(f"Source checkout is missing {len(missing_src)} Caduceus file(s); "
            f"is this a complete Caduceus checkout? e.g. {missing_src[0]}")
        return 2

    # Idempotent re-install: if a prior backup exists, the OLDEST one holds the
    # true originals. Re-installing must NOT create a second backup (that would
    # capture already-Caduceus files as "originals" and break --uninstall).
    # Instead refresh changed files in place and keep the original backup.
    existing = _valid_backups(target)
    if existing and not dry_run:
        orig = os.path.join(target, BACKUP_ROOT, existing[0])
        info(f"Caduceus already installed (originals preserved at {orig}).")
        n = 0
        for rel in MANIFEST:
            src, dst = os.path.join(HERE, rel), os.path.join(target, rel)
            if os.path.exists(dst) and not _files_differ(src, dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            n += 1
        print()
        ok(f"Refreshed Caduceus: {n} file(s) updated, {len(MANIFEST) - n} already current "
           "(no new backup — the original is kept for a clean --uninstall).")
        return 0

    backup_dir = os.path.join(target, BACKUP_ROOT, _ts())
    restore = {"created": _ts(), "built_for": BUILT_FOR_VERSION, "entries": []}

    info(f"Target:  {target}")
    info(f"Source:  {HERE}")
    info(f"Backup:  {backup_dir}" + (" (dry-run, not created)" if dry_run else ""))
    print()

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
    backups = _valid_backups(target)
    if not backups:
        err(f"No valid backup snapshots (with {RESTORE_MANIFEST}) under {root}.")
        return 2
    # The OLDEST snapshot holds the true pre-Caduceus originals (a re-install
    # never shadows it), so restoring from it reverts cleanly to stock.
    oldest = os.path.join(root, backups[0])
    restore = json.load(open(os.path.join(oldest, RESTORE_MANIFEST), encoding="utf-8"))
    info(f"Restoring from {oldest}")
    restored = removed = 0
    for e in restore["entries"]:
        rel, existed = e["path"], e["existed"]
        dst = os.path.join(target, rel)
        if existed:
            bpath = os.path.join(oldest, rel)
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


def _find_packaged_asar(desktop: str) -> str | None:
    """Locate a packaged app.asar under apps/desktop/release/*/resources/."""
    release = os.path.join(desktop, "release")
    if not os.path.isdir(release):
        return None
    for entry in os.listdir(release):
        asar = os.path.join(release, entry, "resources", "app.asar")
        if os.path.exists(asar):
            return asar
    return None


# ---------------------------------------------------------------------------
# Pure-Python asar repack (no node/npx needed)
#
# asar's on-disk format is small and stable, so we read/write it with the
# standard library and recompute per-file SHA256 integrity (matching
# @electron/asar) so Electron's optional asar-integrity fuse stays satisfied.
# This makes --with-desktop work even where `node` is a Windows binary reached
# from WSL (which breaks the `npx` shim) or where the network is unavailable.
#
# Layout (little-endian): u32=4 | u32=len(headerBuf) | u32=4+align4(jsonLen) |
# u32=jsonLen | <JSON><pad-to-4> | <file data>. Each entry "offset" is relative
# to the end of the header.
# ---------------------------------------------------------------------------
_ASAR_BLOCK = 4 * 1024 * 1024  # 4 MiB, matches @electron/asar


def _asar_integrity(data: bytes) -> dict:
    blocks = [hashlib.sha256(data[i:i + _ASAR_BLOCK]).hexdigest()
              for i in range(0, len(data) or 1, _ASAR_BLOCK)] or [hashlib.sha256(b"").hexdigest()]
    return {"algorithm": "SHA256", "hash": hashlib.sha256(data).hexdigest(),
            "blockSize": _ASAR_BLOCK, "blocks": blocks}


def _asar_read(path: str):
    """Return (header_dict, data_offset, raw_bytes)."""
    with open(path, "rb") as f:
        raw = f.read()
    if struct.unpack_from("<I", raw, 0)[0] != 4:
        raise ValueError("not an asar archive (bad header magic)")
    header_len = struct.unpack_from("<I", raw, 4)[0]
    json_len = struct.unpack_from("<I", raw, 12)[0]
    header = json.loads(raw[16:16 + json_len].decode("utf-8"))
    return header, 8 + header_len, raw


def _asar_collect(node, prefix, raw, data_offset, out):
    if "files" in node:
        for name, child in node["files"].items():
            _asar_collect(child, prefix + "/" + name, raw, data_offset, out)
    else:
        off, size = int(node["offset"]), node["size"]
        out[prefix] = raw[data_offset + off: data_offset + off + size]


def _asar_insert(tree: dict, parts, leaf: dict):
    node = tree
    for p in parts[:-1]:
        node = node.setdefault(p, {"files": {}})["files"]
    node[parts[-1]] = leaf


def _asar_write(files: dict, out_path: str) -> str:
    """files: {"/a/b.js": b"..."} → write a complete asar to out_path + '.tmp'."""
    tree: dict = {}
    blobs, offset = [], 0
    for path, data in sorted(files.items()):
        _asar_insert(tree, path.strip("/").split("/"),
                     {"size": len(data), "offset": str(offset), "integrity": _asar_integrity(data)})
        blobs.append(data)
        offset += len(data)
    json_bytes = json.dumps({"files": tree}, separators=(",", ":")).encode("utf-8")
    aligned = (len(json_bytes) + 3) & ~3
    header_buf = (struct.pack("<I", 4 + aligned) + struct.pack("<I", len(json_bytes))
                  + json_bytes + b"\x00" * (aligned - len(json_bytes)))
    size_buf = struct.pack("<I", 4) + struct.pack("<I", len(header_buf))
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(size_buf)
        f.write(header_buf)
        for b in blobs:
            f.write(b)
    return tmp


def _asar_validate(tmp_path: str, new_dist: str) -> list:
    """Re-parse the produced asar; assert it is self-consistent + matches dist."""
    errors = []
    header, data_offset, raw = _asar_read(tmp_path)
    files: dict = {}
    _asar_collect(header, "", raw, data_offset, files)
    idx = os.path.join(new_dist, "index.html")
    if os.path.exists(idx) and files.get("/dist/index.html") != open(idx, "rb").read():
        errors.append("dist/index.html does not match the freshly-built renderer")

    def check(node, prefix=""):
        if "files" in node:
            for k, v in node["files"].items():
                check(v, prefix + "/" + k)
        else:
            blob = raw[data_offset + int(node["offset"]): data_offset + int(node["offset"]) + node["size"]]
            if len(blob) != node["size"]:
                errors.append(f"{prefix}: truncated blob")
            elif hashlib.sha256(blob).hexdigest() != node.get("integrity", {}).get("hash"):
                errors.append(f"{prefix}: integrity hash mismatch")
    check(header)
    return errors


def _repack_asar_python(asar: str, built_dist: str) -> bool:
    """Swap `built_dist` into the /dist subtree of `asar` using only stdlib.

    Builds + validates a fresh archive before atomically replacing the original
    (backed up to app.asar.precaduceus.bak once + a timestamped copy). Returns
    True on success; the running app must be CLOSED (it locks app.asar)."""
    header, data_offset, raw = _asar_read(asar)
    files: dict = {}
    _asar_collect(header, "", raw, data_offset, files)
    merged = {p: b for p, b in files.items() if not p.startswith("/dist/")}
    for root, _dirs, fnames in os.walk(built_dist):
        for fn in fnames:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, built_dist).replace(os.sep, "/")
            with open(full, "rb") as fh:
                merged["/dist/" + rel] = fh.read()
    tmp = _asar_write(merged, asar)
    errs = _asar_validate(tmp, built_dist)
    if errs:
        os.remove(tmp)
        warn("pure-Python asar repack produced an inconsistent archive: " + "; ".join(errs[:3]))
        return False
    bak = asar + ".precaduceus.bak"
    if not os.path.exists(bak):
        shutil.copy2(asar, bak)
    shutil.copy2(asar, asar + f".bak-{_ts()}")
    os.replace(tmp, asar)
    return True


def _repack_asar(asar: str, built_dist: str) -> bool:
    """Swap the freshly-built renderer `dist/` into a packaged app.asar.

    Prefers a self-contained, stdlib-only repack (no node/npx, no network).
    Falls back to the standard `@electron/asar` via npx if that ever fails.
    Backs up the original to app.asar.precaduceus.bak (once) + a timestamped
    copy. Returns True on success. Best-effort — the running app must be CLOSED
    (it holds a lock on app.asar), so this is for fresh installs / closed apps.
    """
    try:
        if _repack_asar_python(asar, built_dist):
            return True
    except OSError as e:
        warn(f"asar repack failed ({e}). The app may be running (it locks app.asar) — close it and retry.")
        return False
    except Exception as e:  # noqa: BLE001 — fall through to the npx path
        warn(f"pure-Python asar repack failed ({e}); trying @electron/asar via npx…")

    npx = shutil.which("npx")
    if not npx:
        warn("npx not found and the built-in repack failed; the renderer is built "
             f"at {built_dist} — repack it into {asar} with `@electron/asar`.")
        return False
    resources = os.path.dirname(asar)
    extracted = os.path.join(resources, "_caduceus_asar_extract")
    bak = asar + ".precaduceus.bak"
    try:
        shutil.rmtree(extracted, ignore_errors=True)
        subprocess.run([npx, "--yes", "@electron/asar", "extract", asar, extracted],
                       check=True, capture_output=True, text=True)
        dist_in = os.path.join(extracted, "dist")
        shutil.rmtree(dist_in, ignore_errors=True)
        shutil.copytree(built_dist, dist_in)
        if not os.path.exists(bak):
            shutil.copy2(asar, bak)
        shutil.copy2(asar, asar + f".bak-{_ts()}")
        subprocess.run([npx, "--yes", "@electron/asar", "pack", extracted, asar],
                       check=True, capture_output=True, text=True)
        shutil.rmtree(extracted, ignore_errors=True)
        return True
    except subprocess.CalledProcessError as e:
        warn(f"asar repack failed: {(e.stderr or e.stdout or e).strip()[:200]}")
        return False
    except OSError as e:
        warn(f"asar repack failed ({e}). The app may be running (it locks app.asar) — close it and retry.")
        return False


def _node_is_windows_exe() -> bool:
    """Detect the WSL trap where `node` resolves to a Windows node.exe — its
    Linux-path npm/npx shims can't load, so `npm run build` won't run. The
    pure-Python repack still works, so we route around it when a build exists."""
    node = shutil.which("node")
    if not node:
        return False
    try:
        return os.path.realpath(node).lower().endswith(".exe")
    except OSError:
        return node.lower().endswith(".exe")


def _dist_has_caduceus(built_dist: str) -> bool:
    """True iff the built renderer bundle actually contains the Caduceus UI.

    Guards against repacking a *stock* dist/ (built before the source overlay),
    which would silently pack a Caduceus-free UI and look like success."""
    assets = os.path.join(built_dist, "assets")
    if not os.path.isdir(assets):
        return False
    needles = (b"Orchestration Theater", b"Caduceus")
    for name in os.listdir(assets):
        if not name.endswith(".js"):
            continue
        try:
            with open(os.path.join(assets, name), "rb") as f:
                data = f.read()
        except OSError:
            continue
        if any(n in data for n in needles):
            return True
    return False


def _repack_existing(desktop: str, *, require_marker: bool = True) -> int:
    """Repack the already-built dist/ into the packaged app.asar (no node)."""
    asar = _find_packaged_asar(desktop)
    built_dist = os.path.join(desktop, "dist")
    if not os.path.exists(os.path.join(built_dist, "index.html")):
        err(f"No built renderer at {built_dist}. Build apps/desktop first "
            "(npm run build), or run --with-desktop.")
        return 1
    if require_marker and not _dist_has_caduceus(built_dist):
        err(f"The build at {built_dist} does not contain the Caduceus UI — it looks "
            "like a stock renderer. Rebuild from the overlaid source (--with-desktop) "
            "before repacking, so you don't pack a Caduceus-free UI.")
        return 1
    if not asar:
        warn("No packaged app.asar found (source checkout?). The renderer is built "
             f"at {built_dist}; nothing to repack.")
        return 0
    info("Repacking app.asar with the built renderer (stdlib-only, no node)…")
    info("(close the Hermes desktop first — a running app locks app.asar.)")
    if _repack_asar(asar, built_dist):
        ok("Repacked app.asar (original saved to app.asar.precaduceus.bak).")
        return 0
    warn("Could not repack — close the Hermes desktop and retry, or repack manually.")
    return 1


def rebuild_desktop(target: str, repack_only: bool = False) -> int:
    desktop = os.path.join(target, "apps", "desktop")
    if not os.path.isdir(desktop):
        warn("No apps/desktop in target; skipping desktop rebuild (backend-only install).")
        return 0
    if repack_only:
        return _repack_existing(desktop)

    npm = shutil.which("npm")
    built_dist = os.path.join(desktop, "dist")
    if not npm or _node_is_windows_exe():
        why = ("npm not found on PATH" if not npm
               else "`node` resolves to a Windows binary (its WSL npm shim can't build)")
        if _dist_has_caduceus(built_dist):
            warn(f"Skipping the renderer build ({why}); the existing build already "
                 "contains the Caduceus UI — repacking it (node-free).")
            return _repack_existing(desktop)
        warn(f"Can't build the desktop renderer here ({why}). The CLI/TUI works now. "
             "Build apps/desktop where Node works (npm run build), then re-run this "
             "installer with --repack-only.")
        return 0

    info("Rebuilding the Electron desktop renderer (npm run build)…")
    try:
        subprocess.run([npm, "run", "build"], cwd=desktop, check=True)
    except subprocess.CalledProcessError as e:
        if _dist_has_caduceus(built_dist):
            warn(f"Desktop build failed ({e}); repacking the existing Caduceus build instead.")
            return _repack_existing(desktop)
        err(f"Desktop build failed ({e}). The backend still works; see "
            "docs/caduceus/IMPLEMENTATION.md to finish the desktop manually.")
        return 1
    ok("Desktop renderer built.")
    return _repack_existing(desktop)


def do_verify(target: str) -> int:
    """Post-install health check: files present, modules compile, wiring in
    place, and (if packaged) the desktop asar carries the Caduceus UI."""
    import py_compile
    if not is_hermes_install(target):
        err(f"Not a Hermes install (missing run_agent.py/cli.py/toolsets.py): {target}")
        return 2
    info(f"Verifying Caduceus at {target}\n")
    problems = 0

    missing = [p for p in MANIFEST if not os.path.exists(os.path.join(target, p))]
    if missing:
        problems += 1
        err(f"{len(missing)} Caduceus file(s) missing (e.g. {missing[0]}). Re-run the installer.")
    else:
        ok(f"All {len(MANIFEST)} Caduceus files present.")

    py_files = [p for p in MANIFEST if p.endswith(".py") and os.path.exists(os.path.join(target, p))]
    bad = []
    for rel in py_files:
        try:
            py_compile.compile(os.path.join(target, rel), doraise=True)
        except py_compile.PyCompileError:
            bad.append(rel)
    if bad:
        problems += 1
        err(f"{len(bad)} Python module(s) failed to compile (e.g. {bad[0]}).")
    else:
        ok(f"All {len(py_files)} Caduceus Python modules compile.")

    def _contains(rel, needle):
        try:
            with open(os.path.join(target, rel), encoding="utf-8", errors="ignore") as f:
                return needle in f.read()
        except OSError:
            return False
    if _contains("hermes_cli/commands.py", "caduceus") and _contains("toolsets.py", "Workflow"):
        ok("Wiring present (/caduceus command + Workflow toolset).")
    else:
        problems += 1
        err("Caduceus wiring missing in hermes_cli/commands.py or toolsets.py.")

    desktop = os.path.join(target, "apps", "desktop")
    asar = _find_packaged_asar(desktop) if os.path.isdir(desktop) else None
    if asar:
        try:
            with open(asar, "rb") as f:
                blob = f.read()
            if b"Orchestration Theater" in blob or b"Caduceus" in blob:
                ok("Packaged app.asar contains the Caduceus desktop UI.")
            else:
                problems += 1
                warn("Packaged app.asar does NOT contain the Caduceus UI. Close Hermes "
                     "and run --with-desktop (or --repack-only after building).")
        except OSError as e:
            warn(f"Could not read {asar}: {e}")
    else:
        info("No packaged desktop app found (CLI/TUI install) — skipping the desktop check.")

    print()
    if problems:
        err(f"Verify found {problems} problem(s) — see above.")
        return 1
    ok("Caduceus verify passed. Restart Hermes and run /caduceus on.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Caduceus 1-click installer for Hermes")
    ap.add_argument("target", nargs="?", help="path to the hermes-agent install (auto-detected if omitted)")
    ap.add_argument("--dry-run", action="store_true", help="show changes, write nothing")
    ap.add_argument("--with-desktop", action="store_true", help="also rebuild the Electron desktop")
    ap.add_argument("--repack-only", action="store_true",
                    help="skip the overlay+build; just repack the already-built renderer into app.asar (node-free)")
    ap.add_argument("--verify", action="store_true", help="check an existing Caduceus install is healthy and exit")
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
    if args.verify:
        return do_verify(target)
    if args.repack_only:
        return rebuild_desktop(target, repack_only=True)

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
