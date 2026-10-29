#!/usr/bin/env python3
"""Headless cross-platform pipeline: board -> DSN -> Freerouting -> SES -> DRC.

A single run executes all 4 stages and reports DRC PASS/FAIL.
Works on Linux, Windows and macOS: if `import pcbnew` fails in the current
interpreter, the script re-execs itself in KiCad's bundled Python (Windows/macOS).

Usage:
  python3 demo_autoroute.py                  # full pipeline through DRC
  python3 demo_autoroute.py --stage create   # board + DSN only
  python3 demo_autoroute.py --stage route    # autorouting only (needs the .dsn)
  python3 demo_autoroute.py --stage import   # import the SES only
  python3 demo_autoroute.py --stage drc      # DRC of an existing board
  python3 demo_autoroute.py --out /tmp/pcb   # write artifacts to another dir

Env vars: KICAD_PYTHON, KICAD_CLI, KICAD10_FOOTPRINT_DIR|KICAD_FOOTPRINT_DIR,
FREEROUTING_EXE, FREEROUTING_JAR (see kicad_paths.py for the full order).
Exit codes: 0 ok | 2 environment/config | 3 stage failed | 4 DRC failures.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kicad_paths as kp  # noqa: E402

BOARD = "demo-unrouted.kicad_pcb"
DSN, SES = "demo.dsn", "demo.ses"
ROUTED = "demo-routed.kicad_pcb"
DRC = "demo-drc.json"


def _import_pcbnew():
    """Imports pcbnew; if unavailable, re-execs in KiCad's Python (Windows/macOS)."""
    with contextlib.redirect_stderr(io.StringIO()):  # PROPERTY_ENUM asserts = harmless noise
        try:
            import pcbnew  # noqa: F401
            return pcbnew
        except ImportError:
            pass
    if os.environ.get("_KICAD_PCB_REEXEC"):  # already tried once — do not loop
        sys.exit("ERROR: pcbnew not importable even in the pointed KiCad Python. "
                 "Check the KiCad install or set KICAD_PYTHON.")
    for py in kp.python_with_pcbnew_candidates():
        if py == Path(sys.executable) or not py.exists():
            continue
        if kp.verify_pcbnew(py):
            env = dict(os.environ, _KICAD_PCB_REEXEC="1", KICAD_PYTHON=str(py))
            print(f"[shim] re-executing in KiCad's Python: {py}")
            r = subprocess.run([str(py), str(Path(__file__).resolve())] + sys.argv[1:], env=env)
            sys.exit(r.returncode)
    sys.exit("ERROR: no Python with pcbnew bindings found. "
             "Install KiCad 10 (includes the bindings) or set KICAD_PYTHON.")


# ------------------------------------------------------------------ stages

def stage_create(pcbnew, outdir: Path) -> None:
    mm = pcbnew.FromMM
    board = pcbnew.NewBoard(str(outdir / BOARD))  # filename arg REQUIRED on KiCad 10

    edge = pcbnew.PCB_SHAPE(board)
    edge.SetShape(pcbnew.SHAPE_T_RECT)
    edge.SetStart(pcbnew.VECTOR2I(mm(0), mm(0)))
    edge.SetEnd(pcbnew.VECTOR2I(mm(30), mm(20)))
    edge.SetLayer(pcbnew.Edge_Cuts)
    edge.SetWidth(mm(0.1))
    board.Add(edge)

    libs = kp.footprints_dir()
    parts = []
    for lib, name, ref, x, y in [
        ("Resistor_SMD", "R_0603_1608Metric", "R1", 8, 10),
        ("LED_SMD", "LED_0603_1608Metric", "D1", 22, 10),
    ]:
        f = pcbnew.FootprintLoad(str(libs / f"{lib}.pretty"), name)
        if not f:
            sys.exit(f"ERROR: footprint {lib}:{name} not found in {libs}")
        f.SetReference(ref)
        f.SetPosition(pcbnew.VECTOR2I(mm(x), mm(y)))
        board.Add(f)
        parts.append(f)

    net = pcbnew.NETINFO_ITEM(board, "N$1")
    board.Add(net)
    parts[0].FindPadByNumber("2").SetNet(net)
    parts[1].FindPadByNumber("1").SetNet(net)
    board.Save(str(outdir / BOARD))
    print(f"[create] board saved: {outdir / BOARD}")

    if not pcbnew.ExportSpecctraDSN(board, str(outdir / DSN)):
        sys.exit("ERROR: ExportSpecctraDSN failed")
    print(f"[create] DSN exported: {outdir / DSN}")


def stage_route(outdir: Path, timeout: int, max_passes: int, threads: int) -> None:
    mode, argv = kp.freerouting()
    cmd = argv + ["-de", str(outdir / DSN), "-do", str(outdir / SES),
                  "-mp", str(max_passes), "-mt", str(threads)]
    print(f"[route] freerouting ({mode}): {' '.join(cmd[:2])} ... -de {DSN} -do {SES}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    tail = "\n".join((r.stdout or "").strip().splitlines()[-3:]
                     + (r.stderr or "").strip().splitlines()[-2:])
    print(tail or f"(exit {r.returncode})")
    if r.returncode != 0 or not (outdir / SES).exists():
        sys.exit(f"ERROR: freerouting exit={r.returncode} (ses exists: {(outdir / SES).exists()})")
    print(f"[route] SES written: {outdir / SES}")


def stage_import(pcbnew, outdir: Path) -> None:
    board = pcbnew.LoadBoard(str(outdir / BOARD))
    if not pcbnew.ImportSpecctraSES(board, str(outdir / SES)):
        sys.exit("ERROR: ImportSpecctraSES failed")
    board.Save(str(outdir / ROUTED))
    n = len(board.GetTracks())
    print(f"[import] {n} track(s) | saved: {outdir / ROUTED}")


def stage_drc(outdir: Path) -> int:
    cli = kp.kicad_cli()
    r = subprocess.run([str(cli), "pcb", "drc", "--format", "json",
                        "--output", str(outdir / DRC), str(outdir / ROUTED)],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0 and not (outdir / DRC).exists():
        sys.exit(f"ERROR: kicad-cli drc exit={r.returncode}: {r.stderr.strip()[:400]}")
    rep = json.loads((outdir / DRC).read_text(encoding="utf-8"))
    errors = [v for v in rep.get("violations", []) if v.get("severity") == "error"]
    warnings = [v for v in rep.get("violations", []) if v.get("severity") != "error"]
    unconn = rep.get("unconnected_items", [])
    parity = rep.get("schematic_parity", [])
    print(f"[drc] violations: {len(errors)} error(s), {len(warnings)} warning(s) | "
          f"unconnected: {len(unconn)} | parity: {len(parity)}")
    for v in (errors + unconn + parity)[:10]:
        print(f"   - {v.get('type', '?')}: {str(v.get('description', ''))[:100]}")
    for v in warnings[:10]:
        print(f"   ~ warning {v.get('type', '?')}: {str(v.get('description', ''))[:100]}")
    if errors or warnings or unconn or parity:
        print("DRC: FAIL")
        return 4
    print("DRC: PASS (0 errors, 0 warnings, 0 unconnected, 0 parity; engineering review still required)")
    return 0


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["create", "route", "import", "drc", "all"], default="all")
    ap.add_argument("--out", type=Path, default=Path.cwd(),
                    help="output dir for artifacts (default: cwd)")
    ap.add_argument("--route-timeout", type=int, default=600,
                    help="autorouter timeout in s (default 600)")
    ap.add_argument("--max-passes", type=int, default=50,
                    help="autorouter max passes, -mp flag (default 50)")
    ap.add_argument("--threads", type=int, default=4,
                    help="autorouter threads, -mt flag (default 4)")
    args = ap.parse_args()
    outdir = args.out.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    pcbnew = None
    if args.stage in ("create", "import", "all"):  # drc/route only need kicad-cli/freerouting
        pcbnew = _import_pcbnew()  # re-exec shim runs here when needed
        print(f"[env] pcbnew {pcbnew.GetBuildVersion()} on {sys.executable}")

    try:
        if args.stage in ("create", "all"):
            stage_create(pcbnew, outdir)
        if args.stage in ("route", "all"):
            stage_route(outdir, args.route_timeout, args.max_passes, args.threads)
        if args.stage in ("import", "all"):
            if not (outdir / SES).exists():
                sys.exit(f"ERROR: {outdir / SES} does not exist — run --stage route first")
            stage_import(pcbnew, outdir)
        if args.stage in ("drc", "all"):
            if args.stage == "drc" and not (outdir / ROUTED).exists():
                sys.exit(f"ERROR: {outdir / ROUTED} does not exist")
            if args.stage == "all" and not (outdir / ROUTED).exists():
                sys.exit("ERROR: import stage did not run — routed board missing")
            return stage_drc(outdir)
    except kp.ResolveError as e:
        print(f"ERROR [{e.component}]: {e.fix}", file=sys.stderr)
        return 2
    except subprocess.TimeoutExpired:
        print(f"ERROR: autorouter exceeded {args.route_timeout}s — try a larger --route-timeout "
              "or smaller -mp/-mt", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
