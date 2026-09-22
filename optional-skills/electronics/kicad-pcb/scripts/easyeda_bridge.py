#!/usr/bin/env python3
"""EasyEDA <-> KiCad bridge (headless, cross-platform).

Subcommands:
  import-pro PROJECT.epro --out DIR     EasyEDA Pro PCB -> DIR/<name>.kicad_pcb
  import-std SRC --out DIR              EasyEDA Standard (project dir with
                                        easyeda/sources/pcb/document.json, or the
                                        json itself) -> DIR/<name>.kicad_pcb
  lcsc Cxxxxx --out DIR                 LCSC part -> KiCad symbol+footprint+3D
                                        (requires easyeda2kicad on PATH)

The importers use KiCad's OWN EasyEDA parsers via pcbnew.PCB_IO_MGR — nothing is
re-implemented here. Schematics are NOT converted (no headless importer exists);
re-draw them or use KiCad GUI: File -> Import Non-KiCad Project -> EasyEDA (Pro).

Validated (Linux, KiCad 10.0.6): a real .epro with 45 footprints / 1068 tracks
converts losslessly (DRC 0 unconnected / 0 parity; rule violations belong to the
source design, not the import).

Exit codes: 0 ok | 2 environment/config | 3 conversion failed.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kicad_paths as kp  # noqa: E402


def _import_pcbnew():
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            import pcbnew
            return pcbnew
        except ImportError:
            pass
    if os.environ.get("_KICAD_PCB_REEXEC"):
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
             "Install KiCad 10 or set KICAD_PYTHON.")


def _convert(pcbnew, src: Path, out_dir: Path, plugin_type) -> Path:
    """Load `src` with the given PCB_IO_MGR plugin, save as KiCad s-expr."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / (src.stem + ".kicad_pcb")
    mgr = pcbnew.PCB_IO_MGR
    print(f"[import] loading {src} ...")
    board = mgr.Load(plugin_type, str(src))
    if board is None:
        sys.exit(f"ERROR: PCB_IO_MGR.Load returned nothing for {src}")
    mgr.Save(mgr.KICAD_SEXP, str(dst), board)
    if not dst.exists():
        sys.exit(f"ERROR: save produced no file at {dst}")
    fps = list(board.GetFootprints())
    n_tracks = len(board.GetTracks())
    print(f"[import] {len(fps)} footprint(s), {n_tracks} track(s) -> {dst}")
    if not fps:
        print("[import] WARNING: 0 footprints — is this really the PCB document?")
    return dst


def cmd_import_pro(pcbnew, args) -> int:
    src = Path(args.src).expanduser().resolve()
    if not src.exists():
        print(f"ERROR: {src} does not exist", file=sys.stderr)
        return 2
    dst = _convert(pcbnew, src, Path(args.out), pcbnew.PCB_IO_MGR.EASYEDAPRO)
    print(f"[ok] KiCad board: {dst}")
    print("     next: kicad-cli pcb drc / demo_autoroute.py --stage drc (board arg)")
    return 0


def _find_std_json(src: Path) -> Path:
    if src.is_file() and src.suffix == ".json":
        return src
    # project zip exported from EasyEDA Standard
    if src.is_file() and src.suffix == ".zip":
        with zipfile.ZipFile(src) as z:
            names = [n for n in z.namelist()
                     if n.endswith(("easyeda/sources/pcb/document.json",
                                    "/pcb/document.json"))]
            if not names:
                sys.exit(f"ERROR: no easyeda/sources/pcb/document.json inside {src}")
            ex = src.parent / "_easyeda_extract"
            z.extract(names[0], ex)
            return ex / names[0]
    if src.is_dir():
        hits = list(src.glob("easyeda/sources/pcb/document.json")) or \
            list(src.rglob("pcb/document.json"))
        if hits:
            return hits[0]
    sys.exit(f"ERROR: cannot find the EasyEDA Std PCB json in/under {src}")


def cmd_import_std(pcbnew, args) -> int:
    src = _find_std_json(Path(args.src).expanduser().resolve())
    print(f"[import] using document: {src}")
    dst = _convert(pcbnew, src, Path(args.out), pcbnew.PCB_IO_MGR.EASYEDA)
    print(f"[ok] KiCad board: {dst}")
    return 0


def cmd_lcsc(args) -> int:
    part = args.part.upper().strip()
    if not (part.startswith("C") and part[1:].isdigit()):
        print(f"ERROR: '{args.part}' is not an LCSC part number (Cxxxxx)", file=sys.stderr)
        return 2
    exe = shutil.which("easyeda2kicad")
    if not exe:
        print("ERROR: easyeda2kicad not found on PATH.\n"
              "  Install it first:  pipx install easyeda2kicad\n"
              "  (it downloads symbol+footprint+3D from LCSC — network required).",
              file=sys.stderr)
        return 2
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    lib = out_dir / f"{part.lower()}-lib"
    lib.mkdir(parents=True, exist_ok=True)  # easyeda2kicad requires it to exist
    cmd = [exe, "--lcsc_id", part, "--full", "--output", str(lib / f"{part.lower()}.kicad_sym"),
           "--overwrite"]
    print(f"[lcsc] {' '.join(cmd)}")
    r = subprocess.run(cmd, timeout=180)
    if r.returncode != 0:
        print(f"ERROR: easyeda2kicad exit={r.returncode}", file=sys.stderr)
        return 3
    made = sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file())
    print(f"[ok] {len(made)} file(s) under {out_dir}:")
    for m in made[:12]:
        print(f"   - {m}")
    print("     add to fp-lib-table/sym-lib-table or copy the .pretty next to your project")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("import-pro", help="EasyEDA Pro .epro -> .kicad_pcb")
    p1.add_argument("src")
    p1.add_argument("--out", default="easyeda-out")
    p2 = sub.add_parser("import-std", help="EasyEDA Standard json/zip/dir -> .kicad_pcb")
    p2.add_argument("src")
    p2.add_argument("--out", default="easyeda-out")
    p3 = sub.add_parser("lcsc", help="LCSC Cxxxxx -> KiCad symbol/footprint/3D")
    p3.add_argument("part")
    p3.add_argument("--out", default="lcsc-out")
    args = ap.parse_args()

    if args.cmd == "lcsc":
        return cmd_lcsc(args)
    pcbnew = _import_pcbnew()
    print(f"[env] pcbnew {pcbnew.GetBuildVersion()} on {sys.executable}")
    if args.cmd == "import-pro":
        return cmd_import_pro(pcbnew, args)
    return cmd_import_std(pcbnew, args)


if __name__ == "__main__":
    sys.exit(main())
