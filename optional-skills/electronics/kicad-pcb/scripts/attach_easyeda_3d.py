#!/usr/bin/env python3
"""Attach EasyEDA/LCSC 3D models to board footprints and export board-level STEP.

Interface (as documented in SKILL.md):
  attach_easyeda_3d.py board.kicad_pcb --map REF=model.step [--map ...]
                        [--step out.step] [--render out.png] [--render-bottom out.png]

Models come from `easyeda_bridge.py lcsc Cxxxxx` (writes <part>-lib/<part>.3dshapes/*.step).
ALWAYS check the model filename matches the footprint package (the names carry it,
e.g. `SOT-223-4P_L6.5-W3.5-H1.6-LS7.0-P2.30`, `R0603`) — a mismatched shell corrupts
the STEP and hides mechanical conflicts. This script warns on suspicious mismatches.

The board is saved in place after attaching (models per ref are REPLACED, so the
command is idempotent). Exit codes: 0 ok | 2 bad args/env | 3 attach/export failed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# package keywords that must roughly agree between footprint name and model name
PKG_HINTS = [
    ("SOT-223", ("SOT-223", "SOT223")),
    ("SOT-23", ("SOT-23", "SOT23")),
    ("SOIC-8", ("SOIC-8", "SOIC8", "SOP-8", "SOP8")),
    ("0402", ("0402", "1005")),
    ("0603", ("0603", "1608")),
    ("0805", ("0805", "2012")),
    ("1206", ("1206", "3216")),
    ("USB", ("USB", "TYPE-C", "TYPEC")),
    ("WROOM", ("WROOM",)),
    ("QFN", ("QFN",)),
    ("TQFP", ("TQFP",)),
]


def package_hints(text: str) -> set[str]:
    t = text.upper().replace("_", "-")
    found = set()
    for key, variants in PKG_HINTS:
        if any(v.upper().replace("_", "-") in t for v in variants):
            found.add(key)
    return found


def warn_mismatch(ref: str, fp_name: str, model_path: Path) -> list[str]:
    fp_pkgs = package_hints(fp_name)
    md_pkgs = package_hints(model_path.stem)
    if fp_pkgs and md_pkgs and not (fp_pkgs & md_pkgs):
        return [f"[warn] {ref}: footprint '{fp_name}' suggests {sorted(fp_pkgs)} "
                f"but model '{model_path.name}' suggests {sorted(md_pkgs)} — verify!"]
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("board", help=".kicad_pcb to modify (saved in place)")
    ap.add_argument("--map", action="append", default=[], metavar="REF=PATH",
                    help="attach 3D model PATH (step/wrl) to footprint REF (repeatable)")
    ap.add_argument("--step", help="export board-level STEP here")
    ap.add_argument("--render", help="render top PNG here (kicad-cli pcb render)")
    ap.add_argument("--render-bottom", help="render bottom PNG here")
    ap.add_argument("--render-size", default="1600x1200", help="render WxH (default 1600x1200)")
    ap.add_argument("--strict", action="store_true", help="fail on any package-mismatch warning")
    args = ap.parse_args()

    board_path = Path(args.board).expanduser().resolve()
    if not board_path.exists():
        print(f"ERROR: board not found: {board_path}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import kicad_paths as kp  # noqa: E403  (resolves the pcbnew interpreter)
    import pcbnew  # noqa: E403

    board = pcbnew.LoadBoard(str(board_path))
    by_ref = {f.GetReference(): f for f in board.GetFootprints()}

    warnings: list[str] = []
    attached = 0
    for entry in args.map:
        if "=" not in entry:
            print(f"ERROR: --map wants REF=PATH, got '{entry}'", file=sys.stderr)
            return 2
        ref, raw = entry.split("=", 1)
        ref = ref.strip()
        model_path = Path(raw.strip()).expanduser().resolve()
        fp = by_ref.get(ref)
        if fp is None:
            print(f"ERROR: no footprint '{ref}' on board (have: {sorted(by_ref)})", file=sys.stderr)
            return 2
        if not model_path.exists():
            print(f"ERROR: model not found: {model_path}", file=sys.stderr)
            return 2
        warnings += warn_mismatch(ref, fp.GetFPIDAsString(), model_path)
        try:  # idempotente: substitui modelos existentes deste footprint
            fp.Models().clear()
        except Exception:
            pass
        m = pcbnew.FP_3DMODEL()
        m.m_Filename = str(model_path)
        m.m_Show = True
        m.m_Opacity = 1.0
        fp.Add3DModel(m)
        attached += 1
        print(f"[attach] {ref} ({fp.GetFPIDAsString().split(':')[-1]}) <- {model_path.name}")

    for w in warnings:
        print(w, file=sys.stderr)
    if warnings and args.strict:
        print("ERROR: --strict and package mismatches found", file=sys.stderr)
        return 3

    board.Save(str(board_path))
    print(f"[ok] {attached} model(s) attached; board saved: {board_path.name}")

    w_str, h_str = args.render_size.lower().split("x")
    rc = 0
    if args.step:
        cmd = [str(kp.kicad_cli()), "pcb", "export", "step", "--output", args.step, str(board_path)]
        print("[exec]", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        (r.stdout + r.stderr).splitlines()[-3:] and print("\n".join((r.stdout + r.stderr).splitlines()[-3:]))
        if r.returncode != 0 or not Path(args.step).exists():
            print("ERROR: STEP export failed", file=sys.stderr)
            rc = 3
        else:
            print(f"[ok] STEP: {args.step}")
    for side, out in (("top", args.render), ("bottom", args.render_bottom)):
        if not out:
            continue
        cmd = [str(kp.kicad_cli()), "pcb", "render", "--output", out, "--side", side,
               "--width", w_str, "--height", h_str, "--quality", "high", str(board_path)]
        print("[exec]", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not Path(out).exists():
            print(f"ERROR: {side} render failed", file=sys.stderr)
            rc = 3
        else:
            print(f"[ok] render {side}: {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
