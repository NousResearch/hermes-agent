#!/usr/bin/env python3
"""Minimal headless board that needs ONLY a stock KiCad 10 install (no Freerouting/Java).

Circuit: 2-pin header (VIN, GND) -> R1 1k -> D1 LED -> GND.  A "hello world"
for the whole pipeline on Linux / Windows / macOS:

  1. schematic (.kicad_sch) with symbols embedded from the official libs
  2. board (.kicad_pcb) linked to the schematic (footprint paths = symbol uuids)
  3. deterministic routing via pcbnew (straight/L tracks, no autorouter)
  4. ERC + DRC with schematic parity (exit 4 on any finding)
  5. gerbers + drill + position file + a fab zip, 3D render, schematic PDF

Usage:
  python simple_board.py --out ./led-board
  python simple_board.py --out ./led-board --vin 5 --led-vf 2.0 --led-ma 3

Exit codes: 0 ok | 2 environment | 3 stage failed | 4 ERC/DRC findings.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kicad_paths as kp  # noqa: E402
import sch_gen as sg  # noqa: E402

NAME = "simple-led"
BOARD_W, BOARD_H = 30.0, 20.0  # mm
TRACK_W = 0.3                  # mm, comfortably above JLCPCB-class minimums
FAB_LAYERS = ("F.Cu,B.Cu,F.Mask,B.Mask,F.Silkscreen,B.Silkscreen,"
              "F.Paste,B.Paste,Edge.Cuts")  # what a 2-layer fab needs, nothing else

# ref -> (symbol lib, symbol, footprint lib, footprint, board x, board y, rotation)
PARTS = {
    "J1": ("Connector_Generic", "Conn_01x02", "Connector_PinHeader_2.54mm",
           # pads centre 1.27 mm low so pin 1 (VIN) sits on the y = 8 signal row
           "PinHeader_1x02_P2.54mm_Vertical", 7.0, 9.27, 0),
    "R1": ("Device", "R", "Resistor_SMD", "R_0805_2012Metric", 14.0, 8.0, 0),
    # rot 180 puts the anode (pad 2) on the left, facing R1
    "D1": ("Device", "LED", "LED_SMD", "LED_0805_2012Metric", 22.0, 8.0, 180),
}
# ref -> {pin number: net}; same net names are used on schematic and board
NETS = {
    "J1": {"1": "VIN", "2": "GND"},
    "R1": {"1": "VIN", "2": "LED_A"},
    "D1": {"2": "LED_A", "1": "GND"},  # LED symbol/footprint: 1 = K, 2 = A
}
# schematic placement (mm, y grows DOWN on the sheet; sch_gen snaps to the 1.27 grid)
SCH_AT = {"J1": (60.96, 76.2), "R1": (91.44, 76.2), "D1": (116.84, 76.2)}
SCH_BBOX: dict[Path, tuple[float, float, float, float]] = {}


def _import_pcbnew():
    """Imports pcbnew; if unavailable, re-execs THIS script in KiCad's Python."""
    with contextlib.redirect_stderr(io.StringIO()):  # PROPERTY_ENUM asserts = noise
        try:
            import pcbnew
            return pcbnew
        except ImportError:
            pass
    if os.environ.get("_KICAD_PCB_REEXEC"):
        sys.exit("ERROR: pcbnew not importable even in KiCad's Python. Set KICAD_PYTHON.")
    for py in kp.python_with_pcbnew_candidates():
        if py == Path(sys.executable) or not py.exists():
            continue
        if kp.verify_pcbnew(py):
            env = dict(os.environ, _KICAD_PCB_REEXEC="1", KICAD_PYTHON=str(py))
            print(f"[shim] re-executing in KiCad's Python: {py}", flush=True)
            r = subprocess.run([str(py), str(Path(__file__).resolve())] + sys.argv[1:], env=env)
            sys.exit(r.returncode)
    sys.exit("ERROR: no Python with pcbnew bindings found. Install KiCad 10 or set KICAD_PYTHON.")


def led_resistor(vin: float, vf: float, ma: float) -> str:
    """Nearest E12 value >= (Vin - Vf) / I, formatted like '1k' / '470'."""
    ohms = (vin - vf) / (ma / 1000.0)
    if ohms <= 0:
        sys.exit(f"ERROR: --vin {vin} must be above the LED forward voltage {vf}")
    e12 = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]
    decade = 1.0
    while decade * 10 <= ohms:
        decade *= 10
    val = next((d * decade for d in e12 if d * decade >= ohms - 1e-9), 10 * decade)
    if val >= 1000:
        k = val / 1000
        return f"{k:g}k"
    return f"{val:g}"


# ------------------------------------------------------------------ schematic

def gen_schematic(outdir: Path, values: dict[str, str], uuids: dict[str, str]) -> Path:
    sch = sg.Schematic(sg.symbols_dir(), NAME, "Simple LED board - headless example")
    for ref, (slib, sym, flib, fp, *_rest) in PARTS.items():
        x, y = SCH_AT[ref]
        sch.add(ref, slib, sym, x, y, values[ref], f"{flib}:{fp}", uid=uuids[ref])
    for ref, pins in NETS.items():
        for pin, net in pins.items():
            # J1 is where power enters the board -> PWR_FLAG on its GND
            sch.connect(ref, pin, net, source=(ref == "J1" and net == "GND"))
    sch.no_connect_rest()
    dst = sch.write(outdir / f"{NAME}.kicad_sch")
    sg.write_lib_tables(outdir, sch.sym_libs(), {v[2] for v in PARTS.values()})
    SCH_BBOX[dst] = sch.bbox()
    print(f"[sch] {dst.name}: {len(PARTS)} symbols")
    return dst


# ------------------------------------------------------------------ board

def gen_board(pcbnew, outdir: Path, values: dict[str, str], uuids: dict[str, str]) -> Path:
    mm = pcbnew.FromMM
    V = lambda x, y: pcbnew.VECTOR2I(int(mm(x)), int(mm(y)))  # noqa: E731
    dst = outdir / f"{NAME}.kicad_pcb"
    board = pcbnew.NewBoard(str(dst))

    edge = pcbnew.PCB_SHAPE(board)
    edge.SetShape(pcbnew.SHAPE_T_RECT)
    edge.SetStart(V(0, 0))
    edge.SetEnd(V(BOARD_W, BOARD_H))
    edge.SetLayer(pcbnew.Edge_Cuts)
    edge.SetWidth(mm(0.1))
    board.Add(edge)

    nets = {}
    for name in sorted({n for m in NETS.values() for n in m.values()}):
        nets[name] = pcbnew.NETINFO_ITEM(board, name)
        board.Add(nets[name])

    fpdir = kp.footprints_dir()
    fps = {}
    for ref, (_sl, _s, flib, fp, x, y, rot) in PARTS.items():
        f = pcbnew.FootprintLoad(str(fpdir / f"{flib}.pretty"), fp)
        if not f:
            sys.exit(f"ERROR: footprint {flib}:{fp} not found in {fpdir}")
        f.SetFPID(pcbnew.LIB_ID(flib, fp))            # parity compares lib:name
        f.SetPath(pcbnew.KIID_PATH(f"/{uuids[ref]}"))  # links to the schematic symbol
        f.SetReference(ref)
        f.SetValue(values[ref])
        f.SetOrientationDegrees(rot)
        board.Add(f)
        # centre the PADS on (x, y): the footprint origin is often pad 1
        xs = [p.GetPosition().x for p in f.Pads()]
        ys = [p.GetPosition().y for p in f.Pads()]
        f.Move(pcbnew.VECTOR2I(int(mm(x)) - (min(xs) + max(xs)) // 2,
                               int(mm(y)) - (min(ys) + max(ys)) // 2))
        for pad in f.Pads():
            if pad.GetNumber() in NETS[ref]:
                pad.SetNet(nets[NETS[ref][pad.GetNumber()]])
        fps[ref] = f

    def pad_xy(ref: str, num: str) -> tuple[float, float]:
        p = fps[ref].FindPadByNumber(num).GetPosition()
        return pcbnew.ToMM(p.x), pcbnew.ToMM(p.y)

    def track(points: list[tuple[float, float]], net: str) -> None:
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            t = pcbnew.PCB_TRACK(board)
            t.SetStart(V(x1, y1))
            t.SetEnd(V(x2, y2))
            t.SetWidth(mm(TRACK_W))
            t.SetLayer(pcbnew.F_Cu)
            t.SetNet(nets[net])
            board.Add(t)

    # deterministic routing: all parts share one row, GND returns underneath
    j1_1, j1_2 = pad_xy("J1", "1"), pad_xy("J1", "2")
    r1_1, r1_2 = pad_xy("R1", "1"), pad_xy("R1", "2")
    d1_2, d1_1 = pad_xy("D1", "2"), pad_xy("D1", "1")
    gnd_y = BOARD_H - 5.0
    track([j1_1, r1_1], "VIN")
    track([r1_2, d1_2], "LED_A")
    track([d1_1, (d1_1[0] + 2.0, d1_1[1]), (d1_1[0] + 2.0, gnd_y),
           (j1_2[0], gnd_y), j1_2], "GND")

    # pin labels beside the header pads, plus a board title
    for text, (x, y) in (("VIN", (j1_1[0] - 3.6, j1_1[1])), ("GND", (j1_2[0] - 3.6, j1_2[1])),
                         (f"LED {values['_vin']}V", (14.0, 4.5)),
                         ("hermes-kicad-pcb", (15.0, 17.5))):
        t = pcbnew.PCB_TEXT(board)
        t.SetText(text)
        t.SetPosition(V(x, y))
        t.SetLayer(pcbnew.F_SilkS)
        t.SetTextSize(V(1.0, 1.0))
        t.SetTextThickness(mm(0.15))
        board.Add(t)

    board.Save(str(dst))
    print(f"[board] {dst.name}: {len(fps)} footprints, {len(board.GetTracks())} tracks")
    return dst


# ------------------------------------------------------------------ checks + outputs

def run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def checks(cli: str, outdir: Path) -> int:
    sch, pcb = outdir / f"{NAME}.kicad_sch", outdir / f"{NAME}.kicad_pcb"
    erc, drc = outdir / "erc.json", outdir / "drc.json"
    run([cli, "sch", "erc", "--format", "json", "--severity-all", "-o", str(erc), str(sch)])
    run([cli, "pcb", "drc", "--format", "json", "--severity-all", "--schematic-parity",
         "-o", str(drc), str(pcb)])
    if not erc.exists() or not drc.exists():
        print("ERROR: kicad-cli did not write the ERC/DRC report", file=sys.stderr)
        return 3
    erc_items = [v for s in json.loads(erc.read_text(encoding="utf-8")).get("sheets", [])
                 for v in s.get("violations", [])]
    rep = json.loads(drc.read_text(encoding="utf-8"))
    groups = {"violations": rep.get("violations", []),
              "unconnected": rep.get("unconnected_items", []),
              "parity": rep.get("schematic_parity", [])}
    print(f"[erc] {len(erc_items)} finding(s)")
    print("[drc] " + ", ".join(f"{k} {len(v)}" for k, v in groups.items()))
    findings = erc_items + [v for g in groups.values() for v in g]
    for v in findings[:15]:
        print(f"   - {v.get('severity', '?')} {v.get('type')}: {str(v.get('description'))[:90]}")
    if findings:
        print("CHECKS: FAIL")
        return 4
    print("CHECKS: PASS (ERC/DRC/parity clean; engineering review still required)")
    return 0


def exports(cli: str, outdir: Path) -> None:
    pcb, sch = outdir / f"{NAME}.kicad_pcb", outdir / f"{NAME}.kicad_sch"
    fab = outdir / "fab"
    fab.mkdir(exist_ok=True)
    run([cli, "pcb", "export", "gerbers", "--layers", FAB_LAYERS, "-o", str(fab), str(pcb)])
    run([cli, "pcb", "export", "drill", "-o", str(fab) + os.sep, str(pcb)])
    run([cli, "pcb", "export", "pos", "--format", "csv", "--units", "mm",
         "-o", str(outdir / f"{NAME}-pos.csv"), str(pcb)])
    run([cli, "sch", "export", "bom", "-o", str(outdir / f"{NAME}-bom.csv"), str(sch)])
    run([cli, "sch", "export", "pdf", "-o", str(outdir / f"{NAME}-schematic.pdf"), str(sch)])
    svg = outdir / f"{NAME}-schematic.svg"
    if sch in SCH_BBOX and sg.export_svg(cli, sch, svg, SCH_BBOX[sch]):
        png = sg.svg_to_png(svg, svg.with_suffix(".png"))
        print(f"[sch] {svg.name} (cropped){' + png' if png else ''}")
    files = sorted(p for p in fab.iterdir() if p.is_file())
    with zipfile.ZipFile(outdir / f"{NAME}-gerbers.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.name)
    print(f"[fab] {len(files)} gerber/drill files -> {NAME}-gerbers.zip")
    for side in ("top", "bottom"):
        png = outdir / f"{NAME}-3d-{side}.png"
        run([cli, "pcb", "render", "--side", side, "--quality", "high", "--width", "1200",
             "--height", "800", "-o", str(png), str(pcb)], timeout=900)
        print(f"[render] {png.name}: {'OK' if png.exists() else 'FAILED'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path.cwd() / "simple-led-out")
    ap.add_argument("--vin", type=float, default=5.0, help="supply voltage (V)")
    ap.add_argument("--led-vf", type=float, default=2.0, help="LED forward voltage (V)")
    ap.add_argument("--led-ma", type=float, default=3.0, help="LED current (mA)")
    ap.add_argument("--no-export", action="store_true", help="skip gerbers/renders")
    args = ap.parse_args()
    outdir = args.out.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    pcbnew = _import_pcbnew()
    print(f"[env] pcbnew {pcbnew.GetBuildVersion()} on {sys.executable}")
    values = {"J1": "VIN/GND", "R1": led_resistor(args.vin, args.led_vf, args.led_ma),
              "D1": "LED", "_vin": f"{args.vin:g}"}
    uuids = {ref: str(uuid.uuid5(uuid.NAMESPACE_URL, f"{NAME}/{ref}")) for ref in PARTS}
    try:
        cli = str(kp.kicad_cli())
        gen_schematic(outdir, values, uuids)
        gen_board(pcbnew, outdir, values, uuids)
        rc = checks(cli, outdir)
        if not args.no_export:
            exports(cli, outdir)
    except kp.ResolveError as e:
        print(f"ERROR [{e.component}]: {e.fix}", file=sys.stderr)
        return 2
    print(f"[done] outputs in {outdir}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
