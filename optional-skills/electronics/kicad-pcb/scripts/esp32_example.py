#!/usr/bin/env python3
"""Full headless example: ESP32 + LED + USB-C minimal dev board.

Generates EVERYTHING without a GUI:
  1. schematic  (.kicad_sch)  - native s-expr, symbols embedded from official libs
  2. board      (.kicad_pcb)  - outline, footprints, nets
  3. project    (.kicad_pro)  - JLCPCB-class design rules
  4. autoroute  (DSN -> Freerouting -> SES)
  5. DRC (json) + 3D render (PNG) + schematic PNG

Circuit: USB-C VBUS 5V -> AMS1117-3.3 -> ESP32-WROOM-32; LED on GPIO2 (module pin 24;
pin 22 is SDI/SD1, the internal flash bus)
via 1k; 5.1k pulldowns on CC1/CC2 (USB-C sink requirement); 10uF decoupling.

Usage:
  python3 esp32_example.py --out /tmp/esp32            # everything (default)
  python3 esp32_example.py --stage sch --out DIR       # schematic only
  python3 esp32_example.py --stage board --out DIR     # board+route+drc+renders
  python3 esp32_example.py --stage render --out DIR    # re-render existing outputs

Outputs (under --out):
  esp32-devboard.kicad_sch  esp32-devboard.kicad_pcb  esp32-devboard.kicad_pro
  esp32-routed.kicad_pcb    esp32-drc.json
  renders/board-top.png     renders/schematic.svg (cropped; + .png with pymupdf)

Exit codes: 0 ok | 2 environment | 3 stage failed | 4 DRC failures.
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
import sch_gen as sg  # noqa: E402

PROJECT = "esp32-devboard"
SCH = "esp32-devboard.kicad_sch"
PCB = "esp32-devboard.kicad_pcb"
PRO = "esp32-devboard.kicad_pro"
ROUTED = "esp32-routed.kicad_pcb"
DRC = "esp32-drc.json"

BOARD_W, BOARD_H = 45.0, 25.0  # mm (JLCPCB min 6x6)

# placement targets are COURTYARD-CENTER-ish (gen_board re-anchors on the PADS
# bbox center). The rot-270 WROOM frees the board midsection, so passives sit
# in the middle column; nothing overlaps the big USB-C / SOT-223 courtyards.
PLACEMENT = [
    # rot 270: the receptacle mouth (footprint +Y) faces the LEFT board edge;
    # gen_board then slides J1 so its "PCB Edge" line sits on x = 0
    ("Connector_USB", "USB_C_Receptacle_GCT_USB4105-xx-A_16P_TopMnt_Horizontal", "J1", 5.5, 12.5, 270),
    ("Package_TO_SOT_SMD", "SOT-223-3_TabPin2", "U2", 14.5, 5.0, 0),
    ("Capacitor_SMD", "C_0805_2012Metric", "C1", 22.0, 12.0, 0),
    ("Capacitor_SMD", "C_0805_2012Metric", "C2", 22.0, 16.0, 0),
    ("Resistor_SMD", "R_0603_1608Metric", "R1", 13.0, 20.5, 0),
    ("LED_SMD", "LED_0603_1608Metric", "D1", 13.0, 16.5, 270),
    ("Resistor_SMD", "R_0603_1608Metric", "R2", 16.5, 20.5, 0),
    ("Resistor_SMD", "R_0603_1608Metric", "R3", 20.0, 20.5, 0),
    # WROOM pads-center at 34.7, rot 270 so the embedded 48mm ANTENNA KEEPOUT
    # extends OFF-board (to x>44, past the 45mm edge). With rot 90 it sweeps
    # the board interior (x 3.3-24.3) and blocks every zone fill + DRC item.
    ("RF_Module", "ESP32-WROOM-32", "U1", 34.7, 12.5, 270),
]
# connectors whose footprint carries a "PCB Edge" line on Dwgs.User: that line
# is moved onto the left board edge so the plug opening faces outwards
EDGE_MOUNT = {"J1"}
REF_BELOW = {"U2"}  # parts near the top edge: silkscreen reference goes under them

# zone x-limit: cover the WROOM's GND pads (up to x=44) but keep 1mm from the
# board edge; the module's antenna keepout is OFF-board (module at the edge).
ZONE_X_MAX = 44.0

# Board nets are NOT typed twice: gen_board reads them from the schematic netlist
# (kicad-cli sch export netlist), like "Update PCB from Schematic" — so schematic
# parity holds by construction and a schematic mistake shows up on the board too.
GND_NETS = {"GND"}

VALUES = {"U1": "ESP32-WROOM-32", "U2": "AMS1117-3.3", "J1": "USB_C_GCT_USB4105",
          "C1": "10uF", "C2": "10uF", "R1": "1k", "R2": "5.1k", "R3": "5.1k",
          "D1": "GREEN"}

# symbol lib (per official KiCad libs) for the schematic
SYMBOLS = {  # ref -> (lib_name, symbol_name)
    "U1": ("RF_Module", "ESP32-WROOM-32"),
    "U2": ("Regulator_Linear", "AMS1117-3.3"),
    "J1": ("Connector", "USB_C_Receptacle_USB2.0_16P"),  # 16 pins = the GCT 16P footprint
    "R1": ("Device", "R"), "R2": ("Device", "R"), "R3": ("Device", "R"),
    "C1": ("Device", "C"), "C2": ("Device", "C"),
    "D1": ("Device", "LED"),
}


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


# ------------------------------------------------------------------ schematic

# schematic: ref -> (x, y) on an A4 sheet (mm, y down), laid out as a signal flow:
# USB-C -> CC pulldowns | 5V -> regulator + caps -> ESP32 -> LED
SCH_PLACE = {
    "J1": (45.72, 101.6), "R2": (78.74, 96.52), "R3": (91.44, 96.52),
    "C1": (111.76, 66.04), "U2": (134.62, 60.96), "C2": (157.48, 66.04),
    "U1": (210.82, 101.6), "R1": (248.92, 91.44), "D1": (254.0, 111.76),
}
# ref -> {symbol pin: net}; stacked pins (USB-C VBUS/GND groups) need one entry
SCH_NETS = {
    "J1": {"A4": "+5V", "A1": "GND", "SH": "GND", "A5": "CC1", "B5": "CC2"},
    "R2": {"1": "CC1", "2": "GND"}, "R3": {"1": "CC2", "2": "GND"},
    "C1": {"1": "+5V", "2": "GND"},
    "U2": {"3": "+5V", "2": "+3V3", "1": "GND"},
    "C2": {"1": "+3V3", "2": "GND"},
    "U1": {"2": "+3V3", "1": "GND", "24": "LED"},
    "R1": {"1": "LED", "2": "LED_A"},
    "D1": {"2": "LED_A", "1": "GND"},
}
SOURCES = {("J1", "A4"), ("J1", "A1")}  # power enters through the USB-C: PWR_FLAGs
# left open on purpose so ERC keeps flagging them: a real ESP32 board needs an
# EN RC (10k pull-up + 1uF) — out of scope for this minimal routing demo
NO_FLAG = {("U1", "3")}
FOOTPRINTS = {ref: f"{lib}:{fp}" for lib, fp, ref, *_ in PLACEMENT}
SCH_BBOX: list[tuple[float, float, float, float]] = []


def gen_schematic(outdir: Path) -> Path:
    sch = sg.Schematic(sg.symbols_dir(), PROJECT,
                       "ESP32 + LED + USB-C - headless example", paper="A4")
    for ref, (lib_name, sym_name) in SYMBOLS.items():
        x, y = SCH_PLACE[ref]
        sch.add(ref, lib_name, sym_name, x, y, VALUES[ref], FOOTPRINTS[ref])
    for ref, pins in SCH_NETS.items():
        for pin, net in pins.items():
            sch.connect(ref, pin, net, source=(ref, pin) in SOURCES)
    for ref, pin in NO_FLAG:  # reserve the point so no_connect_rest() skips it
        sch.taken[sch.parts[ref].pin_xy(pin)] = "(open)"
    n_nc = sch.no_connect_rest()
    dst = sch.write(outdir / SCH)
    sg.write_lib_tables(outdir, sch.sym_libs(), {lib for lib, *_ in PLACEMENT})
    SCH_BBOX[:] = [sch.bbox()]
    print(f"[sch] schematic written: {dst} ({len(sch.parts)} symbols, {n_nc} no-connect flags)")
    return dst


# ------------------------------------------------------------------ board

# Custom DRC rules: scoped, commented exceptions instead of global severity changes.
DRU = """(version 1)

# The ESP32-WROOM-32 antenna overhangs the board edge on purpose (module maker's
# layout guidance: antenna outside the board or over a copper-free area). Only
# U1's silkscreen outline is exempt from the silk-to-edge check; its copper and
# every other footprint keep the normal rules.
(rule "U1 antenna overhang: silkscreen past the edge"
    (layer "F.Silkscreen")
    (constraint silk_clearance (min -100mm))
    (condition "A.memberOfFootprint('U1')")
    (severity ignore))
"""


def gen_project(outdir: Path) -> Path:
    pro = {
        "board": {"design_settings": {
            "rules": {"min_track_width": 0.127, "min_clearance": 0.127,
                       "min_through_hole_diameter": 0.2, "min_via_diameter": 0.45},
            "rule_severities": {
                "solder_mask_bridge": "warning",   # preserve as visible finding; never auto-accept
                "lib_footprint_issues": "warning",
                # the official USB-C footprint places NPTH mounting holes next
                # to its own GND pads by design — inherent to the lib part
                "hole_clearance": "warning",
                "hole_to_hole_clearance": "warning",
                "courtyards_overlap": "warning"}}},
        "meta": {"filename": PRO},
    }
    dst = outdir / PRO
    dst.write_text(json.dumps(pro, indent=2), encoding="utf-8")
    dst.with_suffix(".kicad_dru").write_text(DRU, encoding="utf-8")
    print(f"[pro] project rules (JLCPCB-class) + custom rules: {dst}")
    return dst


def _court_w_h(pcbnew, f) -> tuple[float, float]:
    """Courtyard width/height in mm (cached after BuildCourtyardCaches)."""
    try:
        f.BuildCourtyardCaches()
        box = f.GetCachedCourtyard()  # BOX2I in nm
        return box.GetWidth() / 1e6, box.GetHeight() / 1e6
    except Exception:  # noqa: BLE001 — fallback: bounding box approximation
        bb = f.GetBoundingBox()
        return bb.GetWidth() / 1e6, bb.GetHeight() / 1e6


def gen_board(pcbnew, outdir: Path) -> Path:
    mm = pcbnew.FromMM
    board = pcbnew.NewBoard(str(outdir / PCB))

    edge = pcbnew.PCB_SHAPE(board)
    edge.SetShape(pcbnew.SHAPE_T_RECT)
    edge.SetStart(pcbnew.VECTOR2I(mm(0), mm(0)))
    edge.SetEnd(pcbnew.VECTOR2I(mm(BOARD_W), mm(BOARD_H)))
    edge.SetLayer(pcbnew.Edge_Cuts)
    edge.SetWidth(mm(0.1))
    board.Add(edge)

    libs = kp.footprints_dir()
    nets: dict[str, object] = {}

    def net(name: str):
        if name not in nets:
            n = pcbnew.NETINFO_ITEM(board, name)
            board.Add(n)
            nets[name] = n
        return nets[name]

    if not (outdir / SCH).exists():
        gen_schematic(outdir)
    pad_nets = schematic_pad_nets(outdir / SCH)
    fps = {}
    for lib, fp_name, ref, x, y, rot in PLACEMENT:
        f = pcbnew.FootprintLoad(str(libs / f"{lib}.pretty"), fp_name)
        if not f:
            sys.exit(f"ERROR: footprint {lib}:{fp_name} not found in {libs}")
        f.SetFPID(pcbnew.LIB_ID(lib, fp_name))  # parity compares lib:name
        f.SetPath(pcbnew.KIID_PATH("/" + sg.symbol_uid(PROJECT, ref)))  # link to symbol
        f.SetReference(ref)
        f.SetValue(VALUES[ref])
        f.SetPosition(pcbnew.VECTOR2I(mm(x), mm(y)))  # anchor = pad-1/origin
        f.SetOrientationDegrees(rot)
        board.Add(f)
        # Re-anchor so the PADS BBOX CENTER lands on (x, y). SetPosition anchors
        # at the footprint origin (usually pad 1) and GetCenter() includes
        # text/graphics (a USB-C measured 48mm wide once!) — only the absolute
        # pad positions are rotation-proof and physical.
        xs = [p.GetPosition().x for p in f.Pads()]
        ys = [p.GetPosition().y for p in f.Pads()]
        if xs and ys:
            cx, cy = (min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2
            f.Move(pcbnew.VECTOR2I(int(mm(x)) - cx, int(mm(y)) - cy))
        if ref in EDGE_MOUNT:
            align_to_left_edge(pcbnew, f)
        if ref in REF_BELOW:  # default ref text would be clipped by the board edge
            pads_bb = [p.GetBoundingBox() for p in f.Pads()]
            cx = (min(b.GetLeft() for b in pads_bb) + max(b.GetRight() for b in pads_bb)) // 2
            f.Reference().SetPosition(pcbnew.VECTOR2I(
                cx, max(b.GetBottom() for b in pads_bb) + int(mm(1.2))))
        fps[ref] = f
        for pad in f.Pads():  # every pad sharing a number (WROOM pad 39 x9: pitfall!)
            net_name = pad_nets.get((ref, pad.GetNumber()))
            if not net_name:
                continue
            pad.SetNet(net(net_name))
            if net_name in GND_NETS:
                # solid connection to the zone (no thermal spokes ->
                # no starved_thermal on small pads like USB-C corners)
                pad.SetLocalZoneConnection(pcbnew.ZONE_CONNECTION_FULL)
    pads_on_board = {(r, p.GetNumber()) for r, f in fps.items() for p in f.Pads()}
    missing = sorted(k for k in pad_nets if k not in pads_on_board)

    # NOTE: the GND zone is added AFTER routing (add_gnd_zone) — zones present
    # during DSN export become obstacles for Freerouting.

    board.Save(str(outdir / PCB))
    print(f"[board] {len(fps)} footprints, {len(nets)} nets -> {outdir / PCB}")
    if missing:  # a symbol pin with no pad = wrong symbol/footprint pairing
        print(f"[board] WARNING schematic pins without a pad: {missing}")
    return outdir / PCB


def align_to_left_edge(pcbnew, f) -> None:
    """Slide f along X so its vertical "PCB Edge" line (Dwgs.User) lands on x = 0."""
    xs = [g.GetStart().x for g in f.GraphicalItems()
          if g.GetLayer() == pcbnew.Dwgs_User and isinstance(g, pcbnew.PCB_SHAPE)
          and g.GetShape() == pcbnew.SHAPE_T_SEGMENT and g.GetStart().x == g.GetEnd().x]
    if not xs:
        sys.exit(f"ERROR: {f.GetReference()} has no vertical PCB Edge line after rotation — "
                 "check its rotation in PLACEMENT")
    f.Move(pcbnew.VECTOR2I(-min(xs), 0))


def schematic_pad_nets(sch: Path) -> dict[tuple[str, str], str]:
    """{(ref, pad number): net} from `kicad-cli sch export netlist` (KiCad XML).
    Includes the single-pin `unconnected-(...)` nets KiCad gives no-connect pins,
    which the board must carry too for schematic parity."""
    xml = sch.with_suffix(".net.xml")
    _run([str(kp.kicad_cli()), "sch", "export", "netlist", "--format", "kicadxml",
          "-o", str(xml), str(sch)])
    if not xml.exists():
        sys.exit(f"ERROR: netlist export failed for {sch}")
    import xml.etree.ElementTree as ET
    out = {}
    for net in ET.parse(xml).getroot().iter("net"):
        name = net.get("name")
        if name.startswith(("unconnected-(", "Net-(")):
            # auto names embed pin names; KiCad stores "/" there as {slash}
            # (e.g. SDI/SD1), while the XML netlist prints it unescaped
            head, body = name.split("(", 1)
            name = f"{head}({body.replace('/', '{slash}')}"
        for node in net.iter("node"):
            out[(node.get("ref"), node.get("pin"))] = name
    xml.unlink()
    return out


def stitch_gnd_vias(pcbnew, board) -> int:
    """Drop a GND via beside every SMD GND pad (F.Cu -> B.Cu zone), only where
    it fits: candidate positions are checked against every foreign-net pad,
    track and via (grown by clearance + via radius) before the via is committed.
    A through via touches BOTH layers, so B.Cu tracks under an SMD pad count."""
    mm = pcbnew.FromMM
    gnd = board.FindNet("GND")
    if gnd is None:
        return 0
    clearance = int(mm(0.3))
    via_r = int(mm(0.225))  # 0.45mm via diameter

    # collect foreign-net pad bboxes once
    foreign = []
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            if pad.GetNetname() not in ("GND", ""):
                bb = pad.GetBoundingBox()
                foreign.append(bb)

    # foreign-net tracks/vias as (x1, y1, x2, y2, half width); a via is a 0-length segment
    segs = []
    for t in board.GetTracks():
        if t.GetNetname() == "GND":
            continue
        a, b = t.GetStart(), t.GetEnd()
        # KiCad 10: a via's width is per layer (GetWidth() without one asserts)
        w = t.GetWidth(pcbnew.F_Cu) if t.Type() == pcbnew.PCB_VIA_T else t.GetWidth()
        segs.append((a.x, a.y, b.x, b.y, w // 2))

    def seg_dist(px: int, py: int, x1: int, y1: int, x2: int, y2: int) -> float:
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        u = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
        return ((px - x1 - u * dx) ** 2 + (py - y1 - u * dy) ** 2) ** 0.5

    keep = clearance + via_r
    def hits_foreign(x: int, y: int) -> bool:
        for bb in foreign:
            if (bb.GetLeft() - keep <= x <= bb.GetRight() + keep
                    and bb.GetTop() - keep <= y <= bb.GetBottom() + keep):
                return True
        return any(seg_dist(x, y, x1, y1, x2, y2) < keep + hw
                   for x1, y1, x2, y2, hw in segs)

    n = skipped = 0
    placed: list[tuple[int, int]] = []  # overlapping pads (USB-C A1/B12) share a spot
    min_pitch = 2 * via_r + clearance
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            if pad.GetNetname() != "GND" or pad.GetAttribute() != pcbnew.PAD_ATTRIB_SMD:
                continue
            p = pad.GetPosition()
            # candidate offsets around the pad; pick the first clear one
            step = int(mm(0.3))
            cands = [(p.x + step, p.y), (p.x - step, p.y),
                     (p.x, p.y + step), (p.x, p.y - step),
                     (p.x + step, p.y + step), (p.x - step, p.y - step),
                     (p.x, p.y)]  # center as last resort
            for cx, cy in cands:
                if any(abs(cx - vx) < min_pitch and abs(cy - vy) < min_pitch
                       for vx, vy in placed):
                    break  # a stitch via already serves this spot
                if not hits_foreign(cx, cy):
                    placed.append((cx, cy))
                    via = pcbnew.PCB_VIA(board)
                    via.SetPosition(pcbnew.VECTOR2I(cx, cy))
                    via.SetNet(gnd)
                    via.SetWidth(mm(0.45))
                    via.SetDrill(mm(0.2))
                    via.SetViaType(pcbnew.VIATYPE_THROUGH)
                    board.Add(via)
                    n += 1
                    break
            else:
                skipped += 1
    if skipped:
        print(f"[stitch] {skipped} pad(s) had no clear via spot (zone still connects them)")
    print(f"[stitch] {n} GND stitching vias added")
    return n


def add_gnd_zone(pcbnew, board) -> None:
    """GND copper zones on BOTH layers (2-layer devboard practice): the F.Cu
    pour bridges B.Cu fill islands (single-layer fills fragment around tracks)."""
    mm = pcbnew.FromMM
    gnd = board.FindNet("GND")
    if gnd is None:
        print("[zone] GND net not found — zone skipped")
        return
    margin = 0.3
    for layer in (pcbnew.F_Cu, pcbnew.B_Cu):
        zone = pcbnew.ZONE(board)
        zone.SetLayer(layer)
        zone.SetNet(gnd)
        zone.SetMinThickness(mm(0.2))
        zone.SetIsRuleArea(False)
        chain = pcbnew.SHAPE_LINE_CHAIN()
        chain.Append(mm(margin), mm(margin))
        chain.Append(mm(ZONE_X_MAX), mm(margin))
        chain.Append(mm(ZONE_X_MAX), mm(BOARD_H - margin))
        chain.Append(mm(margin), mm(BOARD_H - margin))
        chain.SetClosed(True)
        zone.AddPolygon(chain)
        board.Add(zone)
    filler = pcbnew.ZONE_FILLER(board)
    ok = filler.Fill(board.Zones())
    print(f"[zone] GND zones on F.Cu+B.Cu filled: {ok}")


def _run(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    print(f"[exec] {' '.join(cmd[:3])} ...")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def route(pcbnew, outdir: Path, timeout: int, passes: int, threads: int) -> Path:
    # DSN comes from the clean unrouted board (no zone = no fake obstacles);
    # the GND zone is added to the ROUTED board afterwards (Astra order).
    board = pcbnew.LoadBoard(str(outdir / PCB))
    if not pcbnew.ExportSpecctraDSN(board, str(outdir / "esp32.dsn")):
        sys.exit("ERROR: ExportSpecctraDSN failed")
    mode, argv = kp.freerouting()
    cmd = argv + ["-de", str(outdir / "esp32.dsn"), "-do", str(outdir / "esp32.ses"),
                  "-mp", str(passes), "-mt", str(threads)]
    print(f"[route] freerouting ({mode}) ...")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    tail = "\n".join((r.stdout or "").strip().splitlines()[-4:])
    print(tail)
    if r.returncode != 0 or not (outdir / "esp32.ses").exists():
        sys.exit(f"ERROR: freerouting exit={r.returncode}")
    board = pcbnew.LoadBoard(str(outdir / PCB))
    if not pcbnew.ImportSpecctraSES(board, str(outdir / "esp32.ses")):
        sys.exit("ERROR: ImportSpecctraSES failed")
    stitch_gnd_vias(pcbnew, board)   # deterministic GND: via per SMD GND pad
    add_gnd_zone(pcbnew, board)      # zone AFTER routing, over the routed board
    board.Save(str(outdir / ROUTED))
    print(f"[route] routed board: {outdir / ROUTED} ({len(board.GetTracks())} tracks)")
    return outdir / ROUTED


def drc(outdir: Path) -> int:
    cli = kp.kicad_cli()
    # --schematic-parity looks for <board name>.kicad_sch next to the board
    for src, ext in ((outdir / SCH, ".kicad_sch"), (outdir / PRO, ".kicad_dru")):
        src = src.with_suffix(ext)
        if src.exists():  # parity + custom rules are looked up by the BOARD's name
            (outdir / ROUTED).with_suffix(ext).write_bytes(src.read_bytes())
    r = _run([str(cli), "pcb", "drc", "--format", "json", "--schematic-parity", "--output",
              str(outdir / DRC), str(outdir / ROUTED)])
    if not (outdir / DRC).exists():
        sys.exit(f"ERROR: DRC failed: {(r.stderr or '')[:300]}")
    rep = json.loads((outdir / DRC).read_text(encoding="utf-8"))
    errors = [v for v in rep.get("violations", []) if v.get("severity") == "error"]
    warns = [v for v in rep.get("violations", []) if v.get("severity") != "error"]
    unconn = rep.get("unconnected_items", [])
    parity = rep.get("schematic_parity", [])
    print(f"[drc] {len(errors)} error(s), {len(warns)} warning(s), "
          f"unconnected {len(unconn)}, parity {len(parity)}")
    for v in (errors + warns + unconn + parity)[:12]:
        print(f"   - {v.get('type')}: {str(v.get('description', ''))[:90]}")
    # Warnings reviewed AGAINST the fabrication process and accepted on purpose.
    # Skill rule: document the exception with its justification, never blanket-ignore.
    REVIEWED_WARNINGS = {
        "hole_clearance": "J1 = GCT USB4105-xx-A vendor footprint: pad-to-own-NPTH gap is "
                          "0.194mm by manufacturer geometry (factory-proven connector, "
                          "assembled by JLCPCB daily); the 0.25mm constraint comes from an "
                          "unrelated demo project, not this fab's limits.",
    }
    reviewed = [v for v in warns if v.get("type") in REVIEWED_WARNINGS]
    unreviewed = [v for v in warns if v.get("type") not in REVIEWED_WARNINGS]
    if not errors and not unreviewed and not unconn and not parity:
        for t in sorted({v.get("type") for v in reviewed}):
            print(f"[reviewed] {t}: {REVIEWED_WARNINGS[t]}")
        print(f"DRC: PASS ({len(reviewed)} reviewed warning(s), 0 unreviewed)")
        return 0
    print("DRC: FAIL")
    return 4


def renders(outdir: Path) -> None:
    cli = str(kp.kicad_cli())
    rdir = outdir / "renders"
    rdir.mkdir(exist_ok=True)
    # 3D render
    r = _run([cli, "pcb", "render", "--side", "top", "--quality", "high",
              "--width", "1200", "-o", str(rdir / "board-top.png"),
              str(outdir / ROUTED)])
    ok = (rdir / "board-top.png").exists()
    print(f"[render] 3D: {'OK' if ok else 'FAILED'} {rdir / 'board-top.png'}")
    if not ok:
        print((r.stderr or "")[:300])
    # schematic: PDF (full sheet) + SVG cropped to the drawing (+ PNG if possible)
    _run([cli, "sch", "export", "pdf", "-o", str(outdir / "esp32-devboard.pdf"), str(outdir / SCH)])
    svg = rdir / "schematic.svg"
    bbox = SCH_BBOX[0] if SCH_BBOX else None
    if bbox is None:  # --stage render: rebuild the layout to know the drawing extents
        gen_schematic(outdir)
        bbox = SCH_BBOX[0]
    if sg.export_svg(cli, outdir / SCH, svg, bbox):
        png = sg.svg_to_png(svg, rdir / "schematic.png")
        print(f"[render] schematic: {svg}{' + png' if png else ' (png: pip install pymupdf)'}")


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["sch", "board", "render", "all"], default="all")
    ap.add_argument("--out", type=Path, default=Path.cwd() / "esp32-out")
    ap.add_argument("--route-timeout", type=int, default=900)
    ap.add_argument("--max-passes", type=int, default=80)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    outdir = args.out.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        if args.stage in ("sch", "all"):
            gen_schematic(outdir)
        if args.stage in ("board", "all"):
            pcbnew = _import_pcbnew()
            print(f"[env] pcbnew {pcbnew.GetBuildVersion()} on {sys.executable}")
            gen_project(outdir)
            gen_board(pcbnew, outdir)
            route(pcbnew, outdir, args.route_timeout, args.max_passes, args.threads)
            rc = drc(outdir)
            renders(outdir)
            return rc
        if args.stage == "render":
            renders(outdir)
            return 0
    except kp.ResolveError as e:
        print(f"ERROR [{e.component}]: {e.fix}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
