#!/usr/bin/env python3
"""Small KiCad 10 schematic writer for headless examples — readable sheets without a GUI.

What it takes care of (each one was a real bug or an ugly sheet before):
  * library Y grows UP, sheet Y grows DOWN: pin (px, py) of a symbol at (x, y) is (x + px, y - py)
  * derived symbols (`extends`) are flattened, otherwise they embed with no body and no pins
  * every connection gets a short wire stub pointing AWAY from the body, then a label or a
    power symbol oriented along that stub, so text never lands on the symbol
  * power nets use real power symbols (GND, +5V, +3V3 ...); PWR_FLAG marks external sources
  * unused pins get no-connect flags (clean ERC); stacked pins are connected once
  * every coordinate is snapped to the 1.27 mm grid
  * `export_svg` writes an SVG cropped to the drawing (no page frame, white background):
    crisp in a README on Linux / Windows / macOS without any rasterizer

Symbols are placed unrotated; rotate the layout, not the parts. See simple_board.py and
esp32_example.py for usage.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kicad_paths as kp  # noqa: E402

GRID = 1.27
# screen angle (0 right, 90 up, 180 left, 270 down) -> sheet delta (sheet y grows down)
DIRS = {0: (1, 0), 90: (0, -1), 180: (-1, 0), 270: (0, 1)}
FONT = '(effects (font (size 1.27 1.27)){extra})'


def symbols_dir() -> Path:
    """Official symbol libs: env override, else next to the footprint libs."""
    for env in ("KICAD10_SYMBOL_DIR", "KICAD9_SYMBOL_DIR", "KICAD_SYMBOL_DIR"):
        p = os.environ.get(env)
        if p and (Path(p) / "Device.kicad_sym").exists():
            return Path(p)
    fp = kp.footprints_dir()
    for c in (fp.parent / "symbols", Path("/usr/share/kicad/symbols"),
              Path("/usr/local/share/kicad/symbols")):
        if (c / "Device.kicad_sym").exists():
            return c
    raise kp.ResolveError("symbols dir", "official KiCad symbol libs not found; set KICAD10_SYMBOL_DIR")


def snap(v: float) -> float:
    return round(round(v / GRID) * GRID, 4)


def balanced(text: str, start: int) -> str:
    """The s-expression that opens at text[start] == '('."""
    depth = 0
    for j in range(start, len(text)):
        depth += {"(": 1, ")": -1}.get(text[j], 0)
        if depth == 0:
            return text[start:j + 1]
    raise ValueError("unbalanced s-expression")


def strip_blocks(text: str, head: str) -> str:
    """Removes every balanced block that starts with `head` (e.g. '(property ')."""
    out, i = [], 0
    while (k := text.find(head, i)) >= 0:
        out.append(text[i:k])
        i = k + len(balanced(text, k))
    return "".join(out) + text[i:]


@dataclass
class Pin:
    number: str
    name: str
    etype: str
    x: float
    y: float
    angle: int      # direction from the connection point INTO the body (library, y up)
    length: float


@dataclass
class LibSymbol:
    lib_id: str
    block: str      # flattened, renamed to lib_id, ready for (lib_symbols ...)
    pins: dict[str, Pin]
    bbox: tuple[float, float, float, float]  # library coords (x0, y0, x1, y1), y up
    power: bool


_LIB_CACHE: dict[Path, str] = {}


def _lib_text(path: Path) -> str:
    if path not in _LIB_CACHE:
        _LIB_CACHE[path] = path.read_text(encoding="utf-8", errors="replace")
    return _LIB_CACHE[path]


def _raw_block(lib_file: Path, name: str) -> str:
    text = _lib_text(lib_file)
    m = re.search(r'\n\s*\(symbol\s+"%s"' % re.escape(name), text)
    if not m:
        raise KeyError(f"symbol {name} not in {lib_file}")
    return balanced(text, text.index("(", m.start()))


def load_symbol(symdir: Path, lib: str, name: str) -> LibSymbol:
    lib_file = symdir / f"{lib}.kicad_sym"
    block = _raw_block(lib_file, name)
    if (m := re.search(r'\(extends\s+"([^"]+)"\)', block)):
        # flatten: parent's graphics + pins, child's properties
        parent = m.group(1)
        pblock = _raw_block(lib_file, parent)
        child_props = dict(re.findall(r'\(property\s+"([^"]+)"\s+"([^"]*)"', block))
        for key, val in child_props.items():
            pblock = re.sub(r'(\(property\s+"%s"\s+)"[^"]*"' % re.escape(key),
                            lambda mm: f'{mm.group(1)}"{val}"', pblock, count=1)
        block = pblock.replace(f'(symbol "{parent}_', f'(symbol "{name}_')
        block = block.replace(f'(symbol "{parent}"', f'(symbol "{name}"', 1)
    block = block.replace(f'(symbol "{name}"', f'(symbol "{lib}:{name}"', 1)

    pins: dict[str, Pin] = {}
    for m in re.finditer(r'\(pin\s+(\w+)\s+\w+', block):
        pb = balanced(block, m.start())
        at = re.search(r'\(at\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\)', pb)
        num = re.search(r'\(number\s+"([^"]*)"', pb)
        nam = re.search(r'\(name\s+"([^"]*)"', pb)
        ln = re.search(r'\(length\s+([-\d.]+)\)', pb)
        if at and num and num.group(1) not in pins:
            pins[num.group(1)] = Pin(num.group(1), nam.group(1) if nam else "", m.group(1),
                                     float(at.group(1)), float(at.group(2)),
                                     int(float(at.group(3))) % 360, float(ln.group(1)) if ln else 0)

    # body extents from graphics (properties excluded) + pin tips
    geo = strip_blocks(strip_blocks(block, "(property "), "(pin ")
    pts = [(float(a), float(b)) for a, b in
           re.findall(r'\((?:start|end|xy|mid|center)\s+([-\d.]+)\s+([-\d.]+)\)', geo)]
    for p in pins.values():
        dx, dy = DIRS.get(p.angle, (0, 0))
        pts += [(p.x, p.y), (p.x + dx * p.length, p.y - dy * p.length)]
    xs, ys = [p[0] for p in pts] or [0], [p[1] for p in pts] or [0]
    return LibSymbol(f"{lib}:{name}", block, pins, (min(xs), min(ys), max(xs), max(ys)),
                     power="(power" in block[:400])


def _uid(*parts: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "/".join(parts)))


def symbol_uid(project: str, ref: str) -> str:
    """Default uuid of symbol `ref`: give the footprint KIID_PATH("/<it>") for parity."""
    return _uid(project, ref)


@dataclass
class Placed:
    ref: str
    sym: LibSymbol
    x: float
    y: float
    uid: str

    def pin_xy(self, num: str) -> tuple[float, float]:
        p = self.sym.pins[num]
        return snap(self.x + p.x), snap(self.y - p.y)

    def body(self) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = self.sym.bbox
        return self.x + x0, self.y - y1, self.x + x1, self.y - y0  # sheet coords


@dataclass
class Schematic:
    symdir: Path
    project: str
    title: str
    paper: str = "A4"
    company: str = "hermes-kicad-pcb"
    power_nets: tuple[str, ...] = ("GND", "+5V", "+3V3", "+12V", "VCC", "VBUS")
    root: str = ""
    parts: dict[str, Placed] = field(default_factory=dict)
    items: list[str] = field(default_factory=list)
    extents: list[tuple[float, float, float, float]] = field(default_factory=list)
    lib: dict[str, LibSymbol] = field(default_factory=dict)
    taken: dict[tuple[float, float], str] = field(default_factory=dict)  # point -> net
    _pwr_n: int = 0

    def __post_init__(self):
        self.root = self.root or _uid(self.project, "root")

    # -------------------------------------------------------------- symbols
    def _symbol(self, lib: str, name: str) -> LibSymbol:
        key = f"{lib}:{name}"
        if key not in self.lib:
            self.lib[key] = load_symbol(self.symdir, lib, name)
        return self.lib[key]

    def add(self, ref: str, lib: str, name: str, x: float, y: float, value: str,
            footprint: str = "", uid: str | None = None) -> Placed:
        sym = self._symbol(lib, name)
        pl = Placed(ref, sym, snap(x), snap(y), uid or symbol_uid(self.project, ref))
        self.parts[ref] = pl
        bx0, by0, bx1, by1 = pl.body()
        if bx1 - bx0 < 6:     # narrow vertical passive: fields to the right of the body
            fx, jr = bx1 + 1.27, " (justify left)"
            ry, vy = pl.y - 1.27, pl.y + 1.27
        elif by1 - by0 < 6:   # short horizontal part (LED, diode): centred above / below
            fx, jr = (bx0 + bx1) / 2, ""
            ry, vy = by0 - 1.27, by1 + 1.905
        else:                 # IC / connector: both above the body (ground pins own the
            fx, jr = bx0, " (justify left)"   # bottom edge), left aligned
            ry, vy = by0 - 5.08, by0 - 2.54
        pins = "".join(f'(pin "{n}" (uuid "{_uid(pl.uid, n)}"))' for n in sym.pins)
        self.items.append(
            f'(symbol (lib_id "{sym.lib_id}") (at {pl.x} {pl.y} 0) (unit 1)\n'
            f'  (exclude_from_sim no) (in_bom yes) (on_board yes) (dnp no)\n'
            f'  (uuid "{pl.uid}")\n'
            f'  (property "Reference" "{ref}" (at {snap(fx)} {snap(ry)} 0) {FONT.format(extra=jr)})\n'
            f'  (property "Value" "{value}" (at {snap(fx)} {snap(vy)} 0) {FONT.format(extra=jr)})\n'
            f'  (property "Footprint" "{footprint}" (at {pl.x} {pl.y} 0) '
            f'{FONT.format(extra=" (hide yes)")})\n'
            f'  {pins}\n'
            f'  (instances (project "{self.project}" (path "/{self.root}" '
            f'(reference "{ref}") (unit 1)))))')
        tw = max(len(ref), len(value)) * 1.0
        self.extents += [(bx0, by0, bx1, by1), (fx - (tw / 2 if not jr else 0), min(ry, vy) - 1.5,
                                                fx + tw, max(ry, vy) + 1)]
        return pl

    # -------------------------------------------------------------- connections
    def _wire(self, a: tuple[float, float], b: tuple[float, float]) -> None:
        self.items.append(f'(wire (pts (xy {a[0]} {a[1]}) (xy {b[0]} {b[1]})) '
                          f'(stroke (width 0) (type default)) (uuid "{_uid(self.root, "w", str(a), str(b))}"))')
        self.extents.append((min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])))

    def _power(self, net: str, at: tuple[float, float], out: int, flag: bool = False) -> None:
        name = "PWR_FLAG" if flag else net
        sym = self._symbol("power", name)
        body = next(iter(sym.pins.values())).angle           # body sits along the pin angle
        rot = (out - body) % 360
        self._pwr_n += 1
        ref = f"#{'FLG' if flag else 'PWR'}{self._pwr_n:02d}"
        dx, dy = DIRS[out]
        gap = 3.81 if out in (90, 270) else 3.3
        tx, ty = at[0] + dx * gap, at[1] + dy * gap          # value text beyond the body
        jx = {0: " (justify left)", 180: " (justify right)"}.get(out, "")
        fa = 90 if rot in (90, 270) else 0  # field angles turn with the symbol: undo it
        uid = _uid(self.root, ref)
        self.items.append(
            f'(symbol (lib_id "power:{name}") (at {at[0]} {at[1]} {rot}) (unit 1)\n'
            f'  (exclude_from_sim no) (in_bom no) (on_board yes) (dnp no) (uuid "{uid}")\n'
            f'  (property "Reference" "{ref}" (at {at[0]} {at[1]} 0) {FONT.format(extra=" (hide yes)")})\n'
            f'  (property "Value" "{name}" (at {snap(tx)} {snap(ty)} {fa}) {FONT.format(extra=jx)})\n'
            f'  (property "Footprint" "" (at {at[0]} {at[1]} 0) {FONT.format(extra=" (hide yes)")})\n'
            f'  (pin "1" (uuid "{_uid(uid, "1")}"))\n'
            f'  (instances (project "{self.project}" (path "/{self.root}" '
            f'(reference "{ref}") (unit 1)))))')
        w = len(name) * 1.0 + 1
        self.extents.append((min(at[0], tx) - (w if out == 180 else 1.5), min(at[1], ty) - 1.5,
                             max(at[0], tx) + (w if out != 180 else 1.5), max(at[1], ty) + 1.5))

    def _label(self, net: str, at: tuple[float, float], out: int) -> None:
        just = "left" if out in (0, 90) else "right"
        self.items.append(
            f'(global_label "{net}" (shape passive) (at {at[0]} {at[1]} {out}) '
            f'{FONT.format(extra=f" (justify {just})")} (uuid "{_uid(self.root, "gl", net, str(at))}"))')
        w = len(net) * 1.0 + 3
        dx, dy = DIRS[out]
        ex, ey = at[0] + dx * w, at[1] + dy * w
        self.extents.append((min(at[0], ex) - 1.3, min(at[1], ey) - 1.3,
                             max(at[0], ex) + 1.3, max(at[1], ey) + 1.3))

    def _junction(self, at: tuple[float, float]) -> None:
        self.items.append(f'(junction (at {at[0]} {at[1]}) (diameter 0) (color 0 0 0 0) '
                          f'(uuid "{_uid(self.root, "j", str(at))}"))')

    def connect(self, ref: str, pin: str, net: str, stub: float = 2.54,
                source: bool = False) -> None:
        """Wire out of the pin (away from the body), then a global label (signals) or a
        power symbol in its conventional pose: ground points down, supplies point up.
        source=True adds a PWR_FLAG on a side branch (net fed from outside the sheet)."""
        pl = self.parts[ref]
        tip = pl.pin_xy(pin)
        if (have := self.taken.get(tip)) is not None:
            if have != net:
                raise ValueError(f"{ref}.{pin}: point {tip} already carries {have}, not {net}")
            return  # stacked pin already connected
        self.taken[tip] = net
        out = (pl.sym.pins[pin].angle + 180) % 360

        def step(pt, d, n=stub):
            return snap(pt[0] + DIRS[d][0] * n), snap(pt[1] + DIRS[d][1] * n)

        if net not in self.power_nets:
            end = step(tip, out)
            self._wire(tip, end)
            self._label(net, end, out)
            return
        natural = 270 if net.upper().startswith(("GND", "VSS")) else 90
        run = 4 * stub if source else stub  # room for the flag branch + both texts
        if out == natural:                                 # straight out
            path = [tip, step(tip, out, run)]
        elif out in (0, 180):                              # sideways pin: bend to natural
            corner = step(tip, out)
            path = [tip, corner, step(corner, natural, run)]
        else:                                              # pin faces the wrong way: loop out
            corner = step(tip, out)
            side = step(corner, 0, 2 * stub)
            path = [tip, corner, side, step(side, natural, run)]
        for a_, b_ in zip(path, path[1:]):
            self._wire(a_, b_)
        self._power(net, path[-1], natural)
        if source:
            # branch off the last run, away from the pin, and hang the flag there
            mid = step(path[-2], natural, 2 * stub)
            away = out if out in (0, 180) else 0
            fl = step(mid, away)
            self._wire(mid, fl)
            self._junction(mid)
            self._power(net, fl, 90, flag=True)          # upright flag, text on top

    def no_connect_rest(self) -> int:
        """No-connect flag on every pin position that carries no net."""
        n = 0
        for pl in self.parts.values():
            for num, p in pl.sym.pins.items():
                tip = pl.pin_xy(num)
                if tip in self.taken or p.etype == "no_connect":
                    continue
                self.taken[tip] = ""
                self.items.append(f'(no_connect (at {tip[0]} {tip[1]}) (uuid "{_uid(self.root, "nc", str(tip))}"))')
                n += 1
        return n

    # -------------------------------------------------------------- output
    def bbox(self, margin: float = 5.0) -> tuple[float, float, float, float]:
        xs0, ys0, xs1, ys1 = zip(*self.extents)
        return min(xs0) - margin, min(ys0) - margin, max(xs1) + margin, max(ys1) + margin

    def sym_libs(self) -> set[str]:
        return {k.split(":", 1)[0] for k in self.lib}

    def write(self, path: Path) -> Path:
        libs = "\n    ".join(s.block for s in self.lib.values())
        body = "\n  ".join(self.items)
        path.write_text(
            f'(kicad_sch (version 20260101) (generator "hermes_sch_gen")\n'
            f'  (uuid "{self.root}")\n  (paper "{self.paper}")\n'
            f'  (title_block (title "{self.title}") (company "{self.company}"))\n'
            f'  (lib_symbols\n    {libs}\n  )\n  {body}\n'
            f'  (sheet_instances (path "/" (page "1")))\n)\n', encoding="utf-8")
        return path


# ------------------------------------------------------------------ project libs

def write_lib_tables(outdir: Path, sym_libs, fp_libs) -> None:
    """Project-local sym-lib-table / fp-lib-table for the libraries a design uses.
    A fresh KiCad install (CI runners, new machines) has NO global tables until the
    GUI runs once, and ERC/DRC then report lib_symbol_issues / lib_footprint_issues.
    Paths use KiCad's own install variables, so the project stays portable."""
    def table(kind: str, var: str, libs, ext: str) -> str:
        rows = "\n".join(f'  (lib (name "{n}") (type "KiCad") (uri "${{{var}}}/{n}{ext}") '
                         f'(options "") (descr ""))' for n in sorted(set(libs)))
        return f"({kind}\n  (version 7)\n{rows}\n)\n"
    (outdir / "sym-lib-table").write_text(
        table("sym_lib_table", "KICAD10_SYMBOL_DIR", sym_libs, ".kicad_sym"), encoding="utf-8")
    (outdir / "fp-lib-table").write_text(
        table("fp_lib_table", "KICAD10_FOOTPRINT_DIR", fp_libs, ".pretty"), encoding="utf-8")


# ------------------------------------------------------------------ images

def export_svg(cli: str, sch: Path, out_svg: Path, bbox: tuple[float, float, float, float]) -> bool:
    """Schematic -> SVG cropped to `bbox` (sheet mm), no drawing sheet, white background."""
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([cli, "sch", "export", "svg", "-e", "-o", td, str(sch)],
                       capture_output=True, text=True, timeout=300)
        svgs = list(Path(td).glob("*.svg"))
        if not svgs:
            return False
        svg = svgs[0].read_text(encoding="utf-8")
    m = re.search(r'viewBox="([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+)"', svg)
    wm = re.search(r'<svg[^>]*\swidth="([\d.]+)mm"', svg)
    if not m or not wm:
        out_svg.write_text(svg, encoding="utf-8")
        return True
    # viewBox units per mm (KiCad plots in its own internal units)
    k = float(m.group(3)) / float(wm.group(1))
    x0, y0, x1, y1 = bbox
    vb = f"{x0 * k:.3f} {y0 * k:.3f} {(x1 - x0) * k:.3f} {(y1 - y0) * k:.3f}"
    svg = svg.replace(m.group(0), f'viewBox="{vb}"', 1)
    svg = re.sub(r'(<svg[^>]*\s)width="[\d.]+mm"', rf'\g<1>width="{x1 - x0:.1f}mm"', svg, count=1)
    svg = re.sub(r'(<svg[^>]*\s)height="[\d.]+mm"', rf'\g<1>height="{y1 - y0:.1f}mm"', svg, count=1)
    bg = f'<rect x="{x0 * k:.3f}" y="{y0 * k:.3f}" width="{(x1 - x0) * k:.3f}" height="{(y1 - y0) * k:.3f}" fill="#FFFFFF"/>'
    svg = re.sub(r'(<svg[^>]*>)', lambda mm: mm.group(1) + "\n" + bg, svg, count=1)
    out_svg.write_text(svg, encoding="utf-8")
    return True


def svg_to_png(svg: Path, png: Path, width_px: int = 1600) -> bool:
    """Optional PNG via pymupdf (any python that has it) or rsvg-convert/inkscape."""
    code = ("import sys,pymupdf as m; d=m.open(sys.argv[1]); p=d[0]; "
            "z=float(sys.argv[3])/p.rect.width; p.get_pixmap(matrix=m.Matrix(z,z)).save(sys.argv[2])")
    pys = [os.environ.get("PNG_PYTHON", ""), shutil.which("python3") or "", shutil.which("python") or ""]
    for py in filter(None, pys):
        r = subprocess.run([py, "-c", code, str(svg), str(png), str(width_px)],
                           capture_output=True, text=True, timeout=120)
        if r.returncode == 0 and png.exists():
            return True
    if (rs := shutil.which("rsvg-convert")):
        subprocess.run([rs, "-w", str(width_px), "-b", "white", "-o", str(png), str(svg)], timeout=120)
        return png.exists()
    return False
