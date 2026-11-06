# pcbnew KiCad 10.0.6 — calls verified on Linux 2026-09-21; cross-platform resolvers (CI smoke-tested)

Import: any Python that can `import pcbnew` — on Linux distro/PPA packages it is the
system `python3`; on Windows/macOS it is KiCad's bundled Python (`demo_autoroute.py`
re-execs itself there automatically; override with the `KICAD_PYTHON` env var).
The 2–3 `PROPERTY_ENUM` asserts on stderr at import time are harmless C++ noise — ignore.

Do NOT hardcode paths: use `scripts/kicad_paths.py` (`footprints_dir()`, `kicad_cli()`,
`freerouting()`) — it resolves per-OS (env var > known locations > PATH).
`pcbnew.GetDefaultFootprintsPath()` does NOT exist in 10.0.6.

| Action | Call |
|---|---|
| New board | `pcbnew.NewBoard(path)` (path REQUIRED) |
| Open | `pcbnew.LoadBoard(path)` |
| Save | `board.Save(path)` |
| Units | `mm = pcbnew.FromMM`; coords `pcbnew.VECTOR2I(x_nm, y_nm)` |
| Shapes | `SHAPE_T_RECT`, `SHAPE_T_SEGMENT`, `SHAPE_T_ARC`, `SHAPE_T_CIRCLE`, `SHAPE_T_POLY` (UPPERCASE) |
| Rectangle | `s = pcbnew.PCB_SHAPE(board); s.SetShape(pcbnew.SHAPE_T_RECT); s.SetStart(...); s.SetEnd(...); s.SetLayer(pcbnew.Edge_Cuts); s.SetWidth(mm(0.1)); board.Add(s)` |
| Load footprint | `pcbnew.FootprintLoad("<footprints_dir>/<Lib>.pretty", "<name>")` |
| Place footprint | `f.SetReference("R1"); f.SetPosition(pcbnew.VECTOR2I(...)); f.SetOrientationDegrees(90); board.Add(f)` |
| Find FP/pad | `board.FindFootprintByReference("R1")`, `fp.FindPadByNumber("2")` |
| Create net | `net = pcbnew.NETINFO_ITEM(board, "N$1"); board.Add(net)` |
| Bind pad | `pad.SetNet(net)` |
| Track | `t = pcbnew.PCB_TRACK(board); t.SetStart(pad1.GetPosition()); t.SetEnd(pad2.GetPosition()); t.SetWidth(mm(0.25)); t.SetLayer(pcbnew.F_Cu); t.SetNetCode(p1.GetNetCode()); board.Add(t)` |
| Count | `len(board.GetTracks())`, `len(list(board.GetFootprints()))` (if iteration crashes → conditional patch below) |
| DSN export | `pcbnew.ExportSpecctraDSN(board, "out.dsn")` → True |
| SES import | `pcbnew.ImportSpecctraSES(board, "out.ses")` → True |
| DRC json | `kicad-cli pcb drc --format json --output drc.json board.kicad_pcb` |
| Import non-KiCad PCB | `board = pcbnew.PCB_IO_MGR.Load(pcbnew.PCB_IO_MGR.EASYEDAPRO, "proj.epro")` (also `EASYEDA`, `EAGLE`, `ALTIUM_DESIGNER`, ...) then `pcbnew.PCB_IO_MGR.Save(pcbnew.PCB_IO_MGR.KICAD_SEXP, "out.kicad_pcb", board)` — see `easyeda_bridge.py` |

## DRC json shape
```json
{"violations": [{"type": "solder_mask_bridge", "description": "...", "severity": "error"}],
 "unconnected_items": [], "schematic_parity": []}
```
Count by type; `unconnected_items` > 0 = unrouted connection; `schematic_parity` =
sch↔pcb mismatch (the check DRC alone does not catch). `solder_mask_bridge` on tight
boards = cosmetic.

## Freerouting 2.4.1 (headless)

**Prefer the OS bundle with embedded runtime** (releases: `linux-x64.zip`,
`windows-x64.msi`, `macos-*.dmg`) — invokes the launcher directly, ZERO system Java:

```bash
# Linux/macOS (bundle under ~/Work/tools/ is auto-detected):
~/Work/tools/freerouting-2.4.1-linux-x64/bin/freerouting -de in.dsn -do out.ses -mp 50 -mt 4
# Windows (MSI):
"C:\Program Files\Freerouting\freerouting\freerouting.exe" -de in.dsn -do out.ses -mp 50 -mt 4
```

The standalone **jar** is the fallback and requires **Java 25+** (2.4.1 = class file 69;
Java 17 dies with `UnsupportedClassVersionError`):
```bash
java -jar freerouting-2.4.1.jar -de in.dsn -do out.ses -mp 50 -mt 4
```
`-mp` = max passes, `-mt` = threads. Score 1000 + "0 unrouted" = fully routed.
Large boards stalling: `-mp 200`, fewer threads. Or let the resolvers find everything:
`python3 demo_autoroute.py --stage route`.

## Compatibility patch — CONDITIONAL (not part of a standard install)

Only if iteration crashes with `'SwigPyIterator' object has no attribute 'next'`
(python 3.14 + older bindings, e.g. Arch/omarchy). Find the file:
`python3 -c "import pcbnew; print(pcbnew.__file__)"` — 3 occurrences of:
`item = it.next()` → `item = it.__next__() if hasattr(it, '__next__') else it.next()`
On Ubuntu 24.04 + KiCad 10.0.6 (python 3.12) NO patch is needed.

## what does NOT exist in kicad-cli 10.0.6
`pcb export dsn`, `pcb import ses`, `sch import` — use the Python functions above.
`kicad-cli pcb import` only handles: pads, altium, eagle, cadstar, fabmaster, pcad,
solidworks (NOT EasyEDA — that one lives in `PCB_IO_MGR` via Python).
kicad-cli has: `pcb {drc,export,import,render,upgrade}`, `sch {erc,export}`, `fp {export svg,upgrade}`, `sym upgrade`.
