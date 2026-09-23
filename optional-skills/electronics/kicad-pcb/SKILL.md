---
name: kicad-pcb
description: 'Headless KiCad PCB automation using pcbnew Python, kicad-cli and optional Freerouting DSN/SES on Linux, Windows and macOS. Use for an existing, electrically reviewed KiCad design when scripting board placement/routing, running ERC/DRC, importing EasyEDA PCB geometry or exporting manufacturing files after the project release gate. This skill does not select circuits, certify electrical safety or replace RF, mechanical and fab review.'
---

# KiCad PCB automation (headless, cross-platform)

Design, route, validate and export PCBs without the GUI: pcbnew Python for board
manipulation, `kicad-cli` for DRC/ERC/export, Freerouting headless for autorouting
(DSN/SES loop), and a headless bridge that imports EasyEDA / EasyEDA Pro boards into
KiCad. **Validated end-to-end on Linux** (KiCad 10.0.6 + Freerouting 2.4.1);
Windows and macOS are covered by runtime path resolvers + a re-exec shim and
smoke-tested on CI — the same commands run on any OS, every path is resolved at
runtime by `scripts/kicad_paths.py` (env var > known location > PATH).

## Toolchain

| Component | Requirement | Notes |
|---|---|---|
| KiCad | 10+ | `kicad-cli` + pcbnew Python bindings (come with KiCad) |
| Freerouting | 2.4+ | **bundle with embedded runtime preferred** — no system Java needed; plain jar needs **Java 25+** (jar 2.4.1 = class file 69) |
| Java | only for the jar | bundle linux-x64.zip / windows-x64.msi / macos .dmg embed their own runtime |
| easyeda2kicad | optional | LCSC part → KiCad symbol+footprint+3D (`pipx install easyeda2kicad`) — used by the bridge for components missing from the official libs |

Per-OS install:
- **Linux**: distro package or PPA — bindings import from system `python3`.
- **Windows**: KiCad installer (includes python + pcbnew) + Freerouting MSI. Resolvers find `C:\Program Files\...` automatically; MSI path may vary by version — `FREEROUTING_EXE` overrides anything.
- **macOS**: KiCad app bundle + Freerouting .dmg.
- Bundles: https://github.com/freerouting/freerouting/releases — unzip to `~/Work/tools/` (auto-detected default) or anywhere.

Environment overrides (all optional):

| Var | Purpose |
|---|---|
| `KICAD_PYTHON` | python interpreter that has pcbnew |
| `KICAD_CLI` | path to `kicad-cli` |
| `KICAD10_FOOTPRINT_DIR` / `KICAD_FOOTPRINT_DIR` | official footprints dir |
| `FREEROUTING_EXE` | launcher binary of a bundle |
| `FREEROUTING_JAR` | freerouting.jar (then system Java 25+ is required) |

## Preflight + pipelines (one command each)

```bash
# 0) preflight: lists what is missing and HOW to fix it (exit 1 only with --strict)
python3 skills/kicad-pcb/scripts/check_env.py

# 1a) smallest end-to-end board, ONLY KiCad needed (no Freerouting/Java):
#     header -> R -> LED; schematic + board + ERC + DRC with schematic parity
#     + gerbers/drill zip + BOM/pos + 3D render. Validated on Windows 11 + Linux (KiCad 10.0.6).
python3 skills/kicad-pcb/scripts/simple_board.py --out /tmp/led --vin 5 --led-ma 3

# 1b) reference pipeline: board -> DSN -> Freerouting -> SES -> DRC (exit 0 = PASS)
python3 skills/kicad-pcb/scripts/demo_autoroute.py --out /tmp/pcb-demo

# 2) full example (ESP32 + LED + USB-C): schematic + board + autoroute + renders
python3 skills/kicad-pcb/scripts/esp32_example.py --out /tmp/esp32
#    outputs: esp32-devboard.kicad_pcb/.kicad_sch, renders/*.png, drc json

# 3) import an EasyEDA / EasyEDA Pro PCB into KiCad
python3 skills/kicad-pcb/scripts/easyeda_bridge.py import-pro project.epro --out outdir/
#    LCSC part -> KiCad symbol/footprint/3D (needs easyeda2kicad installed)
python3 skills/kicad-pcb/scripts/easyeda_bridge.py lcsc C2040 --out outdir/lcsc-lib

# isolated demo stages
python3 demo_autoroute.py --stage create   # board + DSN
python3 demo_autoroute.py --stage route    # autoroute (uses FREEROUTING_EXE/JAR)
python3 demo_autoroute.py --stage import   # imports SES -> routed board
python3 demo_autoroute.py --stage drc      # DRC of an existing board (no pcbnew needed)

# flags: --route-timeout 600  --max-passes 50  --threads 4
# exit codes: 0 ok | 2 environment | 3 stage | 4 DRC FAIL
```

On Windows/macOS, `demo_autoroute.py` / `esp32_example.py` re-exec themselves in
KiCad's bundled Python when `import pcbnew` fails in the current interpreter —
nothing to configure.

## EasyEDA bridge (what it does and does not)

- `import-pro PROJECT.epro` — converts a whole **EasyEDA Pro** PCB to `.kicad_pcb`
  using KiCad's own parser via `pcbnew.PCB_IO_MGR` (headless, no GUI). Validated with
  a real 45-footprint/1068-track board: geometry, nets, zones and outline come
  through (0 unconnected / 0 parity in DRC). Schematics are NOT converted — re-draw
  them or import via KiCad GUI (File → Import Non-KiCad Project → EasyEDA Pro).
- `import-std` — EasyEDA **Standard** JSON (`easyeda/sources/pcb/document.json` inside
  the project zip, or the json itself): tracks, vias, pads, nets and the board
  outline convert. **Caveat (verified):** COPPER_AREA shapes land as misplaced
  drawings (canvas-offset conversion is lossy) — after importing a Standard board,
  re-create its copper zones in KiCad; net names and tracks are fine.
- `lcsc Cxxxxx` — downloads symbol + footprint + 3D model from LCSC via
  `easyeda2kicad` (optional dependency; graceful error with install hint if absent).
  Validated: C14663 → sym + pretty + wrl + step.
- After importing: run `--stage drc` on the output, then route missing nets with the
  autorouter pipeline.

## CRITICAL pitfalls (learned the hard way)

1. **Python interpreter**: any python that can `import pcbnew`. Linux distro/PPA: system `python3`. Windows/macOS: KiCad's bundled python (the scripts re-exec automatically). Override with `KICAD_PYTHON`. The 2–3 `PROPERTY_ENUM` asserts on stderr at import are harmless C++ noise.
2. **py3.14 SWIG patch is CONDITIONAL** (Arch/omarchy + KiCad pkg with python 3.14): only if iteration crashes with `'SwigPyIterator' object has no attribute 'next'` — patch `pcbnew.py` (find via `python3 -c "import pcbnew; print(pcbnew.__file__)"`), 3× `item = it.next()` → `item = it.__next__() if hasattr(it, '__next__') else it.next()`. On Ubuntu 24.04 + KiCad 10.0.6 (python 3.12) NO patch is needed.
3. **Freerouting java version**: the plain jar 2.4.1 needs Java 25+ (Java 17 = `UnsupportedClassVersionError` class file 69). Preferred: OS bundle with embedded runtime — zero Java dependency.
4. `pcbnew.NewBoard(path)` requires the filename arg (KiCad 10).
5. Board outline on Edge.Cuts is REQUIRED before DRC/autorouting. `SHAPE_T_RECT` (uppercase) etc.; coords `VECTOR2I` in nm; `mm = pcbnew.FromMM`.
6. `kicad-cli` 10.0.6 does NOT export DSN / import SES — use `pcbnew.ExportSpecctraDSN(board, path)` / `pcbnew.ImportSpecctraSES(board, path)`. `kicad-cli pcb import` does NOT know EasyEDA — that import lives in Python (`PCB_IO_MGR`).
7. Review every DRC warning against the actual fabrication process. A `solder_mask_bridge` warning may require a mask expansion, aperture or stencil change; it is not automatically cosmetic. Unconnected items and schematic parity are additional gates, not substitutes for warning review.
8. Multi-pad numbers (e.g. ESP32-WROOM-32 thermal pad): several sub-pads share one number — set the net on ALL of them via `fp.Pads()`, else same-number pads short.
9. Symbol pin names ≠ pad names: map via pin number; derived symbols (`extends`) have no own pins — resolve parent and rename nested unit blocks.
10. Footprint anchor is often pad 1, NOT the center. Placement math MUST use real bounding boxes (pads + F.Fab); NEVER `GetBoundingBox()` (includes text + antenna keepout, destroys packers).
11. ESP32/RF modules: use the exact module manufacturer's antenna placement and copper/ground keepout guidance. Edge overhang and clearance are package- and board-specific; inspect the return path and enclosure too.
12. Project-level design rules live in `.kicad_pro`; set track, clearance, via and drill limits from the selected fabricator's current stack/service and the approved net classes. The example's 0.127 mm track and 0.2 mm drill are demo settings, not universal minima.
13. Bounding-box packing with a gap worked in the ESP32 demo (177 tracks), but it proves neither collision freedom nor electrical, RF, thermal or assembly quality on another board. Inspect courtyards, height, orientation, keepouts and service access.
14. Schematic generation: use `scripts/sch_gen.py` (format 20260101, KiCad 10.0.6) rather
    than hand-writing s-expressions. It encodes what bit us: **library Y grows UP, sheet
    Y grows DOWN** (pin `(px, py)` of a symbol at `(x, y)` is `(x + px, y - py)`; `+ py`
    silently put +3V3 on the ESP32 GND pins); `extends` symbols are flattened (else they
    embed with no body/pins); every pin gets a wire stub + label/power symbol pointing
    AWAY from the body; unused pins get no-connect flags; PWR_FLAG marks external
    supplies; everything on the 1.27 mm grid. Pick the symbol whose pins match the
    footprint 1:1 (16-pin GCT USB-C -> `USB_C_Receptacle_USB2.0_16P`, not the 24-pin
    one). Leave pins you have NOT designed (e.g. ESP32 EN) unflagged so ERC keeps
    reporting them; investigate every `pin_not_driven` instead of suppressing it.
15. Imported EasyEDA boards keep the ORIGINAL design rules of the source project — expect DRC violations against KiCad/JLCPCB defaults (a real import showed 498). Triage them; don't blanket-fix.
16. Edge connectors (USB-C etc.): rotate so the mouth faces OUT and slide the
    footprint until its `PCB Edge` line (Dwgs.User) sits on the outline — a
    rot-0 horizontal receptacle points its opening INTO the board. Modules that
    overhang on purpose (ESP32 antenna) leave silkscreen past the edge: do not edit
    the footprint (that trades it for `lib_footprint_mismatch`) and do not relax the
    global severity — write a scoped, commented rule in `<board>.kicad_dru`:
    `(rule "..." (layer "F.Silkscreen") (constraint silk_clearance (min -100mm))
    (condition "A.memberOfFootprint('U1')") (severity ignore))`. (`A.Parent.Reference`
    and `edge_clearance` do NOT match this check.) Prove the scope with a negative test.
17. Post-route stitching vias go through BOTH layers: test candidate spots against
    foreign-net tracks and vias (not only pads) and against vias already placed,
    or a routing change silently turns into `shorting_items` / `holes_co_located`.
    KiCad 10: a via's width is per layer, `via.GetWidth(pcbnew.F_Cu)`.
18. Schematic parity end-to-end: give symbols an `(instances ...)` block, footprints
    `SetPath(KIID_PATH("/<symbol uuid>"))` + `SetFPID(LIB_ID(lib, fp))`, and take the
    board nets FROM the schematic (`kicad-cli sch export netlist --format kicadxml`)
    instead of typing them twice — including the single-pin `unconnected-(...)` nets of
    no-connect pins. The XML prints `/` in those auto names, KiCad stores `{slash}`
    (`unconnected-(U1-SDI{slash}SD1-Pad22)`). `pcb drc --schematic-parity` and custom
    rules are looked up by the BOARD's file name: copy `.kicad_sch` / `.kicad_dru` next
    to a renamed board (e.g. `*-routed.kicad_pcb`), or parity silently checks nothing.
19. Fresh installs (CI runners, a new laptop) have NO global `sym-lib-table` /
    `fp-lib-table` until the KiCad GUI runs once: ERC/DRC then report
    `lib_symbol_issues` / `lib_footprint_issues` / `footprint_link_issues` for every
    part, although your own machine is clean. Always write project-local tables
    (`sch_gen.write_lib_tables`, URIs via `${KICAD10_SYMBOL_DIR}` /
    `${KICAD10_FOOTPRINT_DIR}`). Reproduce locally with an empty `KICAD_CONFIG_HOME`.
    On GitHub Actions: logs need a login but annotations are public — emit
    `::error::` with the output tail; pwsh turns native exit codes into exit 1 unless
    `$PSNativeCommandUseErrorActionPreference = $false`.

## JLCPCB-class fab rules (2-layer)

track ≥ 0.127mm (use 0.2mm), clearance ≥ 0.127mm, via 0.45mm Ø / 0.2mm drill,
annular ring 0.125mm, **board min 6×6mm**, thickness 0.4–2.4mm (default 1.6).
4+ layers: 0.09mm track, via 0.25/0.15. Full tables, assembly (Economic vs
Standard), PCBA BOM/CPL upload workflow and rotation-offset cautions:
**`references/jlcpcb-rules.md`** (adapted from aklofas/kicad-happy, MIT).

## Reviewed-design workflow

The [official GPT-6 Astra example](https://openai.com/pt-BR/index/gpt-6-astra/) shows a KiCad PCB layout with placement and copper routing. It does not specify Freerouting, DSN/SES, headless operation or a particular GUI/API technique. The following is this skill's optional headless workflow for an **already reviewed** circuit:
1. One project folder with datasheet + reference designs.
2. One complete brief: supply voltage, channel count, load impedance, component tech, size limit.
3. Verify the reference design against the CURRENT datasheet revision.
4. Schematic as native KiCad objects (multi-sheet OK), not images.
5. Verify every footprint before layout: pin 1, pad numbering, polarity, land dims.
6. Set design rules to the fab values you will order against; fixed outline; then place.
7. Define and LOCK critical copper first (supply, bootstrap, switching).
8. If Freerouting is appropriate, use it only for ordinary nets after protecting critical RF, high-current, switching, USB and timing-sensitive routes; inspect the imported result, refill zones and confirm protected copper survived.
9. Ship libraries with the project (sym-lib-table / fp-lib-table via ${KIPRJMOD}).
10. Run ERC, DRC, unconnected and schematic-parity checks together; review all warnings and documented exceptions. A zero-item report does not certify electrical safety, RF behavior or factory fit. Export production files only after the project's independent review and release gate.

## Visual feedback loop (mandatory before release)

Every board iteration MUST be checked with vision against reality, not just DRC:
1. Render the board (`kicad-cli pcb render --quality high`, top AND bottom) and export the STEP with real 3D models (`scripts/attach_easyeda_3d.py`).
2. Look at the render with vision and audit: connectors on board edges with openings facing outward; antennas clear of copper AND parts; rotations aligned to natural trace flow and pick-and-place; decoupling caps at their IC's power pins; user-facing items (LEDs, buttons) at reachable edges.
3. Compare against REAL photos of commercial equivalents (vendor docs, review photos). Known-good baselines: Espressif ESP32-DevKitC V4 (EN + Boot buttons, USB-UART bridge, power LED, dual I/O headers), official datasheet layout guidance (docs.espressif.com PCB Layout Design).
4. Copy proven blocks from real designs (open-source dev boards, vendor EVMs) and modify — a layout pattern that already shipped beats a novel one.
5. Record the gap list found by comparison and use it as the next iteration's task list.

## Project deliverables contract (mandatory for every board)

Every board project — KiCad-only or KiCad+EasyEDA — must ship the SAME
software-independent package. The source of parts (official KiCad libs, EasyEDA/LCSC
import) never changes the deliverable format: a consumer of the project needs only KiCad.

1. **Validated design**: native `.kicad_sch` + `.kicad_pcb` + `.kicad_pro` (ERC/DRC/
   unconnected/parity reviewed; exceptions documented with fab justification).
2. **Board 3D**: package-verified 3D models on every footprint
   (`attach_easyeda_3d.py` with LCSC STEP, or kicad-library-3d) + board-level **STEP**.
3. **Project document**: `make_report.py --project DIR --lcsc REF=Cxxxxx ...` writes
   `REPORT.md` with the standard prints: schematic figure, 3D renders top+bottom,
   fabrication-ready **BOM** (refs/qty/value/package/LCSC/JLCPCB Basic-Extended),
   validation summary and a pre-order review checklist.
4. **Fabrication pack**: gerbers + drill + CPL (pos) + BOM CSV, zipped against the
   selected fab's rules (`references/jlcpcb-rules.md`).

The Visual feedback loop (above) gates the release of this package — run it BEFORE
declaring the project done, and attach the gap list found by comparison as the next
iteration's task list. A generated 3D model set and the report document are part of
"done", not optional extras.

## EasyEDA 3D models

`easyeda_bridge.py lcsc Cxxxxx` downloads symbol + footprint + 3D (WRL+STEP) via
`easyeda2kicad`. Then `scripts/attach_easyeda_3d.py board.kicad_pcb --map REF=model.step
--step out.step --render out.png` attaches models and exports the board-level STEP.
ALWAYS verify the downloaded model name matches the footprint package (model names
carry the package, e.g. `SOT-223-4P_L6.5-W3.5-H1.6-LS7.0-P2.30`, `R0603`) — a
mismatched shell corrupts the STEP and hides mechanical conflicts.

## Raw pcbnew calls (quick reference)

```python
import pcbnew
mm = pcbnew.FromMM
board = pcbnew.NewBoard("out.kicad_pcb")            # arg REQUIRED
edge = pcbnew.PCB_SHAPE(board); edge.SetShape(pcbnew.SHAPE_T_RECT)
edge.SetStart(pcbnew.VECTOR2I(mm(0), mm(0))); edge.SetEnd(pcbnew.VECTOR2I(mm(30), mm(20)))
edge.SetLayer(pcbnew.Edge_Cuts); edge.SetWidth(mm(0.1)); board.Add(edge)
f = pcbnew.FootprintLoad(FP_DIR + "/Resistor_SMD.pretty", "R_0603_1608Metric")
f.SetReference("R1"); f.SetPosition(pcbnew.VECTOR2I(mm(8), mm(10))); board.Add(f)
net = pcbnew.NETINFO_ITEM(board, "N$1"); board.Add(net)
f.FindPadByNumber("2").SetNet(net)
board.Save("out.kicad_pcb")
pcbnew.ExportSpecctraDSN(board, "out.dsn")
# freerouting: bundle launcher or java -jar (Java 25+), flags -de out.dsn -do out.ses -mp 50 -mt 4
board = pcbnew.LoadBoard("out.kicad_pcb"); pcbnew.ImportSpecctraSES(board, "out.ses")
# import EasyEDA Pro PCB (headless):
board = pcbnew.PCB_IO_MGR.Load(pcbnew.PCB_IO_MGR.EASYEDAPRO, "proj.epro")
pcbnew.PCB_IO_MGR.Save(pcbnew.PCB_IO_MGR.KICAD_SEXP, "out.kicad_pcb", board)
```

## Verification + export commands

```bash
kicad-cli pcb drc --format json --output drc.json board.kicad_pcb   # violations/unconnected_items/schematic_parity
kicad-cli sch erc --format json --output erc.json board.kicad_sch
kicad-cli pcb export gerbers --layers F.Cu,B.Cu,F.Mask,B.Mask,F.Silkscreen,B.Silkscreen,F.Paste,B.Paste,Edge.Cuts --output gerbers/ board.kicad_pcb
kicad-cli pcb export drill   --output gerbers/ board.kicad_pcb
kicad-cli pcb export pos     --output pos.csv board.kicad_pcb
kicad-cli pcb render --side top --quality high -o top.png board.kicad_pcb   # 3D render PNG
kicad-cli sch export pdf -o sch.pdf board.kicad_sch                          # schematic PDF
kicad-cli sch export svg -e -o dir/ board.kicad_sch   # no frame; sch_gen.export_svg crops it
kicad-cli sch export netlist board.kicad_sch
kicad-cli fp export svg --output dir/ FootprintFile.kicad_mod
```

## Companion skills

For design **review**, BOM/sourcing and manufacturing (not covered here): install
[aklofas/kicad-happy](https://github.com/aklofas/kicad-happy) (MIT) — KiCad project
analysis/DFM, JLCPCB/LCSC/DigiKey BOM workflows, PCBA upload translator.
