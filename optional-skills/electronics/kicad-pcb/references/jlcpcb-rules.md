# JLCPCB fabrication & assembly rules (distilled)

Adapted — paraphrased, factual data only — from
[aklofas/kicad-happy](https://github.com/aklofas/kicad-happy) `skills/jlcpcb`
(MIT License, © 2025 Andrew Klofas). Values shift over time; confirm against
jlcpcb.com/capabilities before ordering.

## Standard PCB, 1–2 layers

| Parameter | Minimum |
|---|---|
| Trace width | 0.127 mm (5 mil) |
| Trace spacing | 0.127 mm (5 mil) |
| Via diameter | 0.45 mm |
| Via drill | 0.2 mm |
| Annular ring | 0.125 mm |
| Min hole | 0.2 mm |
| Thickness | 0.4–2.4 mm (default 1.6) |
| Board size | 6×6 mm to 500×400 mm |

## Multi-layer, 4+

| Parameter | Minimum |
|---|---|
| Trace width / spacing | 0.09 mm (3.5 mil) |
| Via diameter | 0.25 mm |
| Via drill | 0.15 mm |
| Thickness | 0.6–2.4 mm |

## Assembly: Economic vs Standard

| Feature | Economic | Standard |
|---|---|---|
| Sides | Top only | Top + Bottom |
| Component types | SMD only | SMD + through-hole |
| Min component size | 0201 | 01005 |
| BGA/QFP pitch | 0.5 mm | 0.4 mm |
| Extended part fee | $3/unique part | $3/unique part |

- **Basic parts**: no extra fee, pre-loaded on machines. Every part is identified
  by an LCSC number (`Cxxxxx`) — JLCPCB shares LCSC's library.
- Min order for assembly: 5 PCBs.

## PCBA upload that doesn't get rejected (3 steps)

1. BOM CSV → JLCPCB columns: `Comment`, `Designator`, `Footprint`, `LCSC Part #`
   (header must be exactly "LCSC Part #" / "LCSC Part Number").
2. CPL must contain only designators present in the BOM — drop fiducials,
   mechanical holes and test points from the placement file, or the upload is
   rejected.
3. Upload BOM first, then CPL.

kicad-happy's `skills/bom` ships `translate_bom_pnp.py` which does exactly this
translation (including the `--bom` filter mode).

## Rotation offsets, KiCad → JLCPCB pick-and-place

Some footprint families rotate differently between KiCad and JLCPCB's assembler
(commonly SOT-23 family and SOT-223 +180°, QFN +90°, SOIC +90°/+270°; connectors
vary). Verify pin-1 orientation per footprint and correct the CPL `Rotation`
column when needed; the first assembly order is the real test.

## Sourcing

- LCSC = JLCPCB's sister parts library. Free community search API (no key):
  `https://jlcsearch.tscircuit.com/api/search?q=<query>` — useful for finding
  basic-parts alternatives before BOM freeze.
- DigiKey/Mouser for prototype sourcing; LCSC for production runs.

## Importing rules into KiCad

Board Setup → Design Rules → Import Settings from a `.kicad_dru` file.
