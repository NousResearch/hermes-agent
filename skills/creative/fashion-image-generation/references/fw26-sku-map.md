# FW26 Altitude — output destination & SKU renaming map

Authoritative source: `HELMUR-master-prodotti-FW26-27_V3.xlsx` (tab **VARIANTI**), owner
alberto@adelante.digital. The spreadsheet's **`Foto (rif.)`** column already defines the
target filename convention `<modello>-<colore>_NN.jpg` (e.g. `alaska-artic_01.jpg`), and
**`SKU base`** is the product SKU for that model+colorway (e.g. `ALAS-101`).

## Destination tree (Drive) — MODIFICA 3

Generated images are **not** uploaded to the per-garment `_OUTPUT` anymore. Final
destination is **`PRODUCT IMG`** with this structure:

```
PRODUCT IMG (1vy41E81IYScJOVJYYz076sCc_eBCCblN)   ← target root
  ├─ DONNA (1YqQ27arr_CvWIFPZe3XK9B-UpuhDw6l8)
  │    ├─ <MODELLO> e.g. ALASKA, MONTANA
  │    │    └─ <COLORE> e.g. 302-NOCCIOLA
  │    │         └─ <file rinominati con SKU>
  └─ UOMO  (1ZhlmWVWFEnmN6VJpTMEog9zWTQMg4Vdi)
       ├─ <MODELLO> e.g. OXFORD, PORTLAND, LEMANS
       │    └─ <COLORE>
       │         └─ <file rinominati con SKU>
```

- Create subfolders as needed (idempotent: reuse existing, else create).
- Only the **generated** images (ghost + worn variants) go here, split per colorway.
- Renaming is **progressive with the product SKU**, following this convention:

```
<SKU-base>_<modello>-<colore>_<NN>_<posa>.png
```

Examples (Donna):
- `ALAS-101_alaska-artic_00_front.png`   (o `_01` … vedi sotto)
- `ALAS-302_alaska-nocciola_00_front.png`
- `MONT-202_montana-asfalto_00_front.png`
- `MONT-202_montana-asfalto_01_bust34.png`
- `MONT-202_montana-asfalto_02_editorial.png`

`NN` = zero-padded progressive index per file (0,1,2… = front→bust34→editorial per colorway;
ghost and worn are distinct products, see "progressivo" below). Adjust numbering to the
spreadsheet `Foto (rif.)` base when the user requests `_01` start.

## SKU map (model → SKU base; from VARIANTI)

| Modello | Reparto | SKU base | Cod col | Colore | Foto (rif.) |
|---|---|---|---|---|---|
| ALASKA   | Donna | ALAS-101 | 101 | Artic     | `alaska-artic_01.jpg` |
| ALASKA   | Donna | ALAS-201 | 201 | Cenere    | `alaska-cenere_01.jpg` |
| ALASKA   | Donna | ALAS-302 | 302 | Nocciola  | `alaska-nocciola_01.jpg` |
| ALASKA   | Donna | ALAS-999 | 999 | Nero      | `alaska-nero_01.jpg` |
| CAMBRIDGE| Donna | CAMB-13  | 13  | Panna     | `cambridge-panna_01.jpg` |
| CAMBRIDGE| Donna | CAMB-21  | 21  | Sabbia    | `cambridge-sabbia_01.jpg` |
| CAMBRIDGE| Donna | CAMB-44  | 44  | Moro      | `cambridge-moro_01.jpg` |
| CAMBRIDGE| Donna | CAMB-73  | 73  | Avio      | `cambridge-avio_01.jpg` |
| CAMBRIDGE| Donna | CAMB-76  | 76  | Blu       | `cambridge-blu_01.jpg` |
| CAMBRIDGE| Donna | CAMB-80  | 80  | Cenere    | `cambridge-cenere_01.jpg` |
| CAMBRIDGE| Donna | CAMB-84  | 84  | Antracite | `cambridge-antracite_01.jpg` |
| CAMBRIDGE| Donna | CAMB-99  | 99  | Nero      | `cambridge-nero_01.jpg` |
| MONTANA  | Donna | MONT-102 | 102 | Panna     | `montana-panna_01.jpg` |
| MONTANA  | Donna | MONT-202 | 202 | Asfalto   | `montana-asfalto_01.jpg` |
| MONTANA  | Donna | MONT-304 | 304 | Cacao     | `montana-cacao_01.jpg` |
| MONTANA  | Donna | MONT-305 | 305 | Moka      | `montana-moka_01.jpg` |
| MONTANA  | Donna | MONT-999 | 999 | Nero      | `montana-nero_01.jpg` |
| LEMANS   | Uomo  | LEMA-29  | 29  | Nocciola  | `lemans-nocciola_01.jpg` |
| OXFORD   | Uomo  | OXFO-29  | 29  | Nocciola  | `oxford-nocciola_01.jpg` |
| OXFORD   | Uomo  | OXFO-68  | 68  | Oliva     | `oxford-oliva_01.jpg` |
| OXFORD   | Uomo  | OXFO-78  | 78  | Blu       | `oxford-blu_01.jpg` |
| OXFORD   | Uomo  | OXFO-85  | 85  | Antracite | `oxford-antracite_01.jpg` |
| OXFORD   | Uomo  | OXFO-88  | 88  | Perla     | `oxford-perla_01.jpg` |
| PORTLAND | Uomo  | PORT-62  | 62  | Army      | `portland-army_01.jpg` |
| PORTLAND | Uomo  | PORT-77  | 77  | Blu       | `portland-blu_01.jpg` |
| PORTLAND | Uomo  | PORT-82  | 82  | Lavagna   | `portland-lavagna_01.jpg` |
| PORTLAND | Uomo  | PORT-83  | 83  | Pietra    | `portland-pietra_83.jpg` |
| PORTLAND | Uomo  | PORT-99  | 99  | Nero      | `portland-nero_01.jpg` |

## Color-name note
The pipeline's working color tokens (e.g. `MASTICE-202`, `CENERE-201`, `NOCCIOLA-302`) map to
the spreadsheet's canonical names via **Cod. colore** (the number). `MASTICE-202` = **Asfalto**
(`montana-asfalto`); `CENERE-201` = **Cenere**; etc. Always derive the final filename from the
**spreadsheet color name**, not from the working token.

## Generate the map programmatically
Do NOT hand-maintain the table above. Re-derive it from the live spreadsheet each run:
load `HELMUR-master-prodotti-FW26-27_V3.xlsx` (tab VARIANTI), read columns
`SKU base | Modello | Reparto | Categoria | Cod. colore | Colore | … | Foto (rif.)`, and build
`{Modello: {CodColore: {sku, colore, foto, reparto}}}`. See `templates/organize_output.py`.
