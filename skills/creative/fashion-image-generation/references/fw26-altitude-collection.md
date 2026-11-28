# FW26 Altitude — verified collection manifest & Drive topology

Working example of the two-step pipeline, run end-to-end 2026-08-06 on the OpenRouter
gpt-image API (`openai/gpt-5.4-image-2`, key from `.env` `OPENROUTER_API_KEY`). Reuse
these verified file→swatch mappings to resume the batch WITHOUT re-identifying swatches
by vision. Drive owner:  alberto@adelante.digital.

## Drive topology — CAREFUL: two parallel structures exist
There are **TWO** ALTITUDE output trees on this Drive and only one is the user's real target:

✅ **THE CORRECT destination** (where the user looks / where inputs and outputs belong):
```
ALTITUDE (1AZbezfiPHJ54YA6dmexa5d27xOwSbZq)
  └─ "HELMUR - ALTITUDE" (1KNdlz9F3e)
       └─ MARKETING (16mC2V6LVE)
            └─ FW 26 (15lCMoAv1x)
                 └─ UOMO (1yQ5-AZLvdsv_KBjWDW9uckKPzNTS3Hog)
                      ├─ OXFORD  (1No9QGTqeI5ZkZUQ10d-AUn0CZ4mWKiPf)  → _OUTPUT/
                      ├─ PORTLAND(1bgWDqAy4aP-62s1N0ZaVIRbIcp3YvYjU)  → _OUTPUT/
                      ├─ BORMIO   (1wihxZU0ZGg0l3az4l05xdfdJx09za-iL)
                      └─ LE MANS  (1UADhwhf8ovwreOCFOdBqbnXga-sBjG1k)
```
⚠️ **DO NOT use this one** — a parallel `GEN IMAGE` clone (ALTITUDE > GEN IMAGE > UOMO,
OXFORD 1yHFvgwZhjLHKvfZLTz0ud9-JE_veDW-4 / PORTLAND 1UsdqrrdBO9b0S3SDnJLX_us9v-rpg9kW)
exists and LOOKS identical but is NOT what the user sees/expects. Uploading there confuses
the user ("Oxford è vuoto"). When in doubt, verify the parent chain: the correct structure
contains `HELMUR - ALTITUDE > MARKETING > FW 26` in its ancestry; the wrong one contains
`GEN IMAGE`. ALWAYS confirm with the user which tree to use before a full batch, or lift the
destination from where the ghost `IMG_*.jpeg` inputs live.

Per garment: base folder holds the **ghost mannequin** (`IMG_*.jpg`, product shot on flat
grey) + **swatches** (`Bazaart_*.jpeg`, flat tint with `COLOR - CODE` label). Outputs go in
`<CAPO>/_OUTPUT/<COLORE>/` as `ghost_<COLORE>.png` + `indossato_front_<COLORE>.png`.

Verified FW26 `_OUTPUT` folder IDs (created 2026-08-06, in the CORRECT tree):
- OXFORD/_OUTPUT   = `12iIrs_Or9zi1bvNonF7_EkFCGbhmyrDj`
- PORTLAND/_OUTPUT = `1Xc2RP_I-JSu31-joUYgDd8niZbGh-eOj`

## Verified manifest (ghost + swatch file → color)
Model for donna (STEP B ref 1): **`MODELLE/m1.jpg`** (dark-haired female, standing, front).

**ALASKA** — ghost `ALASKA/IMG_5967.jpg` (cropped jacket: faux-fur yoke + cable-knit sleeves)
- `NOCCIOLA-302` ← `ALASKA/Bazaart_663DFE11.jpeg`
- `CENERE-201`   ← `ALASKA/Bazaart_C71C7788.jpeg`
- `ARTIC-101`    ← `ALASKA/Bazaart_D3CC3D77.jpeg`
- `NERO-999`     ← `ALASKA/Bazaart_E23DEE80.jpeg`

**MONTANA** — ghost `MONTANA/IMG_5958.jpg` (long shearling double-breasted coat)
- `MASTICE-202`  ← `MONTANA/Bazaart_16914EC0.jpeg`
- `CACAO-304`    ← `MONTANA/Bazaart_2B14168A.jpeg`
- `MOKA-305`     ← `MONTANA/Bazaart_300F2971.jpeg`
- `PANNA-102`    ← `MONTANA/Bazaart_B35C1478.jpeg`
- `NERO-999`     ← `MONTANA/Bazaart_E23DEE80.jpeg`

**OXFORD** (UOMO, no model provided yet) — ghosts among `OXFORD/IMG_6408..IMG_6425.jpeg`;
colors per spec: ANTRACITE-85, OLIVA-68, BLU-78, PERLA-88, NOCCIOLA-29.
**PORTLAND** (UOMO, no model provided yet) — ghost `PORTLAND/IMG_6397.jpeg`;
colors per spec: ARMY-62, LAVAGNA-82, PIETRA-83, NERO-99, BLU-77.

## Verified costs (measured, not estimated)
- `gpt-5.4-image-2`, `quality=high`, `aspect_ratio=2:3`, 1 reference ≈ **$0.19/image**
  (STEP A ≈ $0.1898, STEP B ≈ $0.1839 — both ~$0.19, near-constant per call).
- STEP A + STEP B for one colorway ≈ $0.37.
- Full donna batch (9 colorways, ALASKA 4 + MONTANA 5, = 18 images) ≈ **$3.17**.

## Resume next session
1. Download any missing garment ghost/swatch files from the Drive topology above.
2. Fill the MANIFEST in `templates/batch_gen.py` (ghost + swatch map per garment; add
   `model:` = `MODELLE/<model>.jpg` for STEP B).
3. Run `batch_gen.py` in background (`notify_on_complete`); it is idempotent so a re-run
   is free. Then upload into each `_OUTPUT/<COLORE>/` folder (re-query folder IDs, do not
   trust a copied listing).