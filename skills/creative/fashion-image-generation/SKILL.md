---
name: fashion-image-generation
description: "Generate fashion/e-commerce product images via cloud image APIs — recolor ghost mannequins to brand colorways, then fit garments on model references. Manifest-driven batch runs over Google Drive folders. Covers OpenRouter gpt-image models, cost control, idempotency, and Drive upload."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [fashion, ecommerce, image-generation, ghost-mannequin, recolor, virtual-try-on, openrouter, google-drive]
    related_skills: [comfyui, google-workspace, vision]
---

# Fashion E-commerce Image Generation

Generate catalog/editorial product imagery from a garment's **ghost mannequin** (the product shot on an invisible mannequin) and **color swatches** (flat tint + color name/code). This is a two-step pipeline that any cloud image-generation API (OpenRouter `gpt-image` family, etc.) can drive.

## When to use
- Recolor a garment photo to a set of brand colorways (swatches).
- Fit a garment (ghost or recolored) onto a real model reference photo (virtual try-on).
- Produce a full batch of product variants for a collection, organized per garment and per colorway.

## References & scripts
- `references/openrouter-image-api.md` — exact OpenRouter image API wire format (endpoint, `input_references`, params, billing).
- `references/openrouter-video-api.md` — OpenRouter image-to-video API (async submit/poll/download), incl. the **Seedance real-person block**, **hailuo-3 2K-only resolution** quirk, and the Drive-public-first-frame pattern. Use when asked to turn a campaign still into a vertical/editorial video.
- `references/fw26-altitude-collection.md` — verified ghost/swatch→file/color manifest + Drive topology for the FW26 Altitude batch (resume without re-identifying swatches). **NOTE: this Drive has TWO parallel ALTITUDE trees (`FW 26` vs `GEN IMAGE`); only `FW 26` under `HELMUR - ALTITUDE` is the user's real destination — see the ⚠️ in the file before uploading. Effective 2026-08-07 the final output destination is `PRODUCT IMG` (see `fw26-sku-map.md`), not `_OUTPUT` — re-run needs confirm.**
- `references/fw26-sku-map.md` — **output destination & SKU renaming** (MODIFICA 3): target `PRODUCT IMG > DONNA|UOMO > MODELLO > COLORE`, and the progressive SKU-based filename convention `<SKU>_<modello>-<colore>_<NN>_<posa>.png` derived from the `HELMUR-master-prodotti` spreadsheet tab VARIANTI. Re-derive the map from the live xlsx — never hand-maintain.
- `templates/batch_gen.py` — known-good manifest-driven batch script (STEP A + STEP B, 3 pose, idempotent, cost-logging). **Output sempre ritagliato/esportato a 800×1000 px** (`OUTPUT_SIZE`). Copy and adapt.
- `templates/organize_output.py` — renames generated outputs with the SKU convention and uploads them into `PRODUCT IMG` (idempotent per exact filename).

## The two-step pipeline
For each garment, for each colorway N:

**STEP A — recolor ghost mannequin** — inputs: [1] ghost mannequin, [2] color swatch.
Prompt: *"Recolor this ghost-mannequin garment to the exact solid color shown in the reference swatch (reference 2). Keep the garment shape, fabric texture, knit/fur structure, stitching, zippers, buttons, pockets and labels exactly identical — change only the color. Same lighting, same neutral grey background, invisible/ghost mannequin product photography. Do not add or alter any text or logos."*
Output → `ghost_<COLOR>.png`

**STEP B — worn on model** — inputs: [1] model reference photo, [2] `ghost_<COLOR>.png`.
Prompt (ecommerce studio): *"The model from the reference image wearing the garment from the reference image, {POSE}, neutral light grey background, full body shot, professional ecommerce fashion photography, clean studio lighting. Preserve all original garment details exactly as shown: stitching, hardware, zippers, buttons and labels must remain identical to the reference, do not generate any text, logos or writing on buttons, zippers or labels, keep all branding elements blank and anonymous."*
`{POSE}` ∈ front / 3-4 / back / detail. Editorial (campaign) variants swap the studio scene for a cinematic European-winter street scene.
Output → `indossato_<POSA>_<COLOR>.png`

**Proportions guard (MANDATORY on every STEP B).** The image model has a tendency to
elongate the figure (legs a little too long, the coat stretched downward), which reads
as "the model looks too tall". Always append this block to any STEP B prompt (studio or
editorial) unless the user explicitly overrides it:
> Keep the model's body proportions anatomically natural and identical to the reference
> photo: the legs must be a realistic length relative to the torso, with no elongation or
> vertical stretching of the figure, and the garment must keep its true length relative to
> the body exactly as in the ghost/reference image — do not lengthen the coat, do not
> stretch the model's height, keep the head-to-body ratio natural (not an unrealistically
> small head on a stretched body). Do not slim or stretch the limbs.

Reference for "good proportions" is the user-confirmed example: `indossato_front_MASTICE-202.png`
(Montana · Mastic) — natural leg length and true coat length. Spot-check every batch against
it; regenerate any shot that looks elongated.

**Required pose set (every garment × every colorway).** Beyond the basic e-commerce front
shot, each garment must also be produced in **two additional variants** (in addition to
`front`), always on-model and always faithful to the ghost mannequin + model reference:
- `bust34` — **three-quarter bust shot** (framed from the waist up, lower body/legs cropped out of frame), slight three-quarter angle.
- `editorial` — **dynamic editorial fashion pose** (weight shifted, one hand in the coat
  pocket or adjusting the collar), on the **same neutral studio background as all other poses**
  (editorial is a POSITION, not a location — never an outdoor/street scene).

So the full per-colorway STEP B set is: `front` (studio), `bust34` (studio), `editorial` (studio — dynamic pose).
Output naming: `indossato_<POSA>_<COLOR>.png` (e.g. `indossato_bust34_NOCCIOLA-302.png`,
`indossato_editorial_NOCCIOLA-302.png`).

**Surrounding-outfit guard (MANDATORY on every STEP B).** The model's base layers must be
coherent with the garment, elegant/modern/casual (refined knitwear, turtleneck or long-sleeve
top, full-length tailored trousers or jeans). **Never shorts, never skirts.** The reference
garment itself must remain exactly identical to the ghost (shape, details, color, relation to
body) — never add/remove/alter it.

**Outfit-consistency guard (MANDATORY per colorway).** For each color the SAME surrounding
outfit must appear across ALL three poses (front/bust34/editorial): same top, same trousers,
same shoes in every shot. Do not change the base layers between poses.

## Workflow (end-to-end)
1. **Enumerate the Drive source** — for each garment folder: identify the ghost mannequin (product photo on flat grey) vs. the color swatches (flat rectangle with a color name+code). Use vision to disambiguate and to map each swatch file → its color code (`NOCCIOLA-302`…). A single `seed` colorway may also come pre-generated.
2. **Scaffold a manifest** of `{garment: {ghost: path, swatch: {COLOR: path}}}`.
3. **Do a single paid test** (one STEP A) and show the user before launching the batch — validates ref-image format and quality for ~one image cost.
4. **Run the batch in background** (STEP A then STEP B per colorway), with cost logging and retries.
5. **Verify** outputs are valid PNGs for every expected colorway (both ghost + worn), then **upload to Drive** into `_OUTPUT/<COLOR>/` under each garment folder.

## Pitfalls (learned the hard way)
- **Model photos are required for STEP B** and are NOT stored in the Drive garment folders. If the previous pipeline used provider-hosted media IDs (e.g. Higgsfield `media_id`), those do NOT carry over to a different API — you must obtain the actual model reference JPGs (the user provides a Drive link). Ask for them before promising a full batch.
- **Verify & normalize the model reference's real format before sending it.** Drive-reported MIME/extension is not trustworthy — a file named `M1.webp` (or even `m1.jpg`) may be a WEBP. Check magic bytes (`file` / read first 12: `RIFF....WEBP`); if it's not a real JPEG/PNG, convert with PIL (`Image.open(p).convert('RGB').save(out,'JPEG',quality=95)`) before building the base64 `input_reference`, or providers can reject or mangle it. WEBP/JPG/AVIF all come up from Drive exports.
- **STEP A and STEP B must each be idempotent**: guard on the existence of the *output* file (`ghost_<C>.png` for A, `indossato_<C>.png` for B), not just the input. A step that only checks its input will re-run (and re-bill) every time on re-execution. See `templates/batch_gen.py`.
- **Cost scales linearly with image count** — at ~$0.19/img (quality high, 2:3) a 9-colorway × 2-step batch ≈ $3–4. Confirm scope before a large run.
- **Batch scripts fail silently when gated by a typo in the manifest key** (e.g. filtering on `"A":"ALASKA"` when the dict key is `"ALASKA"`) → the loop body never executes and reports a bogus `$0.00`. Validate the selector against the manifest before running.
- **Drive folder IDs are copy-sensitive**; a wrong parent ID yields `HttpError 404 File not found` on upload. Re-query the folder ID (`drive search "name='<FOLDER>'"`) rather than trusting a copied listing.
- When the Google CLI shorthand is a two-token command (`GAPI="python .../google_api.py"`), invoking it as `"$GAPI"` treats the whole string as one executable name. Call it unquoted (`$GAPI ...`) so the shell word-splits correctly.
- **`HTTP 402 Payment Required` mid-batch means the OpenRouter key ran out of credit**, not a code bug. Do NOT treat it as a retryable transient error automatically — once it persists across 3 backoff retries, stop, tell the user the credit is exhausted, and let them recharge. Then resume with an **idempotent re-run scoped to the garment** (`batch_gen.py PORTLAND --stepb`): because each step guards on its OUTPUT existing, the re-run regenerates only the single missing image and re-bills only it. Verify which outputs actually landed first (`drive search` for files modified today) so you don't leave a silent gap.
- Keep the same model/face across the collection for coherence; use one female model for donna garments and one male for uomo.
- **Drive search includes trashed files by default** — `drive search "'<folderid>' in parents"` (and bare `name='X'`) returns *trashed* items too. So verifying a folder by counting file names misleads: after a delete, the file still shows up and looks like a duplicate. To count what actually occupies the folder use `trashed = false` in the query, or read each file's `trashed` field. Fail-safe state check (what worked): list `files(id,name,trashed)` via the API and count only `not trashed`.
- **`drive delete` (google_api.py) returns `{"status":"trashed"}` immediately but the item can still appear in `drive search`** — search propagation lags / is not authoritative. An earlier "duplicate" may have already been trashed. Don't trust a search count right after a delete; verify the *active* set (see above) or `drive get` with structured fields (`files().get(fileId=..., fields="id,name,trashed")`).
- **Never dedupe Drive by "keep the oldest copy" using `createdTime`** — the CLI frequently returns it empty, so the sort key is unstable and you may delete the *good* copy rather than the extra (this happened: an `indossato_front` copy got trashed, leaving the folder short). Deterministic, safe pattern: for each target filename, **trash ALL current copies, then upload exactly one fresh copy** from the local source. This yields exactly-one state in a folder regardless of prior duplicates/legacy files and needs no fragile timestamp logic.
- **Uploading to a folder that already holds the same filenames creates duplicates** (Drive does not overwrite-on-same-name). Before (re)uploading outputs, check which of the target filenames are actually *active*; if the folder already has them, use the trash-all-upload-one pattern instead of blind upload.

## Verification
After a batch, programmatically confirm every expected `<garment>/ghost_<C>.png` and `<garment>/indossato_<C>.png` exists, is a real PNG (magic bytes) and is above a size floor (e.g. >100KB). Also do a **re-run idempotency check** with a dummy API key — it must skip all steps (no API calls) and leave the file set unchanged. Spot-check 1–2 images with vision for model/garment fidelity and "no text/logo" compliance.
