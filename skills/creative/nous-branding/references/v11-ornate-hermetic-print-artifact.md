# v11 Lane 3 — Ornate Hermetic Print Artifact

## Purpose

**Ornate Hermetic Print Artifact** is a v11 candidate lane for occult/institutional Nous graphics that should feel like an archival broadside, alchemical seal, Victorian engraving, ritual certificate, or recovered institutional print.

This lane is for **dense symbolic authority**: ornamental borders, seals, medallions, classical or mythic figures, stamps, parchment, and layered archival marks.

It is not the older luminous/cinematic Hermetic lane. The subject can be mythic or occult, but the visual system is primarily print artifact construction: engraving, letterpress, stamp, seal, border, medallion, and archive.

---

## Primary Reference

Canonical local directory: `~/Desktop/Nous-branding-refs/`

| Role | File | Characteristics |
|---|---|---|
| Primary lane DNA | `Hermes-jam-reference.png` | ornate institutional artifact energy, central emblem, layered marks |
| Classical support | `Hermes-classical-reference .png` | statue/classical/hermetic figure language |
| Agent/product support | `Hermes-agent-ollama-reference .png` | product/agent semantics inside ornate frame |
| Psyche support | `Nous-psyche-reference.png` | mind/consciousness emblem tone |
| Print discipline | `style-rough-print.jpg` | aged paper, scuffed ink, letterpress distress |

---

## Core Visual Cues

- Victorian engraving, alchemical broadside, occult certificate, archival seal sheet
- Dense ornamental border architecture, corner flourishes, medallions, seals, stamps
- Centered mythic emblem, abstract agent glyph, classical figure, orb, eye, torch, or instrument
- Parchment/tan/cream substrate with black, brown, or dark teal ink
- Institutional typography: engraved serif, small caps, stamped labels, archival numbers
- `NOUS` as engraved masthead, seal text, stamped archival mark, certificate title, or medallion inscription
- Layered marks: accession numbers, proof seals, registration crosses, marginalia, worn ink
- Tactile damage: paper fibers, foxing, folds, scratches, ink pickup variation, plate pressure

---

## Palette

Primary:
- **Aged Parchment** — `#D8C79F`
- **Oxidized Ink** — `#1E1913`
- **Archive Brown** — `#68442C`
- **Deep Verdigris** — `#173D3F`
- **Bone Highlight** — `#EFE2C0`

Signal accents:
- **Seal Red** — `#7E241D`
- **Cobalt Registration** — `#1E4D7C`
- **Gold Ochre** — `#A77A2D`

Use one signal accent. Seal red is preferred for stamps, rules, wax/seal effects, or archival warnings.

---

## Reference Ordering

Use 2-4 refs. The ornate/hermetic reference must come first.

```bash
# Standard ornate hermetic artifact
-i Hermes-jam-reference.png Hermes-classical-reference\ .png style-rough-print.jpg

# Agent/product certificate variant
-i Hermes-jam-reference.png Hermes-agent-ollama-reference\ .png style-rough-print.jpg

# Consciousness / psyche emblem variant
-i Hermes-jam-reference.png Nous-psyche-reference.png Hermes-classical-reference\ .png style-rough-print.jpg
```

Rule: `Hermes-jam-reference.png` controls the artifact/broadside grammar. Classical/agent/psyche refs provide subject semantics. `style-rough-print.jpg` adds physical print discipline only.

---

## Prompt Block

Append this block to Ornate Hermetic Print Artifact prompts:

```text
STYLE TARGET: Ornate Hermetic Print Artifact — a recovered alchemical broadside / Victorian engraving / institutional occult certificate, not a luminous fantasy poster and not clean digital illustration.
Build the image as a physical print artifact: dense ornamental border, corner flourishes, central medallion or mythic emblem, seals, stamps, small-cap serif labels, archival accession numbers, marginalia, and layered institutional marks.
Make `NOUS` appear as an engraved masthead, certificate title, seal inscription, stamped archival mark, or medallion text printed into the artifact.
Use aged parchment/tan/cream paper, oxidized black-brown or dark teal ink, one seal-red/cobalt/gold accent, visible paper fibers, foxing, plate pressure, letterpress scuffs, scratches, ink dropout, halftone/engraving linework, misregistration, and worn edges.
The central subject should feel engraved, stamped, or printed — not rendered as a glossy 3D/fantasy figure. Prefer symbolic forms: orb, eye, torch, classical fragment, agent glyph, alchemical instrument, nested circles, triangles, and constellation-like network lines.
Avoid neon cyberpunk glow, smooth fantasy illustration, modern vector logo polish, generic occult clipart, overstuffed random sigils, rainbow palette, and readable paragraphs of fake text.
Borrow only print artifact grammar, border density, seal/stamp hierarchy, palette, and mood from references; do not copy the exact reference layout.
```

---

## Prompt Templates

### Agent Seal Broadside

```text
Original Nous Research agent seal broadside as an ornate hermetic print artifact. A centered medallion contains an abstract agent glyph made from nested circles, an eye/orb, and fine network lines. `NOUS` is engraved as the masthead and repeated around a circular seal. Dense Victorian border, corner flourishes, archival stamps, accession numbers, seal-red proof mark, parchment substrate.
[Ornate Hermetic Print Artifact Prompt Block]
--ar 4:5
```

### Classical Archive Certificate

```text
Original archival certificate for a Nous research instrument. A close classical statue fragment or mythic figure is rendered as engraved linework inside a dense ornamental frame. `NOUS RESEARCH` appears as the certificate title; small stamped labels and medallions surround the figure. The page feels foxed, folded, pressed, and recovered from an archive.
[Ornate Hermetic Print Artifact Prompt Block]
--ar 4:5
```

### Alchemical Product Sheet

```text
Original alchemical product sheet for a fictional Nous model release. Central emblem: torch, orb, and nested geometric instrument. Surrounding rings contain tiny proof labels, version number `[version]`, and stamped `NOUS` marks. Dark teal-brown engraving ink on aged parchment, seal-red verification stamp, dense border architecture.
[Ornate Hermetic Print Artifact Prompt Block]
--ar 1:1
```

---

## Post-Processing

Use the shared imprint postprocess pipeline:

```bash
python3 /Users/johann/.hermes/profiles/palantir/skills/creative/nous-branding/scripts/postprocess.py \
  output-raw.png output-final.png --mode imprint --intensity 0.6
```

Calibration:
- `0.5-0.6` for engraving detail and seal legibility.
- `0.65-0.75` if the raw output is too clean or too luminous.
- Avoid very high intensity when borders contain fine linework; excessive thresholding can collapse ornament into mud.

If exact title typography matters, typeset the masthead and seal text after generation.

---

## QA Checklist

Passes if:
- It reads as a physical archival/hermetic print artifact.
- Border architecture, seals, stamps, medallions, and engraved linework are present.
- `NOUS` is embedded as masthead/seal/certificate text, not floating overlay.
- The central subject is printed/engraved/stamped, not glossy rendered.
- Palette stays parchment + dark ink + one signal accent.

Fails if:
- It becomes luminous fantasy/cyberpunk mysticism.
- It becomes clean vector occult clipart.
- It has random sigils without institutional print hierarchy.
- It lacks a dense border/seal system.
- It copies the reference layout rather than translating the print grammar.

---

## Proof Notes

Local deterministic proof generated when external image generation was blocked:

- `/Users/johann/.hermes/agents/palantir/outputs/nous-branding/nous-v11-ornate-hermetic-print-artifact-proof.png`

The proof validates the lane grammar: dense border, masthead, medallion, seals, parchment palette, archival stamps, and imprint post-processing.
