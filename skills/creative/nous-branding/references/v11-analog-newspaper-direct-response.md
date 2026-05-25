# v11 Lane 2 — Analog Newspaper Direct Response

## Purpose

**Analog Newspaper Direct Response** is a v11 candidate lane for release announcements, product updates, model drops, launch posts, and institutional notices that should feel like a recovered newspaper advertisement, mail-order insert, classified listing, or coupon-backed direct-response broadside.

This lane is for **copy-heavy announcement energy**: serif headlines, dense columns, price/version blocks, coupon boxes, and product cutouts embedded in washed newsprint.

It is not a clean magazine ad, not a modern landing page hero, and not a generic technical manual. It should feel like a strange high-signal product announcement printed in a newspaper and archived badly.

---

## Primary Reference

Canonical local directory: `~/Desktop/Nous-branding-refs/`

| Role | File | Characteristics |
|---|---|---|
| Primary lane DNA | `Hermes4.3-reference .png` | version/release hierarchy, product announcement structure, direct-response/ad energy |
| Print discipline | `style-rough-print.jpg` | aged paper, scuffed ink, manual-cover distress |
| Xerox/newsprint support | `style-cyan-xerox-poster.jpg` | scanlines, thresholding, black ink on pale stock |
| Optional sparse structure | `style-minimal-stipple-border.jpg` | coupon/border restraint, negative space, institutional framing |

---

## Core Visual Cues

- Newspaper page, tabloid ad, mail-order insert, classified block, or clipped coupon
- Serif headlines, bold price/version blocks, coupons, ordering forms, dense columns of microcopy
- Product photo/cutout or symbolic object embedded into editorial copy
- Washed gray/tan newsprint, rough halftone, visible paper tooth, ink spread
- `NOUS` as masthead, advertiser, coupon stamp, product label, or classified header
- Direct-response copy hierarchy: headline → offer/version → proof block → coupon/order box
- Imperfect alignment, registration drift, uneven inking, aging, folded/handled paper

---

## Palette

Primary:
- **Newsprint Cream** — `#E3DCC7`
- **Washed Gray Ink** — `#5F625C`
- **Xerox Black** — `#151411`
- **Aged Tan Paper** — `#CDBE9C`
- **Oxidized Brown** — `#6A4E36`

Signal accents:
- **Coupon Red** — `#9B2F24`
- **Classified Blue** — `#1E4F72`

Use one signal accent only. Red is preferred for rules, price circles, coupon stamps, and urgent direct-response marks.

---

## Reference Ordering

Use 2-3 refs. The release/direct-response reference must come first.

```bash
# Standard newspaper direct-response announcement
-i Hermes4.3-reference\ .png style-rough-print.jpg style-cyan-xerox-poster.jpg

# More restrained institutional coupon / order form
-i Hermes4.3-reference\ .png style-minimal-stipple-border.jpg style-rough-print.jpg

# Rougher classifieds / xerox newspaper ad
-i Hermes4.3-reference\ .png style-cyan-xerox-poster.jpg style-rough-grain.jpg
```

Rule: `Hermes4.3-reference .png` controls announcement hierarchy and direct-response grammar. Print refs should add newsprint decay and paper discipline, not turn the output into a generic punk poster.

---

## Prompt Block

Append this block to Analog Newspaper Direct Response prompts:

```text
STYLE TARGET: Analog Newspaper Direct Response — a strange high-signal newspaper advertisement / mail-order insert / classified product announcement, not a clean magazine ad and not a modern landing-page hero.
Use newspaper grammar: serif masthead headline, dense column copy, price/version block, coupon box or order form, small proof marks, product cutout or symbolic device embedded in editorial layout.
Make `NOUS` appear as the advertiser masthead, product stamp, coupon seal, classified header, or printed label inside the newspaper artifact.
Use washed newsprint cream/tan/gray paper, xerox black ink, rough halftone dots, ink spread, paper fibers, folded-page wear, scanner dust, slightly crooked columns, uneven leading, registration drift, and localized exposure loss.
The microcopy can be mostly texture, but the main hierarchy should be legible: NOUS, release/product name, version/offer, and one coupon/action phrase.
Keep palette constrained: newsprint cream, washed gray, xerox black, aged tan, plus one small coupon-red or classified-blue accent.
Avoid glossy advertising, modern SaaS layout, clean vector icons, perfect typography, smooth AI illustration, rainbow colors, luxury product photography, and generic cyberpunk UI.
Borrow only direct-response layout grammar, paper treatment, palette, and mood from references; do not copy the exact reference layout.
```

---

## Prompt Templates

### Release Classified Ad

```text
Original Nous Research release announcement as a recovered newspaper classified advertisement. Masthead reads `NOUS RESEARCH`; main serif headline reads `[release name]`; version `[version]` appears in a price/block badge. Dense columns of tiny product copy surround a rough product cutout or symbolic device. Bottom third contains a clipped coupon box reading `[action phrase]`.
[Analog Newspaper Direct Response Prompt Block]
--ar 4:5
```

### Mail-Order Agent Kit

```text
Original mail-order newspaper insert for a fictional Nous agent kit. Large serif headline, `NOUS` advertiser masthead, product illustration/cutout in black newsprint halftone, comparison chart as dense texture, red coupon seal, and an order form with blank lines. It should feel like a weird archived direct-response ad for technical cognition infrastructure.
[Analog Newspaper Direct Response Prompt Block]
--ar 4:5
```

### Tabloid Product Broadside

```text
Original tabloid-style product broadside for Nous Research. Oversized serif headline, loud but constrained red price/version circle, two narrow columns of ad copy, testimonial pull-quote block, and a torn coupon along the bottom. Printed on yellowed gray newsprint with xerox breakup and imperfect column alignment.
[Analog Newspaper Direct Response Prompt Block]
--ar 16:9
```

---

## Post-Processing

Use the shared imprint postprocess pipeline:

```bash
python3 /Users/johann/.hermes/profiles/palantir/skills/creative/nous-branding/scripts/postprocess.py \
  output-raw.png output-final.png --mode imprint --intensity 0.5
```

Calibration:
- `0.45-0.55` for copy-heavy outputs where hierarchy needs to stay readable.
- `0.6-0.7` for poster-like proofs where text can break down into texture.
- Avoid `0.75+` unless exact copy is irrelevant; it can destroy coupon and headline fidelity.

If exact typography matters, typeset the main labels after generation. Treat most microcopy as texture.

---

## QA Checklist

Passes if:
- It reads immediately as newspaper/classified/direct-response print.
- The primary hierarchy is visible: `NOUS`, release/product, version/offer, action/coupon.
- Dense microcopy and coupon/order-form grammar are present.
- Paper and ink feel physical: newsprint, halftone, scuffs, folds, ink spread.
- Palette stays constrained with at most one signal accent.

Fails if:
- It becomes a modern landing page or magazine ad.
- It becomes a generic technical manual with no coupon/classified grammar.
- It relies on glossy product photography or clean vector illustration.
- The `NOUS` mark floats as a digital overlay instead of being printed into the page.
- The copy is either all illegible mush or too clean/perfect.

---

## Proof Notes

Local deterministic proof generated when external image generation was blocked:

- `/Users/johann/.hermes/agents/palantir/outputs/nous-branding/nous-v11-analog-newspaper-direct-response-proof.png`

The proof validates the layout grammar: masthead, serif headline, price/version block, product cutout, dense columns, coupon box, newsprint palette, and imprint post-processing.
