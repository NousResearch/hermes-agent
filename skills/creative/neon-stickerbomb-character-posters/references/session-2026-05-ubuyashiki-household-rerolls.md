# Demon Slayer Ubuyashiki household rerolls — 2026-05

## Trigger
Nick repeatedly requested variants of:

- `生成 / 重新生成 产屋敷耀哉 发色不同的双胞胎产屋敷日香和产屋敷天音 鬼舞辻无惨`

## Durable lessons

1. **Parse the twin phrase as one combined subject.**
   - `发色不同的双胞胎产屋敷日香和产屋敷天音` should become one poster containing both twins, not separate individual portraits.
   - Keep both girls visibly childlike, covered, wholesome, and non-sexualized.

2. **Put hair-color distinction first.**
   - State the contrast in the first lines, before style mechanics:
     - `产屋敷日香 = deep black hair with violet-blue rim light, left side`
     - `产屋敷天音 = pale white/silver hair with lavender rim light, right side`
   - Add negatives: `no same hair color`, `no two black-haired twins`, `no two white-haired twins`, `no missing twin`, `no adult bodies`.

3. **Use a composition that makes the distinction legible.**
   - Good skeleton: side-profile split twin layout with a diagonal talisman gap between them, faces unobscured, oversized typography pushed to top/bottom borders.
   - Alternative: circular motion ring, but ensure one twin is assigned lower-left foreground and the other upper-right midground.

4. **Retry only failed slots.**
   - In this session, long prompts for 产屋敷耀哉 and 鬼舞辻无惨 sometimes returned `empty_response`, while the twin image succeeded.
   - Preserve successes and retry only failed slots with compact anchor-first prompts.

5. **Compact retry pattern**

```text
Vertical 3:4 neon sticker-bomb poster of [subject]. [Core identity anchors]. [One composition sentence]. huge cropped typography, cyan/magenta rim light, thick black comic outlines, glossy cel shading, halftone, torn decals, graffiti, barcode labels. Small NickZag on [specific object]. No centered pose, no watermark, no photorealism, no sexualization/cute-face drift.
```

## Useful anchors

### 产屋敷耀哉
- long white hair
- gentle pale noble face
- blind/marked eyes
- dark formal kimono / haori
- calm fragile family-head dignity
- good compact composition: foreground glowing wisteria lantern and talisman strips, subject off-left in quiet side profile

### 发色不同的双胞胎：产屋敷日香 / 产屋敷天音
- one combined image
- 日香: deep black hair + violet-blue rim light, left side
- 天音: pale white/silver hair + lavender rim light, right side
- childlike proportions, covered matching kimono layers, calm shrine-family expressions
- good compact composition: side-profile split twin layout with diagonal talisman gap

### 鬼舞辻无惨
- pale flawless face
- sharp red eyes
- black wavy hair
- white fedora
- black luxury suit
- cold demon-king aristocrat
- if direct prompt returns empty, phrase as `a pale demon king aristocrat inspired by 鬼舞辻无惨` while preserving the decisive visual anchors
