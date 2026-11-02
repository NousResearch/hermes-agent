# COMMIT-SHEET — <project>

> Seven decisions before the first line of code. A generic answer ("modern, clean") means the decision hasn't been made — stop and make it. Filled example answers below each field: ✗ is what slop looks like, ✓ is the bar.

## 1. Peak / Signature
<!-- direct: the ONE wow moment. build: the one element a visitor describes to a friend. -->
…
> ✗ "beautiful animations throughout"
> ✓ "peak = scene 4: the espresso machine disassembles into 9 floating parts as you scroll (canvas sequence, 300vh pinned)"

## 2. Color
<!-- primary as OKLCH + tier (restrained / committed / full-palette / drenched) + why it's not lavender, not cream, not the category reflex
     + the BACKGROUND LIGHTNESS as a number: target mean L, and one line on why the page lives at that level.
     "Dark because it's premium" is not a reason — it is the single most common place this skill drifts. -->
…
> ✗ "purple gradient, feels techy" · "dark theme, feels premium"
> ✓ "oklch(0.58 0.19 35) burnt terracotta, tier: committed (~40% of surface). Not the AI lavender; not wellness-beige — terracotta is pulled from the product's clay housing. Background L ≈ 0.55: the machine is photographed in a daylit workshop, and a black page would make the clay read as ceramic-shop-at-night"

## 3. Type
<!-- display + text pair on a contrast axis + why not Inter -->
…
> ✗ "Inter for everything, it's readable"
> ✓ "display: Fraunces (soft wide serif, brand's warmth) / text: Söhne-class grotesque via 'Archivo'. Axis: high-contrast serif × neutral grotesque. Inter rejected as the 2024–26 default"

## 4. Grid break
<!-- the ONE concrete thing that breaks the symmetric grid -->
…
> ✗ "asymmetric layout"
> ✓ "product photo in S2 crosses into S3 over the section boundary (−120px overlap), text wraps around it via shape-outside"

## 5. Motion budget
<!-- ≤3 scroll-pattern families, named -->
…
> ✓ "(1) scrub-pinned hero, (2) per-word text reveals on headings, (3) depth parallax in proof section. Nothing else scroll-triggered"

## 6. Reflex check
<!-- (a) what a generic AI does for this category; (b) what a generic AI avoiding (a) does; (c) our argued deviation from both.
     If recon ran, (a) is evidence, not a guess — cite what design/refs/REFERENCES.md showed repeatedly. -->
a) …
b) …
c) …
> ✓ "a) coffee = warm beige + serif + steam photo; b) 'specialty' = black + neon + brutalist menu; c) ours = terracotta drench + technical cutaway drawings of the machine (the brand is engineering-led, not café-cozy)"

## 7. House tells broken
<!-- The two (minimum) items from taste.md §2.5 you are deliberately NOT doing this time, and what replaces each.
     (a) and (b) above are reflexes of the CATEGORY; these are the reflexes of this SKILL, which recur across
     projects that have nothing to do with each other. Naming them is the only thing that stops them. -->
1. …
2. …
> ✓ "1. near-black background → daylit L 0.55 paper, the shadows do the drama instead; 2. mono service labels in the corners → no chrome at all, the only type on screen is the headline and one caption"
