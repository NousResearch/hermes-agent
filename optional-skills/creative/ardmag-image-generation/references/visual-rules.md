# ARDmag visual rules — quick reference

Captured from the 2026-05-26 Delta Research hero incident and from the broader ARDmag visual direction. The brief builder enforces the prompt-level rules automatically; this document captures the editorial intent that humans should also check.

## Source-of-truth principles

1. **Organic realistic scenes.** Stone workshop / atelier with actual stone samples and worn wood. Not a studio backdrop, not a website hero gradient, not a 3D render.
2. **Same-source previews.** The OG / social preview and the article hero share the same scene. No swapping in a different aesthetic for social.
3. **Article-specific products only.** Use the exact products the article names. If the article says SEAL and QUASAR, the bottles in the image are SEAL and QUASAR.
4. **Generic categories are a fallback, not a default.** Mapping a Delta Research article to `solutii-delta/` instead of to the specific products is the bug, not the feature.
5. **Tenax appears only when the article is about Tenax.** Cross-brand bottles in a Delta scene are wrong, even if they look photogenic.
6. **Varied makers in industry-overview articles.** When an article surveys the category (e.g. "how to choose"), more than one brand may appear — but the brand assignment must still match what the article names.
7. **Context-correct products.** A hydrophobic treatment article should not feature polishing pads in the foreground.
8. **No dark logo patches.** No black bar/strip behind logos.

## Cache-busting filename convention

When replacing a hero whose concept was wrong, do not reuse the existing filename. Pattern:

```
hero-<brand-slug>-<topic-slug>.webp
```

Examples:
- `hero-delta-research-tratamente.webp`
- `hero-tenax-mastici.webp`
- `hero-pad-uri-velcro-7-gradatii.webp` (no brand → just topic)

If the existing hero is `hero.webp`, `hero.jpg`, `hero.png`, or `cover.*`, that's a "generic name" signal — rename on replacement.

## Production validation steps

After push and CI green:

1. `curl -sI https://ardmag.ro/blog/<article-slug>/` → 200
2. View source, grep for the new filename → must be present, old must be absent
3. Hit the image URL directly → 200, expected dimensions
4. Open the blog list page → new card thumbnail matches
5. Take a screenshot of the live page → file with the run

If any step fails, the work is not done. Do not report success on local build alone.
