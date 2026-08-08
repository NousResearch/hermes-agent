---
name: ardmag-image-generation
description: Produce hero / social images for ARDmag blog articles by detecting the real products mentioned in the article, mapping them to real product assets under backend/static/images/<handle>/, and assembling an organic workshop-scene prompt. Refuses to reuse generic category assets when specific products are named, refuses collage/cutout/floating-product framing, and emits a cache-busting semantic filename plus a production-validation checklist.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [creative, image-generation, ardmag, blog, hero, social]
    related_skills: [visual-asset-review, baoyu-article-illustrator]
    category: creative
    homepage: https://ardmag.ro
---

# ARDmag Image Generation

Generate hero and social images for ARDmag blog articles that look like editorial photography from a stone workshop, with the actual products from the article integrated organically into the scene.

## When to Use

- A new ARDmag blog article needs a hero image
- An existing hero is conceptually wrong (generic, missing product labels, doesn't match the article's products)
- Social / OG previews are needed for an existing article
- The production site still shows an old hero after a hero swap (cache-busting may be required)

This skill is for the `ardmag.com/site` content repo. Articles live at `backend-storefront/content/blog/*.md`. Product assets live at `backend/static/images/<handle>/`.

## Required Inputs

- Path to the article `.md` (frontmatter + body)
- Asset root: `<site-root>/backend/static/images/`
- Optional: catalog CSV at `<site-root>/docs/catalog_products-*.csv` or `<site-root>/resources/Wix Products Catalog.csv`

## Procedure

### 1. Build the image brief

Find the skill's script dir:

```bash
SKILL_DIR=$(dirname "$(find ~/.hermes/skills -path '*/ardmag-image-generation/SKILL.md' 2>/dev/null | head -1)")
# If the skill is not yet installed into the hub, fall back to the repo copy:
SKILL_DIR=${SKILL_DIR:-/home/dc/.hermes/hermes-agent/optional-skills/creative/ardmag-image-generation}
```

Run the brief builder:

```bash
python "$SKILL_DIR/scripts/ardmag_brief.py" build-brief \
  --article path/to/article.md \
  --assets-root path/to/backend/static/images \
  --catalog path/to/catalog.csv     # optional
```

The brief is JSON with:

- `brand` — inferred brand (`delta-research`, `tenax`, `mixed`, or `none`)
- `products` — list of `{name, slug, asset_dir, sample_image, in_catalog}` for every product mentioned in the article
- `prompt` — the assembled image-gen prompt, ready to pass to `image_generate`
- `suggested_filename` — semantic, cache-busting filename for the new hero
- `frontmatter_patch` — the new `heroImage:` value to write to the article frontmatter
- `validation_checklist` — steps to verify in production after deploy

If `products` is empty or only contains generic category matches (e.g. `solutii-delta`, `tratamente-specifice`), STOP and ask the user. Generic-only matches are the failure mode this skill exists to prevent.

### 2. Generate the image

Pass `brief.prompt` to `image_generate`. The prompt is pre-loaded with anti-collage / anti-cutout guidance and references the real product names. Do not strip those instructions.

```python
result = image_generate(prompt=brief["prompt"], aspect_ratio="landscape")
```

Save the generated image to the article's media directory using `brief.suggested_filename`:

```
<site-root>/backend-storefront/public/blog/<article-slug>/<suggested_filename>
```

### 3. Update the article

Open the article markdown and replace the `heroImage:` frontmatter value with `brief.frontmatter_patch`. Do NOT overwrite the existing image file with the same filename — use the new semantic name from the brief. Cache-busting is the whole point.

### 4. Verify locally

```bash
cd <site-root>
npm run build           # Next build with dummy env if needed
```

The build must succeed and the new image must resolve.

### 5. Commit, push, validate in production

```bash
git add backend-storefront/content/blog/<article>.md
git add backend-storefront/public/blog/<article-slug>/<suggested_filename>
git commit -m "Replace <article> hero with real product scene"
git push
```

Then run the validation checklist from `brief.validation_checklist`:

1. CI green on GitHub
2. Production article page returns 200 and HTML contains the new filename (not the old one)
3. Production blog list page shows the new hero card
4. The new image URL returns 200 with the expected size
5. Visual confirmation (screenshot) — products labeled, integrated, no collage

If production still shows the OLD image, the cause is one of:
- deploy hasn't completed → wait and retry
- CDN cache on the old filename → if you reused the old filename, this is why we use a new one
- frontmatter still points to the old file → re-check the patch landed

## Hard Rules (Non-Negotiable)

These rules are encoded in `prompts/hero_image.md` and applied by the brief builder. They exist because of a real incident (Delta Research hero, 2026-05-26) where each was violated:

1. **No collages.** No grid of cropped product shots. No cutouts pasted onto a background.
2. **No floating products.** No products hovering against a solid color or studio gradient.
3. **No unlabeled bottles.** If the article names specific Delta Research / Tenax products, the bottles in the scene must show those labels legibly.
4. **No wrong-brand products.** If the article is about Delta Research, do not put Tenax bottles in the scene.
5. **No generic-category-only mapping.** If the article names `SEAL`, `QUASAR`, `IDROREP` etc., the prompt must reference those specific products — not just "treatment products" or `solutii-delta`.
6. **No filename reuse.** When replacing a conceptually wrong hero, use a new semantic filename (e.g. `hero-delta-research-tratamente.webp`), not the old `hero.webp`.
7. **No "done" without production validation.** Build success is not the bar. The production HTML must reference the new filename and the image must render correctly.

## Pitfalls

- **Re-optimizing the old image instead of replacing it.** If the concept is wrong, optimization doesn't fix it. The brief builder flags articles where the current hero is named `hero.webp` / `hero.jpg` / `hero.png` (generic) as needing a semantic rename.
- **Stopping at "I pushed".** A push that doesn't propagate to production is not done.
- **Trusting frontmatter `tags` alone.** Tags may say "delta research" while the body names specific products. The detector reads both.
- **Latinized vs Romanian product names.** `Total Wet`, `Wet Seal`, `Eco Stone Pro` may appear with mixed casing. The detector normalizes case and diacritics.

## Verification

The image is acceptable if:

- The brief's `products` list is non-empty and at least one entry has `asset_dir` populated
- The generated image shows the named products with readable labels, integrated on stone / workshop context, no collage
- `suggested_filename` differs from any existing hero filename for that article
- After push, production HTML contains `suggested_filename` and the image URL returns 200
