# Shopify Alt Text Rules

Use this reference when generating or reviewing alt text candidates.

## Length

- Shopify allows image alt text up to 512 characters, but recommends 125 characters or fewer.
- Use 60-120 characters as the normal target.
- Treat 125 characters as the default soft limit and 512 characters as the hard stop.
- Never submit an alt text longer than 512 characters.
- Rewrite overlong alt text into a shorter complete phrase instead of truncating mid-sentence.

## Quality

Write concise, truthful descriptions of the image in its page context.

Prefer:

- "Black leather tote bag with gold zipper on a white background"
- "Winter skincare gift set with cleanser, serum, and moisturizer"
- "Diagram showing checkout upsell placement in a Shopify cart"

Avoid:

- Keyword lists.
- Repeating the same product title for every image.
- "Image of", "picture of", or "photo of" unless the medium itself matters.
- Claims not visible in the image or supported by surrounding context.
- Decorative details that do not help a shopper or screen-reader user.

## Multimodal Strategy

Use real image understanding for every Alt Text candidate. Do not trust model self-report.

The main agent must test image capability with a real local image input, not a capability question. The model must answer from the pixels, not from the filename, product title, collection title, article title, OCR libraries, metadata, URL, or surrounding text.

Downloading a file is not enough. The agent must open, attach, or view the local image through the host environment's real multimodal image pathway before using `source: vision`.

A valid vision result must include concrete visual evidence, such as color, object type, layout, background, visible text, material, shape, or scene. If the answer only repeats Shopify context or product naming, reject it as context-only.

A vision model is mandatory for this skill's advertised Alt Text result. The agent must download a current image sample to an operating-system temp directory, open it through the host environment's real image-input pathway, and verify at least three pixel-derived facts before generating candidates. If image download, image opening, or pixel interpretation fails, stop with `VISION_MODEL_REQUIRED`; do not produce context-only candidates and do not apply changes.

## Duplicate Control

Within the same product, collection, or article:

- Do not reuse the exact same alt text.
- If two candidates differ only by punctuation or generic words, treat them as duplicates.
- Use grounded distinctions only, such as color, variant, angle, packaging, lifestyle setting, diagram, screenshot, chart, or media position.
- If no grounded distinction exists, prefer a conservative phrase and mark the item for review.

## Resource-Specific Rules

Product media:

- Prefer updating the `MediaImage` file alt through `fileUpdate`.
- Detect and report shared file risk when one `MediaImage` is referenced by more than one product.

Collection featured image:

- Update through `collectionUpdate`.
- Preserve the existing image URL and change only `altText`.

Article featured image:

- Update through `articleUpdate.image.altText`.
- Preserve the existing image URL.

Article inline images:

- Update the `alt` attribute inside `article.body` HTML.
- Preserve every existing `src`, `data-src`, width, height, class, and unrelated attribute.
- Do not rewrite article copy unless the user explicitly asks.
