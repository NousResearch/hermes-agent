# Product Media Alt Text Rules

Use this reference when generating or reviewing product image alt text inside `shopify-product-serp-optimizer`.

## Scope

This skill updates only product media alt text.

- Generate and review alt text inside this skill.
- Include approved alt text in the same product execution bundle as title, description, SEO title, and meta description when relevant.
- Do not route product alt text work to another skill as the default path.

## Length

- Shopify allows image alt text up to 512 characters, but target 60-120 characters.
- Keep 125 characters or fewer by default.
- Never submit alt text longer than 512 characters.
- Rewrite overlong alt text into a shorter complete phrase instead of truncating mid-sentence.

## Quality

Write concise, truthful descriptions of what is visibly in the image and why it matters in the product context.

Prefer:

- `Black leather tote bag with gold zipper and shoulder straps on a white background`
- `Laptop backpack with padded device sleeve and front organizer pocket beside a suitcase handle`
- `Stainless steel water bottle with screw-top lid shown in hand for size context`

Avoid:

- Keyword lists.
- Repeating the same product title for every image.
- `image of`, `picture of`, or `photo of` unless the medium itself matters.
- Claims not visible in the image or supported by merchant product context.
- Decorative details that do not help a shopper or screen-reader user.

## Visual-Evidence Rule

Do not claim `source: "vision"` unless the agent actually inspected pixels during the current run.

Required path for `source: "vision"`:

1. Download only the current product-image batch to an operating-system temp directory.
2. Open the local image through the host environment's real image-input pathway.
3. Record at least three pixel-derived facts.
4. Generate the final alt text from those pixel facts plus merchant context.

Downloading a file is not image inspection. Filenames, URLs, surrounding text, and product titles are not pixel evidence.

**In read-only (public) mode:** Image CDN URLs from the public JSON endpoint are publicly accessible — no API token needed to download. The visual-evidence workflow works identically regardless of access method. If the store's CDN is accessible (which it almost always is), you can download and inspect product images just as in connected mode.

## Context-Only Fallback

When direct image inspection is unavailable, generate a lower-confidence candidate only from merchant product context:

- product title
- product type
- vendor
- tags
- options and variants
- product description
- media position
- image filename or URL slug

For context-only candidates:

- Mark `source` as `context_only`.
- Mark confidence conservatively.
- Do not invent color, material, angle, packaging, dimensions, or scene details that were not visible or explicitly stated in merchant product data.
- Apply only after explicit user approval of the exact candidate.

## Duplicate Control

Within the same product:

- Do not reuse the exact same alt text across images.
- If two candidates differ only by punctuation or generic filler, treat them as duplicates.
- Distinguish images only with grounded differences such as color, angle, open/closed state, packaging, scale reference, lifestyle scene, or feature close-up.
- If uniqueness cannot be achieved without guessing, keep the wording conservative and flag the item for review.

## Shared-File Awareness

- Product media alt text is updated through `fileUpdate`.
- If a file may be shared across product references, warn about shared-file impact before execution.

## Final Gate

Before approval preview:

- Reject alt text above 512 characters.
- Rewrite alt text above 125 characters unless accessibility clearly requires more detail.
- Reject unsupported claims.
- Reject any `vision` candidate without concrete visual evidence.
- Keep product alt text inside the same approval bundle as other product-field changes when the user wants one-round execution.
