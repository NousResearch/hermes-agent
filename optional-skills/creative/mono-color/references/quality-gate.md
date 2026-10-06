# Quality Gate: Originality Firewall, Hard Avoids, Inspection

When to load: before compiling the final prompt (avoids and firewall feed paragraph 5) and after each generation (inspection checklist and retry rules).

## Originality Firewall

A reference is evidence for a visual grammar, never a layout to trace. Before generation, change at least four of these from any supplied reference:

- subject and crop; layout family; headline wording; headline location; image shape or count; grid structure; type pairing; metadata treatment; ratio; disruption device.

Never reproduce a reference's exact object arrangement, line breaks, labels, dates, logos, border system, or distinctive slogan. Never include fake signatures or publication marks. If the user's source image contains protected or branded material, transform only the user's provided material and avoid presenting the result as an official artifact.

## Hard Avoids

Always exclude:

- more than two printing inks, unassigned accent colors, gradients, rainbow accents, neon, or full-color photography;
- clean vector-flat digital poster aesthetics;
- beige lifestyle minimalism or a monochrome color wash;
- glossy mockups, 3D depth, cinematic lighting, lens blur, hard shadows;
- centered template symmetry, card grids, UI panels, stickers, decorative blobs;
- scrapbook collage, uncontrolled overlap, grunge overload, torn-paper styling;
- automatic vintage styling — yellowed paper, sepia aging, distressed borders, nostalgic props, retro type — merely because the image uses halftone or limited inks;
- long paragraphs, marketing copy, CTA buttons, logos, URLs, QR codes;
- exact imitation of a supplied poster or a recognizable artist signature.

## Generation and Inspection

1. Generate the image from the compiled prompt (Hermes `image_generate`).
2. Inspect at full size and thumbnail size.
3. Regenerate once when any of these fail:
   - a one-ink composition shows a second ink, or a two-ink composition shows a third printing ink;
   - a two-ink composition lacks clear plate roles, or the accent covers more than 30% without a subject-driven reason;
   - the page reads as digitally color-graded rather than physically printed;
   - empty paper falls outside 25–55%;
   - the subject is unrecognizable;
   - typography lacks a clear 5x or greater scale jump;
   - long text is garbled or invented branding appears;
   - the composition closely follows a supplied reference;
   - a relaxed result has no immediately identifiable focal event or distributes equal emphasis across the whole page;
   - a theme-only person becomes a complete stock-photo figure, or the page falls into a safe headline-left/photo-right split;
   - the release zone is filled with decorative microcopy, gestures, or secondary focal points.
4. If exact text renders incorrectly after one retry, generate a text-light base image and state that typography should be overlaid in a layout tool. Do not pretend distorted text is correct.

## Final Quality Checklist

- One intentionally selected white/gray/pale-beige substrate; no more than two printing inks.
- Contemporary editorial by default; vintage/aged styling only when requested.
- Two-ink work: each plate has a clear role; the accent remains controlled.
- 25–55% of the page visibly empty.
- Image reproduced through dots or mechanical print texture, not a color filter.
- One object occupies 45–80% of the page (except a justified information-heavy layout).
- Headline visibly crosses, covers, splits around, or locks tightly to the dominant object.
- Exposed paper forms a visible shape inside the image (highlights, gaps, fade-outs, knockouts).
- Exactly one manual gesture family; no mixed decorative doodle styles.
- Type hierarchy: 5–12x scale jump, no more than three type voices.
- Exactly one immediately identifiable focal event and one visibly quieter release zone.
- Relaxed work concentrates energy in the focal event rather than reducing it everywhere.
- With no source image, the figure feels observed in an ordinary in-between moment, not posed as an advertisement.
- Page-filling type is the selected focal event while the remaining devices retreat.
- Language is terse, specific, non-commercial; the user's supplied subject and text are preserved.
- At least four structural features differ from every supplied reference.
- An image was generated unless prompt-only was requested.
