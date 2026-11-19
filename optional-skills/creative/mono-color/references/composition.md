# Composition: Decision Flow, Layout Families, Grammar, Rhythm

When to load: while choosing the manifest's `layout`, `visual_tension`, `focal_event`, `release_zone`, and `unresolved_edge` fields, or writing the composition paragraph of the prompt. Geometry IDs live in `design-system/compositions.json` and `design-system/rhythm.json`; the catalog wins over this prose.

## Space and Grid

- Flat, front-facing paper canvas — no mockup, frame, desk, or cast shadow.
- Default `3:4` vertical poster; respect a user-specified ratio.
- Keep 25–55% of the canvas as visibly empty paper; generous outer margins of 5–9% of page width.
- Align most elements to one invisible left edge or a simple 2–3 column editorial grid.
- Create one deliberate disruption: a floating word, off-center image, oversized title, circular mark, or tiny annotation.
- Never center every element; never distribute objects evenly like a template. Negative space is active pacing, not leftover room.

## Composition Decision Flow

Walk top to bottom; use the first matching rule unless the user explicitly requests a layout:

1. Event, method, schedule, or factual announcement? → **ruled information poster**.
2. Botanical, collected, or taxonomic subject? → **archival plate** (consider botanical green).
3. One ordinary object explicitly requested as a repeated rhythm? → **object field**.
4. Concept explicitly depends on two images/colors/type layers physically crossing? → **overprint collage** with overprint duotone.
5. One supplied portrait or scene photograph?
   - Faithful reproduction → **image field**.
   - Abstract symbol extraction → **editorial cover** by default; **overprint collage** only when two extracted layers must physically cross.
6. 1–3 supplied isolated objects intended for labels or comparison? → **specimen annotation**.
7. The user's phrase itself is the main visual subject? → **type-led declaration**.
8. Reflective, dated, or essay-like content with a primary photograph and readable text? → **editorial journal**.
9. Otherwise → **editorial cover**.

## Layout Families

- **Image field:** large screened image crossing at least one page edge; headline overlaps or locks tightly to it; compact footer.
- **Specimen annotation:** 1–3 isolated cutouts with numbered labels, one oversized phrase, asymmetric empty space.
- **Type-led declaration:** headline controls the page; a smaller screened image interrupts or grounds it.
- **Ruled information poster:** one dominant screened object/scene crossed by a headline; thin one-ink rules form one metadata band; the date stays subordinate.
- **Archival plate:** title, one rectangular image plate, disciplined multi-column caption block.
- **Editorial cover:** title near one edge, one dominant image zone, sparse issue-like microcopy, no fake masthead brand.
- **Object field:** one recognizable object repeated at varied scale/crop/angle to form a printed rhythm; one open zone for title and facts.
- **Overprint collage:** two ink plates carry separate object, image, geometric, or typographic layers and cross in selected zones — deliberate overlap, not everywhere.
- **Editorial journal:** one primary screened photograph, a strong title or date, 2–3 disciplined text columns with real reading size and contrast.

## Composition Grammar (Four Moves)

Build every page from these, in the spirit of object-and-type construction rather than generic retro mood:

1. **One object dominates.** One person, animal, ordinary object, or repeated specimen anchors the page at 45–80% of its area, cropped decisively at one or more edges when scale creates tension. No scattering of small atmospheric props. (A ruled information poster may reduce the image zone to 32–55% only when real supplied information needs the space. Dense overlap is allowed only in overprint collage and must still read as two printing plates.)
2. **Type collides with the object.** One headline crosses, covers, splits around, or aligns tightly against the dominant image, staying readable. Never park every line in a detached safe zone above the image.
3. **Paper cuts through the image.** Clipped highlights, irregular cutout gaps, halftone fade-outs, or plate knockouts make exposed paper a visible shape inside the composition, not only an outer margin.
4. **One manual gesture interrupts the system.** One circled fact, hand-drawn line, registration mark, tiny symbol, rotated label, or ruled data strip — one gesture family only; multiple doodle styles turn the page into scrapbook decoration.

Choose one dominant object and one dominant typographic event before adding secondary information. If either is missing, simplify rather than filling the page with mood-setting decoration.

## Visual Tension and Uneven Energy

Read `design-system/rhythm.json` whenever the user asks for relaxed, loose, effortless, casual, quiet, breezy, or understated work, or when the default intent maps to `relaxed`. Relaxation is a compositional decision, not a soft-focus mood — **uneven energy, not low energy**.

For `relaxed` work choose exactly one strong focal event: oversized type, an extreme crop, one giant object or detail, a concentrated overprint collision, or one abnormal scale relationship. Let it feel decisive; then release the rest of the page with open paper, pale screening, sparse support type, or one quiet alignment. Do not make every element tasteful, small, or equally calm.

- Keep 25–55% visibly empty paper, sized from the focal event rather than a fixed relaxed quota.
- Display type may become large when it is the focal event; when the image or object is the focal event, type supports rather than competes.
- One dominant collision or scale event; secondary elements may be energetic only when they extend that same event.
- Avoid the safe split of headline on one side and a complete photograph on the other — the focal event must cross, crop, interrupt, or materially reorganize the page.
- An unresolved edge (image fade, inferable cropped word, broken alignment, open contour) is optional; use one only when it strengthens the focal event, never as a decorative compliance mark.
- At thumbnail size, the focal event must be immediately identifiable and the release zone visibly quieter.
