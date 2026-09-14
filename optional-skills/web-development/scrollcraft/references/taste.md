# The taste floor

Read this before writing markup, not after. Build without announcing the
checklist.

Everything here is a check on the **rendered result**, not on intention. "I used
a spacing scale" is not evidence; a computed value is.

---

## Spacing

Rhythm comes from the contrast between tight and generous, never from one value
repeated until everything weighs the same. If you can't point at which intervals
are the tight ones and which are the breaks, the page has no rhythm.

- Use the 4px-base scale (`--sc-1` … `--sc-11`). A 4-base gives the useful
  middle steps an 8-only scale misses.
- **More space above a heading than below it.** The gap belongs to the boundary
  between sections, not to the heading-and-body pair. Getting this backwards is
  the single most common spacing error, and it makes a page read as a list.
- Section padding is fluid (`--sc-section`). A phone should not inherit desktop
  air; 8rem of padding on a 375px screen is a scroll tax.
- Group by proximity before reaching for a container. If you added a border to
  show two things are related, the spacing was wrong first.
- Gutters scale with viewport (`--sc-gutter`). Full-bleed media goes edge to
  edge; text never does.

**Optical, not mathematical.** Equal computed padding around a shape with
uneven visual weight looks wrong. Correct against the render, not the number.

---

## Typography

- **Two families maximum.** Display carries voice, text carries prose. A third
  is a costume.
- **Tracking tightens as size grows.** A face set at 6rem with default tracking
  reads loose and amateur. The ramp handles this: `--sc-track-tight` on display,
  `--sc-track-normal` on body. This is optical correction, not decoration.
- **Body measure 45 to 75ch.** `--sc-measure` is 62ch. A full-width paragraph on
  a 1600px monitor is unreadable regardless of font size.
- **Line height inverse to measure.** Wider lines need more leading. Display at
  0.94 to 1.06, body at 1.6.
- **Light text on dark needs compensation on three axes**: slightly more line
  height, a touch more tracking, one step more weight. Dark-mode type set with
  light-mode metrics looks thin and blurry, and this is why.
- `text-wrap: balance` on headings, `pretty` on body. Free, and it removes the
  orphan word that makes a headline look accidental.
- Display max ~6rem outside a genuine hero moment. Bigger is not more confident.
- **Step the hero down one rung below ~700px.** `--sc-t-4xl` floors at 3.4rem,
  which is a *desktop* floor: at 390px it wraps a normal hero headline to six
  lines. `--sc-t-2xl` on the hero inside a phone media query fixes it. The
  portrait crop of the image is covered in assets.md; this is the portrait crop
  of the type, and it is missed more often.

**Font choice.** Inter is discouraged as a default: it is the most-used face in
AI-generated pages and it reads as a non-decision. Reach first for Geist,
Archivo, Outfit, Satoshi, Cabinet Grotesk, or the brand's own face. Inter is
correct when the brand asks for neutral, or when accessibility is the brief.

**Serif is not a synonym for premium.** "It feels editorial" is not a reason.
Use one only when the brand names it, or when the work is genuinely editorial,
luxury, or heritage and you can say why *this* serif fits *this* brand.

**Emphasis inside a headline** uses italic or bold of the same family. Dropping
a serif word into a sans headline for visual interest is amateur.

---

## Colour

- **Six roles, one accent.** Canvas, surface, ink, ink-soft, accent, accent-ink.
  The accent owns a region or a role; scattered tiny accents are confetti.
- **Lock the accent for the whole page.** A warm-grey site does not grow a blue
  CTA in section seven. **The one exception is a page that hard-cuts between
  light and dark grounds**, which physically cannot clear 4.5:1 on both with a
  single stop. That page carries a two-stop accent: one hue, two lightnesses,
  keyed to the ground family, redefined per section alongside the ink. Still one
  accent per ground, and still one hue for the page. Two different hues is not
  what this licenses.
- **Secondary text is tinted, never flat gray.** Derive it from the foreground
  or surface hue. `#888` on a warm dark ground looks dirty.
- **No pure black.** `#000` has no air in it. Off-black at minimum.
- Contrast, measured on the render: body ≥4.5:1, large text ≥3:1, controls and
  focus indicators ≥3:1.
- Drift keeps the whole page in one theme family. See devices.md §10.

**Redefining `--sc-ink` on a subtree does not re-ink the text under it.**
`color` is inherited as a *computed value*, so text whose `color` already
resolved on `<body>` keeps the body's ink no matter what the section redefines
the token to. Every page that inverts a ground mid-page hits this, and it fails
silently: an inverted section renders bone type on concrete at 1.15:1 while the
harness correctly classifies the line as light-on-dark and grades it in the
wrong direction. The fix is one declaration on the same subtree:

```css
.section--light { --sc-ink: #14110C; --sc-ink-soft: #4A443A; color: var(--sc-ink); }
```

Restate `color` wherever you restate the token. The same applies to any other
inherited property you drive from a token on a subtree.

**The premium-consumer palette trap.** Warm cream background, brass or clay
accent, espresso near-black text is the default reach for every artisan, food,
wellness and craft brief, and it makes every such brand look identical. Do not
default to it. Rotate: cold silver and chrome; deep forest with bone and amber;
true off-black with warm tan; cobalt against a single neutral; olive with brick.
Use cream-and-brass only when the brand names those colours.

**The AI-purple trap.** Violet-to-blue gradients, neon glow, glowing buttons.
Not unless the brand asks.

---

## Text over media

"No full-frame overlay" is the rule. Here is what to do instead, because the
rule on its own sends people to a slightly weaker full-frame overlay.

There are three shapes, and which one is right depends only on where the copy is:

1. **A corner** of density, sized to the copy block. `.sc-scrim--lead` /
   `.sc-scrim--trail`. Right when the copy is anchored to a corner on a wide
   screen. An edge gradient has to darken a whole band across the frame to cover
   one corner; a corner gradient puts the density where the text is and leaves
   the photograph alone.
2. **A band**, `.sc-scrim--band`, transparent above roughly 58%. Right whenever
   the copy spans the full width of the frame, which is what *both* corner
   anchors become below 860px. The engine already switches `.sc-scrim--trail` to
   a band there for exactly that reason.
3. **A column** of density under a text column, on an act where the copy holds
   one side of a full-bleed image. Leaves the other half of the frame untouched.

**`width` and `height` attributes are presentational hints, and they come in
pairs.** The reference template ships every `<img>` with both, correctly, because
they reserve the aspect ratio and stop the page reflowing as media arrives. The
trap is that overriding only one of them in CSS leaves the other resolving to the
attribute's raw pixel value, so `width: 100%` on a 1920x1080 image inside a
narrow column renders it 1080px tall and pushes everything under it off the fold.
It looks like a layout bug three elements away from its cause. **Override both or
neither**, usually `width: 100%; height: auto`, or an explicit height plus
`object-fit: cover` when the frame's shape is the design.

And the positive case behind all three: when a photographic ground sits behind a
text column, **mask the image away from the text** rather than laying anything
over it. A `mask-image` or a clip that ends where the column begins gives the
type a clean ground and gives the photograph its full contrast back, and it is
better than any scrim.

**A scrim must not be a child of the text it protects.** The verification pass
hides the copy element and everything inside it to photograph the frame
underneath, so a `::before` on the copy block is hidden too and the scrim is
never measured. Put it in a sibling element. See verify.md.

Then measure it. A scrim tuned by eye is routinely 9:1 where 4.5:1 was needed,
which is a photograph thrown away for nothing, or 2.8:1 on the one frame the
clip brightens under the copy. Both are invisible until the harness reports the
number.

---

## Depth

Depth is the axis that separates a premium page from a styled document, and it
is not one property. Five tools, used together:

1. **Shadow with offset and blur.** Real raised things cast light downward.
   A zero-offset coloured halo is decoration, not depth. Tint the shadow to the
   canvas hue; pure black shadows on a coloured ground look like dirt.
2. **Edge light.** A 1px top highlight (`--sc-edge`) sells a raised surface
   better than any amount of blur, because real lips catch light.
3. **Scale and blur as distance.** Things further away are smaller, softer, and
   lower contrast. Parallax without those reads as sliding, not depth.
4. **Overlap.** One element crossing another's boundary establishes more depth
   than any shadow. Free, and underused.
5. **Grain.** A flat dark ground bands on real displays. `.sc-grain` at 4-5%
   opacity is the difference between "a dark page" and "a lit room".

Three elevation steps (`--sc-e1/2/3`) and no more. If everything is elevated,
nothing is.

---

## Cards

Cards are the lazy container. Before using one, ask what it is doing that
proximity, a hairline, or space could not.

- **Never a grid of identical icon + heading + text cards as the page
  structure.** It is the most recognisable AI-page tell there is.
- **Never nest cards.**
- **Never three equal columns of feature cards.** Use an asymmetric grid, a
  two-column zigzag (max two in a row), a rail, or plain type on space.
- If a multi-cell grid has an empty trailing cell, the grid was planned wrong.
  Reshape it; do not paste a blank tile.
- Pick one corner-radius scale and hold it across the page. Pill buttons on a
  square-card page is broken, not eclectic.

---

## Motion

The scroll devices are the page's motion. Everything else is small and fast.

- `transform` and `opacity` only for anything continuous. `clip-path` is the
  sanctioned third for wipes. Never animate width, height, margin, padding, top
  or left, and never `transition: all`.
- **Never `ease-in` on UI.** It delays the moment the eye is already on.
  `ease-out` at 200ms feels faster than `ease-in` at 200ms.
- Built-in CSS easings are too weak. Use `--sc-ease-out`
  (`cubic-bezier(0.23, 1, 0.32, 1)`).
- **UI transitions under 300ms.** Hover 120-180ms, buttons 100-160ms. Scroll
  devices are exempt: they are paced by the hand, not by a duration.
- **Never `scale(0)`.** Enter from `scale(0.95)` + `opacity: 0`. Nothing in the
  real world appears from nothing.
- Press feedback on anything pressable: `scale(0.97)` or `translateY(1px)`.
- Stagger group entrances 30 to 80ms. Longer feels slow.
- Gate hover motion to `(hover: hover) and (pointer: fine)`; touch fires false
  hovers on tap.
- Reduced motion means **fewer and gentler, not zero**. Keep the opacity that
  carries comprehension, drop every position change.

---

## States and content

- Every interactive element gets hover, focus-visible, active and disabled.
  A page with only the resting state is half-built.
- **Focus-visible must be visible.** Themed to the accent, with offset.
- **Button text fits on one line at desktop.** A wrapped CTA is broken. Primary
  CTA labels are one to three words.
- **One label per intent.** "Get in touch" in the nav and "Let's talk" in the
  footer are the same button with two names. Pick one and use it everywhere.
- **Check button contrast.** White text on a light button, or a ghost button on
  a photo with no scrim, fails.
- Real copy, not lorem. Real names, not "John Doe". Real numbers or no numbers.
- **No invented statistics.** Fake precision (`4.1×`, `92%`, `48k`) is a legal
  and credibility liability, not a design element.

---

## Browser surfaces

The parts you did not draw still carry the design, and this is the cheapest
signal that a page was built rather than assembled. It is also the step that
gets skipped most reliably. `scrollcraft.css` themes all of these; if you fork
it, keep them:

selection colour, caret colour, focus ring, scrollbar, underline offset and
thickness, tabular numerals in anything that counts or tabulates.

---

## The refuse list

Category defaults, not bans on principle. The brief's own words can earn any of
them; reaching for one when the axis is free means you were not deciding.

**Structure**
- Identical cards as page structure. Nested cards. Three equal feature columns.
- The hero-metric template: big number, small label, supporting stats, accent.
- More than two consecutive image-left / text-right zigzag sections.
- The same layout family twice on one page.
- A split header: giant headline left, small explainer paragraph floating right.

**Labels**
- An eyebrow above every section heading. At most one per three sections.
- Section numbers (`01 / 06`, `002 · Capabilities`) unless the sequence itself
  is information the reader needs.
- Scroll cues: "scroll", "↓ scroll", "scroll to explore", animated mouse icons.
  They are looking at the hero. They know.
- Decoration text strips (`BRAND. MOTION. SPATIAL.`) across the hero bottom.
- Locale, time and weather strips unless the brand is genuinely about a place.
- Pills and tags overlaid on photos. Version stamps on a marketing page.

**Surface**
- Gradient text. Neon and outer glows. Hard offset zero-blur shadows outside a
  world that is actually neobrutalist.
- Glass and blur as decoration rather than as a specific effect.
- Coloured `border-left` above 1px on cards, callouts or list items.
- Monospace as a costume for "technical" rather than for code, data, or labels.
- Emoji standing in for an icon system. Use a real icon library.
- Custom cursors.

**Content**
- Em dash anywhere visible. Period, comma, colon, or parentheses.
- Div-built fake screenshots, fake dashboards, fake terminals.
- Text baked into a generated image. Real markup, always.
- Filler verbs: elevate, seamless, unleash, next-gen, revolutionize, supercharge.
- A hero that overflows the viewport. Headline max two lines, subtext max 20
  words, CTA visible without scrolling.
- More than four text elements in the hero. Trust logos, pricing teasers and
  micro-taglines move to their own section below it.

---

## The squint test

Blur the page until detail is gone. You should still be able to name the
primary element, the secondary element, and the major groups, in that order.

If everything greys into one even field, the problem is hierarchy, and no amount
of shadow, gradient or motion will fix it.
