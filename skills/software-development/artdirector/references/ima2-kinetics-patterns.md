# Reference Patterns: ima2-gen and Kinetics

## Evidence

Inspected 2026-07-13 at 1440px desktop and 390px mobile with headless Chromium, full-page screenshots, incremental scroll activation, DOM/computed-style extraction, overflow metrics, and console/page-error capture.

- ima2-gen: https://lidge-jun.github.io/ima2-gen/
- Kinetics: https://kinetics.colorion.co/
- Both returned HTTP 200, had no horizontal overflow at the inspected viewports, and emitted no console/page errors during capture.

## ima2-gen: Campaign Art Direction

### Visual system

- Near-black canvas, cool white text, chrome/foil accents.
- `Clash Display` for brand statements, `Satoshi` for body/UI, monospace for technical metadata, `Pretendard Variable` for Korean fallback.
- Measured hero scale: 128px/700 desktop, 44px/700 mobile; 16px body.
- Oversized product screenshots, rotated/overlapping frames, metallic 3D objects, ticker bands, and scroll reveal establish a launch-campaign voice.

### What works

- The hero communicates product category, workflow, and tone immediately.
- Product screenshots explain the feature rather than serving as decoration.
- Extreme display/body contrast produces a deliberate hierarchy instead of a generic middle-sized scale.
- Marketing art direction and the product's dark studio UI share one visual language.
- Most content sits directly on the canvas; cards appear when they convey a product surface rather than wrapping every paragraph.

### Risks and failures to prevent

- Decoration can become more memorable than the product.
- Mobile hero density is high: logo, language, GitHub, display type, foil treatment, body, installer, 3D object, and product preview compete in one viewport.
- Sticky navigation can overlap hero content.
- Reveal-dependent sections can appear as empty space when JavaScript, reduced motion, automation, or fast scrolling bypasses the expected timeline.
- Repeating “large title → screenshot → explanation” over a long page creates fatigue.

### Reusable principle

Use bold image scale and editorial sequencing, but require static/reduced-motion fallbacks, mobile overlap checks, and a single focal point per viewport.

## Kinetics: System Discipline

### Visual system

- Near-black canvas, warm off-white text, one orange accent family.
- `Archivo` for display, `Inter` for body, monospace for physics parameters and technical labels.
- Measured hero scale: 74px/900 desktop; 16px body.
- Repeated demo-card anatomy: parameter → live interaction → title → description → code action.
- Thin borders and restrained surfaces carry structure; motion states carry personality.

### What works

- Brand promise and product mechanism are identical: “motion that has weight” is demonstrated with visible spring parameters and live interactions.
- Accent use is semantic: primary action, selected control, key number, and motion progress.
- A stable card anatomy lets many examples vary without visual drift.
- Desktop multi-column layouts collapse to a coherent mobile list without changing component identity.
- Motion is product evidence, not ambient decoration.

### Risks and failures to prevent

- A 99-item catalog becomes extremely long on mobile; filtering and representative subsets are needed.
- Uniform cards create fatigue and flatten importance after many repetitions.
- Hover/cursor effects need touch alternatives or desktop-only disclosure.
- Small parameter labels and secondary copy risk low contrast.
- Developer-tool styling should not be transplanted unchanged into consumer products.

### Reusable principle

Use one accent, stable component anatomy, and state-driven feedback. Pair the catalog with hierarchy and device-appropriate interaction alternatives.

## Combined Direction for Consumer and Fashion Products

The preferred synthesis is:

1. **Kinetics for system discipline** — constrained palette, semantic accent, stable component templates, explicit state behavior, and responsive identity.
2. **ima2-gen for editorial scale** — one dominant image, strong display/body contrast, alternating image-copy composition, and product evidence above decorative cards.
3. **Product-specific brand content** — fashion photography, Korean copy, and commerce hierarchy must replace developer-tool motifs.

### Apply

- Let the person or garment occupy 55–70% of the primary stage when styling is the value.
- Use structural look numbering such as `01 / 03` without turning it into badge noise.
- Sequence image → look name → body-fit rationale → adjustment advice.
- Keep one component anatomy across looks; vary content and imagery, not chrome.
- Use one accent only for selection, primary CTA, score, or progress.
- Use restrained damped-spring motion for look changes, selected indicators, saved confirmation, and sheet transitions.
- Ensure unsaved and confirmed CTA states are both active-looking and semantically distinct.

### Do not copy

- Chrome 3D objects, rainbow foil, neon glow, developer-terminal aesthetics, magnetic buttons, cursor trails, particle bursts, rotating frames, or perpetual ambient animation.
- Long reveal-only sections without reduced-motion/static fallbacks.
- Hover-only interactions on touch surfaces.
- A long undifferentiated card catalog.

## Art Director Enforcement Additions

Before approving UI influenced by these references, verify:

- [ ] The reference contribution is named: editorial scale, component discipline, or state behavior.
- [ ] No stylistic motif was copied without matching product semantics.
- [ ] Exactly one dominant focal point exists per mobile viewport.
- [ ] Sticky chrome does not overlap hero or interactive content.
- [ ] Reveal content is visible under reduced motion and when animation fails.
- [ ] Motion communicates state and remains interruptible; it is not ambient garnish.
- [ ] Hover interactions have touch equivalents or clear platform restrictions.
- [ ] Repeated cards preserve anatomy but representative content has hierarchy.
- [ ] A single accent family has documented semantic roles.
- [ ] Product imagery remains the evidence and does not become a small framed decoration.
