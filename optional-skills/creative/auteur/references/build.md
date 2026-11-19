# build.md — the standard register

Not every page is a film, and forcing cinema onto a docs site is its own kind of slop. The build register produces a conventional surface executed at award level: committed color, real typography, one signature, disciplined motion. Same taste core, calmer camera.

## Process

1. **Intake (one message):** product · audience · register of the surface (marketing page / product UI / content site) · brand constraints · stack. Autonomous → write assumptions down.
2. **Recon (bounded, ~5 min):** load `references/recon.md`. `node scripts/refscout.mjs --from awwwards --limit 6` for live references (real stack, page shape, fonts, palette, screenshots — note the ONE mechanic taken from each), and `node scripts/moodboard.mjs "<feeling>" "<treatment>"` when the art direction is still open. This register usually leans harder on the moodboard than on the mechanics. No playwright / no network → skip; taste.md's reflex table carries you. Never cite references you didn't see, and never quote a fingerprint the tool marked NO CAPTURE.
3. **Commit-sheet** (SKILL.md) — all six fields. In this register "Peak" means the **signature element**: the one thing a visitor would describe to a friend. A signature is load-bearing, not decoration: an interactive hero object, a distinctive navigation behavior, an oversized typographic system, a chart that responds to the reader. Pick one, execute it fully.
4. **Mockup gate:** one static throwaway hero screen (`design/mockup-hero.html`) with real copy, the commit-sheet palette and type — screenshot at 1440/390, run `node scripts/slopscan.mjs design/` on it (free, and this is the cheapest place to catch a banned gradient or a contrast failure), look, get a yes (user) or self-check against the commit-sheet (autonomous). Approved CSS custom properties become the project tokens verbatim; "approved with carried notes" is a legal verdict as long as the notes are written down. Minutes now, or a rebuild later.
5. **Skeleton before skin:** semantic HTML for the whole page first — headings hierarchy, landmarks, real copy (write it; lorem hides layout truth). The page must read as a document with CSS off.
6. **Tokens:** define OKLCH custom properties (bg, surface, ink, muted, accent + the commitment-tier colors), the type scale (clamp()-based), the spacing scale — before any component. Load `taste.md` for color/type decisions if not already loaded.
7. **Build top-down**, mobile-first. Each section: layout → type → color → then motion *last* (load `motion.md` before the first animation; respect the page motion budget from the commit-sheet).
8. **States are the product:** hover (gated `@media (hover:hover)`), focus-visible (always, and it must look designed, not default-blue-unless-brand), active, disabled, loading, empty, error. A beautiful happy path with default focus rings is an unfinished page.
9. **Verify** (verify.md): slopscan → shoot → rubric. Same gates as cinema, minus CINEMA-QA.
10. **Lock the style:** fill `templates/DESIGN.md` → `design/DESIGN.md` from the shipped code, so every later edit (the `edit` route) stays in the system instead of drifting back to the mode.

## Modern platform defaults (use, don't ask)

- Container queries for anything that lives in a variable-width slot; viewport queries for page chrome.
- `text-wrap: balance` on headings, `pretty` on prose. `@property` for animatable custom properties (gradient angles, numeric counters).
- Popover API + `<dialog>` for menus/modals — free top-layer, light-dismiss, focus management; a positioned div in an `overflow:hidden` parent is a clipped dropdown waiting to happen.
- View Transitions (same-doc) for SPA state changes; `linear()` easing for spring feels without JS.
- `scroll-margin-top` on anchor targets under sticky headers. `:focus-visible` over `:focus`. `color-scheme` declared.
- Progressive enhancement is the architecture: CSS does the work until JS demonstrably wins; every JS enhancement wraps in a capability check; the un-enhanced page is complete, not broken.

## Craft details that separate good from generated

- Vertical rhythm: section paddings vary with content weight (tight where dense, airy around the signature). No uniform `padding-block: 6rem` down the whole page.
- Max ONE full-width colored band per viewport-height of scroll, or the page becomes a flag.
- Icons: one family, one stroke width, sized to the type scale (1cap or 1.2em), never as filler decoration next to every heading.
- Images get `aspect-ratio` reserved space (CLS), meaningful `alt`, `loading="lazy"` below the fold ONLY (hero is eager + `fetchpriority="high"`).
- Forms: labels always visible (placeholders are not labels), errors inline next to the field with recovery text, submit shows progress state.
- Tables for tabular data — styled, sticky-headed, right-aligned numerals with `font-variant-numeric: tabular-nums` — not card-ified into unscannability.
- Footer is a real place (sitemap, contact, legal), not three centered links.

## When to escalate to the direct register

If during build the commit-sheet's signature keeps growing — the client wants "more wow", the hero wants scroll choreography, assets want to be generated — stop patching. Say the surface has outgrown the register, and restart phase 0 in `direct.md` with the storyboard. A half-cinema page (one heavy scroll-jacked hero bolted onto a static page) is worse than either register done purely.
