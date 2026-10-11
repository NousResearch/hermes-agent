/* ============================================================================
   scrollcraft.css: the taste floor
   ----------------------------------------------------------------------------
   Two layers, and the split matters:

     TOKENS   what you override per brand. Colour roles, the type ramp, the
              spacing scale, elevation, motion. A brand is ~12 values.
     DEVICES  what the engine drives. Do not restyle these to taste; they are
              the mechanism. Style your own markup instead.

   Nothing here draws a "component". There are no card, badge, or pill classes,
   because a stylesheet that ships those is how every page built on it ends up
   with the same shapes. You get a floor and a vocabulary; the composition is
   yours.
   ========================================================================== */

/* ---------------------------------------------------------------- tokens -- */
:root {
  /* --- colour roles. Override these six and the page is rebranded. -------- */
  --sc-canvas:      #08090b;   /* the page ground. Drift interpolates this.   */
  --sc-surface:     #101217;   /* anything raised off the ground             */
  --sc-ink:         #f4f2ef;   /* primary text                               */
  --sc-ink-soft:    #9a9ba1;   /* secondary text. Tinted, never flat gray.   */
  --sc-accent:      #d8ff3e;   /* ONE accent. It owns a region, not confetti. */
  --sc-accent-ink:  #08090b;   /* text that sits on the accent               */

  --sc-hairline:    color-mix(in oklab, var(--sc-ink) 12%, transparent);
  --sc-hairline-strong: color-mix(in oklab, var(--sc-ink) 22%, transparent);

  /* --- type. Two families max. Display carries voice, text carries prose. - */
  --sc-font-display: "Instrument Sans", "Geist", system-ui, sans-serif;
  --sc-font-text:    "Geist", system-ui, sans-serif;
  --sc-font-mono:    "Geist Mono", ui-monospace, monospace;

  /* Fluid ramp. Tracking tightens as size grows because a face set at 6rem
     with 0 tracking reads loose; this is optical correction, not decoration. */
  --sc-t-xs:   clamp(0.75rem, 0.72rem + 0.15vw, 0.82rem);
  --sc-t-sm:   clamp(0.875rem, 0.84rem + 0.18vw, 0.95rem);
  --sc-t-base: clamp(1rem, 0.96rem + 0.22vw, 1.125rem);
  --sc-t-lg:   clamp(1.2rem, 1.1rem + 0.5vw, 1.5rem);
  --sc-t-xl:   clamp(1.6rem, 1.35rem + 1.2vw, 2.25rem);
  --sc-t-2xl:  clamp(2.1rem, 1.6rem + 2.4vw, 3.4rem);
  --sc-t-3xl:  clamp(2.8rem, 1.9rem + 4.2vw, 5rem);
  --sc-t-4xl:  clamp(3.4rem, 1.9rem + 6.6vw, 7.5rem);

  --sc-track-tight:  -0.035em;
  --sc-track-snug:   -0.02em;
  --sc-track-normal: -0.005em;
  --sc-track-wide:    0.08em;

  --sc-leading-none: 0.94;
  --sc-leading-tight: 1.06;
  --sc-leading-body: 1.62;

  --sc-measure: 62ch;

  /* --- space. 4px base: the useful middle steps an 8-only scale misses. --- */
  --sc-1: 0.25rem;  --sc-2: 0.5rem;   --sc-3: 0.75rem;  --sc-4: 1rem;
  --sc-5: 1.5rem;   --sc-6: 2rem;     --sc-7: 3rem;     --sc-8: 4rem;
  --sc-9: 6rem;     --sc-10: 8rem;    --sc-11: 12rem;

  /* Section rhythm is fluid so a phone doesn't inherit desktop air. */
  --sc-section: clamp(4.5rem, 9vw, 11rem);
  --sc-gutter:  clamp(1.25rem, 5vw, 5.5rem);
  --sc-maxw:    82rem;

  /* --- radius. Pick one scale for the page and hold it. ------------------- */
  --sc-r-sm: 6px;  --sc-r-md: 12px;  --sc-r-lg: 20px;  --sc-r-pill: 999px;

  /* --- elevation. Every shadow has an offset AND a soft blur, and is tinted
         to the canvas hue. A zero-offset coloured halo is decoration, not
         depth, so there isn't one here. ---------------------------------- */
  --sc-shadow-color: 220 40% 2%;
  --sc-e1: 0 1px 2px hsl(var(--sc-shadow-color) / 0.28),
           0 2px 6px -1px hsl(var(--sc-shadow-color) / 0.20);
  --sc-e2: 0 2px 4px hsl(var(--sc-shadow-color) / 0.24),
           0 8px 18px -4px hsl(var(--sc-shadow-color) / 0.30);
  --sc-e3: 0 4px 8px hsl(var(--sc-shadow-color) / 0.22),
           0 18px 40px -8px hsl(var(--sc-shadow-color) / 0.38),
           0 48px 90px -24px hsl(var(--sc-shadow-color) / 0.30);

  /* Edge light: a 1px top highlight sells a raised surface better than any
     amount of blur, because real raised things catch light on their lip. */
  --sc-edge: inset 0 1px 0 color-mix(in oklab, var(--sc-ink) 10%, transparent);

  /* --- motion. Built-in CSS easings are too weak for UI. ------------------ */
  --sc-ease-out:    cubic-bezier(0.23, 1, 0.32, 1);
  --sc-ease-in-out: cubic-bezier(0.77, 0, 0.175, 1);
  --sc-ease-drawer: cubic-bezier(0.32, 0.72, 0, 1);
  --sc-d-fast: 160ms;  --sc-d-base: 240ms;  --sc-d-slow: 420ms;

  --sc-z-stage: 1; --sc-z-copy: 20; --sc-z-chrome: 60;
}

/* --------------------------------------------------------------- reset -- */
*, *::before, *::after { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; scroll-behavior: smooth; }
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } }

body {
  margin: 0;
  background: var(--sc-canvas);
  color: var(--sc-ink);
  font-family: var(--sc-font-text);
  font-size: var(--sc-t-base);
  line-height: var(--sc-leading-body);
  letter-spacing: var(--sc-track-normal);
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
  font-synthesis-weight: none;
  overflow-x: clip;
}
img, video, canvas, svg { display: block; max-width: 100%; }
button, input, select, textarea { font: inherit; color: inherit; }

/* ------------------------------------------------- the browser surfaces --
   The parts you did not draw still carry the design. Selection, caret, focus
   ring, scrollbar and tabular numerals all ship with defaults that belong to no
   design system. Theming them is the cheapest signal that a page was built
   rather than assembled, and it is the step that gets skipped most reliably. */
::selection { background: var(--sc-accent); color: var(--sc-accent-ink); }
:root { caret-color: var(--sc-accent); accent-color: var(--sc-accent); }
:focus-visible {
  outline: 2px solid var(--sc-accent);
  outline-offset: 3px;
  border-radius: var(--sc-r-sm);
}
:where(a) { color: inherit; text-decoration-thickness: 1px; text-underline-offset: 0.22em; }
@supports (scrollbar-color: auto) {
  html { scrollbar-color: color-mix(in oklab, var(--sc-ink) 26%, transparent) transparent; scrollbar-width: thin; }
}
:where(table, .sc-nums, [data-sc-count]) { font-variant-numeric: tabular-nums; }

/* ----------------------------------------------------------- typography -- */
.sc-display {
  font-family: var(--sc-font-display);
  font-weight: 500;
  line-height: var(--sc-leading-none);
  letter-spacing: var(--sc-track-tight);
  text-wrap: balance;
  margin: 0;
}
.sc-display--xl { font-size: var(--sc-t-4xl); }
.sc-display--lg { font-size: var(--sc-t-3xl); }
.sc-display--md { font-size: var(--sc-t-2xl); letter-spacing: var(--sc-track-snug); }
.sc-lede {
  font-size: var(--sc-t-lg);
  line-height: 1.44;
  letter-spacing: var(--sc-track-snug);
  color: var(--sc-ink);
  max-width: 46ch;
  text-wrap: pretty;
  margin: 0;
}
.sc-body { max-width: var(--sc-measure); color: var(--sc-ink-soft); text-wrap: pretty; margin: 0; }
.sc-label {
  font-family: var(--sc-font-mono);
  font-size: var(--sc-t-xs);
  letter-spacing: var(--sc-track-wide);
  text-transform: uppercase;
  color: var(--sc-ink-soft);
}

/* ---------------------------------------------------------------- layout -- */
.sc-wrap { width: 100%; max-width: var(--sc-maxw); margin-inline: auto; padding-inline: var(--sc-gutter); }
.sc-section { padding-block: var(--sc-section); }
/* More space above a heading than below it: the gap belongs to the boundary,
   not to the pair. */
.sc-stack > * + * { margin-top: var(--sc-4); }
.sc-stack > :is(h1,h2,h3) { margin-top: var(--sc-7); }
.sc-stack > :is(h1,h2,h3):first-child { margin-top: 0; }
.sc-rule { height: 1px; border: 0; background: var(--sc-hairline); margin: 0; }

/* =============================================================== DEVICES ==
   Driven by the engine. Restyle your own markup, not these.                */

.sc-act--pinned { position: relative; }
.sc-stage {
  position: sticky; top: 0;
  height: 100vh; height: 100svh;
  overflow: clip;
  z-index: var(--sc-z-stage);
}

/* scrub / sequence media */
.sc-stage :is(video[data-sc-scrub], canvas[data-sc-sequence], .sc-stage__poster) {
  position: absolute; inset: 0;
  width: 100%; height: 100%;
  object-fit: cover;
}
video[data-sc-scrub] { opacity: 0; transition: opacity 380ms var(--sc-ease-out); }
.sc-has-clip video[data-sc-scrub] { opacity: 1; }
/* The poster is a live frame-holder, not a placeholder: it stays up until a
   real video frame has painted, which is what stops the blank-stage flash. */
.sc-stage__poster { z-index: 0; transition: opacity 380ms var(--sc-ease-out); }
.sc-has-clip .sc-stage__poster { opacity: 0; }

/* A scrim only where text sits. A full-frame darkening layer flattens the
   image everywhere to fix contrast in one corner. */
.sc-scrim {
  position: absolute; inset: 0; pointer-events: none; z-index: 1;
  background: linear-gradient(
    var(--sc-scrim-angle, 180deg),
    color-mix(in oklab, var(--sc-canvas) var(--sc-scrim-a, 78%), transparent) 0%,
    color-mix(in oklab, var(--sc-canvas) 30%, transparent) 42%,
    transparent 68%);
}
/* Point the scrim at the copy. A scrim gradient that darkens the top while the
   copy sits at the bottom fixes contrast nowhere and flattens the image
   everywhere, which is the most common way text-over-video goes wrong. */
.sc-scrim--bottom { --sc-scrim-angle: 0deg; }
.sc-scrim--left   { --sc-scrim-angle: 90deg; }
.sc-scrim--right  { --sc-scrim-angle: 270deg; }

/* Corner scrims, for copy anchored to a corner. An edge gradient has to darken
   a whole band across the frame to cover one corner; a corner gradient puts the
   density where the text is and leaves the rest of the image alone. Pair with
   .sc-copy--lead / .sc-copy--trail. */
.sc-scrim--lead {
  background: linear-gradient(to top right,
    color-mix(in oklab, var(--sc-canvas) 94%, transparent) 0%,
    color-mix(in oklab, var(--sc-canvas) 72%, transparent) 30%,
    color-mix(in oklab, var(--sc-canvas) 30%, transparent) 52%,
    transparent 72%);
}
.sc-scrim--trail {
  background: linear-gradient(to top left,
    color-mix(in oklab, var(--sc-canvas) 94%, transparent) 0%,
    color-mix(in oklab, var(--sc-canvas) 72%, transparent) 30%,
    color-mix(in oklab, var(--sc-canvas) 30%, transparent) 52%,
    transparent 72%);
}
/* A band across the bottom, for copy that spans the full width of the frame.
   That is what both corner anchors become below 860px, and it is also the right
   shape whenever the copy block is wider than a corner. */
.sc-scrim--band {
  background: linear-gradient(to top,
    color-mix(in oklab, var(--sc-canvas) 94%, transparent) 0%,
    color-mix(in oklab, var(--sc-canvas) 74%, transparent) 22%,
    color-mix(in oklab, var(--sc-canvas) 30%, transparent) 42%,
    transparent 58%);
}
.sc-scrim--vignette {
  background: radial-gradient(120% 90% at 50% 45%, transparent 40%,
              color-mix(in oklab, var(--sc-canvas) 70%, transparent) 100%);
}

/* copy over a stage.
   max-width is in rem, never ch. A `ch` on this container resolves against the
   CONTAINER's font-size (body size), not the display size of the heading inside
   it, so `max-width: 20ch` here silently produces a ~180px column and wraps a
   hero headline to six lines. Set a ch measure on the text element itself. */
.sc-copy {
  position: absolute; z-index: var(--sc-z-copy);
  inset-inline: var(--sc-gutter);
  max-width: min(46rem, 76vw);
  will-change: opacity, transform;
}
.sc-copy--lead   { left: var(--sc-gutter); bottom: clamp(3rem, 12vh, 9rem); }
/* inset-inline FIRST. It is the shorthand for left+right, so declaring it after
   `left: 50%` resets left to auto and the block drifts off the left edge. */
.sc-copy--center { inset-inline: auto; left: 50%; top: 50%; translate: -50% -50%; text-align: center; }
.sc-copy--trail  { inset-inline: auto; right: var(--sc-gutter); bottom: clamp(3rem, 12vh, 9rem); text-align: right; }

/* cue defaults: the engine writes opacity/transform inline, these are the
   pre-paint state so nothing flashes before the first read() */
[data-sc-cue] { opacity: 0; will-change: opacity, transform; }
.sc-ready [data-sc-cue] { transition: none; }

/* kinetic split */
.sc-split { display: inline-block; overflow: hidden; vertical-align: top; }
/* Descenders live below the baseline; a line mask clipped to the line box eats
   the tails off g, y, p, j. The padding buys them room back. */
.sc-split--line { display: block; padding-bottom: 0.14em; margin-bottom: -0.14em; }
.sc-split__i { display: inline-block; will-change: transform, opacity; }
.sc-is-split { opacity: 1 !important; }

/* horizontal rail */
[data-sc-pan] { display: flex; will-change: transform; }

/* reveal + parallax */
[data-sc-reveal] { will-change: clip-path; }
[data-sc-parallax] { will-change: transform; }

/* flow reveal (fires once) */
[data-sc-in], [data-sc-stagger] > * {
  opacity: 0;
  transform: translate3d(0, 14px, 0);
  transition: opacity 620ms var(--sc-ease-out), transform 620ms var(--sc-ease-out);
}
[data-sc-in].sc-in, [data-sc-stagger] > .sc-in { opacity: 1; transform: none; }

/* pointer devices */
[data-sc-tilt] { transform-style: preserve-3d; will-change: transform; }
[data-sc-magnet] { will-change: transform; }
/* isolation always; position only if nothing more specific already set it.
   Written with :where() (zero specificity) on purpose: a plain `position:
   relative` here has the same weight as `.sc-stage` and, being later in the
   file, silently wins. A stage carrying both attributes then stops being
   sticky and the whole act scrolls away instead of pinning. */
[data-sc-spotlight] { isolation: isolate; }
:where([data-sc-spotlight]) { position: relative; }
[data-sc-spotlight]::after {
  content: ""; position: absolute; inset: 0; pointer-events: none; z-index: 2;
  background: radial-gradient(
    28rem 28rem at calc(var(--sc-mx, 0.5) * 100%) calc(var(--sc-my, 0.5) * 100%),
    color-mix(in oklab, var(--sc-accent) 16%, transparent), transparent 70%);
  opacity: 0; transition: opacity var(--sc-d-slow) var(--sc-ease-out);
}
[data-sc-spotlight]:hover::after { opacity: 1; }

/* =========================================================== WORLDFLIGHT ==
   One fixed stage for the whole page. The only thing in document flow is the
   spacer, and it is empty. If you find yourself adding a section here, you want
   act mode instead: the moment a real block scrolls past the fixed stage, the
   page has a seam again and the whole point is gone.                        */

[data-sc-mode="worldflight"] { position: relative; }

.sc-world {
  position: fixed; inset: 0;
  overflow: clip;
  z-index: var(--sc-z-stage);
  background: var(--sc-canvas);
}

/* Every leg is mounted for the life of the page and stacked in the same box.
   The engine owns opacity, visibility and z-index here; do not set them. */
.sc-world__seg {
  position: absolute; inset: 0;
  opacity: 0;
  will-change: opacity;
}
.sc-world__seg > :is(video, img, canvas) {
  position: absolute; inset: 0;
  width: 100%; height: 100%;
  object-fit: cover;
}
/* The poster is the frame-holder AND the first camera move: the engine pushes
   it in slowly until a real decoded frame is available to replace it. */
.sc-world__poster {
  z-index: 0;
  transform-origin: 50% 50%;
  transition: opacity 420ms var(--sc-ease-out);
}
.sc-world__seg > video { z-index: 1; }
.sc-world__seg.sc-has-clip .sc-world__poster { opacity: 0; }
/* The act stylesheet lights a clip through its ancestor act. A worldflight leg
   has no act, so match the class the engine also writes on the clip itself. */
video[data-sc-scrub].sc-has-clip { opacity: 1; }

/* The copy layer is fixed too, and inert by default: the engine hands
   pointer-events back to a block only while it is actually legible. */
.sc-world__copy {
  position: fixed; inset: 0;
  z-index: var(--sc-z-copy);
  pointer-events: none;
}
.sc-world__scrim { position: absolute; inset: 0; pointer-events: none; z-index: 0; }
[data-sc-copy] {
  opacity: 0;
  pointer-events: none;
  will-change: opacity, transform;
}
.sc-ready [data-sc-copy] { transition: none; }

/* The track. Empty on purpose. */
.sc-world__spacer { pointer-events: none; }

/* scroll progress */
[data-sc-progress] {
  position: fixed; inset: 0 0 auto 0; height: 2px; z-index: var(--sc-z-chrome);
  background: var(--sc-accent); transform-origin: 0 50%; transform: scaleX(0);
  will-change: transform;
}

/* ----------------------------------------------------------- atmosphere --
   Depth is not only shadow. Grain keeps a flat dark ground from banding, and
   it is the difference between "a dark page" and "a lit room". */
.sc-grain { position: fixed; inset: 0; pointer-events: none; z-index: 3; opacity: 0.045; }
.sc-grain::before {
  content: ""; position: absolute; inset: -50%;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='140' height='140'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3'/></filter><rect width='140' height='140' filter='url(%23n)'/></svg>");
}

/* -------------------------------------------------------- reduced motion -- */
@media (prefers-reduced-motion: reduce) {
  /* Fewer and gentler, not zero. Opacity still carries the reveal so the page
     is comprehensible; every position change is dropped. */
  [data-sc-in], [data-sc-stagger] > * { transform: none; transition-duration: 220ms; }
  [data-sc-parallax], [data-sc-pan], .sc-split__i { transform: none !important; }
  /* A wipe is a position change too. Show reveal content settled. */
  [data-sc-reveal] { clip-path: none !important; }
  /* A rail's transform IS its navigation, not decoration. Zeroing it the way
     parallax and kinetic type are zeroed parks the act on its first screenful
     and makes everything past the fold unreachable: a device degrading into
     missing content. Hand the travel back to the reader as an ordinary scroll
     region instead, so the same items are all still gettable without motion.
     A build that would rather re-lay the rail out as a grid can still do that;
     this is the floor, not the ceiling. */
  [data-sc-act="pan"] .sc-stage,
  [data-sc-act="pan"] [data-sc-stage] {
    overflow-x: auto; overflow-y: hidden;
    overscroll-behavior-x: contain;
    scroll-snap-type: x proximity;
  }
  [data-sc-pan] > * { scroll-snap-align: center; }
  [data-sc-spotlight]::after { display: none; }
  /* A worldflight keeps its whole story here. No clip is ever fetched, so the
     posters are the film, and they cross-dissolve through exactly the same
     seams at exactly the same scroll positions. The push-in and the copy drift
     are the only things that go. */
  .sc-world__poster, [data-sc-copy] { transform: none !important; }
}

/* --------------------------------------------------------------- mobile -- */
@media (max-width: 860px) {
  .sc-copy { inset-inline: var(--sc-gutter); max-width: none; }
  .sc-copy--trail { text-align: left; }
  .sc-stage { height: 100svh; }
  /* .sc-copy--trail re-anchors to the left and spans the full width here, so a
     bottom-RIGHT corner gradient darkens the one corner the copy just left and
     guarantees a contrast failure over any bright clip. Point the density at
     the copy: the same band .sc-scrim--band paints. */
  .sc-scrim--trail {
    background: linear-gradient(to top,
      color-mix(in oklab, var(--sc-canvas) 94%, transparent) 0%,
      color-mix(in oklab, var(--sc-canvas) 74%, transparent) 22%,
      color-mix(in oklab, var(--sc-canvas) 30%, transparent) 42%,
      transparent 58%);
  }
}
