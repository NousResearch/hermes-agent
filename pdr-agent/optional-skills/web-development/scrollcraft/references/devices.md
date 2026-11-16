# The device kit

Nine ways for scroll to change the page. Each one is a different answer to "what
does the visitor's hand actually do here."

Pick per beat, never per page. The variety law from SKILL.md Step 2 applies:
four or more families, never the same one twice in a row.

Every act publishes `--sc-p` (0 to 1) on its own element, so anything you want
to drive that the kit does not cover, you can drive from CSS with `calc()`
against that variable. Reach for that before asking for a new device.

---

## 1. `scrub`: the wheel is a scrubber

The anchor device. A pre-rendered camera move plays under the reader's hand,
one frame per notch. This is the thing people screenshot and send to each other,
so spend it on the open.

```html
<section data-sc-act="scrub" data-sc-span="2.6" data-sc-dwell="0.35"
         data-sc-drift="#0A0806">
  <div data-sc-stage>
    <img class="sc-stage__poster" src="assets/01-hero.webp" alt="">
    <video data-sc-scrub data-sc-src="assets/01.mp4"
           data-sc-src-mobile="assets/01-m.mp4" playsinline muted></video>
    <div class="sc-scrim"></div>

    <div class="sc-copy sc-copy--lead" data-sc-cue="0.08 0.62 0.34">
      <h1 class="sc-display sc-display--xl" data-sc-kinetic="lines">
        Your morning shouldn't need two drinks.
      </h1>
    </div>
  </div>
</section>
```

- `data-sc-span` is the act's scroll length in viewport-heights. 2.2 to 3.0 for
  a hero. Below 1.8 the clip flies past; above 3.5 the reader starts wondering
  whether the page is broken.
- `data-sc-dwell` (0 to 0.6) remaps time so the camera settles mid-act, exactly
  where the copy peaks, and moves quicker at the edges. It is the difference
  between a clip that plays and a shot that lands. Keep it at or below 0.6.
- `data-sc-src` (not `src`) is deliberate: the engine fetches the clip as a Blob
  so it seeks without needing HTTP range support, and skips the fetch entirely
  under reduced motion.
- The poster is a live frame-holder. It stays up until a real video frame has
  painted, because iOS keeps a seeked-but-never-played muted video blank and
  hiding the poster on metadata alone flashes an empty stage.

**At most two scrub acts per page.** The third one is no longer a surprise, and
it is the heaviest thing on the page.

### Clip time is not cue time

The single most damaging bug this device has, and it is invisible in every
screenshot taken one at a time.

A pinned stage is on screen for **one viewport before** its pinned travel begins,
sliding up into view, and **one viewport after** it ends, sliding off the top.
The act's progress `p` is 0 through the whole entry and 1 through the whole exit.
So a clip driven by `p` sits frozen on its first frame while it slides in, and
frozen on its last frame while it slides out. The reader has been scrubbing a
film with their hand, the film stops, and then the whole page slides a still
photograph past them. It reads as the site breaking, and it is the fastest way to
make an expensive page feel cheap.

The engine therefore maps the clip across the stage's **entire visible life**, not
across its pinned travel, and this is the **default**. Both ends are clamped to
scroll that actually exists, so a hero at the top of the document still starts on
frame one and an act near the bottom still reaches its last frame. Cues keep
using `p`, because cues belong to the pin.

**Pair it with `data-sc-dwell`.** Dwell moves quickly at the edges and settles in
the middle, which is exactly the shape this mapping wants: the fast motion lands
on the two slides, and the settle lands inside the pin where the copy is. The two
were built for each other.

`data-sc-clip-map="travel"` restores the old pinned-travel mapping. There is
almost no reason to reach for it, and reaching for it reintroduces the freeze.

The harness checks this now (see [verify.md](verify.md)), so a frozen clip fails
verification instead of shipping. Do not rely on noticing it by eye: every
individual frame of a frozen clip looks completely correct.

### The playhead is lerped

Scroll never writes `currentTime`. It writes a target, and a standalone rAF loop
walks the clip toward that target at a fixed fraction per frame. Wheel events do
not arrive at a constant rate, so a 1:1 write reproduces every gap in them and
the clip reads as a stutter rather than a glide. Three mechanisms, all on by
default:

- **Lerp 0.18 per frame.** `data-sc-lerp` overrides it, on the mount root for the
  whole page or on one `<video>`. Clamped to 0.02 to 1, and never read as 0, since
  a 0 lerp is a playhead that never moves. Reach for it only when a page's clips
  are short enough that 0.18 visibly lags the hand. Under reduced motion the
  rate is 1.0, which is no smoothing at all.
- **Deadband** of 8ms desktop, 20ms mobile. A write smaller than that costs a
  seek and shows nothing; on a phone it costs more than it shows.
- **Seek coalescing.** No seek is queued while the decoder is still resolving the
  last one. A fast flick otherwise piles seeks up and freezes the clip.

An offscreen clip that has already reached its target stops being touched at all.

This applies to every scrub clip on the page, act or worldflight. It is also why
`shoot.mjs` waits for the playhead to arrive before each shot: a frame captured
mid-lerp is a frame the page never actually holds.

---

## 2. `pin`: the frame holds, the content advances

The workhorse, and the cheapest premium effect there is. The stage sticks for a
few viewport-heights while copy states cross over inside it. Use it when the
beat is an argument rather than an image.

```html
<section data-sc-act="pin" data-sc-span="3" data-sc-drift="#12100E">
  <div data-sc-stage class="pf-argument">
    <p class="sc-lede" data-sc-cue="0.02 0.34">A cup of coffee. Then a shake.</p>
    <p class="sc-lede" data-sc-cue="0.30 0.66">Two things to buy, carry and wash.</p>
    <p class="sc-lede" data-sc-cue="0.62">Or one can.</p>
  </div>
</section>
```

**Minimum useful span is about 1.2.** A pinned act's travel is
`max(height - viewport, 1)`, so at a span of 1 or below that is one pixel:
progress jumps 0 to 1 between two scroll notches and every cue, reveal and
`--sc-p`-driven animation inside the act snaps instead of running. A short quiet
act is exactly when an author reaches for a small span, which is exactly when
this bites. If the beat genuinely wants less than a screen of travel, it is a
`flow` act, not a pinned one.

Cue windows overlap by design: the outgoing line is still fading while the next
arrives, so the reader never faces empty space. A gap between cues reads as a
loading failure. Overlap by roughly 15% of the act.

The last cue takes one value and holds, so the act ends on a statement rather
than fading to nothing before the next section arrives.

### The cue contract

`data-sc-cue="from [to [rampIn [rampOut]]]"`, all in act progress (0 to 1).

| Form | Behaviour |
|---|---|
| `"0.2"` | fades in at 0.2 and **holds to the end of the act** |
| `"0.1 0.6"` | in, plateau, out. Ramps default to 30% of the window each |
| `"0 0.78 0"` | **greet**: already at full opacity when the act begins, then fades |
| `"0.1 0.9 0.15 0.4"` | fast in, long slow out |
| `"0 1 0 0"` | **greet and hold**: full at p = 0, no ramp at either end |

The plateau is the point. Without one a cue is a triangle that touches full
opacity for a single instant, so the reader has to stop on exactly the right
pixel to see the line at full strength and every heading reads slightly faded.

Rules the verification pass will catch you on:

- **A hero cue needs the greet form.** `"0 0.7"` ramps up from nothing, which
  means the landing view, the one screen every visitor sees, has no headline on
  it. Use a third value of `0`.
- **The last act's cue must hold.** Give it one value. A closing CTA on a
  two-value cue fades out before the page ends, and the final screen is empty.
- **Only the last act may hold.** This is the inverse of the rule above and it
  is the one that bites. The engine parks a cue only once its act is a viewport
  and a quarter out of range, so a one-value cue on a *middle* pinned act stays
  lit through the entire un-pin slide: the line travels a full viewport upward,
  crosses any fixed header, and overlaps the section that follows. It is
  invisible until you measure it, and it shows up as a contrast failure on a
  headline nobody meant to still be on screen. Every act except the last closes
  its final cue with a two-value window ending at 1.
- **Ground or greet.** A pinned stage becomes fully visible roughly a viewport
  *before* its own progress leaves 0, so any pinned act whose first content is a
  plain two-value cue shows an empty stage for that whole travel. Give the act
  either a ground that is already there (an image, a held frame, a colour that
  is doing work) or a first cue in the greet form. The closing act needs a
  ground, because its hold cue cannot also greet unless you use `"0 1 0 0"`.
  The rule covers *any* progress-gated content on a pinned act, not just engine
  cues: a bespoke panel that populates from scroll state has the same empty-stage
  window and needs the same ground.

---

## 3. `pan`: vertical scroll, lateral travel

Sideways movement reads as *breadth* where vertical reads as *argument*. Use it
for a range, a lineup, a timeline. Do not use it for a hierarchy: the first item
in a rail is not read as the most important one.

```html
<section data-sc-act="pan" data-sc-span="3.2">
  <div data-sc-stage>
    <div class="pf-rail" data-sc-pan="0.08">
      <article class="pf-flavour">…</article>
      <article class="pf-flavour">…</article>
      <article class="pf-flavour">…</article>
    </div>
  </div>
</section>
```

The engine measures `scrollWidth` against the viewport and travels exactly the
overflow, so the last item lands flush at the right edge. `data-sc-pan="0.08"`
adds 8% overshoot if you want a breath after the final card.

Span rule: roughly 1 viewport-height per item, plus 1. Four cards want ~5.

**Measure the overflow, do not assume it.** The engine travels exactly
`scrollWidth - viewport`, so a rail narrower than the viewport travels **zero**
and the act becomes a pinned stage holding one motionless screen for its whole
span. Three cards at `clamp(16rem, 26vw, 24rem)` measured 1368px against a
1440px viewport: overflow **-72px**, travel 0, and the reader turns the wheel
through two viewport-heights of nothing. It is width-dependent, so it can be
correct on a phone and dead on a desktop at the same time, which is exactly how
it survives review: the mobile sheet pans and the desktop sheet looks like a
still. **The harness did not catch it** and reported `no dead scroll detected`
on every pass, so this is a manual measurement, not something a green run
covers:

```js
const rail = document.querySelector(".rail");
rail.scrollWidth - innerWidth        // must be a healthy positive number
```

Aim for at least half a viewport of overflow. If three items do not reach it,
the fix is not wider cards, it is **more rail**: put the act's heading in as the
first item and a closing note as the last. Both earn their place (the heading
stops competing with fixed chrome, the note gives the rail a resolution instead
of an end), and they add the width the travel needs.

**Give the rail's items a staggered settle driven from `--sc-p`.** Lateral
travel alone reads as a slideshow on rails; items that arrive in sequence read
as a drawer being pulled. Exempt the first item, because a pan act needs its
opening content already present, and floor the opacity around 0.55 so a
not-yet-settled card reads as arriving rather than as failing to load. Gate the
whole thing to `prefers-reduced-motion: no-preference`, or the scroll-region
fallback inherits a page of half-faded items.

**Card copy is read while cropped.** Items enter and leave through the viewport
edges, so a heading is half-shown for most of its life and two half-headings
side by side can read as a third word (`TRACE` beside `EVALUATE` becoming
`RE EVALUATE`). Keep card headings to one short word or two, and expect every
line of card copy to be read partially cut off.

**Reduced motion.** The rail's transform is not decoration, it is the
navigation, so it cannot simply be zeroed the way parallax is: that parks the
act on its first screenful and makes every item past the fold unreachable. The
engine handles the floor for you, turning the stage into a native
`overflow-x: auto` scroll region with proximity snapping, so the same items stay
gettable without motion. If your rail reads better stacked or as a grid at that
point, override it in your own CSS under `prefers-reduced-motion` (the agency
reference build relays its three phases out as a grid at desktop widths).

---

## 4. `reveal`: a wipe is a change of state

`clip-path` eating in from an edge. It costs nothing and it reads as
transformation, which makes it right for the beat where something becomes
something else. Wrong for merely introducing an image, where a cue is enough.

```html
<figure data-sc-reveal="up" data-sc-reveal-at="0.15 0.55">
  <img src="assets/03-carry.webp" alt="…">
</figure>
```

`up` `down` `left` `right` `iris`. Reach for `iris` roughly once per page; it is
the loudest of the five and stops reading as intentional if repeated.

A wipe that runs edge to edge across a full-bleed image is a transition. A wipe
on a small element is a fidget. Use it big.

**`clip-path` is relative to the border box, not to the ink.** A reveal on type
set with `line-height` below 1 has a border box shorter than the glyphs, so the
wipe clips the ascenders off the top and the descenders off the bottom and every
figure renders as a plain bar. Display numerals and drop figures are the usual
casualties, because those are the ones set tight. Either give the reveal element
room (`line-height: 1` and padding to cover the overshoot) or put
`data-sc-reveal` on a wrapper and leave the type's own box alone. No static
audit catches this; only a rendered screenshot does.

---

## 5. `kinetic`: type that assembles

Splits a heading into lines, words or characters and staggers them across the
cue window. Lines are almost always right; words for a short punch line;
characters approximately never, because it turns reading into waiting.

```html
<h2 class="sc-display sc-display--lg" data-sc-cue="0.1 0.7" data-sc-kinetic="lines">
  Coffee that pulls its weight.
</h2>
```

Each unit slides up from behind a mask, so it enters from a clean edge rather
than simply fading. Line masks reserve room for descenders; a mask clipped to
the line box shears the tails off g, y, p and j, and that is the single most
common way this effect looks broken.

Line splitting measures real line boxes, so it re-runs after `document.fonts.ready`.
Do not call it on text that is still loading its face.

**One kinetic headline per act, at most.** Two competing for attention is noise,
and every heading assembling the same way is the templated rhythm this skill
exists to avoid.

---

## 6. `parallax`: layers at different rates

Depth from differential movement. Subtle or nothing: past roughly 200px of total
travel it stops reading as depth and starts reading as a bug.

```html
<div class="pf-layer pf-layer--back"  data-sc-parallax="-1.4">…</div>
<div class="pf-layer pf-layer--mid"   data-sc-parallax="-0.6">…</div>
<div class="pf-layer pf-layer--front" data-sc-parallax="0.35">…</div>
```

**The rate is in hundreds of pixels, not viewport fractions.** The engine writes
`rate * (p - 0.5) * 100` px, so the total travel across a whole act is
`rate * 100` px regardless of screen height. At 0.35 that is 35px across three
viewport-heights of scroll, which is invisible: usable values are roughly 0.3 to
1.5 for a layer inside a frame and 1 to 2 for a full-bleed bed.

Negative moves up faster than the scroll, which pushes an element back. Three
layers is plenty; five is a diorama.

Never put body copy on a parallax layer. Text the reader is trying to read
should not move relative to the thing they are reading it against.

---

## 7. `count`: numbers that land

```html
<span class="sc-nums" data-sc-count="0 4200" data-sc-count-at="0.1 0.5">0</span>
```

Formatting is inferred from the target: decimals from its decimal places,
thousands separators above 10,000 **or whenever the target itself is written
with one**. Write the target exactly as it should render, commas included
(`data-sc-count="0 3,500"`); the engine strips them before parsing. The element
gets `tabular-nums` so the layout does not jitter while the digits change.

**Only real numbers.** A counter is a truth claim with motion attached, which is
what makes it persuasive and what makes an invented one a liability. If the
brand has no verified figure, there is no counter. Check the brand's rules
first; several forbid this outright.

**A concept, fictional or pre-launch brand has no verified figures, so it has no
counters.** The device suits a SaaS or agency page and it will look good in the
score table, which is exactly the trap: every number you could put in it would
be invented. Decide this before you design an act around a number, not after.
Real brand, real stats, or a different device.

---

## 8. `flow` + `in`: ordinary sections, done well

Not everything should be pinned. A page of nothing but pinned acts is
exhausting, and the contrast is what makes the pinned ones land. Normal
document sections with a reveal-on-entry are the rest of the page.

```html
<section class="sc-section">
  <div class="sc-wrap sc-stack" data-sc-stagger="70">
    <h2 class="sc-display sc-display--md">What is actually in it</h2>
    <p class="sc-body">…</p>
  </div>
</section>
```

This fires **once**, on entry, via IntersectionObserver. Content that re-hides
when the reader scrolls back up is a defect, not an effect. Stagger between 30
and 80ms; longer feels slow.

**A flow section directly after a pinned act takes reduced padding.** The pinned
stage needs a full viewport to scroll off, and full `--sc-section` padding on
top of that delays the flow section's first content by another screen. The
harness will not call it dead scroll, correctly, because the stage is moving.
The reader still sees a near-empty screen. Cut the block padding there and start
the first reveal near `p = 0`.

---

## 9. Pointer devices: interactivity that is not scroll

Scroll is a one-dimensional input. A page that only responds to scroll is a
film. These make it respond to the reader being *present*.

```html
<div class="pf-card" data-sc-tilt="7">…</div>
<a class="pf-cta" data-sc-magnet="0.28">Find a stockist</a>
<section data-sc-spotlight>…</section>
```

- `tilt`: 3D rotation toward the pointer. 5 to 9 degrees. Past 12 it is a toy.
- `magnet`: the element drifts toward the pointer. 0.2 to 0.35. Primary CTA
  only; a page of magnetic elements is unusable.
- `spotlight`: publishes `--sc-mx` / `--sc-my` for a light that follows the
  pointer across a surface.

**`magnet`, `parallax` and `cue` all write `transform`, so they cannot share an
element.** The magnet writes every frame in its own rAF loop, so it silently
wins and the cue's entrance rise is discarded. On a magnetic CTA, set
`data-sc-rise="0"` so the cue writes a no-op instead of losing a visible
animation to a race. A cued element that wants its own continuous transform
should drive an inner wrapper from `--sc-p` in CSS rather than stacking a second
device on the same node.

All three interpolate toward the pointer rather than tracking it directly.
Direct tracking reads as artificial because it carries no momentum. All three
are gated to `(hover: hover) and (pointer: fine)` and disabled under reduced
motion, so touch never fires a false hover.

---

## 10. `drift`: the ground moves with you

Not an act. A property of acts, and the thing that makes a page feel like one
continuous place rather than a stack of slides.

```html
<section data-sc-act="scrub" data-sc-drift="#0A0806"> …
<section data-sc-act="pin"   data-sc-drift="#161210"> …
<section data-sc-act="pan"   data-sc-drift="#0E1412"> …
```

The page ground interpolates between the values as each act takes over. Keep the
whole set inside one theme family. Drifting from near-black to cream mid-page
is not atmosphere, it is the reader wondering whether they clicked something.

Three to five stops across a page. Small steps. The effect should be invisible
frame to frame and obvious top to bottom.

**Scoping: drift belongs to the first act whose progress is strictly between 0
and 1.** That is the right pick when acts are long enough that only one is ever
part-way through, and it is wrong the moment several short acts satisfy it at
once. On a page of twelve short cuts the ground shown belongs to a section the
visitor left a screen ago, so a colour arrives late and reads as a bug rather
than as a slow lag. The advice above ("three to five stops") is written for six
long acts.

**If several acts can be part-way through at the same time, paint grounds per
section instead of drifting.** Set an opaque background on each section and let
the change land on a hard edge. That is also what a cutlist or a chaptered page
wants on its own terms: a cut is not an interpolation, and interpolating between
two chapter grounds is precisely the softness those grammars exist to refuse.
Drift is for pages that are one continuous place.

---

## Composing an act

Devices stack inside one act. A pinned stage can hold a scrubbing clip, a
parallax layer, a kinetic headline and a spotlight at once. The limit is
attention, not the engine: **one thing should be the reason each act exists**,
and everything else in it is support.

If you cannot say in one sentence what an act's moment is, it does not have one.
