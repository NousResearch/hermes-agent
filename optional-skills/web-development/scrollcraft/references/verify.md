# Verify

A scroll page cannot be checked by looking at it. It has no single state: every
scroll position is a different frame, and the failures live between the two you
happened to look at. So walk it mechanically.

```bash
cd <build project>
npm i playwright-core                       # once

node <skill>/scripts/serve.mjs --root . --port 4500 &
node <skill>/scripts/shoot.mjs --url http://localhost:4500 --out lab/shots
node <skill>/scripts/shoot.mjs --url http://localhost:4500 --out lab/mobile --width 390 --height 844
node <skill>/scripts/shoot.mjs --url http://localhost:4500 --out lab/reduced --reduced-motion
```

Then **read `sheet.png`**. The whole point of shooting contiguously is looking
at the frames side by side; a folder of PNGs does not get looked at that way.

Two setup facts that will otherwise waste a pass:

- **Serve it.** `file://` blocks the Blob fetch the engine uses for clips, so
  the page silently falls back to posters and proves nothing.
- **Real Chrome, not bundled Chromium.** Chromium ships without an h264
  decoder, so every clip fails to paint and the run "passes" against posters.
  `shoot.mjs` already resolves installed Chrome; override with
  `SCROLLCRAFT_CHROME`.

---

## What the harness reports

It samples **within each act** (default 6 positions per act) rather than
uniformly down the document. Uniform sampling moves every position whenever you
change any section's height, so findings appear and vanish with unrelated edits.

**DEAD SCROLL**: consecutive positions where nothing changed: no cue moved, no
clip time advanced, no rail travelled, no wipe progressed, no stage shifted.
Real dead scroll means the reader is turning the wheel and being given nothing.
Fix by shortening the act's span or adding a cue.

**Bespoke fixed stages must report their visible state.** A split stage, live
canvas, or other page-local system can use ordinary `flow` acts only as scroll
markers while every visible change happens on a fixed layer outside the engine.
The harness cannot infer that layer's semantics. Put `data-sc-verify-state` on
the fixed stage and update its value to a compact signature of the values that
actually paint: divider position, scene opacity, canvas phase, custom film
time, or similar. The detector then checks those flow spans too.

Do not publish raw scroll progress just to make the check green. If progress is
changing while the composition is not, that is the exact failure this path is
meant to catch. Round and publish the rendered values. For an intentional
resolved hold, set `data-sc-verify-hold="true"` only while the hold is active.
Reduced-motion fixed stages may use the same attribute for deliberately stable
frames, which still require manual contact-sheet review.

**FROZEN CLIP**: a scrub stage is on screen, the reader is scrolling, and the
clip's playhead is not moving. Dead scroll cannot see this, because the stage
itself *is* moving: a still photograph is sliding up the page, which is the
worst-looking failure this kit can produce and the one that most reliably makes
a page feel broken.

The harness samples each scrub act's **entry and exit slides**, not only its
pinned travel. That gap is why this went undetected for four builds: a pinned
act's samples were taken at `top + (h - vh) * p`, which never visits the viewport
of scroll on either side where the stage is visible and the clip is parked. A
hold on the first or last frame is always reported. A hold in the middle is only
reported once it outlasts any plausible `data-sc-dwell` settle, since that settle
is a deliberate effect. The check is skipped under reduced motion, where no clip
is ever fetched on purpose.

The fix is almost never per-page: the engine maps clip time across the stage's
whole visible life by default. A page that reports this has usually opted out
with `data-sc-clip-map="travel"`, or is running an engine copy from before that
default existed. See [devices.md §1](devices.md).

**CUES THAT NEVER PEAK**: an element that never reaches full opacity anywhere.
Usually a cue window too narrow for its act, or ramps that eat the whole window.
Widen the window or set explicit ramps. A kinetic heading is read through its
line units, not through the element: the engine forces the element itself to
opacity 1 and carries the real value on `.sc-split__i`, so reading the element
reports every kinetic headline as fully present even on frames where every line
is at 0.

**CONTRAST**: measured on the **composited page**, not on the source video. The
harness hides the text, re-shoots the same frame, and samples the real
background under each line, so scrims, gradients and blends are all included.
Elements with their own opaque background are graded against that fill instead.

Three things it gets right that a hand-rolled version usually does not:

- **The direction is picked per line.** Light type on a dark page fails on the
  brightest patch under it; dark type on a light page fails on the *darkest*
  one, and grading that against the brightest patch is the most lenient reading
  available, so a high-key page can report clean over text that is failing. The
  harness compares the ink to the mean background and grades against whichever
  extreme is on the ink's own side.
- **The sampled rect is clamped to the viewport.** The part of a pinned act's
  copy that has scrolled above the fold is not on screen, so what sits in those
  pixels is not behind anything the reader can see.
- **Fixed chrome is hidden with the text.** A fixed bar paints in *front* of
  what scrolls under it, so its own mark is not the background behind a headline
  passing beneath it.

This is the check no static audit can do: the frame under a headline changes as
the clip scrubs, so text can clear 4.5:1 against the poster and fail badly three
hundred pixels later.

### The scrim has to be a SIBLING of the copy, never a child

The pass hides `[data-sc-cue],[data-sc-cue] *,[data-sc-copy],[data-sc-copy] *`
before photographing the frame underneath a line. `visibility: hidden` hides an
element's pseudo-elements too, so a scrim written as `.mycopy::before` is hidden
along with the text it exists to protect, and the pass grades the line against
the raw film every time.

The tell is unmistakable once you know it: **you strengthen the scrim and the
reported numbers do not move at all.** Not "improve slightly", not "move by a
tenth": byte-identical, because the thing you changed was never in the
measurement. If a contrast number is unchanged to two decimals after a real
change, stop tuning and check what is actually being composited.

A very high mean against a very low worst (`1.21:1 (mean 12.83)`) is the same
finding seen from the other side: the type is fine almost everywhere and there
is a bright patch under it that nothing is covering.

Two shapes that work:

- `.sc-world__scrim` in the copy layer, which is what worldflight.md ships and
  which survives the hide because it carries no `data-sc-copy`.
- One plate per block, mounted as a sibling and driven from the page's own JS.
  `orrery` sizes each plate off its block's untransformed box (set
  `transform:'none'`, read the rect, put it back, so the engine's ±2vh copy
  drift does not skew the measurement) and each frame copies the block's own
  inline opacity onto its plate, so the plate tracks the engine's window with no
  duplicated window maths.

### Known limitations of the contrast pass

Real, and worth knowing before you trust a green run:

- **Cues are keyed by their text.** Two cues that share a string, which
  taste.md's "one label per intent" rule actively encourages, are collapsed into
  one row, and the reported worst frame is the worse of the two.
- **Lines under 0.85 opacity are skipped.** A headline parked at 0.6 over a
  bright frame is never graded, so "contrast clean" can still hide a legibility
  problem. Look at the sheet for anything that reads washed out.

  **Author the fade-outs to land between sample positions.** The harness samples
  a fixed number of positions per act, so a ramp-out that happens to straddle one
  puts a half-faded headline on the sheet: graded by nobody, and read by eye as
  ghost type over the frame. **Fix the ramp, not the sampling.** Shorten
  `rampOut` so the cue is at full opacity at one sample and gone by the next,
  rather than sitting at 0.5 on the sample in between. Widening the sample count
  only finds more half-faded frames; it does not make the page look better,
  because a real reader stopping on that pixel sees exactly what the sheet
  shows. A cue caught mid-fade over a bright frame is a real defect, not a
  sampling artefact.
- **The floor is not size-aware.** It reports below 3:1 as a failure and 3:1 to
  4.5:1 as thin. WCAG allows 3:1 for large text, so a display headline in the
  thin band is usually fine and a 16px caption in it is not.
- **Acts with no `[data-sc-cue]` elements are not graded at all.** Copy on plain
  canvas is a static case, but it is unmeasured.
- **Ordinary `flow` acts are excluded from dead-scroll checks.** Static flow is
  normally correct. A bespoke fixed experience built over flow markers must use
  `data-sc-verify-state`, or the harness will skip its visible timeline and can
  report a dead opening as healthy.
- **A `pan` act whose rail does not overflow is reported as healthy.** The
  `pigment` build ran a rail measuring 1368px inside a 1440px viewport, so it
  travelled zero for its entire 2.1vh span, and every pass printed `no dead
  scroll detected`. Measure `rail.scrollWidth - innerWidth` yourself; a green run
  does not cover it. See devices.md §3.

**Console errors and failed requests**: a 404 on a clip degrades to a poster
silently, which looks fine and is not.

---

## What the harness cannot tell you

Read the sheet for these. They are the ones that matter most.

- **Whether the composition is any good.** Copy landing on the busiest part of
  the frame, a subject cropped at an unfortunate point, an act whose end frame
  is a dark empty corner.
- **Whether the motion is smooth.** Contiguous frames prove the clip advances;
  they do not prove it advances evenly. Watch the contact sheet for a move that
  lurches, reverses, or stalls in the middle.
- **Whether the page means anything.** Six acts that each work and together say
  nothing is the most expensive failure available here.

---

## The manual passes

**Reduced motion.** Clips are never fetched, posters hold, copy still cues. The
page must remain comprehensible, not merely not-crash. This doubles as the
low-bandwidth check.

Comprehensible includes **reachable**: check that no content was deleted rather
than merely stilled. A `pan` rail is the case that bites, because zeroing its
transform parks it on its first screenful. The engine now hands the stage back
as a native scroll region, so confirm on the sheet that the rail shows real
content and that items past the fold can still be got to. Nothing in the harness
reports this; it reads as a page behaving correctly.

**Credit accounting.** `kie.mjs probe` reports a balance, not a delta, so a
build's spend is a before-and-after subtraction. That subtraction is only valid
if nothing else is generating against the same key. When builds run in parallel,
or when a settlement lands late, the deltas overlap and each build will claim
some of another's spend (three parallel builds each read the same 7597 → 7067 and
each reported 530). Either serialise generation, or cost the build from the
per-call model prices in [assets.md](assets.md) against the calls you actually
made, and treat the probe delta as a ceiling.

**And the per-call sum overstates real spend in the other direction.** Two
reconciliations against the account ledger, each with no other consumer, put
actual debits at roughly **0.4x** the documented unit rates: a fleet whose
per-call sums came to ~1447 credits was debited 530, and a three-build run whose
per-call sums came to 2252 was debited 856. Both land near the same ratio. So a
build report should say what the per-call sum is *and* that it is a planning
ceiling rather than the amount billed. Reporting the sum as the cost is the
honest default, because it never under-claims; reporting it as *measured* spend
is wrong. Neither number is the other's substitute: the probe delta bounds a
parallel run from above, the per-call sum bounds a serial one from above, and
only a ledger read with a single consumer settles it.

**Mobile.** Pinned stages use `100svh` so the URL bar does not cause a jump.
Copy reflows and does not collide with the fixed bar. Confirm the phone encodes
actually load. Check the portrait crop of every clip: a 16:9 move composed
around left-hand negative space loses exactly that space at 9:16
(see [assets.md](assets.md)). Mobile is a first-class target, not a check at
the end: the phone clips are cut portrait, the lerp is retuned for touch, tap
targets are grown, and every one of those is authored, not inherited.

### The phone is a different machine

Headless Chrome on the build box cannot reproduce an iPhone's video decoder,
its autoplay policy, Low Power Mode, or touch scrolling. On one build every
probe reported the hero clip scrubbing perfectly for **four consecutive
rounds while the real phone showed a frozen frame**. A green harness run says
the page is correct where the harness runs. It says nothing about iOS video.

What iOS does to a scrub clip, and what the engine now handles for you:

- iOS will not *paint* a muted video that has never been played. Seeks land,
  `seeked` fires, and the picture stays on one frame. The decoder has to be
  primed with one `play()`/`pause()`.
- The engine primes each clip at `loadedmetadata` (a muted inline `play()`
  needs no gesture outside Low Power Mode) and retries on `touchstart`,
  `touchend`, `pointerdown`, `click` and `scroll`. `touchend` matters: the
  HTML spec's activation-triggering events include `touchend` but **not**
  `touchstart`, so a Low Power Mode phone that rejects the touchstart attempt
  gets a valid one when the finger lifts.
- A prime must be re-attemptable per clip. A one-shot prime on first touch
  loses a race: the reader touches to scroll within the first second, while
  the hero's megabytes are still downloading, and the shot is spent on a
  sourceless element. The tell is exactly "the first clip is frozen and every
  later one works".
- iOS may leave a `play()` promise pending forever, and may leave `seeking`
  true forever. Both were permanent silent freezes; the engine now releases
  the priming flag on a timer and re-issues any seek stuck past 700ms. The
  reveal also fires on a 2.5s timeout, never only on `seeked`.

Do not re-implement any of that in page JS, and do not strip it when copying
the engine. If a phone still shows a frozen clip, the cause is past what this
machine can measure, which is what the next section is for.

### Ship the diagnostic with the site

You get one question per round with a real device, so make the round count.
`references/device-diag.html` is a standalone page that scrubs the suspect
clip two ways (blob URL, exactly as the engine loads it, and direct file src)
beside a known-good clip, prints a MOVING / FROZEN verdict over each pane,
and reports prime results, seek counts and distinct painted frames. Edit its
`TESTS` array to point at the build's own clips, deploy it next to the site,
and one screenshot from the phone isolates the layer: blob loading, the file,
the device's decode policy, or the engine's lifecycle. Deploy it **with** the
first mobile fix, not after the fourth.

### Ask what differs before asking what's broken

The debugging lesson that cost three wasted rounds: "desktop works, the phone
does not" reads as a platform difference and invites platform theories
(codecs, keyframes, resolution). **"One clip works and another does not, on
the same device"** cannot be a platform difference. Before theorising, write
down every way the working case differs from the broken one; the bug lives in
that list. On the build above the list had one entry: the hero is first, so
it loads while the first touch is being spent.

**Keyboard.** Tab through. Focus order matches visual order, the focus ring is
visible against every ground it crosses, and nothing reachable is parked at
opacity 0. Cues set `pointer-events: none` when faded, but a focusable element
inside a faded cue is still a trap.

The engine helps here but does not finish the job, and the gap is specific:

- **It handles the ordinary case.** On `focusin`, if the focused element is
  inside a `[data-sc-act]` and its own cue computes under 0.85, the engine
  scrolls it to the centre of the viewport with `behavior: 'instant'`
  (`smooth` would animate a multi-screen glide with focus off screen the whole
  way). On a `flow` act, centring the element also opens its cue, because the
  element's viewport position and the act's progress move together.
- **It does not fix a pinned act, and cannot with this approach.** A pinned
  stage is `position: sticky`, so the control holds *one* viewport position for
  the entire act. Centring it is then only achievable by scrolling backwards out
  of the act, which parks progress at 0 and leaves the cue dark. Measured: a CTA
  cued at 0.75 on a 3vh pinned act sits at viewport y=70 from progress 0 to
  0.875; `scrollIntoView({block:'center'})` from inside the act lands *before*
  the act's top, at progress 0, cue opacity 0. The control is on screen and
  still invisible.

**On a pinned act, park the act at the progress where the focused element's own
cue is open.** That is page-local work, because only the page knows which cue
belongs to which control, and because act progress runs through `dwell()` when
the act has any, so the scroll target is not a straight inverse of the cue
window. The descent build does exactly this. If a pinned act carries a focusable
control, write that handler and assert it; do not assume the engine covered you.

**Fresh eyes.** Look again later. Timing you tuned for twenty minutes reads
differently when you have forgotten what it is supposed to do.

---

## Failures worth knowing about

Each of these shipped once during this skill's own build, and each looked fine
until it was measured.

| Symptom | Cause |
|---|---|
| A hero headline wrapped to six lines | `max-width` in `ch` on a **container**: `ch` resolves against the container's font-size, not the display size of the heading inside it |
| Centred copy hanging off the left edge | `inset-inline` declared **after** `left: 50%`; the shorthand resets `left` to auto |
| An act that never pins, silently | An author rule setting `position` on the stage. The engine now warns in the console |
| A stray headline painted over a later section | Cues frozen at their last value when their act scrolled out of range |
| A clip stuck on its poster at the top of its act | The reveal waits for a `seeked` event, and a clip already at time 0 never seeks |
| A closing CTA that fades out before the page ends | A two-value cue on the last act, plus a tall section after it |
| Copied headings reading "even whenbreakfast" | Line-split spans abutting with no whitespace between them |
| A headline from act 2 overlapping act 3, failing contrast on the way | A one-value hold cue on a middle act. Only the last act may hold |
| A phone-only contrast failure on a trail-anchored act | The trail scrim aimed at the corner the copy leaves below 860px. The engine now switches it to a band |
| A rail act that shows one frozen screenful under reduced motion | `[data-sc-pan] { transform: none }` deleting the navigation. The engine now falls back to a scroll region |
| Two washed-out video acts no scrim tuning could rescue | Flat supplied footage with no white point. Grade the intermediate, not the CSS |
| A blank stage for the first viewport of a pinned act | A two-value first cue with no ground. Ground or greet |
| A rail heading dragged off-screen under reduced motion | The scroll-region fallback snap-centres a single wide track; keep the act heading outside the region, or give the rail multiple snap stops |
| Keyboard focus landing on a control nobody can see | The browser's scroll-into-view parks the element barely on screen, which is where its cue has not opened, and the opacity check still passes. **The engine now centres it on `focusin`** when the element is inside a `[data-sc-act]` and its cue is under 0.85. That fixes the off-screen half. See the note below for what it does not fix |
| A figure or drop numeral rendering as a plain bar | `data-sc-reveal` on type with `line-height` below 1. `clip-path` is relative to the border box, so the wipe eats the ascender and descender. See devices.md §4 |
| An image three times too tall, pushing its own label off the fold | `width` overridden in CSS while `height` still resolves to the HTML attribute. Override both or neither. See taste.md |
| An inverted section rendering its old ink, graded in the wrong direction | `--sc-ink` redefined on the subtree without restating `color`. See taste.md |
| A ground colour arriving a section late | `drift` on a page of short acts; several are part-way through at once. Paint grounds per section. See devices.md §10 |
| Every cue and reveal in a quiet act snapping 0 to 1 | A pinned act at `data-sc-span` ≤ 1, which is one pixel of travel. Minimum useful pinned span is ~1.2 |
| A clip that scrubs beautifully, stops, and then slides up the page as a still photograph | The clip was mapped to the act's pinned travel, which is 0 through the entire entry slide and 1 through the entire exit slide. The engine now maps clip time across the stage's whole visible life by default. See devices.md §1 |
| A custom fixed stage passing while its first screens do nothing | The page used `flow` markers, which are intentionally excluded from ordinary dead-scroll checks, but published no `data-sc-verify-state`. Report the actual rendered state and declare only genuine resolved holds |
| The hero clip frozen on a real iPhone, later clips fine, every probe green | iOS never paints an unplayed muted video, and the one-shot gesture prime was spent while the hero was still downloading. The engine now primes per clip at `loadedmetadata` and retries on every gesture, including `touchend` |
| A phone clip soft and stuttering while the same file is smooth on desktop | A landscape mobile encode in a portrait viewport: cover-fit decoded the full frame and threw three quarters of it away. Cut the phone clips portrait from the masters (see assets.md) |
| Four rounds of mobile fixes verified green, phone still broken | Headless Chrome cannot reproduce the iOS decoder, Low Power Mode, or touch. Deploy `references/device-diag.html` beside the site on the first mobile report and let the phone answer |

The first three are invisible to every check except looking at rendered output.
That is the argument for this whole pass.

Operational note: a `shoot.mjs` run can take the background server process down
with it when it finishes. Check the port before the next pass and restart
`serve.mjs` if it dropped.


## The harness will photograph the wrong site without telling you

`serve.mjs` fails with `EADDRINUSE` if something already holds the port. When
that server was started in the background, the failure is in a log nobody is
reading, and `shoot.mjs` then gets a perfectly good `200` from **whatever else
is on that port**. It walks that page, finds its worldflight, and writes a full
contact sheet and a clean report for a site you did not build.

Confirm the port is serving YOUR build before trusting any run:

```bash
curl -s http://localhost:45XX | grep -o "<title>.*</title>"
curl -s -o /dev/null -w "%{http_code}
" http://localhost:45XX/assets/leg01.mp4
```

A 404 on an asset you know exists is the fastest tell.
