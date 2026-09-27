# Worldflight: the continuous-world page mode

Act mode cuts the page into pinned blocks. That is the right shape for a page of
chapters and the wrong shape for one unbroken camera move, and building a
continuous world out of acts produces exactly the page an owner described as
"awful": you scroll down, the stage unsticks, a static page slides past, clean
horizontal edges travel up the screen, and then you start scrolling down again.
Every one of those defects is the same defect. A pinned act is a block in the
document, and a document made of blocks has seams.

Worldflight removes the seams by removing the blocks.

There is **one** `position: fixed` stage for the whole page. Every leg of the
flight is mounted in it at once and stays mounted. The only element in document
flow is an empty spacer. Scroll drives two things and nothing else: the film
timeline and the opacity of the overlay. Nothing travels, nothing pins, nothing
unpins, and there is no boundary anywhere for a seam to show at.

---

## 1. The markup

```html
<div data-sc-mode="worldflight" data-sc-seam="0.12">

  <div data-sc-world>
    <div data-sc-segment data-sc-w="0.95" data-sc-linger="0.3"
         data-sc-waypoint="Surface">
      <img class="sc-world__poster" src="assets/p1.webp" alt="" decoding="async">
      <video data-sc-src="assets/leg1.mp4"
             data-sc-src-mobile="assets/leg1-m.mp4"></video>
    </div>
    <div data-sc-segment data-sc-w="0.9" data-sc-linger="0.42"
         data-sc-waypoint="Thermocline">
      <img class="sc-world__poster" src="assets/p2.webp" alt="" decoding="async">
      <video data-sc-src="assets/leg2.mp4"
             data-sc-src-mobile="assets/leg2-m.mp4"></video>
    </div>
    <!-- legs in flight order, as many as the world has -->
  </div>

  <div data-sc-world-copy>
    <div class="sc-world__scrim sc-scrim sc-scrim--band"></div>
    <div class="sc-copy sc-copy--lead" data-sc-copy data-sc-window="hero"> … </div>
    <div class="sc-copy sc-copy--trail" data-sc-copy data-sc-window="0.38 0.66"> … </div>
    <div class="sc-copy sc-copy--lead" data-sc-copy data-sc-window="finale"> … </div>
  </div>

  <div data-sc-spacer aria-hidden="true"></div>
</div>
```

`ScrollCraft.mount(document)` as usual. The mode composes with nothing else on
the page: a worldflight page has no acts.

### Attributes

| Attribute | On | Default | What it does |
|---|---|---|---|
| `data-sc-mode="worldflight"` | mode root | n/a | Turns the page into one flight. |
| `data-sc-seam` | mode root | `0.12` | Crossfade band, in viewport-heights of scroll. Clamped 0.02 to 0.4. |
| `data-sc-world` | stage | n/a | The single fixed stage. Gets `.sc-world`. |
| `data-sc-segment` | leg | n/a | One leg. Holds a poster and a clip. |
| `data-sc-w` | leg | `1.3` | Scroll this leg owns, in viewport-heights. |
| `data-sc-linger` | leg | `0` | Dwell remap for this leg only. Clamped to 0.6. |
| `data-sc-waypoint` | leg | n/a | Label published on the waypoint event. |
| `data-sc-world-copy` | copy layer | n/a | Fixed overlay. Gets `.sc-world__copy`. |
| `data-sc-copy` | copy block | n/a | A windowed block of type. |
| `data-sc-window` | copy block | n/a | `hero` \| `finale` \| `from to [in [out]]`. |
| `data-sc-spacer` | spacer | n/a | The scroll track. Engine sets its height. |
| `data-sc-lerp` | root or `<video>` | `0.18` | Playhead smoothing. See devices.md. |

The engine generates no DOM here, the same as in act mode. It sets the spacer's
height, and it writes opacity, visibility and z-index on the legs. Everything
else is markup you wrote.

---

## 2. The scroll track

The spacer's height is **(sum of the leg weights + 1) viewport-heights**, set in
pixels so it and the stage are measured on the same ruler (the stage is sized in
`svh`, and on a phone `vh` and `svh` are different numbers).

The `+ 1` is not padding. Without it the track ends at the exact scroll position
where the last leg reaches progress 1, so the final second of the last clip is a
place the reader can never come to rest. One extra viewport gives the last
flight room to land.

Position along the track, `t`, is measured in viewport-heights, which is the
same unit the weights are written in. Leg *i* owns `[c_i, c_i + w_i)`, and its
local progress is `(t - c_i) / w_i`, remapped through `lingerEase`.

---

## 3. The seam

Two things make a boundary between clips invisible, and both are required.

**The assets have to match at the seam** (section 6). The engine cannot fix a
mismatched cut.

**The crossfade has to be one-sided.** Over the seam band the incoming leg fades
up from 0 to 1 while the outgoing leg holds at full strength underneath it, and
the outgoing leg only drops to zero once it is completely covered. Fading both
sides at once puts the page ground through the middle of every seam, which reads
as a flash, and it is the obvious implementation. z-index favours the current
leg (120) over the rest (100 + opacity × 10).

Each side of the band is half a seam width, so each leg holds its seam frame for
about 0.06vh of scroll. Those are exactly the frames the seam law matched, so a
held frame there is invisible by construction.

Nothing ever swaps a `src`. A src swap is a black frame, and a black frame is
the cut this mode exists to remove.

---

## 4. The copy contract

Copy lives in one fixed layer above the stage. Each block declares a window
against the **whole track**, not against a leg.

- `data-sc-window="hero"`: present from the first pixel, fades out by 0.62 of
  the first leg. A hero that fades IN has to fade in over an empty first screen,
  which is the one moment on the page with nothing else to look at.
- `data-sc-window="finale"`: fades in from 0.4 of the last leg and holds to the
  end.
- `data-sc-window="0.38 0.66"`: a plateau window across those track fractions:
  ramps in over the first 30%, holds at full opacity, ramps out over the last
  30%. Add a third and fourth number to set the ramps yourself. The plateau is
  not decoration: a pure triangle touches opacity 1 for one instant, so the
  reader has to stop on exactly the right pixel to see the line at full strength
  and every heading reads slightly faded.

**The only transform on the copy side is `translateY`, and it is capped at 4vh
across the whole window** (from +2vh to -2vh). Anything larger stops reading as
a layer over a moving world and starts reading as a second page scrolling at a
different speed, which is the cheapness this mode replaces. Pointer events are
handed back to a block only above opacity 0.5.

`.sc-world__scrim` is provided for a scrim div on the copy side. Shape it to
where the copy actually sits. The stock `.sc-scrim--band` tops out at 58% of the
frame, and footage that stays bright past that will fail the contrast pass even
though the page looks fine.

---

## 5. The route rail

The engine publishes the current leg index as `--sc-seg` and its local progress
as `--sc-segp`, on the mode root and on `:root`, and fires a `sc:waypoint`
CustomEvent (bubbling, `detail: { index, count, label, el, progress }`) whenever
the leg changes.

It renders no rail. A gauge, a map, a depth readout, a leg counter and a set of
chapter dots are all the same two numbers, and a runtime that ships one of them
ships it to every page that uses this mode. Build the rail in the page:

```js
addEventListener('sc:waypoint', (e) => {
  document.querySelectorAll('.rail__leg').forEach((el) => {
    el.setAttribute('aria-current', String(+el.dataset.leg === e.detail.index));
  });
});
```

---

## 6. The seam law for assets

A worldflight is only as good as the joins between its clips. Two architectures
work; nothing else does.

**Architecture A (preferred): chain on start images only.** Each leg is
generated from a start image and left to end wherever it ends. The next leg's
start image is a frame pulled from the previous leg's **encoded** mp4. Never
force an end-image wide shot: an image-to-image model asked to hit both ends
resolves the conflict by pulling the camera back, and every leg ends up as the
same wide establishing shot.

**Architecture B: connector legs.** Where two existing clips have to meet, cut a
short connector whose start frame comes from the previous leg and whose end
frame is the next leg's actual first frame.

Extract from the ENCODED mp4, not the source render. The encode changes the
pixels, and a poster or a chain frame taken from the pre-encode master does not
match the frame the browser will actually decode:

```bash
# last frame of the previous leg, as the next leg's start image
ffmpeg -sseof -0.15 -i legN.mp4 -frames:v 1 -q:v 2 chainN.png
# first frame of a leg, for its poster
ffmpeg -i legN.mp4 -frames:v 1 -q:v 3 pN.webp
```

### Encoding

Same rules as any scrub clip, and they matter more here because a worldflight
has more of them mounted at once.

- **GOP 8 desktop, GOP 4 mobile.** Scrubbing is random access; a long GOP means
  every seek decodes a run of frames and the playhead lags behind the hand.
- Ship `data-sc-src-mobile` for every leg. The engine picks it on coarse
  pointers and narrow viewports.
- Posters as WebP, extracted as above.

---

## 7. Loading

A leg is fetched only while the reader is within **±1.6vh** of it. Loading the
whole flight up front is tens of megabytes before the first frame paints;
loading on arrival means arriving at a poster.

Until a leg's first real frame has painted, its poster carries the move with a
push-in (`scale(1.03 + local × 0.14)`). A still that sits perfectly still while
the page scrolls announces itself as a placeholder; a slow push reads as the
camera already flying.

Under reduced motion **no clip is ever fetched**. The posters are the film, they
cross-dissolve through exactly the same seams at exactly the same scroll
positions, the same copy windows open and close, and every transform is dropped.
The whole story still reads.

---

## 7b. The spacer is sized once, at mount

`layout()` writes the spacer height as `(total + 1) * innerHeight`. If
`innerHeight` reports 0 at that moment the spacer is set to **0px**, the page
has no scroll track, and the flight never advances.

It fails silently and it looks like success. The engine mounted, every leg
registered, the clips fetched and decoded, `sc-has-clip` is on the segments, and
there is nothing in the console. The page is simply a still image that cannot be
scrolled. Embedded preview panes and some early loads do exactly this.

Do not fix it in the engine. One resize makes it re-measure correctly, so send
one from the page once the window and the fonts have settled:

```js
function relayout() { dispatchEvent(new Event('resize')); }
addEventListener('load', relayout);
if (document.fonts && document.fonts.ready) document.fonts.ready.then(relayout);
```

The `fonts.ready` half earns its place independently: a webfont swapping in
changes the measured height of every copy block, and anything the page sized
against those blocks (a scrim plate, a rail) is wrong until it re-measures.

Check it with one line, and check it before blaming anything else:

```js
document.documentElement.scrollHeight   // must be ~(sum of weights + 1) * innerHeight
```

## 7c. Pace: one speed, and slower than you think

Two separate faults get described as "it doesn't feel smooth", and only one of
them is smoothing.

**Inconsistent pace is the worse one.** Leg weight divided by clip length is how
fast the world moves under the reader's hand. If that number varies from leg to
leg, the world surges and drags for no reason the reader can see, and it reads
as a fault in the page rather than as pacing. Give every leg with the same clip
length the **same weight**, and give a longer clip a proportional one. On
`orrery` that number varied by 36% across ten legs on the first cut, and the
owner's word for it was "not smooth". Evened to a 6% spread, the same footage
reads as one continuous move.

```
rate = weight / clip_seconds        // hold this within a few percent everywhere
```

**Then slow it down.** The instinct is to spend as little scroll as possible. A
fly-through wants the opposite: the reader is steering a camera, and a camera
that answers too eagerly feels twitchy. **0.21 to 0.22vh per second of film is a
good floor for a world you fly through.** 0.14 to 0.19, which is what the per-8s
line yields, is noticeably fast.

That line is a **dead-scroll guardrail, not a taste ceiling.** Exceeding it is
fine and often correct; exceeding it without checking is not. The harness
defines dead scroll mechanically and will tell you. At 0.216vh/s a 0.12vh sample
gap still advances the clip by half a second, nowhere near dead.

**Damp the playhead and widen the joins.** `data-sc-lerp` defaults to 0.18;
**0.12 is the better default for a worldflight**, because a flight has more legs
mounted and more seams than an act page, and the extra damping is what actually
removes wheel-event judder. Widen `data-sc-seam` from 0.12 to ~0.16 for the same
reason: a longer crossfade band gives each join more room to disappear in.

Changing weights moves every leg boundary, so **every `data-sc-window` has to be
recomputed** against the new track and then re-checked on screen. A copy window
is tuned to a frame of film, not to a number.

## 8. Hard rules

| Rule | Why |
|---|---|
| **Nothing in document flow but the spacer.** | The moment a real block scrolls past the fixed stage, the page has a seam and the mode is pointless. If you want a section, you want act mode. |
| **Copy translate ≤ 4vh across a window.** | Larger reads as a second page scrolling at a different speed. |
| **The lerp is never disabled** except under reduced motion. | A 1:1 playhead reproduces every gap in the wheel event stream as a stutter. |
| **One pace for the whole flight**, and slower than feels necessary. | Weight divided by clip length must match across legs, or the world surges and drags. ~1.5vh per 8s is the dead-scroll guardrail, not the target. See section 7c. |
| **Every clip stays mounted. Never swap a `src`.** | A src swap is a black frame. |
| **Seam frames come from the encoded mp4.** | The encode changes the pixels. |
| **One accent, one scrim shape, copy anchored off the bright centre.** | Verified by the contrast pass, which grades copy blocks exactly like cues, at the worst frame each line is ever shown on. |

---

## 9. Verifying

`shoot.mjs` detects `[data-sc-mode="worldflight"]` and switches modes. It samples
across the spacer track at the same density it samples acts, plus four extra
positions across every seam, and it waits for the lerp to settle before each
shot (a screenshot taken mid-lerp is a frame the page never actually holds, and
it makes the run unrepeatable).

It reports:

- **dead scroll**, defined here as no leg advancing its `currentTime`, no
  crossfade progress, and no copy-window opacity change between two samples more
  than 0.12vh apart. Skipped under reduced motion, where each leg legitimately
  holds one still frame.
- **legs that never reach full opacity**: a weight or a seam that is wrong: the
  reader is shown a permanent dissolve and never the leg itself.
- **legs stuck on poster**: a clip that never loaded or never decoded. It passes
  every other check, because a poster looks exactly like a paused film.
- **contrast** on visible copy blocks, through the same direction-aware
  compositing path as cues.

```bash
node scripts/serve.mjs --root builds/<name> --port 45XX
node scripts/shoot.mjs --url http://localhost:45XX --out lab/<name>-shots --per-act 8
node scripts/shoot.mjs --url http://localhost:45XX --out lab/<name>-reduced --reduced-motion
```

The mechanical assertions ship with the skill as
`scripts/worldflight-assert.mjs` and run against **any** worldflight page, not a
special rig: spacer height, fixed stage, nothing in document flow, lerp
convergence and non-overshoot, seam monotonicity, the copy transform cap, and
the reduced-motion contract.

```bash
node <skill>/scripts/worldflight-assert.mjs --url http://localhost:45XX
```

Run it against your own build before the contact sheet. It answers "does the
mode actually hold" in a way a screenshot cannot.
