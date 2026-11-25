# Verify

Look at the figure the way a demanding reviewer would, before anyone else does. Rendering it is cheap, and guessing is expensive.

## Capture
`scripts/capture.mjs` starts its own headless Chrome (set `CHROME_PATH` if it can't find one), loads a file or URL, and clips to `.iso-plate` by default. It also prints the readout, any console errors or exceptions, and whether the page scrolls sideways.

```sh
node scripts/capture.mjs page.html --out shots/desk.png                         # desktop 1440, dpr 2
node scripts/capture.mjs page.html --out shots/phone.png --width 390            # phone
node scripts/capture.mjs page.html --out shots/hover.png --hover 0.7,0.5        # pointer at 70% across
node scripts/capture.mjs page.html --out shots/tour.png --times 1200,2400,3600  # motion: extra shots over time
node scripts/capture.mjs page.html --out shots/keys.png --key ArrowRight,ArrowRight
node scripts/capture.mjs page.html --out shots/joint.png --zoom 200,80,220,170  # close-up at 4x, in card px
node scripts/capture.mjs page.html --out shots/still.png --reduced              # reduced motion
node scripts/capture.mjs http://localhost:3000/docs --select "figure:nth-of-type(3) .iso-plate"   # one figure on a dev-server page
```
`--select` takes any CSS selector; by default it clips to the first `.iso-plate`. Each `--times` delay counts from the first shot, so `1200,2400,3600` shoots at about 1.2, 2.4 and 3.6 s after it (a capture takes 60–200 ms, so delays closer together than that run late), saved as `name-<ms>.png`. `--key` sends the same key codes as `drive.mjs` (`7` as `Digit7`, `.` as `Period`; write a comma as `Comma`, since the list is comma separated). The other options are `--height` (1000), `--scale` (2), `--wait` (2500 ms before the first shot), `--zoom-scale` (4) and `--theme dark|light`. Read every PNG you make: open it and look.

Both scripts start Chrome with WebGL on (`--enable-unsafe-swiftshader --ignore-gpu-blocklist`), so shaders show in `capture.mjs` stills too. Use `drive.mjs` for anything that has to be timed or slowed down.

## Drive
`scripts/drive.mjs` runs a scripted session in headless Chrome with WebGL on. Use it for sequences: hover, then click, then type, then wait for a phase. Use it also for slowed-down contact sheets that catch pops, and for the no-WebGL fallback. It prints the readout after each shot, then any console errors and exceptions.

```sh
node scripts/drive.mjs page.html '[["wait",1500],["move",".rt-stage",0.7,0.4],["wait",600],["shot","shots/hover.png"]]'
node scripts/drive.mjs page.html --actions steps.json --height 1300
node scripts/drive.mjs page.html --actions steps.json --width 390
node scripts/drive.mjs page-light.html --actions steps.json --reduced
node scripts/drive.mjs page.html --actions steps.json --no-webgl
node scripts/drive.mjs page-light.html --actions steps.json --preload slow.js
```
Options are `--width` (1440), `--height` (1100), `--scale` (1), `--reduced`, `--theme dark|light`, `--no-webgl` (every `getContext("webgl")` or `"webgl2"` returns `null`, to check the SVG fallback), `--preload file.js` (a script that runs before the page's own scripts: counters, stubs, hooks, or `window.__driveSpeed=0.04` to run slowly from the first frame) and `--actions file.json`. Selectors default to the first `.iso-plate`, and fractions run 0..1 across the element.

`--theme` (in both scripts) only emulates `prefers-color-scheme`. Kit pages choose their theme when they are built (`data-theme` on `<body>`) and ignore that media query, so to check the light theme, build with `--light` and drive the `-light.html` file.

| Action | What it does |
|---|---|
| `["scroll", sel]` | Scrolls the element into the middle of the viewport. |
| `["wait", ms]` | Waits. |
| `["move", sel, fx, fy]` | Glides the mouse there in 10 steps, firing `pointermove`. |
| `["click", sel, fx?, fy?]` | Clicks, at the centre by default. |
| `["drag", sel, fx0, fy0, fx1, fy1, steps?]` | Presses at one point, glides to the other in `steps` (20) moves a frame apart, releases. These are real input events (pointer id 1), so a handler that calls `setPointerCapture` works. Test drag-to-turn with it: `PointerEvent`s dispatched from `eval` have no active pointer, and `setPointerCapture` throws `NotFoundError` and aborts the handler. |
| `["key", key, sel?, code?]` | Focuses `sel` (default `[tabindex="0"]`) and presses a key: `"i"`, `"Escape"`, `"ArrowRight"`, `" "`, `"7"`. Digits send `DigitN`, letters `KeyX` and punctuation its proper code (`Space`, `Minus`, `Comma`…). Pass `code` for keys that share a name: `["key", "Shift", null, "ShiftRight"]`, `["key", "Enter", null, "NumpadEnter"]`. |
| `["slow", factor]` | Runs the page's `requestAnimationFrame` timestamps and `performance.now` at `factor` × real time: `["slow", 0.05]` is 20× slow motion, so a sheet catches a 100 ms event in many frames. `["slow", 1]` restores real time. |
| `["until", js, timeoutMs?]` | Polls a JS expression every 25 ms until it is truthy (default 10 s), and prints a line if it times out. |
| `["eval", js]` | Runs JS in the page and prints the result. Promises are awaited. |
| `["shot", out, sel?, scale?, [x,y,w,h]?]` | Screenshots an element, or a region of it in CSS px. |
| `["sheet", out, every, count, sel?, scale?, [x,y,w,h]?, columns?]` | Takes `count` frames `every` ms apart and tiles them into one numbered PNG. |
| `["turn", value, group?]` | 3D figures: sets `window.__isoTurn` (a number sets the first turn group, `group` names another, an object sets several) and waits two frames. |
| `["orbit", out, from, to, step, sel?, scale?, [x,y,w,h]?, columns?]` | 3D figures: a contact sheet across angles of the first turn group, each frame labelled. `["orbit", "swap.png", 129.7, 130.2, 0.1, null, 6, [x, y, w, h]]` is a 6× close-up of one order change. |
| `["lines", out?, sel?, { tolerance }?]` | The line-end check below, in whatever pose the actions before it left the figure (apart, mid-stroke). `out` writes the card with every floating end ringed. A failure makes the run exit 1. |

Traps:
- **Shots are correct after `scroll`**, so you can scroll to a figure lower on a page. A viewport tall enough for the whole card (`--height 1300`) is still the simplest: the figure is visible at scroll 0, and its `IntersectionObserver` starts it.
- **Frame spacing in a sheet is `every` plus the capture time** (60–200 ms, more on heavy WebGL pages, where capture also stalls animation frames). For transitions faster than that, put `["slow", 0.05]` before the sheet and `["slow", 1]` after it. `wait`, `every` and the page's `setTimeout` timers stay in real time, so at 0.05 a sheet every 100 ms is 5 ms of page time apart, and an idle tour's resume timer still fires on the wall clock: a key press holds it off for only about 3.6 s of real time, so press a harmless key again before each slowed sheet. An event that starts on load needs the page slow from its first frame: `--preload` a file with `window.__driveSpeed=0.04`.
- **Slow mode doesn't slow timers.** `setTimeout` runs on the wall clock, so during a long slowed capture an idle tour's resume timer (the Raptor's 3.6 s `quietTour`) can restart the tour and close the engine mid-sheet. Before every slowed sheet stop the tour and wait for rest: preload a stub for that timer (`const st = window.setTimeout; window.setTimeout = (f, d, ...a) => (d === 3600 ? 0 : st(f, d, ...a));`), press `Escape`, `["until", "document.querySelector('[data-readout]').textContent.includes('cold')"]`, and only then the key that starts the transition. A sheet labelled "opening" that starts from a tour that has already opened films a close.
- **Wait for a phase, don't guess.** `["until", "document.querySelector('[data-readout]').textContent.includes('main stage')", 8000]` returns as soon as the readout reaches the phase.
- **A `null` selector is the default.** In every action a missing or `null` selector means the first `.iso-plate` (for `key`, the focusable `[tabindex="0"]`), so `["sheet", "s.png", 200, 16, null, 0.5]` films the card.
- **Long eval strings break inline JSON quoting.** Put anything with `eval` or `until` in `--actions steps.json`.
- **Pages inline their live code as `<script type="module">`,** so top-level variables aren't visible to `eval`. Read state through the DOM instead: readout text, `aria-valuetext`, `data-*` attributes. To count frames or read a value per frame, install a hook with `--preload` (it runs before the page, so it can wrap `requestAnimationFrame` or `HTMLCanvasElement.prototype.getContext`).

## Line ends
The audit sees recorded shapes, never line markup, so it can't see a seam, rule, rib or tick that stops in the middle of a face. `scripts/lines.mjs` checks the drawing itself, in headless Chrome:

```sh
node scripts/lines.mjs page.html --out shots/lines.png
node scripts/lines.mjs page-light.html --out shots/lines-light.png --reduced
node scripts/drive.mjs page.html '[["key"," "],["wait",2000],["lines","shots/lines-apart.png"]]'
```
```
line ends: 3 of 1148 ends on 295 lines float
  iso-line faint in it k -0.080  end  7.25 px from anything   at card 507.4,407.2   length 75.7   d M392.1 381.7L507.4 407.2…
```
Every open subpath of every `.iso-line` in the card (seams, grooves, ribs from `fadedSvg`, rule lines, ticks, needles, pointers, wires) must end within 1 viewBox px of another stroke, a dot or an outline, or under a fill painted after it. A short straight mark (up to 7 px: a tick) needs one end rooted within 2 px; both ends floating fails. The kit's own outlines, creases, bevels and tube edges aren't checked, and neither are ends that a `fadeLineSvg` ramp fades out. `lineSvg(d, { free: true })` (or `"start"`, `"end"`) marks an end that is meant to stop: rule lines that stand for printed text, a needle's tail, a pointer's tip. Use it for nothing else. It prints each floating end in card px, ready for `capture.mjs --zoom`, with the start of its `d` so you can find it in the build; `--out` rings them on a 2× shot of the card. Run it on both themes and in every pose the figure rests in, and fix every line until it prints 0:

| You see | Fix |
|---|---|
| A seam or weld line on a box inset from the corners | `k.sideSeam(plan, z, P, steps)` as the slab's `inner`: it wraps the corners and ends on the silhouette (`craft.md`, section 3). |
| A stiffener or panel line on one face, inset at both ends | Run it edge to edge: from the face's bottom to its top, or from end to end. |
| Ribs that stop short of their seams | `ribsOf(…, { seams: true })`, or `s0`, `s1` on the band's ends. |
| Ticks floating on a bare plate or hanging below an edge | Put them on a ruler strip, rooted on its edge (`topTicks` `edge`, `sideTicks` `top` on the face's top edge). |
| Vent or grille lines | Closed slots: a thin rounded outline, not a line. |
| A line that runs under a part and stops just past it | Draw it before the part, full length, and let the part hide it. |
| Rule lines on a label plate | `free: true`. They stand for text. |

## Audit
Screenshots catch what you think to look at. A figure with pipes, cables and round parts also needs the geometry checked, because a cable through a duct or a part drawn over the thing in front of it is invisible at 1× and obvious at 4×. Wire `kit/audit.mjs` into the build (`kit.md`, section 12): helpers put into one `A.recorder(P, { order, spring })`, the build calls `A.settle(R, P)` and ends with `A.auditOrExit(R, P)`, then run:

```sh
node examples/raptor-engine/build.mjs --audit
```
```
clearance: 0 problems
clearance apart: 0 problems
terminals: 0 ends not on a mount
loose ends: 0
screen folds: 0
support: 0 unsupported solids · 0 hanging tubes · 0 sunk pairs
depth order: 0 problems
through in mid-flight: 0 pairs
self-overlap: 0 chunks whose outline crosses itself on screen
wobble: 0 smoothed profiles whose slope turns between its ends
saddles off their host apart: 0
stand-in shapes: 0 items whose recorded shape covers too little of what it draws
crowding: 0 pairs where a bend or end touches another line on screen
free ends apart (marked free, not failing): 35
audit · 80 tubes · 267 solids · 478 items · 1119 depth states · 1707702 overlaps · 4292 order rules · 151.9 s
audit passed
```
It prints `audit passed` or `audit failed: n problems` and exits non-zero on any failure. Every line counts, including what happens in mid-flight and `items with no shape`, except two lists that are there to be read: `in passing` and `free ends apart`. Depth order and pass-through (tube against everything, and solid against solid) are both checked on the same replayed frames: every frame of the springs opening and closing until they settle, every frame of a reversal 8 ms either side of each stagger boundary (`reversalsOf`: 0.027, 0.043, 0.062, 0.078 … 0.393 s on the Raptor's 35 ms stagger), and the prefix states (`prefixStates`). Depth uses every fourth of those frames. How to read and fix each line:

| Line | Meaning | Fix |
|---|---|---|
| `through A × B  clearance -2.4` | Tube A intersects B over that arc-length span. | Reroute A round B with about 1.5 radii to spare, or move B. Only the shape A's end rim sits on is excused, and only within a radius or so of that end. |
| `close A × B  clearance 0.2 < 0.6` | A passes B with less than half its radius to spare. | Nudge the route; widen harness lane spacing. |
| `clearance apart: through share 1 …` | In the apart state a tube passes through something. | Change that assembly's explode vector. |
| `terminals: A end 1.40 from B (rim -1.1…1.3)` | A's end, or the rim of its end flange, doesn't sit on its mount: it floats, it is tilted, or the flange is wider than the boss under it. | End the route exactly on the surface, along the surface normal; make the boss at least as wide as the flange; on a curved body seat the flange on a saddle pad. A cable never ends on a clamp. |
| `loose ends: share 1 A end touches nothing that moves with it` | Apart, the end hangs in the air: nothing of another owner moves with it (its own flange or shell doesn't count). | Give it a mount that moves with it, or, if it is meant to come off (a flanged duct, a connector on another assembly), mark it `free` so it is listed under "free ends" as a choice. |
| `self-overlap: A:3 its outline folds back on screen` | One chunk's outline crosses itself: the line turns on a screen radius under its own half-width (a leg pointing at the viewer that turns near a port). | Leave the port through an elbow ball (the Raptor's `elbow`), give the turn more straight leg, or bend in a plane that faces the viewer (the Raptor's `sideOn`, `craft.md`, section 8). |
| `wobble: A dr/ds turns at s = …` | A profile smoothed from typed knots has a slope that plateaus and reverses, so its tone bands staircase. | Give it an analytic `r(s)` and pass `smooth: { slope }`. |
| `saddles off their host apart: A (piece) is cut to B (host piece)` | A saddle pad flies off with another assembly, showing its cut base. | Give the pad to the host's assembly; the moving pipe leaves its flat face. |
| `solid  state  A × B  sunk 1.81 deep` | Two solids of different assemblies pass through each other in that frame. | As for a tube: fix the order or the vectors, or open the gap under an overhang. |
| `screen folds: A turns 150° on screen for 78° in 3D` | The route doubles back on screen and reads as a cusp. | Route the leg before the turn sideways on screen. |
| `screen folds: A turns 110° on screen round a 0.4 px bend` | A visible bend pinches to a point: its fillet is seen edge-on, so the two legs meet in a V. | Turn inside a fitting (a ball tee, the line leaving at 45°), bend in a plane that faces the viewer, or move the port so its normal lies across the screen. |
| `through in mid-flight: apart 0.233 s A × B clearance -1.6` | In that replayed frame A passes through B. | Fix the order or the vectors (`motion.md`, "Order the stagger"): the part outside leaves first, a vector that points into another assembly's face needs that assembly to leave earlier, threaded parts share a spring. |
| `crowding: A bend × B outlines overlap 0.05 px` | A's bend or end lands on B's outline on screen and reads as a joint or a line threaded through B. | Reroute so the meeting is a clear crossing of two straights, or leave a visible gap. Excuse a pair (`allow`) only after looking at it at 8×. |
| `crowding: A end × B (rod)` | A's end or bend lands on the outline of a slender solid: a tie rod, piston rod, post, rail or handle (any solid longer than 4× its radius or half-width). It reads as the rod plugged into A's fitting. | Approach the port across the rod, or turn the elbow so the last leg lies across the screen. |
| `stand-in shapes: A its shapes cover 4% of its drawing` | The shapes recorded for A cover much less of the screen than A draws, so the audit checks a sliver and says nothing about the rest. | Record the true shape of what you draw (`A.slide` for a rod that slides out of a gland). Excuse an item (`cover: [name]`) only when its drawing really reaches past its body, and only after looking at it. |
| `crowding: C fitting × B` | A clamp lands within 2 px of another line and reads as that line growing out of it. | Change the explode vector or the clamp's angle. |
| `support: loose A` / `hanging A` / `sunk A × B 1.2 deep` | A solid, or a rigid tube, touches nothing that leads to the stand, or is buried in B. | Add the bracket, stud or plate that holds it; for a sunk part fix the contact (rim on surface, pad cut by its host marked `seated`; `seated` only excuses the host in its `cut`). |
| `order  A is in front of B but drawn first` | A's surface is nearer wherever they overlap, but B paints over it, at rest or in a mid-flight frame. | `orderKeys` normally fixes these; one left over means a rule was dropped from a loop, so look for the interlocked pair in that loop. |
| `cycle  A … (n the other way)` | A and B are each in front of the other somewhere: they interpenetrate on screen. | Change the geometry: gap the pipe at its collar, split the band at the ring, end the rod on the ball, make the volute tangent, put the valve on a straight, keep a pad off its host's silhouette. |
| `in passing` | Interlocks, or can't be ordered by any `k + lift · share`, in a uniform state only (all assemblies at one share, which the page never shows). | Not a failure. The same thing in a replayed frame is a `depth order` problem. |
| `items with no shape` | Something is drawn without a recorded shape, so nothing checks it. | Record it in the helper that draws it. |

A zero audit is necessary, not sufficient: it can't see a seam, a staircase or a texture that stops, and it only checks what you recorded. A stand-in shape that silences it is a bug, which is why it fails one. Run the line check, then look.

## Orbit (3D figures)
A figure with turning groups (`references/3d.md`) adds the orbit audit and the browser checks. Run them on the light page and the dark page.
```sh
node build.mjs --light --audit > audit.txt 2>&1 &   # the full run, in the background: ends "turn audit passed"
node build.mjs --light --audit --quick    # ≈ 10× fewer states while iterating: ends "turn audit quick"
node build.mjs --light --audit --quick --only=order,continuity   # a subset: the rest print "skipped", ends "turn audit partial"
node build.mjs --light --verify           # a page with window.__isoVerify (recorded shapes) for --order
node ../../scripts/turn-check.mjs build.mjs --fidelity
node ../../scripts/turn-check.mjs page-verify-light.html --order --step 1
node ../../scripts/turn-check.mjs page-verify-light.html --order --states clip.json     # hinges and clips: --step turns the first turn group only
node ../../scripts/turn-check.mjs page-light.html --pops --no-webgl --states tilt-path.json   # one hinge as a continuous path at 0.02°
node ../../scripts/turn-check.mjs page-light.html --pops --no-webgl --step 1 --out shots/pops
node ../../scripts/turn-check.mjs page-light.html --lines
node ../../scripts/turn-check.mjs page-light.html --perf --width 1440
node ../../scripts/turn-check.mjs page-light.html --perf --width 390 --throttle 4
node ../../scripts/drive.mjs page-light.html '[["orbit", "shots/orbit.png", 0, 345, 15]]'
```
**Time.** A 28-part figure with one turn, one hinge and one clip took 85 s quick and 17 minutes full, past a 10-minute shell timeout: run the full audit in the background. Quick runs undersample (`continuity` checks 100 halving candidates instead of 800, and the order scan a tenth of the states), so they guide, and only the full run's `turn audit passed` counts. A run with `--only` ends `turn audit partial`, never `passed`.

**States.** The audit sweeps each turn group at 0.3° (order) and 0.05° (continuity), plus 2000 random states and every order-change event ± 0.001°. Motion clips (`clips: [{ name, at(t), from, to, frame }]`) are walked frame by frame: a gesture, a grasp, a break-apart. Free and hinge groups are proved only on the states you give, so give every clip, at several turn angles.

**Lines.** `separation`, `unseparated`, `cycles`, `forced`, `swaps`, `layers`, `rigid`, `swept`, `posed`, `folds`, `self`, `crowding`, `endon`, `coverage`, `depth`, `continuity`. `3d.md`, section 13, has each line's failure and fix. A planted fault fails each line: a sunk pad (`separation`), a pinwheel of three tilted sticks (`cycles`), a fibre bent in a vertical plane (`folds`), a tube at 30° (`endon`), a rib row that stops (`continuity`).

**Browser checks** (each runs its control first; a control that doesn't fail means the check is broken):
- `--order`: the recorded shapes in a WebGL z-buffer against the DOM's fills: 0 live pairs with 4+ interior pixels at every angle and clip state.
- `--pops`: first difference at every degree (θ → θ + 0.02°) and second difference at 0.1° over a full turn: 0 surviving regions, 0 cores of 6+ px. Read the 1–5 px cores it lists.
- `--lines`: 0 floating ends at 24 angles.
- `--perf`: the budgets in `3d.md`, section 15, at 1440 and at 390 with 4× CPU.

**Orbit sheets.** 0–345° at 15°, then 0.1° steps across the three largest order changes (the audit lists them under "order changes"), then every clip slowed (`["slow", 0.05]`). Read them for spider angles (a hand that reads as legs from some side), parts that vanish behind others, and lines that end on another part's outline.

### Checklist for 3D figures
- [ ] `turn audit passed` with the full state set (not `--quick`, not `--only`), clips included.
- [ ] `--order` on `--states` for every hinge and clip, `--pops` on a 0.02° path for every hinge (`3d.md`, section 14).
- [ ] `turn-check` order, pops and lines at 0, fidelity passed, every control failing as designed.
- [ ] Orbit sheets read at 15°; 8× crops of every knuckle, crossing and texture end at 8 angles, light and dark.
- [ ] Materials read as metal on the light page: highlight bands move with the turn, nothing pops at a tone crossfade.
- [ ] Every shader anchored to its group follows it at 0.1° steps; world effects don't turn; ink stops at occluding strokes (8× crop where a cover mask meets a part).
- [ ] Budgets measured at 1440 and 390 × 4, with the half-rate lever if a dense figure needs it.

## Close-ups
The owner found the Raptor's faults on a close-up of the light theme: cables over the main duct, a stepped bell, bolt rows that stopped dead. Shoot that close-up yourself, everywhere, before anyone else does.

1. **Build both themes** (`node build.mjs` and `node build.mjs --light`) and list the junctions: every flange, tee, elbow, ball joint, clamp, valve, bellows, band, saddle pad and crossing, every curved body (bells, throats, domes, cones, valve bulges, balls), every band joint of a body cut with `bandsOf`, and every place a texture ends. On flat parts too: every seam, weld, stiffener and trim line on a box face (does it reach both corners?), every row of ticks or marks on a plate (is it on a ruler strip?), every rib row's two ends along its axis, every textured hose at a chunk joint and at its ferrules, every rod that slides (at rest, mid-stroke and at full stroke), and every window or screen with text or shader content in it.
2. **Shoot each area at 4× in both themes, assembled.** `--zoom x,y,w,h` takes a region in card px and scales it by `--zoom-scale` (4):
   ```sh
   node scripts/capture.mjs page.html --out shots/joint-dark.png --zoom 200,80,220,170
   node scripts/capture.mjs page-light.html --out shots/joint-light.png --zoom 200,80,220,170
   ```
3. **And apart**, if the figure comes apart. Drive it there, wait for rest, then shoot a region at scale 4. The Raptor toggles apart on Space and its readout starts `apart ·` only once every spring has arrived, so `apart.json` is
   ```json
   [["key", " "], ["until", "document.querySelector('[data-readout]').textContent.startsWith('apart')"], ["wait", 400],
    ["shot", "shots/apart-joint.png", null, 4, [200, 80, 220, 170]]]
   ```
   run with `node scripts/drive.mjs page.html --actions apart.json`, then again on `page-light.html`.
4. **Look at each PNG for these faults, in this order:** a line that runs through or into something; a crossing whose nearer line is drawn behind; a bend, end or clamp sitting on another line's outline; a flange standing off its port or sunk into it; tone blocks stepping across a curved surface, scallops or a pale wedge; a band ending on a straight cut; a row of ribs, dots or rule lines stopping in the middle of a surface; a hairline across a tube's shine or shade at a chunk joint; a seam notching the outline; a seam, stiffener or rib that stops short of the edge it should reach; a short mark on a top that touches nothing; a hose texture with straight hard sides, or one that restarts or is cut at a chunk joint; a rod whose outline breaks while it moves; text that crosses its window; shader content that spills past an opening's rim. Compare with the previous round's shot of the same region.
5. **Shaders:** the same regions during the effect, plus contact sheets slowed with `["slow", 0.05]` (`webgl.md`, section 9).

**Detail pass.** Last, sweep at 4× in a grid of crops (the subject and mechanism for a Figure, the whole drawing for a Hero or an Epic; `SKILL.md`, workflow step 1) and look for any face, corner or stretch of base with nothing true on it. Every figure is highly detailed (`SKILL.md`, "Every creation is highly detailed"): add the real parts that belong there (screws, seams, bevels, label plates, scales, the brackets that hold things), never decoration that belongs to no part.

## Light-theme shaders
Light is the shipping theme, and on paper a glow is ink. Every light conversion so far failed review first on something a picture of one frame doesn't show: ink that rose while the light fell, a pale flash before the glow, a salmon stage, a fringe beside a stroke. So measure the ink as well as looking at it (`webgl.md`, section 8).

**Ink per frame.** Save this as `ink.js` and pass it with `--preload`. For every frame the page draws, it logs the mean ink one canvas region lays on white, `255 − lum(rgb + 255 − a)`, read back in the same task as the draw:
```js
const get = HTMLCanvasElement.prototype.getContext;
HTMLCanvasElement.prototype.getContext = function (kind, ...rest) {
  const gl = get.call(this, kind, ...rest);
  if (gl && String(kind).startsWith("webgl") && !this.__gl) {
    this.__gl = gl;
    for (const name of ["drawArrays", "clear"]) {
      const call = gl[name].bind(gl);
      gl[name] = (...args) => { gl.__drew = true; return call(...args); };
    }
  }
  return gl;
};
const raf = window.requestAnimationFrame.bind(window);
window.__ink = [];
window.requestAnimationFrame = (callback) => raf((now) => {
  callback(now);
  const canvas = document.querySelector(window.__inkCanvas || "canvas");
  const gl = canvas && canvas.__gl;
  if (!gl || !gl.__drew) return;
  gl.__drew = false;
  const [x, y, w, h] = window.__inkBox || [0, 0, canvas.width, canvas.height];
  const px = new Uint8Array(w * h * 4);
  gl.readPixels(x, y, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
  let ink = 0;
  for (let i = 0; i < px.length; i += 4) {
    const bare = 255 - px[i + 3];
    ink += 255 - (0.2126 * (px[i] + bare) + 0.7152 * (px[i + 1] + bare) + 0.0722 * (px[i + 2] + bare));
  }
  window.__ink.push(Math.round((ink / (w * h)) * 100) / 100);
});
```
- **Choose what to measure.** Before the transition, set `window.__inkCanvas` (a selector) and `window.__inkBox` (`[x, y, w, h]` in device px of the canvas, measured from its bottom-left; scale CSS px by `canvas.width / canvas.clientWidth`) with an `eval`. Read the rows afterwards with `["eval", "window.__ink.join(' ')"]`.
- **Do it once per region that matters:** a cavity, the core, an area only the fallback covers. Run it slowed, across first light (`--preload` a second file with `window.__driveSpeed=0.04`), every phase change and the fade-out.
- **What to look for.** Ink follows the physical quantity monotonically: up as the light comes in, down as it goes. There is no hump at a hand-off and no dip at first light.
- **What it caught.** The afterburner's cavity peaked at 3600 on the cut before its fix; afterwards it fell monotonically from 1894 to 155. Those rows are in that figure's own units, a sum over its window region rather than this hook's mean. On the ripple tank's light page the hook reads 14.2 at rest and falls as the lamp comes in, because that page's light brightens the paper.

**Colour along the ramp.** For anything that heats or cools, `eval` a pixel read at the centre and the rim at several temperatures (or several temperatures' worth of a slowed run) and write them down. The order must be dull red → orange → yellow → near-white, with the rim cooler than the centre and no salmon or grey stage on the way.

**Cross-sections.** Read one row of ink across a flame, a jet or a stream. It peaks on the axis and falls to translucent edges. An edge-heavy profile is a hollow, outlined flame.

**Strokes at 8–16×.** Crop every place where ink meets an SVG stroke (cut faces, rims, mouths, creases, a canvas edge with a body entering) at 8–16× in light. Look for a pale gap between the ink and the line, ink over the line, or a stroke the ink cleared.

**Both paths, and nested.** The dark build should match its baseline, because every light change is gated on `uLight`. Render a dark plate on a light page and a light plate on its own, each with and without WebGL: a `:not([data-theme="dark"])` rule can reach through the page into a dark plate.

## The review checklist
**It is an object**
- [ ] At a glance it reads as one real, well-made thing, and you can name it.
- [ ] Its mechanism maps to the concept: what moves and what that means can be said in one sentence.
- [ ] It isn't a stack of abstract plates and isn't one plain slab.

**Parts and volume**
- [ ] 30–60 solids for an inline figure; a hero subject that really has more parts can go far past that (the Raptor has about 270). Every part rests on another and has a job.
- [ ] Sides shaded and tops lighter; bevels on the main slabs; a halo under the base and under lifted parts.
- [ ] Small details belong to parts: screws at the corners of plates, grooves, rulers, knurls, label plates.

**Lines**
- [ ] At 4×, every line ends on a face, another line or a terminal dot. Nothing stops in the air.
- [ ] No leader lines, balloons, dimension arrows or floating labels; at most one engraved label.
- [ ] Painter's order is right everywhere: nothing behind shows through, springs and frames are split.
- [ ] **Audit zero.** With pipes, cables or round parts in the figure, `node build.mjs --audit` ends `audit passed`: zero on every failing line, assembled, apart and in every frame of the replayed explode (opening, closing and reversing). No interlock was hidden with a key bias, and every excused pair (`allow`, `crowd`) has been looked at at 8×.
- [ ] No pipe or cable passes through another pipe, cable or solid, and every crossing is drawn in depth order: the nearer line on top, crossing on straights, well apart in depth.
- [ ] No line runs across the subject's face where it could run beside the structure; no bend pinches to a V on screen; no bend, end or clamp lands on another line's outline, assembled or apart.
- [ ] Every flange sits on its port (on a saddle pad if the port is curved); every collar, clamp, band, bellows and valve gaps the pipe it wraps; every valve and bellows sits on a straight; no route doubles back on screen; every end, apart too, ends on a flange, coupling, connector or terminal dot.
- [ ] **Junction close-ups.** Every junction area shot at 4× in both themes, assembled and apart (see "Close-ups"), and each PNG read.
- [ ] **No staircase.** Every curved surface is smooth-shaded at 4×: no stepped tone blocks, no scalloped terraces, no pale wedge, no band that ends on a straight cut, no kink at a band joint; waisted bodies follow their true outline; seams meet the outline without a nick.
- [ ] **Line ends zero.** `node scripts/lines.mjs` prints `line ends: 0` on both themes and in every resting pose; every `free` end is a rule line standing for text, a needle tail or a pointer tip.
- [ ] **Shapes are true.** No stand-in shapes: what is recorded is what is drawn, including rods that slide.
- [ ] **No abrupt texture ends.** Every row of ribs, bolt dots, rule lines or grooves ends on a real edge or fades out; none stops at full strength in the middle of a surface, flat faces included. Seams on boxes wrap their corners, rib rows end on their seams, ticks sit on a ruler strip, and a texture on a hose follows its cross-section and runs through every chunk joint. Pipe stripes run onto their end caps and show no hairline at any chunk joint, also on a strongly curved tapered tube (a volute).

**Tone and colour**
- [ ] Base `lo`, structure `mid`, subject `hi`, exactly one `lit` accent.
- [ ] Monochrome unless colour is the subject.

**Card and words**
- [ ] `Fig n` / title / hint / live readout are in the corners, with legend keys and a caption under the card.
- [ ] Every number is true and comes from the model. The readout changes as things move, and its words and numbers come from the same instant: a phase word from the model next to a number from a lagging gauge reads `relief 85 bar` before the valve has cracked.
- [ ] Text the machine displays fits its window at its widest string and is clipped to it.

**Motion and access**
- [ ] Eased, calm, no snapping; an idle tour; pauses on interaction.
- [ ] Stops offscreen; still under `--reduced`; no console errors (including React key warnings).
- [ ] Keyboard works, the focus ring shows, and `aria-valuetext` reads the state in words.

**Shaders (if the figure has one; `references/webgl.md`)**
- [ ] A contact sheet across every transition (start, each phase change, steady, stop), slowed with `["slow", 0.05]` (or `--preload` with `window.__driveSpeed=0.04` for an effect that starts on load) so every frame of each crossfade shows: no sudden colour switch, no one-frame flash, nothing cut out, and brightness never rises during a fade-out.
- [ ] 2× and 4× crops where the effect meets the SVG: no seams, hard edges, banding, dark collars or double lines; things on a plane register with that plane's lines.
- [ ] The canvas's outer rows and columns read alpha 0 in every phase.
- [ ] Without WebGL (`--no-webgl`) the SVG alone still reads; under reduced motion you get one still, representative frame; the loop stops when idle and offscreen.
- [ ] **Light theme** (`webgl.md`, section 8; "Light-theme shaders" above):
  - Glows read through saturation, with a saturated body, a pale core and a coloured rim, never as dark or brown ink.
  - What the light falls on is tinted, and the cavity that holds the light is a darker physical ground.
  - Hot bodies run in blackbody order with no salmon stage.
  - Smoke and steam are light grey.
  - There are no halos in empty air.
  - Ink per frame follows the light monotonically through first light, every hand-off and the fade-out.
  - Ink stops at each stroke's inner edge at 8–16×.
  - The fallback shows the same inks and ground, and hands over to the shader without a dip.
  - Dark is unchanged, and a dark plate on a light page still works.

**Composition**
- [ ] Centred, filling about 55–65% of the card; it doesn't touch the corner labels.
- [ ] Still readable at 390 px with no sideways scroll, with a part selected and during a burn, not only at rest: the longest readout must not run under the controls.

## Common faults and fixes
| You see | Fix |
|---|---|
| A part shows through something in front of it | It is painted too late. Move it earlier, or split the front part. |
| An ellipse shows across a post where it meets a face | A full `ring` drawn over a cylinder that continues shows its back half. Use `sideArc` (front half only), or let the cylinder's own outline make the edge. |
| A post looks pasted onto the block it passes through | Draw it in two segments, below and above, with the block painted between them. |
| A crease or bevel stops short of a corner | Use `bevel` on `slabOf`/`cylinder`. It already ends on the silhouette. Don't hand-draw edge lines. |
| Flat, dead sides | You passed `flat`, or the solid has height 0. Give it thickness. |
| Too sparse | Add structure, not texture: feet, rails with rulers and end stops, a label plate, screws, a knob, a second stage. |
| Too busy | Drop texture-like repeats and tone grooves down to `faint`; keep one lit accent. |
| The figure is tiny in the card | Fit with a smaller `pad`, or tighten the points passed to `fitProjection`. |
| It jumps when a tab regains focus | Clamp `dt` to 1/30 s. |
| Key warnings only on the big page | See `react.md`: key the roots of element props. |
| A shader colour jumps at a phase change | Give colour its own slow envelope, and drive it from the physical quantity, not from a phase flag. See `webgl.md`, section 7. |
| A straight edge in the glow, often at the canvas border | A term isn't zero at a boundary. Window it to zero before every `if`, bound and canvas edge. |
| A glow runs through the floor or over a column | It is in screen space. Rebuild it in world space with the ray and the ground hit, and give simple parts to the shader as occluders. |
| A dark collar under a glowing rim | The rim is normal-blended over the shader. Move it to an overlay SVG with `mix-blend-mode: screen`. |
| Flat bands in soft glows | Add gated triangular dither after `tonemap`. |
| In light, a glow reads as dark red or brown smoke | It is the dark colour darkened, with alpha set to brightness. Solve the ink from a paper ramp with a saturated body, a pale core and an alpha floor (`webgl.md`, section 8.1). |
| In light, first light goes pale and then dark, or a fade-out greys or gets heavier | Per-term fades, an alpha-only fade, or a hue that reddens at fixed alpha. Fade the finished ink once, dim energy and alpha together, and take the hue at steady brightness (`webgl.md`, section 8.2). |
| In light, a pale fringe or white seam along an edge | The ink stops short of the stroke, or clears a crease nobody strokes. End it at the stroke's inner edge (0.3 vb) and draw the crease in ink (`webgl.md`, section 8.6). |
| A dark plate's fallback vanishes on a light page | A `.iso:not([data-theme="dark"])` rule matched through the page. Put the dark rule last, or key on the plate's own theme (`webgl.md`, section 8.5). |
| A pipe or cable looks pasted on or cuts through a part | Its pieces are in one layer. Depth-sort each `tubePieces` chunk by `mid`, and use shorter pieces where it weaves. If it really passes through the part, reroute it; `--audit` lists every case. |
| Cables or sense lines fan across the subject's face and over a main duct | They were routed point to point. Re-route along the structure: up beside a pipe, round the body in clamped lanes, round the side to a far port (`craft.md`, "Choosing waypoints"). |
| Fixing one crossing's order breaks another, or the apart pose | Keys tuned by hand. Record shapes, let `A.settle` set the keys, and fix whatever it reports as interlocked in the geometry. |
| A collar, band or clamp hides the pipe's near side, or the pipe paints over the collar's face | The pipe runs through it as one chunk. Gap the pipe at the collar (`gaps`), so a piece ends on each face. |
| A ring on a body shows its back half over the body | The body is one long band under the ring. Split it at the ring. |
| A valve or bellows cuts into its pipe at the ends | It sits on a bend. Move it to a straight at least its own length plus both fillet cuts. |
| Stepped tone blocks on a bell, dome or cone | Facet shading. Mark the curved knots smooth (`[s, r, 1]`, shaded smoothly by default) or pass `smooth: true`; cut a body into bands with `G.bandsOf`. |
| Tone bands on a throat, cone or puck stall and lurch in scallops | A smoothed profile of hand-typed knots: its slope plateaus at each knot. Write the profile as an analytic `r(s)` and pass its derivative as `smooth: { slope }` (`wobble` lists them). |
| Hairline seams inside a smooth-shaded part | The tone sheets abut. Use the kit's `solidSvg` with `seamless` paths (smooth lathes and `sphereOf` set it) and `ISO_CSS`. |
| A row of ribs or bolt dots stops in the middle of a surface | A hard `least` cut-off. Use `fade: [lo, hi]` on `ribsOf` or `dotsOf`. |
| A pale wedge with straight edges on a smooth-shaded cone or bell | Bare fill between tone pieces: pieces of one tone wound in opposite directions cancel under the nonzero rule, or a band ended without tapering. The kit orients every tone polygon and tapers bands; if you build tone paths yourself, orient them. |
| A narrowing body looks like a straight cone with a pale strip | It was hulled. Let `lathe` pick `fill: "auto"` (or pass `"surface"`). |
| A hairline crosses a tube's shine or shade band | Body and stripe cut on the same line at a chunk joint. `tubePieces` runs the stripes past the body; keep that if you change it. |
| The outline is dashed where seams meet it | Seams drawn after the solid. Pass them as `inner`. |
| A pipe's stripes stop short of its end | An end without a cap. `tubePieces` caps every end by default; don't pass `caps: false` for an end that shows. |
| Parts slide along their mount while the figure comes apart | They move with one assembly's vector on another's spring. Give the item that assembly's `ride`. |
| A long turned body has a kinked outline at band joints | Build it with `G.bandsOf` and draw the sides once with its `sidesOf` (`kit.md`, section 10). |
| A texture line stops in the middle of a face | End it on a real edge, or draw it with `fadeLineSvg(points, { fade })` so it ramps out. `scripts/lines.mjs` lists every one. |
| A seam on a box stops short of its corners | `k.sideSeam(plan, z, P, steps)` as `inner`. |
| A hose braid or corrugation with straight hard sides, or broken at chunk joints | `tubePieces(…, { rings: { pitch, twist, cross: true } })`. |
| A rod's outline notched while it slides | One rod from `G.rodOut`, clipped at the gland, recorded with `A.slide`; never overlapping copies. |
| LCD or counter digits cross the window | Fit the type to the window's height and widest string, and clip it. |
| Stray triangles of outline across a valve or bulge | An old kit joined a broken silhouette; the kit now falls back to the hull when a side's silhouette has gaps. Rebuild with the current `lathe.mjs`. |
