# Motion and interaction

A figure is alive when its mechanism moves the way the real thing would: a carriage glides and settles, a spring overshoots once, a needle swings to the new reading. Motion should be calm enough to follow and true to the model.

## Principles
- **Time-based.** Take `dt` from `requestAnimationFrame` timestamps and clamp it to 1/30 s so a background tab doesn't jump. Never step a fixed amount per frame.
- **Ease toward targets.** Pointer and keyboard set a *target*, and the part follows with exponential easing (`x += (target − x)·(1 − e^(−dt/τ))`, τ = 0.1–0.3 s) or a spring. Parts never snap to the cursor.
- **Real springs where the subject is springy.** Use stiffness and damping from the real model, sub-stepped about 240 times a second for stability. Snap to the target once both error and velocity fall below 5e-4.
- **One loop per figure.** A single `tick(dt)` updates the model, then writes the DOM through refs: `setAttribute("transform" | "d")`, `textContent`, `dataset`. Don't re-render React per frame.
- **Stop when settled or offscreen.** Use an `IntersectionObserver` with rootMargin `120px 0px`, and read the newest entry (`entries[entries.length - 1]`): when the stage leaves and comes back between two callbacks, one callback delivers both entries oldest first, and reading the first one leaves the figure frozen on screen. Stop the loop once everything has reached its target.
- **Reduced motion.** When `prefers-reduced-motion: reduce` is set, jump to targets with no tour, flow or pulse. The figure must still respond to input.
- **Idle tour.** When nobody is interacting, play a scripted loop after about 1.4 s that shows the point of the figure. Pause it on hover, focus or press, and resume about 3.6 s after the last interaction. The tour drives the same targets the user would.
- **Readout updates live** from the same model, in true units and short: `block 2 · reads 14.0 px`, `120.0 × 32.0 · lean 10.6 px`. Read it from where the parts are, not from the target: the Raptor says `taking apart` or `closing up` while any spring is moving or pending, and only then `apart · 11 assemblies` or `1,630 kg · cold`. Take a phase word and the number beside it from the same instant of the same source: a blind-test bench printed `end of stroke · relief 85 bar` (the word from the model, the number from a gauge needle lagging 0.25 s behind it) although its relief valve cracks at 92, and `idle · 44 bar` just after retracting.

## Pointer
- Map the pointer to world space with the screen vector of an axis. Use `axisVector(P, [1, 0, 0])` for motion along x, or map the figure's width fraction onto the part's travel range. The range mapping is simpler and feels better when the travel is short.
- Hover previews and press commits. On touch there is no hover, so make a tap act and keep `touch-action: pan-y` so the page still scrolls.
- Use pointer capture while dragging, and release it on `pointerup` or `pointercancel`.

## Keyboard and accessibility
- The figure's root is focusable (`tabIndex=0`) with `role="slider"` (or `"group"` plus buttons when there are discrete choices), an `aria-label` that explains the controls, and `aria-valuenow`/`aria-valuetext` in words, such as "Pressed: 129.6 by 34.6 pixels, leaning 10.6 px". Write the value text when the reader's choice changes (a part selected, apart or assembled, firing or cold), not from the loop: screen readers announce every change of a focused slider's value text, so a readout ticking each frame ("Pc 290 bar · 222 tf") became a stream of numbers. Keep the visual readout and the accessible value separate (the Raptor's `say()` writes only the readout, `describe()` the value text); the live region carries the phases.
- Keys: arrows step by a sensible unit (Shift for bigger steps), Home and End go to the extremes, Space or Enter presses or toggles, and Escape releases.
- Show a focus ring: `outline: 1px solid var(--anatomy-muted); outline-offset: 6px`.
- The `<svg>` carries `role="img"` and an `aria-label` describing the object and what it shows.

## Vanilla loop (framework-free)
```js
let frame = 0, last = 0, visible = false, touring = !still.matches, clock = 0;
function tick(now) {
  const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
  last = now;
  const calm = still.matches;
  if (touring) { clock += dt; target = tourAt(clock); }
  x = calm ? target : x + (target - x) * (1 - Math.exp(-dt / 0.28));
  springTo(tip, heightUnder(x), dt, 420, 26, calm);
  draw();                                   // setAttribute on refs, readout.textContent = …
  const settled = !touring && x === target && tip.rate === 0;
  frame = visible && !settled ? requestAnimationFrame(tick) : 0;
}
const run = () => { if (!frame && visible) { last = 0; frame = requestAnimationFrame(tick); } };
new IntersectionObserver((entries) => { visible = entries[entries.length - 1].isIntersecting; if (visible) run(); }, { rootMargin: "120px 0px" }).observe(stage);
```
`examples/dial-indicator/build.mjs` contains a complete version, with the tour, pointer mapping and keyboard stops.

## React loop
Keep static art in props built on the server and move only `<g>` wrappers and the paths that change shape:

```tsx
"use client";
export function RigLive(props: { base: React.ReactNode; carriage: React.ReactNode; dial: React.ReactNode }) {
  const { anchor, say } = useReadout<HTMLDivElement>();
  const car = React.useRef<SVGGElement>(null);
  const needle = React.useRef<SVGPathElement>(null);
  const state = React.useRef({ x: REST, target: REST, tip: { at: 0, rate: 0 } });
  const visible = useInView(anchor);
  const kick = useLoop((dt, calm) => {
    const s = state.current;
    s.x = follow(s.x, s.target, dt, 0.28, calm);
    const settled = spring(s.tip, heightUnder(s.x), dt, 420, 26, calm);
    car.current?.setAttribute("transform", translateAlong(P, [s.x, 0, 0]));
    needle.current?.setAttribute("d", needleAt(s.tip.at));
    say(readoutOf(s));
    return !(settled && s.x === s.target);
  }, visible);
  return (
    <div ref={anchor} tabIndex={0} role="slider" onPointerMove={(e) => { state.current.target = toWorld(e); kick(); }}>
      <IsoFigure width={W} height={H} label={LABEL}>
        {props.base}
        <g ref={car}>{props.carriage}</g>
        {props.dial}
        <path ref={needle} className="iso-line" data-tone="lit" d={NEEDLE_REST} />
      </IsoFigure>
    </div>
  );
}
```
`examples/test-rig/live.tsx` is the full production version, with springs, a tour, press and pull, keyboard handling and aria.

## Shapes that change
When a part changes shape rather than position (a swelling capsule, a compressing spring, a needle turning), rebuild only its paths each frame with the kit, using `slabOf`, `coil` or `lineOnTop`, and write them to its paths: `fill`, `shades[0..3]`, `top`, `crease`, `bevel`, `outline`. One solid costs well under 0.1 ms. Measure it: a frame should stay under about 1 ms of script.
- **Precomputed frames step.** A lever drawn from 41 prebuilt frames over ±15° moved its knob about 0.8 px per frame, visible in slow motion. Rebuild a rotating part each frame, or precompute enough frames that one step moves nothing more than 0.25 px.
- **A rod that slides out of a gland** doesn't change shape: it is one full-length rod that moves, clipped at the gland face by a clip that stays put (`G.rodOut`, `holdStill`, `craft.md`, section 3). Never draw it as several overlapping copies.

## Envelopes for one-shot events
A flash, a click, a pop, a shake or a shock is an event with an age, not a value you set. Drive it with an envelope that has an attack and a decay. `kit/gl.mjs` exports `smooth(a, b, x)` (smoothstep) and `burst(age, rise, fall)`, which is `smooth(0, rise, age)` times `exp(−(age − rise)/fall)` after the peak and 0 before the event:
```js
import { smooth, burst, settle } from "../../kit/gl.mjs";
const flash = calm ? 0 : burst(clock - mainAt, 0.08, 0.13);
```
On a standalone page `glScript()` already inlines all three, so use them directly and don't declare your own: a second `const smooth` in the same module script is a SyntaxError.
- The attack is 40–120 ms, and the smoothstep gives it zero slope at onset. Setting a value to 1 in one frame is a pop.
- Store the event's start time (`mainAt = clock`) and compute every envelope from its age each frame. Two events that overlap then simply add.
- Give each event its own flag (`popped`), so a quick restart can't swallow the next one and can't cut off one that is still fading.
- Under reduced motion every event envelope is 0.

## Critically damped settle
For a value that should arrive without overshoot (a hue, a brightness, a lamp, a reach, a frequency), use the kit's `settle(state, target, dt, seconds, calm)`: a critically damped spring on `state = { x, v }` with time constant `seconds`, `v += (w²·(target − x) − 2w·v)·dt` with `w = 1/seconds`, in four sub-steps. It is smoother than exponential follow, because velocity is continuous too. It snaps to the target once error and velocity are both below 1e-4, jumps there under `calm`, and returns true while still moving, so it can keep the loop alive:
```js
const moving = settle(freq, state.tf, dt, 0.625, calm);
```
The ripple tank runs its frequency (0.625 s) and dipper spacing (0.4 s) through `settle`, so a key press or a pointer move never jumps the wavelength.
- **Separate what a phase changes.** When one physical quantity (say chamber pressure) drives colour, shape and brightness, a fast spring on that quantity makes all three jump at once. Give colour its own slower `settle` (0.3–1 s), and compute brightness so it moves one way through the change.
- **A fade-out only gets darker.** When a stop begins, freeze the brightness at its current level and scale it only by quantities that fall.
- **Changing a spring changes its neighbours.** Damping one spring critically slows every phase that relied on its old overshoot, so recheck all of them.

## Exploded views
Taking a machine apart on hover is a strong way to show its parts (`examples/raptor-engine/`):
- **Build every piece at its assembled position**, tagged with an assembly name, and give each assembly a world move vector (`PIECES.nozzle.move = [0, 0, -120]`). Moves follow the real assembly: the nozzle drops furthest, side units slide out radially (`out(deg, r, z)`), and the stand doesn't move.
- **One spring per assembly**, with value 0 for assembled and 1 for apart. The Raptor uses stiffness 64 and drag 14.2, just under critical, so the parts settle with a hint of weight, and sub-steps 4 times per frame.
- **Stagger the springs.** Opening, each assembly's target changes 35 ms after the previous one, in an order that lets nothing pass through anything (the Raptor: `nozzle, downcomer, controller, plumbing, lox, ftp, hotgas, chamber, orpb, otp, gimbal, stand`, see "Order the stagger" below). Closing uses the reverse order, so the machine folds back together from the inside out. Keep pending targets as `{ value, at }` and apply them when `clock >= at`.
- **On a reversal, reschedule only what changes.** When the reader reverses mid-way (a quick hover, Space, a tap), cancel the pending target of every assembly that hasn't left yet and stagger only the ones whose target changes, by their index in that shorter list. The last one out is then the first one back, at once. Rescheduling the whole list left the early assemblies flying out toward apart for 0.3 s while later ones sat at home, and the explode only worked because the next assembly moved out of the way 35 ms later: a reversal in that window drove the hot-gas duct into a methane pump that hadn't left. With this rule, shares along the stagger stay in order in every frame (earlier assemblies always lead), which is what the prefix rule below relies on.
- **Move with transforms.** Each item's screen offset is its move vector, projected and multiplied by the spring value: `translate(tx ty)`. Remove the attribute at 0, so an assembled figure is exactly the static art.
- **Keep the lines landing.** Parts that join two assemblies (an actuator between the frame and a pump lug, or a cable between a box and a plug) belong to the assembly they physically stay with, and they still end on something when apart. Cables get terminal dots at both ends.
- **Selection.** Hover, the keyboard arrows and a parts list all select one assembly. The selected parts become the one `lit` accent: CSS on `.it[data-on]` moves `hi` to `lit` and `mid` to `hi`, and switches tops and sides to the lit fills. Every other part steps down one tone (`hi` to `mid`, `mid` to `lo`) through CSS on `[data-focus]`, and the readout says what the selected part does.

## Depth re-sorting of moving parts
Parts that move toward or away from the viewer change painter's order as they move. When they do, sort every frame:
- Give every item a static key `k = dot3(ref, viewOf(P)) + bias`, where `ref` is its centre and `bias` settles ties between touching parts. Also give it a lift: `lift = dot3(move, viewOf(P))`.
- Each frame, its key is `k + lift × spring`. Sort the indices by that key, and **re-append the DOM nodes only when the order actually changed** (one `DocumentFragment`). Most frames change nothing.
- Store the key and the move vector on the element (`data-k`, `data-m`), so the live script needs no geometry.
- Tubes are sorted per piece (`tubePieces` → `piece.mid`), so a pipe can pass in front of one part and behind another.
- **Settle the static keys from the motion, not just the rest poses.** Each item's key is `k + lift × spring`, but with staggered springs every assembly has its own share in any frame, so two items that are ordered right assembled and apart can still be drawn the wrong way round in between: in the first Raptor a pressurisation line hid behind an actuator column it was in front of for 40% of every explode. The Raptor measures true depth for every overlapping pair at 0, 0.1 … 1, at every fourth frame of its own springs replayed with the live stagger until they settle, opening, closing and reversed either side of every stagger boundary, and in the prefix states (`A.explodeStates(STAGGER, SPRING).depth`, which is `[...A.uniformStates(10), ...A.springStates(STAGGER, { ...SPRING, stride: 4, reversals: A.reversalsOf(STAGGER, SPRING.stagger) }), ...A.prefixStates(STAGGER)]`, 1,119 states, and the default of `A.settle(R, P)` for a recorder made with `{ order: STAGGER, spring: SPRING }`), and `orderKeys` solves all of them at once (each rule uses that frame's shares). Opening and closing alone left wrong pairs in the frames a quick hover makes: a purge flange that should sit in front of the chamber band vanished behind it on a reversal at 0.20 s. The audit fails on any wrong pair at rest and in any replayed frame; only the uniform states, which nobody sees, may hold an interlock.
- **One source for the springs.** Keep the stagger order and the spring constants (`SPRING = { stagger, stiffness, damping, substeps }`) in the build and pass them to the page in its data, so the audit replays exactly what the viewer sees.
- **Move a part with what it is mounted on (`ride`).** A sensor bolted to the chamber highlights with the plumbing but rides the chamber's spring. Each item records `ride` (the assembly whose spring moves it, found from its move vector) next to `piece` (the assembly it highlights with), the page reads `data-ride`, and `place()` uses `piece[node.ride].x` for both the translate and the sort. An item that borrows another assembly's vector but keeps its own spring slides along its mount and sinks into it mid-flight.
- **The prefix rule.** Every assembly's path must be clear of every later-leaving assembly parked at home, with the earlier ones out: that is the frame a reversal at a stagger boundary freezes. `A.prefixStates(STAGGER)` poses exactly those (the first `k − 1` out, the `k`-th at 0.25 … 1, the rest home) and the audit checks them with the replayed frames. On the Raptor it moved the hot-gas manifold after the methane pump it bolts to (its vector points into the pump), and the controller before the plumbing (the oxygen-to-preburner line is enclosed by the controller's sense lines, so it can only leave after them).
- **Order the stagger so nothing has to pass through.** The springs overlap: with 35 ms between starts and about a second to settle, every assembly is moving at once, and relative motion is a blend of every vector. The first Raptor explode passed 66 pairs through each other in mid-flight. The rules that took it to zero, checked every frame and on reversals by the audit:
  - **Whoever is outside leaves first.** On each side, the part that everything else is tucked behind or under moves first: the Raptor's nozzle drops before anything (everything sits on it), and the downcomer, which runs outside every sense line and harness that reaches the methane pump, goes second. Closing reverses it, so the outer part is also the last one home.
  - **When a part's explode vector points into a face of another assembly, that assembly comes earlier in the order.** The downcomer drops 84 onto the nozzle's port tee; the nozzle drops 120 and leaves 35 ms earlier, so the port is always below the flange, opening and closing.
  - **Point every vector away from every face the part touches while the other side is still in place**, not only the two it bolts to: a line threaded between the chamber and the pump must move in the sector both mount normals allow (the Raptor's jacket lines go to −120° with no drop, sliding off both pads), a cable wrapped round a core part may only move in the sector that is outward for its whole wrap (a 90° wrap allows 90°; a wrap over 180° allows nothing, so shorten it or end it on the near side), and a line that would sweep down over a part bolted below it must not drop.
  - **Things threaded through each other share a spring.** Sense lines that rise through the harness ring into a block, and the harness under that block, cannot have separate springs: with 35 ms between them they shear past each other. The Raptor's sense lines and block ride the controller (`BLOCK_MOVE = PIECES.controller.move`, so `ride` resolves to `controller`) while still highlighting as plumbing.
  - **Give each line that is mounted on one part only the move of that part.**
  - **Check the paths a viewer can make**, not only the two from rest: a quick hover reverses the springs halfway (`reversals`), and an assembly that is late on the way back can sweep through a cable that is already home. Reverse at every stagger boundary (`A.reversalsOf`), not at a few round times: the bad windows are about 15 ms wide, and 0.15, 0.3 and 0.5 s missed all of them. Feed those frames to the depth states as well as to the clearance check, and test solids against solids too (`A.solidClearances`): a flange tucked under an overhang on its host met the host's spring tail on the way home, 0.67 deep, with every tube check at zero.

## Turning by hand
For a 3D figure (`references/3d.md`):
- **Drag.** `controller.angleAt(clientX, clientY, zFace, "dial")` is the pointer's angle on the face's plane, unwrapped across turns. On pointer down, store it and the current target. On move, set target = start + Δ, and capture the pointer once the move passes about 1.5°. The group follows with exponential follow (τ = 0.12 s). The stage keeps `touch-action: pan-y`.
- **Keys.** Arrow keys step by one division, Shift steps by ten, Home returns to the index.
- **Coupled groups.** Drive them from one value through `follow(values)` (a wheel at a quarter of the dial, a finger chain from one `lift`), so a tool, a drag and the tour all agree.
- **Gestures and clips are pure functions of time** (`tapAt(t)`, `breakAt(t)`): the build samples them, the audit walks them frame by frame, and the page can play them backwards to reset with no jump.
- **Hook takeover.** Any `window.__isoTurn` call runs `onHold`: stop the tour, and keep values a tool set until the reader acts again.
- **Half rate.** A dense figure keeps an average of `controller.set` time and renders on alternate frames above 14 ms, every frame below 9 ms, with time-based motion unchanged.
- Reduced motion gets one still frame. Offscreen, the loop stops (IntersectionObserver).

## Arming
If an action needs the machine in a certain state (assembled before firing, door closed before spinning), arm first:
1. Set the targets that bring it into that state (`setApart(false)`, `select(null)`).
2. Mark it armed: the button shows `data-state="armed"` and the readout says `closing up · arming`.
3. Start the action's clock only when every spring has arrived and nothing is pending.

Never run the action on a half-exploded machine, and never snap the parts home to save time.

## Taking over from the tour
When the user acts during the idle tour, stop the tour and leave the state where it is. If your take-over helper resets something (for example, it forces the power on), and the user's action is itself a toggle of that thing, the two cancel out in the same frame and replay an animation. Pass the helper a flag (`takeOver(switching)`), and test every toggle in every phase of the tour, not only the default one.

A step key pressed during the tour must step from the value the reader sees, not from the tour's target, which may be far ahead. The ripple tank's keys start from `freq.x` and `gap.x` while touring (`state.tf` and `state.td` otherwise), and `steer()` zeroes both springs' velocity when it takes over, so the value only moves in the key's direction.

## Announcing phases
For a machine with phases, add a visually hidden `aria-live="polite"` element and write only phase changes to it: `arming`, `chill-down, venting`, `preburners lit`, `main stage`, `shutdown`, `cold`. Never write per-frame numbers to it. For a continuous control, wait until the input has been quiet for about 700 ms before you announce (the ripple tank's `announce`). Leave the live region empty at load and fill it with the current state on `focus`, so a screen reader hears the state when it arrives rather than a stale line. A button whose label changes ("Ignite" to "Shut down") doesn't also need `aria-pressed`.

## CSS motion
- `.iso-line[data-flow]` animates dashes, for things travelling along a path.
- `.iso-dots circle[data-pulse]` blinks a dot grid slowly, for an "active" surface.
- For hover lifts, wrap a group with `transition: transform 700ms var(--anatomy-ease)` and translate it up on hover.

All of these switch off under reduced motion in `ISO_CSS`.
