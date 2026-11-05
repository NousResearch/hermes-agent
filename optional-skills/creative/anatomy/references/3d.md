# 3D: parts that turn in the drawing

The optional 3D mode ("turning groups") lets some parts of a figure move as rigid bodies: a dial or turntable that spins, fingers that flex, a piece that breaks off and tumbles. Those parts are re-projected and re-shaded every frame by the kit's own rules and painted in an order solved from separating planes. Everything else stays ordinary static kit art. A figure that doesn't opt in is untouched.

**Status: beta.** The engine is proved on `examples/turning-dial` and one test figure (a desk fan built from this document alone), not yet on a production figure. Turn one subject at a time, keep clutter static, and measure your figure against section 15 before you add parts.

## Contents
1. When a figure earns 3D
2. The model
3. A minimal page
4. The authoring API
5. Part kinds and the rules that keep them true
6. Materials
7. Free groups: parts that break off
8. Hinge chains: hands and fingers
9. Painter's order and why it is correct
10. Shading and fades
11. Static layers and a messy workshop around a turntable
12. Shaders that follow the turn and the groups
13. The orbit audit
14. Browser checks
15. Performance
16. Known limits

## 1. When a figure earns 3D

Only when the motion is the explanation: a dial whose numbers are read as it turns, a turntable that shows an object from every side, a wheel pack, a hand that grips or breaks. Never as decoration. A figure that only needs a part to slide or a needle to swing stays 2D: `translateAlong` and a re-drawn needle are cheaper and exact.

## 2. The model

- **Groups.** A group is a rigid frame with a pose. Kinds: `turn` (about a vertical axis through `pivot`, any angle), `slide` (along a direction, `0…travel`), `hinge` (about any axis through a point, within `range`, in degrees) and `free` (any rotation plus a translation, set by the figure every frame). Groups nest: a pose is relative to the parent.
- **Rest.** Every constructor takes **world coordinates at rest** (every group value 0, every free group `null`). The engine subtracts each group's origin, so the recorder (`A.recorder`), the kit and the engine share one world.
- **Live layers.** A live layer is one `<svg class="iso-svg iso-live" data-live="name">` in the stage. Live and static layers stack in the order of `T.stack([...])`: for example `back svg → [dial live] → back canvas → [hand live] → front canvas → front svg`.
- **The identity.** Turning a group by θ about a vertical axis is exactly the kit drawing at azimuth `a − θ`, because the kit's light is camera-locked. So every frame of a turn group is the kit's own picture at P′(θ), shifted; the kit builders at P′(θ) are the fidelity reference, and every kit screen-space audit runs unchanged at P′(θ). Hinges and free groups compose matrices instead (correct, slightly slower).

## 3. A minimal page

`examples/turning-dial/` is the reference scene: a dial with a knurl, ticks and numerals, a second turn group (an index wheel under a window), a hinge (a pawl riding the knurl), a hand of about 80 parts with materials, a three-hinge finger that lifts, a hex pod that breaks off as a free group, and a WebGL LED, light pool and sparks. `examples/turning-dial/stress.mjs` is the performance scene (a dense hand on a turntable in a cluttered workshop); it is not audit-clean, so take patterns from `build.mjs`.

`build.mjs`, reduced to its skeleton:
```js
import * as k from "../../kit/iso-kit.mjs";
import * as A from "../../kit/audit.mjs";
import { turning } from "../../kit/turn-build.mjs";
import * as TA from "../../kit/turn-audit.mjs";
import { turnScript, glScript } from "../../scripts/inline-kit.mjs";

export const P = k.fitProjection(points, W, H, { pad: 22, azimuth: 40, elevation: 31 });
export const R = A.recorder(P);
export const T = turning(P, { recorder: R });
const dial = T.group("dial", { turn: { pivot: [0, 0] } });
const lift = T.group("f2.mcp", { parent: dial, hinge: { point: C1, axis: w, range: [-22, 0] } });
const hand = T.layer("hand");
T.stack(["back", "hand", "front"]);
export const follow = (values) => ({ ...values, "f2.mcp": -22 * (values.lift ?? 0) });
export const poses = () => Array.from({ length: 720 }, (_, i) => follow({ dial: i * 0.5, lift: (i % 90) / 90 }));

T.round({ name: "dial.body", group: dial, layer: hand, F: VERT(0, 0), s0: 10, s1: 14.4, r: 50, material: "steel", details: { ribs: { s0: 10.9, s1: 13.7, count: 150, seams: true, tone: "lo" } } });
T.prism({ name: "hand.palm", group: dial, layer: hand, plan: PALM, z: 19.4, h: 8, bevel: 0.9, material: "gold" });
T.tube({ name: "cable.red", group: dial, layer: hand, route, r: 0.7, hue: "red", material: "rubber", gaps });
// … static kit art: R.put({ name, layer: "back", svg, shapes }) as usual …

const out = T.build({ statics: R.items, poses, verify: argv.includes("--verify") });
A.settle(R, P);
if (argv.includes("--audit")) {
  A.auditOrExit(R, P);
  TA.orbitOrExit(T, R, P, { sweep: { at: (x) => follow({ dial: x }), from: 0, to: 360, frame: 0.3 }, clips: [{ name: "tap", at: (t) => follow({ dial: 40 + 12 * t, lift: tapAt(t) }), from: 0, to: 4, frame: 1 / 60 }] });
}
const stage = `<div class="iso-stage iso-turn" tabindex="0" role="slider" …>${backSvg}${out.layerSvg("hand", { width: W, height: H })}${frontSvg}</div>`;
const page = k.pageHtml({ title, theme, body: k.plateHtml({ …, body: `<style>${out.css}</style>${stage}` }), script: turnScript() + LIVE });
```
`live.js`:
```js
const controller = TURN.mount(stage, DATA.turn, { follow, onHold: () => (touring = false), onFrame: () => paint() });
function tick(now) {
  …
  controller.set({ dial: angle, lift });
  frame = visible && moving ? requestAnimationFrame(tick) : 0;
}
```
The figure owns the loop (`motion.md`); the controller never runs its own `requestAnimationFrame`. `out.css` is `TURN_CSS` (the face band rules, materials, hue tubes and the `.iso-turn` stacking): include it once.

**In React or Next.js** (`references/react.md`): copy `turn.mjs`, `turn-build.mjs` and `turn-audit.mjs` beside the kit with the imports rewritten (`./iso-kit.mjs` → `./kit`, and so on), and `TURN_CSS` as a `turn.css` imported once. The geometry module (server) runs `turning(…)` and `T.build(…)` and returns the stage markup (`out.layerSvg(name, { width, height })` for each live layer, in stack order between the static SVGs and canvases) plus `out.data` in the figure's data. The client mount calls `TURN.mount(stageElement, data.turn, { follow, onHold, onFrame })` and drives `controller.set(...)` from its own loop; return `controller.destroy` from the effect's cleanup. A verify build can put `out.verify` in the data and have the mount assign `window.__isoVerify` for `turn-check --order`.

## 4. The authoring API

### Scene
| Call | What it does |
|---|---|
| `turning(P, { recorder, scale = 1 })` | The builder `T`. Pass the kit recorder so every live shape is recorded for the kit audit too. |
| `T.group(name, { parent?, turn: { pivot: [x, y] } })` | Rotation about the vertical through `pivot`, value in degrees. |
| `T.group(name, { parent?, slide: { direction, travel } })` | Translation, value `0…travel` world units. |
| `T.group(name, { parent?, hinge: { point, axis, range: [lo, hi] } })` | Rotation about `axis` through `point`, value in degrees within `range`. |
| `T.group(name, { parent?, free: { origin } })` | A free rigid body (section 7). Value `{ R: [9] \| q: [x, y, z, w] \| axis, angle, t: [3], world? }` or `null` for rest. |
| `T.layer(name)`, `T.stack([…])` | Live layers and the full stack of live and static layer names, back to front. |
| `T.build({ statics, poses, verify = false, strict = true })` | Solves every pair, proves every static item's layer, packs the page data. `poses` is the figure's reachable states (a list or a function returning one): every motion clip, sampled. Throws a readable error on any pair with no separating plane, any static that can't be proved, any non-convex part. `strict: false` keeps interlocked pairs (so a planted fault reaches the audit). |

`T.build` returns `out = { svg, data, css, report, layerSvg(name, { width, height }), verify }`. `out.data` goes to the page (`DATA.turn`). `out.report` has `parts`, `layers` (per layer: `planar`, `framed`, `dynamic`, `skip` counts), `tight`, `margins`, `proofs`, `seconds`.

### Constructors
Common options on every constructor: `{ name, group, layer, tone = "hi", crease = "faint", lit = false, material, owner, details = {} }`.

| Constructor | Part |
|---|---|
| `T.prism({ plan, z, h, steps, bevel })` | An upright kit slab (the kit `extrude` rule). |
| `T.prism({ F, polygon, s0, s1, round = 0, steps = 4, bevel = 0 })` | A convex polygon in frame `F = { o, a, u, v }` extruded along `F.a` from `s0` to `s1`, any axis (the `prismOf` rule). |
| `T.round({ F, s0, s1, r \| [r0, r1], ends = ["flat", "flat"], fixed })` | Cylinder or cone frustum; each end `"flat"`, `"dome"` (radius must equal the side's: a capsule) or `"open"`. One dome and one flat end is a half-capsule, and any length works, even one shorter than the radius (a nose cone on a motor can). A vertical round in an upright group becomes a `fixed` picture automatically (`fixed: false` stops it). |
| `T.lathe({ F, profile, slope, radius })` | A general convex smooth body (r(s) concave). Costly; prefer rounds. |
| `T.ball({ c, r, flats: [{ n, d }] })` | A sphere; each flat keeps n·(p − c) ≤ d and draws its rim as a cap, so `n` is the flat's outward normal. Without flats it is a `fixed` picture, and it takes no `details`. It is the costliest kind to draw (section 15). |
| `T.tube({ route, r, chunk = 6, turn = 20, gaps, breaks, caps, rings, hue, material, bundle, owner, touch })` | The kit `tubePieces` semantics, chunked; `hue` is `"red" \| "green" \| "blue"`; records `R.route`. The route runs from mount surface to mount surface. The port, gland or clamp at each end is a separate part with `owner` set to the tube's name, and its span along the route goes in `gaps`. A route that starts at a port's outer face fails `terminals` in the kit audit (below). |
| `T.fixed({ svg, anchor, shapes, hull, F })` | Any kit markup whose picture never changes under the group's motion, moved by `translate`. |
| `T.plane({ part, o, u, v, svg, fade = [0.04, 0.3], normal })` | Line or text markup in a plane of the part (ticks, numerals, label plates), placed by one matrix, faded by facing. |
| `T.billboard({ part, at, normal, svg, fade })` | Screen-aligned markup at a projected point (an SVG fallback for a shader glow), faded by facing. `svg` is in viewBox px centred on the projected point, not in world units: multiply world sizes by `G.scaleOf(P)` (`<circle r="${(1.1 * G.scaleOf(P)).toFixed(2)}"/>` fills a lens of radius 1.1). |
| `T.split(part, { n, d })` | Two parts cut by the plane (prism: any plane through it along or across the axis; round: across). The cut edges are not stroked. Use it to break a genuine cycle. |

### Details (in the part's frame: `s` along the axis, `x` and `y` in `u` and `v`)
Each constructor has its own frame and takes its own detail kinds. Anything else is ignored without a warning.

| Constructor | Frame `[s, x, y]` | Takes |
|---|---|---|
| `T.round`, `T.lathe` | `F`: `s` along `F.a`, `x` along `F.u`, `y` along `F.v` | `seams`, `ribs`, `bolts`, `dots`, `rings`, `rules` |
| `T.prism({ F, polygon })` | `F` | `dots`, `seams`, `bevel` |
| `T.prism({ plan, z, h })` | The world: `s` is z, `x` and `y` are world x and y | `dots`, `seams` (z values), `bevel` |
| `T.ball` with flats | Origin `c`, `a` = the first flat's `n`, `u` and `v` from `G.frameAlong(c, n)`. The flat is at `s = d` and the far pole at `s = −r` | `bolts`, `dots`, `rings`, `rules` |
| `T.ball` without flats | None: a fixed picture | Nothing |
| `T.fixed({ F })` | `F` (no `F`: no details) | `bolts`, `dots`, `rings`, `rules` |

A centre dot on a domed spinner `T.ball({ c, r: 4.5, flats: [{ n: [-1, 0, 0], d: 0 }] })` (the flat faces −x, the dome +x) sits at `at: [-4.5, 0, 0], normal: [1, 0, 0]`: `s = −r` is the pole. `[4.5, 0, 0]` would land 4.5 behind the flat, inside whatever the spinner is mounted on. For a ring or a rule on a prism face, use `T.plane`.
```js
details: {
  seams: [s…],                                              // rings on rounds, ring seams on prisms
  ribs: { s0, s1, count, phase, twist, fade: [0.1, 0.55], seams: true, tone },   // knurl rows on rounds
  bolts: [{ s, r, count, phase, size: 0.5, tone: "mid", fade: [0.04, 0.42] }],   // bolt circles
  dots: [{ at: [s, x, y], normal, size, tone }],
  rings: [{ at: [s, x, y], normal, r, tone }],              // become planes
  rules: [{ points: [[s, x, y]…], tone, free }],            // become planes
  bevel: 0.4…1.6,                                           // prism cap bevel (faded slots)
}
```
Every detail fades with facing: dots and bolts by `smoothstep(0.04, 0.42, n·V)`, ribs per rib, bevel lines per edge, planes on the group's opacity. No texture stops at full strength.

### Runtime (`kit/turn.mjs`, inlined with `turnScript()`)
| Call | What it does |
|---|---|
| `TURN.mount(stage, data, { onHold, onFrame, follow })` | Binds every live layer in `stage`, renders the rest pose, installs `window.__isoTurn`. `follow(values)` maps figure values to group values on every `set` (one input drives coupled groups). `onFrame(controller)` runs after each render (draw shaders here). `onHold()` runs when a tool calls the hook. |
| `controller.set(values \| θ, { detail })` | Sets group values (a number sets the first turn group), renders synchronously, returns stats. `detail` (0…1) scales faded lines, ribs and dots (section 15). |
| `controller.angleAt(clientX, clientY, z, group)` | The pointer as an unwrapped angle about the group's pivot on the plane at height `z` (drag a dial). |
| `controller.pose(name)` | The group's world pose `{ R, t }`. |
| `controller.detach(name)` | The `{ world: true, R, t }` value that continues a free group's current pose in the world frame (no jump). |
| `controller.anchor(group, restPoint, restDir?)` | Where a rest-world point (and direction) of a group is now: `{ world, screen, direction }`. Spark sources, flame roots, LED positions. |
| `controller.cover(names, options)`, `order(layer)`, `orderIndex(layer)`, `events()`, `stats()`, `frames()`, `cams()`, `scene()`, `values`, `destroy()` | Cover masks (section 12), paint order, order-change events, `{ ms, writes, moved, forced, refined, unseparated, emitted }`. |
| `TURN.cover(controller, names \| [names, names…], { polygons = 6, edges = 10, pad = 0.3, layers })` | `{ uCoverEdge, uCoverCount, dropped, parts }` for `coverOf`. |
| `TURN.poseUniforms(controller, group, prefix = "uGroup")` | `{ prefixX, prefixY, prefixZ, prefixT }` such that `toGroup(p)` gives the group's **rest-world** coordinates. |
| `TURN.materials`, `TURN.materialCss()` | The material names; the material CSS alone (it is already in `TURN_CSS`). |
| `TURN.exact(on)` | Write path coordinates unrounded (6 decimals) instead of to 0.01 px. The orbit audit turns it on for the continuity line; a page never needs it. |
| `TURN.rotation(axis, degrees)`, `TURN.quat(q)`, `TURN.compose(A, B)` | 3×3 rotations as arrays, for free poses. |
| `TURN.floorOf(scene, group, R)` | The lowest z of a group's parts under rotation `R` (relative to its origin): to set a piece down flush. |
| `window.__isoTurn` | `{ controller, set, get, order, events, stats, layers }`; every call holds the figure (`onHold`). `set` takes a number or a values object. |

## 5. Part kinds and the rules that keep them true

These are binding; the build and the audit fail without them.
- **Convex parts only.** A stepped dial is tiers; a notched wheel is a hub disc plus rim sectors; a U-bracket is three prisms; a hand's shell is plates. `T.prism` checks its polygon, `T.lathe` its profile.
- **Contacts are flush, tangent or rim-on-flat.** A rim on a sphere is not separable: give the ball a flat (`T.ball({ flats })`) and land the rim on it. A pad sunk into its host is not allowed in a live layer (`separation` fails with `interlock A × B −0.6`).
- **Joints.** The two links stay at least 0.2 apart, and the joint is bridged by a knuckle disc on the child group: coaxial with the hinge axis, and touching the parent's link, flush or tangent. A disc coaxial with its hinge (or any flat face turning in its own plane) is invariant under the hinge, so it never collides, and it is the contact the kit audit needs. With nothing in the child group touching the parent, `rigid` reports every part of the child group as `support loose`. Links meeting at a joint share one radius; dome centres sit the same distance from the joint so the gap holds at any angle (two domes of radius r at distance c from the joint stay apart while cos(β/2) ≥ r/c). A trunnion on a yoke:
  ```js
  T.round({ name: "yaw.arm+", group: yaw, layer, F: VERT(0, 11.7), s0: 40, s1: 61, r: 1 });
  T.round({ name: "head.can", group: tilt, layer, F: frameOf([0, 0, 56], [1, 0, 0], [0, 0, 1]), s0: -12, s1: 8, r: 9 });
  T.round({ name: "head.trunnion+", group: tilt, layer, F: frameOf([0, 9, 56], [0, 1, 0], [0, 0, 1]), s0: 0, s1: 1.7, r: 4.5 });
  ```
  `tilt` hinges about y through `[0, 0, 56]`. The can reaches y = 9 and the arm starts at y = 10.7, so the links are 1.7 apart. The trunnion runs from 9 to 10.7: it sits on the can and its end face is tangent to the arm. Ending it at 10.5, with a 0.2 gap to the arm, fails `rigid` for the whole head.
- **No live line spans two moving groups.** A cable or hose crossing a flexing joint is two routes ending in a clip or port on each side of the joint, or crosses at the hinge axis through a rotary union (banjo). A cable that must reach a turning part from static ground goes up the axis through a slip ring at the hub (section 11).
- **Tubes bend nearly flat or inside fittings.** A bend of radius R in a plane tilted θ from horizontal, toward the viewer, is an ellipse on screen whose tightest radius is R·sin²(e − θ) (in world units; × `G.scaleOf(P)` for px). `folds` fails a bend that turns more than 90° on screen round less than 2r, and `self` one under r. So a bend needs **R ≥ 2r / sin²(e − θmax)**, where θmax is the plane's own tilt plus the pitch range of every hinge it rides on (a turn group sweeps the tilt through every azimuth, so the worst case always comes round). At e = 30°:

  | θmax | 0° | 5° | 8° | 10° | 12° | 15° | 20° |
  |---|---|---|---|---|---|---|---|
  | R ≥ | 8r | 11r | 14r | 17r | 21r | 30r | 66r |

  8r is the bound itself on a horizontal plane, not a margin: take 10r. Past about 15° no practical radius holds, so turn inside a fitting (clip block, banjo, elbow ball, a junction box) or run the line straight. Draw bends with `G.fillet(points, R)`, never the view-dependent fillet. The audit's `folds` and `self` run at every angle and pose.
- **Tubes never point along the view.** A tube's tangent elevation stays out of 15°–45° and −45°–−15° at e = 30° (`endon`), in every pose: on a hinge chain, check the link's pitch range too (a cable on a link that flexes 30° will fail).
- **Lathes are the one kind with small residual steps** (≤ 2.5 tone·px² where a band changes topology). Prefer rounds, domes and balls with flats for live parts.

## 6. Materials

A part's material is one option on its constructor (`material: "gold"`). Tones still come from facing exactly as the kit's do; a material remaps the ramp (the four shades, the top, paper, outline and line tones, dots and the tube body, shine and shade), with a light and a dark set each.

| Material | Look |
|---|---|
| `gold` | Champagne-gold anodised aluminium: warm pale top, bronze shadows, dark bronze outline. |
| `chrome` | Bright polished steel: a light far band, a dark reflection band (n·L 0.2–0.42) next to a bright band and a white highlight. Non-monotonic on purpose: the bands move with facing, never jump. |
| `steel` | Satin steel, monotonic and cool. |
| `gunmetal` | Dark blue-grey metal with light dots. |
| `rubber` | Black rubber and cable sheath with a soft sheen. With `hue`, a tube becomes a coloured sheath (the hue mixed into the rubber ramp). |
| `brass`, `copper` | Warm metals for fittings, ports, valves. |

Static parts use the same families as plain classes: wrap kit markup in `<g data-mat="brass">…</g>` (the example's index wedge). The rules live in `TURN_CSS`; a page with static art only can include `TURN.materialCss()`. A dark plate inside a light page picks the dark set (`.iso[data-theme="dark"] [data-mat]` is ordered last).

On a light page a material figure is far stronger than kit greys: keep the ground (bench, base, plates) in kit tones and put materials on the subject, so it reads as the lit, finished object.

## 7. Free groups: parts that break off

`T.group("pod", { parent: dial, free: { origin } })` makes a rigid body whose pose the figure sets every frame. At rest (`null`) it sits exactly where it was built, attached: the parent carries it. To break it off, start setting a pose relative to the parent: there is no single-frame change because the first value is the identity.
- **Falling onto the turntable**: keep the parent as the turntable group, so the piece lands and then turns with it. Compute the flight as a pure function of time (`breakAt(t)` in the example: a somersault about the centroid, a ballistic arc, a flush landing found with `TURN.floorOf`, a decaying slide and yaw). A pure function can be sampled by the build and the audit, and played backwards.
- **Falling off onto the bench**: switch to world space with `controller.detach(name)`, which returns `{ world: true, R, t }` continuing the current pose, then integrate from there.
- **Order**: a free group's parts are paired at runtime with every part of their layer (by screen box, then a GJK plane cached per relative pose); intra-group pairs are planar. Cycles among free parts are possible in principle; the audit samples the clips and reports them.
- **Proof**: give the build the clips in `poses()` (every frame of the break at several turn angles), and the audit the same clips (`clips`). The audit poses every shape in every clip frame and checks clearances against live and static parts (`posed`), order, cycles and continuity.

## 8. Hinge chains: hands and fingers

Each link is a hinge group whose parent is the previous link (`T.group("f0.j1", { parent: f0j0, hinge: { point: J1, axis: w, range } })`). A wrist is a hinge carrying the palm; a thumb base is a hinge (vertical axis) carrying the thumb chain. Drive the chain from one figure value through `follow` (`lift` → three angles).
- Pairs on different links are solved at build where possible: a pair whose swept volumes over the build's `poses()` have a plane in their common ancestor's frame with margin ≥ 0.1 becomes a **framed** plane (no runtime GJK). Only near-joint pairs stay dynamic (GJK, warm-started, cached while the two groups keep their relative pose). So `poses()` must cover the motion: if a gesture leaves the sampled range, the audit's `unseparated` line names the pair ("framed plane: … add it to poses()").
- Keep ranges physical and small: a three-link finger at ±5° per joint already reads as a hand moving; 20° per joint needs every route and contact re-proved.
- **A route on a hinged link carries the link's pitch range.** Section 5's `endon` rule and its bend bound both use the tilt the line can reach, not the tilt at rest: a horizontal 90° bend at R = 8.6r (inside the 8r rule for a flat plane) on a ±12° hinge fails `folds` and `self`, because at 12° of pitch it needs 21r. Make the line straight into a junction box, or move the bend to the parent.

## 9. Painter's order and why it is correct

Every pair of parts that can overlap on screen gets one relation: a plane fixed in their shared group (planar), a plane fixed in their common ancestor's frame (framed), a GJK plane each frame (dynamic, different groups that move relative to each other) or a skip (neighbouring chunks of one route). B is in front of A when n·V > 0 in that frame. Order is a stable Kahn sort keyed by the previous frame's order; on a stall, edges between parts whose exact screen hulls are apart (or touch within 0.1 px) are dropped; a remaining stall is a forced pick, counted and made fatal by the audit.

Correct because two disjoint convex solids whose projections overlap are both hit by some view ray, and a separating plane crosses that ray once. Flicker-free because a plane's relation flips only when the plane is edge-on, and then the two footprints lie on opposite sides of a line: swapping them changes no pixel. Edge-on relations (|n·V| < 1e-7) add no edge.

## 10. Shading and fades

- Flat faces (prism sides, caps, ball flats) use the kit's thresholds (`EB` for upright prisms, `LB` otherwise) through a smoothed staircase: a face holds the kit tone except within ±0.035 of a threshold, where it crossfades in 64 steps per band (`data-q`, 0…256). That fine step keeps chrome's high-contrast bands smooth.
- **The continuity line measures the unrounded geometry.** The page writes path coordinates rounded to 0.01 px, which moves an edge by at most 0.005 px, far below a device pixel. Measured on those rounded paths, a long edge's rounding steps never shrink, so the halving test once read large flat faces (the example's palm, a desk fan's 21-tall prism arms) as jumps that were not there. `orbit` therefore switches the runtime to exact coordinates (`TURN.exact(true)`) for the continuity line and back afterwards; large flat faces are fine on turning groups.
- Curved surfaces always use analytic iso-light bands (`labelledArcs`), never facets. Capsule domes are nested hulls; balls with flats are clipped caps. Bands are born at zero width.
- Faded-line slots hold ribs and bevels: alphas quantised to 1/32, as many slots as the build's orbit sweep shows a set needs.
- `lit: true` uses the lit tokens; with a material, the material's lit set.

## 11. Static layers and a messy workshop around a turntable

Every static item carries `layer` (`R.put({ …, layer: "back" })`) and the build proves a constant relation with every live layer over all reachable states, by the first rule that holds: a **horizontal plane** (the item is wholly under or over the layer's z range), the **cylinder rule** (the item lies wholly behind or in front of the turn group's swept cylinder along the view's floor direction), **never overlap** on screen, or a **swept-envelope plane**. Otherwise: `static S interleaves live layer Λ: split S at z = …, or move it`.

Authoring rules for a dense workshop (`examples/turning-dial/stress.mjs` proves 450 relations: bench, turntable base, power unit, tool rack, cable reel, tray, wrench, three bench cables):
- **Keep clutter outside the swept cylinder or under the turntable plane.** Tall things (power units, racks, lamps) go behind or in front of the cylinder of radius R (the live layer's reach from the axis) along the view's floor direction (cos a, sin a). Things beside it, neither behind nor in front, must be low.
- **Anything under the turntable's lowest live z is behind the live layer, even when it sits in front on the bench.** It belongs in a layer under the live one (`back`), not `front`: the build says so.
- **A cable that must reach the turning part** runs on the bench to the stationary base, climbs inside it and reaches the turning part through a slip ring or rotary joint at the hub. Draw the static run to a port on the base, and start the live cables from a terminal block on the turning platter. No route ever spans static ground and a turning group.
- Static cables lie flat on the bench (bends in the bench plane, R ≥ 8r) so they never fold, and end on a port, a reel or a free end marked `free`. A cable resting on the bench (centre at bench top + r) touches it, which the kit's `clearance` check fails (`close stand.feed × stand.bench clearance 0.00 < 0.40`) unless the route says so: `R.route(name, route, r, { owner: name, limp: true, touch: ["stand.bench"] })`.
- `stress.mjs` is a performance and layer-proof scene, not an audit-clean one: its bench cables have no `touch` and its hand has loose links, so the kit audit fails it. Copy authoring patterns from `build.mjs`.

## 12. Shaders that follow the turn and the groups

`kit/gl.mjs` gains (both opt-in; old fragments see nothing new):
- `GLSL_TURN`: `uGroupX/Y/Z/T`, `toGroup(p)`, `toWorld(g)`, `dirToGroup(d)`, `dirToWorld(d)`, and the cover masks `uCoverEdge[60]`, `uCoverCount[6]`, `coverOf(vb, soft)`.
- `glslPose(Name, prefix = "uName")`: the same four functions for any group (`toDial`, `fromDial`, `dirToDial`, `dirFromDial`), so one shader can follow several groups.
- `glLayer` sends arrays longer than 4 to `vec2`/`vec3`/`vec4` uniform arrays by their declared type (float arrays unchanged).

How a shader follows the motion:
1. **Content on a moving part** (an LED on a fingertip, a flame from a ruptured line's end, oil jetting from a pipe end): take the anchor every frame with `controller.anchor(group, restPoint, restDir)` and pass `world`, `screen` and `direction` as uniforms. Draw it on the **front canvas** (over the live layer) and occlude it with the parts painted after it: `const c = TURN.cover(controller, [partName]); layer.draw({ ...c, … })` and `ink *= 1. - coverOf(vb, .7 * px)`. Every live part is convex, so its hull is its exact silhouette; the cover is pushed out 0.3 vb so ink stops at the occluder's stroke, never in it. Several targets share the six polygons: `TURN.cover(controller, [["hand.f2.p3"], ["hand.pod.socket"]])`.
2. **Content fixed to a turning surface** (oil pooling on the turntable, a texture revealed by light on a dial): pass `TURN.poseUniforms(controller, "dial", "uDial")` and evaluate in rest-world coordinates: `vec3 q = toDial(onFloor(vb, FACE_Z));` The pool turns with the dial.
3. **Content in world space** (smoke rising, sparks after they leave the source, oil pooling on the bench outside the turntable): never touch a group transform. Emit particles from an anchor's world position at birth and integrate them in world space (the example's sparks): the turn doesn't drag them along.
4. **Back canvas** (light falling on a surface under the parts that cover it): put the canvas between two live layers. The example splits its dial into its own live layer `dial` under `hand`, so the LED's pool lands on the dial face and the hand covers it with no masks at all.

Every rule of `webgl.md` still holds: envelopes, windows to 0 before any bound, light-theme inks (saturated bodies, pale cores, tinted ground, ink only grows with light), dither, canvas edges at 0, an SVG fallback (`T.billboard` inside the owning part, hidden under `[data-gl]`, handed over with `(1 − u)/(1 − u·α)`). The cover chunk uses 70 uniform vectors: read `MAX_FRAGMENT_UNIFORM_VECTORS` and keep the SVG fallback below 128. `GLSL_TURN` defines `toWorld`: don't prepend it to a fragment that declares its own.

## 13. The orbit audit

`node build.mjs --audit` runs the kit audit, then `TA.orbitOrExit(T, R, P, options)`, which prints one line per check and ends `turn audit passed` or `turn audit failed: n problems` (exit 1). Only a full run ends `turn audit passed`: a run with `only` ends `turn audit partial: no problems in what ran (n lines skipped by only)`, and a quick one `turn audit quick: no problems in what ran (quick run, undersampled: the full run is the proof)`.

Options: `sweep: { at(x), from, to, frame }` (the turn), `clips: [{ name, at(t), from, to, frame }]` (gestures, the break-apart: every frame at 60 fps), `poses` (extra states), `crowd: [[route, part]…]` (pairs exempt from `crowding`, after you have looked), `cover`, `fittings: [[route, from, to]]` (stretches inside a fitting, exempt from `endon`), `quick` (≈ 10× fewer states, for iteration), `only: ["order", "rigid", "swept", "posed", "routes", "coverage", "continuity", "depth"]` (an unknown name throws), `log`. The example takes `--quick`, `--only=a,b` and `--verbose`. `only` runs these lines: `order` → `unseparated`, `cycles`, `forced`, `swaps`; `routes` → `folds`, `self`, `crowding`, `endon`; `depth` → `depth`, `layers`; the others their own line. `separation` always runs. A line that did not run prints `skipped`.

**Time and how to iterate.** The orbit audit is slow, and its time grows with parts × states. Measured on a 28-part desk fan (one turn, one hinge, one clip): `--quick` 85 s, full 17 minutes. The example scene takes several minutes; budget about 30–40 s per part for a full run. That is past a 10-minute shell timeout, so:
1. Iterate with `--quick --only=…` on the lines you are fixing.
2. `--quick` alone before the final run. It undersamples: it checked 100 halving candidates against 800 in the full run and missed 20 `continuity` failures the full run found.
3. The final full run in the background (`node build.mjs --audit > audit.txt 2>&1 &`), read when it ends. Only that run's `turn audit passed` counts.

| Line | Fails on | Fix |
|---|---|---|
| `separation` | A same-group pair with no plane (margin < −0.08) | Split, a flat, or a flush contact |
| `unseparated` | A cross-group pair interpenetrating while their hulls overlap; a framed plane that a state breaks | Fix the contact or the motion; add the motion to `poses()` |
| `cycles` | A cycle among overlapping parts in some state; prints the parts, the state range and the split plane (the cycle edge with the largest margin) | `T.split` that part on that plane |
| `forced` | The runtime had to force a pick | As cycles |
| `swaps` | Two overlapping parts swap order while overlapping more than 0.25 px² (0.5 px² for a pair built flush, margin under 0.05: at the instant their contact plane is edge-on, the true overlap is a line, and the drawn curves leave a sliver about 0.1 px wide) | Look first: `drive.mjs page.html '[["orbit", "swap.png", θ−0.3, θ+0.2, 0.1, null, 6, [x, y, w, h]]]'` (θ from the line; the region round the pair in CSS px of the plate, at scale 6). Something that appears or vanishes in one step is a drawing bug, not an order bug. Engines before 2026-10-06 drew a round with one `"dome"` end shorter than its radius with a wedge behind its flat end, which vanished when that end went edge-on: copy the current `turn.mjs`. A clean swap with sane order on both sides (`__isoTurn.order`) is a wrong plane: report it with the part pair and θ |
| `layers` | A static item on the wrong side of a live part (ray-cast) | Move it or split it (section 11) |
| `rigid` | Kit clearances, terminals, supports, solid clearances at rest | As in the kit audit. `support loose` on every part of a hinged group: nothing in it touches the parent (the knuckle disc, section 5). `terminal … from X`: the route starts at a port's face, not the mount (the `T.tube` row, section 4). `clearance … × stand.bench`: a bench cable without `touch` (section 11) |
| `swept` | A static shape inside a turn group's swept envelope | Move it out of the cylinder |
| `posed` | Shapes of different groups (and moving live × static) passing through each other in any sampled state | The motion or the geometry |
| `folds`, `self`, `endon` | Live routes at every angle and posed state | Section 5 (bend radius with the pitch range, section 8) |
| `crowding` | A live route's end or bend lands on the outline of a slender solid (a rod, post, stem) at some angle | A line parallel to a turning rod crosses the rod's outline at some angle whatever its offset, so you can't design it out. Look at the angles it names (`["turn", θ]` and a 4× shot), and if it reads as two separate parts, list the pair in `crowd: [["cable.a", "yaw.stem"]]`, or route the line inside the rod. A line that is not parallel: approach across the rod |
| `coverage` | A part's recorded shape covers < 60% of its drawing | Record the true shape |
| `depth` | The ray-cast oracle disagrees with the engine order | Report it |
| `continuity` | Per 0.05°: tone step > 1/16 band, a birth > 2 px², a dot α step > 0.01, a faded α step > 1/32 + 0.005, outline motion over 1.5·ω·Δθ + 0.05 px; and the halving test | A detail that stops: fade it. A part whose outline or face boundary jumps in one step: find the angle with an orbit sheet at 0.1°; on a plain kit primitive it is an engine bug, report it with the part and θ. `--quick` undersamples: only the full run counts |

The halving test re-emits each candidate step at ½, ¼ and ⅛ of itself with exact coordinates, rasterises the frames at 4× (jittered samples on 8 sub-scanlines) and measures the pixels that change owner, weighted by tone. A smooth change halves with the step; a jump doesn't. A step is a discontinuity when it stays above 60% at each halving and is still ≥ 2 tone·px² at ⅛, measured at 4× and again at 16×: a step that fails at 4× fails only if it also fails at 16×. Measured on the page's rounded paths instead, every long edge keeps a floor of about its length × 0.01 px that never shrinks; that floor is what flagged large flat faces before the switch to exact coordinates. Every line has a planted fault that fails it (`3d/engine/plants.mjs` in the build session: a sunk pad, a pinwheel of three tilted sticks, a vertical-plane fibre bend, a tube at 30°, a stopped rib row).

## 14. Browser checks

```sh
node examples/turning-dial/build.mjs --light --verify           # writes turning-dial-verify-light.html
node scripts/turn-check.mjs examples/turning-dial/build.mjs --fidelity
node scripts/turn-check.mjs page-verify.html --order --step 1
node scripts/turn-check.mjs page-verify.html --order --states clip.json
node scripts/turn-check.mjs page.html --pops --no-webgl --step 1 --out shots/pops
node scripts/turn-check.mjs page.html --lines                    # and the dark page
node scripts/turn-check.mjs page.html --perf --width 1440
node scripts/turn-check.mjs page.html --perf --width 390 --throttle 4
node scripts/turn-bench.mjs examples/turning-dial/build.mjs
node scripts/drive.mjs page.html '[["turn", 37], ["shot", "a.png"], ["orbit", "orbit.png", 0, 345, 15]]'
```
- `--fidelity` (Node): prisms against `k.extrude`/`G.prismOf` (outline, top and faces each ≤ 0.06 px), rounds and lathes against `G.lathe(…, smooth)` (outline ≤ 0.05 px against the kit drawn at 10×, bands ≤ 0.5 px; the line also prints the outline against the kit's own paths, which round to 0.1 px, so 0.08–0.13 px there is expected and not judged), tubes against `tubePieces` (0.05 px), at P′(θ). Control: one prism drawn 2° off must fail.
- `--order`: the recorded shapes in a WebGL z-buffer against the DOM painter's fills, 2×; a live pair with 4+ interior pixels fails. Control: each live layer reversed must give ≥ 1000 px. Static × live mismatches are listed (static recorded shapes are often stand-ins, such as a slab for a plate with a window). `--part name` prints a part's footprint in both images. `--step` sweeps the **first turn group only**, with every other group at rest: a hinge, a slide or a clip needs `--states`.
- `--pops`: first difference (θ → θ + 0.02°, two erosions) and second difference (0.1°, |2b − a − c| > 56, or > 6 on flat pixels, one erosion; a core of 6 px fails). Control: the largest part hidden for one frame. `--out dir` saves the first core's frames. With `--states`, consecutive entries are treated as neighbouring frames and only the first difference runs, so the list must be **one continuous path in steps of about 0.02°**: 529 unrelated yaw × tilt poses gave 404 false failures; a tilt sweep of 1201 states at 0.02° gave 0.
- `--states file.json`: a list of value objects instead of angles. Write them from the build module, which exports `follow` and the clip functions:
  ```js
  import { writeFileSync } from "node:fs";
  import { follow, nodAt } from "./build.mjs";
  const at = (t) => follow({ yaw: 20 + 30 * t, tilt: nodAt(t) });
  writeFileSync("clip.json", JSON.stringify(Array.from({ length: 241 }, (_, i) => at(i / 60))));
  writeFileSync("tilt-path.json", JSON.stringify(Array.from({ length: 1201 }, (_, i) => follow({ yaw: 30, tilt: -12 + i * 0.02 }))));
  ```
  `clip.json` (every 60 fps frame of the clip) is for `--order`; `tilt-path.json` (one hinge swept at 0.02°) is for `--pops`. Run `--pops` on one path per hinge, at a turn angle where the hinge's parts overlap the most.

## 15. Performance

Measured by `turn-bench` (Node CPU) and `turn-check --perf` (browser, 240 frames at 1.1°).

**What a part costs.** `turn-bench` prints the Node time per part by kind. In prism units (one live prism, about 0.03–0.04 ms per frame on the build machine):

| Kind | Cost | Kind | Cost |
|---|---|---|---|
| `fixed` (vertical rounds, plain balls) | 0.5 | `round` | 2.5–4 |
| `prism` | 1 | `ball` with flats | 8–10 |
| tube chunk | 2–3 | `lathe` | about 25 |

A figure's weight is the sum. A **small figure** weighs up to about 50 (a few dozen parts, mostly prisms and fixed pictures); a **dense** one up to about 230. The desk fan (28 parts: 6 prisms, 7 fixed, 5 tube chunks, 8 rounds, 2 balls with flats) weighs about 65, and measured 1.6–1.9 ms of Node CPU per frame and a 6.0 ms browser task at 1440, so it is over the small-figure line, mostly because of its two balls. The turning dial (88 parts, 42 rounds, one lathe) weighs about 225 and measures 6.4–7.2 ms. A ball with flats is the right part for a domed end with a rim landing on it (section 5), but it costs as much as nine prisms; a plain ball or a vertical round is a fixed picture and nearly free. Times vary by machine and load: compare kinds within one run, and measure your own figure.

| Measure | Budget for a small figure (weight ≤ 50) | Budget for a dense figure (weight ≤ 230) |
|---|---|---|
| Node CPU per frame | ≤ 2 ms | ≤ 7 ms |
| Browser main-thread task at 1440, DPR 2 | ≤ 6 ms | ≤ 13 ms (60 fps) |
| Main thread at 390, DPR 3, 4× CPU | ≤ 16 ms | ≤ 33 ms (30 fps) |
| `data.turn` JSON | | ≤ 250 KB |

Levers, in order:
1. `fixed` for every vertical-axis turned part and every plain sphere (automatic).
2. Planes for every set of lines or text on a face; upright prism tops are one matrix.
3. Small parts sample by device pixels: rings, caps and domes of a part under 30 px round take 12–16 samples, never the large-part minimums.
4. **Framed planes and the relative-pose cache** (automatic): cross-group pairs cost nothing while their groups keep their relative pose (a pure turntable turn), and pairs whose motion the build sampled are framed planes.
5. **Adaptive half rate** (the figure's loop): keep an exponential moving average of `controller.set(...)` time; above 14 ms render the live layers on alternate frames (time-based motion unchanged), below 9 ms every frame again (`stress.mjs`'s loop).
6. **Detail by speed** (opt-in): `controller.set(values, { detail })` with `detail = settle(1 − smoothstep(120, 240, |ω|°/s), 0.25 s)`: faded lines, ribs and dots scale and are skipped below 0.01. Outlines and tones are never reduced.

Authoring rules that keep a figure inside budget: about 150–250 live parts per moving group; small details as `details` (dots, rings, bolts) rather than parts; static clutter as static art; one live layer per independently moving region; tubes chunked at 6 with as few chunks as the bends allow. The measured numbers for the example and the stress scene are in the skill's build notes; a 380-part flexing hand on this machine is a 30 fps figure at 1440 with the half-rate lever, not a 60 fps one.

## 16. Known limits

- Vertical turn axes take the azimuth shortcut; other rotations compose matrices.
- Only convex parts. Concave bodies are convex pieces; ball flats replace rim-on-sphere joints.
- Live tubes obey the fold and end-on rules: there are no swept-tube silhouettes, so turns happen flat or in fittings, and no route spans two moving groups.
- No automatic splitting: a cycle is an audit failure with a suggested plane.
- Flat faces crossfade over ±0.035 of score (about 8° of turn).
- Lathes keep small residual steps where a band changes topology.
- Free groups pair at runtime with their whole layer; dozens of debris pieces cost GJK while they move.
- Phones: the budget aims at 30 fps, not 60, and only a figure inside it gets there. The 380-part stress scene (`examples/turning-dial/stress.mjs`) measured about 20 fps at 1440 and 13–15 fps at 390 with 4× CPU throttle, with the half-rate lever, on a heavily loaded machine. There is no canvas painter.
