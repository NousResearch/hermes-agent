# Kit API

Everything is in `kit/iso-kit.ts`, with the same API in `kit/iso-kit.mjs`. World coordinates are `[x, y, z]`, with x and y on the floor and z up, in units of about 1 px. Screen coordinates are `[sx, sy]` in the SVG viewBox. Paths come back as SVG `d` strings, rounded to 0.1.

## Contents
1. Types
2. Camera
3. Solids
4. Lines on faces
5. Parts library
6. Moving things
7. SVG strings (framework-free)
8. Card and page
9. React components
10. Round parts on any axis (`kit/lathe.mjs`)
11. Pipes, tubes and cables (`kit/tube.mjs`)
12. The audit (`kit/audit.mjs`)
13. The WebGL layer (`kit/gl.mjs`)
14. Tools
15. Turning groups (`kit/turn.mjs`, `turn-build.mjs`, `turn-audit.mjs`)

## 1. Types
```ts
type Flat = [number, number];            // screen point
type Point3 = [number, number, number];  // world point
type Plane = [number, number];           // floor point (x, y)
interface Plan { x: number; y: number; w: number; d: number; r: number }   // rounded rectangle on the floor
interface Projection { origin: Flat; scale: number; azimuth?: number; elevation?: number }
interface SolidPaths { fill; outline; crease; top; shades: string[4]; bevel?; seamless? }
type Tone = "hi" | "mid" | "lo" | "faint";
type LineTone = Tone | "lit" | "red" | "green" | "blue";
```

## 2. Camera
| Function | What it does |
|---|---|
| `fitProjection(points: Point3[], width, height, { pad = 24, headroom = 0, azimuth = 45, elevation = 30 })` | A projection that centres and fits these points in the viewBox. Pass every extreme point, such as `boxCorners(plan, z0, z1)` and the tallest tip. |
| `iso(point, P)` | World to screen. |
| `depthOf(point, P)` | Larger means nearer the viewer. Sort ascending to paint back to front. |
| `towardViewer(P)` | The floor direction pointing at the viewer, `[cos a, sin a]`. Faces whose normal points this way are visible. |
| `screenRise(h, P)` | How many screen px a height of `h` covers. |
| `cameraOf(P)` | The raw sines and cosines. |

## 3. Solids
All solids are prisms: a floor outline extruded straight up.

| Function | What it does |
|---|---|
| `slabOf(plan, z, h, P, steps = 8, bevel = 0)` | A rounded box. `steps` is the points per corner (2–4 for small parts, 8–10 for large rounded ones). `bevel` (0.3–1.6) adds a highlight line inset on the top along the visible edges, running into the silhouette corners. |
| `cylinder(cx, cy, r, z, h, P, steps = 32, bevel = 0)` | A round post, puck, foot, knob or pad. Use 16–20 steps for small ones and 56–72 for big discs. |
| `extrude(ring: Plane[], z, h, P, { convex = true, bevel })` | Any floor outline. For concave outlines pass `convex: false` (the fill becomes faces rather than a hull). |
| `ringOf(plan, steps)` | A rounded-rectangle outline as points. |
| `roundedPlan(plan, steps)` | The same thing. |
| `circleRing(cx, cy, r, steps)` | A circle outline as points. |
| `insetPlan(plan, by)` | Shrinks a plan (use a negative `by` to grow it). Keeps the radius sensible. |
| `corners(plan, inset)` | Four floor points inset from the corners, for feet and screws. |
| `boxCorners(plan, z0, z1)` | The eight corners, for `fitProjection`. |
| `haloOf(plan, z, h, P)` | A fill path for a soft shadow: `<path class="iso-halo" d=…>`. |

What a `SolidPaths` contains:
- `fill`: paper-coloured, it hides whatever is behind.
- `shades[0..3]`: visible sides, bucketed by how they face the light.
- `top`: the top face, lighter.
- `crease`: where top meets the visible sides.
- `bevel`: optional.
- `outline`: the silhouette.

`solidSvg` and `<Solid>` paint them in that order.

## 4. Lines on faces
| Function | What it draws |
|---|---|
| `onTop(points: Plane[], z, P, closed?)` | A polyline on a horizontal face. |
| `lineOnTop(a, b, z, P)` | One segment on a horizontal face. |
| `planOutline(plan, z, P, steps)` | A rounded-rectangle outline on a face, used for grooves and inlays (`insetPlan`). |
| `ring(cx, cy, r, z, P, steps)` | A circle on a face, used for screw rings, bezels and seats. |
| `segment(a, b, P)` | Any 3D segment. |
| `rise(at, z0, z1, P)` | A vertical segment. |
| `pointsAt(points, P)` | Projects a list of points, for screw and rivet dots. |
| `dotGrid(x, y, cols, rows, gap, z, P)` | Projected perforations. |
| `topMatrix(at, z, P, along = "x")` | A transform that lays text flat on a top face, for `faceTextSvg`/`FaceText`. |
| `sideMatrix(at, z, P, "left" \| "right")` | The same, on a vertical face. |

## 5. Parts library
| Function | Returns |
|---|---|
| `knurl(cx, cy, r, z0, z1, count, P, inset = 0.4)` | Vertical lines on the visible half of a cylinder. |
| `sideArc(cx, cy, r, z, P, steps)` | The visible front half of a horizontal circle on a cylinder side: seams and bands. |
| `sideSeam(plan, z, P, steps = 8)` | The same for a box: the visible near half of the rounded plan at height `z`, wrapping its corners and ending exactly on the slab's vertical silhouette edges. Pass the slab's own `steps`, and draw it as the slab's `inner`. It is the way to draw a weld bead, a band, a lid seam or a trim line on a box: never a `segment` inset from the corners. |
| `ringSeam(ring, z, P)` | `sideSeam` for any floor outline (the ring you gave `extrude`). |
| `sideRing(center: Point3, r, "x" \| "y", P)` | A circle on a vertical face: bores, rod ends, lenses. |
| `coil(from, to, [cy, cz], r, turns, "x" \| "y", P)` | `{ back, front }` spring halves, so you can paint back → rod → front. |
| `topTicks(from, to, step, every, edge, z, [minor, major], P, axis)` | `{ minor, major }` ruler ticks on a top face, growing from the line `edge` in +y (or +x). |
| `sideTicks(from, to, step, every, face, top, [minor, major], P, axis)` | Ticks hanging down a vertical face. |
| `radialTicks(cx, cy, r, count, every, z, [minor, major], P, from = 0, span = 2π)` | Dial and turntable ticks. Angle 0 points to −y, increasing clockwise as seen from above. |

## 6. Moving things
| Function | What it does |
|---|---|
| `translateAlong(P, [dx, dy, dz])` | A `"translate(x y)"` string that moves a pre-built group by a world offset. Use it for carriages, drawers, plungers and lifts. |
| `holdStill(element, [dx, dy])` | Gives every `[data-hold]` element inside `element` the opposite translate, so a clip stays where it is while the group it sits in moves by `[dx, dy]` screen units. Call it with the same offset whenever you move an item that holds a `G.rodOut` clip. |
| `axisVector(P, [1, 0, 0])` | The screen vector of one world unit. Use it to turn pointer movement into world movement. |

Rotations about z can't be done with a screen transform: rebuild the rotating paths for each frame or step. That is cheap for a needle, a pointer or a small part.

## 7. SVG strings (framework-free)
| Function | Notes |
|---|---|
| `solidSvg(paths, { tone = "hi", crease = "faint", halo, lit, flat, className, inner })` | When `paths.seamless` is set (smooth lathe shading, `sphereOf`), the group gets `data-seamless` and `ISO_CSS` strokes each tone region in its own colour, 0.4 wide, so the many abutting tone sheets show no hairline seams. `inner` is extra line markup placed just before the outline (`iso-edge`), for seams, weld lines and band lines that must sit under the silhouette stroke; the React `Solid` takes it as a prop. |
| `lineSvg(d, { tone = "lo", dotted, dashed, flow, className, free })` | `flow` animates dashes; use it for things moving along a path. `free: true` (or `"start"`, `"end"`) writes `data-free`, telling `scripts/lines.mjs` that an end is meant to stop on a bare face: rule lines that stand for printed text, a needle's tail, a pointer's tip. Every other line end must land on something. |
| `dotsSvg(points, { size = 0.5, tone = "mid", pulse })` | A point may carry a third value, its opacity (`[x, y, 0.4]`), which is how `dotsOf(..., { fade })` fades a bolt circle out instead of stopping it. |
| `fadedSvg(list, style)` | Draws a set of lines that each carry an opacity, `[{ d, alpha }]` or `[d, alpha]`, as returned by `ribsOf(..., { fade })`. Lines are grouped by opacity in steps of 0.05, each group in one `<g opacity>`, and lines under 0.02 are dropped. Use it for any texture whose members fade one by one toward a limb: ribs, fins, knurls, rule lines. |
| `fadeLineSvg(points, { fade = [start, end], tone = "lo", className, stops = 6 })` | One line (screen points) that fades out over its first `start` and last `end` px instead of stopping (each ramp carries `data-free` on its faded end, so the line check passes it). Each ramp is stroked with a `userSpaceOnUse` linear gradient along its own chord (a smoothstep in `stops` steps, the colour taken from `var(--anatomy-<tone>)`, so both themes work), and the middle is a plain `lineSvg`, so there are no overlapping caps, beads or seams. Use it for a texture line that ends in the middle of a face: a groove that runs out, a weld line that stops short of a corner, a rule on a curved plate. Keep a ramp shorter than a quarter turn of the line, since a linear gradient follows the chord. Gradient ids are a hash of the path, so the same line drawn twice shares one. |
| `wireSvg(from, to, { via = [], tone = "lo", dotted, dashed, flow, ends: "both" \| "from" \| "to" \| "none", size = 1.2 })` | A connector with terminal dots. |
| `faceTextSvg(transform, text, { size = 3, tone = "mid", anchor })` | |
| `groupSvg(children, { class, transform, "data-part": … })` | |
| `figureSvg({ width, height, label, body, standalone: false \| "dark" \| "light", className })` | The `<svg>`. `label` is the accessible description: say what the object is and what it shows. `className` is added to its class, which helps when you stack it under a canvas (`rt-back`, `rp-back`). |

## 8. Card and page
| Function | Notes |
|---|---|
| `plateHtml({ fig, title, hint, readout, keys: [{ mark, label }], caption, body })` | The card. `mark` is one of `raised`, `flat`, `lit`, `edge`, `dim`, `dotted`, `dashed`, `red`, `green`, `blue`. The readout element carries `data-readout`, so live code writes to it. |
| `pageHtml({ title, theme = "dark", body, script, width = 650 })` | A standalone document with `ISO_CSS` inlined. |
| `kitScript()` | From `scripts/inline-kit.mjs`. Prepend it to `script` so the page can call kit functions such as `lineOnTop` and `translateAlong` while animating. |
| `glScript()` | From `scripts/inline-kit.mjs`. The source of `kit/gl.mjs` (`GLSL_ISO`, `glCamera`, `glLayer`, `smooth`, `burst`, `settle`), to inline after `kitScript()` on a page with a shader: `script: kitScript() + glScript() + live`. It doesn't depend on the kit, so a page whose live code calls no kit function can inline it alone (the Raptor uses `glScript() + LIVE`). Everything is one module script, so live code must not redeclare `smooth`, `burst` or `settle`: a second `const smooth` is a SyntaxError that kills the page. |
| `turnScript()` | From `scripts/inline-kit.mjs`. The 3D runtime (`TURN`, `TURN_CSS`) with `export` stripped, for a page with live parts: `script: turnScript() + glScript() + live`. Its internals are prefixed `t3`/`T3_`, so it inlines beside the kit and the WebGL layer. |
| `ISO_CSS` | All the styles. `kit/iso.css` is the same text. |

## 9. React components
From `kit/react/draw.tsx`, which is server-safe:

- `<Solid paths tone crease halo lit flat className inner />` (`inner` is line markup placed under the outline, as for `solidSvg`)
- `<Line d tone dotted dashed flow free />`
- `<Wire from to via tone dotted dashed flow ends size />`
- `<Dots points size tone pulse />`
- `<FaceText transform size tone anchor>text</FaceText>`
- `<IsoFigure width height label>` …svg children… `</IsoFigure>`
- `<Plate fig title hint readout keys caption>` …figure… `</Plate>`

From `kit/react/live.tsx`, which is `"use client"`:

- `useReadout()` returns `{ anchor, say }`. Put `anchor` on your root element; `say(text)` writes the card's readout.
- `useLoop(tick: (dt, calm) => keepGoing, running)` runs a frame loop while `running` and while `tick` returns true. It returns `kick()` to restart it.
- `follow(current, target, dt, seconds, calm)` is exponential easing.
- `spring(state, target, dt, stiffness, damping, calm)` is a sub-stepped spring. It returns true when settled.
- `useEased(target, seconds)` returns an eased number as React state. Fine for small figures.
- `useInView(ref)`, `useStillness()`, `asksForStillness()`.
- `Stage` is the wrapper `IsoFigure` uses. It adds `data-inview` for CSS animations.

## 10. Round parts on any axis (`kit/lathe.mjs`)
`kit/iso-kit.mjs` builds prisms that stand on the floor. `kit/lathe.mjs` turns a profile around **any** axis, for bells, chambers, flanges, valve bodies, pump volutes, pipe fittings, actuators lying at an angle, and connectors on a vertical face. It returns the same `SolidPaths` as `slabOf`, so `solidSvg` paints them. Import it as `import * as G from "../../kit/lathe.mjs"`.

**Frames.** A frame is `{ o, a, u, v }`: an origin, the unit axis, and two unit radial directions. A point on the part is `(s, r, angle)`: `s` along the axis from `o`, `r` from the axis, and `angle` from `u` toward `v`.

| Function | What it does |
|---|---|
| `frameAlong(o, a)` | A frame on axis `a` through `o`, with `u` and `v` chosen for you. Use it for pipes, bosses and flanges at any angle. |
| `frameOf(o, a, u, v)` | A frame with explicit radial directions, so angles mean something: `G.frameOf([0, 0, z], [0, 0, 1], [1, 0, 0], [0, 1, 0])` makes angle 0 point along +x. |
| `pointOf(F, s, r, angle)` | The world point. |
| `radialOf(F, angle)`, `tangentOf(F, angle)`, `angleToward(F, direction)` | Directions in the frame, and the angle that faces a direction. |

**Solids.** A meridian is a list of `[s, r]` points along the axis. Add a third element (`[s, r, 1]`) to mark a smooth ring, so no crease line is drawn there. Use that for curved profiles such as a bell or a dome. A meridian with any smooth vertex is shaded smoothly on its own (`smooth` defaults to `"auto"`, which is `softOf(meridian)`); pass `smooth: false` to opt out, `true` or `{ rows }` to force it on a profile without smooth vertices, and `{ slope }` or `{ slope, radius }` for an analytic curve.

**Smooth shading.** Without it every facet row gets one flat tone, and a curved profile shows a staircase of tone blocks (the first Raptor bell and throat did, and the reviewer saw it at once). With `smooth` the side of each meridian segment is shaded from an interpolated normal: the tone boundaries become the true iso-light curves, sampled at `rows` levels and emitted as one polygon per tone region, so they run as clean curves across the whole part.
- `smooth: true` interpolates each run of smooth vertices (`[s, r, 1]`) with a monotone cubic Hermite curve (Fritsch–Carlson slopes: zero where the profile turns, a weighted harmonic mean elsewhere), and uses that one curve for the normals, the tone sheets, the silhouette and the hull, resampled until it is within 0.2 px of the chord. Shading and outline then describe the same shape, and the normal can only change sign where the profile does. Use it for any profile with smooth vertices: domes, throats, cones, valve bulges, a flame cone. (A natural spline overshoots near a peak; on a valve bulge it zig-zagged the tone bands across the body.)
- `smooth: { slope }` takes the normal from an analytic `dr/ds`, so a body cut into bands shares one normal field and the tone curves cross the band joints without a step. With a `slope` alone the profile points are joined straight, so sample the analytic `r(s)` densely (20–30 knots on a throat), and the silhouette of a waisted body uses the same `slope`, so the outline follows the true curve too.
- `smooth: { slope, radius }` also takes the surface from an analytic `r(s)`: the tone sheets, the hull and the silhouette all follow `radius` between the knots, so two knots are enough for a band. This is what `bandsOf` passes to every band.
- **Hand-typed knots are not a curve.** The monotone Hermite is only C1: its `dr/ds` is a piecewise quadratic that plateaus at every knot and overshoots between them, so tone bands stall and then lurch, a staircase in scallops. The Raptor throat (`[128, 31, 1], [134, 27.6, 1], …`) showed it at 5×. Any curved body larger than a fitting gets an analytic profile and `smooth: { slope }`: the throat is `r = √(24.2² + (c·(s − 146))²)`, the flame cone `r = 48·(1 − x)^0.8` closed to `r = 0` at the tip, the valve bulge an ellipse, the gimbal puck `18.4 − a·|s − 297.5|⁴`. The audit's `wobble` lists every profile smoothed without a `slope` whose `dr/ds` turns between its ends.
- `rows` (default 6) is the starting sampling along each segment. Rows are then split until every tone boundary is within `error` (0.35 px) of a straight chord, and wherever the number of bands changes down to `depth` (8) levels or to `fineness` (0.1 px) on screen, whichever comes first, so a band that appears or disappears is born at zero width in a sliver instead of ending on a straight cut. Each tone is then emitted as one polygon for as long as its own number of regions holds, across every meridian segment and smooth knot of the body, and only that tone is bridged where its count changes; a band that comes and goes no longer cuts the other tones into rows. That made a small turned ball about half the size and removed the row seams. Visibility uses the drawn facet's normal, so the tones always reach the outline; every tone polygon is wound the same way, so overlapping pieces of one tone never cancel into a hole.
- The result is marked `seamless` (see `solidSvg`). Flat caps (rings where `s` doesn't change) keep their facet shading.

| Function | What it does |
|---|---|
| `lathe(meridian, F, P, { steps = stepsFor(rMax, 32, 160, scaleOf(P)), fill = "auto", hole = null, bevel = 0, rims = "all", smooth = "auto", waist = 0.5 })` | The general solid of revolution. Facets are shaded in four tones by the kit's light and bucketed like `extrude`. `fill: "auto"` uses the hull of the rings unless the true silhouette lies more than `waist` px inside it (a throat, a spool), and then fills and outlines the body with `silhouettePoints` on both sides joined by the visible end arcs (only when the silhouette is unbroken along the whole side: a body whose surface faces you head-on somewhere, like a valve bulge that leaves its neck at a right angle, falls back to the hull, because joining the broken pieces drew stray triangles across it); `"surface"` forces that, `"hull"` forces the hull; `fill: "facets"` fills only the visible facets and draws no outline, which you need for a concave or open shape seen from inside. `hole: [s, r]` cuts a round hole into the fill and outline, for the end of a tube. `bevel` adds a highlight inset from a closed cap (a meridian end at `r = 0`). `rims` keeps the outline on the end rings: `"all"`, `"none"`, `"first"`, `"last"` or `"ends"`. Only hull edges along the first or last ring are dropped; edges along interior rings always stay, so the side silhouette stays whole. |
| `solidOf(profile, F, P, options)` | Closes a profile to the axis at both ends and picks `steps` from the largest radius. It is the usual way to make a turned part: `G.solidOf([[0, 12], [6, 12], [8, 9, 1], [20, 9]], F, P)`. |
| `bandOfProfile(profile, F, P, options)` | An open band with no caps, for one stretch of a long body that is split into depth-sorted pieces. Prefer `bandsOf`, which cuts the whole body at once. |
| `bandsOf(profile, cuts, F, P, { slope, radius, steps, rows = 6, rims = "none", fill = "auto", samples = 72 })` | One curved body (`s` rising along `profile`) cut at the `s` values in `cuts` into depth-sortable bands that share one normal field, one surface and one facet count. Pass the analytic `slope` and `radius`, or neither: then `profileCurve(profile)` makes them from the profile (the monotone Hermite through each run of smooth knots, straight elsewhere), so no band ever has its own end slopes. Returns `{ bands, sidesOf, radius, slope, steps, s0, s1 }`. Each band is `{ s0, s1, profile, paths, outline, seam }`: `paths` is ready for `solidSvg` with its outline cleared, `outline` is the band's own chord outline if you need it (the last band of the Raptor skirt keeps it), `seam` is the visible arc where it meets the next band (put it in `inner`), and `profile` is what to record for the audit (`A.solid(F, band.profile, { radius })`). `sidesOf(from = s0, to = s1, { ends = false \| true \| "first" \| "last", count = samples })` returns `{ d, points }`: the body's two analytic side edges over that span as one path, optionally with the visible rim arcs at its ends, plus their 3D points to record (`points.map((list) => A.tube(list, 0.4 / scale))`). Draw `d` as one item with a small depth bias over the bands, so the outline is one continuous curve. |
| `profileCurve(profile, { slope, radius })` | `{ s0, s1, radius(s), slope(s) }` for a profile whose `s` rises: the curve `smooth` draws through it. Throws if `s` doesn't rise, since a step in a profile is a crease and belongs to a separate solid. |
| `outlineOf(meridian, F, P, { steps, fill, hole, rims, smooth, waist })` | Only the `fill` and `outline` of `lathe`, without the shading, for a body whose outline you draw separately. |
| `softOf(meridian)` | True when any vertex is marked smooth. |
| `disc(s0, s1, r, F, P, options)` | A short cylinder on the axis: pucks, flanges, bosses, pins, plugs. |
| `ringBand(s0, s1, rIn, rOut, F, P, options)` | An annulus with thickness: a flange or collar around something. |
| `stepsFor(radius, least = 16, most = 112, scale = 0)` | A facet count. With `scale` (screen px per unit, `scaleOf(P)`) it keeps every facet about 3 px long on screen; `disc`, `ringBand`, `solidOf`, `bandOfProfile` and `lathe` use that (32–160) when you pass no `steps`. Facets only set the tone fills: outlines, rims, creases and bevels are sampled from the true circle at about 1.5 px and end exactly where the faces turn away. |
| `saddleOf(F, a, start, s1, P, { steps })` | A saddle pad: a cylinder of radius `a` along `F.a` from a curved host surface to a flat face at `s1`. `start(angle)` is where each generator leaves the host (find it by bisecting the host's `sdf`), or a number `R` for a host cylinder perpendicular to `F.a` (then `saddleStart(R, a, angle)` gives each generator's start). The fill is the union of its visible facets and face, the outline its two side generators, the visible part of the saddle curve and the face rim, so it meets the host along the true intersection. Seat flanges on it wherever a flat flange on a curved body would leave a rim gap. |
| `prismOf(F, polygon, s0, s1, P)` and `stadium(r0, r1, offset)` | A prism of any convex `[x, y]` polygon in the frame's `u`, `v` plane, extruded along `F.a`, with facet tones, the hull as fill and outline, and creases only at sharp corners. `stadium` is the hull of two circles, for a gearbox plate joining a motor to an actuator. |
| `sphereOf(centre, r, P, { steps = 48 })` | A ball with exact smooth tones: each tone region is the projected hull of a spherical cap, so it costs four small polygons instead of a turned profile. Use it for rod-end bearings, tee fittings and knobs. It is `seamless`. |
| `silhouettePoints(profile, F, P, { samples = 72, slope, radius })` | The two side edges of `silhouetteOf` as 3D point lists, to record the outline's shape for the audit. |
| `silhouetteOf(profile, F, P, { samples = 72, slope, radius })` | The two analytic side edges of the surface of revolution through `profile` (`[s, r]` points, linearly interpolated), as one path of two polylines: the points where the surface turns away from the viewer (`cos(θ − φ) = r′·aV/|V⊥|`). Use it to draw one continuous outline over a body built in bands. Where the surface faces the viewer head-on (`|c| > 1`) a sample is skipped. Pass the analytic `slope` (the same one the bands are shaded with) so the edge doesn't kink at every profile point. |

**Lines and dots on round surfaces.** These draw only what faces the viewer. `slope` is `dr/ds` of the surface at that ring, so a line on a cone or bell hides where the real surface turns away.

| Function | What it does |
|---|---|
| `arcOf(F, s, r, P, { slope = 0, steps, least = 0.02, inward = false, from = 0, to = 2π, trim = 0.3 })` | The visible arc of a ring on the surface: seams, bands, weld lines, a lip. The limb angles are solved exactly (`t = φ ± acos(c)`), and each end that stops at the limb is shortened by `trim` px so its round cap stays inside the outline stroke. `inward: true` draws the arc you see on an inner surface, such as the inside of a nozzle. `least: -1` gives the whole ring. Draw seams *under* the outline: `solidSvg(paths, { inner: lineSvg(arcOf(…), { tone: "faint" }) })`. Appended after the solid, a faint seam cap paints over the edge stroke and notches the silhouette at every seam. |
| `circleOf(F, s, r, P, steps)` | A full closed circle, such as a face ring, a bore or a void you fill. |
| `ribsOf(F, s0, s1, r, count, P, { phase, twist, least = 0.12, r1 = r, fade, seams })` | Visible axial lines: a knurl, ribs or fins along any axis. `twist` makes them helical and `r1` lets them run along a cone. With `fade: [lo, hi]` it returns `[{ d, alpha }]`, each rib's opacity a smoothstep of how much it faces you, so the set fades out toward the turn instead of stopping at `least`. Draw them with `fadedSvg`. `seams: true` (or `"first"`, `"last"`) adds the seam arcs at `s0` and `s1` to the result (at full opacity), so a rib row and the seams it ends on always share one `s`; leave it off only where the ribs end on a step or a band end that is already drawn. |
| `rodOut(F, from, to, r, P, { stroke, ...latheOptions })` | A rod that slides out of a gland at `s = from` (`F.a` points out of it), its tip at `to` at rest, travelling up to `stroke` further out, or the same rod retracting into it. Returns `{ paths, clip, from, to, stroke, reach }`: `paths` is one rod from `from − stroke` to `to`, and `clip` a path covering the rod from the gland face to `to + stroke`. Draw `<clipPath id><path data-hold d=clip/></clipPath><g clip-path=url(#id)>solidSvg(paths)</g>`, move the item, call `holdStill`, and record `A.slide(F, from, to, r)` (`craft.md`, section 3). |
| `spokesOf(F, s, r0, r1, count, P, { phase, sweep, from, to })` | Radial lines on a face, for vanes and turbine blades. |
| `dotsOf(F, s, r, count, P, { phase, slope, least = 0.08, all = false, fade })` | Screen points of a bolt circle on the visible side. Pass them to `dotsSvg`. Use `all: true` for a face you see head-on. A `least` well above 0 stops the circle in the middle of the surface; pass `fade: [0.04, 0.42]` instead and the dots carry a falling opacity. |
| `fadeOf(value, [lo, hi])` | The smoothstep both fades use. |
| `facingOf(F, angle, P, slope)` | How much a surface at that angle faces the viewer. Positive means visible. |

**Paths in 3D**, for pipes and cables.

| Function | What it does |
|---|---|
| `fillet(points, radius, perArc = 7, { P, tube, margin = 1.3 })` | Rounds every corner of a polyline with a bend of `radius`, clamped to the room available (0.48 of the shorter leg). Lay out a pipe as a few straight legs, then fillet it. With `P` and `tube` (the pipe radius) each corner's bend grows to `margin · tube / (0.7 · f²)`, where `f = |n · V|` is how squarely the bend's plane faces the viewer: a bend seen edge-on projects to a radius `R·f²`, and below the tube's own radius the outline folds back on itself. The legs still cap it, so a turn next to a port facing you needs long legs or an elbow fitting. |
| `resample(points, spacing)`, `pathLength(points)`, `cut(points, from, to)` | Even spacing, total length, and a sub-path by arc length. |
| `pointAlong(points, at)`, `tangentAlong(points, at)` | Position and direction at an arc length. Use them to place flanges, bands, valves and clamps on a route. |
| `spiral(F, s, r0, r1, a0, a1, steps = 28)` | 3D points of a widening spiral, such as a pump volute. Feed them to `tubePieces` with a radius function. |
| `arcPoints(F, s, r, a0, a1, steps = 24)` | 3D points around a ring, such as a manifold, a deluge ring or a harness lane. |

**Camera and vectors.**
- `viewOf(P)` is the unit direction to the viewer, so `dot3(p, viewOf(P))` equals `depthOf(p, P)`.
- `lightOf(P)` is the kit's light.
- `scaleOf(P)` is the screen units per world unit.
- `toneOf(normal, P)` and `bandOf(score)` give the four-tone bucket.
- `hullOf(points)` and `runsOf(flags)` are the hull and run helpers the module uses itself.
- The vector helpers are `dot3`, `add3`, `sub3`, `mul3`, `len3`, `unit3`, `cross3` and `lerp3`.

**A long curved body in bands.** Cut it with `bandsOf` so each band depth-sorts against the pipes that wrap it. Every band shares the body's `steps`, its normal field and its surface, so tone curves cross every joint without a step, and the outline comes from `sidesOf` as one curve instead of a chord per band that kinks at every joint. The Raptor bell:
```js
const BELL = G.bandsOf([[3.2, bell(3.2)], [116, bell(116)]], [14, 28, 44, 60, 76, 92, 106], ENGINE, P, { radius: bell, slope: bellSlope, steps: 96, rows: 8 });
BELL.bands.forEach((band, index) => put("nozzle", ex(ENGINE, (band.s0 + band.s1) / 2), S(band.paths, { tone: "hi", inner: index === 2 ? Ln(band.seam, { tone: "faint" }) : "" }), { shapes: [record(A.solid(ENGINE, band.profile, { radius: bell }))] }));
const SIDES = BELL.sidesOf(3.2, 106, { ends: "first" });
put("nozzle", ex(ENGINE, 105), Ln(SIDES.d, { tone: "hi" }), { bias: 0.2, shapes: SIDES.points.map((points) => A.tube(points, 0.4 / KS)) });
```
Don't build bands by hand from separate `bandOfProfile` calls on two-knot chords with their own Hermite fits: each band then has its own end slopes, its tones step at every joint, and its chord outline kinks.

**Known traps.**
- **Four-tone quantisation follows the slope of each facet row** when a body isn't smoothed. Mark every curved knot smooth (`[s, r, 1]`) and the shading follows on its own; a profile of hard knots that still describes a curve needs `smooth: true`. Use `bandsOf` on a body cut into bands.
- **Smooth shading is heavier.** A small turned ball with `smooth` costs about 9 KB; `sphereOf` costs under 3 KB. Use `sphereOf` for balls and keep `rows` low on small parts.

## 11. Pipes, tubes and cables (`kit/tube.mjs`)
```js
import { tubePieces, tubeSvg, runsOf, crossingsOf } from "../../kit/tube.mjs";
```
| Function | What it does |
|---|---|
| `tubePieces(points3, radius, P, { maxLength = 30, spacing = 2.2, closed = false, breaks = [], gaps = [], caps = true, stripes = "auto" })` | Splits a 3D route into chunks of at most `maxLength` world units. Each one is drawn as a 2D ribbon with a paper body, a shine band toward the light, a shade band away from it, and two edge lines. Neighbouring pieces overlap by 0.5 units, and the edge lines are cut from the same overlapped span as the body, so neither a seam nor a notch in the outline shows at a joint. The shine and shade run 0.5 further than the body at every open end, so a stripe covers the anti-aliased cut of the body under it and its own cut lies on the neighbour's identical stripe; every frame normal is a central difference along the whole route, so the stripes of two chunks coincide exactly where they overlap (with both cut on one line, a hairline crossed the volute's shade band at every joint). `radius` is a number or a function of `t` (0 to 1 along the route), for tapers and volutes. `breaks` are arc lengths where a chunk must end (at a crossing, or where the pipe turns back past a part). `gaps` are `[from, to]` arc-length spans that are not drawn because a collar, clamp, valve or fitting covers them; the pipe ends flat on each side of a gap. Every end next to a gap gets an elliptical cap (the projected cross-section, with its outer half-arc as an edge), and so do the route's own two ends unless you pass `caps: false` (or `[start, end]`, such as `[false, true]` for a branch whose start hides inside a tee collar), so a pipe meets a flange face along the right curve instead of a straight cut. The shine and shade stripes follow the cap: onto the outward half of the ellipse when the end face is turned away, and up to its inner rim (drawn as an edge) when the face is turned toward you, so no stripe stops short of the end. A closed route with no gaps wraps its first and last pieces round each other. Returns `[{ mid, body, shine, shade, edges, wide, points, radii, ends, from, to }]`: `mid` is the chunk's centre in world space, to depth-sort it; `wide` is true when the route is wide enough on screen for its shine and shade bands (decided once for the whole route from its widest point, so a taper never drops its stripes at one chunk joint; `stripes: true` or `false` overrides it); `points` and `radii` are the chunk's 3D centreline and radii (record them for the audit); `ends` says which ends are real ends rather than joints with a neighbour. |
| `tubeSvg(piece, { tone = "hi" })` | One piece as `<g class="tb">`. Shine and shade appear only on `wide` pieces, then the piece's rings (`.tb-ring`), then the edges. |
| `tubePieces(…, { rings: { pitch = 3, twist = 0, cross = false, fade = [0.1, 0.55], phase = 0, steps = 16, tone = "lo" } })` | A texture that follows the pipe's cross-section: a braid, corrugation, tape wrap or armour. Stations sit every `pitch` units of 3D arc length from the route's start (`phase` shifts them), so the pattern is continuous through every chunk joint; each is drawn as the visible front arc of the cross-section there, split into short segments whose opacity is `fadeOf(facing, fade)`, so the texture fades toward both limbs instead of ending on a hard line. `twist` (in radii) turns each ring into a helix that advances that far from limb to limb, and `cross: true` adds the other hand, for a braid. Rings run over each chunk's stripe overhang, so the later chunk at a joint draws an identical piece over its own body's cut, and stop on real ends and gap faces (rings keep a third of a pitch off an end face, helices end on it). Each piece carries `rings: [{ d, alpha }]`; `ringsSvg(rings, tone)` draws them alone. |
| `runsOf(total, gaps)` | The drawn `[from, to]` spans of a route of length `total` once `gaps` are taken out. Record each run as its own tube for the audit. |
| `crossingsOf(points, centre, axis, radius)` | Arc lengths where a route crosses the disc of a clamp or collar (the plane through `centre` normal to `axis`, within `radius`). Turn each into a gap of the clamp's thickness. |

The fills for `.tb-body`, `.tb-shine` and `.tb-shade` are part of `ISO_CSS` (and `kit/iso.css`), so `tubeSvg` works on any kit page with no extra CSS. Override them in page CSS only to restyle tubes, as the Raptor does for selected parts (`.it[data-on] .tb-body`).

**Depth-sorting tubes.** A pipe runs between parts at many depths, so draw each piece at its own depth: put it in your item list with key `G.dot3(piece.mid, G.viewOf(P))`, next to the solids it passes. Use short pieces (`maxLength` 6–18) where a pipe weaves between neighbours, and long ones (400) for a ring that lies flat on the floor. To split a pipe between SVG layers, test `piece.mid`. The ripple tank's motor cable goes three ways: pieces over the bridge (`piece.mid[2] > Z.top - 1 && piece.mid[1] < IN.y + IN.d - 10`) to the front SVG before the bar, the rest above the rim to the front SVG after the near caps, and the run down the leg to the control box to the middle SVG. Its lamp cable sends the pieces that hang over the base edge (`piece.mid[1] < BASE.y - 0.2`) to the back SVG. In a figure as dense as the Raptor, don't tune these keys by eye: record the pieces and let `orderKeys` (section 12) settle them.

**Anything that wraps a pipe splits it.** A collar, band, bellows disc, clamp or valve body is wider than the pipe it sits on. If the pipe runs through it as one chunk, the chunk is both behind and in front of the collar and no key can order them: the collar either hides the pipe's near side or the pipe paints over the collar's face. Gap the pipe at the collar (`gaps: [[at - t/2, at + t/2]]`): the far piece ends at the collar's back face, the near piece starts at its front face, and both order cleanly. The Raptor's `pipe()` does this for its own end flanges, bands, bellows and valve, and `cable()` for the harness clamps via `crossingsOf`.

The Raptor's `pipe()` helper is a good pattern:
1. `fillet` the control points with a bend of 2.6 × r (at least 3). A valve or bellows needs a straight leg at least as long as its body plus both fillet cuts (`r_bend · tan(turn / 2)` each); put it on a bend and its ends cut into the pipe.
2. Work out the collars first (end flanges, bands, bellows discs, the valve body) and pass their spans as `gaps`.
3. `tubePieces` the route, recording each chunk with its `points` and `radii`, and each drawn run as a tube for the audit.
4. Draw the collars, owned by the pipe (`owner`), so the audit doesn't count a pipe's own flange as a collision.

`cable()` adds a terminal dot at an end that stays with its own assembly and a connector shell (`shells: [start, end]`) at an end that lands on another one, so every cable still ends on something when the figure comes apart. Both helpers take `free: [start, end]` for an end that comes off its mount when the figure is apart, and fillet with `{ P, tube: r }` (see `fillet`). A line that must leave a port facing the viewer starts from an `elbow` (a ball on the boss, rim on sphere at both joints, the line leaving it toward its next point), so the turn happens inside a fitting. `pipe()` puts each end flange's back face on the route end and can end a line in a `coupling` (a union nut at a quick-disconnect) instead of a flange.

## 12. The audit (`kit/audit.mjs`)
```js
import * as A from "../../kit/audit.mjs";
```
**The one-line hook.** Make a recorder, let every drawing helper put its items and record its shapes into it, settle the keys, and end the build with `auditOrExit`:
```js
const R = A.recorder(P, { order: STAGGER, spring: SPRING });
R.put({ piece: "base", at: centre, bias: 0.1, svg, shapes: [R.solid(A.slab(plan, 0, 6, { name: "base.plate" }))] });
R.route("base.cable", route, 0.6, { piece: "base", gaps, limp: true });
A.settle(R, P);
R.items.sort((a, b) => a.key - b.key);
A.auditOrExit(R, P, { crowd: [] });
```
`node build.mjs --audit` then prints the report and exits non-zero on any failing line; without the flag `auditOrExit` returns at once. A figure that doesn't come apart passes no `order` and every explode check reduces to the rest pose.

| Function | What it does |
|---|---|
| `recorder(P, { order = [], spring = {}, ground = grounded })` | `{ items, solids, tubes, routes, put, solid, route, nameFor }`. `put({ at, bias, key, name, piece, move, ride, shapes, ...rest })` adds a drawn item (its key is `dot3(at, V) + bias` unless you pass `key`; the rest, such as `svg`, `route` and `chunk`, is kept on the item) and stamps its shapes with the item's `move`, `ride` and `piece`. `solid(shape)` records a shape and returns it, for the item's `shapes` or for static parts drawn outside the sorted layer. `route(name, points, radius, { owner, touch, bundle, limp, gaps, closed, ends, rims, free, piece, move, ride })` records a pipe or cable: one tube per drawn run between `gaps`, with its `tips`, `tipAxes`, `tipRims` and `free` ends, plus the route for `folds` and `crowding`. `nameFor(base)` numbers repeated names (`base`, `base#2`, …). `order` and `spring` are the live page's stagger and spring. |
| `explodeStates(order, spring, { uniform = 10, stride = 4 })` | `{ depth, frames, apart }`: the depth-solver states (uniform shares, every `stride`th replayed frame with reversals at every stagger boundary, prefix states) and the clearance frames (every replayed frame plus the prefix states). `{ depth: [0], frames: [] }` when nothing comes apart. |
| `settle(R, P, { states, skip = sameRun, ...overlapOptions })` | Runs `overlapsOf` and `orderKeys` and writes the keys onto the items. It keeps the order the items had, so you may sort `R.items` afterwards. Run it on every build. |
| `audit(R, P, { allow, crowd, cover, ground, frames, apart })` | Every check below on the recorded shapes, as one result for `report` and `failures`: clearance at rest, apart and in every frame, solid clearance, crowding at rest and apart (against slender solids too), stand-in shapes, terminals, loose ends, folds, self-overlaps, wobble, seats, supports and depth order. `allow` excuses depth pairs, `crowd` crowding pairs, `cover` stand-in items. It never sees line markup: run `scripts/lines.mjs` on the page for that. |
| `auditOrExit(R, P, { flag = "--audit", argv, ...auditOptions })` | With the flag: `audit`, print the report and a summary line, and set `process.exitCode = 1` on any failure. |
| `sameRun(a, b)`, `grounded(shape)` | Neighbouring chunks of one route (skipped by the overlap pass), and the default ground (a shape with `ground: true` in its meta or a name starting `stand.`). |
| `keyAt(item, state, V)`, `labelOf(state)`, `cameraBasis(P)` | An item's sort key in an explode state (`key + lift · share`), a state's printable label, and the camera's screen-right, screen-down and toward-viewer unit vectors `{ R, D, V }`, for probes of your own. |

A detailed figure fails in ways the eye misses until a reviewer zooms in: a pipe or cable that passes through something, at rest or in any frame of the explode, a flange that floats off its mount, a part that hangs in the air or is sunk into its neighbour, a route that doubles back or pinches to a point on screen, a bend or end that lands on another line's outline, a cable end that dangles when the figure comes apart, and two things drawn in the wrong order where they overlap. The audit measures all of them from the real geometry. On the Raptor (478 items, 1,119 depth states, about 4,400 replayed and prefix frames for clearance) the keys take about a minute on every build and a full `--audit` about three minutes in all, most of it the per-frame clearance.

**Shapes.** Record a shape for everything you draw, in world units, as you build it:

| Constructor | Shape |
|---|---|
| `solid(F, profile, { radius, ...meta })` | A turned part: the profile closed to the axis at both ends, the same `[s, r]` list you give `solidOf`. Runs of smooth knots are traced along the same Hermite curve the drawing uses (or along `radius(s)`, as for a `bandsOf` band), so the recorded surface is the drawn one, not its chords. |
| `lathe(F, meridian, { radius, ...meta })` | The shape of `G.lathe(meridian, F, P)` with the same tracing: any closed meridian (rings, valve bodies, cans). |
| `body(F, polygon, meta)` | Any closed `[s, r]` polygon turned round `F`, taken as it is. Edges that lie on the axis are not surface, so a point deep inside a disc measures its true depth. |
| `disc(F, s0, s1, r, meta)`, `ring(F, s0, s1, rIn, rOut, meta)` | Flanges, bosses, pins, collars. |
| `box(min, max, meta)`, `orientedBox(centre, axes, half, meta)` | Square blocks; a beam at 45°. |
| `slab(plan, z, height, meta, steps = 8)`, `extruded(ring, z, height, meta)`, `cylinder(cx, cy, r, z, height, meta)` | The shapes of `slabOf`, `extrude` and `cylinder`: the same rounded plan or any polygon ring (convex or not) stood on the floor, so a rounded deck corner or an L-shaped bracket is checked as drawn. |
| `prism(F, polygon, s0, s1, meta)` | The shape of `G.prismOf`: any polygon in the frame's `u`, `v` plane extruded along `F.a`. |
| `traced(meridian, { radius, error = 0.02, depth = 6 })` | The polygon `solid` and `lathe` record. |
| `ball(at, r, meta)` | Spheres and terminal dots (`size / scale` world units). `meta.flats: [[nx, ny, nz, d]]` cuts flats (the half-space n·(p − c) ≤ d is kept): `sdf` is `max(\|p − c\| − r, n·(p − c) − d …)`, surface samples include the flats' rims, `boundsOf` is unchanged (conservative). The 3D mode's knuckle balls record this. |
| `tube(points, radius, meta)` | A pipe or cable centreline with a radius (number, function of 0..1, or a list per point). |
| `slide(F, from, to, r, meta)` | A rod sliding out of a gland at `from` (a `disc` from `from` to `to`). Posed in an explode or stroke state, it stretches from the gland face to `to + offset · F.a` instead of translating, so every check sees the rod that is drawn. Put it on the moving item (the one with the stroke `move`). |
| `moved(shape, offset)`, `sdf(shape, p)`, `boundsOf(shape)` | Helpers: translate, signed distance (negative inside), world bounds. |

`meta` carries `name` (used in reports, and matched by prefix: `"nozzle.bell*"`), `owner` (a pipe's own flanges, bands and valve), `touch` (names a tube may touch on purpose, such as the clamps round a harness), `bundle` (cables in one harness may lie side by side, and a clamp carries its harness's bundle), `joined` (runs of one gapped pipe), `limp` (a cable: it holds nothing up), `anchor: false` (a clamp: an end may not count it as its mount), `fitting` (a clamp or collar whose centre `crowding` checks against other lines), `seated` (a pad or rim-on-sphere joint sunk into its host by design), `cut` (a shape or list of shapes subtracted from this one: a saddle pad is a disc cut by its host) and `curve` (the smoothed profile of a turned part with no analytic slope, for `wobble`). Tube runs also carry `tips` (the route's true end points, or `null` for an end that merges into a body, like a volute's start), `tipAxes`, `tipRims` (the flange or connector radius at each end), `free` (ends that are meant to come off their mount apart), `offset` and `total` (where the run sits on its route). Every shape carries the `move`, `ride` and `piece` of the item that draws it, so it can be posed in any explode state.

**States.** An explode state is a number (every assembly at that share) or `{ label, shares }`, one share per assembly. `uniformStates(10)` gives 0, 0.1 … 1; they are not frames anyone sees, so they only feed the key solver. `springStates(order, { stagger, stiffness, damping, substeps, fps = 60, settle = 1e-3, stride = 1, reversals = [] })` replays the live page's own springs with their stagger, opening from rest and closing from apart, one state per frame at `fps` until every spring is within `settle` of its target (about 1.2 s each way on the Raptor, so the slow tail of every collapse is checked too). `stride: 4` keeps every fourth frame, for the depth raster. `reversals` adds the paths a quick hover makes: open, then close again after that many seconds (and the reverse), with the page's own scheduling: on a reversal only the assemblies whose target changes are rescheduled, staggered among themselves, so one that has just left comes straight back and one still waiting simply stays home. **`reversalsOf(order, stagger)`** gives a reversal 8 ms either side of every stagger boundary: that is where the bad windows are (a reversal 15 ms before the next start left the hot-gas duct flying out into a methane pump still parked at home), and three hand-picked times missed every one. **`prefixStates(order, { shares = [0.25, 0.5, 0.75, 1] })`** adds the static states behind the prefix rule: the first `k − 1` assemblies fully out, the `k`-th at each share, the rest home. Shares along the stagger are always non-increasing (earlier assemblies lead, opening, closing and reversing), so every reachable frame lies between these. A 40 ms sample missed whole pass-throughs: between two samples the nozzle travelled 15 units, more than twice the widest pipe. `shareOf(item, state)` reads an item's share from its `ride` (or `piece`), and `offsetOf(item, state)` is its move times that share.

**`overlapsOf(items, P, { step = 0.25, tolerance = 0.3, least = 0.25, states = [0], skip })`** finds every pair of drawn items whose screen footprints overlap in each state and, on a grid of `step` viewBox units over the overlap, ray-casts both shapes along the camera ray. A cell counts when one surface is more than `tolerance` nearer. Each side's flagged cells are closed (one dilate, one erode, so an aliased sliver becomes solid), eroded once (so the hairline along a real contact drops out), and the largest connected piece is measured in px²; a side is "in front" when that area is at least `least`. At the old 0.5 step with a 4-neighbour core count, a whole flange sunk into a pump housing and every cable crossing (a 0.5 cable is under three cells wide) read as noise. Each item is ray-cast once, at home; in any state its raster is that one shifted on screen by its offset (rounded to the grid) and in depth by `offset · V`, which is exact for a translation. Pair results are cached by the pair's relative shift, so pairs that move together are measured once. On the Raptor, 1,119 states (uniform, every fourth frame of every replayed path, prefix states) cost less than the old 61. `skip(a, b)` excludes pairs, such as neighbouring chunks of one pipe.

**`orderKeys(items, overlaps, { gap = 0.02 })`** turns every one-sided pair into a rule in that pair's state: `keyFront − keyBack ≥ gap + shareBack · liftBack − shareFront · liftFront`. It is linear in the keys, so one longest-path relaxation settles all states at once and only ever moves an item later. A loop that can't be met drops its weakest rule (`dropped`). Pairs in front of each other in different places are `cycles`.

**`depthOrder(items, P, { overlaps, solved, rest, allow })`** checks the final order in every state. At rest (`rest`, default the states 0 and 1) and in every replayed spring frame, every wrong pair is a problem: a clean misorder, an interlock and a pair no key of the form `k + lift · share` can hold all show on screen. In the uniform states only a clean misorder counts; interlocks there are listed as passing. `allow` takes `[patternA, patternB]` pairs to excuse explicitly.

**`clearances(tubes, solids, { ratio = 0.5, least = 0.2, state, only, mounts })`** samples every tube and lists every place it comes closer than `max(least, ratio · r)` to a solid or another tube. Near a run's flat end it measures the end ring, not a sphere that would overhang the cut. Only an end's own mount (the shape its rim sits on, see `mountsOf`), that mount's pipe, and the fittings of a line that shares the mount or lands on this tube are excused, and only within `r + gap + 1` of the end; near an end anything else may still not be entered by more than 0.1. Pass `state` to pose everything (the excuse holds only while tube and mount are within 0.05 of where they sit together) and `only: "through"` to list penetrations alone. A tube's distance to another tube is the true capped-cylinder distance, so a run that ends flat under a flange doesn't grow a phantom round cap that the next run seems to enter.

**`mountsOf(tubes, solids)`**, **`terminals(tubes, solids)`** find, for every tube end, the nearest anchored shape to its end rim (12 points at the flange or connector radius) and list every end whose rim isn't within 0.15 of it: a flange standing off its port, a flat flange on a curved body (the rim gap), a cable that stops beside the bundle or inside a clamp.

**`looseEnds(tubes, solids, { state = 1 })`** lists every tube end that, in that explode state, touches no solid of another owner that moves with it. Its own flange, coupling or connector shell does not count (with it, the check could never fail: 25 Raptor ends rested only on their own fitting). Ends marked `free` are reported separately as a deliberate list; everything else fails.

**`supports(solids, tubes, { ground, touch = 0.15, sunk = 0.5 })`** builds a contact graph of solids and rigid tubes and lists every solid not connected to the `ground` (default: names starting `stand.`), plus every pair of solids sunk more than `sunk` into each other unless they share an owner or one is `seated` on the other (named in its `cut`). A seated pad is excused against its host only: the Raptor's igniter flange sunk 0.5 into the hot-gas pad of another assembly hid behind a blanket `seated` flag. `hanging` lists rigid tubes that connect to nothing held, and fails too.

**`solidClearances(solids, { state, sunk = 0.5 })`** poses the solids and, for every pair that sits at a different offset in that state, tests each one's surface points against the other's `sdf`; anything more than `sunk` inside fails (a seated pad and its host are excused). Run it at state 1 and in every replayed and prefix state: a housing, pad or flange of one assembly sweeping through another's body is invisible to the tube checks.

**`seats(solids)`** lists every `seated` shape cut to a host that moves with another assembly. Apart, its cut base is on show: the hot-gas duct's saddle, cut to the methane pump's turbine and ring, flew out as a jagged shard with a pointed tail. A saddle belongs to its host's assembly; the moving line meets it on its flat face.

**`selfOverlaps(items, P, { fold = 0.25, least = 0.5 })`** checks each drawn tube chunk against itself: where the screen centreline turns on a radius under the tube's screen half-width, the inner edge of the ribbon runs backwards and the single-chunk outline crosses itself (a knot with the hidden leg's edges showing through). It fails when that fold is at least `least` px and the spot isn't covered by another item. A leg that points at the viewer and turns within a radius or two of a port is the usual cause.

**`wobble(solids, { samples = 20 })`** samples `dr/ds` of every recorded `curve` at 20 points per segment and lists the `s` values where it turns between the run's ends.

**`folds(routes, P, { fold = 140, turn = 110, pinch = 2, items })`** lists every route (`{ name, points, r }`) whose screen direction turns more than `fold` degrees within four radii while its real turn is under `turn` (the cusp of a pipe that runs away from the viewer and then drops), and every bend that turns more than 90° on screen round a radius under `pinch` × the tube's screen radius, measured over spans of 1.5, 3 and 6 radii (a fillet seen edge-on: the V of a leg that points at the viewer and then climbs). A pinch within three tube widths of an end is a stub seen end-on and doesn't count. With `items`, only bends you can see count: each is ray-cast against every drawn shape.

**`crowding(routes, P, { gap = 1, bend = 30, links, items, allow, state, clear = 2 })`** lists every place where a route's bend (a 3D turn over `bend` degrees within ±2 radii) or end lands on another route's outline on screen: the outlines closer than `gap` px or overlapping, with both lines visible there (ray-cast against `items`; a route's own flanges, shells and clamps count as the route). Those read as a joint or a line threaded through another, whatever the true depth. A bend is never excused by overlapping deeply; an end only when the other line is straight there and the end's tip itself is clear of it. Every `fitting` (a harness clamp) is checked too: a clamp whose outline lands within `clear` px of another line reads as that line growing out of the clamp, as the Raptor's harness clamp did on the oxygen-to-preburner line in the apart pose. `links` (from **`linksOf(tubes, mounts)`**) excuses lines that really meet: one mounted on the other, both on one mount. `state` poses routes and items in an explode state, so the apart pose is checked too (routes carry `move`, `ride`, `piece`). `allow` takes `[patternA, patternB]` pairs you have looked at and judged clear.

**`slenderOf(solids, { ratio = 4, least = 0.3 })`** turns every slender recorded solid (a turned part, prism or box longer than `ratio` × its radius or half-width: tie rods, piston rods, posts, rails, handles) into a pseudo-route, which `audit` passes to `crowding` as `rods`. A tube's bend or end that lands on one fails like one landing on another tube (`crowding: hose.B end × cyl.tie-rod (rod)`), unless the tube is mounted on it or on something linked to it. Rods are only ever the line landed on: their own ends aren't checked.

**`coverage(items, P, { least = 0.6, pad = 1, allow })`** compares, for every item with an `svg` string, the screen box of what it draws (`markupBox(svg)`: every path, circle, line, rect and polyline, through `transform`s and `clip-path`s, without halos, text and defs) with the screen box of its recorded shapes (`footprintOf(shape, cameraBasis(P))`), and lists every item whose shapes cover less than `least` of its drawing: `stand-in shapes: rod.rod  its shapes cover 4% of its drawing`. A sliver recorded for a long rod, or one shape recorded for an item that draws three parts, makes every other check blind to the rest. `audit` takes `cover: [names]` to excuse items whose drawing really reaches past their body; look at each first.

**`report(result)`** formats everything; **`failures(result)`** counts what should fail the build.

The Raptor wires it like this (`examples/raptor-engine/build.mjs`): its `put`, `tubeItems` and `record` are thin wrappers over `R.put`, `R.route` and `R.solid`, and the end of the build is
```js
A.settle(R, P);
items.sort((a, b) => a.key - b.key);
A.auditOrExit(R, P, { crowd: CROWD_OK });
```
The live page reads the same `STAGGER` and `SPRING` from its data, so the audit and the page can't drift apart. Every helper (`turned`, `band`, `flangeAt`, `boss`, `padOn`, `boxAt`, `valveOn`, `pipe`, `cable`, `clampAt`, `strut`, the actuators) records its shape as it puts its item, and the static stand drawn in the back layer (deck, flame cone, deluge ring and feed, columns, splices) is recorded as fixed shapes owned by `stand.frame`, so an explode vector that drove a part into the stand would fail too. Nothing escapes the check.

## 13. The WebGL layer (`kit/gl.mjs`)
| Export | What it is |
|---|---|
| `GLSL_ISO` | The GLSL prelude that `glLayer` prepends. It holds the camera uniforms (`uCanvas`, `uView`, `uOrigin`, `uK`, `uCam`, `uTime`) and the functions `viewOf`, `project`, `towardViewer`, `rayOf`, `pixelOf`, `lightDir`, `onFloor`, `onWallX`, `onWallY`, `hash12`, `hash13`, `noise2`, `noise3`, `fbm2`, `fbm3`, `window` and `tonemap`. Don't redefine any of them in your fragment: GLSL rejects a second `rayOf`. |
| `rayOf(vb)` (GLSL) | The world point under viewBox point `vb` on the vertical plane through the origin that faces the camera. `rayOf(vb) + t·towardViewer()` is that pixel's whole ray; larger `t` is nearer the viewer. |
| `pixelOf()` (GLSL) | World units per canvas pixel, `(uView.x / uCanvas.x) / uK`. Divide by `uCam.z` for the floor's depth direction, or by `uCam.w` up a wall. |
| `glCamera(P)` | `{ origin, k, cam }` for a projection. Works in Node, so put it in the page `DATA`. |
| `glLayer({ canvas, fragment, view, camera, uniforms = [], fixed = {}, extensions = [], maxRatio = 1.5, onError })` | Returns `{ gl, extensions, set(values), draw(values, time), clear() }`, or `null` without WebGL or on a compile error. `uniforms` names the uniforms you will send per frame. `fixed` is `{ name: value }`, set once at creation, for everything that never changes (plans, heights, theme constants). `extensions` lists GLSL extensions such as `"GL_OES_standard_derivatives"`; each one the context supports is enabled and prepended as `#extension … : enable`, and the returned `extensions` lists those that were. |
| `set(values)` | Sets uniforms without drawing, for a value that changes rarely. |
| `draw(values, time)` | Fits the canvas, sets the camera uniforms and `uTime`, then `values`, and draws. |
| `clear()` | Clears to transparent, only if something was drawn. |
| `smooth(a, b, x)` | Smoothstep in JS. |
| `burst(age, rise, fall)` | An event envelope: a smoothstep attack over `rise` seconds, then exponential decay with time constant `fall`. 0 for `age ≤ 0`. |
| `settle(state, target, dt, seconds, calm = false)` | A critically damped spring on `state = { x, v }` with time constant `seconds`, four sub-steps. Snaps to the target once both error and velocity are below 1e-4. Returns true while still moving. `calm` jumps straight to the target. |

| `GLSL_TURN` | Opt-in GLSL for 3D figures, prepended by the figure to its fragment: `uGroupX/Y/Z/T`, `toGroup(p)` (world → the group's rest-world coordinates), `toWorld(g)`, `dirToGroup(d)`, `dirToWorld(d)`, and the cover masks `uCoverEdge[60]`, `uCoverCount[6]`, `coverOf(vb, soft)` (1 inside a part painted after the effect). It defines `toWorld`: don't prepend it to a fragment that declares its own. The chunk uses 70 uniform vectors. |
| `glslPose(Name, prefix = "uName")` | The same four functions for any group: `toName`, `fromName`, `dirToName`, `dirFromName` over `prefixX/Y/Z/T`, so one shader follows several groups. Feed it `TURN.poseUniforms(controller, group, prefix)`. |

A value for `fixed`, `set` or `draw` can be a number (float), an array of length 2–4 (`vec2`–`vec4`), or an array or `Float32Array` longer than 4, which goes to the uniform array of the declared type: `uniform3fv` for `uniform vec3 name[N]` (cover edges, spark positions), `uniform2fv`/`uniform4fv` likewise, and `uniform1fv` for `uniform float name[N]`. There are no ints, matrices or samplers through these calls; a `sampler2D` defaults to texture unit 0, so bind a texture with `layer.gl` (the ripple tank's history texture). Names that aren't in `uniforms` or `fixed`, and uniforms the compiler optimised away, are skipped.

`smooth`, `burst` and `settle` are plain JS: import them from `kit/gl.mjs` in a module, or get them for free on a page that inlines `glScript()`. `references/webgl.md` has the full contract, a minimal page and the craft rules.

## 14. Tools
| Script | Use it for |
|---|---|
| `scripts/capture.mjs` | One-shot screenshots: desktop, phone, hover, keys, timed shots, 4× zooms, reduced motion, console check. Chrome starts with WebGL on (SwiftShader allowed), so shaders show. `--times` delays are measured from the first shot; `--key` sends the same key codes as `drive.mjs` (`7` is `Digit7`, `Comma` sends `,`). |
| `scripts/drive.mjs` | Scripted sessions in headless Chrome with WebGL on: scroll, move, click, key, slow, until, eval, shots and contact sheets. `--no-webgl` checks the SVG fallback and `--preload file.js` runs a script before the page's own. Use it for shader transitions and anything with a sequence. See `references/verify.md`. |
| `scripts/lines.mjs` | The line-end check: every texture line in the card must end on a stroke, a dot or an outline, or under a later fill. `--out` rings the failures; exits 1 on any. The same check is the `["lines"]` action of `drive.mjs`. See `references/verify.md`, "Line ends". |
| `scripts/inline-kit.mjs` | `kitScript()`, `glScript()` and `turnScript()` return the kit, the WebGL layer and the 3D runtime as plain script, to inline into a standalone page. |
| `scripts/drive.mjs` `["turn", value, group?]` | 3D figures: sets the first turn group (or `group`, or a values object) through `window.__isoTurn` and waits two frames. |
| `scripts/drive.mjs` `["orbit", out.png, from, to, step, sel?, scale?, [x,y,w,h]?, columns?]` | 3D figures: a contact sheet across angles, each frame labelled θ°. 15° over a full turn; 0.1° across an order change; 1° across a fold-prone bend. |
| `scripts/turn-check.mjs` | 3D figures. `build.mjs --fidelity` (Node: prisms, rounds, lathes and tubes against the kit builders at P′(θ)); `page.html --order` (needs a `--verify` build), `--pops`, `--lines`, `--perf`, `--states file.json`, `--part name`, `--out dir`. Every check runs its planted control first. `references/3d.md`, section 14. |
| `scripts/turn-bench.mjs` | 3D figures: `emit` and `order` CPU time in Node over 360 angles, per part kind (median of 7). |

## 15. Turning groups (`kit/turn.mjs`, `turn-build.mjs`, `turn-audit.mjs`)
The optional 3D mode. `references/3d.md` is the guide; this is the reference.
```js
import { turning, separate, gjk } from "../../kit/turn-build.mjs";
import { TURN, TURN_CSS } from "../../kit/turn.mjs";
import * as TA from "../../kit/turn-audit.mjs";
```

### `turning(P, { recorder, scale = 1 })` → `T`
| Member | What it does |
|---|---|
| `T.group(name, { parent?, turn: { pivot } \| slide: { direction, travel } \| hinge: { point, axis, range } \| free: { origin } })` | A rigid group. Values: degrees (turn, hinge), world units (slide), `{ R \| q \| axis+angle, t, world? }` or `null` (free). |
| `T.layer(name)`, `T.stack(names)` | Live layers and the full back-to-front stack of live and static layer names. |
| `T.prism`, `T.round`, `T.lathe`, `T.ball`, `T.tube`, `T.fixed`, `T.plane`, `T.billboard`, `T.split` | Parts, in world coordinates at rest. Common options `{ name, group, layer, tone, crease, lit, material, owner, details }`. Signatures in `3d.md`, section 4. |
| `T.build({ statics, poses, verify, strict })` | `{ svg, data, css, report, layerSvg(name, { width, height }), verify, planesByName }`. |
| `T.parts`, `T.groups`, `T.layers`, `T.routes`, `T.data`, `T.built` | The scene, for the audit and the checks. |

`details`: `seams`, `ribs: { s0, s1, count, phase, twist, fade, seams, tone }`, `bolts: [{ s, r, count, phase, size, tone, fade }]`, `dots: [{ at, normal, size, tone }]`, `rings: [{ at, normal, r, tone }]`, `rules: [{ points, tone, free }]`, `bevel`. Each constructor takes its own subset in its own frame (a ball's `s` runs along its first flat's normal; a prism takes dots, seams and bevel only): the table in `3d.md`, section 4. Materials: `gold`, `chrome`, `steel`, `gunmetal`, `rubber`, `brass`, `copper`.

`separate(A, B, hints)` and `gjk(A, B, warm)` return `{ n, d, margin }` for two point sets (the build's plane solver).

### `TURN` (runtime)
| Member | What it does |
|---|---|
| `TURN.mount(stage, data, { onHold, onFrame, follow })` | The controller: `set(values \| θ, { detail })`, `render()`, `angleAt(x, y, z, group)`, `values`, `pose(name)`, `detach(name)`, `anchor(group, restPoint, restDir?)`, `cover(names, options)`, `order(layer)`, `orderIndex(layer)`, `events()`, `stats()`, `frames()`, `cams()`, `scene()`, `destroy()`. Installs `window.__isoTurn`. |
| `TURN.cover(controller, names \| [names…], { polygons = 6, edges = 10, pad = 0.3, layers })` | `{ uCoverEdge, uCoverCount, dropped, parts }` for `coverOf`. |
| `TURN.poseUniforms(controller, group, prefix = "uGroup")` | Four `vec3` for `GLSL_TURN` or `glslPose`. |
| `TURN.anchor(scene, cams, group, point, dir)`, `TURN.floorOf(scene, group, R)` | Anchors outside a controller; the lowest z of a group under a rotation. |
| `TURN.rotation(axis, deg)`, `TURN.quat(q)`, `TURN.compose(A, B)` | Rotations as `[9]`, for free poses. |
| `TURN.materials`, `TURN.materialCss()`, `TURN.css()`, `TURN_CSS` | Material names; the CSS. |
| `TURN.camera`, `TURN.poses`, `TURN.cams`, `TURN.prepare`, `TURN.emit`, `TURN.order`, `TURN.frame`, `TURN.gjk`, `TURN.disjoint`, `TURN.q`, `TURN.levels` | The pipeline, for tools: cameras per group, a prepared scene, one part's frame, a layer's order. |

### Orbit audit (`kit/turn-audit.mjs`)
| Export | What it does |
|---|---|
| `orbitOrExit(T, R, P, { flag = "--audit", argv, sweep, clips, poses, crowd, cover, fittings, quick, only, log })` | Runs `orbit` when the flag is present, prints the report, sets exit code 1 on any failure. Ends `turn audit passed` only for a full run; with `only` it ends `turn audit partial`, with `quick` `turn audit quick` (`3d.md`, section 13). |
| `orbit(T, R, P, options)`, `orbitReport(result)`, `orbitFailures(result)` | The checks, the text, the count. `result.skipped` lists the lines `only` left out (printed `skipped`); `result.quick` is set for a quick run. |
| `posed(shape, pose)`, `eventsOf(T, layer)` | A recorded shape moved by a group pose; a layer's edge-on angles. |
