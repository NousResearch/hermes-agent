# Craft

How a figure comes to read as a crafted object rather than a diagram. Every rule here came from a figure that was rejected or approved by a demanding reviewer.

## Contents
1. What "crafted" means
2. Anti-patterns
3. Parts vocabulary, with recipes
4. Tones
5. Painter's order
6. Camera and composition
7. Themes
8. Round parts and pipes
9. Exploded views
10. Screens and things on faces
11. Shaders

## 1. What "crafted" means

The approved figures share five qualities.

1. **A real object you could build.** The reader names it at a glance: a test rig, a dial indicator, a chest of drawers, a gantry plotter, a lock. It is invented for the subject, never traced from a reference.
2. **Many small, precise parts.** A good inline figure has 30–60 solids and 300–500 paths. A hero subject that really has more parts can carry far more: the desk computer has about 85 solids, and the Raptor about 270 solids in 478 depth-sorted items. Every part rests on another part, aligns to its neighbours and has a job. Density comes from structure (feet, rails, stops, bushings, knobs), not from texture.
3. **Volume.** Parts have real thickness, sides shaded in four tones, lighter tops, bevels that catch light, and a soft halo underneath.
4. **Calm.** Thin 0.6 px lines and a restrained palette. One lit accent tells the eye where to look. Motion is eased and slow enough to follow.
5. **Truth.** The readout shows live, correct numbers, and the motion follows the real model: springs with real stiffness, true sizes and thresholds.

## 2. Anti-patterns

Reviewers rejected each of these, so don't do them:

- **Abstract stacks.** Floating plates, slabs or tiles standing for "layers" are too abstract to read as anything. If you need layers, make them drawers, trays in a rack, plates on guide pillars, or cards in a holder.
- **One big plain slab.** A large surface with a few lines is too flat and bland. Break it into a base, a carriage and parts; add a groove, screws, a ruler and a label plate.
- **Blueprint furniture.** Leader lines, balloons with numbers, dimension arrows, hatching, dashed hidden edges and floating labels belong in the legend and caption, not the drawing.
- **Lines that stop in mid-air.** Guide fans, connectors that end short of a part, or a crease that stops before the silhouette corner. Every line must land on something.
- **Wireframes.** See-through boxes where the back edges show.
- **Colour everywhere.** Keep it monochrome unless colour is the subject: one lit accent plus true RGB only for colour channels. A fluid is not colour by default: oil in a sight glass, water in a tank and coolant in a pipe are tints of the figure's own inks, and the `--no-webgl` fallback draws the same tint the shader does. A blind-test bench painted its oil saturated orange in an otherwise grey figure, and its fallback oil was grey.
- **Text in the drawing.** Allow at most one tiny engraved label (`FaceText` on a face). Everything else goes in the card corners, legend or caption.
- **Copying a reference.** Take mood and finish from a reference, then invent the object.
- **Raw pointer jumps.** Parts that snap to the cursor. Ease everything.
- **Pipes through pipes.** On the first Raptor the owner zoomed in and found cables and lines passing through the main ducts and through each other, "which makes no sense". At 1× it looked fine. Every pipe and cable keeps clear of everything but its own connection, every crossing is drawn in true depth order, and `kit/audit.mjs` proves both (section 8, "Clearance and crossings").
- **Textures that stop.** On the same close-up, rows of bolt dots on the chamber stopped dead in the middle of the surface, and the bell and throat were shaded as a staircase of flat tone blocks. Both read as broken. The owner's words: if a texture has to end, "we'd rather fade them out at the end than an instant stop". Shade curved surfaces smoothly, and make a texture end on a real edge or fade out (section 8, "Shading and textures on curved surfaces").
- **Shaders that pop or are cut.** The first Raptor plume changed colour instantly between its phases, "as if cut out", and a glow ended in a straight line at the canvas edge. Every shader term rides an envelope and reaches zero before any bound (`webgl.md`, section 7).
- **Textures that stop, on flat faces too.** The same fault came back on a later figure's boxes, where no curved-surface rule reached it: a tank's weld seam inset 6 units from each rounded corner, a manifold face line inset 4, a row of faint ticks floating on a bare bedplate with no ruler under them, cooling ribs that stopped 2 units short of the seam arcs at both ends, and a hose braid drawn as a dashed stroke down the centreline that ended in two straight hard edges along the hose and broke at every chunk joint. A seam on a box runs edge to edge round the visible corners (`k.sideSeam`), ribs end on seams at the same `s` (`ribsOf(…, { seams: true })`), ticks hang from a ruler strip's edge, and a texture on a pipe comes from `tubePieces(…, { rings })`. `node scripts/lines.mjs page.html` fails every line end that touches nothing (`verify.md`, "Line ends").
- **Stand-in shapes.** To quiet the audit a builder recorded a 0.45-thick disc at the tip of each of five overlapping rod copies, and a sliver for a second rod that was drawn sunk into its cylinder head. The audit passed and said nothing about either rod, and the lit rod's outline was notched every 12 units while it moved. Record the true shape of what you draw; the audit's stand-in check fails an item whose recorded shapes cover less than 60% of its drawing. A rod that slides out of a gland is one rod (section 3, "Rod out of a gland").
- **Dark-theme glows pasted on white.** The owner made light the shipping theme for every site figure, and seven shader figures built on a dark plate had to be converted. Taking a dark glow and darkening it gave "red ink smoke" (the Raptor's light plume). It also gave flames drawn as outlines, sparks that looked like dirt, salmon-coloured cooling glass, grey fades and cavities that got heavier as the fire went out. On paper, light reads through saturation and a darker physical ground (section 7, and `webgl.md`, section 8).
- **Fixing one close-up by eye.** Tuning a painter's key or nudging a waypoint until one screenshot looks right breaks another frame or another theme. The Raptor's hand-set keys looked right at 1× and had 323 depth-order problems. Measure (section 8, "Clearance and crossings"), then look.

## 3. Parts vocabulary, with recipes

`P` is the projection and `k` is the kit, as in `import * as k from "./iso-kit"`. Heights are in world units, with z up. Draw each recipe's solids in the order listed.

### Base plate on feet
```js
const PLATE = { x: -128, y: -64, w: 262, d: 128, r: 8 };
halo:  `<path class="iso-halo" d="${k.haloOf(PLATE, -17, 12, P)}"/>`
feet:  for (const [x, y] of byDepth(k.corners(PLATE, 22))) {
         k.cylinder(x, y, 7, -17, 1.6, P, 28)            // flange, tone lo
         k.cylinder(x, y, 5.4, -15.4, 3.4, P, 28)        // foot, tone mid
       }
slab:  k.slabOf(PLATE, -12, 12, P, 8, 1.6)                // tone mid, bevel 1.6
groove: k.planOutline(k.insetPlan(PLATE, 5), 0, P)         // line lo
screws: k.corners(PLATE, 10) → k.ring(x, y, 2.2, 0, P, 16) (line lo) + dots size .5
```
Sort feet with `byDepth = (pts) => [...pts].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P))`.

### Ruler strip and label plate
```js
const strip = { x: -110, y: PLATE.y + PLATE.d - 15, w: 230, d: 7, r: 1 };
k.slabOf(strip, 0, 0.6, P, 3)                                   // tone lo, crease none
const t = k.topTicks(strip.x + 4, strip.x + strip.w - 4, 4, 5, strip.y + 0.8, 0.6, [2, 3.6], P);
t.minor → line lo, t.major → line mid
label plate: k.slabOf(plan, 0, 0.8, P, 4) + k.planOutline(k.insetPlan(plan, 1.6), 0.8, P) (faint)
             + three k.lineOnTop rule lines of different lengths (lo, { free: true }) + two screw dots
```
Ticks only live on a ruler strip or a dial face, rooted on its edge: `topTicks` grows each tick from `edge`, so put `edge` on (or within 1 unit of) the strip's edge, and `sideTicks` hangs them from `top`, so pass the face's top edge, not 2 units below it. A row of marks on a bare plate is a texture that floats at both ends. The label plate's rule lines stand for printed text, so they are the one place a line may stop on a bare face: pass `free: true` so the line check knows it was meant.

### Seams, welds and trim lines on a box
```js
const seam = k.sideSeam(TANK, 14, P, 6);                         // visible near half of the plan at z = 14, round the corners
k.solidSvg(k.slabOf(TANK, 6, 98, P, 6, 1.2), { tone: "mid", inner: k.lineSvg(seam, { tone: "faint" }) })
```
A seam, weld bead, band or lid line on a box wraps the visible corners and ends exactly on the vertical silhouette edges, like `sideArc` on a cylinder. Pass the slab's own `steps` so the seam follows the same facets, and pass it as `inner` so its ends sit under the outline. A line that belongs to one face only (a stiffener, a panel joint) runs the full face, edge to edge: `segment([x, y, z0], [x, y, z1])` with `z0` and `z1` the face's bottom and top, never `z0 + 3`. Slots and vents are closed outlines (a `planOutline` of a thin rounded plan, or the same in a face plane), so they have no ends at all. Never inset a line by a few units from its corners: at 4× it reads as a texture that stops.

### Rod out of a gland
A piston rod, a plunger or a slide whose visible length changes as it moves is one rod, built once at full length (visible plus stroke), with the part inside the gland clipped away:
```js
const ROD = G.rodOut(F, gland, tip, r, P, { stroke });           // F.a points out of the gland
svg = `<clipPath id="rod"><path data-hold d="${ROD.clip}"/></clipPath><g clip-path="url(#rod)">${k.solidSvg(ROD.paths, { lit: true })}</g>`
R.put({ ..., move: G.mul3(F.a, stroke), svg, shapes: [R.solid(A.slide(F, gland, tip, r, { name: "rod" }))] })
live: el.setAttribute("transform", `translate(${dx} ${dy})`); holdStill(el, [dx, dy])   // dx, dy: the item's screen offset this frame
```
`rodOut` returns the rod's paths from `gland − stroke` to `tip` and a clip that covers it from the gland face to `tip + stroke`. The clip path carries `data-hold`; whatever moves the item calls `holdStill(el, [dx, dy])` with the same screen offset, which moves the clip the other way, so the gland face stays put while the rod slides through it. The same helper retracts a rod into a gland (a load cylinder) when its item moves the other way. Draw a faint seal ring (`circleOf` at the gland face, radius `r + 0.4`) on the gland. Record it with `A.slide`, which the audit poses by stretching the rod from the gland face instead of translating it, so clearances, depth and solid checks see the true rod in every frame. Never fake the extension with overlapping copies: each copy's lit edge stops where the next one's fill starts, and the outline is notched every few units while it moves.

### Rails, end stops and a carriage
```js
rails:  for y of [-20, 15]: k.slabOf({ x: 2, y, w: 122, d: 5, r: 1 }, 0, 3, P, 3, 0.4)   // mid
stops:  k.slabOf({ x, y: -23, w: 5, d: 46, r: 1.2 }, 0, 7, P, 3, 0.5) + 2 screw dots on top
carriage (built at x = 0, moved with translateAlong):
        k.slabOf({ x: -28, y: -24, w: 56, d: 48, r: 3 }, 3, 5, P, 5, 0.6)
        bushings: small slabs at the four corners, taller than the plate, with screw dots
        rivets: a row of 0.4 dots along the front edge
        handle: a small neck slab plus a knurled cylinder on the front face
```

### Post, collar, clamp and thumb screw
```js
k.cylinder(px, py, 5, z0, 4, P, 28)              // collar
k.cylinder(px, py, 3.2, z0 + 4, h, P, 28)        // post, crease none
k.sideArc(px, py, 3.2, z, P, 16)                 // faint seam rings on the post
k.slabOf({ x: px - 7, y: py - 7, w: 14, d: 14, r: 2.5 }, zc, 12, P, 4, 0.6)   // clamp block
thumb screw: a 2.8 × 3 neck slab from the clamp face + k.cylinder(r 2.6, h 6) + k.knurl(..., 18)
```

### Knurled knob with a pointer
```js
k.cylinder(cx, cy, 6, z, 4, P, 32)                          // mid
k.knurl(cx, cy, 6, z, z + 4, 36, P)                         // line lo
k.lineOnTop([cx - 3, cy], [cx + 4.6, cy], z + 4, P)         // pointer, line hi
```

### Coil spring around a rod (split for painter's order)
```js
const c = k.coil(xFrom, xTo, [rodY, rodZ], 4.2, 6, "x", P);
c.back  → line lo, drawn BEFORE the rod
rod     → k.slabOf({ x, y: rodY - 2, w, d: 4, r: 1.8 }, rodZ - 2, 4, P, 3)
c.front → line mid, drawn AFTER the rod
```
Rebuild the coil each frame when it compresses, by changing `xFrom`/`xTo`. It is cheap.

### Dial gauge (face up)
```js
k.cylinder(x, y, R, z, 10, P, 64, 0.8)                       // puck, tone hi, crease lo
k.knurl(x, y, R, top - 3.4, top, 84, P, 0.3)                 // knurled bezel band, lo
k.sideArc(x, y, R, top - 3.4, P, 40)                         // band seam, faint
k.ring(x, y, R - 2.2, top, P, 64)                            // bezel ring, lo
const t = k.radialTicks(x, y, R - 3, 50, 5, top, [1.3, 2.8], P)   // minor lo, major mid
crown: two small slabs on the +x side; lug: a slab joining the arm
needle: k.lineOnTop(tail, tip, top + 0.2, P), tone lit, stroke-width 1.1, rebuilt per frame
hub: k.ring(x, y, 1.6, top, P, 16)
```

### Drawer with a pull and label holder
```js
case:   k.slabOf(casePlan, z, h, P, 4, 0.8)
front:  a thin slab on the near face; pull: two posts plus a bar (small slabs)
holder: an inset outline on the front with a card slab inside; rivets as 0.45 dots
open:   translate the drawer along its axis with translateAlong; its box (floor plus 3 walls)
        is drawn before the front, and its contents are engraved on the floor
```

### Turntable with degree ticks
```js
k.cylinder(cx, cy, R, z, 4, P, 72, 0.8)
k.radialTicks(cx, cy, R - 1, 72, 6, z + 4, [1.2, 2.6], P)
rotate: rebuild only what turns (or the object on top) at the new angle each frame
```

### Perforations, vents and a pulsing dot grid
```js
k.dotGrid(x, y, cols, rows, gap, zTop, P) → dotsSvg(points, { size: .45, tone: "lo" })
// add { pulse: true } for a slow pulsing grid on an "active" part
```

### Lamp, probe and finger
These are stacks of cylinders: housing, cap ring, collar, stem, pad. Seam rings come from `sideArc`, and the lock screw is a small slab plus a cylinder head on the side.

## 4. Tones

| Element | Tone |
|---|---|
| Base, feet, floors | `lo` |
| Structure: posts, rails, frame, carriage, case | `mid` |
| The subject: the thing being explained | `hi` |
| The part the reader should watch now | `lit` (one only) |
| Grooves, inset outlines, floor grid | `faint` |
| Minor ticks | `lo` |
| Major ticks | `mid` |
| Zero mark, pointer line | `hi` |
| Screws and rivets | dots `mid` at 0.4–0.55 |
| Crease (top/side join) | `faint` (default), `lo` on hi parts |

A `Solid` with `lit` gets a white edge and a lighter top. Use it for the moving subject, the plunger or a picked drawer.

## 5. Painter's order

- Draw the base first, then everything resting on it.
- Within a level, sort by `depthOf(centre, P)`, ascending: far before near.
- **Split wrapping parts.** For a spring or a frame around a carriage, draw the back half, then the inner part, then the front half. In the test rig: back frame bars → rods with the back coil halves → carriage → glass → front coil halves → front bars → dial.
- **Moving parts** are drawn at their slot in the order, inside a `<g transform>` that moves them. If a moving part can pass in front of and behind the same neighbour, split the neighbour or bound the motion so the order never changes.
- **Things standing on a stepped surface.** Order the steps so the taller ones are further back. A probe resting on a lower step then can't be hidden by a taller one in front of it.
- Vertical parts that pass through a horizontal one (a post through a clamp) are drawn in two segments, below and above, around the horizontal part.
- **A clip cut is an edge.** Where a clip ends one part against another (a cable-chain run entering its loop, a rod at a gland), make the clip's section flush with the joint plane, and draw a stroked edge along the cut that moves with the clip (`data-clip`). On the plasma cutter, a run's clip section reached past the joint. The run's top face and slot bars were drawn 26 units into the loop's hollow and ended on a vertical cut with no stroke, which is a line stopping in mid-air. Re-ordering the painter's keys wouldn't have fixed it: the loop link's end face really is behind the run.
- **Dense figures settle their keys from geometry.** Past a few dozen overlapping parts, hand-set biases fight each other. Give every item its true shape and let `kit/audit.mjs` measure which surface is nearer wherever two items overlap, then `orderKeys` raises each key just enough to satisfy every pair (`kit.md`, section 12). What it can't satisfy is a pair that interlocks, which needs a split. The first Raptor's hand-set keys left 190 pairs drawn the wrong way round assembled and 133 apart; measured keys took both to zero.
- **Crossings sort per chunk.** A pipe or cable is drawn as `tubePieces` chunks, each its own item at its own depth, so one line can pass in front of one part and behind the next. Where two lines cross on screen, the chunk nearer the camera is drawn later; a crossing can only be ordered if neither chunk also runs behind the other somewhere else, so break long chunks at a crossing (`breaks`).

## 6. Camera and composition

- Default camera: azimuth 45°, elevation 30°. Rotate the azimuth (35–60°) so the face that carries the mechanism points at the viewer. The test rig uses 57° to show the long side. Raise the elevation to look into something open: the ripple tank uses 44° so the water surface and the paper under the tank both read.
- Fit with `fitProjection(boxCorners(...) ∪ tallest points, width, height, { pad: 24–34, azimuth })`. Typical viewBoxes are 600 × 340–380.
- The object fills about 55–65% of the card. Leave air around it, don't let it touch the corner labels, and keep sizes consistent across a set of figures.
- On phones (390 px), check it still reads. If parts get too small, scale the SVG up to about 110–118% inside a card with `overflow: hidden` and a negative margin, as `stretch.css` does in the test-rig example.

## 7. Themes

`ISO_CSS` defines tokens on `.iso`, with dark as the default and `.iso[data-theme="light"]` for light. Every fill and stroke reads a token, so a theme is just a token set. To tune one:

- **Paper** (`--anatomy-paper`) equals the card colour, so fills hide what is behind without showing as patches.
- **Shades.** Four steps from darkest (0, facing away from the light) to lightest (3). Keep them close: about 3% apart in luminance.
- **Lines.** In dark mode, `hi` is near-white and `faint` barely above the paper. Light mode inverts this.
- **Halo.** A blurred dark fill under the base: strong in dark mode, about 30% in light.

**Light ships.** Every figure on the isometrics site is light, and that includes its shaders. Design light first and keep dark working where that is cheap. What the seven shader conversions taught about the drawing:
- **Every colour is a token.** A colour typed into a figure's CSS or its shader is a theme bug waiting to happen. Give it a light value and a dark override (`.x { --x: … }`, then `.iso[data-theme="dark"] .x { --x: … }`). Where the shader needs the same colour, read it from the token (`webgl.md`, section 8.3).
- **Cavities get their own token set.** The inside of a furnace mouth, a ladle, a vessel seen through a cutaway, a water column or a duct window is darker for a reason: soot, refractory, depth, shadow. Give its group a class that redefines `--anatomy-paper`, `--anatomy-face`, `--anatomy-top`, the four shades and the line tones to darker, warmer or bluer greys. Keep all the tones so it still has form, mix it toward the light's hue by a variable the live code writes, and set the tokens to `inherit` in dark. Glows then have a ground, without a black box.
- **Mind nested themes.** A dark plate can sit on a light page. A rule written `.iso:not([data-theme="dark"]) .x` also matches through the page around it. Order the dark rule after the light one, or key on the plate's own theme.
- **Opacity fades a pale colour to grey.** Anything that fades by opacity in light (a fallback glow, a lit accent) gets a saturated token, and its tints are multiply-blended so the lines under them stay visible.

## 8. Round parts and pipes

Engines, pumps, valves, lamps and instruments are mostly turned parts and pipes. `kit/lathe.mjs` and `kit/tube.mjs` build them along any axis (`references/kit.md`, sections 10 and 11). `examples/raptor-engine/build.mjs` has every recipe below in working form.

### Turned parts: think in meridians
Write a part as its half-section: a list of `[s, r]` from one end to the other, and turn it with `G.solidOf(profile, F, P)`.
- **Steps and shoulders** are where the radius jumps at one `s`: `[6, 12], [6, 9]`. They make the crease lines that read as machining.
- **Curves** are many points marked smooth (`[s, r, 1]`), so no crease is drawn across them: a bell, a dome, a fillet into a flange. The kit shades any profile with a smooth vertex smoothly on its own (`smooth: "auto"`), so the shading follows the curve instead of stepping facet by facet; pass `smooth: false` only to opt out.
- **A waisted body is never hulled.** A profile that narrows in the middle (a throat, a spool) has a concave outline. `lathe` notices when its true silhouette lies more than half a pixel inside the hull of its rings and then fills and outlines it with the silhouette and the visible end arcs instead (`fill: "auto"`, or force `"surface"`). Drawn as a hull, the Raptor throat was a straight cone with a pale unshaded strip and a seam that stopped 8 px short of the outline.
- **Seams and weld lines** on a curved surface are `G.arcOf(F, s, r, P, { slope })`, with the profile's slope at that ring, drawn `faint` and passed as the solid's `inner` (`solidSvg(paths, { inner: lineSvg(arcOf(…), { tone: "faint" }) })`; the Raptor's `S` and `Ln` are one-line wrappers over these), so they sit under the outline. They end exactly at the limb, a little short of it. Concatenated after the solid, each seam's round cap lands on the edge stroke and the dark-theme outline of the Raptor's actuator barrels looked dashed at 4×.
- **A long body** (a nozzle, a barrel) is cut into bands with `G.bandsOf(profile, cuts, F, P, { radius, slope })`, so each band depth-sorts against the pipes that wrap it. The bands share one `steps`, one normal field and one surface, their own outlines come cleared, and `sidesOf` draws one continuous outline for the whole body (`kit.md`, section 10), so the tone curves run through every joint without a step and the sides don't kink. Without an analytic `radius` and `slope` it fits one curve through the whole profile, never one per band.
- **A ring around a band** (a stiffening ring on a chamber, a collar) splits the band: band, ring, band. Drawn as one long band with the ring on top, the ring's back half paints over the band (or the band hides the ring's front); split, each piece orders itself. The Raptor chamber jacket is three bands between its two rings.
- **An open end** is a `ringBand` lip, with a filled void (`circleOf`) for the dark inside and the inside arc from `arcOf(..., { inward: true, least: -1 })`. Bolt dots go around the lip.

### Parts on a round body
- **Flange:** its back face sits on the route end, so the route ends exactly on the mount's surface and the disc runs from there back along the pipe (`centre = end − tangent · t/2`). `disc(-t/2, t/2, r, frameAlong(centre, axis), P)`, with a bolt circle `dotsOf(F, faceS, 0.74r, 8, P, { all: true })` on the face that points at the viewer. Skip the bolts when the face is edge-on (`|dot(a, V)| < 0.12`).
- **Boss or port** on a curved wall: a short `disc` along the outward normal, with a `faint` face ring (`circleOf`) when it faces you. This is where a sense line, igniter or plug lands. Make it at least as wide as the flange that lands on it; a flange that overhangs its boss floats at the rim.
- **Saddle pad.** A flat flange on a curved body leaves a gap at its rim (the sagitta, `R − √(R² − a²)`): an 11.6 flange on an 18.4 pump body stood 4 units off at the sides. Wherever that gap would be more than about 0.15, seat the flange on a pad: a short cylinder whose base follows the host surface (`G.saddleOf(F, a, start, h, P)` with `start(angle)` found from the host's shape, the Raptor's `padOn`/`boss(..., { host })`). Record it as a disc `cut` by its host and marked `seated`, so the audit sees its true shape. Keep the pad inside one band of the host (an overhang under a step is drawn in front of the band that hides it; a Raptor pad lowered until its rim touched the step below became an interlock in every frame) and off the host's silhouette: a pad that straddles the silhouette is half in front of and half behind its host, and no painter's order draws it. The Raptor's LOX duct moved from 130° to 150° for that reason. **A saddle belongs to its host's assembly.** Its base is cut to the host's curve, so if it flies out with the pipe the cut base is on show apart: the Raptor's hot-gas pad, cut to the methane pump's turbine and ring, became a jagged shard with a pointed tail. Give the pad to the host's piece (`boss("ftp", …, { host: FTP_TURBINE })`) and let the moving pipe's flange leave its flat top; the audit's `seats` lists every seated shape whose host moves with another assembly.
- **Igniter plug, one terminal.** A plug carries either the igniter line (leaving its outer face along its axis) or the exciter cable, never both. The cable lands on its own small boss beside it.
- **Valve on a pipe:** a bulged `lathe` profile on `frameAlong(point, tangent)` turned with `smooth: true`, with bolt dots on both end faces, a round neck and an actuator box with a lid outline and two screws. It sits on a straight leg (body length plus both fillet cuts), the pipe is gapped under its body, and the actuator stands perpendicular to the pipe: up on a level pipe, sideways (a round can) on a vertical one, never along the pipe.
- **Volute:** `spiral(F, s, r0, r1, a0, a1)` fed to `tubePieces` with a radius that grows along it. Keep its centreline at housing radius plus tube radius plus 0.3, so it wraps the housing tangentially; a volute sunk into the housing is a pipe through a solid.
- **Ball joints and tees** are spheres (`sphereOf`): a rod-end bearing on an actuator, a cast tee where a branch meets a manifold.
- **Knurls, ribs and fins on a tilted axis:** `ribsOf`. Vanes on a face: `spokesOf` with a `sweep`.

### Pipes
- Lay out a route as a few straight legs between real ports, then `fillet` it with a bend of about 2.6 × the pipe radius. Pipes bend; they don't kink.
- End every pipe on a port with a flange, or on a block with a terminal dot. A pipe that stops in the air is a line that stops in the air.
- Add the parts a real line has: bands where it is clamped or welded (`bands: [0.5]`), a bellows where it must flex (a run of thin `disc`s, `bellows: [from, to, count]`), and a valve where flow is controlled. These are the Raptor `pipe()` helper's options (`kit.md`, section 11).
- Use few sizes, all big enough to tell apart: main ducts (r 5–8), feed lines (r 2–4), sense lines (r 0.9–1.5), cables (r 0.5–0.65).

### Choosing waypoints
Route the big lines first (ducts, downcomers, manifolds), then feed lines, then sense lines, then cables, so each smaller line routes round what is already there. For each route:
- **Start on the port's normal.** The first leg leaves the port straight out along its surface normal, long enough for the flange, then the bend: `fillet` takes at most 0.48 of the shorter leg, so a leg shorter than about twice the bend radius gives a tight bend, and a turn right at a port facing the viewer pinches ("Clearance and crossings", step 5).
- **Follow structure.** Legs run parallel to the body they follow (along a chamber, up a downcomer) or round it on a ring at a fixed radius (`arcPoints`), offset from its skin by the body's own clearance: skin radius plus about 1.5 × the line's radius. The owner's close-up showed sense lines fanning across the chamber's face and over the main duct to reach their block; on the new build the long line from the nozzle manifold runs up the left side, the fuel-valve cable runs up beside the downcomer, and the harness comes down from the controller in parallel drops into clamped lanes.
- **Stay on your side of the subject.** A line that serves a port on the far side goes round the back or round the side at a ring, never diagonally across the front of the thing being explained.
- **Prefer legs across the screen.** A leg that runs straight toward or away from the viewer projects short and makes every turn at its ends a cusp or a V ("Clearance and crossings", steps 4 and 5).
- **Leave room for fittings.** A valve, bellows or clamp needs a straight leg at least its body plus both bend allowances (`r_bend · tan(turn / 2)` each). On a bend the straight body cuts into the pipe; the Raptor's downcomer was rerouted so its valve sits on a straight run, and the oxygen duct gained a vertical straight for its valve.
- **Don't land on a rod.** A pipe's last leg must not run along a rod, tie rod, post or rail on screen, and its end fitting must not sit on one's outline: the blind-test bench's hose ran its last leg parallel to a tie rod, 2 px above it, so the rod read as plugged into the hose fitting. Approach the port across the rods, or turn the elbow so the leg lies across the screen. The audit's `crowding` treats every slender solid (length over 4× its radius or half-width) as a line, so it fails these. Check every pose too: a ferrule that sits on a moving plate's edge at the end of the stroke reads as touching it.
- **Cross deliberately.** If two lines must cross on screen, they cross once, on straights, at a clear angle, and far enough apart in depth that the clearance check passes with room to spare. A crossing near a bend or an end reads as a joint ("Clearance and crossings", step 6).
- **Split, don't overlap.** Where something wraps a pipe (flange, band, bellows disc, clamp, valve body), the pipe is gapped there and a piece ends on each face (`tubePieces` `gaps`, `crossingsOf` for clamps). Run through it as one chunk, the pipe is both behind and in front of the collar and no key can order it. The Raptor chamber jacket was split at its two rings for the same reason.

### Contacts and junctions
Every place two parts meet is either a flat face on a flat face, a tangent contact, or a rim landing on a round body. Nothing is sunk into anything: a sunk part is both in front of and behind its host, and no painter's order can draw it.
- **Flush.** A rod ends on the face of its clevis; a boss starts on the surface it stands on; a lug is a short pin from the housing to the ball.
- **Rim on a round body.** A cylinder of radius `a` meeting a sphere or tube of radius `R` ends where its rim lies on the surface, `sqrt(R² − a²)` from the centre. Its end face is then hidden inside the host and nothing shows through. This is how the Raptor's actuator rods meet their ball joints and how the manifold tube and the downcomer meet their tee.
- **Tees.** A branch into a pipe is a sphere fitting a little larger than both: the run is gapped where its rim meets the sphere and the branch ends on the sphere. Size the sphere so neither pipe's rim pokes into the other: for a branch of radius `b` into a run of radius `a`, `R ≥ sqrt(a² + b²)`.
- **Stand-offs.** A ring that would have to sink into a curved body to touch it (a manifold round a bell) stands clear by at least half its tube radius on small radial pads.
- **Valves and bellows on straights.** A straight lathe on a curved pipe cuts into it at both ends. Put them where the route is straight for their whole length.
- **Two flanges, one joint.** Where a duct bolts to a pump outlet flange, the duct's own flange meets the outlet flange face to face; two flanges in the same place interlock. Each part then keeps a flange when the figure comes apart.
- **Branches have fittings.** A branch leaves its run from a tee (a ball the run is gapped under, the branch starting where its rim meets the ball) or from a saddle pad on the run, never from a bare spot on the run's skin.
- **Struts end in ball joints.** A strut's eye disc laid along an oblique axis sinks into whatever it meets. Give each end a short stem along the host's normal, a ball on the stem, and the rod between the balls (rim on sphere at all four joints).
- **Balls with flats for joints that move.** In a 3D figure a rim landing on a sphere can't be ordered at every angle. Give the ball a flat (`T.ball({ flats: [{ n, d }] })`, recorded as `A.ball(c, r, { flats })`) and land the rim on the flat, or use coaxial knuckle discs. `references/3d.md`, section 5.
- **Everything is held.** Every solid is connected through contacts to the stand: brackets carry boxes, studs carry harness clamps, a gearbox plate joins a motor to its actuator. A clamp that touches only the cable it holds floats.

### Clearance and crossings
A pipe keeps about 1.5 radii between its centreline and anything but its own connection, so a 0.6 cable passes 0.3 clear of a duct's skin and a duct keeps half its radius off its neighbours. Where two lines must cross on screen, they cross in depth, one well in front of the other, and the chunk keys order them; where a pipe passes through a collar, clamp or valve it is gapped there (`kit.md`, section 11). Prove it rather than eyeballing it:
1. Record a shape for every item and every tube as you build, into one `A.recorder(P, { order, spring })` (`kit.md`, section 12): `R.put` for drawn items with their `shapes`, `R.solid` for each shape, `R.route` for each pipe and cable. Make the helpers do it, so nothing escapes. There is a recorded shape for everything the kit draws: `A.slab`, `A.extruded` and `A.cylinder` for floor prisms, `A.solid`, `A.lathe` and `A.disc` for turned parts (traced along the same curve the drawing uses), `A.prism` for `prismOf`, `A.ball`, and tubes.
2. Settle the painter's keys with `A.settle(R, P)` on every build: it measures the overlaps in every state the figure rests in (assembled and apart) and in every frame of its real explode (`explodeStates` replays the live springs and stagger) and runs `orderKeys`.
3. End the build with `A.auditOrExit(R, P)` and run `node build.mjs --audit` until every failing line prints zero. An interlocked pair is a geometry bug (a split, a gap or a tangent contact fixes it); don't bias it away.
4. Lines don't double back on screen. A horizontal leg that runs away from the viewer reads as going up, so a pipe that runs away and then drops reads as a cusp, a pipe ending in mid-air. Route the leg sideways on screen first (the audit's `folds` finds every case: a screen turn over 140° within 4 radii while the real turn is under 110°).
5. No outline folds back on itself, and no bend pinches to a point on screen. A leg that points at the viewer and then turns (a sense line leaving a jacket boss radially and climbing to a block, a pigtail leaving a transducer tip and hooking back) projects its fillet edge-on, and the two legs meet in a V with no visible radius. `folds` measures the screen radius of every visible bend over 90° and fails it when it is under twice the tube's screen radius. Fixes, in order of preference: let the turn happen inside a fitting (a ball tee on the boss, the line leaving it toward its next point: the Raptor's `elbow`, used by sense0, sense2, sense5 and the fuel-rich igniter line); bend in a plane that faces the viewer (the Raptor's pigtails loop in the transducer's own radial plane, which faces the camera; its local helper `sideOn(d) = unit3(cross3(unit3(V − d·(V·d)), d))`, with `V = G.viewOf(P)`, gives the direction across the screen from a leg along `d`, so a bend in the plane of `d` and `sideOn(d)` faces the viewer); move the port to where its normal lies across the screen. A bend in a plane whose normal makes `f = |n·V|` with the view projects to a screen radius of about `R·f²`; once that is under the tube's screen half-width, the inner edge of a single chunk runs backwards and the outline crosses itself in a little knot (the audit's `selfOverlaps`). `fillet(…, { P, tube })` raises each corner's radius to clear that, but only as far as the legs allow, so give a turn next to a port at least `R + 3r` of straight leg. A port that faces the viewer squarely (the Raptor's speed pickup on the methane pump) makes every approach a hairpin: move it.
6. Crossings happen on straights, clearly. Where one line's bend or end lands on another's outline on screen (touching, or overlapping by any amount), the eye reads a joint or a line threaded through another, whatever the true depth: a pigtail hook that peeks out from behind a sense-line elbow, a flange that lands on another line's band in the apart pose, a cable that runs along a short duct on screen and over its end flange. A fitting counts too: a harness clamp whose outline lands on another line reads as that line growing out of the clamp, as the Raptor's clamp did on the oxygen-to-preburner line apart. The audit's `crowding` lists every visible case, assembled and apart; reroute one line so the meeting is a clear crossing of two straights or a visible gap of at least a pixel (two for a clamp). Excuse a pair only after you have looked at it at 8× (`CROWD_OK`).

The first Raptor had 54 clearance problems (39 pass-throughs) and 323 depth-order problems; the routes in this section took both to zero.

### Shading and textures on curved surfaces
- **Smooth shading** (automatic on any profile with a smooth knot; `smooth: { slope }` or `{ slope, radius }` for an analytic curve; `bandsOf` for a body in bands; `sphereOf` for balls) on every curved surface: bells, throats, domes, cones, valve bulges, balls. A flat facet tone per row is only right on a cylinder. Each tone is one polygon over the whole body for as long as its region count holds, so no row seams show.
- **A texture ends on an edge or fades, along the axis too.** Each rib's `s0` and `s1` land exactly on a drawn step, crease or seam arc at the same `s`: pass `seams: true` and `ribsOf` returns the two seam arcs with the ribs, so they can't drift apart (the blind-test motor ran its ribs from 26 to 88 with seams at 24 and 90, and every rib stopped 2 units short at both ends). Where a rib row has no seam, let it run onto the face that ends the band (`s0`, `s1` equal to the band's ends) or fade it.
- **A texture on a pipe follows its cross-section.** Braid, corrugation, tape wrap and armour come from `tubePieces(route, r, P, { rings: { pitch, twist, cross, fade } })`: rings (or with `twist`, helices, and with `cross: true`, a braid of both hands) placed by 3D arc length along the whole route, each drawn as the visible front arc of its cross-section, fading toward both limbs and ending on the ferrule and gap faces. Never a `stroke-dasharray` along a chunk's centreline: it ends in straight hard edges along the hose, doesn't wrap the round section, piles up on the inside of bends, and restarts at every chunk joint.
- **Textures end on an edge or fade.** Ribs and bolt circles that are limited to the side facing you stop wherever the cut-off falls, in the middle of the surface: the bolt circles on the Raptor's chamber rings did, in the owner's close-up. Use `ribsOf(..., { fade })` with `fadedSvg`, and `dotsOf(..., { fade })`, so they fade out toward the turn (the Raptor uses `fade: [0.04, 0.42]` for bolt dots and `[0.1, 0.55]` for ribs). Seam arcs and band lines already end on the silhouette. A single texture line that has to stop in the middle of a face (a groove that runs out, a weld bead short of a corner) is drawn with `fadeLineSvg(points, { fade: [start, end] })`, which ramps it out over a few px with a gradient along the line; never let it end on a bare cut. Choose the cut-off from the way the feature faces, not from the ring it sits on: spray holes on top of a deluge ring face up and are seen all the way round, so they take `all: true`. Culled by the ring's outward normal (`least: -0.2`), the Raptor's row stopped at full strength just past the ring's sides while the ring ran on round the back, bare. Never cut a row on an up-facing feature with a radial `least`; if part of it really is hidden, `fade` toward the turn.
- **Smooth tones taper.** Where a tone band appears or disappears along a curved profile, it is born at zero width and grows (`smooth` refines rows by screen error until each boundary is within 0.35 px and every change of band count happens in a sliver). A band that ends on a straight cut across the surface, or a pale wedge of bare fill, is the bug the reviewer finds first.
- **Shape and shading describe one curve.** Smooth runs of a profile are interpolated with a monotone cubic Hermite curve (Fritsch–Carlson), and the kit uses that same curve for the normals, the tone sheets, the silhouette and the hull. A natural spline overshoots near a peak: on the Raptor's valves it put three sign changes of slope into a body that rises and falls once, and the tone bands zig-zagged across the bulge like wood grain while the outline stayed a polyline with a nub at every profile point. A monotone curve can only turn where the profile turns.
- **The monotone Hermite is only C1.** Its `dr/ds` is a piecewise quadratic: with hand-typed knots it plateaus at each knot and overshoots between them, and tone bands, which follow the slope, stall and then lurch, a staircase in scallops. The Raptor throat had extrema of slope at five places on a profile that bends one way; its deck cone wobbled more mildly. Any curved body larger than a fitting needs an analytic profile `r(s)` and its derivative passed as `smooth: { slope }`, sampled at 20–30 knots: the throat is `√(24.2² + (c·(s − 146))²)`, the cone `48·(1 − x)^0.8` closed to `r = 0` at its tip with a vertical tangent, so the outline rounds over the tip instead of peaking above a flat tip disc. The audit's `wobble` fails any profile smoothed without a `slope` whose `dr/ds` turns between its ends.
- **Outlines are ellipses, not facet polygons.** Facet counts are for tone fills only (about 3 px per facet on screen, `stepsFor(r, 32, 160, scale)`); every silhouette, rim and crease is sampled from the analytic circle at about 1.5 px, and creases and bevels end exactly where the faces turn away. A 20-sided flange outline next to a 96-sample torus reads as a notch at 4×.
- **Tube stripes.** Shine and shade bands run continuously through chunk joints and on to the very end: at every capped end they follow the outward half of the end ellipse, and at an end face turned toward you they stop on its inner rim, which is drawn. At a joint the stripes overhang the body by a further 0.5 (bodies overlap by 0.5, stripes by 1), and every frame is computed from the whole route, so the later chunk's stripe covers its own body's anti-aliased cut and ends on an identical stripe. With body and stripe cut on one line, a hairline crossed the volute's shade band at every joint, light in light theme and dark in dark. Check them at 4× at every bend, tee, bellows and flange, and on a strongly curved tapered tube. Anything you draw on a chunk yourself (a texture, flow marks) follows the same rule: compute it from the whole route, phase it by 3D arc length from the route's start, and extend it over the chunk's stripe overhang (1 unit past each joint), or the joint shows as a half-dash, a doubled mark or a notch. The `rings` option does this for you.

### Harness routing
Cables are the easiest way to ruin a figure: a cable drawn across the subject's face reads as a scribble. Route them like a real harness:
- **Lanes.** Run cables in parallel lanes on a ring around the body (`arcPoints` at a fixed `s`, each lane offset 1.4 units out and 0.2 along for r 0.62 cables). They leave the lane only to branch to their plug.
- **Enter in order.** Cables come down from the connectors in parallel drops and join the ring where their drop meets their lane (at `acos(x / laneR)`), so the innermost lane is fed by the connector nearest the front and no drop crosses another.
- **Leave in order.** The outermost lane leaves the ring first; an inner cable that turns outward while outer lanes still run beside it has to cross them.
- **Follow structure.** A cable that has to climb runs up beside a pipe (the Raptor's valve cable runs up the downcomer) instead of crossing open space in front of the engine.
- **Short branches.** Go from the lane straight to the port. Don't run diagonally across the subject's face.
- **Clamps** (`disc(-0.9, 0.9, r, frameAlong(point, tangent))`, tone `mid`) hold the lanes to the body at regular angles. Gap each cable where it passes through a clamp (`crossingsOf`), or the clamp and the cable interlock.
- **Ends land.** Each cable starts at a connector on the controller or a junction box and ends on a plug, sensor tip, valve lid or bulkhead connector, never on a clamp or in the air beside the bundle. An end that stays with its own assembly gets a terminal dot; an end on another assembly gets a connector shell (`cable(..., { shells: [false, true] })`) that moves with the cable.
- **Count what you draw.** Push every route into a list (`CABLES`) and build the readout from its length. The first Raptor readout claimed 8 cables while 11 were drawn.

## 9. Exploded views
An exploded view earns its place when the parts are the explanation (what feeds what, in what order):
- Choose the assemblies the real thing comes apart into (the Raptor has 11), not single solids.
- Move each assembly along its real disassembly direction, far enough that its silhouette clears its neighbours: down along the axis for stacked stages, out radially for side units.
- Every part keeps a home. Parts that bridge two assemblies (actuators, cables) go with the one they stay attached to, and when apart they still end on something of their own: a flange, a coupling nut at a quick-disconnect, a connector shell on a cable. Strut ends are ball studs on their hosts with the rod leaving with its assembly, and saddle pads stay on their hosts. Mark each end that comes off its mount when apart (`free: [true, false]`); the audit's `looseEnds` fails every other end that touches nothing of another owner in the apart state, and lists the free ones so each is a choice.
- Highlight and motion are separate. A sensor bolted to the chamber highlights with the plumbing but rides the chamber's spring (`ride`, emitted as `data-ride`). Moved with another assembly's vector on its own spring, it slides along its mount and sinks into it mid-flight.
- Choose explode vectors and the stagger so nothing passes through anything, in any frame: `references/motion.md` ("Order the stagger so nothing has to pass through") has the rules, and the audit replays every frame of the real springs, opening, closing and reversing halfway, and fails on any pass-through.
- List the assemblies in the legend as numbered buttons (`01 Gimbal and thrust puck`…) that select, the same as hovering does.
- Springs, staggering and per-frame depth re-sorting are in `references/motion.md`.

## 10. Screens and things on faces
A screen, a dial face or a display is drawn in the plane of the face it sits on (`examples/desk-computer/`):
- **The face plane.** `sideMatrix([0, y], 0, P, "left")` maps a group's local `(x, −z)` coordinates onto the vertical face at `y`. Draw everything on the screen (the glow, scanlines, a mark, `<text>` and a cursor) in those flat coordinates inside that `<g transform>`.
- **A recess.** Clip the screen to its rounded outline with a `clipPath` in the same plane, and draw the recess's inner sides as quads between the face and the face moved `depth` back. Shade the sides by facing (tops `iso-top`, sides by shade). Draw the outline `lo` on the face and `faint` at the back.
- **Motion on the face** is a transform in face coordinates. The CRT power-off scales the picture about the screen centre to a line, then to a dot, with a beam line and a fading dot.
- **Text the machine displays** (a typed line, a counter's digits) is part of the object, not a label, so the one-label rule doesn't apply. Keep it in the screen's own type and size, and fit it: the cap height is about 0.55 of the window's height and the widest string you will ever show (`8.88`, the longest unit) must fit its width with a margin of about a tenth of the height. Clip it to the window like any screen. A blind-test flow meter set 5.6 px digits in a 12.8 × 8 window, so the leading digit crossed the window's left border and the digit tops ran into the unit label.
- **Content seen through an opening** (a sight glass, a gauge window, a recess) sits behind the face, so draw the opening's inner walls between the face and the back plane, shaded by facing, and clip the content to the opening in the face plane as well as its own. A strip of bare face inside the opening reads as a hole drawn on the surface. For a shader behind an opening, `webgl.md`, section 5.

## 11. Shaders
Some subjects can't be drawn with lines: fire, exhaust, water, caustics, real lamp light, steam. These go in a WebGL layer *inside* the drawing, in world space and in the same camera, between a back SVG and a front SVG. `references/webgl.md` has the layer stack, the camera contract, the craft rules (nothing pops, no hard cut-offs, tonemap and dither, blending per theme) and how to verify them. Everything in this page still applies to the SVG around the effect. Its section 8 covers the light theme, which ships: inks solved from the colour the paper should show, ink that grows with the light, a dark physical ground for every glow, recipes for hot metal, glass, sparks, flame, steam, plasma and Cherenkov light, and a no-WebGL fallback that matches the shader.
