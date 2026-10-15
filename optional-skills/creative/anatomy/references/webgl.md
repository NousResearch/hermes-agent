# WebGL inside the drawing

How to put a shader effect (fire, water, light, glow) inside an isometric figure so it reads as part of the same object, seen through the same camera. Two shipped examples do this: `examples/raptor-engine/` (a plume, shock, splash and steam under a rocket engine) and `examples/ripple-tank/` (caustics, shadows and a rippled water surface). Read their `live.js` next to this page.

The user loved the first shader figure, but noticed that the colour changed instantly between its phases, as if the effect had been cut. So the bar is: **nothing pops, nothing is cut, and everything sits in the drawing's perspective.** Every rule below came from a bug that broke that bar.

## Contents
1. When a shader earns its place
2. The layer stack
3. The camera contract
4. Ray-marching in the iso camera
5. Registering shader content to the SVG
6. API and a minimal page
7. Craft rules, each with the bug that taught it
8. Light theme (8b. Shaders on a turning group)
9. Verify
10. Performance

## 1. When a shader earns its place

Use a shader for things lines can't draw: emitted light and glow, fire and exhaust, water surfaces, caustics, soft shadows from a real lamp, smoke and steam. These are continuous fields, and they're the subject or what the subject does.

Don't use one for:
- **Solids.** Parts, edges, ticks and screws stay SVG from the kit. The shader never draws a part that the SVG could draw.
- **Decoration.** Shimmer, background gradients, bloom on every edge, or noise added to make the figure look "rich". If you could delete the effect and the explanation still holds, delete it.
- **Something the SVG already shows well.** A glowing LED is a `lit` dot, not a shader.

The SVG must still read on its own. The Raptor without WebGL still shows the engine, with a rim light and a CSS deck glow. The ripple tank draws its crests and nodal lines as SVG lines (`patternOf` in `model.mjs`), clipped to the drawn outline of the tank's shadow on the paper, and hides them with `.rt-stage[data-gl] .rt-fallback { display: none }` once WebGL is running.

## 2. The layer stack

A shader can't be depth-tested against SVG, so you order layers instead. The basic stack is three layers in one `position: relative` stage, all with the same viewBox:

```
back SVG     what the effect covers        (k.figureSvg: role="img" and the label)
canvas       the effect                     (position:absolute; inset:0; pointer-events:none)
front SVG    what covers the effect        (aria-hidden)
```

For every part, ask one question: **is it ever covered by the effect, or does it cover the effect?** Parts that are covered go in the back SVG, and parts that cover go in the front SVG. If a part does both, you have three choices:

- **Split the art.** Cut the part into pieces and send each piece to the layer, and the slot in that layer, where it belongs. The ripple tank's motor cable is three groups of `tubePieces`: the run over the bridge goes in the front SVG before the bar, the run along the near rim goes in the front SVG after the near caps, and the run down the left leg to the control box goes in the middle SVG.
- **Model it in the shader.** If the part is a simple shape (box, cylinder, torus) and stays in the back SVG, give the shader that shape as an occluder. The Raptor's columns are opaque boxes (`columnNear`), so every air integral starts at `max(tGround, tColumn)`. The deluge ring is a torus inside `groundHit`, so the splash lights its inner side and leaves its outer side dark.
- **Split the effect.** When the effect passes in front of front-SVG parts, add a second canvas above the front SVG. An analytic proxy of the occluding part decides, for each sample, whether it goes to the back canvas (weight `1 − f`) or the front canvas (weight `f`). Each sample is drawn exactly once, so there is no seam. Make the proxy generous: outside the real silhouette both layers look the same.

### Worked case: the Raptor
```
back SVG    deck, feet, flame cone, deluge ring, columns, ladder, cable tray
canvas      .rp-plume: plume volume, shock (back half), flash, splash, ground ring, steam behind the engine
front SVG   every engine item (depth-sorted .it groups), pipes, harness, crosshead
glow SVG    .rp-glow: the two nozzle-lip rims, screen-blended so they can only add light
canvas      .rp-veil: steam and shock samples that lie in front of the engine
div         .rp-flash: a CSS radial gradient (closest-side) for the ignition flash
```
The engine proxy is `engineR(z)`, a radius function of the bell, chamber and pumps. `inFront(p, V)` says whether sample `p` lies between the engine and the viewer. `side(front, veil)` sends the sample to one canvas or the other.

### Worked case: the ripple tank
```
back SVG    base plate, feet, ruler, label plate, paper sheet with tape, lamp cable over the base edge, SVG fallback pattern
canvas      .rt-paper-gl: lamp light on the paper and base, caustics, shadows of the legs, post, bridge, motor, control box and foam beach, refraction
middle SVG  post, arm, lamp and its cables, control box, motor cable down the left leg in clips, legs with levelling screws,
            far rails, three corner posts, far glass, foam beach, plain-water fill
canvas      .rt-water-gl: window reflection and room-light glints, meniscus tilt at the walls, dipper contact rings,
            the water band behind the near glass, glass tint, the lamp's beam (dark only)
front SVG   far caps, motor cable over the bridge, far pillar and spring, bar, dippers, motor, near spring, near rails,
            glass bottom, water lines, near glass, near caps, motor cable on the rim, near corner post, near pillar, beam
```
Two canvases sit at two depths because there are two effects at two heights: light on the paper (under the legs), and the water surface (inside the tank). The control box stands on the base beside the tank, below the water canvas, so it lives in the middle SVG; the near corner post is painted after the caps it stands on.

CSS for the stack (adapted from `ripple.css`):
```css
.fx-stage { position: relative; touch-action: pan-y; }
.fx-stage > svg, .fx-stage > canvas { display: block; width: 100%; height: auto; }
.fx-stage > .fx-layer { position: absolute; inset: 0; height: 100% !important; pointer-events: none; }
.iso:not([data-theme="light"]) .fx-gl { mix-blend-mode: screen; }
```
Screen blending can only brighten. A canvas that must also darken what it covers (the ripple tank's water, which tints the paper seen through it) stays on normal premultiplied blending in dark too: only `.rt-paper-gl` is screen-blended.
If the front SVG must take hover (the Raptor's parts do), give it `pointer-events: none` and its items `pointer-events: visiblePainted`, so the canvas and empty areas never catch the pointer.

## 3. The camera contract

`kit/gl.mjs` prepends `GLSL_ISO` to your fragment. It declares the uniforms and functions that make the shader see the same camera as `iso()`:

| Name | What it is |
|---|---|
| `uCanvas` | Canvas size in device pixels. |
| `uView` | The viewBox size `[W, H]`. |
| `uOrigin`, `uK` | The projection origin and the screen units per world unit (`glCamera(P).k`, equal to `scaleOf(P)`). |
| `uCam` | `(sinA, cosA, sinE, cosE)` of the azimuth and elevation. |
| `uTime` | The time you pass to `draw`. |
| `viewOf(gl_FragCoord.xy)` | Device pixel to viewBox point (it flips y). |
| `project(vec3 p)` | World to viewBox, identical to the kit's `iso()`. |
| `towardViewer()` | The unit 3D view direction, `(cosA·cosE, sinA·cosE, sinE)`. Every `p + t·towardViewer()` lands on the same pixel. Not the same as the kit's JS `towardViewer(P)`, which returns the 2D floor direction. |
| `rayOf(vb)` | The world point under `vb` on the vertical plane through the origin that faces the camera, so `rayOf(vb) + t·towardViewer()` is the pixel's ray. |
| `pixelOf()` | World units per canvas pixel across the screen, `(uView.x / uCanvas.x) / uK`. |
| `lightDir()` | The kit's light, equal to `lightOf(P)`, so shader shading agrees with the SVG tones. |
| `onFloor(vb, z)` | The world point on the horizontal plane at height `z` under this pixel. |
| `onWallX(vb, x)`, `onWallY(vb, y)` | The world point on the vertical plane `x = const` or `y = const` under this pixel. |
| `hash12`, `hash13`, `noise2`, `noise3`, `fbm2`, `fbm3` | Hashes and value noise, plus 4-octave fbm. |
| `window(x, a, b, soft)` | 1 inside `[a + soft, b − soft]` and 0 outside `[a − soft, b + soft]`: a smoothstep `2·soft` wide centred on each bound, so it is 0.5 at `a` and `b`. |
| `tonemap(c, exposure)` | `1 − exp(−c·exposure)`. Use it instead of `clamp`. |

`onFloor` and the wall functions round-trip exactly with `iso()`: `project(onFloor(vb, z)) == vb`. That is what lets shader content register with SVG lines.

**Why world space.** The figure's perspective is the projection. If you draw an effect in screen space (an ellipse sized by eye, a glow column in pixels), it can't agree with the drawing. It floats above the floor, runs through the deck, or keeps the same shape when the camera changes. Build everything from world points and world sizes. Screen sizes come only from world units times `uK`.

Antialiasing in world units: `pixelOf()` is world units per device pixel across the screen. On the floor in the depth direction, divide it by `uCam.z` (sinE), and up a wall by `uCam.w` (cosE). Soften registered edges by about `0.7 * px`, so they come out one pixel soft: crisp like the SVG, but not aliased. Where refraction stretches the image, a fixed softness stair-steps; the ripple tank enables `GL_OES_standard_derivatives` and widens its shadow penumbra by `length(fwidth(q.xy)) / length(fwidth(rest))`, the shifted point's footprint over the unshifted one, which is exactly 1 at rest, with an `#ifdef GL_OES_standard_derivatives` fallback.

## 4. Ray-marching in the iso camera

The camera is orthographic with uniform scale: both rows of the projection are orthonormal. Three things follow.

- **Every pixel is a straight ray** `p(t) = p0 + t·V`, where `V = towardViewer()` and `p0` is any world point under the pixel. Take `p0 = onFloor(vb, h)`, or the kit's `rayOf(vb)`, the point on the vertical plane through the origin that faces the camera (the Raptor's ray state starts there: `S.g0 = rayOf(vb)`). Larger `t` is nearer the viewer.
- **Distances to points are screen distances.** Screen distance divided by `uK` is the exact perpendicular distance from the pixel's ray to a world point. A spherical flash, shock shell or glow ball can be computed from that distance and still be world-correct. The closest-approach parameter is `tc = dot(C − p0, V)`.
- **Visible fraction behind an occluder.** If the first opaque surface along the ray is at `tLow`, a Gaussian centred at `tc` with width `sigma` is visible by `1/(1 + exp(2.41·(tLow − tc)/sigma))` (the Raptor's `beyond`). That gives a soft cut, but a geometric one.

To march a volume:
1. **Find the first opaque hit** along the ray: the floor, plus any heightfield the SVG draws. The Raptor steps 28 times from above the flame cone's tip down to the deck and then bisects 7 times on a `ground(r)` heightfield built from the same numbers as the SVG lathe profile. Only the part of the ray in front of the hit (`t > tLow`) is air.
2. **Bound the march** with a simple shape (the Raptor uses a radius-132 cylinder from the ground hit up to the exit plane). Window every term to exactly zero before that bound, so the bound is only an optimisation.
3. **Sample with dither.** Use `t = ta + (i + hash12(gl_FragCoord.xy))·dt`, so 28 samples don't band.
4. **Normalise each term by its chord**, so the centre pixel gets the intended brightness whatever the step count. For a vertical-axis volume `r² = u² + (t·cosE)²`, so a Gaussian of radius `σ` integrates to `σ√π / cosE` along the ray: weight each sample by `dt·cosE/(σ·1.772)`.
5. **Expect a uniform cylinder to look softer.** Its projected profile is sqrt-shaped, so it loses the defined edge a flat drawing had. In dark, add a limb shell (the Raptor's orange shear layer at `0.9·Re`) to bring the edge back. In light the same shell inks an outline round a hollow flame, so there you weight the body toward the axis instead (section 8.4).

Some terms don't need marching:
- A thin horizontal disc seen along the ray is an exact ellipse. The Raptor's Mach disks stay analytic: `u² + ((s0 − sc)/tanE)² < rm²`.
- A thin spherical shell integrated along a ray is limb-brightened: brightness is proportional to `rs·w / sqrt(rs² − d² + 2·rs·w)`. Square the normalised profile, or the inside reads as a milky bubble.

## 5. Registering shader content to the SVG

Anything that lies on a plane must line up exactly with that plane's SVG lines.
- **Same numbers.** Pass the SVG's plans and heights to the shader through `DATA` (the ripple tank passes `inside`, `outside`, `paper`, `base`, `round`, `z` and the plans of everything that casts a shadow), and hand them to `glLayer` once as `fixed` uniforms (its `FIXED`). Test against the same rounded rectangles with a signed-distance `sdRect`, so a lit paper corner ends on the drawn corner radius.
- **Same plane.** The ripple tank uses `onFloor(vb, uZ.x)` (the paper), `onFloor(vb, uZ.z)` (the water) and `onWallX(vb, uNear.x)` and `onWallY(vb, uNear.y)` (the two near glass walls) for the water band behind the glass. Never use screen offsets.
- **Seen through an opening, masked twice.** Content behind an opening in a nearer plane (a sight glass, a window, a recess, a gauge face under its bezel) is masked by its own plane (`onWallY(vb, back)`) and by the opening tested in the opening's plane (`onWallY(vb, face)`), the two soft masks multiplied together. Masked only in the back plane, a blind-test sight glass spilled its oil and glass about 0.4 px past the top rim onto the body face, and at the bottom the offset showed a strip of bare body inside the opening. Draw the recess's inner walls in the front SVG, shaded by facing (`craft.md`, section 10), so the offset between the two planes reads as wall, not as bare face.
- **Only the deviation moves.** If an effect displaces what you see (refraction through ripples), apply only the part that differs from the rest state. The ripple tank computes the sight line bent by the rippled surface and by flat water, and applies only the difference. At rest, the paper under the water lines up with the SVG paper. The shift is also faded to zero wherever the sight line to the paper passes within a few units of a leg (`legClear`), so refracted paper never slides out from behind a leg that the middle SVG draws over it.
- **Heightfields match the lathe.** The Raptor's `ground(r)` is the deck disc plus the flame-cone profile, with the same radii and heights as the `G.solidOf` profile in `build.mjs`. Flow behind the cone is hidden geometrically, and hard edges appear only where the SVG draws a real silhouette.

Check registration at 4× (section 9).

## 6. API and a minimal page

```js
import { GLSL_ISO, glCamera, glLayer, smooth, burst, settle } from "../../kit/gl.mjs";
import { kitScript, glScript } from "../../scripts/inline-kit.mjs";
```

| Function | What it does |
|---|---|
| `glCamera(P)` | `{ origin, k, cam }` for a kit projection. Works in Node and the browser. Put it in your page `DATA`. |
| `glLayer({ canvas, fragment, view, camera, uniforms = [], fixed = {}, extensions = [], maxRatio = 1.5, onError })` | Compiles `GLSL_ISO + fragment` on `canvas` with a premultiplied-alpha WebGL1 context. `view` is `[W, H]` of the viewBox, `camera` is `glCamera(P)` and `uniforms` lists the uniform names you send per frame. `fixed` is `{ name: value }` for uniforms set once at creation (geometry, theme constants). `extensions` lists GLSL extensions (`["GL_OES_standard_derivatives"]`); each one the context has is enabled and prepended as an `#extension` directive. Returns `{ gl, extensions, set(values), draw(values, time), clear() }`, where `extensions` lists the ones actually enabled, or `null` when WebGL is missing or the shader fails to compile (`onError`, which defaults to `console.error`, gets the log). |
| `set(values)` | Sets uniforms without drawing. |
| `draw(values, time)` | Resizes the canvas to its CSS box × `min(maxRatio, devicePixelRatio)`, sets the camera uniforms, sets `uTime = time`, sets each value in `values` and draws one full-screen triangle. |
| `clear()` | Clears to transparent, only if something was drawn. |
| `smooth(a, b, x)`, `burst(age, rise, fall)`, `settle(state, target, dt, seconds, calm)` | The JS envelopes every shader uniform should ride on (section 7 and `motion.md`). `settle` returns true while still moving. |
| `glScript()` | The source of `kit/gl.mjs` with `export` stripped, for inlining into a standalone page after `kitScript()`. Live code on that page uses `glLayer`, `smooth`, `burst` and `settle` directly and must not declare its own. |

A uniform value (in `fixed`, `set` or `draw`) can be a number (float), an array of length 2–4 (`vec2`–`vec4`), or an array or `Float32Array` longer than 4, sent with `uniform1fv` to a `uniform float name[N]`: a 24-sample history is one `uH[24]`. There are no ints, matrices or samplers through these calls. For more data than a uniform array holds, use a texture: a `sampler2D` reads unit 0 by default, so create and update it on `layer.gl`. The ripple tank packs a 32 × 2 history into an RGBA8 texture, 16 bits per value (`PACK`, `put`, `unpack`), one texture per layer because each layer has its own context. Unknown names and uniforms the compiler optimised away are skipped.

Your fragment declares its own extra uniforms and a `main()` that writes **premultiplied** `gl_FragColor`. The canvas must cover exactly the same box as the SVGs, with the same aspect ratio, because `viewOf` assumes one uniform scale.

A minimal page: a hot plate glowing on top of a square deck. The plate registers with an SVG ring of radius 40 (`k.ring(0, 0, 40, 0, P, 72)`), and the glow stops at the deck's rounded edge, since there is nothing beyond it to light.

`build.mjs`:
```js
const P = k.fitProjection(points, W, H, { pad: 28, azimuth: 40 });
const DECK = { x: -90, y: -90, w: 180, d: 180, r: 8 };
const DATA = { W, H, camera: glCamera(P), plateR: 40, deck: [DECK.w / 2, DECK.r] };
const backSvg = k.figureSvg({ width: W, height: H, label: LABEL, body: back });
const frontSvg = `<svg xmlns="http://www.w3.org/2000/svg" class="iso-svg fx-layer" viewBox="0 0 ${W} ${H}" aria-hidden="true">${front.join("")}</svg>`;
const stage = `<div class="fx-stage" tabindex="0">${backSvg}<canvas class="fx-layer fx-gl" aria-hidden="true"></canvas>${frontSvg}</div>`;
const page = k.pageHtml({ title, theme, body: `<style>${CSS}</style>${k.plateHtml({ ...card, body: stage })}`, script: kitScript() + glScript() + LIVE.replace("__DATA__", JSON.stringify(DATA)) });
```

`live.js`:
```js
const D = __DATA__;
const stage = document.querySelector(".fx-stage");
const still = matchMedia("(prefers-reduced-motion: reduce)");
const LIGHT = document.body.dataset.theme === "light" ? 1 : 0;
const FRAGMENT = `
uniform float uHeat;
uniform float uLight;
const float R=${D.plateR.toFixed(1)};
const vec2 DECK=vec2(${D.deck[0].toFixed(1)},${D.deck[1].toFixed(1)});
float deckDist(vec2 p){vec2 q=abs(p)-DECK.x+DECK.y;return length(max(q,0.))+min(max(q.x,q.y),0.)-DECK.y;}
void main(){
  vec2 vb=viewOf(gl_FragCoord.xy);
  float px=pixelOf()/uCam.z;
  vec3 p=onFloor(vb,0.);
  float r=length(p.xy);
  float disc=1.-smoothstep(R-.7*px,R+.7*px,r);
  float onDeck=1.-smoothstep(-.7*px,.7*px,deckDist(p.xy));
  float halo=exp(-pow(r/(R*1.6),2.))*onDeck;
  vec3 col=tonemap(vec3(1.,.42,.12)*(disc*.9+halo*.35)*uHeat,1.2);
  float a=max(col.r,max(col.g,col.b));
  vec4 dark=vec4(col,a);
  vec4 light=vec4(vec3(.62,.24,.08)*a*.8,a*.8);
  gl_FragColor=mix(dark,light,uLight);
}`;
const layer = glLayer({ canvas: stage.querySelector(".fx-gl"), fragment: FRAGMENT, view: [D.W, D.H], camera: D.camera, uniforms: ["uHeat"], fixed: { uLight: LIGHT } });
if (layer) stage.setAttribute("data-gl", "");
const heat = { x: 0, v: 0 };
let target = 1, clock = 0, last = 0, frame = 0, visible = false;
function tick(now) {
  const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
  last = now;
  clock += dt;
  const moving = settle(heat, target, dt, 0.35, still.matches);
  if (layer) heat.x > 1e-3 ? layer.draw({ uHeat: heat.x }, still.matches ? 0.8 : clock) : layer.clear();
  frame = visible && moving ? requestAnimationFrame(tick) : 0;
}
const run = () => { if (!frame && visible) { last = 0; frame = requestAnimationFrame(tick); } };
new IntersectionObserver((entries) => { visible = entries[entries.length - 1].isIntersecting; if (visible) run(); }, { rootMargin: "120px 0px" }).observe(stage);
```
Without `uTime` animation this loop stops once the heat has settled. With an animated field (flicker, flow), keep running while the effect is above its activity threshold and visible, and stop otherwise.

## 7. Craft rules, each with the bug that taught it

### Nothing pops
Every uniform moves along a smooth envelope in time, with an attack and a decay. Never set a value to 1 in a single frame.

- **One-shot events use `burst`.** The first ignition flash went from 0 to full in one frame and whited out the card. Now every event (flash, shock, ground ring, shutdown pop, card shake) uses the kit's `burst(age, rise, fall)` from `kit/gl.mjs`, a smoothstep attack then an exponential decay:
  ```js
  const flash = calm ? 0 : burst(sinceMain, 0.08, 0.13);
  ```
  with a 50–90 ms attack (zero slope at onset). An alpha function `t·e^(−t)` ties the decay to the rise and stays white too long. Keep whole-card flashes low too: a flash plus a CSS veil at 0.9 opacity whited out the whole card for six frames, so the veil is now `flash × 0.32`.
- **Phase changes are crossfades driven by the physical quantity.** A separate preburner term that died while the main plume grew read as an instant colour switch. Now colour, brightness, raggedness, tongue length and splash are all functions of chamber pressure, and pressure moves on a spring toward each phase's target.
- **Colour gets its own slow envelope.** Even when driven by pressure, the orange-to-violet change crossed its smoothstep window in about 0.23 s on a fast spring, and brightness dipped 33% and came back inside the 130 ms flash: a cut hidden by a flash. The fix:
  - Hue has its own critically damped envelope, the kit's `settle`: `settle(burn.hue, lit && burn.pc > 0.2 ? 1 : 0, dt, lit ? 0.32 : 0.5, calm)`, sent as `uHue`.
  - Brightness is a separate `uGlow`, computed in JS so it falls monotonically from 1.15 to 1.0 as hue rises.
  - The pressure spring is critically damped (stiffness 20, drag 2√20), so it doesn't overshoot to 318 bar.
- **Brightness during a fade-out only falls.** On shutdown the plume flared back up as an orange flame, because `mix(a, b, hue)` with `b < a` rose again as hue decayed. Now the glow level freezes when shutdown starts and is then scaled only by quantities that fall.
- **Recheck neighbours when you change a spring.** Once the pressure spring was made critically damped, the preburner tongue nearly vanished. That leg needed its own stiffer critical spring (stiffness 64).
- **One-shot flags belong to the event.** Re-igniting within 2 s of a shutdown lost the next shutdown pop, because the pop's clock was only reset after a quiet period. Now a per-burn `popped` flag decides it. Resetting the clock in `ignite()` would have cut off a pop that was still fading.
- **Changes travel through the medium.** When the ripple tank's frequency changes, the new wavelength doesn't re-lay the whole field at once. Each frame the page records the source's true phase, ω, k, damping per mm and 1/group speed, keeps 32 samples 50 ms apart, and uploads them as a 32 × 2 texture to both shaders. Each point finds its age `r / groupSpeed` with one fixed-point step on its own history's group speed, and reads k and damping from that age, so the new wavelength spreads outward from the dippers the way it really would.
- **Keep the phase continuous through a change.** The first version used `θ = k(age)·r − φ(t)`. During a sweep the term `r·dk/dt` made far crests race and alias, and damping the amplitude while the frequency moved (`calm`) blanked the pattern: mean paper brightness dipped 17%. Now `θ = kE·r − uPhase + Ψ(age)`, where `Ψ` is the phase the source gained since that sample beyond what its ω predicts (`lag` in `packHistory`). At a fixed point the crests then move only as fast as ω changes, so no amplitude damping is needed, and a small motion-blur term (`gMove`, how far crests move per frame) softens the caustics during fast changes.
- **First light fades in, and in light it only brightens.** The ripple lamp fades in over 0.9 s on first view (`smooth(0, 1, (lampClock − 0.15) / 0.9)`). In dark the SVG plain-water fill fades out on the same value through a CSS variable (`--rt-lamp`). In light the first version started from bare paper, so the troughs' ink made the tank go darker as the lamp came on, and the plain-water fill darkened it further. Now the water fill is hidden whenever WebGL runs, and the paper is drawn from the first frame (lamp 0) at its unlit grey (`off`) and mixed toward lit ink by `uLight`, so first light only brightens. For an emitter on paper the same rule reads the other way round: its ink only grows as the light comes in, and the scene never goes pale first or dips dark (section 8.2).

### No hard spatial cut-offs
Every term reaches zero before any boundary: an `if`, a bounding shape, a plane, or the canvas edge. Use `if` only as an optimisation where the term is provably about zero.
- A bead term was cut off sharply at both ends of its range by an `if`. It now fades in and out with smoothstep.
- The plume had an `if` box, a `s < .01 continue` and a `step(ax, R0)` halo mask. All three showed as seams.
- A screen-space glow column ran from the nozzle to the bottom of the canvas. The last rows of the canvas still had alpha 11–19/255, so it ended in a straight horizontal line under the deck. It is now a world-space halo bounded by the ground, and the last 10 rows and columns read exactly 0.
- A wide `exp()` flash and the floor-coordinate steam were still visible at the canvas edges. Both now fall to zero inside the canvas.
- Cell noise looks only inside its own cell, so a feature whose offset plus radius plus soft tail reaches past the cell's half-width is clipped flat at the cell edge. Keep `offset + wobble + radius + tail` under half a cell, or test the 3 × 3 neighbouring cells. A blind-test bubble field let a centre sit 0.31 of a cell off-centre with a reach of 0.86 against a 0.85 half-cell.
- Radial noise that used `atan` had a seam along one angle. Use `noise3(vec3(dir, r·k − t))` of the direction vector instead.
- A CSS `radial-gradient` with percentage stops ended outside its element. Use `circle closest-side` so the gradient ends inside it.

### World space, in the camera
- The ignition shock was a 2D circle around the projected exit. Its lower arc ran over and under the deck and painted over both columns. It is now a real sphere: each pixel ray hits it at two roots, roots below the ground are hidden softly, each root is limb-weighted, and each one goes to the back or front canvas by the engine proxy.
- The floor splash was laid flat over the deluge ring, so the tube lost its volume. The ring is now a torus occluder, lit with `max(dot(n, toPlume), 0.)`.
- Haze behind the near column painted over its front face. Columns are now opaque boxes in the shader.

### Layering and blending
- In the dark theme, canvases that only emit light use `mix-blend-mode: screen` and write premultiplied colour with alpha = the largest channel. A canvas that also darkens what it covers stays normal-blended: the ripple tank's water tints the paper seen through it, which screen blending would silently drop. In the light theme every canvas is normal-blended, and its ink is solved from the colour the paper should show (section 8), not darkened from the dark-theme colour.
- A blurred orange rim drawn in normal blend over the white-hot core read as a dim brown collar under the nozzle lip, and drawn slightly off the lip it made a double arc. Glow outlines over a shader now go in their own overlay SVG (`.rp-glow`), screen-blended in dark, at the exact lip. In light nothing can add light, so a rim is part of the ink itself: a saturated band at the body's edge around a pale core (section 8.1).
- Steam that should pass in front of the bell was always behind it. It is now split across two canvases with the engine proxy (section 2).

### Tonemap and dither
- Use `tonemap()`, never `clamp`, so bright cores roll off instead of flattening.
- Without output dither, the large soft terms became 8-bit staircases (runs of 18–34 px per level), and screen blending made them worse. Dither after tonemap with about ±1 LSB of animated triangular noise, gated so empty pixels stay exactly 0 (the Raptor's `inkOf`):
  ```glsl
  col=tonemap(col,1.15);
  float m=max(col.r,max(col.g,col.b));
  float n=(hash12(gl_FragCoord.xy+fract(uTime*.61)*91.)+hash12(gl_FragCoord.yx*1.37+fract(uTime*.37)*57.)-1.)/255.;
  col=max(col+n*smoothstep(0.,1.5/255.,m),0.);
  ```
- Band-limit fine detail. The ripple tank fades out each wave whose projected wavelength drops below 2.5–5 device pixels (its `fine` term). Otherwise a 28 Hz pattern turns into moiré.

### Render only while it matters
- Cap the device ratio at 1.5 (the `maxRatio` default). Draw only while the effect is active: the Raptor clears both canvases and lets the loop stop once every envelope is below 1e-3. Stop offscreen with the same `IntersectionObserver` as the rest of the figure, and redraw on `resize`.
- An effect that never rests (bubbles in a returning flow, a pilot flame) keeps the shader drawing at idle, but not the rest of the figure: split the loop so the SVG work (depth sort, gauges, readout) runs only while something in it changes, and run the effect alone, at a reduced rate if it is slow. A blind-test bench kept its whole tick at 60 fps forever because one envelope settled at 0.54 instead of 0.
- Gate costly terms with uniform checks (`if(uSmoke<.0004&&uVent<.0004)return vec3(0.);`). These are the same for the whole draw, so they cost nothing when off.

### Reduced motion
Jump straight to a representative steady state (the Raptor jumps to main stage, and the ripple tank to its current f and d), freeze `uTime` (the Raptor uses 0.8), zero every event envelope (no flash, shock or shake), and redraw only when the JSON of the uniforms changes. The figure must still respond to input.

### Fallback
`glLayer` returns `null`, so guard every call. Mark the stage (`data-gl`) when WebGL runs so CSS can swap between the SVG fallback and the shader. Test by stubbing WebGL (section 9).

## 8. Light theme

Light is the shipping theme. A page that is light gets shaders designed for paper, not dark-plate glows converted at the end. Seven figures on the isometrics site were built on a dark plate and then converted: a pool reactor, a foundry pour, an afterburner, a glory hole, a tokamak, a plasma cutter and a forge. Each conversion took three review rounds, and every rule in this section is one of their findings.

The Raptor's light plume (`inkOf` in `examples/raptor-engine/live.js`) shows what not to do. It multiplies every hue by a fixed dark red, `vec3(.62,.34,.3)`, and sets alpha to brightness, so the hottest gas comes out darkest and most opaque. Reviewers called it "red ink smoke". Don't copy it. The ripple tank's grey ink (`uPale`) is still a good model for light that isn't a glow: a tone curve `1 − uExpo.x/(1 + (e/uExpo.y)^uExpo.z)` takes the crests to near paper-white and the troughs to the kit's greys, and its water writes only the change from flat water.

### 8.1 Ink on paper
Over white, the canvas is normal-blended, premultiplied, and never `screen`, because screen over white is white. No pixel can be brighter than the paper, so light has to read through **saturation and local contrast**:
- **A glow is a saturated body, a pale near-white core and a coloured rim.** The pale core reads as hot only because the body around it is saturated and the ground around that is darker.
- **What the light falls on takes a tint** (multiply-like, with capped alpha). It doesn't get brighter.
- **The volume that holds the light is darker, for a physical reason:** soot, refractory, deep water or shadow (section 8.3). It is never a black box.

**Design the colour the paper should show, then solve for the ink.** Write a ramp from light energy to the colour the reader should see over white. Then invert normal compositing over white, which shows `rgb + (1 − a)`:
```glsl
vec4 inkFor(vec3 seen,float cover){
  vec3 absorbed=1.-seen;
  float a=max(max(absorbed.r,max(absorbed.g,absorbed.b)),cover);
  return vec4(vec3(a)-absorbed,a);
}
```
Over white this composites to exactly `seen`.
- **Reactor.** `inkOf` does this with its `paperOf(e)` ramp, where `e = 1 − exp(−2.3·max(lin))`. The ramp runs white → `TINT (.85,.93,1)` → `BODY (.2,.46,.94)` (by e .12–.5) → `RIM (.2,.68,1)` (.4–.65) → `ICE (.9,.98,1)` (.6–.9).
- **Afterburner.** Its `paperOf` mixes white → body → a pale core, and then inverts the same way.
- **`cover` is an alpha floor at the hot end:** `WASH·smoothstep(.6,.9,e)` with WASH .52 in the reactor, and `COVER·smoothstep(.45,.92,m)` with COVER .94 in the afterburner. Without it, a near-white core has alpha near 0, so the SVG lines behind it show straight through and it reads as a hole, not as light.

The other figures use the form `vec4(colour·a, a)`: `colour` comes from a ramp and `a` from a saturating curve of energy. The tokamak's `glowInk` is an example. Both forms work; `inkFor` also guarantees the colour the reader sees.

**One ink per kind of light, composited in the shader.** Keep separate accumulators, give each its own ink, and stack them back to front with `vec4 over(vec4 top,vec4 under){return top+under*(1.-top.a);}`:

| Term | Ink | Seen in |
|---|---|---|
| Emission, the hot thing itself | ramp body plus a pale core, alpha up to .96–.97 | plasma cutter `bodyInk`, forge `barInk`, tokamak `glowInk` |
| Light falling on a surface | hue pulled warm or cool, alpha capped at .5–.8 | foundry `castInk` and `tintInk`, tokamak `tintInk` (.5), plasma cutter `tintInk` (.75–.8) |
| Haze: steam, smoke, fume, breakdown gas | grey or lightly tinted, alpha `1 − exp(−density·k)` | plasma cutter `hazeInk`, tokamak `gasInk`, forge steam |
| Sparks and streaks | on top of everything | foundry and forge `sparks` |

The plasma cutter's `lightInkOf` is the clearest example: `over(arcCore, over(bodyInk(body), over(arc, over(fumeInk, over(steamInk, tintInk(tint))))))`. If you sum the energies first and ink the sum, everything gets one hue and the pale core is lost.

**Finish.**
- Clamp `rgb ≤ a` after every sum: the afterburner showed a 1-LSB premultiply overflow.
- Dither the final premultiplied value, rgb and alpha both, gated by `smoothstep(0.,1.5/255.,a)` (the glass furnace's `paperOut`, the foundry's `dithered`).
- Keep the canvas edges at exactly 0.
- Tonemap energy before it reaches a ramp (`tonemap`, or `1 − exp(−k·e)`), never `clamp`.

### 8.2 Ink grows with light
In dark, more light only adds. On paper, more light means more ink, and a careless formula can add ink while the light falls or take ink away while it rises. These were the faults reviewers found most often:
- **Fade the finished ink once.** Scaling a premultiplied ink by `u` is an exact linear crossfade from bare paper to the full picture, monotonic in every pixel: the reactor's `ink*=uFirst`, the glass furnace's `paperOut(overInk(gas, ink)*uLit)`. The glory hole first faded each term before inking it (`*uLit` in `holeOf`, the port bore and the ball). The pale core arrived before its saturated body, then the ground darkened, so first light went pale and then dark. Build every light ink from unfaded tones (its `litOf()` returns 1 in light) and fade only the result.
- **Fade a glow by energy and alpha together.** If you fade only alpha, a pale core stays pale and the thinning glow turns milky grey: the tokamak's contact band did. If you fade only energy, the glow runs back down its ramp. The tokamak now scales energy down to a floor and alpha the rest of the way: `hot+=glowInk(c*level*(.3+.7*hold))*hold*(1.-hot.a)`. The band stays pink-mauve as it goes.
- **Take the hue at steady brightness.** As energy falls, the tonemap reddens and darkens the hue, and with alpha near its cap the ink rises: the afterburner's cavity got 55% heavier as the flame went out. The fix:
  - Divide the burn by the local lit strength before taking the hue (`burn/lit`, `lit = la.x·glowOf(la.y)/LIT_GLOW`).
  - Blend the hues by each part's share of the energy.
  - Tie alpha to energy: `cap·(1 − exp(−1.6·e/cap))`.
  - At low cover, warm the hue toward `EMBER` and scale alpha by `inkOf(hue)/inkOf(ember)`, so the ink stays the same.

  With that, the window's ink fell monotonically from 1894 to 155 on the cut.
- **Hand-offs are plain premultiplied mixes.** The afterburner's jet has two copies, dry blue and lit warm, crossfaded with `mix(blue, warm, lift)`. A `max(alpha)` cover there doubled the dry jet's ink. Also gate each copy so it can't pick up the other's terms: the dry shock disc leaked into the warm copy as a cream patch.
- **Envelopes that multiply can rise during a fade.** If a first-light envelope is still rising while the effect's own edge falls, the fade-out brightens: the tokamak did this when Home was pressed in its first half second. Freeze first-light growth while the effect is falling.
- **The handover from the SVG fallback is exact,** or the scene dips while the shader comes in (section 8.5).

### 8.3 A ground for the light
A glow on white with nothing around it reads as a stain. Give it the ground it really has:
- **Cavities in the SVG take a darker token set.** A group class redefines the kit tokens for everything inside it:
  - the afterburner's `.ab-inside`: paper `#4a453f`, shades `#3c3833` to `#544e47`, lines `#5e5952` to `#9b958b`;
  - the tokamak's `.tk-cavity`: `#66646e` to `#8b8992`;
  - the foundry's sooted bore tones (`--fd-bore-*-sooted`, a `color-mix` of `--anatomy-shade-0` with `--fd-soot-ink: #2f2a25`);
  - the glory hole's `--gf-void` and `--gf-bore`;
  - the reactor's water-column gradient (`--rx-water-top`, `-mid`, `-deep`, from `#edf2f4` to `#c9d7e0`).

  In dark the same tokens are set to `inherit`. All four shade tones stay, so the inside still has form and reads as refractory, soot or deep water rather than a hole.
- **The ground takes the light's tint, live.** The same tokens `color-mix` toward the light's hue by a variable the loop writes: `--ab-flame` at 30% on faces and 55% on lines and back plates, `--tk-cold-mix` for the violet of breakdown. The SVG ground then warms with the flame even without WebGL.
- **The shader's ground equals the SVG's.** Where the shader paints an opaque cavity (the glory hole's `holeOf = overInk(emberInk(t,1.), vec4(SOOT,1.))`), read the token rather than retyping it. The foundry adds a probe path, sets its fill to each `var(--fd-…)`, reads `getComputedStyle(probe).fill` and passes the result as a `fixed` uniform (`paintOf` in its `mount.ts`).
- **Light around the source is a tint with a long tail.** The tokamak's vessel read grey at flat-top until a long-range wash (`skin*(.14*exp(-gap/(.22*U))+.17*exp(-gap/(.9*U)))`) tinted its walls magenta. Measured on the column: `#6e6c76` with the plasma off, `#786778` with the wash at .17, which reads as lit. At .1 it still read grey.
- **No bloom in empty air.** In light, a halo around a stream or sheet hanging in the air is a smudge on the paper (the foundry's stream and notch glows were). Drop halos in air in light. Keep the limb rim inside the body, and the tint on the surfaces the light reaches.
- **Light only lands where the source can see.** The reactor's surface light leaked onto a shield's cut face and coping until its seal test used the true top of the tank lip (`RIMZ`) rather than the deck. The plasma cutter lit the slat bay under the jet's lower end while the jet was visibly in the next bay; its `bayLight` now lights each bay by its share of the jet column, so the light crossfades from bay to bay as the jet moves.

### 8.4 Recipes by phenomenon
Hot bodies run dull red → orange → yellow → near-white, in that order and no other. Keep one stop table in the model, compile it to GLSL and use the same table for the fallback: the foundry's `INK_STOPS` (`inkRampGlsl`, and `glowInk` for the SVG), the forge's `STEEL_INK` (`steelInk`, `steelInkOf`).
- **Molten metal, forged steel, billets, kerfs.**
  - Index the table by brightness temperature, so a dim far wall and a bright pour read from the same table. The foundry's `brightOf` inverts Wien from the green channel.
  - The forge's stops: 560 °C `(.45,.09,.05)`, 680 `(.56,.07,.04)`, 780 `(.78,.10,.03)`, 950 `(.95,.38,.06)`, 1150 `(1,.78,.12)`, 1260 `(1,.93,.78)`.
  - Alpha opens at visible glow: `steelGlows` over 560–820 °C, or the foundry's `INK_FROM`/`INK_FULL` over 470–590 °C, crossfading from the metal's own ground tone.
  - The rim is cooler and darker (`steelInk(t − 140)`, the foundry's 150 K limb drop). A cream core appears only in the hottest centre (`steelPale` over 1050–1280 °C, at .8).
  - A dark skin sits under the glow and grows as the metal cools: the forge's `barInk` runs from oxide `(.28,.29,.32)` to skin `(.24,.22,.23)` with cover up to .86.
  - Reviewers' words for the foundry stream: "a cream core with an orange skin".
- **Glass gathers.**
  - Alpha is a luminance gate: `GLOW_SEEN` .008–.028, about 600–710 °C.
  - The colour is `pow(hue, mix(1.6,.8,warm))*mix(.62,1.,warm)`, which runs deep cherry → red → orange.
  - The pale core (`CORE_PALE (1,.88,.62)`) sits where the pixel's own emission is brightest (`cored`, luminance .14–.24), not at a fixed radius.
  - The cool skin (`SKIN_COOL (.3,.11,.07)`) only recedes.
  - The first version had a salmon stage. A red mixed toward white is salmon, and no temperature looks like that: go through orange and yellow before anything turns pale.
- **Sparks.**
  - A saturated head and a darker, cooler tail. The plasma cutter's `tailCooling` sets the tail at 0.62 of the head's temperature; the forge's tail ink is `(.42,.07,.02)`, alpha .75 capped at .8. Only the youngest third of a spark gets a pale core (`u < .35`).
  - Keep the streak at least about 1.25 device px wide (`max(.35+.6px, 1.25px)`), or diagonal streaks bead at 10–24×.
  - On white a spark is ink, so over a hotter emitter it must vanish. The foundry multiplies spark ink by `1 − cover·smoothstep(sparkT−60, sparkT+60, tb)`, against both the metal and the glowing bore.
  - A spark that passes behind a jet fades by depth: `cover*smoothstep(-.4,.4,tJet-tS)`.
- **Flame and exhaust.**
  - A saturated orange-yellow body with translucent edges, weighted toward the axis in light: `mix(1.,1.3-.6*smoothstep(0.,rf,r),uLight)`.
  - Cut the limb mantle (`mix(.8,.3,uLight)`). The limb shell that gives a dark plume its edge (section 4) draws an outline on paper and leaves the flame hollow.
  - Keep raggedness to the outer part, widen the hot axis, and put the pale core at the nozzle, fading out over .4–.9 of the length.
  - Check a cross-section. The afterburner's final flame reads `0 2 13 33 31 49 78 72 78 52 40 32 8` in ink from edge to edge.
  - The dry jet is a faint blue column. Shock diamonds are pale cream discs (`CREAM (1,.95,.86)`) with a warm rim and a soft edge (about .65 of the disc radius). Pushed stronger, a second diamond read as a knob or a washer, so it stays a faint lens.
- **Smoke and steam.**
  - Light grey, never with a dark glow ink laid over it. Edge `(.84,.87,.90)`, crown `(.97,.975,.98)`, and the underside `(.72,.76,.81)` by the cloud's own self-shadow; alpha `1 − exp(−density·k)`.
  - Near a hot source, warm the cloud toward pale amber (`mix(steelInk(T), CREAM, .5)`). The forge first laid glow ink over the cloud, and reviewers saw a salmon blob.
  - A quench puff is a `burst` that leaves the tub: it climbs from 4 to about 40 units above the water, slowing as it goes (`PUFF` in the forge's `shader.ts`), clears the rim within about 0.15 s, widens from 7 to 13, thins as it grows, and shares the column's height window, so it never cuts off.
- **Plasma (tokamak).**
  - Saturated magenta-violet: `inkBody(chroma)=pow(chroma,vec3(1.8))*vec3(.93,.86,.97)`.
  - Above m .55 the core goes pale, toward white with 8% of the chroma kept.
  - Alpha: `.97*smoothstep(.02,.42,m)*(1.-exp(-2.6*(m+2.*m*m)))`.
  - Breakdown gas is a violet haze, `(.52,.44,1)` to `(.74,.68,1)`, alpha at most .64. Seen through the cutaway, the vessel's interior uses the cavity token set.
- **Cherenkov (reactor).**
  - Vivid cyan-blue in a blue-tinted water column: the paper ramp in 8.1, over the SVG water gradient, with the hot end floored by `WASH` so the core covers the fuel lines behind it.
  - Compress the pulse above critical (`1.+.75*log(level)`), so a 1.5 GW pulse brightens the pool without flooding the card.
  - The glow on the hall wall uses a separate, weaker ink (`HALL_INK` .35).
- **Plasma arc (plasma cutter).**
  - A violet arc `vec3(.6,.34,1.)` with a pale core `vec3(.96,.95,1.)`.
  - Warm tint ink on the plate and water: `mix(vec3(1.,.5,.08),vec3(1.,.68,.3),…)`, alpha `.75*(1.-exp(-3.2*sqrt(m)))`.

### 8.5 The SVG and the fallback in light
- **Read the theme once.** Choose the theme at build time (`pageHtml({ theme })`, or `data-theme` on the React wrapper; on the isometrics site the figure's build returns `theme` and `LivePlate` wraps a dark plate in its own `.iso[data-theme="dark"]`). The live code reads it from the nearest `[data-theme]` (`root.closest("[data-theme]")?.getAttribute("data-theme") !== "dark"`) and passes `uLight` as `fixed`. Gate every light-only change on `uLight` (`mix(dark, light, uLight)`, or a branch on the uniform), so dark output stays as it was. Keep dark working if that is cheap.
- **No colour literals.** If a figure hard-codes dark colours in its CSS, convert them to tokens with dark overrides: `.x { --x: light }` then `.iso[data-theme="dark"] .x { --x: dark }`. Shader constants that must match a token come from a probe (8.3).
- **Nested themes.** `.iso:not([data-theme="dark"]) .x` also matches through the light page around a dark plate. That is how the reactor's dark fallback got multiplied into black. Put the dark rule after the light one, at equal specificity, or key on the plate's own theme: the forge uses `.iso-plate[data-plate="light"]`.
- **How the fallback blends.** Multiply its saturated tints, which darken and leave the lines under them visible. Normal-blend its pale cores, because multiply can't lighten: the reactor's no-WebGL core is a radial gradient from `#fff` at .85 to `#dcefff` at .3, blended normally.
  - Fallback colours come from the same ramp as the shader: the foundry's `glowLightCss`, and the glass furnace's `gatherInk`, which mirrors the shader's gather ink exactly.
  - Opacity on a pale token fades to grey, so give the fallback the saturated hue: the tokamak's fallback rings set `--anatomy-lit: #e0409c` and `--anatomy-hi: #f27ec0`.
- **Fallback shape changes crossfade.** The tokamak's no-WebGL plasma ring is 11 precomputed rings (`LIMITED_RINGS`). Neighbouring rings are crossfaded with linear weights that sum to 1, on the same eased radius the shader draws. With 6 rings a doubled outline showed mid-fade.
- **Nothing with gaps goes under a translucent ink.** Whatever the SVG draws shows through an ink that isn't fully opaque. The forge's heat band was a set of strokes with paper between them, and it printed grey streaks through the bar's 14% transparency. It is now a face filled with the heat gradient.
- **The handover from fallback to shader is exact.** At first light the canvas ink rises by `u` while the fallback under it goes. A linear `1 − u` on the fallback makes the middle of the fade paler than either end. For an exact linear crossfade, set the fallback's opacity to `(1 − u)/(1 − u·α)`, where α is the canvas ink's alpha where it matters: the glass furnace's `--gf-gather-fade`, with α the gather's centre stop. Its other fallbacks stay at full strength until the shader is nearly in: `clamp(0, calc(var(--gf-fallback, 1) * 12), 1)`.
- **Every body the shader draws has a fallback.** The foundry's tap stream had none, so it was missing without WebGL. It is now an SVG outline rebuilt each frame from the same history the shader reads.

### 8.6 Where the ink meets a line
The canvas lies over the back SVG's 0.6 px strokes. In light, a misregistered edge shows as a pale gap or a coloured line over the stroke, rather than as a dark seam.
- **Stop at the stroke's inner edge.** Ramp a face's ink to full under the stroke and end it at the stroke's inner edge, half the line weight in: 0.3 vb for the kit's 0.6 px lines. Measure the computed stroke width; the forge's reviewer guessed 1.1. Examples: the reactor's `cutFade=smoothstep(-SEAM-.5*px,-SEAM+.5*px,gap)` with `SEAM=.3`, and the forge's `strokeMeet`. Pad a tint by the same half stroke (the foundry's `ringLight`), so it never paints over the line and leaves no pale band inside it.
- **Don't clear the ink where nothing is stroked.** The forge's veil cleared at the crease two faces share, which the SVG doesn't stroke, and white seams showed along the bar. Draw such a crease in the ink instead: ×0.72, at least 0.8 canvas px wide (sharper came out dotted).
- **Supersample where neighbouring rays disagree.** The reactor traces four sub-pixel rays and takes 16 samples when their hit, depth, kind, normal or cut share differ (`apart`), so cut-face edges don't stair-step. At dpr 1 a canvas pixel is about as wide as the stroke, so one pixel of softness remains.
- **A body that enters from out of frame** may fade in from the canvas edge in screen space. The foundry's tap stream fades in over 11 vb, starting 10.5 device px below the canvas top, so the edge still reads 0 and the stream reads as coming from above.

### 8.7 What the reviewers saw, and the fix

| You see in light | Cause | Fix |
|---|---|---|
| A red-brown "ink smoke" plume | Hue × a dark multiplier, alpha set to brightness | A paper ramp: saturated body, pale core, alpha floor (8.1) |
| A flame drawn as an outline, hollow inside | The dark limb mantle | Weight the body toward the axis, cut the mantle (8.4) |
| First light goes pale, then dark | Terms faded one by one before inking | Fade the finished premultiplied ink once (8.2) |
| A glow fades out through milky grey | Alpha-only fade of a pale ink | Dim energy and alpha together (8.2) |
| A cavity gets darker as the fire goes out | Tonemap reddens the hue while alpha holds | Hue at steady brightness, alpha from energy, ember match (8.2) |
| A salmon stage as glass or steel cools | Red mixed toward white | A blackbody-ordered ramp, pale only past yellow, core where emission peaks (8.4) |
| Sparks look like dirt on the metal | Dark ink over a hotter emitter | Drown sparks over hotter bodies; saturated head, darker tail (8.4) |
| Spark streaks beaded at zoom | Tails too thin | At least about 1.25 px wide; thin the tail less (8.4) |
| A spark shows through the jet | No depth test against the jet | Fade it by depth behind the jet (8.4) |
| A halo stains the paper around a stream | Dark-theme bloom in empty air | No halos in air in light (8.3) |
| A salmon blob in the steam | Glow ink laid over the cloud | Tint the cloud toward pale amber (8.4) |
| The vessel reads grey with the plasma on | Tint falls off too fast | A long-range wash term (8.3) |
| A pale fringe or white seam along an edge | Ink stops short of the stroke, or clears an unstroked crease | End the ink at the stroke's inner edge; draw the crease in ink (8.6) |
| A cream patch in the jet | One copy picks up another copy's term | Gate each copy (8.2) |
| The dark plate's fallback is invisible | A `:not([data-theme="dark"])` rule matched through the light page | Order the dark rule last, or key on the plate (8.5) |
| A grey, washed-out fallback glow | Opacity on a pale token | Saturated tokens; multiply tints, normal pale cores (8.5) |
| Grey streaks through a hot face | SVG strokes with gaps under a translucent ink | Fill the face (8.5) |
| Light on a face the source can't see | A surface light without the true occluder | Use the same geometry the SVG draws (8.3) |

## 8b. Shaders on a turning group
In a 3D figure (`references/3d.md`) parts move as rigid groups, so a shader has to follow them.
- **Anchor to a part**: `controller.anchor(group, restPoint, restDir)` gives `{ world, screen, direction }` every frame. Pass them as uniforms. That is how an LED rides a flexing fingertip, a flame stays on a ruptured line's end and a spark source sits on a knuckle.
- **Anchor to a surface**: prepend `glslPose("Dial")` (or `GLSL_TURN` for one group) and pass `TURN.poseUniforms(controller, "dial", "uDial")`. Then `vec3 q = toDial(onFloor(vb, z))` is the surface point in the group's rest-world coordinates. Oil pooling on a turntable, a texture revealed by light and a scorch mark all turn with it.
- **World space stays world space.** Smoke, sparks after they leave the source and oil on the bench never touch a group transform. Emit at the anchor's world position, then integrate in world.
- **Occlusion by live parts.** A front canvas sits over a live layer. Mask its ink with the parts painted after the effect's own part: `TURN.cover(controller, [partName])` → `uCoverEdge`, `uCoverCount`, then `ink *= 1. - coverOf(vb, .7 * px)`. The masks are the exact convex silhouettes pushed out 0.3 vb, so ink stops at the occluder's stroke. Six polygons of ten edges each. Read `MAX_FRAGMENT_UNIFORM_VECTORS` and keep the SVG fallback below 128.
- **A back canvas between live layers** needs no masks: put the surface that receives light in its own live layer under the parts that cover it (the turning dial's face under the hand).
- Vector uniform arrays (`uniform vec3 uSpark[24]`) take a `Float32Array` through `draw`.
- **The SVG fallback on a moving part** is a `T.billboard({ part, at, normal, svg })`. Its markup is in viewBox px centred on the projected point, not in world units, so scale world sizes by `G.scaleOf(P)`: a glow sized for a lens of radius 1.1 is `<circle r="${(1.1 * G.scaleOf(P)).toFixed(2)}"/>`. Written as `r="1.1"`, it draws a pin-dot about a third of the lens's size. Check it with `--no-webgl` and a 4× zoom.

## 9. Verify

Use `scripts/drive.mjs` for anything that moves; `capture.mjs` also has WebGL now, so it is fine for still shots of a shader page. `references/verify.md` has the usage. On top of the normal checklist:

- **A contact sheet across every transition:** start, each phase change, steady state and stop. Real-time sheets on a heavy WebGL page are too sparse, because a headless capture takes 60–200 ms and blocks animation frames. Slow the page's clock first with `["slow", 0.05]` (20× slow motion for `requestAnimationFrame` and `performance.now`), take the sheet, then `["slow", 1]`. To catch something that starts on the first frame, such as a first-light fade-in, pass `--preload` a file containing `window.__driveSpeed=0.04` so the page runs slowly from its first frame. Wait for a phase with `["until", "document.querySelector('[data-readout]').textContent.includes('main stage')", 8000]` rather than a guessed `wait`. Hold any idle tour off before each sheet: its resume timer runs on the wall clock, not the slowed one, so it can take the targets back mid-sheet. A key press holds it for a few real seconds; for a long slowed sheet, stub the timer with `--preload` and wait for rest first (`verify.md`, "Slow mode doesn't slow timers").
- **Numbers, not just pictures.** `canvas.getContext("webgl")` returns the page's own context. In a `requestAnimationFrame` callback, read uniforms with `gl.getUniform(gl.getParameter(gl.CURRENT_PROGRAM), location)` and read the mean canvas RGB with `gl.readPixels`. Readback only works in the same task as the draw, because `preserveDrawingBuffer` is off: all-zero frames from a separate task are a test artefact. Log one row per frame across a whole run. Hue and mean luminance should each move monotonically through a phase change, apart from deliberate bursts.
- **Edges read 0.** Read back the last 10 rows and columns of the canvas during every phase. Anything above 0 is a cut that will show under screen blending.
- **In light, log ink, not brightness.** Per frame, log the ink a region lays on white, `1 − lum(rgb + 1 − a)`, with the hook in `verify.md` ("Light-theme shaders"). Do it for each region that matters (a cavity, the core, a fallback-only area) across first light, every phase change and the fade-out. Ink must follow the physical quantity: up as the light comes in, down as it goes, with no hump and no dip. Stated as numbers, the afterburner's cut window fell 1894 → 155, and its light-off rose 154 → 1959 and settled at 1914.
- **In light, probe colours along the physical ramp.** For something that heats or cools, read the centre and rim pixel at several temperatures. They must run dull red → orange → yellow → near-white, with the rim cooler than the centre and no salmon or grey stage. The glory hole's table runs from 255/179/99 centre and 252/114/25 rim at 930 °C, through 211/63/6 and 165/28/1 at about 620 °C, to clear glass by 465 °C. For a volume, read a cross-section of ink from edge to edge: it should peak on the axis.
- **In light, check the dark path too.** Every light change sits behind `uLight`, so the dark build's frames should match its baseline. Also render a dark plate inside a light page, with and without WebGL (the nested-theme trap, section 8.5).
- **2× and 4× crops** of every place the effect meets the SVG: rims, lips, deck edges, wall bands, the paper corners, occluders, and every opening the effect is seen through (does it stop at the rim on all four sides, with the recess wall showing below?). Look for seams, hard edges, banding, a dark collar, a double line or a sudden colour switch. In light, also shoot 8–16× crops wherever ink meets a stroke (cut faces, rims, mouths, creases), and look for a pale gap between ink and line, ink painted over a line, or a stroke the ink cleared. Compare the same crop with the front SVG hidden and with the canvas hidden (set `style.display="none"` with an `eval`).
- **Fallback.** Run `drive.mjs` with `--no-webgl` (every `getContext("webgl")` returns `null`) and check that `data-gl` is absent and the SVG alone still reads, in both themes. In light, the fallback should show the same inks as the shader and the same dark ground, and every body the shader draws should be there. Film first light on a sheet with `--preload` slowing the page from its first frame: the handover from fallback to shader must not dip (section 8.5).
- **Reduced motion** (`--reduced`): one still, representative frame.
- **Console clean.** A shader compile error arrives as `console.error` from `onError`.

## 10. Performance

- Cost is roughly pixels × samples. The Raptor runs 28 volume samples, each with four shock-cell terms, plus a 28-step ground search with 7 bisections, an 8-sample halo, a 10-sample splash curtain and 6 steam samples per canvas, at up to 1.5× device ratio. It runs in SwiftShader, but its frame time hasn't been measured on a real GPU. Measure in a real browser's Performance panel before shipping something heavier, and if it's too slow, cut samples first, then `maxRatio`.
- The ripple tank's paper shader evaluates the two-source field four times per pixel (once for the refraction shift and three times for the caustic's fixed-point inverse), each source costing six texture reads of its history instead of a 24-step uniform loop. It measured about 1 ms per draw on a laptop GPU; phone cost is unmeasured.
- Know which GPU you measured. Headless Chrome in `drive.mjs` and `capture.mjs` uses the real GPU when it can and SwiftShader otherwise; log `gl.getParameter(gl.RENDERER)` before trusting a frame time.
- Spend samples only where they matter. Bound the march (a cylinder or box), start at the ground hit, and gate whole terms on uniforms.
- Two canvases cost two full-screen passes. Clear the one that has nothing to show (the Raptor's veil is drawn only while steam, vapour or the shock is present).
- Don't animate `uTime` if nothing in the field needs it. The ripple tank passes `draw(values, 0)` and moves only the wave phase, so a reduced-motion frame is truly still.
- When idle, stop the loop. A shader page that stays at 60 fps with nothing happening is a bug.
