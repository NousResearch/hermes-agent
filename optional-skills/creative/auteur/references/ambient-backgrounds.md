# ambient-backgrounds — quiet texture for secondary sections

For sections and simpler builds where a bold WebGL hero is overkill but a flat
fill reads as dull. An ambient background is **texture, not a feature**. If a
visitor *notices* it while reading, it has failed.

## The one rule (memorize)

> The background's strongest change — in luminance, colour, or motion — must stay
> **weaker than the quietest meaningful element of the foreground.** Invisible in
> peripheral vision until the reader has already begun parsing the copy. Zero
> focal point.

Operationally: contrast of any background detail against the base ≤ **~8%**
(colour distance small enough that text/background contrast is untouched — keep
body text ≥ 4.5:1 *as if the texture weren't there*), and motion slow enough it
reads as texture, not animation (a full cycle measured in tens of seconds, or no
motion at all).

## Composition limits

- **One ambient per page.** Two only when the page genuinely splits into distinct
  bands (e.g. a light editorial section then a dark one) — and **never two in the
  same viewport**. A different effect per section reads as a background sampler.
- **One committed hue.** Monochrome in the project's brand hue (same-hue,
  lower-lightness/opacity variations only). The moment a second hue appears it
  stops being ambient and becomes decoration.
- **Reduced-motion is mandatory** and easy here: every effect below degrades to a
  *rich still* (its last/seeded frame), never a blank fill.

## Banned (slopscan-adjacent)

The 250–290° purple→blue gradient · neon pulse / "cosmic ripple" (the template-
generator defaults) · glassmorphism-by-default · big blurred glowing orbs ·
animated `feTurbulence` re-rastered on scroll (see perf note) · rainbow/2-hue
anything · a texture legible enough to compete with 16–18px body copy.

## Performance facts (verified, load-bearing)

- **CSS and SVG (static) cost ~nothing** — composited once, no per-frame JS.
  Prefer them. `canvas2d` with < ~100 primitives on a throttled rAF is cheap.
  A **single** small WebGL fragment shader is fine; multi-pass WebGL is not
  "ambient" — it belongs to the hero tier.
- **SVG `feTurbulence` is CPU-rasterised in Chromium and expensive to re-raster.**
  Use it ONLY as a **static bake** (render once into a tiled data-URI). Never
  animate `baseFrequency`, and never let it re-rasterise on a scroll transform —
  that alone can drop a page below 60fps on a weak iGPU.

---

## The set (6 + a zero-motion default)

Each: technique · how it works · **NOT** (the taste trap). All assume a
`prefers-reduced-motion: reduce` branch that freezes to the still.

### 0. Static mesh (the always-safe default — zero motion, zero JS)
`pure CSS`. Two or three large, soft, same-hue radial/linear gradients at 150%
size, hand-placed. It's just a considered, non-flat ground.
```css
.amb-mesh{position:fixed;inset:0;z-index:0;pointer-events:none;
  background:
    radial-gradient(60% 50% at 18% 12%, color-mix(in srgb,var(--accent) 7%,transparent), transparent 70%),
    radial-gradient(50% 60% at 88% 90%, color-mix(in srgb,var(--accent) 5%,transparent), transparent 70%),
    var(--bg);}
```
**NOT:** the cool-blue/lavender default mesh, or SaaS-cream — commit to the brand hue.

### 1. Paper grain
`SVG, static bake`. A `feTurbulence` tile baked once into a data-URI, tiled and
held at 3–6% opacity — analog tooth that kills the flat-digital deadness.
```css
.amb-grain{position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.05;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.82' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='160' height='160' filter='url(%23n)'/%3E%3C/svg%3E");
  background-size:160px 160px;}
```
**NOT:** animate it, or push opacity past ~0.08 — grain is felt, not seen.

### 2. Ledger / blueprint rules
`pure CSS`. Same-hue hairlines (a baseline rhythm, with a bolder every-Nth),
static, ≤6% opacity — an editorial/instrument scaffold.
```css
.amb-ledger{position:fixed;inset:0;z-index:0;pointer-events:none;
  --l:color-mix(in srgb,var(--ink) 6%,transparent);
  --lb:color-mix(in srgb,var(--ink) 10%,transparent);
  background:
    repeating-linear-gradient(0deg,transparent 0 27px,var(--l) 27px 28px),
    repeating-linear-gradient(0deg,transparent 0 139px,var(--lb) 139px 140px);}
```
**NOT:** colour the lines, add verticals, or tighten spacing until it reads as a grid/table.

### 3. Topographic contour drift
`canvas2d`. A handful of low-frequency flow-lines from a layered-sine field,
drifting imperceptibly; one context, tiny primitive count.
```js
const cv=document.getElementById('amb'),g=cv.getContext('2d');
const RM=matchMedia('(prefers-reduced-motion: reduce)').matches;
function fit(){cv.width=innerWidth;cv.height=innerHeight;}
function frame(t){fit();g.clearRect(0,0,cv.width,cv.height);
  g.strokeStyle=getComputedStyle(cv).getPropertyValue('--line')||'rgba(40,36,32,.05)';g.lineWidth=1;
  const T=RM?0:t*0.00004;
  for(let i=0;i<9;i++){g.beginPath();
    for(let x=0;x<=cv.width;x+=14){
      const y=cv.height*(i+1)/10 + Math.sin(x*0.006+T+i)*12 + Math.sin(x*0.013-T*1.7)*7;
      x?g.lineTo(x,y):g.moveTo(x,y);}
    g.globalAlpha=.5;g.stroke();}
  if(!RM)requestAnimationFrame(frame);}
requestAnimationFrame(frame); // RM draws one still frame and stops
```
**NOT:** high curvature/density that forms a recognisable landscape silhouette, or bright/coloured strokes.

### 4. Sumi / ink tide
`canvas2d`. Two–three desaturated low-frequency bands advected slowly on a
low-res buffer (upscaled) — a breathing wash, monochrome.
Sketch: draw 2–3 soft horizontal `createLinearGradient` bands into a small
offscreen (e.g. 64px tall), offset each by `sin(t)` at different phases, then
`drawImage` upscaled with `imageSmoothing` on. Alpha ≤ 0.06.
**NOT:** fluid-sim turbulence, saturated colour, or crushed blacks — it's a whisper, not a lava lamp.

### 5. Sparse dust
`canvas2d`. Fewer than ~40 specks, drifting < 0.04px/frame on a throttled rAF;
static seeded frame for reduced-motion.
```js
const pts=Array.from({length:34},(_,i)=>({x:Math.abs(Math.sin(i*99.7))%1,y:Math.abs(Math.cos(i*57.3))%1,r:.5+(i%3)*.4}));
// per frame: y -= 0.00006 (wrap), draw each at alpha .07 in the brand hue; RM → draw once.
```
**NOT:** a dense twinkling starfield, trails, or high speed — that's a screensaver, not ambient.

### 6. Heat-haze / paper-ripple
`single WebGL fragment shader` (the one shader you're allowed). A ≤1px domain-warp
of a monochrome field — the surface subtly "breathes". Use the standard fullscreen-
quad harness (see `scroll-cinema.md`); fragment core:
```glsl
// uv in [0,1]; uTime slow; brand hue in uBase; result stays near-monochrome
float n = sin(uv.x*8.+uTime*.15)*.5 + sin(uv.y*11.-uTime*.11)*.5;
vec2 warp = vec2(n)*0.004;                 // <= a few px at 1080p
float g = texture(uTex, uv+warp).r;        // or a baked gradient
outColor = vec4(mix(uBase, uBase*1.03, g), 1.);
```
Fallback: a pre-rendered static frame (or effect #0). **NOT:** chromatic aberration,
liquid blobs, or any warp large enough to visibly bend text edges.

---

## Wiring notes

- The ambient layer is `position:fixed; inset:0; z-index:0; pointer-events:none`;
  content sits above on its own stacking context. Keep a solid `--bg` under it so
  text contrast is guaranteed by the base, not the texture.
- Trigger any scroll-linked drift with `IntersectionObserver` (pause the rAF when
  the section is off-screen), never `addEventListener('scroll')`.
- Reduced-motion: one `matchMedia('(prefers-reduced-motion: reduce)')` check that
  draws a single frame and returns — every effect above already shows this shape.
- Distinctiveness is the point: reach for #2/#3/#6 and the sumi/letterpress
  register before plain grain — "grain + grid" alone is now a SaaS default. The
  authored textures (ledger, contour, ink, heat-haze) are what separate auteur
  from a prompt generator.
