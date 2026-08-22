# scroll-cinema — Cinematic Scroll Recipes

Stack default 2026: **GSAP 3.13 + ScrollTrigger 3.13 (free), Lenis 1.3, Three.js r170+, CSS scroll-driven animations (Chrome 115+ / Safari 26) with `@supports` fallback on GSAP. View Transitions API — 88.64% global support.**

---

## Technique selection

| Narrative need | Technique |
|---|---|
| Product in motion — user scrubs through a scene | Scroll-scrubbed video |
| Mood/state shift between two AI-generated keyframes | WebGL displacement shader transition |
| Atmosphere with available video asset | Scroll-scrubbed video |
| Text as hero — no assets | Kinetic typography (SplitType stagger) |
| Simple section reveals — minimal JS | CSS scroll-driven animations with GSAP fallback |
| Page-to-page continuity, shared-element morphs | View Transitions API |
| Pixel-accurate product animation, any frame swappable | Canvas image sequence |
| 3D object or abstract CGI scene | Three.js + ScrollTrigger |
| Atmospheric brand presence | Ambient audio + mute toggle |

---

## Scroll-scrubbed video

Viewer scrolls — a pre-generated video scrubs forward/backward in sync with scroll progress. First frame = one composition, last frame = another (product revealed, camera moved). This is the primary wow technique: igloo.website, Runway, Veo launch pages, Apple AirPods.

> The GSAP recipe below is the minimal single-clip scrubber. For a **photoreal, multi-scene "fly through the world"** — mobile/iOS-hardened, blob-loaded, seek-coalesced, with crossfade or seamless seams — use the drop-in engine in `templates/scroll-flight-engine.js` and the full recipe in **`references/scroll-flight.md`** (it also carries the critical `-g 8` encode and the SSIM seam gate). Reach for that when the hero is footage/AI-video rather than a single generated clip.

**Video prep:**
```bash
# H.264 baseline for Safari/old Android, VP9 for Chrome
ffmpeg -i source.mp4 -c:v libx264 -profile:v baseline -level 3.1 -pix_fmt yuv420p \
  -movflags +faststart -crf 23 hero.mp4
ffmpeg -i source.mp4 -c:v libvpx-vp9 -crf 30 -b:v 0 hero.webm

# Teaser (first 1-2s, < 200KB) — poster effect while main loads
ffmpeg -i source.mp4 -t 2 -c:v libx264 -profile:v baseline -level 3.1 -crf 28 teaser.mp4
```

```html
<video id="hero" src="/hero.webm" muted playsinline preload="metadata"></video>
```

Scrubbing needs a scroll-progress engine. A raw `window.addEventListener('scroll')` loop is banned by this skill's own rules (and slopscan) — ScrollTrigger is the sanctioned driver:

**GSAP ScrollTrigger + Lenis (idiomatic Awwwards stack):**
```js
import gsap from 'gsap'
import ScrollTrigger from 'gsap/ScrollTrigger'
import Lenis from 'lenis'

gsap.registerPlugin(ScrollTrigger)

const lenis = new Lenis()
lenis.on('scroll', ScrollTrigger.update)
gsap.ticker.add((time) => lenis.raf(time * 1000))
gsap.ticker.lagSmoothing(0)

const video = document.querySelector('#hero')
video.pause()

ScrollTrigger.create({
  trigger: video,
  start: 'top top',
  end: '+=400%',
  pin: true,
  scrub: 0.5,          // 0 = instant, number = smoothing in seconds; use 0.3–0.8
  onUpdate: (self) => {
    video.currentTime = self.progress * video.duration
  },
})
```

| Pitfall | Fix |
|---|---|
| Video > 2 MB | Compress to ≤ 2 MB; loop 5–8 s; serve teaser as first paint |
| `currentTime` resets after full load | Wait for `loadedmetadata`, then set; `preload="metadata"` is required |
| Seek jitter on scrub | Move `currentTime` only while scrolling; never touch it on pause |
| iOS Safari autoplay | `muted playsinline` attributes are mandatory — video won't play without them |
| `prefers-reduced-motion` | Show static poster + `autoplay muted loop` video, no scrub |
| Mobile > 1080p lag | Serve `hero-mobile.mp4` at 720p via media query |
| `content-visibility: auto` on parent | Breaks ScrollTrigger — never use on pinned ancestors |

**Use when:** hero section, 3–6 s sell-the-product moment, user = director concept.  
**Avoid when:** long-form content pages, information-first UX, full-page application.

---

## Canvas image sequence

60–240 pre-rendered frames drawn to `<canvas>` per scroll progress. Deterministic, no seek jitter, better first-frame LCP, swappable per A/B test. Apple AirPods Pro pattern (2019+).

**Export frames:**
```bash
# From video/CGI at 30 fps:
ffmpeg -i render.mp4 -vf "fps=30" -q:v 5 frames/frame_%04d.jpg
# WebP (smaller):
ffmpeg -i render.mp4 -vf "fps=30" -lossless 0 -compression_level 4 -q:v 70 frames/frame_%04d.webp
```
Frame budget: 24 fps × 2–5 s = **48–120 frames** for one camera pass. 60–240 for complex scenes.

**HTML:**
```html
<canvas id="seq" width="1920" height="1080" role="img" aria-label="Product animation"></canvas>
<div style="height: 400vh"></div>
```

**Preload + draw:**
```js
const canvas = document.getElementById('seq')
const ctx = canvas.getContext('2d')
const FRAME_COUNT = 120
const frames = new Array(FRAME_COUNT)

function preloadFrame(i) {
  return new Promise(resolve => {
    const img = new Image()
    img.onload = () => { frames[i] = img; resolve() }
    img.onerror = resolve  // don't crash on a broken frame
    img.src = `/frames/frame_${String(i + 1).padStart(4, '0')}.webp`
  })
}

async function preloadAll() {
  // batch 8 at a time — avoids saturating the network
  for (let i = 0; i < FRAME_COUNT; i += 8) {
    await Promise.all(
      Array.from({ length: 8 }, (_, k) => i + k)
        .filter(k => k < FRAME_COUNT)
        .map(k => preloadFrame(k))
    )
  }
}

function draw(progress) {
  const idx = Math.min(FRAME_COUNT - 1, Math.floor(progress * FRAME_COUNT))
  if (frames[idx]) ctx.drawImage(frames[idx], 0, 0, canvas.width, canvas.height)
}

// Drive with ScrollTrigger (raw scroll listeners are banned by this skill):
preloadAll().then(() => {
  ScrollTrigger.create({
    trigger: '#seq',
    start: 'top top',
    end: '+=400%',
    pin: true,
    scrub: 0.3,
    onUpdate: (self) => draw(self.progress),
  })
})
```

| Pitfall | Fix |
|---|---|
| 240 frames × 100 KB = 24 MB | WebP lossless 0 q70 → ~30–60 KB/frame → 7–15 MB total; serve smaller set to mobile |
| Blocking preload (5+ s) | Batch 8 simultaneously; `<link rel="preload" as="image">` for first 5–10 frames |
| Retina blur | Multiply canvas size by `devicePixelRatio`; `ctx.scale(dpr, dpr)` |
| Screen reader sees nothing | `role="img"` + `aria-label` on canvas; add `<p>` description in section |
| PageSpeed "large network payload" | CDN + HTTP/2 multiplexing + Service Worker cache; minimal first frame as LCP |

**Use when:** pixel-accurate product spin, text-in-frame scenes, A/B-swappable keyframes.  
**Avoid when:** AI scene > 10 s (video is cheaper); dynamic/personalized content.

---

## GSAP ScrollTrigger + Lenis foundation

The load-bearing skeleton. Wire this first — every other scroll effect depends on it being present and correct.

```bash
npm i gsap lenis
# ScrollTrigger is free since 2024; only SplitText and MorphSVG are Club-only
```

**Foundation (copy verbatim, do not reorder):**
```js
import gsap from 'gsap'
import ScrollTrigger from 'gsap/ScrollTrigger'
import Lenis from 'lenis'
import 'lenis/dist/lenis.css'

gsap.registerPlugin(ScrollTrigger)

const lenis = new Lenis()
lenis.on('scroll', ScrollTrigger.update)      // keep ScrollTrigger in sync with Lenis
gsap.ticker.add((time) => lenis.raf(time * 1000))
gsap.ticker.lagSmoothing(0)                  // prevent GSAP catching up after tab blur
```

**Common recipes built on the foundation:**
```js
// Fade-up on enter
gsap.from('.card', {
  y: 60, opacity: 0, duration: 1, ease: 'power3.out',
  scrollTrigger: { trigger: '.card', start: 'top 80%', toggleActions: 'play none none reverse' },
})

// Horizontal scroll panel (pinned, 3 screen-widths of horizontal travel)
gsap.to('.pin-inner', {
  x: () => -(document.querySelector('.pin-inner').scrollWidth - window.innerWidth),
  ease: 'none',
  scrollTrigger: { trigger: '.pin-section', pin: true, scrub: 1, end: '+=300%' },
})

// Parallax background (slower than content)
gsap.to('.bg', {
  yPercent: -30, ease: 'none',
  scrollTrigger: { trigger: '.section', start: 'top bottom', end: 'bottom top', scrub: true },
})

// Section snapping
ScrollTrigger.create({
  trigger: '.section', start: 'top top',
  snap: { snapTo: 1, duration: 0.4 },
})
```

| Pitfall | Fix |
|---|---|
| Lenis and ScrollTrigger drift apart | Mandatory: `lenis.on('scroll', ScrollTrigger.update)` + `lagSmoothing(0)` |
| ScrollTrigger misses dynamically added DOM | Call `ScrollTrigger.refresh()` or `ScrollTrigger.sort()` after dynamic inserts |
| Firefox anchor jumps fight smooth scroll | `lenis.scrollTo(target, { lock: true, duration: 1.2 })` |
| `content-visibility: auto` on pinned parent | Never use — breaks ScrollTrigger |
| GSAP SplitText is Club (paid) | Use **SplitType** (`npm i split-type`) instead — MIT, same API shape |

**Use when:** any site with more than one scroll effect, pin/parallax/horizontal scroll needed.  
**Avoid when:** pure CSS page with no animations, or JS-budget-critical JAMStack.

---

## CSS scroll-driven animations

Native `animation-timeline` API — scroll-linked animations with zero JS. Chrome 115+ / Edge 115+ / Safari 26 stable. Firefox 132+ behind flag. ~75–80% global coverage; use `@supports` + dynamic GSAP import as fallback.

```css
/* Fade-up as element enters viewport */
.card {
  opacity: 0;
  transform: translateY(40px);
  animation: card-in linear both;
  animation-timeline: view();                 /* always declare AFTER the animation shorthand */
  animation-range: entry 10% cover 40%;
}
@keyframes card-in {
  to { opacity: 1; transform: translateY(0); }
}

/* Page-wide scroll progress bar */
#progress {
  transform-origin: 0 50%;
  transform: scaleX(0);
  animation: grow auto linear;
  animation-timeline: scroll();
}
@keyframes grow { to { transform: scaleX(1); } }

/* Named scroll container (not the whole document) */
.gallery {
  overflow-x: scroll;
  scroll-timeline: --gallery inline;
}
.gallery__progress {
  animation: grow auto linear;
  animation-timeline: --gallery;
}
```

**`@supports` + dynamic GSAP fallback:**
```css
@supports not (animation-timeline: scroll()) {
  .card { opacity: 1; transform: none; }
}
```
```js
if (!CSS.supports('animation-timeline', 'scroll()')) {
  import('gsap').then(({ default: gsap }) => {
    import('gsap/ScrollTrigger').then(({ default: ScrollTrigger }) => {
      gsap.registerPlugin(ScrollTrigger)
      gsap.from('.card', { y: 40, opacity: 0, scrollTrigger: { trigger: '.card', start: 'top 80%' } })
    })
  })
}
```

| Pitfall | Fix |
|---|---|
| `animation-timeline` in shorthand gets reset | Declare `animation-timeline` separately, always after the `animation` shorthand |
| Firefox off by default | `@supports` + JS fallback; flag: `about:config → layout.css.scroll-driven-animations.enabled` |
| Layout properties (`width`, `top`) are slow | Animate only composited properties: `transform`, `opacity`, `filter` |
| `view()` fails inside a scroll container | Read MDN `view-timeline-scope` |

**Use when:** simple pages, no GSAP budget, maximum GPU-composited performance.  
**Avoid when:** you need pin sections, snapping, complex timelines — that's GSAP territory.

---

## Text reveal / stagger

Text appears line-by-line, word-by-word, or character-by-character from behind a mask. Controls reading pace and creates cinematic rhythm.

**Option A — SplitType (free, MIT):**
```js
import SplitType from 'split-type'

const text = new SplitType('.hero h1', { types: 'lines, words, chars' })

gsap.from(text.chars, {
  opacity: 0,
  y: 20,
  stagger: 0.025,
  duration: 0.6,
  ease: 'power2.out',
  scrollTrigger: { trigger: '.hero', start: 'top 70%' },
})

// Recompute on resize (only needed for absolute-position mode)
window.addEventListener('resize', () => text.split())
```
Add `font-kerning: none` on the target element — prevents 1–2 px character jumps after split.

**Option B — GSAP SplitText (Club GSAP, paid):**
```js
import SplitText from 'gsap/SplitText'
gsap.registerPlugin(SplitText)

const split = SplitText.create('.hero h1', {
  type: 'chars, lines',
  mask: 'lines',          // reveal mask — lines don't reflow out of container
  autoSplit: true,        // reacts to resize automatically
  onSplit(self) {
    return gsap.from(self.chars, { yPercent: 110, stagger: 0.04, duration: 1.2, ease: 'expo.out' })
  },
})
```

**Option C — pure CSS (zero dependencies):**
```css
.reveal-text > span {
  display: inline-block;
  transform: translateY(100%);
  opacity: 0;
  transition: transform 1s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.6s;
}
.reveal-text.is-visible > span { transform: translateY(0); opacity: 1; }
.reveal-text > span:nth-child(1) { transition-delay: 0.00s; }
.reveal-text > span:nth-child(2) { transition-delay: 0.04s; }
/* continue for each child */
```
```js
const io = new IntersectionObserver(
  (entries) => entries.forEach(e => e.isIntersecting && e.target.classList.add('is-visible')),
  { threshold: 0.2 }
)
document.querySelectorAll('.reveal-text').forEach(el => io.observe(el))
```

| Pitfall | Fix |
|---|---|
| `<a>` inside split text loses semantics | `aria: 'none'` (SplitText) + hidden duplicate text for screen readers |
| Lines reflow on resize | `autoSplit: true` (SplitText) or `ResizeObserver` → `text.split()` |
| Layout shift after split | Reserve height with `min-height`; or use mask variant so no element leaves flow |
| `prefers-reduced-motion` | Skip transition entirely — show final state immediately |

**Use when:** hero headings, key subheads, manifesto lines, pull quotes.  
**Avoid when:** body-text articles, documentation, every other paragraph — it desensitizes.

---

## Three.js displacement shader transition

Two AI-generated keyframe images (frame A → frame B, produced per assets.md: B is an *edit* of A) morphed by a displacement map, scrubbed by scroll. The skill's signature move: a living, filmic transition between two stills — no video needed. One full-viewport quad, one ShaderMaterial; simpler and richer-looking than scene compositing. Copy-pasteable — all pieces are here.

```bash
npm i three
```

```js
import * as THREE from 'three'
import gsap from 'gsap'
import ScrollTrigger from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

const canvas = document.querySelector('#morph')
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false })
renderer.setPixelRatio(Math.min(2, devicePixelRatio)) // cap DPR — saves FPS on Retina
renderer.outputColorSpace = THREE.SRGBColorSpace

const scene = new THREE.Scene()
// Orthographic camera + 2×2 plane = a screen-space quad, no perspective math
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)

const loader = new THREE.TextureLoader()
const [texA, texB, disp] = await Promise.all([
  loader.loadAsync('/assets/s3-peak-a.webp'),
  loader.loadAsync('/assets/s3-peak-b.webp'),
  // Displacement map: any soft grayscale — clouds/noise, or frame A itself
  // blurred to ~20px grayscale (free, and the morph follows the image's own forms)
  loader.loadAsync('/assets/s3-disp.webp'),
])
;[texA, texB].forEach(t => { t.colorSpace = THREE.SRGBColorSpace })

const material = new THREE.ShaderMaterial({
  uniforms: {
    texA: { value: texA },
    texB: { value: texB },
    disp: { value: disp },
    progress: { value: 0 },
    intensity: { value: 0.3 },              // 0.15 subtle · 0.3 default · 0.6 dramatic
    // cover-fit correction: plane aspect vs image aspect
    scale: { value: coverScale(texA, canvas) },
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
    uniform sampler2D texA, texB, disp;
    uniform float progress, intensity;
    uniform vec2 scale;
    varying vec2 vUv;

    void main() {
      vec2 uv = (vUv - 0.5) * scale + 0.5;          // object-fit: cover
      float d = texture2D(disp, uv).r;
      // classic displacement morph: A slides out along the map, B slides in
      vec2 uvA = uv + vec2( progress        * d * intensity, 0.0);
      vec2 uvB = uv - vec2((1.0 - progress) * d * intensity, 0.0);
      vec4 a = texture2D(texA, uvA);
      vec4 b = texture2D(texB, uvB);
      gl_FragColor = mix(a, b, smoothstep(0.15, 0.85, progress));
    }
  `,
})
scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))

// cover-fit helper: returns UV scale so the image covers the canvas without stretching
function coverScale(tex, el) {
  const ia = tex.image.width / tex.image.height
  const ca = el.clientWidth / el.clientHeight
  return ca > ia ? new THREE.Vector2(1, ia / ca) : new THREE.Vector2(ca / ia, 1)
}

// --- Scroll → progress (render only on change; no idle rAF loop) ---
ScrollTrigger.create({
  trigger: '#scene-peak',
  start: 'top top',
  end: '+=250%',
  pin: true,
  scrub: 0.5,
  onUpdate: (self) => {
    material.uniforms.progress.value = self.progress
    renderer.render(scene, camera)
  },
})
addEventListener('resize', () => {
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false)
  material.uniforms.scale.value = coverScale(texA, canvas)
  renderer.render(scene, camera)
})
renderer.setSize(canvas.clientWidth, canvas.clientHeight, false)
renderer.render(scene, camera)
```

Displacement direction variants: use `vec2(0.0, d * intensity)` for vertical flow; `(uv - 0.5) * d * intensity` for radial bloom; rotate the vector by scene's camera motion for directed morphs. Small A/B drift from the CLI edit hides inside the mid-morph distortion — that's why this rung of the ladder tolerates imperfect keyframe pairs.

**For full 3D scene→scene transitions** (two live THREE scenes, not stills): render both via `EffectComposer` with two `RenderPass`es and blend in a final `ShaderPass` with the same progress uniform — same scrub wiring, heavier GPU bill; reach for it only when both scenes genuinely need live geometry.

---

## 3D beyond the morph

Three compact recipes for when a scene needs live 3D. All share the morph section's hygiene: DPR cap at 2, render-on-demand where possible, dispose on teardown, `<picture>` fallback for no-WebGL/low-end (`navigator.hardwareConcurrency < 4`), reduced-motion → static render.

### GLB product model scrubbed by scroll
The product rotates/travels as the user scrolls — the "turn it in your hands" scene. Needs a .glb (from the client, a 3D artist, or an AI mesh generator — check quality eyes-on; auto-generated topology is often mush up close).
```js
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js'

const draco = new DRACOLoader()
draco.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.7/')
const loader = new GLTFLoader().setDRACOLoader(draco)

const { scene: model } = await loader.loadAsync('/assets/product.glb')
scene.add(model)
// Environment lighting sells realism more than any material tweak:
scene.environment = await new THREE.PMREMGenerator(renderer)
  .fromEquirectangular(await new THREE.TextureLoader().loadAsync('/assets/studio.hdr.jpg')).texture

ScrollTrigger.create({
  trigger: '#scene-product', start: 'top top', end: '+=300%', pin: true, scrub: 0.5,
  onUpdate: (self) => {
    model.rotation.y = self.progress * Math.PI * 1.5   // 270° over the scene
    camera.position.z = 4 - self.progress * 1.2        // slow push-in
    renderer.render(scene, camera)                     // render only on scroll change
  },
})
```
| Pitfall | Fix |
|---|---|
| GLB is 20 MB | Draco/meshopt compression (`gltf-transform optimize in.glb out.glb`), target ≤3 MB |
| Model pops in late | preload + show poster until `loadAsync` resolves; reserve canvas size (CLS) |
| Materials look flat | environment map (above) beats adding lights; `ACESFilmicToneMapping` |

### Particle field (depth without a model)
2–5k points drifting with scroll-linked parallax — atmosphere for a hero when there's no asset at all.
```js
const N = 3000
const pos = new Float32Array(N * 3)
for (let i = 0; i < N * 3; i++) pos[i] = (Math.random() - 0.5) * 10
const geo = new THREE.BufferGeometry()
geo.setAttribute('position', new THREE.BufferAttribute(pos, 3))
const pts = new THREE.Points(geo, new THREE.PointsMaterial({
  size: 0.02, color: 0xd4a24e, transparent: true, opacity: 0.7, depthWrite: false,
}))
scene.add(pts)
ScrollTrigger.create({
  trigger: '#hero', start: 'top top', end: 'bottom top', scrub: 0.5,
  onUpdate: (s) => { pts.rotation.y = s.progress * 0.6; pts.position.y = s.progress * -1.5; renderer.render(scene, camera) },
})
```
Cap N by device: `navigator.hardwareConcurrency < 6 ? 1200 : 3000`. Points must never carry meaning — they're weather, not content.

### Animated shader background (one quad, no geometry)
Same screen-space quad rig as the displacement morph, but the fragment shader generates the visual: flowing noise, grain-drenched gradient in the brand hue, contour lines. Swap the morph's fragmentShader for a noise-driven one and feed `uTime` from GSAP's ticker only while the section is on screen (IntersectionObserver gate). This is the cheapest "expensive-looking" background that isn't a stock video — and it recolors with the token palette for free. Keep chroma/hue inside the commit-sheet palette; a shader background in an off-brand hue is just animated slop.

**Other shader variants (swap the `fragmentShader` body):**
- Cross-fade: `gl_FragColor = mix(c1, c2, progress);`
- Radial reveal: `float r = length(vUv - 0.5); gl_FragColor = mix(c1, c2, step(r, progress));`
- Pixelation: round `vUv` to a grid scaled by `progress` before sampling
- Glitch: offset UV with noise on `progress > 0.3`

| Pitfall | Fix |
|---|---|
| Low-end mobile FPS drops | Check `navigator.hardwareConcurrency < 4` or `navigator.deviceMemory < 4` → fall back to `<picture>` |
| Backgrounded tab keeps rendering | `cancelAnimationFrame` on `visibilitychange: hidden` |
| Texture color shift | `renderer.outputColorSpace = THREE.SRGBColorSpace` + `texture.colorSpace = THREE.SRGBColorSpace` per texture |
| GPU memory leak on teardown | `scene.traverse(o => { o.geometry?.dispose(); o.material?.dispose() })` |
| Long `if` blocks in shader | Replace all branching with `step()` / `smoothstep()` — branch divergence kills GPU perf |
| iOS Safari WebGL 2 | Available on A12+ (2018+); add `WebGL1Renderer` fallback for older |

**Use when:** two AI keyframes, brand identity moment, abstract state transition.  
**Avoid when:** product catalog, text-first page, information-priority UX.

---

## View Transitions API

Native animated transitions between pages or DOM states. 88.64% global support. Chrome 111+, Safari 18+, Firefox 144+. No libraries needed.

**Same-document (SPA) transition:**
```js
async function navigate(newHtml) {
  if (!document.startViewTransition) {
    document.documentElement.innerHTML = newHtml  // silent fallback
    return
  }
  document.startViewTransition(() => {
    document.documentElement.innerHTML = newHtml
  })
}

document.querySelectorAll('a[data-spa]').forEach(a => {
  a.addEventListener('click', async e => {
    e.preventDefault()
    navigate(await (await fetch(a.href)).text())
  })
})
```

**CSS transition styling:**
```css
/* Default: whole-document crossfade */
::view-transition-old(root),
::view-transition-new(root) { animation-duration: 0.3s; }

/* Named element morphs between pages (shared-element transition) */
.product-card        { view-transition-name: product-card-main; }
.product-detail__img { view-transition-name: product-card-main; }

::view-transition-old(product-card-main),
::view-transition-new(product-card-main) {
  animation-duration: 0.5s;
  animation-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
}
```

**Cross-document MPA (Chromium-only as of 2026 stable):**
```html
<!-- Both pages: -->
<style>@view-transition { navigation: auto; }</style>
```
```css
/* Slide-up enter/exit */
::view-transition-old(root) { animation: 0.4s ease-in both slide-up-out; }
::view-transition-new(root) { animation: 0.4s ease-in both slide-up-in; }
@keyframes slide-up-out { to   { transform: translateY(-100%); } }
@keyframes slide-up-in  { from { transform: translateY(100%);  } }
```

| Pitfall | Fix |
|---|---|
| Safari < 18, Firefox < 144 | Wrap in `if (document.startViewTransition)` — falls back to instant swap |
| Duplicate `view-transition-name` on same page | Each name must be unique per frame; use data-attribute selectors |
| Glitchy positioned/transformed elements | View Transitions capture layout boxes; rebuild manually for complex absolute layouts |
| `prefers-reduced-motion` | `@media (prefers-reduced-motion: reduce) { ::view-transition-* { animation-duration: 0.01ms !important; } }` |

**Use when:** e-commerce card → detail, portfolio thumbnail → case study, SPA navigation.  
**Avoid when:** you want GLSL noise/glitch — use a persistent WebGL scene instead.

---

## Ambient audio + mute toggle

Background ambient sound (hum, rain, synth) launched on first user gesture, with a persistent mute toggle. Default off for first-time visitors — 90% of users dislike auto-sound.

```html
<button id="mute" aria-label="Toggle sound" class="fixed bottom-4 right-4">
  <span data-on>Sound on</span>
  <span data-off hidden>Muted</span>
</button>
<audio id="ambient" src="/ambient.opus" loop preload="none"></audio>
```

```js
const audio  = document.getElementById('ambient')
const btn    = document.getElementById('mute')
const onSpan = btn.querySelector('[data-on]')
const offSpan = btn.querySelector('[data-off]')

const saved = localStorage.getItem('ambient-muted')
let muted = saved === null ? true : saved === 'true'  // default OFF on first visit

const apply = () => {
  audio.muted  = muted
  audio.volume = muted ? 0 : 0.4
  onSpan.hidden  = muted
  offSpan.hidden = !muted
  localStorage.setItem('ambient-muted', String(muted))
}
apply()

btn.addEventListener('click', async () => {
  muted = !muted
  apply()
  if (!muted && audio.paused) {
    await audio.play().catch(e => console.warn('autoplay blocked', e))
  }
})

// Soft-start on first pointer interaction (if not muted by user preference)
const start = async () => {
  if (muted) return
  try { await audio.play() } catch {}
}
window.addEventListener('pointermove', start, { once: true })
```

**WebAudio for filter/reverb control:**
```js
// Must be created inside a user gesture handler (AudioContext unlock rule)
const ctx    = new (window.AudioContext || window.webkitAudioContext)()
const source = ctx.createMediaElementSource(audio)
const gain   = ctx.createGain()
gain.gain.value = muted ? 0 : 0.4
source.connect(gain).connect(ctx.destination)
// chain ctx.createBiquadFilter / ctx.createConvolver for EQ and reverb
```

```bash
# Encode ambient loop: 64–96 kbps Opus, 30–60 s loop, < 200 KB
ffmpeg -i source.wav -c:a libopus -b:a 64k ambient.opus
```

| Pitfall | Fix |
|---|---|
| Browser blocks autoplay | Only call `audio.play()` inside a click/touch handler |
| Tab hidden — audio continues | `audio.volume = 0` on `visibilitychange: hidden` |
| `prefers-reduced-motion` | Default muted; do not auto-start |
| iOS Safari `<audio loop>` glitch | Use WebAudio `BufferSource` with `loop = true`; or crossfade two instances |
| Mobile data cost | 64 kbps Opus loop ≈ 30 KB/30 s — acceptable; VP9 video costs more |

**Use when:** product demo, generative art, film/game promo, atmospheric brand.  
**Avoid when:** corporate, documentation, news, e-commerce (unless explicitly cinematic brand).

---

## Cinematic transition library

Drop-in CSS/GSAP recipes. Each is 10–20 lines and self-contained. Wire to a ScrollTrigger `onEnter` or a View Transition callback.

### Wipe-mask reveal
Clip-path inset animates from right to left, revealing the element underneath. *When:* section-to-section reveal, image hero entrance.
```js
// Scrubbed by scroll progress:
gsap.fromTo('.wipe-target',
  { clipPath: 'inset(0 100% 0 0)' },
  {
    clipPath: 'inset(0 0% 0 0)',
    ease: 'none',
    scrollTrigger: { trigger: '.wipe-target', start: 'top 80%', end: 'top 30%', scrub: true },
  }
)
```

### Curtain split
Two halves (`::before` / `::after`) translate apart to reveal content behind. *When:* section entrance, modal reveal.
```css
.curtain { position: relative; overflow: hidden; }
.curtain::before,
.curtain::after {
  content: '';
  position: absolute; inset: 0;
  width: 50%;
  background: var(--curtain-color, #0a0a0a);
  transition: transform 0.9s cubic-bezier(0.76, 0, 0.24, 1);
  z-index: 2;
}
.curtain::after  { left: 50%; }
.curtain.is-open::before { transform: translateX(-100%); }
.curtain.is-open::after  { transform: translateX(100%);  }
```
```js
ScrollTrigger.create({
  trigger: '.curtain', start: 'top 70%',
  onEnter: () => document.querySelector('.curtain').classList.add('is-open'),
})
```

### Letterbox bars
21:9 black bars animate in when a key cinematic scene is pinned. *When:* hero climax, product reveal moment.
```css
.letterbox::before,
.letterbox::after {
  content: '';
  position: fixed; left: 0; right: 0;
  height: 0;
  background: #000;
  transition: height 0.6s ease;
  z-index: 100;
}
.letterbox::before { top: 0; }
.letterbox::after  { bottom: 0; }
.letterbox.is-cinema::before,
.letterbox.is-cinema::after { height: 13.5vh; } /* 27% total = 21:9 ratio approximation */
```
```js
ScrollTrigger.create({
  trigger: '.hero-climax', start: 'top top',
  onEnter:     () => document.body.classList.add('is-cinema'),
  onLeaveBack: () => document.body.classList.remove('is-cinema'),
})
```

### Shutter
Horizontal slats (pseudo-elements on children) flip open. *When:* image grid entrance, dramatic reveal of a collection.
```css
.shutter-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); }
.shutter-grid > * {
  overflow: hidden;
  clip-path: inset(0 0 100% 0);
  transition: clip-path 0.7s cubic-bezier(0.16, 1, 0.3, 1);
}
.shutter-grid.is-open > * { clip-path: inset(0 0 0% 0); }
/* stagger via nth-child transition-delay */
.shutter-grid > *:nth-child(2) { transition-delay: 0.07s; }
.shutter-grid > *:nth-child(3) { transition-delay: 0.14s; }
.shutter-grid > *:nth-child(4) { transition-delay: 0.21s; }
```

### Depth parallax
2–4 layers move at different speeds, creating foreground/background depth. *When:* hero section with layered illustration or photo composite.
```html
<section class="parallax-scene" data-parallax-container>
  <div class="layer" data-speed="0.1"><!-- sky / far background --></div>
  <div class="layer" data-speed="0.3"><!-- midground --></div>
  <div class="layer" data-speed="0.6"><!-- foreground --></div>
  <div class="layer" data-speed="1.0"><!-- subject (moves with scroll) --></div>
</section>
```
```js
document.querySelectorAll('[data-parallax-container]').forEach(container => {
  container.querySelectorAll('.layer[data-speed]').forEach(layer => {
    const speed = parseFloat(layer.dataset.speed)
    gsap.to(layer, {
      yPercent: -30 * speed,
      ease: 'none',
      scrollTrigger: {
        trigger: container,
        start: 'top bottom',
        end: 'bottom top',
        scrub: true,
      },
    })
  })
})
```

### Cursor parallax (pointer-driven depth — "flat photo feels 3D under the mouse")

The Depth parallax above shifts layers by *scroll*; this shifts them by the *cursor*, so a still composite gains dimension as the mouse moves. Perfect for a white-on-white hero where a subject generated on the page's own background (assets.md §0 "match the background") should feel dimensional, not pasted. Reuse the same 2–4 layers (subject cut from background — assets.md ladder rung 4).

```html
<section class="tilt-scene" data-pointer-parallax>
  <img class="layer" data-depth="0.03" src="bg.webp"      alt="">
  <img class="layer" data-depth="0.07" src="mid.webp"     alt="">
  <img class="layer" data-depth="0.14" src="subject.webp" alt="">
</section>
```
```js
document.querySelectorAll('[data-pointer-parallax]').forEach(scene => {
  const layers = [...scene.querySelectorAll('.layer[data-depth]')]
  let tx = 0, ty = 0, raf = 0
  const apply = () => {
    raf = 0
    for (const l of layers) {
      const d = parseFloat(l.dataset.depth) * 100        // px of travel at screen edge
      l.style.transform = `translate3d(${(-tx*d).toFixed(1)}px, ${(-ty*d).toFixed(1)}px, 0)`
    }
  }
  const onMove = e => {
    const r = scene.getBoundingClientRect()
    tx = ((e.clientX - r.left) / r.width  - 0.5) * 2      // -1..1 from center
    ty = ((e.clientY - r.top)  / r.height - 0.5) * 2
    if (!raf) raf = requestAnimationFrame(apply)          // never write transform per event
  }
  // desktop pointer only; touch has no hover, and respect reduced-motion
  if (matchMedia('(hover: hover) and (pointer: fine)').matches &&
      !matchMedia('(prefers-reduced-motion: reduce)').matches) {
    scene.addEventListener('pointermove', onMove, { passive: true })
    scene.addEventListener('pointerleave', () => { tx = ty = 0; apply() })
  }
})
```
```css
.tilt-scene .layer { transition: transform 120ms ease-out; will-change: transform; }
```

Rules: rAF-throttle (one transform write per frame, never per event); clamp so the *farthest* layer travels ≤ ~12–16px — big travel reads as a cheap gimmick, subtle depth reads as craft; animate only `transform`; disabled on touch (`hover: none`) and under `prefers-reduced-motion`; the `120ms ease-out` gives a soft settle instead of rubber-banding. **Fancier (peak only):** one image + a generated depth map in a WebGL shader (parallax-occlusion) for a true "3D photo" — reach for it only when this hero is THE wow peak, and generate the depth pass as its own asset.

---

## State-machine cinema — A→B→C morph + audio-reactive (Tier-1 engine)

The signature 2026 move: ONE world that transforms through a chain of scene-consistent frames
(assets.md §2 — grok edit-chain A→B→C…N), scrubbed by scroll AND driven by an audio track, both feeding
the SAME `uMix`/`uEnergy` uniforms so picture and sound move as one. Extends the two-frame displacement
to N frames. One persistent WebGL context; swap textures, never remount.

### 1. Frame-chain scrubber (scroll → uProgress → GPU morph)

Preload N consistent frames. Scroll maps 0→1 across the chain; the shader mixes the two frames bracketing
the current position with a noise-driven wipe, so it reads as a morph, not a crossfade.

```html
<section id="stage" style="height:400vh">
  <canvas id="c" style="position:sticky;top:0;height:100vh;width:100%"></canvas>
</section>
```
```js
import * as THREE from 'three'
import Lenis from 'lenis'
const FRAMES = ['/gen/s1-a.webp','/gen/s1-b.webp','/gen/s1-c.webp','/gen/s1-d.webp'] // grok edit-chain, in order
const canvas = document.getElementById('c')
const renderer = new THREE.WebGLRenderer({ canvas, antialias:true })
renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
const scene = new THREE.Scene(), cam = new THREE.OrthographicCamera(-1,1,1,-1,0,1)
const load = new THREE.TextureLoader()
const tex   = FRAMES.map(f => load.load(f))
const noise = load.load('/gen/noise.webp')                 // generated grayscale (assets.md §7)
const u = {
  uFrom:{value:tex[0]}, uTo:{value:tex[1]}, uMix:{value:0}, // 0..1 between the two bracketing frames
  uEnergy:{value:0}, uNoise:{value:noise},                 // audio pushes displacement
}
const mat = new THREE.ShaderMaterial({ uniforms:u,
  vertexShader:`varying vec2 vUv; void main(){ vUv=uv; gl_Position=vec4(position,1.); }`,
  fragmentShader:`precision highp float; varying vec2 vUv;
    uniform sampler2D uFrom,uTo,uNoise; uniform float uMix,uEnergy;
    void main(){
      float n = texture2D(uNoise, vUv).r;
      float amt = smoothstep(n-0.15, n+0.15, uMix);                 // noise wipe between frames
      vec2 disp = (uMix*(1.-uMix)+uEnergy) * 0.06 * vec2(n-0.5);    // bulge peaks mid-morph + on the beat
      vec3 a = texture2D(uFrom, vUv+disp).rgb;
      vec3 b = texture2D(uTo,   vUv-disp).rgb;
      gl_FragColor = vec4(mix(a,b,amt), 1.);
    }`})
scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2), mat))
const resize=()=>renderer.setSize(innerWidth,innerHeight); addEventListener('resize',resize); resize()

let target=0, cur=0
const seg = 1/(FRAMES.length-1)                            // scroll span per frame pair
const stage = document.getElementById('stage')
const setTarget=()=>{ const r=stage.getBoundingClientRect();
  target = Math.min(1, Math.max(0, -r.top/(r.height-innerHeight))) }
const lenis = new Lenis(); lenis.on('scroll', setTarget)
function raf(t){
  lenis.raf(t)
  cur += (target-cur)*0.08                                 // scrub smoothing (0.3-0.8 feel)
  const p = cur/seg, i = Math.min(FRAMES.length-2, Math.floor(p))
  u.uFrom.value = tex[i]; u.uTo.value = tex[i+1]; u.uMix.value = p - i
  u.uEnergy.value += (audioEnergy()-u.uEnergy.value)*0.2   // smoothed beat (see §2; 0 until audio starts)
  renderer.render(scene,cam); requestAnimationFrame(raf)
}
requestAnimationFrame(raf)
```

### 2. Audio-reactive — the same uniforms, on the beat

MiniMax score (assets.md §8) → Web Audio `AnalyserNode` → low-band energy → `uEnergy`, so the morph
pulses with the music. Start on the mute-toggle gesture (see Ambient audio); default OFF, and NEVER gate
the visual on audio — with sound off the scroll scrub is the whole show.

```js
let analyser, freq
function startAudio(){
  if (analyser) return
  const ctx = new AudioContext()
  const src = ctx.createMediaElementSource(document.getElementById('ambient'))
  analyser = ctx.createAnalyser(); analyser.fftSize = 256
  src.connect(analyser); analyser.connect(ctx.destination)
  freq = new Uint8Array(analyser.frequencyBinCount); ctx.resume()
}
document.getElementById('mute').addEventListener('click', startAudio)
function audioEnergy(){                                     // called from raf()
  if (!analyser) return 0
  analyser.getByteFrequencyData(freq)
  let bass=0; for (let k=0;k<8;k++) bass+=freq[k]
  return bass/8/255                                         // 0..1 low-band energy
}
```

### 3. Depth-map 2.5D composite (flat still → dimensional)

A hero still + its depth map (assets.md §2.5) → shader offsets UV per-pixel by depth × (pointer + a little
scroll) → parallax-occlusion "3D photo" + optional rack focus. Controlled, cheap, no 3D model.

```glsl
// uColor (hero), uDepth (grayscale 0 far…1 near), uPointer (vec2 -1..1), uFocus (0..1)
float d   = texture2D(uDepth, vUv).r;
vec2  off = uPointer * (d-0.5) * 0.04;          // near pixels shift more → parallax-occlusion
vec3  col = texture2D(uColor, vUv + off).rgb;
float coc = abs(d - uFocus);                     // rack focus: distance from the focal plane
// feed coc to a cheap multi-tap blur (or a separate DOF pass) for depth-of-field on scroll
gl_FragColor = vec4(col, 1.);
```
Pointer → `uPointer` (rAF-throttled, travel ≤ ~0.04 — Cursor-parallax rules); scroll can drift `uFocus`
for a rack-focus reveal as the section enters.

### Rules (or it's slop)
- ONE persistent WebGL context; swap textures, never remount per scene (memory climb + stutter = the #1 reported pitfall).
- Write uniforms / `element.style` in rAF ONLY — never React `setState` per scroll/audio frame (that's a 120Hz slideshow).
- Clamp: displacement & parallax travel small; `uEnergy` capped so the beat *breathes* the frame, not seizures it.
- `prefers-reduced-motion` → freeze on the nearest frame + a static end-state, no audio drive; content readable with JS off.
- Audio OFF by default, user-gesture to start; the scroll story must stand alone in silence.
- LOD: fewer / lower-res frames + drop the depth pass under `(max-width:768px)` or low-power — verify.md checks the tiers.
- Preload the frame chain and decode to `ImageBitmap` before first paint — mid-scroll texture decode is visible jank.
- **One function owns each shared quantity, and everything else calls it.** Scroll progress, camera path, the height of a terrain, the position of the peak: the moment two subsystems compute the same value separately they drift, and the drift looks like a rendering bug rather than a duplicated formula. A terrain whose height function lives inside the mesh component gets fog that lies flat and slices through the hills, and a cursor that dents the snow somewhere other than where it points — three readers, one truth, or three different worlds.
- **Every number that shapes the scene lives in one config module, with a comment saying why it is that number and what breaks otherwise.** Constants scattered across components make tuning a search problem, and a bare `0.36` teaches the next session nothing — `bevel: 0.36 // never 0: the highlight needs a chamfer to run along, a sharp edge reads as plastic` survives being "cleaned up".

---

## Tier-2 scene recipes

### 3D scroll-camera dolly — Blender path → glTF → Three.js

Author the move once in Blender (assets.md §9), then scrub the exported camera action. `quickTo` retargets six reusable values; no tween is created per frame.

```html
<section id="dolly-stage">
  <picture id="dolly-poster">
    <source srcset="/assets/gen/dolly-poster.webp" type="image/webp">
    <img src="/assets/gen/dolly-poster.png" alt="Bronze forms in a dark gallery">
  </picture>
  <canvas id="dolly" aria-hidden="true"></canvas>
</section>
```

```css
#dolly-stage { position: relative; display: grid; min-height: 400vh; }
#dolly, #dolly-poster {
  grid-area: 1 / 1;
  position: sticky;
  top: 0;
  display: block;
  width: 100%;
  height: 100vh;
}
#dolly { opacity: 0; }
#dolly-poster { opacity: 1; }
#dolly-poster img { width: 100%; height: 100%; object-fit: cover; }
#dolly, #dolly-poster { transition: opacity 200ms cubic-bezier(0.23, 1, 0.32, 1); }
```

```js
import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import gsap from 'gsap'
import ScrollTrigger from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

async function initDolly() {
  const canvas = document.getElementById('dolly')
  const poster = document.getElementById('dolly-poster')
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches

  let renderer
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true })
  } catch {
    return                                      // poster remains the complete no-WebGL scene
  }
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
  renderer.outputColorSpace = THREE.SRGBColorSpace

  const scene = new THREE.Scene()
  const gltf = await new GLTFLoader().loadAsync('/assets/gen/dolly.glb')
  scene.add(gltf.scene)

  const pathCamera = gltf.cameras.find(c => c.name === 'PathCamera')
  const clip = gltf.animations.find(c =>
    c.name === 'CameraPath' || c.tracks.some(track => track.name.startsWith('PathCamera.'))
  )
  if (!pathCamera || !clip) throw new Error('dolly.glb needs PathCamera + CameraPath')

  const mixer = new THREE.AnimationMixer(gltf.scene)
  mixer.clipAction(clip).play()
  const camera = pathCamera.clone(false)
  const sampledPosition = new THREE.Vector3()
  const sampledDirection = new THREE.Vector3()
  const sampledLook = new THREE.Vector3()
  const look = new THREE.Vector3()

  const moveX = gsap.quickTo(camera.position, 'x', { duration: 0.28, ease: 'power3.out' })
  const moveY = gsap.quickTo(camera.position, 'y', { duration: 0.28, ease: 'power3.out' })
  const moveZ = gsap.quickTo(camera.position, 'z', { duration: 0.28, ease: 'power3.out' })
  const lookX = gsap.quickTo(look, 'x', { duration: 0.28, ease: 'power3.out' })
  const lookY = gsap.quickTo(look, 'y', { duration: 0.28, ease: 'power3.out' })
  const lookZ = gsap.quickTo(look, 'z', { duration: 0.28, ease: 'power3.out' })

  const resize = () => {
    const width = canvas.clientWidth
    const height = canvas.clientHeight
    renderer.setSize(width, height, false)
    camera.aspect = width / height
    camera.updateProjectionMatrix()
  }
  addEventListener('resize', () => requestAnimationFrame(resize), { passive: true })
  resize()

  const samplePath = (progress, immediate = false) => {
    mixer.setTime(progress * clip.duration)
    pathCamera.updateWorldMatrix(true, false)
    pathCamera.getWorldPosition(sampledPosition)
    pathCamera.getWorldDirection(sampledDirection)
    sampledLook.copy(sampledPosition).add(sampledDirection)
    if (immediate) {
      camera.position.copy(sampledPosition)
      look.copy(sampledLook)
      return
    }
    moveX(sampledPosition.x); moveY(sampledPosition.y); moveZ(sampledPosition.z)
    lookX(sampledLook.x); lookY(sampledLook.y); lookZ(sampledLook.z)
  }

  const revealCanvas = () => {
    canvas.style.opacity = '1'
    poster.style.opacity = '0'
  }

  if (reduceMotion) {
    samplePath(0.55, true)                      // static authored composition, not a blank canvas
    camera.lookAt(look)
    renderer.render(scene, camera)
    requestAnimationFrame(revealCanvas)
    return
  }

  samplePath(0, true)
  camera.lookAt(look)
  renderer.render(scene, camera)
  let progress = 0
  let active = true
  const trigger = ScrollTrigger.create({
    trigger: '#dolly-stage',
    start: 'top top',
    end: 'bottom bottom',
    scrub: 0.5,
    onToggle: self => { active = self.isActive },
    onUpdate: self => { progress = self.progress },       // state only; WebGL writes stay in rAF
  })
  const tick = () => {
    if (!active) return
    samplePath(progress)
    camera.lookAt(look)
    renderer.render(scene, camera)
  }
  gsap.ticker.add(tick)
  tick()
  requestAnimationFrame(revealCanvas)

  return () => {
    trigger.kill()
    gsap.ticker.remove(tick)
    mixer.stopAllAction()
    gltf.scene.traverse(o => { o.geometry?.dispose(); o.material?.dispose() })
    renderer.dispose()
  }
}

initDolly()
```

If the Tier-1 engine already owns `renderer`, pass that renderer/scene into `initDolly`; never construct a second context. glTF carries Blender's coordinate conversion, lens, camera transform, and baked animation.

**Rules (or it's slop):** one persistent renderer; `ScrollTrigger` owns progress; all pose/render/style writes happen in GSAP's rAF ticker; `quickTo` is created once; reduced motion renders one authored camera frame; keep the poster for no-WebGL and dispose on route teardown.

### Signature post-FX pass — photochemical misregistration

Commit to one look: restrained channel misregistration plus 2% moving grain. No bloom stack; highlights and color stay authored in the scene, while scroll and Tier-1 `uEnergy` only make the print breathe.

```js
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js'
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js'

const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches
let scrollFxTarget = 0
let scrollFx = 0

ScrollTrigger.create({
  trigger: '#stage',
  start: 'top bottom',
  end: 'bottom top',
  scrub: 0.5,
  onUpdate: self => { scrollFxTarget = self.progress },    // state only
})

let composer, signature
if (!reduceMotion) {
  composer = new EffectComposer(renderer)                  // same renderer/context as Tier-1
  composer.addPass(new RenderPass(scene, cam))
  signature = new ShaderPass({
    uniforms: {
      tDiffuse: { value: null },
      uTime: { value: 0 },
      uScroll: { value: 0 },
      uEnergy: { value: 0 },
    },
    vertexShader: `
      varying vec2 vUv;
      void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
    `,
    fragmentShader: `
      precision highp float;
      uniform sampler2D tDiffuse;
      uniform float uTime, uScroll, uEnergy;
      varying vec2 vUv;

      float hash(vec2 p) {
        return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
      }
      void main() {
        float edge = 0.00035 + 0.00065 * uScroll + 0.00045 * uEnergy;
        vec2 split = vec2(edge, 0.0);
        float r = texture2D(tDiffuse, vUv + split).r;
        float g = texture2D(tDiffuse, vUv).g;
        float b = texture2D(tDiffuse, vUv - split).b;
        float grain = (hash(gl_FragCoord.xy + floor(uTime * 24.0)) - 0.5) * 0.02;
        gl_FragColor = vec4(vec3(r, g, b) + grain, 1.0);
      }
    `,
  })
  composer.addPass(signature)                              // the ONE signature pass
}

// Call from the existing Tier-1 requestAnimationFrame loop in place of renderer.render().
function renderWithSignature(time) {
  scrollFx += (scrollFxTarget - scrollFx) * 0.08
  if (reduceMotion) {
    renderer.render(scene, cam)                            // clean static/gentle art direction
    return
  }
  signature.uniforms.uTime.value = time * 0.001
  signature.uniforms.uScroll.value = scrollFx
  signature.uniforms.uEnergy.value = Math.min(0.35, u.uEnergy.value)
  composer.render()
}
```

Non-WebGL layers get the same imperfect-print idea without another canvas: one SVG displacement filter, drifting slowly on GSAP's rAF ticker.

```html
<svg width="0" height="0" aria-hidden="true" focusable="false">
  <filter id="paper-warp" x="-3%" y="-3%" width="106%" height="106%">
    <feTurbulence id="paper-noise" type="fractalNoise" baseFrequency="0.009 0.013"
      numOctaves="1" seed="7" result="noise"/>
    <feDisplacementMap in="SourceGraphic" in2="noise" scale="2.5"
      xChannelSelector="R" yChannelSelector="G"/>
  </filter>
</svg>
<figure class="paper-fx"><img src="/assets/gen/scene-poster.webp" alt="Product still"></figure>
```

```css
.paper-fx { filter: url("#paper-warp"); }
@media (prefers-reduced-motion: reduce) {
  .paper-fx { filter: none; }
}
```

```js
const turbulence = document.getElementById('paper-noise')
if (turbulence && !reduceMotion) {
  gsap.ticker.add(time => {
    const x = 0.009 + Math.sin(time * 0.35) * 0.0007
    const y = 0.013 + Math.cos(time * 0.27) * 0.0007
    turbulence.setAttribute('baseFrequency', `${x.toFixed(4)} ${y.toFixed(4)}`)
  })
}
```

**Rules (or it's slop):** one custom pass, no effect buffet; aberration ≤0.0015 UV and grain ≈2%; update uniforms/SVG attributes in rAF only; reuse Tier-1 `uEnergy`; reduced motion removes both GPU and SVG passes.

### Physics pointer-trail — alpha cut-outs with gravity

Transparent PNG cut-outs inherit pointer velocity, then fall through a tiny Verlet step. Pointer events only record state; spawn, physics, draw, fade, and cull happen in rAF.

```html
<canvas id="pointer-trail" aria-hidden="true"></canvas>
```

```css
#pointer-trail {
  position: fixed;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 60;
}
@media (hover: none), (prefers-reduced-motion: reduce) {
  #pointer-trail { display: none; }
}
```

```js
const canTrail =
  matchMedia('(hover: hover) and (pointer: fine)').matches &&
  !matchMedia('(prefers-reduced-motion: reduce)').matches

if (canTrail) {
  const canvas = document.getElementById('pointer-trail')
  const ctx = canvas.getContext('2d', { alpha: true })
  const paths = ['/assets/cutout-01.png', '/assets/cutout-02.png', '/assets/cutout-03.png']
  const sprites = paths.map(src => {
    const image = new Image()
    image.decoding = 'async'
    image.src = src
    return image
  })
  Promise.allSettled(sprites.map(image => image.decode()))

  const MAX_SPRITES = 28
  const SPAWN_DISTANCE = 46
  const GRAVITY = 0.28
  const DRAG = 0.985
  const particles = []
  const pointer = { x: 0, y: 0, dx: 0, dy: 0, travel: 0, pending: false, seen: false }
  let raf = 0
  let spriteIndex = 0

  const resize = () => {
    const dpr = Math.min(devicePixelRatio, 2)
    canvas.width = Math.round(innerWidth * dpr)
    canvas.height = Math.round(innerHeight * dpr)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  }
  resize()
  addEventListener('resize', () => requestAnimationFrame(resize), { passive: true })

  const spawn = () => {
    const image = sprites[spriteIndex++ % sprites.length]
    if (!image.complete || !image.naturalWidth || pointer.y > innerHeight - 120) return
    if (particles.length === MAX_SPRITES) particles.shift()
    const vx = Math.max(-7, Math.min(7, pointer.dx * 0.32))
    const vy = Math.max(-5, Math.min(5, pointer.dy * 0.22))
    const size = 42 + Math.random() * 34
    particles.push({
      image,
      x: pointer.x,
      y: pointer.y,
      px: pointer.x - vx,
      py: pointer.y - vy,
      size,
      age: 0,
      life: 72 + Math.random() * 30,
      rotation: (Math.random() - 0.5) * 0.25,
      spin: (Math.random() - 0.5) * 0.035,
    })
  }

  const wake = () => {
    if (!raf) raf = requestAnimationFrame(frame)
  }

  function frame() {
    raf = 0
    if (pointer.pending) {
      if (pointer.travel >= SPAWN_DISTANCE) {
        pointer.travel %= SPAWN_DISTANCE
        spawn()                                      // at most one spawn per animation frame
      }
      pointer.pending = false
    }

    ctx.clearRect(0, 0, innerWidth, innerHeight)
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i]
      const vx = (p.x - p.px) * DRAG
      const vy = (p.y - p.py) * DRAG
      p.px = p.x; p.py = p.y
      p.x += vx
      p.y += vy + GRAVITY
      p.rotation += p.spin
      p.age++

      const remaining = 1 - p.age / p.life
      if (remaining <= 0 || p.y - p.size > innerHeight ||
          p.x + p.size < 0 || p.x - p.size > innerWidth) {
        particles.splice(i, 1)
        continue
      }

      ctx.save()
      ctx.translate(p.x, p.y)
      ctx.rotate(p.rotation)
      ctx.globalAlpha = Math.min(1, p.age / 6) * Math.min(1, remaining * 4)
      ctx.drawImage(p.image, -p.size / 2, -p.size / 2, p.size, p.size)
      ctx.restore()
    }
    if (particles.length || pointer.pending) wake()
  }

  addEventListener('pointermove', event => {
    if (!pointer.seen) {
      pointer.x = event.clientX
      pointer.y = event.clientY
      pointer.seen = true
      return
    }
    pointer.dx = event.clientX - pointer.x
    pointer.dy = event.clientY - pointer.y
    pointer.travel += Math.hypot(pointer.dx, pointer.dy)
    pointer.x = event.clientX
    pointer.y = event.clientY
    pointer.pending = true
    wake()                                           // event writes state; rAF owns every visual write
  }, { passive: true })
}
```

**Rules (or it's slop):** cap at 28 sprites; gravity ≈0.28px/frame²; one spawn maximum per rAF; cull by lifetime and viewport; no DOM sprite churn; disable on touch and reduced motion.

### Living type — variable axes + gooey nav

Flex one display/nav face, not the reading layer. Scroll sets the phrase's posture; Web Audio adds a small pulse through the same rAF-owned CSS variables.

```html
<h2 class="living-title">Matter changes under pressure.</h2>
<p class="scene-copy">Body copy stays fixed at a tested reading weight.</p>

<svg width="0" height="0" aria-hidden="true" focusable="false">
  <filter id="nav-goo" x="-20%" y="-40%" width="140%" height="180%">
    <feGaussianBlur in="SourceGraphic" stdDeviation="7" result="blur"/>
    <feColorMatrix in="blur" type="matrix" values="
      1 0 0 0 0
      0 1 0 0 0
      0 0 1 0 0
      0 0 0 18 -7" result="goo"/>
    <feComposite in="SourceGraphic" in2="goo" operator="over"/>
  </filter>
</svg>

<nav class="goo-nav" aria-label="Primary">
  <a href="#work">Work</a>
  <a href="#process">Process</a>
  <button id="type-audio" type="button" aria-pressed="false">Sound off</button>
</nav>
<audio id="type-score" src="/assets/ambient.opus" muted loop preload="none"></audio>
```

```css
@font-face {
  font-family: "Brand Variable";
  src: url("/fonts/brand-variable.woff2") format("woff2");
  font-weight: 300 800;
  font-stretch: 75% 125%;
  font-display: swap;
}
.living-title {
  --vf-wght: 570;
  --vf-wdth: 96;
  font-family: "Brand Variable", serif;
  font-variation-settings: "wght" var(--vf-wght), "wdth" var(--vf-wdth);
}
.scene-copy {
  max-width: 70ch;
  font-family: "Source Sans 3", system-ui, sans-serif;
  font-weight: 420;
  line-height: 1.5;
}
.goo-nav {
  display: flex;
  gap: 0.2rem;
  align-items: center;
  filter: url("#nav-goo");
}
.goo-nav :is(a, button) {
  border: 0;
  border-radius: 999px;
  padding: 0.65rem 0.9rem;
  color: #fff;
  background: #b43d18;
  font: 600 0.9rem/1 "Brand Variable", serif;
  text-decoration: none;
  transition: transform 140ms cubic-bezier(0.23, 1, 0.32, 1);
}
@media (hover: hover) and (pointer: fine) {
  .goo-nav :is(a, button):hover { transform: translateY(-2px); }
}
@media (prefers-reduced-motion: reduce) {
  .living-title { font-variation-settings: "wght" 600, "wdth" 98; }
  .goo-nav :is(a, button) { transition: none; }
}
```

```js
const title = document.querySelector('.living-title')
const audio = document.getElementById('type-score')
const audioButton = document.getElementById('type-audio')
const reduceTypeMotion = matchMedia('(prefers-reduced-motion: reduce)').matches
let scrollTypeTarget = 0
let scrollType = 0
let audioContext, analyser, bins

ScrollTrigger.create({
  trigger: title,
  start: 'top bottom',
  end: 'bottom top',
  scrub: 0.5,
  onUpdate: self => { scrollTypeTarget = self.progress },  // state only
})

audioButton.addEventListener('click', async () => {
  if (!audioContext) {
    audioContext = new AudioContext()
    const source = audioContext.createMediaElementSource(audio)
    analyser = audioContext.createAnalyser()
    analyser.fftSize = 256
    bins = new Uint8Array(analyser.frequencyBinCount)
    source.connect(analyser).connect(audioContext.destination)
  }
  await audioContext.resume()
  audio.muted = !audio.muted
  if (!audio.muted && audio.paused) await audio.play()
  audioButton.setAttribute('aria-pressed', String(!audio.muted))
  audioButton.textContent = audio.muted ? 'Sound off' : 'Sound on'
})

function typeEnergy() {
  if (!analyser || audio.muted) return 0
  analyser.getByteFrequencyData(bins)
  let low = 0
  for (let i = 0; i < 8; i++) low += bins[i]
  return low / 8 / 255
}

function animateType() {
  scrollType += (scrollTypeTarget - scrollType) * 0.08
  const energy = typeEnergy()
  const weight = 570 + scrollType * 52 + energy * 16       // 570..638
  const width = 96 + scrollType * 4 + energy * 1.5         // 96..101.5
  title.style.setProperty('--vf-wght', weight.toFixed(1))
  title.style.setProperty('--vf-wdth', width.toFixed(1))
  requestAnimationFrame(animateType)
}
if (!reduceTypeMotion) requestAnimationFrame(animateType)   // reduced branch stays at CSS 600/98
```

The SVG filter merges adjacent nav pills; `feComposite` restores crisp labels after the alpha threshold. Verify the font actually exposes `wght`/`wdth`; substitute `opsz` only when the file declares it.

**Rules (or it's slop):** axis travel stays subtle; CSS variables update in rAF only; reduced motion freezes the axes; body copy never flexes; audio is muted by default and starts only on the button gesture; goo belongs to one compact nav, not the page.

## Assembly order

Build in this sequence — each step depends on the previous.

1. **Smooth scroll foundation first.** Wire Lenis + GSAP integration (`lenis.on('scroll', ScrollTrigger.update)` + `lagSmoothing(0)`). Do not skip — all scroll effects jitter without it.
2. **Hero section.** Scrub video or Three.js displacement. This sets the site's tone and reveals performance constraints early.
3. **Sections top-to-bottom.** After each section's ScrollTrigger is initialized, call `ScrollTrigger.refresh()` so positions are recalculated with the full DOM height.
4. **Text reveals last.** SplitType/SplitText on headings ties the sections together and is cheapest to adjust.

**Every effect ships with its fallback chain:**

| Effect | Primary | Fallback 1 | Fallback 2 |
|---|---|---|---|
| Three.js / WebGL scene | WebGL | `<picture>` static poster | Nothing (CSS background) |
| CSS `animation-timeline` | Native API | Dynamic GSAP import via `@supports` check | Static visible state |
| Scrub video | `<video>` + ScrollTrigger | Canvas image sequence on `canplay` timeout | Static poster image |
| Ambient audio | WebAudio + `<audio>` | Muted by default | Silent |
| Text stagger | SplitType + GSAP | CSS transition + IntersectionObserver | Visible immediately |

**Reduced-motion rule:** all techniques must check `prefers-reduced-motion: reduce`. For scroll-driven effects, show static poster + soft `opacity` only. For text reveals, skip to final visible state. For audio, default muted and do not auto-start.

```js
// Utility — gate any animation behind the media query
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

if (!reduceMotion) {
  // init SplitType, ScrollTrigger scrub, etc.
} else {
  // reveal all text immediately, show poster frame
  document.querySelectorAll('.reveal-text').forEach(el => el.classList.add('is-visible'))
}
```

**easing calibration:** scrub speed 0.3–0.8 (never 1.0 — too rigid). Entrance easing: `expo.out`, `power3.out`, or `cubic-bezier(0.16, 1, 0.3, 1)`. Nothing linear except scrub progress mapping itself.
