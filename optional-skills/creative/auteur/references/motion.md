# Motion Reference — auteur skill

Numeric, enforceable animation rules distilled from 13 motion sources. Every number is exact. Conflicts are resolved; only the winning rule appears.

---

## When to animate

Animate only when the motion answers one of these six questions:

1. **Hierarchy** — does it show what matters most?
2. **Storytelling** — does it narrate a sequence?
3. **Feedback** — does it confirm an action?
4. **State transition** — does it show what changed?
5. **Spatial consistency** — does it orient the user in space?
6. **Preventing jarring change** — does it smooth a discontinuity?

"Looks cool" is not a reason. If none of the six apply, delete the animation.

**Frequency decision framework** — stop at the first row that matches:

| How often the user triggers this | Rule |
|---|---|
| 100+ times/day (keyboard shortcuts, command palette) | Zero animation, ever |
| Tens/day (hover, list navigation) | Drastically reduce — near zero |
| Occasional (modal, drawer, toast) | Standard motion allowed |
| Rare / first-time experience | Can add delight |

Apply this before writing any transition. A command palette toggle with a 200ms fade is a P1 block.

---

## Easing

**The resolved policy** (Emil over raphaelsalaja for UI):

- **Enter → `ease-out`**. Arrives fast, settles gently. Feels faster than `ease-in` at identical duration.
- **Exit → `ease-out`** (same as enter for UI menus, drawers, toasts — this is the system-response model).
- **`ease-in` is banned on all UI motion.** Reserve it exclusively for Web Audio gain envelopes (exponential release before silence).
- **Marquee / progress bars / time representation → `linear`** only. Never use linear for positional motion.
- **On-screen morph (element repositions while visible) → `ease-in-out`.**
- **Hover / color → `ease` (CSS default).**

Built-in CSS easing curves are too weak. Always use custom curves:

```css
:root {
  --ease-out-quart: cubic-bezier(0.23, 1, 0.32, 1);      /* default for enter/exit */
  --ease-in-out-quart: cubic-bezier(0.77, 0, 0.175, 1);  /* on-screen morphs */
  --ease-drawer: cubic-bezier(0.32, 0.72, 0, 1);         /* large panel slides */
}
```

For spring-like bounces without a spring library, use `linear()` with sampled keyframes (CSS `linear()` function, widely supported 2024+).

---

## Duration

Default table — apply literally, justify any deviation in a comment:

| Element | Duration |
|---|---|
| Button press / tap feedback | 100–160 ms |
| Tooltip appear | 125–200 ms |
| Dropdown / select open | 150–250 ms |
| Modal / drawer enter | 200–500 ms |
| Marketing / explanatory sequences | Longer allowed |

**Hard rule: any UI transition over 300 ms requires a written justification** (comment in code or design note). No exceptions. If the animation feels slow, shorten the duration first — do not sharpen the curve as the primary fix.

Similar elements must use identical timing. `button-primary 200ms` vs `button-secondary 150ms` is a fail.

Modal exit is faster than enter (release snap): enter 200 ms, exit 150 ms.

---

## Spring vs easing

Decision table — pick one row and commit:

| Motion type | Best choice | Why |
|---|---|---|
| User-driven (drag, flick, gesture) | Spring | Survives interruption; preserves velocity |
| System-driven (state change, feedback) | Easing | Clear start/end, predictable timing |
| Time representation (progress, loading) | Linear | 1:1 time-to-progress |
| High-frequency (typing, fast toggles) | None | Adds noise, makes UI feel slower |

**Spring parameters:**
- Gesture / drag: `stiffness: 500, damping: 30` — balanced, no excessive bounce.
- Apple-style (preferred for simplicity): `{ type: "spring", duration: 0.5, bounce: 0.2 }`.
- Bounce > 0.3 only for drag-to-dismiss and explicitly playful contexts. Never in standard UI.
- Preserve velocity on flick: `animate(target, { x: 0 }, { type: "spring", velocity: info.velocity.x })`.

**Rapidly-triggered elements (toasts, toggles) → CSS `transition`, not `@keyframes`.** Keyframes restart from zero on re-trigger; transitions retarget mid-flight smoothly.

**Modal system state change → 200 ms `ease-out`, not spring.** Spring on a toast feels restless.

---

## Physicality

**Never `transform: scale(0)` for entrance.** Nothing in the real world appears from nothing. Start at `scale(0.95)` + `opacity: 0` at minimum; `scale(0.97)` is the safe default for small UI elements.

**Press / tap squash-stretch:** `scale` range `0.95–1.05`. The standard:

```css
button:active {
  transform: scale(0.97);
  transition: transform 160ms var(--ease-out-quart);
}
```

`whileTap={{ scale: 0.8 }}` is a P1 fail — too exaggerated.

**Origin-aware popovers and dropdowns** — the element must scale from its trigger, not from its own center:

```css
/* When using Radix UI */
[data-radix-popper-content-wrapper] > * {
  transform-origin: var(--radix-popover-content-transform-origin);
}

/* When using Base UI */
[data-popup] {
  transform-origin: var(--transform-origin);
}
```

**Modals are exempt from origin-awareness** — keep `transform-origin: center` on modals. They represent a system interrupt, not a trigger-anchored element.

Never set `transform-origin: center` on trigger-anchored popovers, tooltips, or dropdowns.

---

## Performance

**GPU-composited properties only: `transform` and `opacity`.** Animating `width`, `height`, `top`, `left`, `margin`, or `padding` forces layout → paint → composite on every frame. This is unanimously banned across all 13 sources.

**`window.addEventListener('scroll', …)` is banned** — jank-prone, no batching, blocks main thread. Use instead:

- Framer Motion: `useScroll()` + `useTransform()`
- GSAP: `ScrollTrigger`
- Vanilla: `IntersectionObserver`
- CSS: `animation-timeline: view()`

**Framer Motion shorthands (`x`, `y`, `scale` as separate props) are not hardware-accelerated under load** — they run on the main thread via rAF. For pinned sections and scroll-scrubbed animations, use full transform strings or GSAP:

```tsx
// Weak under scroll load:
<motion.div animate={{ x: 100, scale: 1.2 }} />

// Correct for pinned / scroll-driven:
<motion.div animate={{ transform: "translateX(100px) scale(1.2)" }} />
// or migrate to GSAP for the section
```

**Never drive a child's transform via a CSS variable on a parent** — causes style-recalc storm on all children. Set `transform` directly on the target element.

**Continuous values (mouse position, scroll progress, pointer physics) → `useMotionValue` + `useTransform`, never `useState`.** `useState` triggers a React re-render per scroll tick; `useMotionValue` updates the DOM directly.

```tsx
// Banned:
const [scrollY, setScrollY] = useState(0);
useEffect(() => { window.addEventListener('scroll', () => setScrollY(window.scrollY)); }, []);

// Correct:
const { scrollY } = useScroll();
const opacity = useTransform(scrollY, [0, 300], [1, 0]);
```

`useEffect` animations must always include cleanup (`gsap.context()` + `ctx.revert()`, or Motion's unsubscribe).

`will-change: transform` — use sparingly, only on elements that are actively animating. It promotes to a GPU layer immediately; overuse wastes VRAM.

Grain / noise filter overlays: only on `position: fixed; inset: 0; pointer-events: none; z-index: 60` pseudo-elements. Never on scrolling containers — continuous GPU repaints destroy mobile FPS.

### Fullscreen passes are priced per pixel, not per object

A scene rarely dies of geometry. Hundreds of thousands of triangles, thousands of particles, shadows and volumetric fog all fit inside a 16.7ms frame. What eats the budget is every pass that touches the whole screen, because those cost the same whether the frame contains one sphere or a city. Order of magnitude, measured on a retina laptop (1440×900 @2x = 5.2MP) for one WebGL scene:

| Pass | ~cost / frame | |
|---|---|---|
| chromatic aberration + grain | 8ms | the "free" cinematic layer is the most expensive thing on the page |
| bloom | 7ms | at half-res; dropping to quarter-res saved 0.7ms — the cost is compositing over the frame, not the blur |
| custom transition shader | 5ms | |
| depth of field | 17ms | over the entire budget alone; it was cut, not optimized |

Read the **order**, not the absolutes — your GPU differs, and summing these is meaningless because passes overlap. Three rules follow:

- **Pixel count is the main lever — for pages that have these passes.** A scene carrying DoF + bloom + grain runs 60fps at 2MP and 30fps at 4.5MP. A scene with no fullscreen pass barely notices: measured on three showcase sites at 4× CPU throttle, DPR 1 → 2 moved minFps by 0–1 (53→54, 53→53, 54→54), because there the ceiling is the main thread, not fillrate. Still measure at DPR 2 — the day a bloom lands, the honest number is already the one you have been quoting. Cap `renderer.setPixelRatio(Math.min(devicePixelRatio, 2))`, and when a scene is over budget, cut resolution or a pass before you cut geometry.
- **Measure by ablation** — switch passes off one at a time and re-measure. Intuition is wrong about which one hurts: shadows usually turn out nearly free, and the effect that "barely does anything" is often the 8ms one.
- **Measure the production build.** A dev server costs roughly 2× per frame (HMR client, unminified bundles, no asset pipeline), so its numbers describe a page nobody will load. `motionqa.mjs` flags a detected dev server, but it cannot detect every one of them.

---

## Stagger and orchestration

- **Stagger delay: 30–80 ms between items.** Upper bound is 50 ms per item for lists — anything longer makes the reveal feel broken.
- Stagger is decorative. **It must never block interaction.** The list is interactive from the moment it renders; the stagger is cosmetic only.
- **Reveal animations must enhance an already-visible default.** Content must be readable with JavaScript disabled, because CSS transitions pause in hidden tabs — a section that starts `opacity: 0` via JS will ship blank in that case.

```tsx
// Motion RevealStagger skeleton (feature lists, testimonials, logo walls):
initial={{ opacity: 0, y: 24 }}
whileInView={{ opacity: 1, y: 0 }}
viewport={{ once: true, amount: 0.3 }}
transition={{ duration: 0.6, delay: i * 0.06, ease: [0.16, 1, 0.3, 1] }}
```

`staggerChildren` in Framer Motion requires parent and child to be in the same Client Component tree. Async data → pass through props into a centralized parent Motion wrapper.

**Library routing:**
- Framer Motion — UI components, Bento layouts, state-change animations.
- GSAP + ScrollTrigger — full-page scrolltelling, pinned sections, horizontal pans.
- Never mix GSAP/Three.js and Framer Motion in the same component tree.

---

## Motion budget

Page-level constraints that most motion guidance omits:

- **Max 3 distinct scroll-triggered animation families per page.** (A "family" = a combination of easing + distance + direction. Three fade-up variants count as one if identical.)
- **Each additional scroll reveal must differ from the previous in at least one dimension** — easing, distance, or direction. Uniform fade-in on every section is a fail.
- **Marquee: max 1 per page.**
- **One primary "wow" peak per page.** Supporting scenes run at lower visual intensity. Two hero-level spectacles compete and cancel each other.
- If a storyboard scene claims intensity >4, the scene must visibly move. If it can't (asset missing, perf budget), downshift the scene's intensity honestly instead of faking it with decoration.

---

## Modals, drawers, toasts

- **Modals:** `transform-origin: center`. Enter 200 ms `ease-out`; exit 150 ms (faster, release snap). Spring is wrong here — use easing.
- **Drawers / toasts:** CSS `transition`, not `@keyframes` — these are rapidly triggered and must retarget smoothly on re-trigger. `@starting-style { opacity: 0; transform: translateY(100%); }` for CSS-only entry without JS.
- **Tooltips:** suppress delay and animation on subsequent hovers — after the first tooltip, all are instant:
  ```css
  [data-instant] { transition-duration: 0ms; }
  ```
- **Drag-to-dismiss:** use momentum, not distance threshold. `Math.abs(distance) / elapsedTime > 0.11` → dismiss. A flick is enough.
- Enable pointer capture during drag so motion continues after the cursor leaves the element.
- Multi-touch protection: `if (isDragging) return;` — ignore new touch points after drag begins.

---

## Reduced motion

`@media (prefers-reduced-motion: reduce)` is **mandatory for any scroll-driven animation, parallax, or large-scale motion.** Not optional.

**Reduced = gentler, not zero.** Treat it as an alternative art direction:

| Keep | Drop |
|---|---|
| `opacity` transitions | `transform` movement |
| `color` / `background` transitions | Parallax offsets |
| Subtle scale (≤ 2%) | Scroll-scrubbing |
| State indication | Entrance slide-in |

```css
@media (prefers-reduced-motion: reduce) {
  .animated-section {
    /* opacity-only fallback — transforms removed, state still visible */
    transform: none !important;
    animation: none !important;
    transition: opacity 200ms ease;
  }
}
```

---

## Hover

Gate all hover effects behind the pointer media query — touch devices fire false hover states on tap:

```css
@media (hover: hover) and (pointer: fine) {
  .card:hover {
    transform: translateY(-4px);
    transition: transform 200ms var(--ease-out-quart);
  }
}
```

No hover animation outside this gate. Ever.

---

## Sound

Sound is a parallel channel to motion — it follows the same budget discipline.

**Use sound only for:**
- Confirmation (payment completed, file uploaded, form submitted)
- Error state
- Notification / alert

**Never use sound for:** typing, hover, scroll events, keyboard navigation — keyboard nav with click sounds becomes unbearable immediately.

**Implementation rules:**

```ts
// singleton — new AudioContext() per call leaks nodes and hits mobile context limits
let _ctx: AudioContext | null = null;
function getAudioContext(): AudioContext {
  if (!_ctx) _ctx = new AudioContext();
  if (_ctx.state === 'suspended') _ctx.resume();
  return _ctx;
}

function playConfirm() {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return; // doubles as reduced-sound
  const ctx = getAudioContext();
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.connect(gain); gain.connect(ctx.destination);
  gain.gain.setValueAtTime(0.3, ctx.currentTime); // default 0.3, never 1.0
  gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4); // exponential, not linear
  osc.start(); osc.stop(ctx.currentTime + 0.4);
  osc.onended = () => { osc.disconnect(); gain.disconnect(); };
}
```

- Default volume: **0.3**. Never 1.0.
- Envelope decay: **`exponentialRampToValueAtTime(0.001, t)`**, not `linearRampToValueAtTime(0, t)`. Linear sounds mechanical; exponential matches human perception. Always call `setValueAtTime` before ramping.
- `prefers-reduced-motion` doubles as reduced-sound — if the media query matches, skip playback entirely.
- Provide an explicit sound toggle in settings: `<SoundProvider enabled={soundEnabled} />`.
- Sound weight must match action weight: soft click for toggle, success chime for purchase. A loud buzzer for form validation is punishing — never do this.
- Click/tap sounds: 5–15 ms duration, bandpass filter 3 000–6 000 Hz, Q 2–5.
- Rapid re-trigger: `audio.currentTime = 0` before `play()`.

---

## Anti-pattern quick reference

| Pattern | Why it fails |
|---|---|
| `transition: all` | Animates every property including layout — unbounded |
| `scale(0)` entrance | Nothing appears from nothing; start at 0.95 |
| `ease-in` on UI | Feels slower than ease-out at identical duration |
| Animation on 100+/day actions | Accumulates into constant noise |
| UI duration > 300 ms, no justification | Noticeably slow |
| `transform-origin: center` on trigger-anchored popovers | Scales from wrong origin |
| `@keyframes` on toasts / toggles | Restarts from zero on re-trigger |
| Animating `width/height/margin/top/left` | Forces layout + paint every frame |
| Framer Motion `x/y/scale` under scroll load | Main-thread rAF, not composited |
| CSS variable on parent to drive child transform | Style-recalc storm on all children |
| Missing `prefers-reduced-motion` | Accessibility block |
| `:hover` without `(hover: hover) and (pointer: fine)` | False-fires on touch |
| Uniform fade-in on every scroll section | Violates motion budget |
| > 1 marquee per page | Visual noise |
| `new AudioContext()` per call | Leaks nodes; crashes on mobile |
| `linearRampToValueAtTime(0, t)` for decay | Sounds mechanical |
| Sound on hover / scroll / keyboard nav | Unbearable at speed |
| Default volume 1.0 | Jarring |
