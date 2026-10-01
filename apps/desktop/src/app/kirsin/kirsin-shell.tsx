import { useStore } from '@nanostores/react'
import { type CSSProperties, useCallback, useContext, useEffect, useRef, useState } from 'react'

import { OverflowTip, Tip } from '@/components/ui/tooltip'
import { useThemeEpoch } from '@/hooks/use-theme-epoch'
import { useI18n } from '@/i18n'
import { $busy, $currentModel } from '@/store/session'

import { ContribWiringContext } from '../contrib/context'
import { WiredPane } from '../contrib/wiring'

import { useKirsinHeaderDrag } from './kirsin-drag'
import { KirsinSessionSwitcher } from './kirsin-session-switcher'

// Renderer-side mirror of the window's geometry (see electron/kirsin-geometry.ts).
// The window is created non-resizable, so the ONLY size changes are these
// manual morphs: the header chevron drives setBounds between the 560px chat
// and the 96px pill, and the orb button (Rosario's Augmentor reference)
// tweens the window down to the 196px orb (a 150px glowing circle). Width
// is 576 in the chat/pill states; only the orb narrows it.
const KIRSIN_W = 576
const KIRSIN_H_COMPACT = 96
const KIRSIN_H_EXPANDED = 560
// The orb's WINDOW is 196: the painted circle is 150px, centered in the
// window, with a ~23px margin around it so the glow halo is never clipped
// by the window edge (see "Orb morph (Phase 9)" in styles.css).
const KIRSIN_ORB = 196

// The ORB is a Jarvis-style HOLOGRAM (Rosario 2026-09-26: "create something
// Jarvis-like" — the Iron-Man hologram reference). It is a single rAF-driven
// <canvas> (NOT an SVG, NOT a CSS animation): the user's OS has "show
// animations" OFF, so the app's blanket reduced-motion rule freezes every CSS
// animation, but canvas drawing is unaffected — the hologram keeps living.
//
// The hologram (see drawHolo below): a soft reactor GLOW at the core, a slowly
// rotating dot-matrix SPHERE (the reference's "energy ball"), a few thin
// concentric RINGS, an organic DOT RING floating outside the rim (each dot
// creeps, wobbles and twinkles on its own clock — it never spins as a body),
// and an energy MEMBRANE on
// the edge whose shape undulates with random organic bumps (Approach B, but
// random). Every layer derives from ONE clock (the accumulated `acc`) so the
// motion is a single coherent phase, not a scatter of independent timers.
// (The membrane bumps additionally keep a tiny frame-delta-driven state, so
// the rim stays organic rather than a fixed repeating sine.)
// Canvas logical size — equals the orb WINDOW (196), so the hologram's center
// is the window center and the 75px ring lands exactly on the 150px disc's
// edge (the rim rides the visible border; the membrane + outer dot ring's
// glow reach out into the 23px halo margin, never clipped by the window).
const HOLO_SIZE = 196
// Main ring radius — matches the 150px disc (the hologram's rim rides the
// disc's edge, so the membrane + outer dot ring align with the border).
const HOLO_RING = 75
// The dot-matrix sphere's radius, as a fraction of the ring (a touch under
// half the ring so it sits as the core with room around it). 2026-09-26
// (Rosario): 0.55 → 0.62 (core read small) → 0.68 (push a bit more) → 0.72
// (final: "just a tiny bit bigger and we're done with the dot matrix").
// 0.72 · 75px ring = 54px core — still clear of the inner sonar ring
// (61.5px). FINAL value; the dot-matrix sizing is closed.
const HOLO_SPHERE = 0.72
// Beat clock: drives the sonar ping here in the canvas and the CSS label
// letter wave (kirsin-think-wave) — the confirmed thinking tempo. The ping
// TRAVELS over HOLO_SWEEP_S, then HOLDS dissolved at the rim for
// HOLO_RETRIGGER_S before the next ping is born — a re-trigger gap, so each
// pulse reads as a distinct sonar hit instead of back-to-back laps.
// 2026-09-28 (Rosario): 1.2s → 2s (label + circle "too fast") → 1.5s travel +
// 0.5s re-trigger → 1.2s travel + 0.5s re-trigger ("even 1.5s is too slow").
const HOLO_SWEEP_S = 1.2
// Re-trigger gap after the ping dissolves at the rim (seconds).
const HOLO_RETRIGGER_S = 0.5
// Outer DOT RING — 2026-09-28 (Rosario: "sub the outside lines into another
// dot matrix with the organic movement. Not fixed rotation like the central
// dot matrix, but a dot matrix that never enters the main circle and moves
// randomly on the outside"). Replaces the 72-tick dial + its traveling wave:
// a second, FLATTER dot ring floating just OUTSIDE the rim. Every dot keeps
// its own angular creep + radial wobble + twinkle (independent clocks), so
// the ring shimmers and creeps organically — it NEVER rotates as a rigid
// body and NEVER crosses into the main circle (worst case the innermost dot
// sits ~82px, 7px clear of the 75px rim). Always alive: a gentle drift at
// idle, brighter + faster while thinking.
const HOLO_OUTER_DOTS = 260
// Resting ring radius, as a fraction of the rim (R·1.12 ≈ 84px).
const HOLO_OUTER_R0 = 1.12
// Per-dot radial wobble, in px — kept small enough that even at peak
// inward the dot stays clear of the rim (1.12R − 2.6 − r ≈ 78px > 75px).
const HOLO_OUTER_WOB = 2.6
const HOLO_OUTER_TW = 0.4 // twinkle speed, Hz
const HOLO_OUTER_IDLE = 0.45 // idle multiplier on angular speed + alpha
const HOLO_OUTER_BUSY = 1.35 // busy multiplier
// Membrane: a handful of RANDOM organic bumps undulating the rim (Approach-B
// behavior — the line's shape deforms — but random instead of a fixed
// repeating sine). Each bump swells in, crawls slowly around the ring, and
// sinks out before a new one is born elsewhere. See holoBumpOffset below.
const HOLO_BUMP_MAX = 10

// Fibonacci-lattice points on the unit sphere — the dot-matrix globe. Computed
// once at module load (deterministic, no per-frame allocation).
function buildSpherePoints(count: number) {
  const pts: { x: number; y: number; z: number }[] = []
  const golden = Math.PI * (3 - Math.sqrt(5))

  for (let i = 0; i < count; i++) {
    const y = 1 - (i / (count - 1)) * 2 // 1 → -1
    const r = Math.sqrt(1 - y * y)
    const th = golden * i
    pts.push({ x: Math.cos(th) * r, y, z: Math.sin(th) * r })
  }

  return pts
}

const HOLO_SPHERE_POINTS = buildSpherePoints(300)

// Organic rim bumps — the membrane's "Approach B, but random" (2026-09-26,
// Rosario: "the line should behave like the B sample but randomly"). A small
// set of bumps lives on the rim at once; each is born at a random angle with
// a random size, eases UP (swells out of the line), crawls slowly around the
// ring (random speed + direction), eases DOWN (sinks back), then retires — a
// new one is born elsewhere. The shape itself undulates (like B); nothing
// thickens or brightens, so the line stays one constant stroke.
type RimBump = {
  age: number // seconds alive
  life: number // seconds total
  a0: number // birth angle (radians)
  drift: number // radians per second (sign = crawl direction)
  amp: number // peak bulge in px at scale 1 (× the live ampScale when drawn)
  width: number // angular half-width (radians)
}

let holoBumps: RimBump[] = []
let holoBumpsSeeded = false

function makeBump(startAge: number): RimBump {
  const life = 1.5 + Math.random() * 1.5 // 1.5–3.0s — quick enough to stay lively

  return {
    age: startAge,
    life,
    a0: Math.random() * Math.PI * 2,
    drift: (0.35 + Math.random() * 0.55) * (Math.random() < 0.5 ? -1 : 1),
    amp: 2.5 + Math.random() * 2.5,
    width: 0.25 + Math.random() * 0.3
  }
}

function holoStepBumps(dt: number) {
  if (!holoBumpsSeeded) {
    // Seed the crests with STAGGERED ages so they never swell, peak, or sink
    // together — the ring undulates continuously instead of pulsing in one
    // batch (the "only one spike" fix). Ages spread across one full life so
    // the ring is fully populated from the first frame.
    holoBumps = Array.from({ length: HOLO_BUMP_MAX }, (_, i) => makeBump(i * 2.25 / HOLO_BUMP_MAX))
    holoBumpsSeeded = true
  }

  for (const b of holoBumps) {b.age += dt}

  holoBumps = holoBumps.filter(b => b.age < b.life)

  // Replace a dead crest ONE per frame (born at age 0 so it swells in
  // smoothly). Because the deaths are staggered, the births are too — the rim
  // always has a few crests coming and going, never a flat ring.
  if (holoBumps.length < HOLO_BUMP_MAX) {holoBumps.push(makeBump(0))}
}

// The rim's radial offset at angle `th` this frame: the sum of every bump's
// Gaussian hump (so the line deforms smoothly, never jagged) × the bump's
// ease-in/out life envelope × the live amplitude scale. Returns px.
function holoBumpOffset(th: number, ampScale: number): number {
  let off = 0

  for (const b of holoBumps) {
    const lifeT = b.age / b.life
    const env = Math.sin(Math.PI * lifeT) // 0 → 1 → 0, ease in/out
    const a = b.a0 + b.drift * b.age
    let d = Math.abs(((th - a) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2))

    if (d > Math.PI) {d = Math.PI * 2 - d}
    const g = Math.exp(-(d * d) / (2 * b.width * b.width))
    off += b.amp * ampScale * g * env
  }

  return off
}

// Outer ring dots — each one a free agent (2026-09-28, Rosario: "moves
// randomly on the outside", "not fixed rotation like the central dot
// matrix"). Every dot carries its own angular base + creep speed (mixed
// directions), radial wobble (random phase + rate + amplitude within
// HOLO_OUTER_WOB), and twinkle (random phase + rate). Nothing is shared, so
// the ring never rotates as a body — it just shimmers and creeps.
type OuterDot = {
  a0: number // base angle (radians)
  speed: number // angular creep, rad/s (sign = direction)
  wobA: number // wobble amplitude, px (0..HOLO_OUTER_WOB)
  wobW: number // wobble angular frequency, rad/s
  wobP: number // wobble phase
  twA: number // twinkle amplitude (0..1, added to base alpha)
  twW: number // twinkle angular frequency, rad/s
  twP: number // twinkle phase
  r: number // dot radius, px
}

let holoOuterDots: OuterDot[] = []
let holoOuterSeeded = false

function makeOuterDot(): OuterDot {
  return {
    a0: Math.random() * Math.PI * 2,
    speed: (0.08 + Math.random() * 0.3) * (Math.random() < 0.5 ? -1 : 1),
    wobA: Math.random() * HOLO_OUTER_WOB,
    wobW: 0.5 + Math.random() * 1.5,
    wobP: Math.random() * Math.PI * 2,
    twA: 0.25 + Math.random() * 0.35,
    twW: HOLO_OUTER_TW * (Math.PI * 2) * (0.5 + Math.random()),
    twP: Math.random() * Math.PI * 2,
    r: 0.55 + Math.random() * 0.6
  }
}

// Seed once at module scale; afterwards the dots are immortal (they drift
// forever), so there is nothing to step per frame — their position is a pure
// function of the accumulated clock `t` (see drawHolo).
function holoEnsureOuterDots() {
  if (!holoOuterSeeded) {
    holoOuterDots = Array.from({ length: HOLO_OUTER_DOTS }, makeOuterDot)
    holoOuterSeeded = true
  }
}

// Parse a CSS color (rgb()/rgba()/#hex) into [r, g, b] so the canvas can build
// rgba() strings with variable alpha (the theme accent, not hard cyan).
function parseColor(c: string): [number, number, number] {
  const m = c.match(/rgba?\(([^)]+)\)/)

  if (m) {
    const [r, g, b] = m[1].split(',').map(s => parseInt(s, 10))

    return [r, g, b]
  }

  const hex = c.replace('#', '')
  const full = hex.length === 3 ? hex.split('').map(h => h + h).join('') : hex

  return [parseInt(full.slice(0, 2), 16), parseInt(full.slice(2, 4), 16), parseInt(full.slice(4, 6), 16)]
}

// Paint one hologram frame. `S` is the logical canvas size (HOLO_SIZE), `acc`
// the accumulated elapsed ms (the single clock), `busy` whether a turn runs
// (arms the tick-wave + intensifies the glow/membrane), `col` the [r,g,b]
// theme accent. Everything is additive ('lighter') so overlapping light blooms
// like the reference's holographic glow.
function drawHolo(
  ctx: CanvasRenderingContext2D,
  S: number,
  acc: number,
  dt: number,
  busy: boolean,
  col: [number, number, number]
) {
  const t = acc / 1000
  const cx = S / 2
  const cy = S / 2
  const R = HOLO_RING * (S / HOLO_SIZE)
  const rgba = (a: number) => `rgba(${col[0]}, ${col[1]}, ${col[2]}, ${a})`
  const TAU = Math.PI * 2

  ctx.clearRect(0, 0, S, S)
  ctx.save()
  ctx.globalCompositeOperation = 'lighter'

  // 1) Reactor glow — a soft radial bloom at the core (brighter while thinking).
  const glowR = R * 1.2
  const ga = busy ? 0.3 : 0.16
  const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, glowR)
  g.addColorStop(0, rgba(ga))
  g.addColorStop(0.6, rgba(ga * 0.35))
  g.addColorStop(1, rgba(0))
  ctx.fillStyle = g
  ctx.fillRect(0, 0, S, S)

  // 2) Dot-matrix sphere — the rotating "energy ball". One clock drives the
  //    Y-rotation (φ = k·t); front dots are bigger + brighter, back dots small
  //    + dim, which sells the 3D spin.
  const sphR = R * HOLO_SPHERE
  const rot = t * 0.6
  const cosR = Math.cos(rot)
  const sinR = Math.sin(rot)

  for (const p of HOLO_SPHERE_POINTS) {
    const xr = p.x * cosR + p.z * sinR
    const zr = -p.x * sinR + p.z * cosR
    const px = cx + xr * sphR
    const py = cy + p.y * sphR
    const d01 = (zr + 1) / 2 // 0 back → 1 front
    ctx.globalAlpha = 0.1 + d01 * 0.55
    ctx.fillStyle = rgba(1)
    ctx.beginPath()
    ctx.arc(px, py, 0.6 + d01 * 1.2, 0, TAU)
    ctx.fill()
  }

  ctx.globalAlpha = 1

  // 3) Concentric rings — thin sonar circles (outermost = the rim, dimmer ones
  //    step inward for depth). The middle ring PULSES while thinking as a sonar
  //    ping: born right on the dot-matrix edge (0.72R), it expands out to the
  //    rim over the 1.2s beat and HARD-RESETS back to the core — the return is
  //    NOT animated (no shrink-back). Between beats it HOLDS dissolved at the
  //    rim for the 0.5s re-trigger gap, so each ping reads as a distinct hit.
  //    2026-09-26 (Rosario): "start right on the dot matrix, go the outermost
  //    circle and then start again, don't animate the return to small state".
  //    2026-09-28: 1.2s travel + 0.5s re-trigger (1.7s cycle).
  ctx.lineWidth = 1
  const beatCycle = (t / (HOLO_SWEEP_S + HOLO_RETRIGGER_S)) % 1 // 0..1 per 1.7s
  const ping = busy ? Math.min(beatCycle / (HOLO_SWEEP_S / (HOLO_SWEEP_S + HOLO_RETRIGGER_S)), 1) : 0 // travel, then hold
  // Idle: the ring rests where it has always sat (0.82R, dim). Busy: born on
  // the core edge, fades as it travels so it dissolves at the rim like a
  // true sonar ping.
  const pingR = busy ? R * (HOLO_SPHERE + ping * (1 - HOLO_SPHERE)) : R * 0.82
  const pingA = busy ? 0.7 * (1 - ping) + 0.12 : 0.2

  const rings: [number, number][] = [
    [R, busy ? 0.5 : 0.32],
    [pingR, pingA],
    [R * 0.66, 0.12]
  ]

  for (const [rr, aa] of rings) {
    ctx.globalAlpha = aa
    ctx.strokeStyle = rgba(1)
    ctx.beginPath()
    ctx.arc(cx, cy, rr, 0, TAU)
    ctx.stroke()
  }

  // 4) Outer DOT RING — a second dot matrix floating OUTSIDE the rim
  //    (2026-09-28, Rosario: "sub the outside lines into another dot matrix
  //    with the organic movement... never enters the main circle and moves
  //    randomly on the outside"). Replaces the 72-tick dial + its wave. Each
  //    dot is a free agent (its own angular creep, radial wobble, twinkle —
  //    see makeOuterDot), so the ring shimmers and creeps organically: it
  //    NEVER rotates as a rigid body (unlike the central sphere) and NEVER
  //    crosses the rim (base 1.12R − max wobble keeps it ~7px clear).
  //    Always alive — gentle at idle, brighter + faster while thinking.
  holoEnsureOuterDots()
  const outerMul = busy ? HOLO_OUTER_BUSY : HOLO_OUTER_IDLE
  const outerBase = busy ? 0.16 : 0.1
  const outerR0 = R * HOLO_OUTER_R0

  for (const d of holoOuterDots) {
    const a = d.a0 + d.speed * outerMul * t
    const r = outerR0 + d.wobA * Math.sin(d.wobW * t + d.wobP)
    const tw = 0.5 * (1 + Math.sin(d.twW * t + d.twP)) // 0..1 twinkle
    const alpha = outerBase + d.twA * tw * outerMul
    ctx.globalAlpha = Math.min(1, alpha)
    ctx.fillStyle = rgba(1)
    ctx.beginPath()
    ctx.arc(cx + r * Math.cos(a), cy + r * Math.sin(a), d.r, 0, TAU)
    ctx.fill()
  }

  ctx.globalAlpha = 1

  // 6) Energy membrane — the rim's SHAPE undulates (Approach B: the line
  //    itself deforms, radius bulging out where a bump is). Instead of a fixed
  //    repeating sine, a handful of RANDOM organic bumps live on the rim at
  //    once: each swells in, crawls slowly around the ring, sinks out, then a
  //    new one is born elsewhere. The stroke stays constant (no thickening /
  //    brightening — the shape IS the effect). Subtle at rest, livelier while
  //    thinking. 2026-09-26 (Rosario): "behave like the B sample but randomly".
  const ampScale = (S / HOLO_SIZE) * (busy ? 1 : 0.55)

  if (dt > 0) {holoStepBumps(dt)}

  ctx.globalAlpha = busy ? 0.8 : 0.5
  ctx.strokeStyle = rgba(1)
  ctx.lineWidth = 1.8
  ctx.shadowColor = rgba(0.9)
  ctx.shadowBlur = busy ? 12 : 6
  ctx.beginPath()

  for (let i = 0; i <= 180; i++) {
    const th = (i / 180) * TAU
    const rr = R + holoBumpOffset(th, ampScale)
    const x = cx + rr * Math.cos(th)
    const y = cy + rr * Math.sin(th)

    if (i === 0) {ctx.moveTo(x, y)}
    else {ctx.lineTo(x, y)}
  }

  ctx.closePath()
  ctx.stroke()

  ctx.shadowBlur = 0
  ctx.globalAlpha = 1
  ctx.restore()
}

// Morph timing — the JS size tween and the CSS radius/opacity transitions
// (styles.css, "Orb morph (Phase 9)") share these so the native frame and
// its content stay in step. Collapse is fast (the reference's ~0.3s);
// expand is a touch slower so the content can fill in AFTER the frame
// regrows.
const MORPH_COLLAPSE_MS = 300
const MORPH_EXPAND_MS = 450

type KirsinMode = 'expanded' | 'pill' | 'orb'

/**
 * Kirsin mode's shell — the Kirsin Agent Window's chrome.
 *
 * Deliberately almost nothing, the same way {@link HudShell} is: it mounts the
 * SAME wired chat surface the workspace pane does, so the composer here IS the
 * app's composer (slash commands, `@` refs, attachments, queue, voice, model
 * pill) and the transcript is the app's transcript, rendered by the app's
 * renderer. The window carries no `profile=` override, so the gateway boot
 * adopts the PRIMARY (default) backend — this panel is a quick-chat surface
 * onto the default agent (it was originally pinned to a now-deleted
 * `kirsin` profile; it kept its name but now fronts the primary agent).
 *
 * The one thing that differs from the HUD is persistence: the HUD is an
 * auto-hiding band (Spotlight-shape, click-through, edge-detect, fade-on-idle),
 * where the Kirsin window is a PERSISTENT, always-on-top panel — a draggable
 * chat that remembers its position and never dismisses on blur. So this shell
 * carries none of the transient-band machinery; it is a solid frame around the
 * same chat surface.
 *
 * Three states, all MANUAL toggles (Rosario's choice — not Augmentor's
 * auto-morph-on-thinking):
 * - EXPANDED: the full chat surface.
 * - PILL (header chevron): a short status strip (live "Thinking…" line + the
 *   active model) while the chat stays mounted underneath.
 * - ORB (header orb button): the window morphs to a 196px glowing circle
 *   (a 150px painted disc with a soft halo margin) with the live turn status
 *   inside ("Thinking…" letter-wave + pulsing dot), exactly the Augmentor
 *   reference's collapse. Click the orb to re-grow the panel; drag it to move
 *   the window.
 *
 * The header is the drag handle; the chevron morphs to the pill; the orb
 * button morphs to the circle; the pin unpins; the × closes.
 */
export function KirsinShell() {
  // Force the HOST layers transparent. index.html's pre-paint script writes an
  // opaque themed background onto <html> as an INLINE style (the anti-white-
  // flash trick), and an inline style beats any stylesheet rule — so without
  // this the window is a solid slab and the panel below is just glass over a
  // white wall. A `!important` style tag is what the pet overlay, quick entry,
  // and the HUD already do; Kirsin is not a bespoke root, so it needs the same.
  useEffect(() => {
    const style = document.createElement('style')
    style.textContent = 'html,body,#root{background:transparent !important;}'
    document.head.appendChild(style)
    document.documentElement.setAttribute('data-kirsin-window', '')

    return () => {
      style.remove()
      document.documentElement.removeAttribute('data-kirsin-window')
    }
  }, [])

  const [mode, setMode] = useState<KirsinMode>('expanded')
  // The header + chat body are display-swapped only AFTER the morph tween
  // settles (the onDone below), so during the 300ms shrink they stay laid out
  // and just fade out — the content never pops away mid-shape.
  const [surfaceHidden, setSurfaceHidden] = useState(false)
  // The CSS radius/opacity transitions read this, so the frame rounds in step
  // with whichever tween is running (300ms collapse / 450ms expand).
  const [morphMs, setMorphMs] = useState(MORPH_COLLAPSE_MS)
  // Live turn state + the active model for this window's own backend — the
  // pill's and the orb's lines. Both are nanostores on the session store, so
  // the status stays truthful even while the chat surface is hidden.
  const busy = useStore($busy)
  const model = useStore($currentModel)
  const { t } = useI18n()
  const morphRaf = useRef<number | null>(null)

  const stopMorph = useCallback(() => {
    if (morphRaf.current !== null) {
      cancelAnimationFrame(morphRaf.current)
      morphRaf.current = null
    }
  }, [])

  // Tween the NATIVE window to (width, height), top-left anchored, so the
  // CSS border-radius transition (which runs over the same duration with the
  // same ease) makes the frame round into a circle as it shrinks. Each frame
  // is a fire-and-forget setBounds (same channel the drag uses, 60/s) — the
  // main handler clamps to the 196/96 floors, which every intermediate size
  // already satisfies.
  const morphTo = useCallback(
    (width: number, height: number, durationMs: number, onDone?: () => void) => {
      stopMorph()
      const set = window.hermesDesktop?.kirsin?.setBounds

      if (!set) {
        onDone?.()

        return
      }

      const startX = window.screenX
      const startY = window.screenY
      const startW = window.outerWidth
      const startH = window.outerHeight
      // Top-CENTER anchor (the reference's morph): the frame shrinks toward
      // the panel's horizontal center, so the orb parks centered under the
      // panel it came from. Widths that don't change (chat↔pill) keep x.
      const targetX = startX + (startW - width) / 2
      const t0 = performance.now()

      const step = (now: number) => {
        const p = durationMs <= 0 ? 1 : Math.min(1, (now - t0) / durationMs)
        const eased = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2
        set({
          x: Math.round(startX + (targetX - startX) * eased),
          y: startY,
          width: Math.round(startW + (width - startW) * eased),
          height: Math.round(startH + (height - startH) * eased)
        })

        if (p < 1) {
          morphRaf.current = requestAnimationFrame(step)
        } else {
          morphRaf.current = null
          onDone?.()
        }
      }

      morphRaf.current = requestAnimationFrame(step)
    },
    [stopMorph]
  )

  // A morph in flight at unmount must not keep driving setBounds.
  useEffect(() => stopMorph, [stopMorph])

  // The hologram is rAF-driven on a <canvas>, NOT a CSS animation. The user's
  // OS has "show animations" turned OFF, so Chromium reports
  // prefers-reduced-motion: reduce and the app's blanket rule (styles.css:
  // `animation-duration: 0.01ms !important` on `*`) freezes every CSS
  // animation — but canvas drawing (and rAF) is unaffected, so the hologram
  // keeps living even with that rule active (2026-09-25: every CSS-animation
  // version of the rim read as "still not animated" for exactly this reason).
  // The loop runs whenever the orb is VISIBLE (armed): at rest it paints the
  // calm hologram (rotating sphere + gentle membrane), and while a turn runs
  // (busy) it arms the tick-wave + intensifies the glow/membrane. One clock
  // (`acc`, accumulated ms) drives every layer — a single coherent phase.
  const holoCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const holoCtxRef = useRef<CanvasRenderingContext2D | null>(null)
  const holoColorRef = useRef<[number, number, number]>([64, 224, 208])
  // The accent must re-resolve when the theme repaints (the app rewrites the
  // computed tokens on <html>); useThemeEpoch ticks AFTER that paint, so the
  // probe reads a fresh computed color (see src/hooks/use-theme-epoch.ts).
  const themeEpoch = useThemeEpoch()
  // eslint-disable-next-line no-restricted-syntax -- canvas 2D context + one-time DOM color probe (DOM-instance refs, not a mirrored atom)
  useEffect(() => {
    const el = holoCanvasRef.current

    if (!el) {return}
    holoCtxRef.current = el.getContext('2d')
    // --kirsin-accent is a VAR CHAIN (var(--ui-accent)), so getComputedStyle's
    // getPropertyValue returns it UNRESOLVED. Resolve it to a concrete rgb() by
    // reading the computed `color` of a throwaway element that USES the var
    // (the browser resolves the whole chain), then parse it.
    const probe = document.createElement('span')

    probe.style.color = 'var(--kirsin-accent)'
    ;(el.parentElement ?? document.body).appendChild(probe)
    const accent = getComputedStyle(probe).color
    probe.remove()
    holoColorRef.current = parseColor(accent)
  }, [themeEpoch])
  useEffect(() => {
    const el = holoCanvasRef.current
    const ctx = holoCtxRef.current

    if (!el || !ctx) {return}
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    el.width = Math.round(HOLO_SIZE * dpr)
    el.height = Math.round(HOLO_SIZE * dpr)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  }, [mode])
  useEffect(() => {
    const ctx = holoCtxRef.current

    if (!ctx) {return}
    const armed = mode === 'orb'

    if (!armed) {
      ctx.clearRect(0, 0, HOLO_SIZE, HOLO_SIZE)

      return
    }

    let raf = 0
    let last = performance.now()
    // rAF pauses when the window is hidden, so wall-clock would make the
    // hologram "jump" on resume; accumulate elapsed ms per frame instead.
    // `acc` is MILLISECONDS — drawHolo's periods are in SECONDS, so it
    // divides by 1000 (a missing /1000 made the old rotation strobe ~1000x).
    let acc = 0

    const tick = (now: number) => {
      const dt = now - last
      acc += dt
      last = now
      drawHolo(ctx, HOLO_SIZE, acc, dt / 1000, busy, holoColorRef.current)
      raf = requestAnimationFrame(tick)
    }

    raf = requestAnimationFrame(tick)

    return () => {
      cancelAnimationFrame(raf)
      ctx.clearRect(0, 0, HOLO_SIZE, HOLO_SIZE)
    }
  }, [mode, busy])

  const collapseToPill = useCallback(() => {
    setMorphMs(MORPH_COLLAPSE_MS)
    setMode('pill')
    morphTo(KIRSIN_W, KIRSIN_H_COMPACT, MORPH_COLLAPSE_MS, () => setSurfaceHidden(true))
  }, [morphTo])

  const collapseToOrb = useCallback(() => {
    setMorphMs(MORPH_COLLAPSE_MS)
    setMode('orb')
    morphTo(KIRSIN_ORB, KIRSIN_ORB, MORPH_COLLAPSE_MS, () => setSurfaceHidden(true))
  }, [morphTo])

  const expand = useCallback(() => {
    setMorphMs(MORPH_EXPAND_MS)
    setMode('expanded')
    // Re-layout the surface IMMEDIATELY: it fades in via CSS (240ms opacity
    // with a 220ms delay) so it fills back in after the frame has regrown.
    setSurfaceHidden(false)
    morphTo(KIRSIN_W, KIRSIN_H_EXPANDED, MORPH_EXPAND_MS)
  }, [morphTo])

  const toggleExpanded = useCallback(() => {
    if (mode === 'pill') {
      expand()
    } else {
      collapseToPill()
    }
  }, [mode, expand, collapseToPill])

  const close = useCallback(() => window.hermesDesktop?.kirsin?.close(), [])

  // The header is the drag handle (see kirsin-drag.ts). The header buttons
  // stopPropagation on pointerdown, so a press on one never starts a drag.
  const { dragging, onPointerDown: onHeaderPointerDown } = useKirsinHeaderDrag()

  // The orb's Stop button drives the SAME cancel the composer's stop uses (the
  // controller's `onCancel` → cancelRun). It's reached through the stable wiring
  // actions bag, so this shell needs no gateway of its own. The orb's Expand
  // button is the explicit "reopen the chat" control — the disc itself is now
  // drag-only (no more "click the dot to expand").
  const wiring = useContext(ContribWiringContext)

  const onOrbStop = useCallback(() => {
    void wiring?.actions.onCancel?.()
  }, [wiring])

  // A dictated transcript from the listen overlay (Ctrl+Shift+L, Dictate
  // mode) lands here as plain text and rides the SAME onSubmit path a typed
  // message uses — this window's own composer/session, nothing bespoke.
  // Only ever fires when the user explicitly picked Dictate for that
  // capture (see electron/listen-overlay.ts's 'final' handler).
  useEffect(() => {
    const off = window.hermesDesktop?.kirsin?.onDictate(text => {
      void wiring?.actions.onSubmit?.(text)
    })

    return off
  }, [wiring])

  // Mic button: opens/advances the SAME listen overlay Ctrl+Shift+L drives —
  // one control surface, reachable either way. Toggling from here has no
  // special mode of its own; the overlay's own Subtitle/Dictate picker
  // decides what happens to the transcript.
  const onToggleListen = useCallback(() => {
    window.hermesDesktop?.listenOverlay?.toggle()
  }, [])

  const isExpanded = mode === 'expanded'
  const collapsed = !isExpanded

  return (
    <div
      className="relative flex h-screen w-screen flex-col overflow-hidden"
      data-kirsin-busy={busy ? 'true' : 'false'}
      data-kirsin-collapsed={collapsed ? '' : undefined}
      data-kirsin-orb={mode === 'orb' ? '' : undefined}
      data-kirsin-orb-thinking={busy ? 'true' : 'false'}
      data-kirsin-shell
      data-kirsin-surface-hidden={surfaceHidden ? '' : undefined}
      style={{ '--kirsin-morph-ms': `${morphMs}ms` } as CSSProperties}
    >
      {/* The header strip — the drag handle (Phase 4 wires the pointer
          handlers to data-kirsin-header). Left: logo chip + wordmark. Right:
          the session switcher, the orb button (morph to circle), the collapse
          chevron (morph to pill), the pin (unpins/closes), and close. */}
      <header
        className="flex shrink-0 select-none items-center"
        data-kirsin-dragging={dragging ? '' : undefined}
        data-kirsin-header
        onPointerDown={onHeaderPointerDown}
        style={{ height: 38 }}
      >
        <span aria-hidden data-kirsin-logo>
          {/* 4-point sparkle on the teal chip — the Kirsin brand mark (teal,
              not pink; the reference's accent chip, re-skinned to the theme). */}
          <svg fill="currentColor" role="img" viewBox="0 0 24 24">
            <path d="M12 1.5l2.6 7.9 7.9 2.6-7.9 2.6L12 22.5l-2.6-7.9L1.5 12l7.9-2.6z" />
          </svg>
        </span>
        <span data-kirsin-wordmark>Kirsin</span>
        <span className="flex-1" />
        <Tip label={t.kirsin.listenToggle} side="bottom">
          <button
            aria-label={t.kirsin.listenToggle}
            data-kirsin-ctl
            data-kirsin-listen-ctl
            onClick={onToggleListen}
            onPointerDown={event => event.stopPropagation()}
            type="button"
          >
            <svg fill="none" role="img" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} viewBox="0 0 24 24">
              <rect height="11" rx="3.5" width="7" x="8.5" y="2" />
              <path d="M5 11a7 7 0 0 0 14 0" />
              <line x1="12" x2="12" y1="18" y2="22" />
              <line x1="8" x2="16" y1="22" y2="22" />
            </svg>
          </button>
        </Tip>
        <KirsinSessionSwitcher />
        <Tip label={t.kirsin.orbCollapse} side="bottom">
          <button
            aria-label={t.kirsin.orbCollapse}
            data-kirsin-ctl
            data-kirsin-orb-ctl
            onClick={collapseToOrb}
            onPointerDown={event => event.stopPropagation()}
            type="button"
          >
            <svg fill="none" role="img" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="8" />
              <circle cx="12" cy="12" fill="currentColor" r="2.5" stroke="none" />
            </svg>
          </button>
        </Tip>
        <Tip label={isExpanded ? t.kirsin.collapse : t.kirsin.expand} side="bottom">
          <button
            aria-expanded={isExpanded}
            aria-label={isExpanded ? t.kirsin.collapse : t.kirsin.expand}
            data-kirsin-collapse
            data-kirsin-ctl
            onClick={toggleExpanded}
            onPointerDown={event => event.stopPropagation()}
            type="button"
          >
            <svg
              fill="none"
              role="img"
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              style={{ transform: isExpanded ? 'rotate(0deg)' : 'rotate(180deg)', transition: 'transform 140ms ease' }}
              viewBox="0 0 24 24"
            >
              <path d="M6 9l6 6 6-6" />
            </svg>
          </button>
        </Tip>
        <Tip label={t.kirsin.pinClose} side="bottom">
          <button
            aria-label="Unpin"
            data-kirsin-ctl
            data-kirsin-pin
            data-kirsin-pinned="true"
            onClick={close}
            onPointerDown={event => event.stopPropagation()}
            type="button"
          >
            <svg
              fill="none"
              role="img"
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              viewBox="0 0 24 24"
            >
              <path d="M9 4h6l-1 5 3.5 3.5V15H6.5v-2.5L10 9z" />
              <line x1="12" x2="12" y1="15" y2="20" />
            </svg>
          </button>
        </Tip>
        <Tip label={t.kirsin.close} side="bottom">
          <button
            aria-label="Close"
            data-kirsin-close
            data-kirsin-ctl
            onClick={close}
            onPointerDown={event => event.stopPropagation()}
            type="button"
          >
            <svg
              fill="none"
              role="img"
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              viewBox="0 0 24 24"
            >
              <path d="M6 6l12 12M18 6L6 18" />
            </svg>
          </button>
        </Tip>
      </header>

      {/* Compact strip — visible only in the PILL state (the orb state hides
          it; see styles.css). A single row: the live turn status (pulsing
          teal dot + "Thinking…" while busy) and the active model, truncated
          to fit the pill. */}
      <div data-kirsin-compact>
        <span data-kirsin-status data-kirsin-thinking={busy ? 'true' : 'false'}>
          {busy ? t.kirsin.thinking : t.kirsin.ready}
        </span>
        {model ? (
          <OverflowTip label={model} side="bottom">
            <span data-kirsin-model>{model}</span>
          </OverflowTip>
        ) : null}
      </div>

      {/* The real chat surface — always MOUNTED (the turn keeps running and the
          composer draft survive while collapsed); hidden via CSS after the
          morph settles. This is the app's transcript + composer, bound to the
          default backend this window booted against. */}
      <div className="min-h-0 flex-1 overflow-hidden" data-chat-surface data-kirsin-body>
        <WiredPane part="chatRoutes" />
      </div>

      {/* ORB — the third state (the Augmentor reference): a glowing circle
          with the live turn status inside (spinner ring + "Thinking…" +
          pulsing dot). The disc is now DRAG-ONLY — the two explicit controls
          below do the real work: Stop (only while a turn runs) and Expand
          (re-opens the chat). No more "click the dot to expand". */}
      <div
        data-kirsin-dragging={dragging && mode === 'orb' ? '' : undefined}
        data-kirsin-orb
        onPointerDown={onHeaderPointerDown}
      >
        <span data-kirsin-orb-label>
          {(busy ? t.kirsin.thinking : t.kirsin.ready).split('').map((ch, i) => (
            <span key={i} style={{ animationDelay: `${i * 50}ms` }}>
              {ch === ' ' ? '\u00A0' : ch}
            </span>
          ))}
        </span>
        {/* The Jarvis-style HOLOGRAM (Approach C, 2026-09-26) — a single
            rAF-driven <canvas>, NOT an SVG wave (the two old wave layers are
            gone). It paints the rotating dot-matrix sphere core, the
            concentric sonar rings, an organic outer DOT RING (each dot
            creeps, wobbles and twinkles on its own clock — never a rigid
            spin, never crossing the rim), over an energy-membrane edge whose
            SHAPE undulates with random organic bumps (Approach B, but random
            — each bump swells in, crawls around the rim, and sinks out), all
            in the theme accent with additive bloom (drawHolo, kirsin-shell.tsx).
            The canvas is window-sized (196) so its center is the window center
            and the 75px rim rides the 150px disc's edge; see the "hologram"
            block in styles.css for the positioning + z-order. Canvas is
            unaffected by the blanket reduced-motion rule (CSS-only), so the
            hologram stays alive even with OS "show animations" off. The comet
            orbit + label letter-wave coexist (see below). */}
        <canvas
          aria-hidden
          data-kirsin-orb-holo
          height={HOLO_SIZE}
          ref={holoCanvasRef}
          width={HOLO_SIZE}
        />
        <div data-kirsin-orb-controls>
          <button
            aria-label={t.kirsin.orbStop}
            data-kirsin-ctl
            data-kirsin-orb-stop
            data-kirsin-orb-stop-visible={busy ? '' : undefined}
            onClick={onOrbStop}
            onPointerDown={event => event.stopPropagation()}
            type="button"
          >
            <svg fill="currentColor" role="img" viewBox="0 0 24 24">
              <rect height="12" rx="2" width="12" x="6" y="6" />
            </svg>
          </button>
          <button
            aria-label={t.kirsin.orbExpand}
            data-kirsin-ctl
            data-kirsin-orb-expand
            onClick={expand}
            onPointerDown={event => event.stopPropagation()}
            type="button"
          >
            <svg
              fill="none"
              role="img"
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              viewBox="0 0 24 24"
            >
              <path d="M9 3H3v6M15 3h6v6M9 21H3v-6M15 21h6v-6" />
            </svg>
          </button>
          <button
            aria-label={t.kirsin.listenToggle}
            data-kirsin-ctl
            data-kirsin-orb-listen
            onClick={onToggleListen}
            onPointerDown={event => event.stopPropagation()}
            type="button"
          >
            <svg fill="none" role="img" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} viewBox="0 0 24 24">
              <rect height="11" rx="3.5" width="7" x="8.5" y="2" />
              <path d="M5 11a7 7 0 0 0 14 0" />
              <line x1="12" x2="12" y1="18" y2="22" />
              <line x1="8" x2="16" y1="22" y2="22" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  )
}
