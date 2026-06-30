import { darken, luminance, mixRgb, rgba } from './color'
import {
  LIT_BAND_ALPHA,
  NODE_SHAPE,
  ORB_DARKEN,
  RING_INNER,
  RING_PARAMS,
  TILT,
  WHITE,
  WHITEISH_SHEEN
} from './constants'
import { clamp, fitScale, nodeRadius, recencyInk, shapePath } from './geometry'
import { countLabel, ellipsize, metaBadges, nodeFooter, wrapText } from './text'
import type {
  FadeBuckets,
  MemoryCard,
  Palette,
  Rect,
  Rgb,
  Ring,
  RingLabelRect,
  SimLink,
  SimNode,
  Viewport
} from './types'

export interface Scene {
  adjacency: Map<string, Set<string>>
  byId: Map<string, SimNode>
  ctx: CanvasRenderingContext2D
  dpr: number
  fades: FadeBuckets
  focusId: null | string
  hoverId: null | string
  hoverLink: null | string
  hoverRing: null | number
  links: SimLink[]
  memById: Map<string, MemoryCard>
  nodes: SimNode[]
  palette: Palette
  // Time scrubber: only paint nodes/links whose recency has been reached. 1 =
  // everything (the default, idle state); lower values "build up" the map.
  reveal: number
  rings: Ring[]
  selectedRing: null | number
  size: { h: number; w: number }
  // Scrub jumps: snap every ease to its target this frame (no birth/fade replay).
  snapMotion?: boolean
  vp: Viewport
}

export interface DrawResult {
  animating: boolean
  ringLabelRects: RingLabelRect[]
}

// Smoothstep — eases the birth animations (position grow-out) in and out.
const ease = (t: number): number => {
  const u = t < 0 ? 0 : t > 1 ? 1 : t

  return u * u * (3 - 2 * u)
}

// EVE-style warp arrival for node births: the star streaks outward fast, then
// decelerates hard (exponential ease-out) and drops onto its ring — like a ship
// dropping out of warp. WARP_FROM is how deep toward the core it launches from.
const WARP_FROM = 0.32

const warpIn = (t: number): number => {
  const u = t < 0 ? 0 : t > 1 ? 1 : t

  return u >= 1 ? 1 : 1 - 2 ** (-9 * u)
}

// Layered birth speeds for the scrubber's parallax: rings expand slowly and
// grandly in the background, stars pop in quicker up front — both well below the
// default hover/focus speeds so the build-up reads as a cinematic settle.
const RING_BIRTH = { down: 0.055, up: 0.032 }
const NODE_BIRTH = { down: 0.11, up: 0.075 }

// Glyph pool for the empty-core scramble: Matrix-style half-width katakana plus
// a few digits/symbols for the "digital rain / decoding" look.
const SCRAMBLE_CHARS =
  'ﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾜﾝｦｱｳｴｵｶｷｹｺｻｼｽｾﾀﾁﾂﾃﾅﾆﾇﾈ0123456789:.=*+<>Ξ╳'

// Sphere-sprite atlas: a lit orb is the same picture at every size, so we render
// each distinct (ink, sheen, darken) appearance ONCE into an offscreen sprite and
// blit it (scaled) per node — instead of allocating a fresh radial gradient for
// every star on every frame. Keyed by appearance, not size; drawImage scales it.
// Reference radius the sprite is rendered at — larger than the usual billboarded
// screen-space orb, so sprites scale down in normal use and stay crisp.
const SPRITE_R = 96

const spriteCache = new Map<string, HTMLCanvasElement>()

// Build (or fetch) the orb sprite for one appearance: an offset radial gradient
// from a hot core → darkened body → translucent rim, clipped to the disk, so a
// flat circle reads with volume. `strength` is how white the core is; `bodyDarken`
// darkens the body (0 for active/hover nodes so they pop full bright). Near-white
// inks skip the darken and force a near-full sheen so the white core still reads.
function sphereSprite(ink: Rgb, strength: number, bodyDarken: number): HTMLCanvasElement {
  const key = `${ink.r},${ink.g},${ink.b}|${strength}|${bodyDarken}`
  const cached = spriteCache.get(key)

  if (cached) {
    return cached
  }

  const R = SPRITE_R
  // Margin for the gradient's rim (extends to 1.15·R) so it isn't clipped.
  const pad = Math.ceil(R * 0.15) + 1
  const size = (R + pad) * 2
  const c = R + pad
  const cv = document.createElement('canvas')
  cv.width = size
  cv.height = size
  const g2 = cv.getContext('2d')

  if (!g2) {
    return cv
  }

  const mx = Math.max(ink.r, ink.g, ink.b)
  const mn = Math.min(ink.r, ink.g, ink.b)
  const sat = mx ? (mx - mn) / mx : 0
  const whiteness = clamp((luminance(ink.r, ink.g, ink.b) - 0.7) / 0.3, 0, 1) * (1 - sat)
  const eff = strength + (WHITEISH_SHEEN - strength) * whiteness
  const hi = mixRgb(ink, WHITE, 0.7 * eff)
  const body = darken(ink, bodyDarken * (1 - whiteness))
  const grad = g2.createRadialGradient(c - R * 0.35, c - R * 0.4, R * 0.05, c, c, R * 1.15)
  grad.addColorStop(0, rgba(hi, 1))
  grad.addColorStop(0.5, rgba(body, 1))
  grad.addColorStop(1, rgba(body, 0.85))
  g2.fillStyle = grad
  g2.beginPath()
  g2.arc(c, c, R, 0, Math.PI * 2)
  g2.fill()
  spriteCache.set(key, cv)

  return cv
}

// Paint a lit orb of radius `r` centered at (x, y) by blitting its cached sprite.
// Honors the caller's globalAlpha (drawImage multiplies it), matching the old
// gradient fill. No path needed — the sprite already carries the disk + AA rim.
function sphereFill(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  r: number,
  ink: Rgb,
  strength: number,
  bodyDarken: number
): void {
  const sprite = sphereSprite(ink, strength, bodyDarken)
  const scale = r / SPRITE_R
  const drawSize = sprite.width * scale
  ctx.drawImage(sprite, x - drawSize / 2, y - drawSize / 2, drawSize, drawSize)
}

const rectsOverlap = (a: Rect, b: Rect) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y

// Paint a full frame of the star map. Pure given its inputs (draws to the
// canvas + advances the fade buckets); returns whether it's still animating and
// the ring-label hit rects for pointer picking.
export function drawScene(scene: Scene): DrawResult {
  const {
    adjacency,
    byId,
    ctx,
    dpr,
    fades,
    focusId,
    hoverId,
    hoverLink,
    hoverRing,
    links,
    memById,
    nodes,
    palette,
    reveal,
    rings,
    selectedRing,
    size,
    snapMotion = false,
    vp
  } = scene

  // Small epsilon so a node exactly at the playhead counts as revealed.
  const seen = (rec: number) => rec <= reveal + 1e-3
  // Recency for styling is RELATIVE to the newest revealed node — the current
  // "present" — not the bare playhead. So a lone frontier node still reads as
  // fresh (bright/full size) even with empty space between it and the scrubber.
  // At reveal = 1 the frontier is the newest node, collapsing back to raw recency.
  let frontier = 0

  for (const fn of nodes) {
    if (fn.rec <= reveal + 1e-3 && fn.rec > frontier) {
      frontier = fn.rec
    }
  }

  const erec = (rec: number) => (frontier > 0 ? clamp(rec / frontier, 0, 1) : 1)
  const { h, w } = size
  const { bandInk, base, bg, c, chipBg, darkTheme, inkInv, memoryInk, skillInk } = palette
  const { bandAlpha, lightSize, ringAlpha, sheen } = RING_PARAMS[darkTheme ? 'dark' : 'light']

  let animating = false
  const ringLabelRects: RingLabelRect[] = []

  // Eased opacity per element: snaps up when newly highlighted, eases otherwise.
  // `rates` overrides the default in/out lerp speed (the slow births pass their
  // own gentler pair so the build-up reads as a graceful settle, not a flash).
  const fadeAlpha = (
    bucket: Map<string, number>,
    key: string,
    target: number,
    snapUp = false,
    rates?: { down: number; up: number }
  ) => {
    const targetAlpha = clamp(target, 0, 1)
    const prev = bucket.get(key)

    // Scrub: jump straight to the target so a fast drag doesn't replay easing.
    if (snapMotion) {
      bucket.set(key, targetAlpha)

      return targetAlpha
    }

    if (prev == null || (snapUp && targetAlpha > prev)) {
      bucket.set(key, targetAlpha)

      return targetAlpha
    }

    const up = rates?.up ?? 0.22
    const down = rates?.down ?? 0.32
    const rate = targetAlpha > prev ? up : down
    const next = prev + (targetAlpha - prev) * rate

    if (Math.abs(next - targetAlpha) < 0.01) {
      bucket.set(key, targetAlpha)

      return targetAlpha
    }

    animating = true
    bucket.set(key, next)

    return next
  }

  const shade = (a: number) => `rgba(${base.r},${base.g},${base.b},${a})`
  const projX = (wx: number) => wx * vp.k + vp.x
  const projY = (wy: number) => wy * vp.k * TILT + vp.y
  // Baseline node scale: the rested fit, held stable while the playback camera
  // dives into the core — so t≈0 nodes don't balloon (see fitScale).
  const nodeK = fitScale(w, h, rings)

  // Two composable layers: node highlight (selected ?? hovered) in full ink, and
  // a selection-only ring/date filter that only shifts alpha.
  const focusSet = focusId ? (adjacency.get(focusId) ?? new Set<string>()) : null
  const ringIdx = selectedRing
  const ring = ringIdx != null ? (rings[ringIdx] ?? null) : null
  // A selected ring owns the band it caps: previous ring → this ring. Ring 0 is
  // visual-only/unlabeled, so the first selectable date naturally owns shell 0→1.
  const ringLo = ring && ringIdx != null ? (rings[ringIdx - 1]?.ratio ?? 0) - 1e-3 : 0
  const ringHi = ring ? ring.ratio + 1e-3 : 1

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, w, h)

  ctx.globalAlpha = 1

  // Tilted world transform for the disk structure.
  ctx.setTransform(vp.k * dpr, 0, 0, vp.k * TILT * dpr, vp.x * dpr, vp.y * dpr)

  // The "lit" date = hovered (preview) or selected (locked) — drives the band
  // flatten only; the ring outline reacts to selection.
  const litRingIdx = hoverRing ?? ringIdx

  // A ring is "laid" one band AHEAD of the playhead — it appears the moment the
  // scrubber enters the band beneath it (its inner neighbor's date), so the date
  // gridline that caps a region is always drawn before any node in that region.
  // Ring 0 is just the visual core; the first real shell still needs non-zero
  // playback progress so replay starts empty instead of showing it pre-laid.
  const ringSeen = (i: number) => {
    const threshold = rings[i - 1]?.ratio ?? 0

    return i === 0 || (threshold <= 0 ? reveal > 1e-3 : reveal + 1e-3 >= threshold)
  }

  // Per-ring "grow out" progress (advanced once per frame, reused by bands /
  // outlines / labels): a revealed ring eases its radius from its inner neighbor
  // outward to its resting radius, so it expands into place instead of popping.
  const ringAppear = rings.map((rg, i) =>
    ease(fadeAlpha(fades.appear, `ring:${i}`, ringSeen(i) ? 1 : 0, false, RING_BIRTH))
  )

  // Direction-based origin (the sign of the reveal): a ring growing IN expands
  // outward from its inner neighbour — never from the dead centre — while a ring
  // fading OUT collapses all the way to the core. ringSeen is the direction tell:
  // true = revealing/at rest, false = receding.
  const ringDrawR = rings.map((rg, i) => {
    const startR = ringSeen(i) ? (rings[i - 1]?.r ?? rg.r) : RING_INNER

    return startR + (rg.r - startR) * (ringAppear[i] ?? 1)
  })

  // Opacity envelope that stays near-full through most of the grow/shrink and
  // only fades in the final stretch — so the radius TRAVEL is visible (the ring
  // shrinks back into place) instead of just dimming out where it stands.
  const ringVis = ringAppear.map(a => clamp(a / 0.55, 0, 1))

  // Inter-ring bands: a theme-tinted wash sliver at the outer edge; the lit
  // date's band flattens to an even wash.
  if (bandAlpha > 0 || litRingIdx != null) {
    for (let i = 0; i < rings.length - 1; i += 1) {
      const lit = litRingIdx != null && i + 1 === litRingIdx

      if (!lit && bandAlpha <= 0) {
        continue
      }

      // The band tracks its outer ring's grow-in.
      if ((ringAppear[i + 1] ?? 1) <= 0.01) {
        continue
      }

      const inner = ringDrawR[i] ?? 0
      const outer = ringDrawR[i + 1] ?? 0

      if (lit) {
        ctx.fillStyle = rgba(bandInk, LIT_BAND_ALPHA)
      } else {
        const grad = ctx.createRadialGradient(0, 0, inner, 0, 0, outer)
        grad.addColorStop(0, rgba(bandInk, 0))
        grad.addColorStop(clamp(1 - lightSize, 0.01, 0.99), rgba(bandInk, 0))
        grad.addColorStop(1, rgba(bandInk, bandAlpha))
        ctx.fillStyle = grad
      }

      ctx.beginPath()
      ctx.arc(0, 0, outer, 0, Math.PI * 2)
      ctx.arc(0, 0, inner, 0, Math.PI * 2, true)
      ctx.fill()
    }
  }

  // Ring outline: brightens only on selection — the selected ring + its inner
  // neighbor (the two bounding the lit band).
  ctx.lineWidth = c.ringWidth / vp.k
  ctx.setLineDash(c.ringDashed ? [c.ringDash / vp.k, c.ringDash / vp.k] : [])
  rings.forEach((rg, i) => {
    const emphasized = ringIdx != null && (i === ringIdx || i === ringIdx - 1)
    // Reveal in/out rides the smooth (slow) ringAppear envelope so a ring fades
    // out as gracefully as it grew in; the alpha bucket only carries the snappy
    // selection emphasis.
    const emphasisAlpha = emphasized ? clamp(LIT_BAND_ALPHA * 2, 0, 1) : ringAlpha
    const ringAlphaNow = fadeAlpha(fades.rings, String(i), emphasisAlpha, emphasized) * (ringVis[i] ?? 1)

    if (ringAlphaNow < 0.004) {
      return
    }

    ctx.strokeStyle = shade(ringAlphaNow)
    ctx.beginPath()
    ctx.arc(0, 0, ringDrawR[i] ?? rg.r, 0, Math.PI * 2)
    ctx.stroke()
  })
  ctx.setLineDash([])

  // Screen space for the jump routes and glyphs (crisp, easy to trim). The empty
  // core's animated scramble is NOT painted here — it's the only perpetually
  // moving layer, so it's drawn live each frame by drawScramble() on top of the
  // (cached) static scene. Everything else in this function is static until an
  // input changes, which is why `animating` now reflects only in-flight fades.
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

  // Jump routes — a focused node's links stop at its selection ring.
  const focusNode = focusId ? (byId.get(focusId) ?? null) : null
  const focusRingR = focusNode ? nodeRadius(focusNode) * nodeK + 4 : 0

  for (const link of links) {
    const s = typeof link.source === 'object' ? link.source : byId.get(String(link.source))
    const t = typeof link.target === 'object' ? link.target : byId.get(String(link.target))

    if (!s || !t) {
      continue
    }

    // A jump route only exists once both of its endpoints have ignited.
    const revealed = seen(s.rec) && seen(t.rec)

    const lit =
      revealed &&
      !!focusId &&
      (s.id === focusId || t.id === focusId || (!!focusSet && focusSet.has(s.id) && focusSet.has(t.id)))

    let x1 = projX(s.x)
    let y1 = projY(s.y)
    let x2 = projX(t.x)
    let y2 = projY(t.y)

    if (s.id === focusId) {
      const d = Math.hypot(x2 - x1, y2 - y1) || 1
      x1 += ((x2 - x1) / d) * focusRingR
      y1 += ((y2 - y1) / d) * focusRingR
    }

    if (t.id === focusId) {
      const d = Math.hypot(x1 - x2, y1 - y2) || 1
      x2 += ((x1 - x2) / d) * focusRingR
      y2 += ((y1 - y2) / d) * focusRingR
    }

    const key = `${s.id}->${t.id}`
    const ambient = recencyInk(erec((s.rec + t.rec) / 2)) * c.lineAlpha

    // Hovering a line fades it in a bit (×2, capped — never full white).
    const targetAlpha = !revealed
      ? 0
      : lit
        ? 1
        : key === hoverLink
          ? clamp(ambient * 2, 0, 0.7)
          : focusId || ring
            ? 0.025
            : ambient

    const linkAlpha = fadeAlpha(fades.links, key, targetAlpha, lit)

    if (linkAlpha < 0.004) {
      continue
    }

    ctx.strokeStyle = shade(linkAlpha)
    ctx.setLineDash(lit || !c.lineDashed ? [] : [c.lineDash, c.lineDash])
    ctx.lineWidth = lit ? 1.5 : c.lineWidth
    ctx.beginPath()
    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2)
    ctx.stroke()
  }

  ctx.setLineDash([])

  // Nodes: the node layer paints pure ink (focused node + neighbors); the date
  // filter is alpha-only, so the two states compose. Track which rings have at
  // least one revealed node so a ring's date only shows once it has content.
  const revealedRings = new Set<number>()

  for (const n of nodes) {
    // The land comes first: a node waits for the ring that CAPS its region (its
    // outer date gridline) to grow in before it ignites — so the ring is always
    // drawn before any star inside it, not after.
    const landLaid = (ringAppear[n.outerRingIndex] ?? 1) >= 0.5
    const revealed = seen(n.rec) && landLaid

    if (revealed) {
      revealedRings.add(n.outerRingIndex)
    }
    const isFocus = revealed && n.id === focusId
    const isNeighbor = revealed && !!focusSet && focusSet.has(n.id)
    const inRing = !!ring && n.rec >= ringLo && n.rec < ringHi
    const nodeHigh = isFocus || isNeighbor
    const er = erec(n.rec)
    const ageScale = nodeHigh || inRing ? 1 : 0.34 + Math.min(1, er / 0.4) * 0.66
    // Stable screen-space radius: use the graph's resting fit zoom, not the
    // current playback camera zoom. Full-map views keep their original density,
    // while t≈0 spore-zoom no longer inflates nodes into bubbles.
    const r = nodeRadius(n) * nodeK * ageScale

    const baseAlpha = nodeHigh ? 1 : ring ? (inRing ? (focusId ? 0.55 : 1) : 0.16) : focusId ? 0.16 : recencyInk(er)
    const alpha = fadeAlpha(fades.nodes, n.id, revealed ? baseAlpha : 0, nodeHigh || inRing)

    // Birth fade + warp rise are coupled (slow rates) so a star grows in instead
    // of flashing. Focus snaps (no drift).
    const rawBorn = fadeAlpha(fades.appear, n.id, revealed ? 1 : 0, nodeHigh || inRing, NODE_BIRTH)
    const born = ease(rawBorn)
    const vis = alpha * born

    if (vis < 0.004) {
      continue
    }

    // Warp-in: streak outward from WARP_FROM·radius and decelerate hard onto the
    // ring (origin = disk core), echoing an EVE ship dropping out of warp.
    const posScale = WARP_FROM + (1 - WARP_FROM) * warpIn(rawBorn)
    const sx = projX(n.x * posScale)
    const sy = projY(n.y * posScale)

    ctx.globalAlpha = vis
    const nodeInk = nodeHigh ? base : n.kind === 'memory' ? memoryInk : skillInk
    const shape = NODE_SHAPE[n.kind]

    if (shape === 'circle') {
      // Highlighted orbs pop full bright; others darken so the sheen reads. The
      // sprite carries the disk, so no path is built for circles.
      sphereFill(ctx, sx, sy, r, nodeInk, sheen, nodeHigh ? 0 : ORB_DARKEN)
    } else {
      shapePath(ctx, shape, sx, sy, r)
      ctx.fillStyle = rgba(nodeInk, 1)
      ctx.fill()
    }

    if (isFocus) {
      ctx.globalAlpha = 1
      ctx.strokeStyle = rgba(nodeInk, 1)
      ctx.lineWidth = 1.4
      shapePath(ctx, shape, sx, sy, r + 4)
      ctx.stroke()
    }
  }

  ctx.globalAlpha = 1
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

  // Ring date labels (top of each ellipse) — hoverable to focus the ring. Many
  // adaptive rings can crowd the top, so labels thin out: skip any that would
  // land within LABEL_GAP of the last one drawn (the gridline still shows).
  ctx.font = '10px ui-sans-serif, system-ui, sans-serif'
  ctx.textAlign = 'center'
  const LABEL_GAP = 15
  let lastLabelY = Number.POSITIVE_INFINITY
  // A ring's date only shows once it actually has a revealed node — no floating
  // date over a blank disk (t=0) or a lone empty ring.
  rings.forEach((rg, i) => {
    if (!rg.label || !revealedRings.has(i)) {
      return
    }

    const sx = projX(0)
    // Track the growing radius so the date rides the ring as it expands out.
    const sy = projY(-(ringDrawR[i] ?? rg.r))

    if (sy < 8 || sy > h - 8 || lastLabelY - sy < LABEL_GAP) {
      return
    }

    lastLabelY = sy
    const tw = ctx.measureText(rg.label).width
    const boxW = tw + 6
    const isThis = ringIdx === i || hoverRing === i
    const faded = (focusId != null || ringIdx != null) && !isThis
    // The date rides the same smooth ringAppear envelope, so it recedes as
    // gently as it appears; the bucket carries only the snappy focus/selection dim.
    const emphasisAlpha = faded ? 0.33 : 1
    const labelAlpha = fadeAlpha(fades.labels, String(i), emphasisAlpha, isThis) * (ringVis[i] ?? 1)

    if (labelAlpha < 0.01) {
      return
    }

    ctx.globalAlpha = labelAlpha
    ctx.fillStyle = rgba(bg, 1)
    ctx.fillRect(sx - boxW / 2, sy - 6, boxW, 13)
    ctx.fillStyle = shade(isThis ? 1 : 0.2)
    ctx.fillText(rg.label, sx, sy + 3)
    ctx.globalAlpha = 1
    // Hidden labels (mid fade-out / not yet reached) drop out of hit-testing.
    ringLabelRects.push({ h: 18, i, w: boxW + 6, x: sx - boxW / 2 - 3, y: sy - 10 })
  })

  // Tooltip on focus — measured first so its rect joins the avoidance set and
  // neighbor labels route around it.
  const tipNode = focusId ? byId.get(focusId) : null
  const tip = tipNode && seen(tipNode.rec) ? tipNode : null
  let tipRect: null | Rect = null

  if (tip) {
    const PADX = 6
    const PADY = 4
    const BADGE_H = 14
    const ROW_GAP = 3
    const LINE_H = 16
    const ITEM_GAP = 8
    const badgeFont = '9px ui-sans-serif, system-ui, sans-serif'
    const monoFont = '9px ui-monospace, SFMono-Regular, Menlo, monospace'
    const titleFont = '600 11px ui-sans-serif, system-ui, sans-serif'
    const footerFont = '9px ui-sans-serif, system-ui, sans-serif'
    const FOOTER_H = 13
    // The date (index 0) stays sans; the rest of the tags are monospace.
    const badgeFontFor = (i: number) => (i === 0 ? badgeFont : monoFont)

    const badges = metaBadges(tip)
    const use = countLabel(tip)
    const titleText = tip.kind === 'memory' ? memById.get(tip.id)?.body.split('\n')[0]?.trim() || tip.label : tip.label

    const badgeW = badges.map((b, i) => {
      ctx.font = badgeFontFor(i)

      return ctx.measureText(b).width
    })

    const rowW = badgeW.reduce((a, b) => a + b, 0) + ITEM_GAP * Math.max(0, badges.length - 1)
    ctx.font = monoFont
    const useW = use ? ctx.measureText(use).width : 0
    const metaW = rowW + (use ? ITEM_GAP + useW : 0)

    ctx.font = titleFont
    const maxTitleW = Math.min(380, w - 16) - PADX * 2
    const titleLines = wrapText(ctx, titleText, maxTitleW)
    const titleW = Math.max(0, ...titleLines.map(l => ctx.measureText(l).width))
    const titleBgW = titleW + PADX * 2
    const titleBgH = titleLines.length * LINE_H + PADY * 2

    const footerText = nodeFooter(tip)
    ctx.font = footerFont
    const footerW = footerText ? ctx.measureText(footerText).width : 0

    const totalW = Math.max(metaW, footerW, titleBgW)
    const totalH = BADGE_H + ROW_GAP + titleBgH + (footerText ? ROW_GAP + FOOTER_H : 0)
    const bx = clamp(projX(tip.x) - totalW / 2, 4, Math.max(4, w - totalW - 4))
    const by = clamp(projY(tip.y) - (nodeRadius(tip) * nodeK + 8) - totalH, 4, Math.max(4, h - totalH - 4))
    tipRect = { h: totalH, w: totalW, x: bx, y: by }

    ctx.textAlign = 'left'
    ctx.textBaseline = 'middle'
    const badgeMidY = by + BADGE_H / 2

    // Metadata row, flush at the left edge.
    ctx.fillStyle = shade(0.7)
    let cx = bx
    badges.forEach((label, i) => {
      ctx.font = badgeFontFor(i)
      ctx.fillText(label, cx, badgeMidY)
      cx += badgeW[i] + ITEM_GAP
    })

    if (use) {
      ctx.font = monoFont
      ctx.fillStyle = shade(0.5)
      ctx.fillText(use, cx, badgeMidY)
    }

    // Title: inverted (fg/bg flipped) so the focused tooltip pops.
    const ty = by + BADGE_H + ROW_GAP
    ctx.fillStyle = shade(1)
    ctx.fillRect(bx, ty, titleBgW, titleBgH)
    ctx.font = titleFont
    ctx.fillStyle = inkInv
    titleLines.forEach((line, i) => {
      ctx.fillText(line, bx + PADX, ty + PADY + LINE_H * i + LINE_H / 2)
    })

    if (footerText) {
      ctx.font = footerFont
      ctx.fillStyle = shade(0.45)
      ctx.fillText(footerText, bx, ty + titleBgH + ROW_GAP + FOOTER_H / 2)
    }

    ctx.textBaseline = 'alphabetic'
  }

  // Neighbor constellation labels — greedy placement that clamps to the overlay
  // and dodges placed labels (date labels + tooltip) so nothing overlaps/clips.
  ctx.font = '11px ui-sans-serif, system-ui, sans-serif'
  ctx.textAlign = 'center'
  const LBL_M = 6
  const LBL_H = 15
  const placed: Rect[] = ringLabelRects.map(r => ({ h: r.h, w: r.w, x: r.x, y: r.y }))

  if (tipRect) {
    placed.push(tipRect)
  }

  for (const id of focusSet ?? []) {
    if (id === hoverId) {
      continue
    }

    const n = byId.get(id)

    if (!n || !seen(n.rec)) {
      continue
    }

    const label = ellipsize(ctx, n.label, Math.min(180, w * 0.32))
    const bw = ctx.measureText(label).width + 8
    const x = clamp(projX(n.x) - bw / 2, LBL_M, Math.max(LBL_M, w - bw - LBL_M))
    const top = projY(n.y) - (nodeRadius(n) * nodeK + 7) - LBL_H + 4
    const clampY = (v: number) => clamp(v, LBL_M, Math.max(LBL_M, h - LBL_H - LBL_M))
    const step = LBL_H + 3
    let y: null | number = null

    // Prefer above the node, then fan outward; skip if nothing stays clear (a
    // label on the tooltip reads worse than no label).
    for (let k = 0; k <= 7 && y == null; k += 1) {
      for (const dy of k === 0 ? [0] : [-k * step, k * step]) {
        const cand = { h: LBL_H, w: bw, x, y: clampY(top + dy) }

        if (!placed.some(p => rectsOverlap(cand, p))) {
          y = cand.y

          break
        }
      }
    }

    if (y == null) {
      continue
    }

    placed.push({ h: LBL_H, w: bw, x, y })
    ctx.fillStyle = chipBg
    ctx.fillRect(x, y, bw, LBL_H)
    ctx.fillStyle = shade(0.85)
    ctx.fillText(label, x + bw / 2, y + 11)
  }

  return { animating, ringLabelRects }
}

// Glyph cells from the core's center to its rim — the target density. In the mid
// range the field is this many cells across (constant "amount of text"), and the
// glyph size tracks the camera. Bump for denser, drop for sparser.
const SCRAMBLE_RADIUS = 6

// Glyph size (px) is clamped to this band: the font grows with the camera but
// never balloons on a big/zoomed-in core — past the ceiling the core fills with
// MORE, smaller glyphs instead of fewer huge ones — and stays legible when tiny.
const SCRAMBLE_CELL_MIN = 5
const SCRAMBLE_CELL_MAX = 13

// The empty-core scramble: a tilted, Matrix-style decoding-glyph field laid on
// the disk plane (rows squashed by TILT, clipped to the core ellipse) so the
// empty center reads as "computing", not missing. PURELY decorative — the glyphs
// are a seeded PRNG field, never derived from nodes/memories. Drawn live each
// frame on top of the cached static scene, since it's the only animated layer.
export function drawScramble({
  ctx,
  dpr,
  palette,
  rings,
  vp
}: {
  ctx: CanvasRenderingContext2D
  dpr: number
  palette: Palette
  rings: Ring[]
  vp: Viewport
}): void {
  const { darkTheme, primary } = palette
  const projX = (wx: number) => wx * vp.k + vp.x
  const projY = (wy: number) => wy * vp.k * TILT + vp.y

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

  const coreX = projX(0)
  const coreY = projY(0)
  // Fill to the innermost ring (the core shell), not the RING_INNER constant —
  // the ring sits in lead-in space, so derive the radius from it directly.
  const coreRx = (rings[0]?.r ?? RING_INNER) * vp.k * 0.94

  if (coreRx <= 0) {
    return
  }

  // Target ~SCRAMBLE_RADIUS cells to the rim (camera-scaled glyphs), but clamp the
  // glyph SIZE so a big/zoomed-in core scales the font DOWN — packing in more,
  // smaller glyphs rather than a few giant ones — and stays legible when tiny.
  const cell = clamp(coreRx / SCRAMBLE_RADIUS, SCRAMBLE_CELL_MIN, SCRAMBLE_CELL_MAX)
  // Aspect-correct on the tilt: rows are spaced by the full glyph height (square
  // cells, no vertical squish), but the field is clipped to the disk's ELLIPSE
  // (vertical extent = coreRx * TILT), so it sits on the tilted plane while the
  // glyphs themselves stay un-squished. Fewer rows fit vertically — that's it.
  const coreRy = coreRx * TILT
  const half = Math.max(3, Math.round(coreRx / cell))
  const now = performance.now()

  ctx.save()
  ctx.font = `${cell}px "JetBrains Mono", "Hiragino Sans", "Noto Sans JP", ui-monospace, monospace`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'

  for (let r = -half; r <= half; r += 1) {
    // Per-row flow: half the rows drift left, half right, each at its own speed.
    // The drift is a continuous pixel scroll (not a per-cell swap), and each
    // glyph's identity is tied to its slot index — so a character visibly slides
    // across instead of the whole row flickering in place. Combined with the
    // TILT squash + opposite directions, the field reads as a turning surface.
    const rowSeed = (r * 19349663) >>> 0 || 1
    const dir = rowSeed & 1 ? 1 : -1
    const speed = 8 + (rowSeed % 16) // px/sec
    const scroll = (now / 1000) * speed * dir
    const ny = (r * cell) / coreRy
    // Latitude dimming: rows away from the equator fade, selling the sphere read.
    const rowDim = 1 - 0.5 * Math.min(1, Math.abs(ny))
    const kMin = Math.floor((-coreRx - scroll) / cell) - 1
    const kMax = Math.ceil((coreRx - scroll) / cell) + 1

    for (let k = kMin; k <= kMax; k += 1) {
      const sx = k * cell + scroll // screen-space x relative to the core center
      const nx = sx / coreRx
      const d2 = nx * nx + ny * ny

      if (d2 > 1) {
        continue
      }

      const seed = (rowSeed ^ ((k >>> 0) * 73856093)) >>> 0
      const ch = SCRAMBLE_CHARS[seed % SCRAMBLE_CHARS.length] ?? '0'
      // Mostly flat brightness, fading only near the rim (reduced gradient).
      const edge = clamp((1 - Math.sqrt(d2)) / 0.4, 0, 1)
      const flick = 0.7 + 0.3 * (((seed >>> 5) % 100) / 100)
      // Fake depth: a stable per-slot value pops a subset of glyphs forward, so
      // some characters read as nearer/brighter and drift across in front.
      const depth = ((seed >>> 11) % 100) / 100
      const pop = depth > 0.92 ? 2.6 : depth > 0.78 ? 1.6 : 1
      const a = clamp((darkTheme ? 0.25 : 0.33) * edge * flick * rowDim * pop, 0, 0.85)

      if (a < 0.02) {
        continue
      }

      ctx.fillStyle = rgba(primary, a)
      ctx.fillText(ch, coreX + sx, coreY + r * cell)
    }
  }

  ctx.restore()
  ctx.globalAlpha = 1
}
