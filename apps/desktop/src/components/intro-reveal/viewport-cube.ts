/** Canvas geometry and texture noise derive from score time, so every playback draws the same frames. */

import { INTRO_BEATS } from './timeline'

const N = 4

/** End of the cube's draw window. use-intro-clock.ts stops calling drawViewport at this time. */
export const VIEWPORT_END_MS = INTRO_BEATS.find(b => b.id === 'everywhere')!.t + 700

/** Each material has an explicit start time, so the two `texture` slots sit at chosen moments in the
 *  sequence. Cycling the list at a fixed interval placed them wherever the modulo fell. */
const VIEWPORT_SCHEDULE = [
  { at: 0, mode: 'standard' },
  { at: 1900, mode: 'metal' },
  { at: 3700, mode: 'texture' },
  { at: 5800, mode: 'glass' },
  { at: 7000, mode: 'wireframe' },
  { at: 8300, mode: 'texture' },
  { at: 10500, mode: 'wireframe' }
] as const

const CROSSFADE_MS = 620

export type ViewportMode = (typeof VIEWPORT_SCHEDULE)[number]['mode']

export interface ViewportSlot {
  at: number
  index: number
  mode: ViewportMode
  until: number
}

/** The material showing at `t`, with the start and end of its window. Callers use the bounds for the
 *  crossfade, the tear ramps and the label's decode. */
export function viewportSlot(t: number): ViewportSlot {
  let index = 0

  for (let i = 0; i < VIEWPORT_SCHEDULE.length; i += 1) {
    if (VIEWPORT_SCHEDULE[i].at <= t) {
      index = i
    }
  }

  return {
    at: VIEWPORT_SCHEDULE[index].at,
    index,
    mode: VIEWPORT_SCHEDULE[index].mode,
    until: VIEWPORT_SCHEDULE[index + 1]?.at ?? VIEWPORT_END_MS
  }
}

interface Quad {
  z: number
  pts: [number, number][]
  shade: number
  /** Grid cell on its face, for texture coordinates. */
  cell: [number, number]
}

/** Subdivided cube quads, rotated and projected. The subdivision keeps each texture cell small enough that
 *  an affine map reads as a perspective one. */
function cubeQuads(t: number, w: number, h: number): Quad[] {
  const rx = t * 0.00042
  const ry = t * 0.00071
  const cx = Math.cos(rx)
  const sx = Math.sin(rx)
  const cy = Math.cos(ry)
  const sy = Math.sin(ry)
  // Scaled off the smaller side so the cube stays clear of the viewport frame.
  const scale = Math.min(w, h) * 0.24
  const quads: Quad[] = []

  const vert = (u: number, v: number, face: number): [number, number, number] => {
    const a = -1 + (2 * u) / N
    const b = -1 + (2 * v) / N

    const p: [number, number, number] =
      face === 0
        ? [a, b, 1]
        : face === 1
          ? [a, b, -1]
          : face === 2
            ? [1, a, b]
            : face === 3
              ? [-1, a, b]
              : face === 4
                ? [a, 1, b]
                : [a, -1, b]

    const x1 = p[0] * cy + p[2] * sy
    const z1 = -p[0] * sy + p[2] * cy
    const y2 = p[1] * cx - z1 * sx
    const z2 = p[1] * sx + z1 * cx

    return [x1, y2, z2]
  }

  for (let face = 0; face < 6; face += 1) {
    for (let u = 0; u < N; u += 1) {
      for (let v = 0; v < N; v += 1) {
        const c: [number, number, number][] = [
          vert(u, v, face),
          vert(u + 1, v, face),
          vert(u + 1, v + 1, face),
          vert(u, v + 1, face)
        ]

        const z = (c[0][2] + c[1][2] + c[2][2] + c[3][2]) / 4
        const ux = c[1][0] - c[0][0]
        const uy = c[1][1] - c[0][1]
        const uz = c[1][2] - c[0][2]
        const vx = c[3][0] - c[0][0]
        const vy = c[3][1] - c[0][1]
        const vz = c[3][2] - c[0][2]
        const nx = uy * vz - uz * vy
        const ny = uz * vx - ux * vz
        const nz = ux * vy - uy * vx
        const nl = Math.hypot(nx, ny, nz) || 1
        const shade = Math.abs(nz / nl)

        quads.push({
          z,
          shade,
          cell: [u, v],
          pts: c.map(([x, y, zz]): [number, number] => {
            const persp = 3.6 / (3.6 - zz * 0.9)

            return [w / 2 + x * scale * persp, h / 2 + y * scale * persp]
          })
        })
      }
    }
  }

  return quads.sort((a, b) => a.z - b.z)
}

// Texture pass: the image mapped onto the cube, with an RGB channel separation.
// The cube renders once, is split into red, green and blue copies, and the three
// are re-composited with 'lighter' at diverging offsets. At zero offset they sum
// back to the untouched image.

const CHANNEL_TINTS = ['#ff0000', '#00ff00', '#0000ff'] as const
const SLICES = 14

let texture: HTMLImageElement | null = null
let textureRequested = false

/** Starts the image load on the first call and returns the cached image afterwards. Returns null until the
 *  image decodes, and paintTexturedCube draws nothing on those frames. */
function textureImage(): HTMLImageElement | null {
  if (textureRequested || globalThis.document === undefined) {
    return texture
  }

  textureRequested = true

  const img = new Image()

  img.onload = () => {
    texture = img
  }

  // Not `nous-girl.jpg`: that asset is the BrandMark tile art, dark on white, and
  // reads as a solid white block once it is wrapped around a cube. This one is
  // light-on-dark line work, so the faces keep their shading and the channel
  // split has edges to offset.
  img.src = `${import.meta.env.BASE_URL}intro-nous-girl.png`

  return null
}

const scratch = new Map<string, HTMLCanvasElement>()

/** A cleared offscreen context at device resolution. `scale` applies the DPR, so callers keep drawing in the
 *  same CSS pixels the quads are projected into. */
function buffer(key: string, w: number, h: number, scale: number): CanvasRenderingContext2D {
  let canvas = scratch.get(key)

  if (!canvas) {
    canvas = document.createElement('canvas')
    scratch.set(key, canvas)
  }

  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w
    canvas.height = h
  }

  const ctx = canvas.getContext('2d')!

  ctx.setTransform(scale, 0, 0, scale, 0, 0)
  ctx.globalCompositeOperation = 'source-over'
  ctx.globalAlpha = 1
  ctx.clearRect(0, 0, w, h)

  return ctx
}

/** Deterministic value noise. The tear has to replay identically on every playback. */
function hash(n: number): number {
  const s = Math.sin(n * 12.9898) * 43758.5453

  return s - Math.floor(s)
}

/** 0 is settled, 1 is fully torn. Starts fully torn and settles over the first 460ms, stutters at random
 *  during the hold, then tears out over the last 420ms of the slot. */
function tearAmount(local: number, span: number): number {
  const arriving = 1 - Math.min(1, local / 460)
  const leaving = Math.max(0, (local - (span - 420)) / 420)
  const step = Math.floor(local / 90)
  const stutter = hash(step) > 0.88 ? hash(step * 1.7) * 0.5 : 0

  return Math.min(1, Math.max(arriving, leaving, stutter))
}

let scanPattern: CanvasPattern | null = null

/** Scanline pattern of one dark row in three, built once. It is filled under the buffer's DPR transform, so
 *  the lines keep a constant weight in CSS pixels at any scale. */
function scanlines(ctx: CanvasRenderingContext2D): CanvasPattern | null {
  if (!scanPattern) {
    const canvas = document.createElement('canvas')

    canvas.width = 1
    canvas.height = 3

    const tile = canvas.getContext('2d')!

    tile.fillStyle = 'rgba(0, 0, 0, 0.34)'
    tile.fillRect(0, 0, 1, 1)
    scanPattern = ctx.createPattern(canvas, 'repeat')
  }

  return scanPattern
}

function inflate(pts: [number, number][], px: number): [number, number][] {
  const cx = (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4
  const cy = (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4

  return pts.map(([x, y]): [number, number] => {
    const dx = x - cx
    const dy = y - cy
    const d = Math.hypot(dx, dy) || 1

    return [x + (dx / d) * px, y + (dy / d) * px]
  })
}

function paintTexturedCube(
  ctx: CanvasRenderingContext2D,
  quads: Quad[],
  w: number,
  h: number,
  slot: ViewportSlot,
  t: number,
  alpha: number
): void {
  const img = textureImage()

  if (!img) {
    return
  }

  const local = t - slot.at

  // The surface passes a DPR-scaled context and CSS-pixel geometry. The offscreens
  // use the same DPR, otherwise this pass renders at 1x and is upscaled.
  const dpr = ctx.getTransform().a || 1
  const dw = Math.ceil(w * dpr)
  const dh = Math.ceil(h * dpr)

  const tear = tearAmount(local, slot.until - slot.at)
  const cube = buffer('cube', dw, dh, dpr)
  const sw = img.naturalWidth / N
  const sh = img.naturalHeight / N

  for (const q of quads) {
    const [p0, p1, , p3] = q.pts
    // Two clips that meet on an edge each antialias to half cover, which on a
    // white texture reads as a grey hairline grid. The cells overlap instead: the
    // texture is opaque and drawn back to front, so a later cell repaints the seam.
    const poly = inflate(q.pts, 0.6)

    cube.save()
    cube.beginPath()
    cube.moveTo(poly[0][0], poly[0][1])

    for (let i = 1; i < 4; i += 1) {
      cube.lineTo(poly[i][0], poly[i][1])
    }

    cube.closePath()
    cube.clip()
    // The image is light-on-dark line work, so each face needs an opaque fill
    // first. Without it the cube's unlit areas are the same black as the viewport
    // behind it and only stray white curves show. The fill also covers the
    // inflated overlap before the line work is drawn.
    cube.fillStyle = `rgb(${12 + q.shade * 20}, ${13 + q.shade * 22}, ${17 + q.shade * 28})`
    cube.fillRect(0, 0, w, h)
    // Affine map from the unit cell to this quad. It ignores the fourth corner, so
    // the texture swims slightly across a face (affine texture warping).
    cube.transform(p1[0] - p0[0], p1[1] - p0[1], p3[0] - p0[0], p3[1] - p0[1], p0[0], p0[1])
    // 'lighter' adds the line work instead of covering the fill: black in the
    // image contributes nothing, and the alpha below scales with the face's shade.
    cube.globalCompositeOperation = 'lighter'
    cube.globalAlpha = 0.55 + q.shade * 0.45
    cube.drawImage(img, q.cell[0] * sw, q.cell[1] * sh, sw, sh, -0.06, -0.06, 1.12, 1.12)
    cube.restore()
  }

  // CRT pass, drawn with 'source-atop' so it stays inside the cube's own alpha and
  // does not touch the empty space around the solid. It is composited before the
  // channel split below, so the split offsets the sweep and the grille too.
  cube.globalCompositeOperation = 'source-atop'

  const sweep = ((local % 1150) / 1150) * 1.3 - 0.15
  const bar = cube.createLinearGradient(0, (sweep - 0.13) * h, 0, (sweep + 0.13) * h)

  bar.addColorStop(0, 'rgba(120, 200, 255, 0)')
  bar.addColorStop(0.5, 'rgba(165, 220, 255, 0.26)')
  bar.addColorStop(1, 'rgba(120, 200, 255, 0)')
  cube.fillStyle = bar
  cube.fillRect(0, 0, w, h)

  const grille = scanlines(cube)

  if (grille) {
    cube.fillStyle = grille
    cube.fillRect(0, 0, w, h)
  }

  const step = Math.floor(local / 90)
  // The 0.7 term keeps a minimum separation, so held frames still show a small
  // offset instead of a clean image.
  const rip = (tear * 8 + 0.7) * dpr

  ctx.save()
  // Composite in device space: the offsets are in device pixels and the buffers
  // are already at device resolution.
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.globalCompositeOperation = 'lighter'
  ctx.globalAlpha = alpha

  for (let c = 0; c < 3; c += 1) {
    const chan = buffer('chan', dw, dh, 1)

    chan.drawImage(cube.canvas, 0, 0)
    // Isolate one channel: multiply by a primary, then re-apply the cube's own
    // alpha, because the full-canvas fill also tints the empty space.
    chan.globalCompositeOperation = 'multiply'
    chan.fillStyle = CHANNEL_TINTS[c]
    chan.fillRect(0, 0, dw, dh)
    chan.globalCompositeOperation = 'destination-in'
    chan.drawImage(cube.canvas, 0, 0)

    // dx offsets red left and blue right and leaves green at 0. The per-slice
    // jitter below breaks the silhouette as well as the colour.
    const dx = (c - 1) * rip

    for (let s = 0; s < SLICES; s += 1) {
      // Bands are integer and abutting. Overlapping rows would be summed twice by
      // 'lighter' and show as bright lines across the cube.
      const y0 = Math.round((s * dh) / SLICES)
      const band = Math.round(((s + 1) * dh) / SLICES) - y0
      const jitter = (hash(step * 31 + s) - 0.5) * 2 * tear * 11 * dpr

      ctx.drawImage(chan.canvas, 0, y0, dw, band, dx + jitter, y0, dw, band)
    }
  }

  ctx.restore()
}

export function drawViewport(ctx: CanvasRenderingContext2D, w: number, h: number, t: number) {
  // Start the image load on the first frame. The first `texture` slot opens at
  // 3700ms, and a decode that arrives during a crossfade would show the cube
  // empty for those frames.
  textureImage()

  const slot = viewportSlot(t)
  const mode = slot.mode
  // Materials crossfade at slot boundaries: the incoming material fades up over
  // the outgoing one.
  const prevEntry = VIEWPORT_SCHEDULE[slot.index - 1]
  const prevMode = prevEntry?.mode ?? mode
  const fade = Math.min(CROSSFADE_MS, (slot.until - slot.at) * 0.4)
  const blendF = Math.min(1, (t - slot.at) / fade)
  const blend = blendF * blendF * (3 - 2 * blendF)

  ctx.clearRect(0, 0, w, h)

  const quads = cubeQuads(t, w, h)

  ctx.save()
  ctx.font = "8px 'JetBrains Mono', monospace"

  const rx = t * 0.00042
  const ry = t * 0.00071
  const gx = 24
  const gy = h - 22

  const axes: [string, number, number, number, string][] = [
    ['x', 1, 0, 0, 'rgba(248, 113, 113, 0.8)'],
    ['y', 0, -1, 0, 'rgba(74, 222, 128, 0.8)'],
    ['z', 0, 0, 1, 'rgba(96, 165, 250, 0.8)']
  ]

  for (const [label, ax, ay, az, color] of axes) {
    const x1 = ax * Math.cos(ry) + az * Math.sin(ry)
    const z1 = -ax * Math.sin(ry) + az * Math.cos(ry)
    const y2 = ay * Math.cos(rx) - z1 * Math.sin(rx)
    const px = gx + x1 * 13
    const py = gy + y2 * 13

    ctx.strokeStyle = color
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(gx, gy)
    ctx.lineTo(px, py)
    ctx.stroke()
    ctx.fillStyle = color
    ctx.fillText(label, px + 2, py + 3)
  }

  // The 6 * 5 * 5 below is 6 faces times (N + 1)² shared grid vertices, for N = 4.
  const deg = (r: number) => ((((r * 180) / Math.PI) % 360) | 0).toString().padStart(3, ' ')

  ctx.fillStyle = 'rgba(255,255,255,0.22)'
  ctx.fillText(`rx ${deg(rx)}\u00b0  ry ${deg(ry)}\u00b0`, 12, 14)
  const verts = `${6 * 5 * 5} verts \u00b7 ${quads.length} faces`

  ctx.fillText(verts, w - ctx.measureText(verts).width - 12, h - 10)
  ctx.restore()

  // One painter per material. `texture` is not handled here: it is a whole-cube
  // pass below, because its channel split has to happen in screen space.
  const paint = (m: ViewportMode, q: Quad, alpha: number) => {
    if (alpha <= 0.01 || m === 'texture') {
      return
    }

    ctx.globalAlpha = alpha

    if (m === 'standard') {
      const l = 152 + q.shade * 88

      ctx.fillStyle = `rgb(${l}, ${l}, ${l + 2})`
      ctx.fill()
      ctx.strokeStyle = 'rgba(0,0,0,0.16)'
      ctx.lineWidth = 0.5
      ctx.stroke()
    } else if (m === 'metal') {
      const s = Math.pow(q.shade, 2.6)
      const v = 26 + s * 205

      ctx.fillStyle = `rgb(${v * 0.92}, ${v * 0.97}, ${Math.min(255, v * 1.06 + 6)})`
      ctx.fill()
      ctx.strokeStyle = 'rgba(255,255,255,0.07)'
      ctx.lineWidth = 0.5
      ctx.stroke()
    } else if (m === 'glass') {
      const rim = 1 - q.shade

      ctx.fillStyle = `rgba(140, 180, 255, ${0.05 + rim * 0.17})`
      ctx.fill()
      ctx.strokeStyle = `rgba(170, 200, 255, ${0.1 + rim * 0.38})`
      ctx.lineWidth = 0.7
      ctx.stroke()
    } else {
      ctx.strokeStyle = `rgba(255,255,255,${0.14 + q.shade * 0.2})`
      ctx.lineWidth = 1
      ctx.stroke()
    }
  }

  for (const q of quads) {
    ctx.beginPath()
    ctx.moveTo(q.pts[0][0], q.pts[0][1])

    for (let i = 1; i < 4; i += 1) {
      ctx.lineTo(q.pts[i][0], q.pts[i][1])
    }

    ctx.closePath()

    if (blend < 1) {
      paint(prevMode, q, 1 - blend)
    }

    paint(mode, q, blend)
  }

  ctx.globalAlpha = 1

  if (mode === 'texture') {
    paintTexturedCube(ctx, quads, w, h, slot, t, blend)
  } else if (prevMode === 'texture' && prevEntry) {
    // The outgoing texture keeps its own slot bounds, so the tear-out that began
    // at the end of its window carries through the crossfade. Reading the incoming
    // slot's local time restarts the tear ramp instead.
    paintTexturedCube(
      ctx,
      quads,
      w,
      h,
      { at: prevEntry.at, index: slot.index - 1, mode: 'texture', until: slot.at },
      t,
      1 - blend
    )
  }
}
