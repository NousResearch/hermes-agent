import type { OrbMotionTargets } from './particle-orb'
import type { PlasmaPalette, PlasmaSurface } from './plasma'

export interface OrbRenderState {
  count: number
  electrons: readonly { from: number; to: number; progress: number }[]
  level: number
  links: readonly number[]
  motion: Readonly<OrbMotionTargets>
  outline: Readonly<Float32Array>
  positions: Float32Array
  projected: Float32Array
}

const LINK_DISTANCE = 0.3
const ALPHA_BUCKETS = 5
const OUTLINE_BINS = 48
const linkBuckets: number[][] = Array.from({ length: ALPHA_BUCKETS }, () => [])

function bodyPath(cx: number, cy: number, radius: number, radii: Readonly<Float32Array>): Path2D {
  const path = new Path2D()
  const step = (Math.PI * 2) / OUTLINE_BINS

  const point = (bin: number): [number, number] => {
    const index = (bin + OUTLINE_BINS) % OUTLINE_BINS
    const angle = (index + 0.5) * step - Math.PI
    const r = radii[index] * 1.02 + radius * 0.02

    return [cx + Math.cos(angle) * r, cy + Math.sin(angle) * r]
  }

  const [x0, y0] = point(0)
  const [x1, y1] = point(1)
  path.moveTo((x0 + x1) / 2, (y0 + y1) / 2)

  for (let bin = 1; bin <= OUTLINE_BINS; bin += 1) {
    const [x, y] = point(bin)
    const [nx, ny] = point(bin + 1)
    path.quadraticCurveTo(x, y, (x + nx) / 2, (y + ny) / 2)
  }

  path.closePath()

  return path
}

function drawBody(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  radius: number,
  radii: Readonly<Float32Array>,
  palette: PlasmaPalette,
  light: boolean,
  level: number,
  time: number
): void {
  const path = bodyPath(cx, cy, radius, radii)
  const fill = ctx.createRadialGradient(cx - radius * 0.2, cy - radius * 0.25, radius * 0.05, cx, cy, radius * 1.12)

  if (light) {
    fill.addColorStop(0, 'rgba(255, 255, 255, 0.5)')
    fill.addColorStop(0.48, `rgba(${palette.highlight}, ${0.07 + level * 0.06})`)
    fill.addColorStop(0.8, `rgba(${palette.front}, ${0.2 + level * 0.08})`)
    fill.addColorStop(1, `rgba(${palette.rim}, 0.34)`)
  } else {
    fill.addColorStop(0, `rgba(${palette.core}, ${0.16 + level * 0.1})`)
    fill.addColorStop(0.3, `rgba(${palette.highlight}, 0.09)`)
    fill.addColorStop(0.62, `rgba(${palette.front}, 0.06)`)
    fill.addColorStop(0.85, `rgba(${palette.back}, ${0.14 + level * 0.08})`)
    fill.addColorStop(1, `rgba(${palette.rim}, 0.32)`)
  }

  ctx.fillStyle = fill
  ctx.fill(path)

  const sheen = ctx.createConicGradient(time * 0.35, cx, cy)
  sheen.addColorStop(0, `rgba(${palette.front}, 0)`)
  sheen.addColorStop(0.18, `rgba(${palette.highlight}, ${light ? 0.12 : 0.1})`)
  sheen.addColorStop(0.36, `rgba(${palette.back}, 0)`)
  sheen.addColorStop(0.62, `rgba(${palette.back}, ${light ? 0.1 : 0.08})`)
  sheen.addColorStop(0.8, `rgba(${palette.rim}, 0)`)
  sheen.addColorStop(1, `rgba(${palette.front}, 0)`)
  ctx.fillStyle = sheen
  ctx.fill(path)

  // A soft outer pass gives the rim width without making its hard edge louder.
  ctx.save()
  ctx.shadowColor = `rgba(${palette.rim}, ${light ? 0.3 : 0.65})`
  ctx.shadowBlur = radius * (0.2 + level * 0.1)
  ctx.strokeStyle = `rgba(${palette.rim}, ${light ? 0.16 : 0.2})`
  ctx.lineWidth = Math.max(2, radius * 0.035)
  ctx.stroke(path)
  ctx.restore()

  ctx.save()
  ctx.shadowColor = `rgba(${palette.rim}, ${light ? 0.45 : 0.85})`
  ctx.shadowBlur = radius * (0.12 + level * 0.1)
  ctx.strokeStyle = `rgba(${palette.rim}, ${(light ? 0.55 : 0.6) + level * 0.25})`
  ctx.lineWidth = Math.max(1, radius / 110)
  ctx.stroke(path)
  ctx.restore()

  ctx.save()
  ctx.clip(path)
  const hx = cx - radius * 0.34
  const hy = cy - radius * 0.42
  const highlight = ctx.createRadialGradient(hx, hy, 0, hx, hy, radius * 0.55)
  highlight.addColorStop(0, `rgba(255, 255, 255, ${light ? 0.55 : 0.22})`)
  highlight.addColorStop(1, 'rgba(255, 255, 255, 0)')
  ctx.fillStyle = highlight
  ctx.fillRect(hx - radius * 0.55, hy - radius * 0.55, radius * 1.1, radius * 1.1)
  ctx.restore()
}

function drawLinks(
  ctx: CanvasRenderingContext2D,
  state: OrbRenderState,
  palette: PlasmaPalette,
  light: boolean,
  near: boolean
): void {
  const { level, links, motion, positions: p, projected: out } = state

  if (motion.links <= 0.02 || links.length === 0) {
    return
  }

  const max = LINK_DISTANCE * (1 + level * 0.45)
  ctx.lineWidth = 0.7 + level * 0.5

  for (const bucket of linkBuckets) {
    bucket.length = 0
  }

  for (let k = 0; k < links.length; k += 2) {
    const a = links[k] * 3
    const b = links[k + 1] * 3

    // A link crossing the equator belongs to the near side, in front of the glass.
    if (Math.max(out[a + 2], out[b + 2]) >= 0.5 !== near) {
      continue
    }

    const distance = Math.hypot(p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2])
    const depth = (out[a + 2] + out[b + 2]) / 2
    const strength = Math.max(0, 1 - distance / max) * (0.35 + depth * 0.65)
    const bucket = Math.min(ALPHA_BUCKETS - 1, Math.floor(strength * ALPHA_BUCKETS))
    linkBuckets[bucket].push(out[a], out[a + 1], out[b], out[b + 1])
  }

  for (let bucket = 0; bucket < ALPHA_BUCKETS; bucket += 1) {
    const segments = linkBuckets[bucket]

    if (segments.length === 0) {
      continue
    }

    const alpha = ((bucket + 1) / ALPHA_BUCKETS) * motion.links * (0.3 + level * 0.25) * (light ? 2.1 : 1)
    ctx.strokeStyle = `rgba(${near ? palette.front : palette.back}, ${alpha.toFixed(3)})`
    ctx.beginPath()

    for (let index = 0; index < segments.length; index += 4) {
      ctx.moveTo(segments[index], segments[index + 1])
      ctx.lineTo(segments[index + 2], segments[index + 3])
    }

    ctx.stroke()
  }
}

/**
 * Whether a projected point has fallen outside the glass body's silhouette.
 * The outline carries one radius per angular bin (see `ParticleOrb.traceOutline`),
 * so the bin the point falls in holds the cloud's edge in that direction.
 */
function outsideBody(cx: number, cy: number, state: OrbRenderState, x: number, y: number): boolean {
  const bins = state.outline.length

  if (bins === 0) {
    return false
  }

  const dx = x - cx
  const dy = y - cy
  const angle = Math.atan2(dy, dx) + Math.PI
  const bin = Math.min(bins - 1, Math.floor((angle / (Math.PI * 2)) * bins))

  return Math.hypot(dx, dy) > state.outline[bin] * 1.02
}

function drawParticles(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  state: OrbRenderState,
  palette: PlasmaPalette,
  light: boolean,
  near: boolean,
  radius: number
): void {
  const { count, level, motion, projected: out } = state
  const dot = Math.max(0.7, radius / 142) * (1 + level * 0.3)

  const alpha = near
    ? Math.min(1, (motion.brightness + level * 0.15) * (light ? 1.5 : 1))
    : motion.brightness * (light ? 0.9 : 0.55)

  if (near) {
    // Sparse bloom pass: only the brightest near points get a halo.
    ctx.save()
    ctx.shadowColor = `rgba(${palette.highlight}, ${light ? 0.25 : 0.7})`
    ctx.shadowBlur = dot * (light ? 3 : 6)
    ctx.fillStyle = `rgba(${palette.highlight}, ${light ? 0.14 : 0.24})`
    ctx.beginPath()

    for (let index = 0; index < count; index += 7) {
      const i3 = index * 3

      if (out[i3 + 2] < 0.65 || outsideBody(cx, cy, state, out[i3], out[i3 + 1])) {
        continue
      }

      ctx.moveTo(out[i3] + dot * 1.8, out[i3 + 1])
      ctx.arc(out[i3], out[i3 + 1], dot * 1.8, 0, Math.PI * 2)
    }

    ctx.fill()
    ctx.restore()
  }

  ctx.fillStyle = `rgba(${near ? palette.core : palette.back}, ${alpha.toFixed(3)})`
  ctx.beginPath()

  for (let index = 0; index < count; index += 1) {
    const i3 = index * 3
    const depth = out[i3 + 2]

    if (depth >= 0.5 !== near) {
      continue
    }

    // A dot projected past the glass outline would read as a speck floating
    // beside the orb instead of part of it. The outline smooths the cloud's
    // silhouette (and perspective pushes the nearest points outward), so a
    // handful of dots always landed outside it: drop those.
    if (outsideBody(cx, cy, state, out[i3], out[i3 + 1])) {
      continue
    }

    const size = dot * (0.55 + depth * 0.9)
    ctx.moveTo(out[i3] + size, out[i3 + 1])
    ctx.arc(out[i3], out[i3 + 1], size, 0, Math.PI * 2)
  }

  ctx.fill()
}

function drawElectrons(
  ctx: CanvasRenderingContext2D,
  state: OrbRenderState,
  palette: PlasmaPalette,
  light: boolean,
  radius: number
): void {
  const dot = Math.max(0.6, radius / 150) * (1 + state.level * 0.25)
  const out = state.projected

  for (const electron of state.electrons) {
    const a = electron.from * 3
    const b = electron.to * 3

    // Travelling particles remain on the visible face of the glass.
    if (Math.max(out[a + 2], out[b + 2]) < 0.5) {
      continue
    }

    const x = out[a] + (out[b] - out[a]) * electron.progress
    const y = out[a + 1] + (out[b + 1] - out[a + 1]) * electron.progress
    const tail = Math.min(0.18, electron.progress)
    const tx = out[a] + (out[b] - out[a]) * (electron.progress - tail)
    const ty = out[a + 1] + (out[b + 1] - out[a + 1]) * (electron.progress - tail)
    const trail = ctx.createLinearGradient(tx, ty, x, y)
    trail.addColorStop(0, `rgba(${palette.front}, 0)`)
    trail.addColorStop(1, `rgba(${palette.highlight}, ${light ? 0.55 : 0.85})`)
    ctx.save()
    ctx.strokeStyle = trail
    ctx.lineWidth = dot * 2.5
    ctx.shadowColor = `rgba(${palette.core}, ${light ? 0.35 : 0.75})`
    ctx.shadowBlur = dot * 5
    ctx.beginPath()
    ctx.moveTo(tx, ty)
    ctx.lineTo(x, y)
    ctx.stroke()
    ctx.restore()
    const glow = ctx.createRadialGradient(x, y, 0, x, y, dot * 6)
    glow.addColorStop(0, light ? `rgba(${palette.core}, 0.95)` : 'rgba(255, 255, 255, 0.95)')
    glow.addColorStop(0.3, `rgba(${palette.core}, 0.55)`)
    glow.addColorStop(1, `rgba(${palette.front}, 0)`)
    ctx.fillStyle = glow
    ctx.beginPath()
    ctx.arc(x, y, dot * 6, 0, Math.PI * 2)
    ctx.fill()
  }
}

/** Far network, glass, near network, then moving light. */
export function drawOrb(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  radius: number,
  palette: PlasmaPalette,
  surface: PlasmaSurface,
  time: number,
  state: OrbRenderState
): void {
  const light = surface === 'light'
  drawLinks(ctx, state, palette, light, false)
  drawParticles(ctx, cx, cy, state, palette, light, false, radius)
  ctx.save()
  ctx.globalCompositeOperation = 'source-over'
  drawBody(ctx, cx, cy, radius, state.outline, palette, light, state.level, time)
  ctx.restore()
  drawLinks(ctx, state, palette, light, true)
  drawParticles(ctx, cx, cy, state, palette, light, true, radius)
  drawElectrons(ctx, state, palette, light, radius)
}
