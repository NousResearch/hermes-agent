/**
 * The Jarvis orb's particle network: a hollow sphere of drifting particles,
 * wired together by faint links between the ones that drift close enough,
 * with bright "electrons" travelling along those links while Jarvis works.
 *
 * Every state has its own body language — at rest the cloud is wide and slow,
 * listening draws it in, working packs it tight with dense links and
 * electrons, speaking lets the measured voice level breathe it outward — and a
 * change of state gives the cloud a short tumble so the change is felt, not
 * just recoloured.
 *
 * Canvas 2D only (docs/product/AI_EVOLUTION_JARVIS_DESIGN.md §6: no WebGL on
 * the light path). The simulation is framerate-independent: `step(dt)` eases
 * with `1 - exp(-k·dt)`, so a 144 Hz panel and a throttled 30 Hz one settle
 * the same way.
 */

import { drawOrb } from './particle-orb-render'
import type { PlasmaPalette, PlasmaSurface, PlasmaTone } from './plasma'

export interface OrbMotionTargets {
  /** Shell radius as a fraction of the orb radius. */
  radius: number
  /** Wander strength. */
  speed: number
  /** Particle opacity. */
  brightness: number
  /** How visible the proximity links are, 0…1. */
  links: number
  /** Whether electrons are sent along links, 0…1. */
  electrons: number
  /** How far the shell departs from a sphere (fraction of the radius). */
  morph: number
  /** How fast the shape's waves travel over the surface. */
  wave: number
}

// Each state moves differently: at rest the shell slowly flows, listening
// lets the voice push it out of round, speaking sends quick waves across it,
// working folds it into slow rotating lobes, an error withdraws it.
const TARGETS: Record<PlasmaTone, OrbMotionTargets> = {
  idle: { brightness: 0.6, electrons: 0, links: 0.42, morph: 0.09, radius: 1, speed: 0.25, wave: 0.45 },
  listening: { brightness: 0.78, electrons: 0, links: 0.62, morph: 0.12, radius: 0.84, speed: 0.4, wave: 0.76 },
  working: { brightness: 0.84, electrons: 1, links: 1, morph: 0.17, radius: 0.64, speed: 0.68, wave: 0.88 },
  speaking: { brightness: 0.88, electrons: 0, links: 0.88, morph: 0.1, radius: 0.74, speed: 0.3, wave: 1.16 },
  approval: { brightness: 0.78, electrons: 0.4, links: 0.72, morph: 0.08, radius: 0.8, speed: 0.2, wave: 0.5 },
  success: { brightness: 0.66, electrons: 0, links: 0.45, morph: 0.07, radius: 0.92, speed: 0.22, wave: 0.4 },
  // Withdrawn and slow: unmistakably not "about to answer".
  error: { brightness: 0.42, electrons: 0, links: 0.18, morph: 0.03, radius: 0.7, speed: 0.1, wave: 0.2 }
}

/** How much a measured voice level adds to the shape's departure from round. */
const VOICE_MORPH: Partial<Record<PlasmaTone, number>> = { listening: 0.3, speaking: 0.35 }

export function orbMotionTargets(tone: PlasmaTone): OrbMotionTargets {
  return TARGETS[tone]
}

export interface ParticleOrbInput {
  tone: PlasmaTone
  /** Measured audio level, 0…1. */
  level: number
  /** Backend task pressure, 0…1. */
  signal: number
}

interface Electron {
  from: number
  to: number
  progress: number
  speed: number
}

/** Deterministic PRNG, so every orb starts from the same cloud. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0

  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)

    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const MAX_ELECTRONS = 8
const ELECTRON_INTERVAL_S = 0.22
const LINK_REFRESH_S = 0.1
/** Link distance in orb-radius units. */
const LINK_DISTANCE = 0.3
/** Camera distance for the perspective projection, in orb-radius units. */
const CAMERA = 3.4
/** Angular resolution of the glassy body's outline. */
const OUTLINE_BINS = 48
const OUTLINE_SMOOTHING = 4

function ease(current: number, target: number, rate: number, dt: number): number {
  return current + (target - current) * (1 - Math.exp(-rate * dt))
}

export class ParticleOrb {
  readonly count: number
  readonly positions: Float32Array
  private readonly velocities: Float32Array
  private readonly phases: Float32Array
  /** Each particle's depth inside the shell, 0.9…1: a shell with some body, thin enough to show its shape. */
  private readonly depths: Float32Array
  private readonly projected: Float32Array
  /** The projected cloud's silhouette: a radius per angular bin, around the last projection centre. */
  private readonly outlineRadii = new Float32Array(OUTLINE_BINS)
  private readonly outlineScratch = new Float32Array(OUTLINE_BINS)
  private readonly random: () => number
  /** Every `stride`-th particle takes part in links: O(n²) stays bounded. */
  private readonly stride: number

  private time = 0
  /** Accumulated wave phase: integrating the speed keeps a speed change smooth. */
  private phase = 0
  private tone: PlasmaTone = 'idle'
  private current: OrbMotionTargets = { ...TARGETS.idle }
  private level = 0
  private yaw = 0
  private spinX = 0
  private spinZ = 0
  private tumble = 0
  private lastElectron = 0
  private readonly electrons: Electron[] = []
  private readonly links: number[] = []
  private linkElapsed = LINK_REFRESH_S

  constructor(count = 420, seed = 0x0a1e7) {
    this.count = count
    this.random = mulberry32(seed)
    this.positions = new Float32Array(count * 3)
    this.velocities = new Float32Array(count * 3)
    this.phases = new Float32Array(count)
    this.depths = new Float32Array(count)
    this.projected = new Float32Array(count * 3)
    this.stride = Math.max(1, Math.round(count / 240))

    for (let index = 0; index < count; index += 1) {
      const z = this.random() * 2 - 1
      const angle = this.random() * Math.PI * 2
      const ring = Math.sqrt(1 - z * z)
      const r = 0.62 + this.random() * 0.4
      const i3 = index * 3
      this.positions[i3] = ring * Math.cos(angle) * r
      this.positions[i3 + 1] = z * r
      this.positions[i3 + 2] = ring * Math.sin(angle) * r
      this.phases[index] = this.random() * 1000
      this.depths[index] = 0.9 + this.random() * 0.1
    }
  }

  /** The eased motion values, for tests and for callers that tint around the orb. */
  get motion(): Readonly<OrbMotionTargets> {
    return this.current
  }

  get electronCount(): number {
    return this.electrons.length
  }

  /** Apply an input as a single still state, without advancing the animation clock. */
  settle({ level, signal, tone }: ParticleOrbInput): void {
    const pressure = Math.min(1, Math.max(0, signal))
    this.level = Math.min(1, Math.max(0, level))
    this.tone = tone
    const target = TARGETS[tone]
    this.current = {
      ...target,
      links: Math.min(1, target.links + pressure * 0.2),
      morph: target.morph + (VOICE_MORPH[tone] ?? 0) * this.level,
      radius: target.radius * (1 - pressure * 0.08),
      speed: target.speed + pressure * 0.2
    }

    for (let index = 0; index < this.count; index += 1) {
      const i3 = index * 3
      const x = this.positions[i3]
      const y = this.positions[i3 + 1]
      const z = this.positions[i3 + 2]
      const distance = Math.hypot(x, y, z) || 1
      const nx = x / distance
      const ny = y / distance
      const nz = z / distance
      const shape = Math.sin(nx * 4.4 + this.phases[index]) * 0.65 + Math.cos(ny * 5.1 - nz * 3.2) * 0.35
      const shell = this.current.radius * (1 + this.current.morph * shape) * this.depths[index] + this.level * 0.06
      this.positions[i3] = nx * shell
      this.positions[i3 + 1] = ny * shell
      this.positions[i3 + 2] = nz * shell
    }

    this.velocities.fill(0)
    this.electrons.length = 0
    this.updateLinks()
    this.linkElapsed = 0
  }

  step(dtSeconds: number, { level, signal, tone }: ParticleOrbInput): void {
    // A long gap (tab hidden, debugger) must not fling the cloud apart.
    const dt = Math.min(Math.max(dtSeconds, 0), 0.05)

    if (dt === 0) {
      return
    }

    this.time += dt
    const t = this.time

    if (tone !== this.tone) {
      this.tone = tone
      this.tumble = 1
    }

    const target = TARGETS[tone]
    // Task pressure tightens and quickens the cloud a little on top of its state.
    const pressure = Math.min(1, Math.max(0, signal))

    // A raw meter flickers; the cloud should breathe.
    this.level = ease(this.level, Math.min(1, Math.max(0, level)), 10, dt)

    this.current = {
      brightness: ease(this.current.brightness, target.brightness, 5, dt),
      electrons: ease(this.current.electrons, target.electrons, 4, dt),
      links: ease(this.current.links, Math.min(1, target.links + pressure * 0.2), 4, dt),
      // Shape follows the voice quickly but not instantly: it flows, never jumps.
      morph: ease(this.current.morph, target.morph + (VOICE_MORPH[tone] ?? 0) * this.level, 3.2, dt),
      radius: ease(this.current.radius, target.radius * (1 - pressure * 0.08), 1.4, dt),
      speed: ease(this.current.speed, target.speed + pressure * 0.2, 2.2, dt),
      wave: ease(this.current.wave, target.wave, 2.4, dt)
    }

    this.phase += dt * this.current.wave

    this.tumble *= Math.exp(-1.1 * dt)
    this.yaw += dt * (0.1 + this.current.speed * 0.35 + this.tumble * 1.4)
    this.spinX += dt * this.tumble * 0.9 * Math.sin(t * 1.7)
    this.spinZ += dt * this.tumble * 0.5 * Math.cos(t * 1.3)

    const { morph, radius, speed } = this.current
    const audio = this.level
    const speaking = tone === 'speaking'
    // Near-critical damping for the shell spring below: the cloud settles into
    // each new shape without wobbling past it.
    const damping = Math.exp(-5 * dt)
    const p = this.positions
    const v = this.velocities
    const w = this.phase
    // Three wave axes that drift slowly, so the lobes wander over the surface.
    const a1 = [Math.cos(t * 0.13), Math.sin(t * 0.13), 0.3]
    const a2 = [0.4, Math.cos(t * 0.11 + 2), Math.sin(t * 0.11 + 2)]
    const a3 = [Math.sin(t * 0.09 + 4), 0.2, Math.cos(t * 0.09 + 4)]

    for (let index = 0; index < this.count; index += 1) {
      const i3 = index * 3
      const phase = this.phases[index]
      const x = p[i3]
      const y = p[i3 + 1]
      const z = p[i3 + 2]
      const wander = 0.25 * speed * dt

      v[i3] += Math.sin(t * 0.5 + phase + y * 1.6) * wander
      v[i3 + 1] += Math.cos(t * 0.6 + phase * 1.3 + z * 1.6) * wander
      v[i3 + 2] += Math.sin(t * 0.55 + phase * 0.7 + x * 1.6) * wander

      const distance = Math.hypot(x, y, z) || 0.001
      const nx = x / distance
      const ny = y / distance
      const nz = z / distance

      // The shell's radius in this particle's direction: a sphere pushed in and
      // out by three travelling waves.
      // Low frequencies only: a soft, liquid silhouette rather than a rock.
      const shape =
        0.6 * Math.sin(1.4 * (nx * a1[0] + ny * a1[1] + nz * a1[2]) * Math.PI + w * 2.4) +
        0.3 * Math.sin(2.2 * (nx * a2[0] + ny * a2[1] + nz * a2[2]) * Math.PI - w * 3.1 + 1.7) +
        0.1 * Math.sin(3 * (nx * a3[0] + ny * a3[1] + nz * a3[2]) * Math.PI + w * 1.7 + 4.1)

      const shell = radius * (1 + morph * shape) * this.depths[index]
      // A spring onto the shell, so the whole cloud takes the new shape together.
      let radial = (shell - distance) * 10

      radial += audio * 0.6

      if (speaking && audio > 0.05) {
        radial += Math.sin(t * 8 + phase) * audio * 0.8
      }

      v[i3] += nx * radial * dt
      v[i3 + 1] += ny * radial * dt
      v[i3 + 2] += nz * radial * dt

      v[i3] *= damping
      v[i3 + 1] *= damping
      v[i3 + 2] *= damping

      p[i3] = x + v[i3] * dt
      p[i3 + 1] = y + v[i3 + 1] * dt
      p[i3 + 2] = z + v[i3 + 2] * dt
    }

    this.linkElapsed += dt

    if (this.linkElapsed >= LINK_REFRESH_S) {
      this.updateLinks()
      this.linkElapsed = 0
    }

    this.updateElectrons(dt)
  }

  private updateLinks(): void {
    const p = this.positions
    const max = LINK_DISTANCE * (1 + this.level * 0.45)
    const maxSq = max * max
    const links = this.links
    links.length = 0

    if (this.current.links > 0.02) {
      for (let i = 0; i < this.count; i += this.stride) {
        const i3 = i * 3

        for (let j = i + this.stride; j < this.count; j += this.stride) {
          const j3 = j * 3
          const dx = p[j3] - p[i3]
          const dy = p[j3 + 1] - p[i3 + 1]
          const dz = p[j3 + 2] - p[i3 + 2]

          if (dx * dx + dy * dy + dz * dz < maxSq) {
            links.push(i, j)
          }
        }
      }
    }
  }

  private updateElectrons(dt: number): void {
    for (let index = this.electrons.length - 1; index >= 0; index -= 1) {
      const electron = this.electrons[index]
      electron.progress += electron.speed * dt

      if (electron.progress >= 1) {
        this.electrons.splice(index, 1)
      }
    }

    if (
      this.current.electrons > 0.5 &&
      this.links.length > 0 &&
      this.electrons.length < MAX_ELECTRONS &&
      this.time - this.lastElectron > ELECTRON_INTERVAL_S / this.current.electrons
    ) {
      const pair = Math.floor(this.random() * (this.links.length / 2)) * 2

      this.electrons.push({
        from: this.links[pair],
        progress: 0,
        speed: 0.5 + this.random() * 0.4,
        to: this.links[pair + 1]
      })
      this.lastElectron = this.time
    }
  }

  /**
   * Project the cloud onto the screen around `(cx, cy)` with `radius` CSS
   * pixels per orb unit, and trace its silhouette for the body.
   */
  project(cx: number, cy: number, radius: number): void {
    const p = this.positions
    const out = this.projected
    const cosY = Math.cos(this.yaw)
    const sinY = Math.sin(this.yaw)
    const pitch = 0.3 + this.spinX
    const cosP = Math.cos(pitch)
    const sinP = Math.sin(pitch)
    const cosR = Math.cos(this.spinZ)
    const sinR = Math.sin(this.spinZ)

    for (let index = 0; index < this.count; index += 1) {
      const i3 = index * 3
      const x0 = p[i3] * cosY + p[i3 + 2] * sinY
      const z0 = -p[i3] * sinY + p[i3 + 2] * cosY
      const y1 = p[i3 + 1] * cosP - z0 * sinP
      const z1 = p[i3 + 1] * sinP + z0 * cosP
      const x2 = x0 * cosR - y1 * sinR
      const y2 = x0 * sinR + y1 * cosR
      const perspective = CAMERA / (CAMERA - z1)

      out[i3] = cx + x2 * radius * perspective
      out[i3 + 1] = cy + y2 * radius * perspective
      // Depth 0 (far) … 1 (near).
      out[i3 + 2] = Math.min(1, Math.max(0, (z1 + 1.2) / 2.4))
    }

    this.traceOutline(cx, cy)
  }

  /**
   * The silhouette's radius in each angular bin, in CSS pixels, as of the
   * last `project()`. The body is filled inside it, so it bends with the cloud.
   */
  get outline(): Readonly<Float32Array> {
    return this.outlineRadii
  }

  private traceOutline(cx: number, cy: number): void {
    const out = this.projected
    const radii = this.outlineRadii
    const scratch = this.outlineScratch
    radii.fill(0)

    for (let index = 0; index < this.count; index += 1) {
      const i3 = index * 3
      const dx = out[i3] - cx
      const dy = out[i3 + 1] - cy
      const angle = Math.atan2(dy, dx) + Math.PI
      const bin = Math.min(OUTLINE_BINS - 1, Math.floor((angle / (Math.PI * 2)) * OUTLINE_BINS))
      radii[bin] = Math.max(radii[bin], Math.hypot(dx, dy))
    }

    // Too few particles in a bin under-reads the edge: blur round the circle
    // (it wraps) toward the neighbours, keeping the higher of the two so the
    // body never shrinks inside the dots that define it.
    for (let pass = 0; pass < OUTLINE_SMOOTHING; pass += 1) {
      for (let bin = 0; bin < OUTLINE_BINS; bin += 1) {
        const before = radii[(bin + OUTLINE_BINS - 1) % OUTLINE_BINS]
        const after = radii[(bin + 1) % OUTLINE_BINS]
        const blurred = before * 0.25 + radii[bin] * 0.5 + after * 0.25
        scratch[bin] = pass === 0 ? Math.max(blurred, radii[bin] * 0.96) : blurred
      }

      radii.set(scratch)
    }
  }

  /** Paint the projected cloud around its glass body. */
  draw(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    radius: number,
    palette: PlasmaPalette,
    surface: PlasmaSurface = 'dark',
    time = 0
  ): void {
    this.project(cx, cy, radius)
    drawOrb(ctx, cx, cy, radius, palette, surface, time, {
      count: this.count,
      electrons: this.electrons,
      level: this.level,
      links: this.links,
      motion: this.current,
      outline: this.outlineRadii,
      positions: this.positions,
      projected: this.projected
    })
  }
}
