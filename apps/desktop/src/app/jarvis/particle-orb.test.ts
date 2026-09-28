import { describe, expect, it, vi } from 'vitest'

import { orbMotionTargets, ParticleOrb, type ParticleOrbInput } from './particle-orb'
import { drawOrb } from './particle-orb-render'
import { plasmaPalette } from './plasma'
import type { PlasmaTone } from './plasma'

const TONES: PlasmaTone[] = ['idle', 'listening', 'working', 'speaking', 'approval', 'success', 'error']

function run(orb: ParticleOrb, input: ParticleOrbInput, seconds: number, fps = 60) {
  for (let frame = 0; frame < seconds * fps; frame += 1) {
    orb.step(1 / fps, input)
  }
}

function meanRadius(orb: ParticleOrb): number {
  let total = 0

  for (let index = 0; index < orb.count; index += 1) {
    const i3 = index * 3
    total += Math.hypot(orb.positions[i3], orb.positions[i3 + 1], orb.positions[i3 + 2])
  }

  return total / orb.count
}

/** Spread of the particles' distance from the centre: 0 for a perfect shell. */
function roughness(orb: ParticleOrb): number {
  const radii: number[] = []

  for (let index = 0; index < orb.count; index += 1) {
    const i3 = index * 3
    radii.push(Math.hypot(orb.positions[i3], orb.positions[i3 + 1], orb.positions[i3 + 2]))
  }

  const mean = radii.reduce((sum, r) => sum + r, 0) / radii.length

  return Math.sqrt(radii.reduce((sum, r) => sum + (r - mean) ** 2, 0) / radii.length) / mean
}

describe('ParticleOrb', () => {
  it('changes shape with the voice: a loud reply bends the shell further from round than rest', () => {
    const resting = new ParticleOrb(300)
    const talking = new ParticleOrb(300)

    run(resting, { level: 0, signal: 0, tone: 'idle' }, 6)
    run(talking, { level: 0.9, signal: 0, tone: 'speaking' }, 6)

    expect(talking.motion.morph).toBeGreaterThan(resting.motion.morph * 3)
    expect(roughness(talking)).toBeGreaterThan(roughness(resting))
  })

  it('draws the cloud in while working and lets it rest wider when idle', () => {
    const idle = new ParticleOrb(200)
    const working = new ParticleOrb(200)

    run(idle, { level: 0, signal: 0, tone: 'idle' }, 8)
    run(working, { level: 0, signal: 0.7, tone: 'working' }, 8)

    expect(meanRadius(working)).toBeLessThan(meanRadius(idle))
    expect(orbMotionTargets('working').radius).toBeLessThan(orbMotionTargets('idle').radius)
  })

  it('sends electrons along links only while working', () => {
    const idle = new ParticleOrb(200)
    const working = new ParticleOrb(200)
    let sawElectron = false

    for (let frame = 0; frame < 6 * 60; frame += 1) {
      idle.step(1 / 60, { level: 0, signal: 0, tone: 'idle' })
      working.step(1 / 60, { level: 0, signal: 0.7, tone: 'working' })
      sawElectron ||= working.electronCount > 0
      expect(idle.electronCount).toBe(0)
    }

    expect(sawElectron).toBe(true)
  })

  it('breathes outward with the measured voice level', () => {
    const quiet = new ParticleOrb(200)
    const loud = new ParticleOrb(200)

    run(quiet, { level: 0, signal: 0, tone: 'speaking' }, 4)
    run(loud, { level: 0.9, signal: 0, tone: 'speaking' }, 4)

    expect(meanRadius(loud)).toBeGreaterThan(meanRadius(quiet))
  })

  it('settles the same way at 30 Hz and 144 Hz', () => {
    const slow = new ParticleOrb(200)
    const fast = new ParticleOrb(200)

    run(slow, { level: 0, signal: 0, tone: 'listening' }, 6, 30)
    run(fast, { level: 0, signal: 0, tone: 'listening' }, 6, 144)

    expect(Math.abs(slow.motion.radius - fast.motion.radius)).toBeLessThan(0.01)
  })

  it('stays bounded through every state and a long stall', () => {
    const orb = new ParticleOrb(200)

    for (const tone of TONES) {
      run(orb, { level: 1, signal: 1, tone }, 2)
      // A backgrounded tab resumes with a huge gap.
      orb.step(30, { level: 1, signal: 1, tone })
    }

    for (const value of orb.positions) {
      expect(Number.isFinite(value)).toBe(true)
      expect(Math.abs(value)).toBeLessThan(3)
    }
  })

  it('wraps the glassy body round the cloud as one closed edge that bends with the shape', () => {
    const outlineOf = (tone: PlasmaTone, level: number) => {
      const orb = new ParticleOrb(340)
      run(orb, { level, signal: 0, tone }, 4)
      orb.project(200, 200, 100)
      const radii = [...orb.outline]
      const mean = radii.reduce((sum, r) => sum + r, 0) / radii.length

      return { mean, radii, spread: (Math.max(...radii) - Math.min(...radii)) / mean }
    }

    const resting = outlineOf('idle', 0)
    const talking = outlineOf('speaking', 0.9)

    // No holes or spikes: every direction reaches the cloud's edge.
    for (const { mean, radii } of [resting, talking]) {
      for (const r of radii) {
        expect(r).toBeGreaterThan(mean * 0.6)
        expect(r).toBeLessThan(mean * 1.6)
      }
    }

    expect(talking.spread).toBeGreaterThan(resting.spread)
  })

  it('settles directly to a distinct still shape and clamps the audio brightness input', () => {
    const idle = new ParticleOrb(240)
    const working = new ParticleOrb(240)
    const talking = new ParticleOrb(240)
    const clamped = new ParticleOrb(240)

    idle.settle({ level: 0, signal: 0, tone: 'idle' })
    working.settle({ level: 0, signal: 0.7, tone: 'working' })
    talking.settle({ level: 1, signal: 0, tone: 'speaking' })
    clamped.settle({ level: 5, signal: 0, tone: 'speaking' })

    expect(working.motion.radius).toBeLessThan(idle.motion.radius)
    expect(meanRadius(working)).toBeLessThan(meanRadius(idle))
    expect(talking.motion.morph).toBeGreaterThan(idle.motion.morph)
    expect(roughness(talking)).toBeGreaterThan(roughness(idle))
    expect([...clamped.positions]).toEqual([...talking.positions])
    expect(talking.electronCount).toBe(0)
  })

  it('keeps several working electrons in flight and lets them expire after work stops', () => {
    const orb = new ParticleOrb(340)
    let peak = 0

    for (let frame = 0; frame < 8 * 60; frame += 1) {
      orb.step(1 / 60, { level: 0, signal: 0.7, tone: 'working' })
      peak = Math.max(peak, orb.electronCount)
    }

    expect(peak).toBeGreaterThan(3)
    run(orb, { level: 0, signal: 0, tone: 'idle' }, 4)
    expect(orb.electronCount).toBe(0)
  })

  it('draws far links before the glass and near links after it', () => {
    const operations: string[] = []

    class TestPath {
      moveTo() {}
      quadraticCurveTo() {}
      closePath() {}
    }
    vi.stubGlobal('Path2D', TestPath)
    const gradient = () => ({ addColorStop() {} })

    const ctx = {
      arc() {},
      beginPath() {},
      clip() {},
      createConicGradient: gradient,
      createLinearGradient: gradient,
      createRadialGradient: gradient,
      fill(path?: TestPath) {
        operations.push(path ? 'body' : 'particles')
      },
      fillRect() {},
      lineTo() {},
      moveTo() {},
      restore() {},
      save() {},
      stroke(path?: TestPath) {
        operations.push(path ? 'rim' : 'link')
      }
    } as unknown as CanvasRenderingContext2D

    const orb = new ParticleOrb(4)

    const state = {
      count: 4,
      electrons: [],
      level: 0,
      links: [0, 1, 2, 3],
      motion: orb.motion,
      outline: new Float32Array(48).fill(30),
      positions: new Float32Array([0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0.1, 0, 0]),
      projected: new Float32Array([0, 0, 0.2, 10, 0, 0.2, 0, 10, 0.8, 10, 10, 0.8])
    }

    try {
      drawOrb(ctx, 5, 5, 30, plasmaPalette('idle'), 'dark', 0, state)
    } finally {
      vi.unstubAllGlobals()
    }

    const body = operations.indexOf('body')
    expect(body).toBeGreaterThan(operations.indexOf('link'))
    expect(operations.lastIndexOf('link')).toBeGreaterThan(body)
  })

  it('never paints a dot past the glass outline, so none floats beside the orb', () => {
    const arcs: { x: number; y: number }[] = []

    class TestPath {
      moveTo() {}
      quadraticCurveTo() {}
      closePath() {}
    }
    vi.stubGlobal('Path2D', TestPath)
    const gradient = () => ({ addColorStop() {} })

    const ctx = {
      arc(x: number, y: number) {
        arcs.push({ x, y })
      },
      beginPath() {},
      clip() {},
      createConicGradient: gradient,
      createLinearGradient: gradient,
      createRadialGradient: gradient,
      fill() {},
      fillRect() {},
      lineTo() {},
      moveTo() {},
      restore() {},
      save() {},
      stroke() {}
    } as unknown as CanvasRenderingContext2D

    // One dot sits well outside the silhouette, three inside it.
    const orb = new ParticleOrb(4)

    const state = {
      count: 4,
      electrons: [],
      level: 0,
      links: [],
      motion: orb.motion,
      outline: new Float32Array(48).fill(10),
      positions: new Float32Array(12),
      projected: new Float32Array([0, 0, 0.8, 6, 6, 0.8, 0, -5, 0.8, 40, 0, 0.8])
    }

    try {
      drawOrb(ctx, 0, 0, 10, plasmaPalette('idle'), 'light', 0, state)
    } finally {
      vi.unstubAllGlobals()
    }

    expect(arcs.length).toBeGreaterThan(0)

    for (const dot of arcs) {
      expect(Math.hypot(dot.x, dot.y)).toBeLessThanOrEqual(10)
    }
  })
})
