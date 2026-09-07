import './particle-field.css'

import { type CSSProperties, type ReactNode, useEffect, useMemo, useRef, useState } from 'react'

import { cn } from '@/lib/utils'

/**
 * Reusable float-up particle emitter. It owns the motion (rise + organic sway +
 * springy pop) and lifecycle (staggered burst, lifetime, cleanup); callers just
 * hand it a `glyph` (any element using `currentColor`) and `colors`, then place
 * it with `className` / `style`. See {@link VibeHearts} for the chat-hearts use.
 */

type Range = readonly [min: number, max: number]

const rand = ([min, max]: Range) => min + Math.random() * (max - min)
/** Sample a range along `t` (0→min, 1→max) — couples travel/lifetime to `life`. */
const lerp = ([min, max]: Range, t: number) => min + (max - min) * t

export interface ParticleFieldConfig {
  /** Particles per burst when `burst()` is called without a count. */
  count: number
  /** Window (ms) over which a burst releases its particles — small = one poof. */
  spawnWindowMs: number
  /** Glyph edge size (px), uniform. */
  size: Range
  /** Vertical travel before fade-out (% of field height), `life`-biased. */
  rise: Range
  /** Rise duration (ms), `life`-biased so short-lived particles also rise less. */
  duration: Range
  /** Sway amplitude each side of center (px), uniform. */
  swayAmp: Range
  /** Peak tilt into the sway (deg), uniform. */
  bank: Range
  /** Sway period (ms), uniform — independent of the rise so paths never repeat. */
  swayDuration: Range
  /** Cap on simultaneously-alive particles. */
  maxAlive: number
}

export const DEFAULT_PARTICLE_CONFIG: ParticleFieldConfig = {
  count: 12,
  spawnWindowMs: 550,
  size: [6, 13],
  rise: [6.75, 15.75],
  duration: [320, 700],
  swayAmp: [9, 24],
  bank: [7, 16],
  swayDuration: [1300, 2800],
  maxAlive: 200
}

export interface ParticleEmitter {
  /** Fire a burst (defaults to the field's configured `count`). */
  burst: (count?: number) => void
  /** Internal: field subscription. */
  subscribe: (fn: (count?: number) => void) => () => void
}

/** Create an emitter handle. `burst()` is safe to call from anywhere. */
export function createParticleEmitter(): ParticleEmitter {
  const listeners = new Set<(count?: number) => void>()

  return {
    burst: count => listeners.forEach(fn => fn(count)),
    subscribe: fn => {
      listeners.add(fn)

      return () => void listeners.delete(fn)
    }
  }
}

interface Particle {
  id: number
  leftPct: number
  size: number
  color: string
  delayMs: number
  durationMs: number
  rise: number
  swayAmp: number
  bank: number
  swayDurationMs: number
  swayDelayMs: number
}

let nextId = 1

function spawn(cfg: ParticleFieldConfig, colors: readonly string[]): Particle {
  // Short-lived particles fade out lower; a few live longer and rise higher.
  const life = Math.random() ** 1.7
  const swayDurationMs = Math.round(rand(cfg.swayDuration))

  return {
    id: nextId++,
    // Spread edge to edge across the lane, not clustered near center.
    leftPct: 4 + Math.random() * 92,
    size: rand(cfg.size),
    color: colors[Math.floor(Math.random() * colors.length)]!,
    delayMs: Math.round(Math.random() * 120),
    durationMs: Math.round(lerp(cfg.duration, life)),
    rise: lerp(cfg.rise, life),
    swayAmp: rand(cfg.swayAmp),
    bank: rand(cfg.bank),
    swayDurationMs,
    // Negative delay drops each particle in mid-swing (desynced phases).
    swayDelayMs: -Math.round(Math.random() * swayDurationMs)
  }
}

const prefersReducedMotion = () =>
  typeof window !== 'undefined' && Boolean(window.matchMedia?.('(prefers-reduced-motion: reduce)').matches)

export interface ParticleFieldProps {
  emitter: ParticleEmitter
  /** Any element that paints with `currentColor` (SVG, glyph, …). */
  glyph: ReactNode
  colors: readonly string[]
  config?: Partial<ParticleFieldConfig>
  className?: string
  style?: CSSProperties
}

export function ParticleField({ emitter, glyph, colors, config, className, style }: ParticleFieldProps) {
  const cfg = useMemo(() => ({ ...DEFAULT_PARTICLE_CONFIG, ...config }), [config])
  const [particles, setParticles] = useState<Particle[]>([])
  const timers = useRef<Set<ReturnType<typeof setTimeout>>>(new Set())

  useEffect(() => {
    const pool = timers.current
    const add = () => setParticles(prev => [...prev, spawn(cfg, colors)].slice(-cfg.maxAlive))

    // Release a burst across a tight window so it reads as one poof, each node
    // with its own random birth time; reduced motion gets a single flash.
    const onBurst = (count?: number) => {
      const n = Math.max(1, Math.min(cfg.maxAlive, Math.round(count ?? cfg.count)))

      if (prefersReducedMotion()) {
        add()

        return
      }

      for (let i = 0; i < n; i++) {
        const timer = setTimeout(() => {
          pool.delete(timer)
          add()
        }, Math.random() * cfg.spawnWindowMs)

        pool.add(timer)
      }
    }

    const unsubscribe = emitter.subscribe(onBurst)

    return () => {
      unsubscribe()
      pool.forEach(clearTimeout)
      pool.clear()
    }
  }, [cfg, colors, emitter])

  const remove = (id: number) => setParticles(prev => prev.filter(p => p.id !== id))

  if (particles.length === 0) {
    return null
  }

  return (
    <div aria-hidden className={cn('particle-field', className)} style={style}>
      {particles.map(p => (
        <span
          className="particle"
          key={p.id}
          // Retire on the RISE track only (sway is infinite, pop is shorter).
          onAnimationEnd={e => {
            if (e.animationName === 'particle-rise' || e.animationName === 'particle-flash') {
              remove(p.id)
            }
          }}
          style={
            {
              '--particle-left': `${p.leftPct}%`,
              '--particle-size': `${p.size}px`,
              '--particle-color': p.color,
              '--particle-delay': `${p.delayMs}ms`,
              '--particle-duration': `${p.durationMs}ms`,
              '--particle-rise': p.rise,
              '--particle-sway': `${p.swayAmp}px`,
              '--particle-bank': `${p.bank}deg`,
              '--particle-sway-duration': `${p.swayDurationMs}ms`,
              '--particle-sway-delay': `${p.swayDelayMs}ms`
            } as CSSProperties
          }
        >
          <span className="particle__sway">
            <span className="particle__glyph">{glyph}</span>
          </span>
        </span>
      ))}
    </div>
  )
}
