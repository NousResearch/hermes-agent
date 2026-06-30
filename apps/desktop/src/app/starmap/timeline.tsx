import { memo, useCallback, useEffect, useMemo, useRef } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { useTheme } from '@/themes/context'

import type { TimeAxis } from './time-axis'

interface TimelineProps {
  axis: TimeAxis
  // Colour for memory stars — matches the map's memory glyph.
  memoryColor?: string
  onScrub: (reveal: number) => void
  onTogglePlay: () => void
  playing: boolean
  revealStore: RevealSignal
  // Reveal positions (0–1) where rings spawn — drawn as anchor ticks.
  ringStops?: number[]
}

interface RevealSignal {
  get: () => number
  subscribe: (listener: (value: number) => void) => () => void
}

interface Star {
  delay: number
  duration: number
  kind: 'memory' | 'skill'
  leftPct: number
  opacity: number
  size: number
  topPct: number
}

const ACTIVE_MARKER_CLASS = 'opacity-100'
const INACTIVE_MARKER_CLASS = 'opacity-30'
// Busiest bucket gets this many stars; quieter ones scale down proportionally.
const MAX_STARS_PER_BUCKET = 7

// Deterministic PRNG (mulberry32) so a bucket's stars stay put across renders.
function rng(seed: number): () => number {
  let a = seed >>> 0

  return () => {
    a += 0x6d2b79f5
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)

    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// Scatter each time bucket's activity into stars: count ∝ events, split between
// skill- and memory-coloured stars, jittered within the bucket's horizontal slot
// and across the track height. Vertical placement is biased toward the midline
// (a triangular distribution) so stars cluster near the centre more often than
// the edges. A starmap timeline for a starmap.
function buildStars(axis: TimeAxis): Star[] {
  const n = Math.max(1, axis.buckets.length)
  const stars: Star[] = []

  axis.buckets.forEach((b, i) => {
    if (b.total === 0) {
      return
    }

    const intensity = axis.maxTotal > 0 ? b.total / axis.maxTotal : 0
    // sqrt curve so a single co-timed burst (a packed core ring) doesn't crush
    // every quieter bucket down to the 1-star floor and read as blank.
    const count = Math.max(1, Math.round(Math.sqrt(intensity) * MAX_STARS_PER_BUCKET))
    const skillCount = Math.round((b.skill / b.total) * count)
    const r = rng(i * 9973 + 7)
    const slot = 1 / n
    const center = (i + 0.5) / n

    for (let s = 0; s < count; s++) {
      const jitter = (r() - 0.5) * slot * 0.9
      // Average of two uniforms → triangular peak at 0.5, pulling stars toward
      // the midline more often while still reaching the edges occasionally.
      const vertical = (r() + r()) / 2

      stars.push({
        delay: r() * 3,
        duration: 2.4 + r() * 2.6,
        kind: s < skillCount ? 'skill' : 'memory',
        leftPct: Math.max(0, Math.min(1, center + jitter)) * 100,
        // Brighter, slightly larger stars are rarer.
        opacity: 0.5 + r() * 0.5,
        // Floor at 2px so a lone star in a quiet bucket still reads on black.
        size: 2 + Math.round(r() * r() * 2.2),
        topPct: 12 + vertical * 76
      })
    }
  })

  return stars
}

// Playback scrubber as a constellation: dim stars are the unrevealed future; a
// scanner sweep ignites them (bright + twinkling) left→right as the reveal
// advances. The bright layer is clipped by the reveal CSS var, so the rAF sweep
// in StarMap drives it with zero per-frame JS.
export const Timeline = memo(function Timeline({
  axis,
  memoryColor = 'var(--theme-secondary)',
  onScrub,
  onTogglePlay,
  playing,
  revealStore,
  ringStops = []
}: TimelineProps) {
  const trackRef = useRef<HTMLDivElement | null>(null)
  const draggingRef = useRef(false)
  const markerRefs = useRef<HTMLDivElement[]>([])
  // Star glow halos read as depth on a dark track but smear on a light one, so
  // the bloom is dark-mode only.
  const { resolvedMode } = useTheme()
  const glow = resolvedMode === 'dark'

  const stars = useMemo(() => buildStars(axis), [axis])

  const syncReveal = useCallback(
    (value: number) => {
      const reveal = Math.max(0, Math.min(1, value))
      const track = trackRef.current

      if (track) {
        track.style.setProperty('--starmap-reveal', String(reveal))
        track.setAttribute('aria-valuenow', String(Math.round(reveal * 100)))
      }

      ringStops.forEach((stop, i) => {
        const el = markerRefs.current[i]

        if (!el) {
          return
        }

        const active = stop <= reveal

        el.classList.toggle(ACTIVE_MARKER_CLASS, active)
        el.classList.toggle(INACTIVE_MARKER_CLASS, !active)
      })
    },
    [ringStops]
  )

  useEffect(() => revealStore.subscribe(syncReveal), [revealStore, syncReveal])

  useEffect(() => {
    markerRefs.current.length = ringStops.length
    syncReveal(revealStore.get())
  }, [revealStore, ringStops.length, syncReveal])

  const ratioAt = (clientX: number): number => {
    const rect = trackRef.current?.getBoundingClientRect()

    if (!rect || rect.width === 0) {
      return revealStore.get()
    }

    return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
  }

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    draggingRef.current = true
    e.currentTarget.setPointerCapture(e.pointerId)
    onScrub(ratioAt(e.clientX))
  }

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (draggingRef.current) {
      onScrub(ratioAt(e.clientX))
    }
  }

  const onPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    draggingRef.current = false

    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId)
    }
  }

  const colorFor = (kind: Star['kind']) => (kind === 'skill' ? 'var(--theme-primary)' : memoryColor)

  return (
    <div className="pointer-events-auto flex w-[28rem] max-w-full items-center gap-3 [-webkit-app-region:no-drag]">
      <style>{'@keyframes starmap-twinkle{0%,100%{opacity:var(--o,1)}50%{opacity:calc(var(--o,1) * 0.35)}}'}</style>

      <button
        aria-label={playing ? 'Pause' : 'Play timeline'}
        className="flex size-5 shrink-0 items-center justify-center text-foreground/75 transition-colors hover:text-foreground"
        onClick={onTogglePlay}
        type="button"
      >
        <Codicon name={playing ? 'debug-pause' : 'triangle-right'} size={playing ? '0.8rem' : '0.95rem'} />
      </button>

      <div
        aria-label="Timeline scrubber"
        aria-valuemax={100}
        aria-valuemin={0}
        aria-valuenow={Math.round(revealStore.get() * 100)}
        className="relative h-7 min-w-0 flex-1 cursor-pointer select-none touch-none"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        ref={trackRef}
        role="slider"
        style={{ '--starmap-reveal': revealStore.get() } as React.CSSProperties}
        tabIndex={0}
      >
        {/* Dashed midline — a faint horizontal axis the stars ride over. */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-x-0 top-1/2 -translate-y-1/2 border-t border-dashed border-foreground/5"
        />

        {/* Dim constellation — the unrevealed future. */}
        <div aria-hidden className="absolute inset-0">
          {stars.map((star, i) => (
            <div
              className="absolute -translate-x-1/2 -translate-y-1/2 rounded-full"
              key={i}
              style={{
                backgroundColor: colorFor(star.kind),
                height: star.size,
                left: `${star.leftPct}%`,
                opacity: 0.22,
                top: `${star.topPct}%`,
                width: star.size
              }}
            />
          ))}
        </div>

        {/* Ignited constellation — bright + twinkling, clipped to the reveal. */}
        <div
          aria-hidden
          className="absolute inset-0"
          style={{ clipPath: 'inset(0 calc((1 - var(--starmap-reveal, 1)) * 100%) 0 0)' }}
        >
          {stars.map((star, i) => {
            const color = colorFor(star.kind)

            return (
              <div
                className="absolute -translate-x-1/2 -translate-y-1/2 rounded-full"
                key={i}
                style={
                  {
                    '--o': star.opacity,
                    animation: `starmap-twinkle ${star.duration}s ease-in-out ${star.delay}s infinite`,
                    backgroundColor: color,
                    boxShadow: glow ? `0 0 ${star.size + 1}px ${color}` : 'none',
                    height: star.size,
                    left: `${star.leftPct}%`,
                    opacity: star.opacity,
                    top: `${star.topPct}%`,
                    width: star.size
                  } as React.CSSProperties
                }
              />
            )
          })}
        </div>

        {/* Ring-spawn anchor ticks — small bright stars that light up on pass. */}
        {ringStops.map((stop, i) => (
          <div
            aria-hidden
            className={`pointer-events-none absolute top-1/2 size-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-[var(--theme-primary)] ${glow ? 'shadow-[0_0_4px_var(--theme-primary)]' : ''} ${INACTIVE_MARKER_CLASS}`}
            key={i}
            ref={el => {
              if (el) {
                markerRefs.current[i] = el
              }
            }}
            style={{ left: `${stop * 100}%` }}
          />
        ))}

        {/* Playhead — a thin white sweep line. */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-y-0 w-px -translate-x-1/2 bg-foreground"
          style={{ left: 'calc(var(--starmap-reveal, 1) * 100%)' }}
        />
      </div>
    </div>
  )
})
