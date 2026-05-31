import { type ComponentProps, useEffect, useRef } from 'react'

import { cn } from '@/lib/utils'

export const LOADER_TYPES = [
  'original-thinking',
  'thinking-five',
  'thinking-nine',
  'rose-orbit',
  'rose-curve',
  'rose-two',
  'rose-three',
  'rose-four',
  'lissajous-drift',
  'lemniscate-bloom',
  'hypotrochoid-loop',
  'three-petal-spiral',
  'four-petal-spiral',
  'five-petal-spiral',
  'six-petal-spiral',
  'butterfly-phase',
  'cardioid-glow',
  'cardioid-heart',
  'heart-wave',
  'spiral-search',
  'fourier-flow'
] as const

export type LoaderType = (typeof LOADER_TYPES)[number]

interface Point {
  x: number
  y: number
}

interface LoaderCurve {
  durationMs: number
  name: string
  particleCount: number
  point: (progress: number, detailScale: number) => Point
  pulseDurationMs: number
  rotate: boolean
  rotationDurationMs: number
  strokeWidth: number
  trailSpan: number
}

interface LoaderProps extends Omit<ComponentProps<'div'>, 'children'> {
  label?: string
  pathSteps?: number
  strokeScale?: number
  type?: LoaderType
}

interface BaseCurveOptions extends Pick<
  LoaderCurve,
  'durationMs' | 'particleCount' | 'pulseDurationMs' | 'strokeWidth' | 'trailSpan'
> {
  point?: LoaderCurve['point']
  rotate?: boolean
  rotationDurationMs?: number
}

const TWO_PI = Math.PI * 2

const LOADER_CURVES: Record<LoaderType, LoaderCurve> = {
  'original-thinking': thinkingCurve('Original Thinking', 7, {
    durationMs: 4600,
    particleCount: 64,
    pulseDurationMs: 4200,
    rotationDurationMs: 28000,
    trailSpan: 0.38
  }),
  'thinking-five': thinkingCurve('Thinking Five', 5, {
    durationMs: 4600,
    particleCount: 62,
    pulseDurationMs: 4200,
    rotationDurationMs: 28000,
    trailSpan: 0.38
  }),
  'thinking-nine': thinkingCurve('Thinking Nine', 9, {
    durationMs: 4700,
    particleCount: 68,
    pulseDurationMs: 4200,
    rotationDurationMs: 30000,
    trailSpan: 0.39
  }),
  'rose-orbit': {
    ...baseCurve('Rose Orbit', {
      durationMs: 5200,
      particleCount: 72,
      pulseDurationMs: 4600,
      rotate: true,
      rotationDurationMs: 28000,
      strokeWidth: 5.2,
      trailSpan: 0.42
    }),
    point(progress, detailScale) {
      const t = progress * TWO_PI
      const r = 7 - 2.7 * detailScale * Math.cos(7 * t)

      return {
        x: 50 + Math.cos(t) * r * 3.9,
        y: 50 + Math.sin(t) * r * 3.9
      }
    }
  },
  'rose-curve': roseCurve('Rose Curve', 5, {
    durationMs: 5400,
    particleCount: 78,
    pulseDurationMs: 4600,
    strokeWidth: 4.5,
    trailSpan: 0.32
  }),
  'rose-two': roseCurve('Rose Two', 2, {
    durationMs: 5200,
    particleCount: 74,
    pulseDurationMs: 4300,
    strokeWidth: 4.6,
    trailSpan: 0.3
  }),
  'rose-three': roseCurve('Rose Three', 3, {
    durationMs: 5300,
    particleCount: 76,
    pulseDurationMs: 4400,
    strokeWidth: 4.6,
    trailSpan: 0.31
  }),
  'rose-four': roseCurve('Rose Four', 4, {
    durationMs: 5400,
    particleCount: 78,
    pulseDurationMs: 4500,
    strokeWidth: 4.6,
    trailSpan: 0.32
  }),
  'lissajous-drift': {
    ...baseCurve('Lissajous Drift', {
      durationMs: 6000,
      particleCount: 68,
      pulseDurationMs: 5400,
      strokeWidth: 4.7,
      trailSpan: 0.34
    }),
    point(progress, detailScale) {
      const t = progress * TWO_PI
      const amp = 24 + detailScale * 6

      return {
        x: 50 + Math.sin(3 * t + 1.57) * amp,
        y: 50 + Math.sin(4 * t) * (amp * 0.92)
      }
    }
  },
  'lemniscate-bloom': {
    ...baseCurve('Lemniscate Bloom', {
      durationMs: 5600,
      particleCount: 70,
      pulseDurationMs: 5000,
      rotationDurationMs: 34000,
      strokeWidth: 4.8,
      trailSpan: 0.4
    }),
    point(progress, detailScale) {
      const t = progress * TWO_PI
      const scale = 20 + detailScale * 7
      const denom = 1 + Math.sin(t) ** 2

      return {
        x: 50 + (scale * Math.cos(t)) / denom,
        y: 50 + (scale * Math.sin(t) * Math.cos(t)) / denom
      }
    }
  },
  'hypotrochoid-loop': {
    ...baseCurve('Hypotrochoid Loop', {
      durationMs: 7600,
      particleCount: 82,
      pulseDurationMs: 6200,
      rotationDurationMs: 42000,
      strokeWidth: 4.6,
      trailSpan: 0.46
    }),
    point(progress, detailScale) {
      const t = progress * TWO_PI
      const r = 2.7 + detailScale * 0.45
      const d = 4.8 + detailScale * 1.2
      const x = (8.2 - r) * Math.cos(t) + d * Math.cos(((8.2 - r) / r) * t)
      const y = (8.2 - r) * Math.sin(t) - d * Math.sin(((8.2 - r) / r) * t)

      return {
        x: 50 + x * 3.05,
        y: 50 + y * 3.05
      }
    }
  },
  'three-petal-spiral': spiralPetalCurve('Three-Petal Spiral', 3, 82),
  'four-petal-spiral': spiralPetalCurve('Four-Petal Spiral', 4, 84),
  'five-petal-spiral': spiralPetalCurve('Five-Petal Spiral', 5, 85),
  'six-petal-spiral': spiralPetalCurve('Six-Petal Spiral', 6, 86),
  'butterfly-phase': {
    ...baseCurve('Butterfly Phase', {
      durationMs: 9000,
      particleCount: 88,
      pulseDurationMs: 7000,
      rotationDurationMs: 50000,
      strokeWidth: 4.4,
      trailSpan: 0.32
    }),
    point(progress, detailScale) {
      const t = progress * Math.PI * 12

      const butterfly = Math.exp(Math.cos(t)) - 2 * Math.cos(4 * t) - Math.sin(t / 12) ** 5

      const scale = 4.6 + detailScale * 0.45

      return {
        x: 50 + Math.sin(t) * butterfly * scale,
        y: 50 + Math.cos(t) * butterfly * scale
      }
    }
  },
  'cardioid-glow': cardioidCurve('Cardioid Glow', {
    a: 8.4,
    particleCount: 72,
    pointFor(t, r, scale) {
      return {
        x: 50 + Math.cos(t) * r * scale,
        y: 50 + Math.sin(t) * r * scale
      }
    },
    rFor(t, a) {
      return a * (1 - Math.cos(t))
    }
  }),
  'cardioid-heart': cardioidCurve('Cardioid Heart', {
    a: 8.8,
    particleCount: 74,
    pointFor(t, r, scale) {
      const baseX = Math.cos(t) * r
      const baseY = Math.sin(t) * r

      return {
        x: 50 - baseY * scale,
        y: 50 - baseX * scale
      }
    },
    rFor(t, a) {
      return a * (1 + Math.cos(t))
    }
  }),
  'heart-wave': {
    ...baseCurve('Heart Wave', {
      durationMs: 8400,
      particleCount: 104,
      pulseDurationMs: 5600,
      rotationDurationMs: 22000,
      strokeWidth: 3.9,
      trailSpan: 0.18
    }),
    point(progress, detailScale) {
      const root = 3.3
      const xLimit = Math.sqrt(root)
      const x = -xLimit + progress * xLimit * 2
      const safeRoot = Math.max(0, root - x * x)
      const wave = 0.9 * Math.sqrt(safeRoot) * Math.sin(6.4 * Math.PI * x)
      const curve = Math.abs(x) ** (2 / 3)
      const y = curve + wave

      return {
        x: 50 + x * 23.2,
        y: 18 + (1.75 - y) * (24.5 + detailScale * 1.5)
      }
    }
  },
  'spiral-search': {
    ...baseCurve('Spiral Search', {
      durationMs: 7800,
      particleCount: 86,
      pulseDurationMs: 6800,
      rotationDurationMs: 44000,
      strokeWidth: 4.3,
      trailSpan: 0.28
    }),
    point(progress, detailScale) {
      const t = progress * TWO_PI
      const angle = t * 4
      const radius = 8 + (1 - Math.cos(t)) * (8.5 + detailScale * 2.4)

      return {
        x: 50 + Math.cos(angle) * radius,
        y: 50 + Math.sin(angle) * radius
      }
    }
  },
  'fourier-flow': {
    ...baseCurve('Fourier Flow', {
      durationMs: 8400,
      particleCount: 92,
      pulseDurationMs: 6800,
      rotationDurationMs: 44000,
      strokeWidth: 4.2,
      trailSpan: 0.31
    }),
    point(progress, detailScale) {
      const t = progress * TWO_PI
      const mix = 1 + detailScale * 0.16
      const x = 17 * Math.cos(t) + 7.5 * Math.cos(3 * t + 0.6 * mix) + 3.2 * Math.sin(5 * t - 0.4)
      const y = 15 * Math.sin(t) + 8.2 * Math.sin(2 * t + 0.25) - 4.2 * Math.cos(4 * t - 0.5 * mix)

      return {
        x: 50 + x,
        y: 50 + y
      }
    }
  }
}

export function Loader({
  className,
  label = 'Loading',
  pathSteps = 240,
  role = 'status',
  strokeScale = 1,
  type = 'rose-curve',
  ...props
}: LoaderProps) {
  const config = LOADER_CURVES[type]
  const groupRef = useRef<SVGGElement | null>(null)
  const particleRefs = useRef<Array<SVGCircleElement | null>>([])
  const pathRef = useRef<SVGPathElement | null>(null)

  useEffect(() => {
    let animationFrame = 0
    const startedAt = performance.now()
    const phaseOffset = Math.random()
    particleRefs.current.length = config.particleCount

    const render = (now: number) => {
      const time = now - startedAt
      const progress = ((time + phaseOffset * config.durationMs) % config.durationMs) / config.durationMs
      const detailScale = detailScaleFor(time, config, phaseOffset)
      const rotation = rotationFor(time, config, phaseOffset)

      groupRef.current?.setAttribute('transform', `rotate(${rotation} 50 50)`)
      pathRef.current?.setAttribute('d', buildPath(config, detailScale, pathSteps))

      particleRefs.current.forEach((node, index) => {
        if (!node) {
          return
        }

        const particle = particleFor(config, index, progress, detailScale, strokeScale)
        node.setAttribute('cx', particle.x.toFixed(2))
        node.setAttribute('cy', particle.y.toFixed(2))
        node.setAttribute('r', particle.radius.toFixed(2))
        node.setAttribute('opacity', particle.opacity.toFixed(3))
      })

      animationFrame = window.requestAnimationFrame(render)
    }

    render(performance.now())

    return () => window.cancelAnimationFrame(animationFrame)
  }, [config, pathSteps, strokeScale])

  return (
    <div
      {...props}
      aria-label={props['aria-label'] ?? label}
      className={cn('inline-grid size-10 place-items-center text-primary', className)}
      role={role}
    >
      <svg aria-hidden="true" className="size-full overflow-visible" fill="none" viewBox="0 0 100 100">
        <g ref={groupRef}>
          <path
            opacity="0.1"
            ref={pathRef}
            stroke="currentColor"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={config.strokeWidth * strokeScale}
          />
          {Array.from({ length: config.particleCount }, (_, index) => (
            <circle
              fill="currentColor"
              key={`${type}-${index}`}
              ref={node => {
                particleRefs.current[index] = node
              }}
            />
          ))}
        </g>
      </svg>
    </div>
  )
}

function baseCurve(name: string, options: BaseCurveOptions): LoaderCurve {
  return {
    durationMs: options.durationMs,
    name,
    particleCount: options.particleCount,
    point: options.point ?? (() => ({ x: 50, y: 50 })),
    pulseDurationMs: options.pulseDurationMs,
    rotate: options.rotate ?? false,
    rotationDurationMs: options.rotationDurationMs ?? 36000,
    strokeWidth: options.strokeWidth,
    trailSpan: options.trailSpan
  }
}

function thinkingCurve(
  name: string,
  petalCount: number,
  options: Pick<LoaderCurve, 'durationMs' | 'particleCount' | 'pulseDurationMs' | 'rotationDurationMs' | 'trailSpan'>
): LoaderCurve {
  return {
    ...baseCurve(name, {
      ...options,
      rotate: true,
      strokeWidth: 5.5
    }),
    point(progress, detailScale) {
      const t = progress * TWO_PI
      const x = 7 * Math.cos(t) - 3 * detailScale * Math.cos(petalCount * t)
      const y = 7 * Math.sin(t) - 3 * detailScale * Math.sin(petalCount * t)

      return {
        x: 50 + x * 3.9,
        y: 50 + y * 3.9
      }
    }
  }
}

function roseCurve(
  name: string,
  k: number,
  options: Pick<LoaderCurve, 'durationMs' | 'particleCount' | 'pulseDurationMs' | 'strokeWidth' | 'trailSpan'>
): LoaderCurve {
  return {
    ...baseCurve(name, {
      ...options,
      rotate: true,
      rotationDurationMs: 28000
    }),
    point(progress, detailScale) {
      const t = progress * TWO_PI
      const a = 9.2 + detailScale * 0.6
      const r = a * (0.72 + detailScale * 0.28) * Math.cos(k * t)

      return {
        x: 50 + Math.cos(t) * r * 3.25,
        y: 50 + Math.sin(t) * r * 3.25
      }
    }
  }
}

function spiralPetalCurve(name: string, spiralR: number, particleCount: number): LoaderCurve {
  return {
    ...baseCurve(name, {
      durationMs: 4600,
      particleCount,
      pulseDurationMs: 4200,
      rotate: true,
      rotationDurationMs: 28000,
      strokeWidth: 4.4,
      trailSpan: 0.34
    }),
    point(progress, detailScale) {
      const t = progress * TWO_PI
      const spiralr = 1
      const d = 3 + detailScale * 0.25
      const baseX = (spiralR - spiralr) * Math.cos(t) + d * Math.cos(((spiralR - spiralr) / spiralr) * t)
      const baseY = (spiralR - spiralr) * Math.sin(t) - d * Math.sin(((spiralR - spiralr) / spiralr) * t)
      const scale = 2.2 + detailScale * 0.45

      return {
        x: 50 + baseX * scale,
        y: 50 + baseY * scale
      }
    }
  }
}

function cardioidCurve(
  name: string,
  options: {
    a: number
    particleCount: number
    pointFor: (t: number, r: number, scale: number) => Point
    rFor: (t: number, a: number) => number
  }
): LoaderCurve {
  return {
    ...baseCurve(name, {
      durationMs: 6200,
      particleCount: options.particleCount,
      pulseDurationMs: 5200,
      rotationDurationMs: 36000,
      strokeWidth: 4.9,
      trailSpan: 0.36
    }),
    point(progress, detailScale) {
      const t = progress * TWO_PI
      const a = options.a + detailScale * 0.8
      const r = options.rFor(t, a)

      return options.pointFor(t, r, 2.15)
    }
  }
}

function buildPath(config: LoaderCurve, detailScale: number, steps: number) {
  return Array.from({ length: steps + 1 }, (_, index) => {
    const point = config.point(index / steps, detailScale)

    return `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`
  }).join(' ')
}

function detailScaleFor(time: number, config: LoaderCurve, phaseOffset: number) {
  const pulseProgress =
    ((time + phaseOffset * config.pulseDurationMs) % config.pulseDurationMs) / config.pulseDurationMs

  const pulseAngle = pulseProgress * TWO_PI

  return 0.52 + ((Math.sin(pulseAngle + 0.55) + 1) / 2) * 0.48
}

function normalizeProgress(progress: number) {
  return ((progress % 1) + 1) % 1
}

function particleFor(config: LoaderCurve, index: number, progress: number, detailScale: number, strokeScale: number) {
  const tailOffset = index / (config.particleCount - 1)
  const point = config.point(normalizeProgress(progress - tailOffset * config.trailSpan), detailScale)
  const fade = (1 - tailOffset) ** 0.56

  return {
    opacity: 0.04 + fade * 0.96,
    radius: (0.9 + fade * 2.7) * strokeScale,
    x: point.x,
    y: point.y
  }
}

function rotationFor(time: number, config: LoaderCurve, phaseOffset: number) {
  if (!config.rotate) {
    return 0
  }

  return (
    -(((time + phaseOffset * config.rotationDurationMs) % config.rotationDurationMs) / config.rotationDurationMs) * 360
  )
}
