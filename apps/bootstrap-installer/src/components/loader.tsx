import { type ComponentProps, useEffect, useRef } from 'react'

import { cn } from '../lib/utils'

/*
 * Loader — the desktop's "Fourier Flow" curve, ported standalone.
 *
 * The shim can't import apps/desktop's 559-line multi-curve <Loader> (cross-app
 * coupling + bundle bloat that defeats the point of a lightweight installer), so
 * this is just the one curve the installer uses. Math + tuning lifted verbatim
 * from apps/desktop/src/components/ui/loader.tsx ('fourier-flow'); rotation is
 * dropped because that curve never rotates. Keep the constants in sync if the
 * desktop's curve is retuned.
 */

const TWO_PI = Math.PI * 2

const CURVE = {
  durationMs: 2200,
  particleCount: 92,
  pulseDurationMs: 2000,
  strokeWidth: 4.2,
  trailSpan: 0.31,
  point(progress: number, detailScale: number) {
    const t = progress * TWO_PI
    const mix = 1 + detailScale * 0.16
    const x = 17 * Math.cos(t) + 7.5 * Math.cos(3 * t + 0.6 * mix) + 3.2 * Math.sin(5 * t - 0.4)
    const y = 15 * Math.sin(t) + 8.2 * Math.sin(2 * t + 0.25) - 4.2 * Math.cos(4 * t - 0.5 * mix)

    return { x: 50 + x, y: 50 + y }
  }
}

const norm = (progress: number) => ((progress % 1) + 1) % 1

function detailScaleFor(time: number, phaseOffset: number) {
  const p = ((time + phaseOffset * CURVE.pulseDurationMs) % CURVE.pulseDurationMs) / CURVE.pulseDurationMs

  return 0.52 + ((Math.sin(p * TWO_PI + 0.55) + 1) / 2) * 0.48
}

function buildPath(detailScale: number, steps: number) {
  return Array.from({ length: steps + 1 }, (_, i) => {
    const { x, y } = CURVE.point(i / steps, detailScale)

    return `${i === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
  }).join(' ')
}

function particleFor(index: number, progress: number, detailScale: number, strokeScale: number) {
  const tail = index / (CURVE.particleCount - 1)
  const { x, y } = CURVE.point(norm(progress - tail * CURVE.trailSpan), detailScale)
  const fade = (1 - tail) ** 0.56

  return { x, y, opacity: 0.04 + fade * 0.96, radius: (0.9 + fade * 2.7) * strokeScale }
}

interface LoaderProps extends Omit<ComponentProps<'div'>, 'children'> {
  label?: string
  pathSteps?: number
  strokeScale?: number
}

export function Loader({
  className,
  label = 'Loading',
  pathSteps = 240,
  role = 'status',
  strokeScale = 1,
  ...props
}: LoaderProps) {
  const particleRefs = useRef<Array<SVGCircleElement | null>>([])
  const pathRef = useRef<SVGPathElement | null>(null)

  useEffect(() => {
    let frame = 0
    const startedAt = performance.now()
    const phaseOffset = Math.random()
    particleRefs.current.length = CURVE.particleCount

    const render = (now: number) => {
      const time = now - startedAt
      const progress = ((time + phaseOffset * CURVE.durationMs) % CURVE.durationMs) / CURVE.durationMs
      const detailScale = detailScaleFor(time, phaseOffset)

      pathRef.current?.setAttribute('d', buildPath(detailScale, pathSteps))

      particleRefs.current.forEach((node, index) => {
        if (!node) {
          return
        }

        const p = particleFor(index, progress, detailScale, strokeScale)
        node.setAttribute('cx', p.x.toFixed(2))
        node.setAttribute('cy', p.y.toFixed(2))
        node.setAttribute('r', p.radius.toFixed(2))
        node.setAttribute('opacity', p.opacity.toFixed(3))
      })

      frame = window.requestAnimationFrame(render)
    }

    render(performance.now())

    return () => window.cancelAnimationFrame(frame)
  }, [pathSteps, strokeScale])

  return (
    <div
      {...props}
      aria-label={props['aria-label'] ?? label}
      className={cn('inline-grid size-10 place-items-center text-primary', className)}
      role={role}
    >
      <svg aria-hidden="true" className="size-full overflow-visible" fill="none" viewBox="0 0 100 100">
        <path
          opacity="0.1"
          ref={pathRef}
          stroke="currentColor"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={CURVE.strokeWidth * strokeScale}
        />
        {Array.from({ length: CURVE.particleCount }, (_, index) => (
          <circle
            fill="currentColor"
            key={index}
            ref={node => {
              particleRefs.current[index] = node
            }}
          />
        ))}
      </svg>
    </div>
  )
}
