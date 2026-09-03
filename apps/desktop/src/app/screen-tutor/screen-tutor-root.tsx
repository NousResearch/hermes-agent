import { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'

import type { ScreenTutorAnnotation, ScreenTutorAnnotationsPayload, ScreenTutorPoint } from '@/global'

const ink = {
  amber: '#f59e0b',
  cyan: '#22d3ee',
  emerald: '#34d399',
  rose: '#fb7185',
  white: '#f8fafc'
} as const

const pct = (value: number | undefined) => `${(value ?? 0) * 100}%`
const svg = (value: number | undefined) => (value ?? 0) * 1000

function AnnotationLabel({ annotation }: { annotation: ScreenTutorAnnotation }) {
  if (!annotation.label) {return null}

  const color = ink[annotation.color ?? 'cyan']

  return (
    <div
      className="absolute max-w-72 -translate-y-full rounded-md border bg-[#101722]/94 px-2.5 py-1.5 font-mono text-[12px] font-semibold tracking-[0.01em] text-slate-50 shadow-[0_10px_32px_rgba(0,0,0,0.48)] backdrop-blur-md"
      style={{ borderColor: `${color}99`, left: pct(annotation.x), top: pct(annotation.y) }}
    >
      <span className="mr-1.5 inline-block h-1.5 w-1.5 rounded-full align-middle" style={{ background: color }} />
      {annotation.label}
    </div>
  )
}

function ScreenAnnotation({ annotation, index }: { annotation: ScreenTutorAnnotation; index: number }) {
  const colorName = annotation.color ?? 'cyan'
  const color = ink[colorName]
  const x1 = svg(annotation.x)
  const y1 = svg(annotation.y)
  const x2 = svg(annotation.x2)
  const y2 = svg(annotation.y2)
  const left = Math.min(x1, x2)
  const top = Math.min(y1, y2)
  const width = Math.abs(x2 - x1)
  const height = Math.abs(y2 - y1)

  if (annotation.kind === 'label') {return <AnnotationLabel annotation={annotation} />}

  if (annotation.kind === 'point') {
    return (
      <>
        <div
          className="absolute -translate-x-1/2 -translate-y-1/2"
          style={{ left: pct(annotation.x), top: pct(annotation.y) }}
        >
          <div
            className="absolute left-1/2 top-1/2 h-14 w-14 -translate-x-1/2 -translate-y-1/2 animate-ping rounded-full border-2 opacity-75"
            style={{ borderColor: color }}
          />
          <div
            className="h-5 w-5 rounded-full border-[3px] border-white shadow-[0_0_24px_currentColor]"
            style={{ background: color, color }}
          />
        </div>
        <AnnotationLabel
          annotation={{
            ...annotation,
            x: Math.min(0.96, annotation.x + 0.025),
            y: Math.min(0.96, annotation.y + 0.06)
          }}
        />
      </>
    )
  }

  return (
    <>
      <svg
        aria-hidden
        className="absolute inset-0 h-full w-full overflow-visible"
        preserveAspectRatio="none"
        viewBox="0 0 1000 1000"
      >
        <defs>
          <marker
            id={`hud-arrow-${colorName}-${index}`}
            markerHeight="8"
            markerWidth="8"
            orient="auto"
            refX="7"
            refY="4"
          >
            <path d="M0,0 L8,4 L0,8 Z" fill={color} />
          </marker>
        </defs>
        {annotation.kind === 'rect' && (
          <rect
            fill={`${color}18`}
            height={height}
            rx="10"
            stroke={color}
            strokeDasharray="12 8"
            strokeWidth="4"
            width={width}
            x={left}
            y={top}
          />
        )}
        {annotation.kind === 'circle' && (
          <ellipse
            cx={left + width / 2}
            cy={top + height / 2}
            fill={`${color}16`}
            rx={width / 2}
            ry={height / 2}
            stroke={color}
            strokeWidth="4"
          />
        )}
        {(annotation.kind === 'arrow' || annotation.kind === 'line') && (
          <line
            markerEnd={annotation.kind === 'arrow' ? `url(#hud-arrow-${colorName}-${index})` : undefined}
            stroke={color}
            strokeLinecap="round"
            strokeWidth="5"
            x1={x1}
            x2={x2}
            y1={y1}
            y2={y2}
          />
        )}
      </svg>
      <AnnotationLabel annotation={annotation} />
    </>
  )
}

function ScreenTutorOverlay() {
  const [annotations, setAnnotations] = useState<ScreenTutorAnnotation[]>([])

  useEffect(() => {
    const stopPoint = window.hermesDesktop.screenTutor?.onPoint((point: ScreenTutorPoint) => {
      setAnnotations([{ color: 'cyan', kind: 'point', label: point.label, x: point.x, y: point.y }])
    })

    const stopAnnotations = window.hermesDesktop.screenTutor?.onAnnotations(
      (payload: ScreenTutorAnnotationsPayload) => {
        setAnnotations(current =>
          payload.mode === 'append' ? [...current, ...payload.annotations].slice(-24) : payload.annotations
        )
      }
    )

    return () => {
      stopPoint?.()
      stopAnnotations?.()
    }
  }, [])

  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 z-[2147483647] overflow-hidden select-none">
      {annotations.map((annotation, index) => (
        <ScreenAnnotation
          annotation={annotation}
          index={index}
          key={`${annotation.kind}-${index}-${annotation.x}-${annotation.y}`}
        />
      ))}
    </div>
  )
}

export function mountScreenTutor(): void {
  document.title = 'Hermes Screen Annotations'
  document.documentElement.style.background = 'transparent'
  document.body.style.background = 'transparent'
  createRoot(document.getElementById('root')!).render(<ScreenTutorOverlay />)
}
