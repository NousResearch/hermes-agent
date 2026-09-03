import type { Display, Rectangle } from 'electron'

export const SCREEN_TUTOR_MAX_LONG_EDGE = 2048
export const SCREEN_TUTOR_POINT_TTL_MS = 8_000
export const SCREEN_TUTOR_ANNOTATION_TTL_MS = 30_000
export const SCREEN_TUTOR_MAX_ANNOTATIONS = 24

export type ScreenTutorAnnotationColor = 'amber' | 'cyan' | 'emerald' | 'rose' | 'white'
export type ScreenTutorAnnotationKind = 'arrow' | 'circle' | 'label' | 'line' | 'point' | 'rect'

export interface ScreenTutorAnnotation {
  color: ScreenTutorAnnotationColor
  kind: ScreenTutorAnnotationKind
  label?: string
  x: number
  x2?: number
  y: number
  y2?: number
}

export interface ScreenTutorGuideStep {
  id: string
  instruction: string
  step: number
  successCheck?: string
  title: string
  total: number
}

export interface ScreenTutorAnnotationsPayload {
  annotations: ScreenTutorAnnotation[]
  displayId: string
  frozen: boolean
  guide?: ScreenTutorGuideStep
  mode: 'append' | 'replace'
  ttlMs: number
}

function normalizeGuide(value: unknown): ScreenTutorGuideStep | undefined {
  if (!value || typeof value !== 'object') {
    return undefined
  }

  const record = value as Record<string, unknown>
  const id = typeof record.id === 'string' ? record.id.trim().slice(0, 80) : ''
  const instruction = typeof record.instruction === 'string' ? record.instruction.trim().slice(0, 240) : ''
  const title = typeof record.title === 'string' ? record.title.trim().slice(0, 100) : ''
  const step = Math.max(1, Math.min(99, Math.round(Number(record.step))))
  const total = Math.max(step, Math.min(99, Math.round(Number(record.total))))

  if (!id || !instruction || !title || !Number.isFinite(step) || !Number.isFinite(total)) {
    return undefined
  }

  const successCheck = typeof record.successCheck === 'string' ? record.successCheck.trim().slice(0, 240) : ''

  return { id, instruction, step, title, total, ...(successCheck ? { successCheck } : {}) }
}

export interface ScreenTutorPointPayload {
  displayId: string
  label?: string
  x: number
  y: number
}

const coordinate = (value: unknown): number | null => {
  const number = Number(value)

  return Number.isFinite(number) && number >= 0 && number <= 1 ? number : null
}

const annotationColor = (value: unknown): ScreenTutorAnnotationColor => {
  return value === 'amber' || value === 'emerald' || value === 'rose' || value === 'white' ? value : 'cyan'
}

export function normalizeScreenTutorAnnotations(value: unknown): ScreenTutorAnnotationsPayload | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const record = value as Record<string, unknown>
  const displayId = typeof record.displayId === 'string' ? record.displayId.trim() : ''

  if (!displayId || !Array.isArray(record.annotations)) {
    return null
  }

  const annotations: ScreenTutorAnnotation[] = []

  for (const candidate of record.annotations.slice(0, SCREEN_TUTOR_MAX_ANNOTATIONS)) {
    if (!candidate || typeof candidate !== 'object') {
      continue
    }

    const item = candidate as Record<string, unknown>
    const kind = item.kind

    if (
      kind !== 'arrow' &&
      kind !== 'circle' &&
      kind !== 'label' &&
      kind !== 'line' &&
      kind !== 'point' &&
      kind !== 'rect'
    ) {
      continue
    }

    const x = coordinate(item.x)
    const y = coordinate(item.y)

    if (x === null || y === null) {
      continue
    }

    const label = typeof item.label === 'string' ? item.label.trim().slice(0, 120) : ''
    const annotation: ScreenTutorAnnotation = { color: annotationColor(item.color), kind, x, y }

    if (label) {
      annotation.label = label
    }

    if (kind === 'arrow' || kind === 'circle' || kind === 'line' || kind === 'rect') {
      const x2 = coordinate(item.x2)
      const y2 = coordinate(item.y2)

      if (x2 === null || y2 === null) {
        continue
      }

      annotation.x2 = x2
      annotation.y2 = y2
    }

    annotations.push(annotation)
  }

  if (!annotations.length) {
    return null
  }

  const requestedTtl = Number(record.ttlMs)
  const guide = normalizeGuide(record.guide)

  const ttlMs = Number.isFinite(requestedTtl)
    ? Math.min(300_000, Math.max(3_000, Math.round(requestedTtl)))
    : SCREEN_TUTOR_ANNOTATION_TTL_MS

  return {
    annotations,
    displayId,
    frozen: record.frozen === true,
    ...(guide ? { guide } : {}),
    mode: record.mode === 'append' ? 'append' : 'replace',
    ttlMs
  }
}

export function screenTutorThumbnailSize(
  bounds: Pick<Rectangle, 'height' | 'width'>,
  scaleFactor: number,
  maxLongEdge = SCREEN_TUTOR_MAX_LONG_EDGE
): { height: number; width: number } {
  const nativeWidth = Math.max(1, Math.round(bounds.width * Math.max(1, scaleFactor)))
  const nativeHeight = Math.max(1, Math.round(bounds.height * Math.max(1, scaleFactor)))
  const scale = Math.min(1, maxLongEdge / Math.max(nativeWidth, nativeHeight))

  return {
    height: Math.max(1, Math.round(nativeHeight * scale)),
    width: Math.max(1, Math.round(nativeWidth * scale))
  }
}

export function selectScreenTutorSource<T extends { display_id?: string }>(
  sources: readonly T[],
  displayId: string
): T | null {
  return (
    sources.find(source => String(source.display_id ?? '') === displayId) ?? (sources.length === 1 ? sources[0] : null)
  )
}

export function normalizeScreenTutorPoint(value: unknown): ScreenTutorPointPayload | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const record = value as Record<string, unknown>
  const displayId = typeof record.displayId === 'string' ? record.displayId.trim() : ''
  const x = Number(record.x)
  const y = Number(record.y)

  if (!displayId || !Number.isFinite(x) || !Number.isFinite(y) || x < 0 || x > 1 || y < 0 || y > 1) {
    return null
  }

  const label = typeof record.label === 'string' ? record.label.trim().slice(0, 120) : ''

  return { displayId, x, y, ...(label ? { label } : {}) }
}

export function screenTutorDisplayForPoint(
  displays: readonly Display[],
  point: Pick<ScreenTutorPointPayload, 'displayId'>
): Display | null {
  return displays.find(display => String(display.id) === point.displayId) ?? null
}

export function screenTutorPointInDisplay(
  point: Pick<ScreenTutorPointPayload, 'x' | 'y'>,
  bounds: Pick<Rectangle, 'height' | 'width'>
): { x: number; y: number } {
  return {
    x: Math.round(point.x * bounds.width),
    y: Math.round(point.y * bounds.height)
  }
}
