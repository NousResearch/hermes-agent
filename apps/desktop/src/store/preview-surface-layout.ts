export interface FloatingGeometry {
  height: number
  width: number
  x: number
  y: number
}

export interface PreviewViewport {
  bottomInset?: number
  height: number
  topInset?: number
  width: number
}

export type PreviewSnapSlot =
  | 'bottom-half'
  | 'bottom-left-quarter'
  | 'bottom-right-quarter'
  | 'center-third'
  | 'left-half'
  | 'left-third'
  | 'left-two-thirds'
  | 'right-half'
  | 'right-third'
  | 'right-two-thirds'
  | 'top-half'
  | 'top-left-quarter'
  | 'top-right-quarter'

export type PreviewEdgePlacement = 'maximized' | PreviewSnapSlot

export interface SnapLayoutDefinition {
  id: 'halves' | 'quarters' | 'thirds' | 'two-thirds-left' | 'two-thirds-right'
  slots: readonly PreviewSnapSlot[]
}

export const WIN11_SNAP_LAYOUTS: readonly SnapLayoutDefinition[] = [
  { id: 'halves', slots: ['left-half', 'right-half'] },
  { id: 'two-thirds-left', slots: ['left-two-thirds', 'right-third'] },
  { id: 'two-thirds-right', slots: ['left-third', 'right-two-thirds'] },
  { id: 'thirds', slots: ['left-third', 'center-third', 'right-third'] },
  {
    id: 'quarters',
    slots: ['top-left-quarter', 'top-right-quarter', 'bottom-left-quarter', 'bottom-right-quarter']
  }
]

export const PREVIEW_FLOATING_MIN_HEIGHT = 240
export const PREVIEW_FLOATING_MIN_WIDTH = 320

function usableViewport(viewport: PreviewViewport): FloatingGeometry {
  const y = Math.max(0, viewport.topInset ?? 0)
  const bottomInset = Math.max(0, viewport.bottomInset ?? 0)

  return {
    height: Math.max(0, viewport.height - y - bottomInset),
    width: Math.max(0, viewport.width),
    x: 0,
    y
  }
}

function partition(total: number, start: number, span: number): { offset: number; size: number } {
  const offset = Math.round((total * start) / 6)
  const end = Math.round((total * (start + span)) / 6)

  return { offset, size: end - offset }
}

export function surfaceGeometryForPlacement(
  placement: 'maximized' | PreviewSnapSlot,
  viewport: PreviewViewport
): FloatingGeometry {
  const usable = usableViewport(viewport)
  let horizontal = { offset: 0, size: usable.width }
  let vertical = { offset: 0, size: usable.height }

  switch (placement) {
    case 'left-half':
      horizontal = partition(usable.width, 0, 3)

      break

    case 'right-half':
      horizontal = partition(usable.width, 3, 3)

      break

    case 'left-third':
      horizontal = partition(usable.width, 0, 2)

      break

    case 'center-third':
      horizontal = partition(usable.width, 2, 2)

      break

    case 'right-third':
      horizontal = partition(usable.width, 4, 2)

      break

    case 'left-two-thirds':
      horizontal = partition(usable.width, 0, 4)

      break

    case 'right-two-thirds':
      horizontal = partition(usable.width, 2, 4)

      break

    case 'top-left-quarter':
      horizontal = partition(usable.width, 0, 3)
      vertical = partition(usable.height, 0, 3)

      break

    case 'top-right-quarter':
      horizontal = partition(usable.width, 3, 3)
      vertical = partition(usable.height, 0, 3)

      break

    case 'bottom-left-quarter':
      horizontal = partition(usable.width, 0, 3)
      vertical = partition(usable.height, 3, 3)

      break

    case 'bottom-right-quarter':
      horizontal = partition(usable.width, 3, 3)
      vertical = partition(usable.height, 3, 3)

      break

    case 'top-half':
      vertical = partition(usable.height, 0, 3)

      break

    case 'bottom-half':
      vertical = partition(usable.height, 3, 3)

      break

    case 'maximized':
      break
  }

  return {
    height: vertical.size,
    width: horizontal.size,
    x: usable.x + horizontal.offset,
    y: usable.y + vertical.offset
  }
}

export function clampFloatingGeometry(geometry: FloatingGeometry, viewport: PreviewViewport): FloatingGeometry {
  const usable = usableViewport(viewport)
  const minWidth = Math.min(PREVIEW_FLOATING_MIN_WIDTH, usable.width)
  const minHeight = Math.min(PREVIEW_FLOATING_MIN_HEIGHT, usable.height)
  const width = Math.min(Math.max(geometry.width, minWidth), usable.width)
  const height = Math.min(Math.max(geometry.height, minHeight), usable.height)

  return {
    height,
    width,
    x: Math.min(Math.max(geometry.x, usable.x), usable.x + usable.width - width),
    y: Math.min(Math.max(geometry.y, usable.y), usable.y + usable.height - height)
  }
}

export function edgeSnapPlacement(
  point: { x: number; y: number },
  viewport: PreviewViewport,
  threshold = 24
): PreviewEdgePlacement | null {
  const usable = usableViewport(viewport)
  const nearLeft = point.x <= usable.x + threshold
  const nearRight = point.x >= usable.x + usable.width - threshold
  const nearTop = point.y <= usable.y + threshold
  const nearBottom = point.y >= usable.y + usable.height - threshold

  if (nearTop && nearLeft) {
    return 'top-left-quarter'
  }

  if (nearTop && nearRight) {
    return 'top-right-quarter'
  }

  if (nearBottom && nearLeft) {
    return 'bottom-left-quarter'
  }

  if (nearBottom && nearRight) {
    return 'bottom-right-quarter'
  }

  if (nearTop) {
    return 'maximized'
  }

  if (nearLeft) {
    return 'left-half'
  }

  if (nearRight) {
    return 'right-half'
  }

  if (nearBottom) {
    return 'bottom-half'
  }

  return null
}
