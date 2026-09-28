import type { Rectangle } from 'electron'

export interface GrowRequest {
  bottom?: number
  left?: number
  minWidth?: number
  right?: number
  top?: number
}

export interface GrowInputs {
  bounds: { height: number; width: number }
  frameWidth?: number
  workArea: { height: number; width: number; x: number; y: number }
  zoom?: number
}

const MAX_DELTA_PX = 4000

const MAX_WORK_AREA = 0.92

export function growWindowBounds(
  request: GrowRequest | null | undefined,
  { bounds, frameWidth = 0, workArea, zoom = 1 }: GrowInputs
) {
  const dip = (value: number | undefined, round: (n: number) => number) =>
    Math.max(0, Math.min(MAX_DELTA_PX, round((Number(value) || 0) * zoom)))

  const toDip = (value?: number) => dip(value, Math.round)

  const requestedMin = dip(request?.minWidth, Math.ceil)
  const grown = bounds.width + toDip(request?.left) + toDip(request?.right)

  const width = Math.min(
    Math.max(grown, requestedMin ? requestedMin + frameWidth : 0),
    Math.round(workArea.width * MAX_WORK_AREA)
  )

  const height = Math.min(
    bounds.height + toDip(request?.top) + toDip(request?.bottom),
    Math.round(workArea.height * MAX_WORK_AREA)
  )

  return centeredBounds(workArea, width, height)
}

export function centeredBounds(workArea: Rectangle, width: number, height: number): Rectangle {
  return {
    height,
    width,
    x: Math.round(workArea.x + (workArea.width - width) / 2),
    y: Math.round(workArea.y + (workArea.height - height) / 2)
  }
}
