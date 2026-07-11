import { pathToFileURL } from 'node:url'

const DEFAULT_MARGIN = 8

export type CompanionWindowSize = {
  height: number
  width: number
}

export type Rect = {
  height: number
  width: number
  x: number
  y: number
}

export function buildMenuBarCompanionWindowUrl({
  devServer,
  rendererIndexPath
}: {
  devServer?: string
  rendererIndexPath?: string
}): string {
  if (devServer) {
    const base = devServer.endsWith('/') ? devServer.slice(0, -1) : devServer

    return `${base}/?win=menu-bar-companion#/`
  }

  if (!rendererIndexPath) {
    throw new Error('rendererIndexPath is required outside dev mode')
  }

  return `${pathToFileURL(rendererIndexPath).toString()}?win=menu-bar-companion#/`
}

function clamp(value: number, min: number, max: number): number {
  if (max < min) {
    return min
  }

  return Math.min(Math.max(value, min), max)
}

export function positionMenuBarCompanionWindowBounds({
  displayBounds,
  margin = DEFAULT_MARGIN,
  trayBounds,
  windowSize
}: {
  displayBounds: Rect
  margin?: number
  trayBounds: Rect
  windowSize: CompanionWindowSize
}): Rect {
  const width = Math.round(windowSize.width)
  const height = Math.round(windowSize.height)
  const minX = displayBounds.x + margin
  const maxX = displayBounds.x + displayBounds.width - width - margin
  const centeredX = trayBounds.x + trayBounds.width / 2 - width / 2
  const belowY = trayBounds.y + trayBounds.height + margin
  const aboveY = trayBounds.y - height - margin
  const maxY = displayBounds.y + displayBounds.height - height - margin
  const fitsBelow = belowY + height + margin <= displayBounds.y + displayBounds.height
  const preferredY = fitsBelow ? belowY : aboveY

  return {
    x: Math.round(clamp(centeredX, minX, maxX)),
    y: Math.round(clamp(preferredY, displayBounds.y + margin, maxY)),
    width,
    height
  }
}
