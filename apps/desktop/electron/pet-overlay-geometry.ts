/** Pure display geometry for the floating pet overlay. No Electron imports. */

export interface PetOverlayRectangle {
  x: number
  y: number
  width: number
  height: number
}

export interface PetOverlayDisplay {
  workArea: PetOverlayRectangle
}

const MIN_SIZE = 80

/**
 * Safe overlap margins for the pet overlay window. The transparent window has
 * padding around the sprite (see pet-overlay.ts): the sprite sits at
 * bottom-center with ~50px padding on each side horizontally and ~176px above
 * + ~24px below vertically. These margins let the window overlap the screen
 * edge by the padding amount so the sprite can reach the edge.
 */
export const PET_OVERLAY_SAFE_MARGIN_X = 50
export const PET_OVERLAY_SAFE_MARGIN_Y = 176

const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
const rounded = (value: unknown, fallback: number): number => (finite(value) ? Math.round(value) : fallback)

const clamp = (value: number, minimum: number, maximum: number): number => Math.max(minimum, Math.min(value, maximum))

function sanitizeBounds(bounds: Partial<PetOverlayRectangle> | null | undefined): PetOverlayRectangle {
  return {
    x: rounded(bounds?.x, 0),
    y: rounded(bounds?.y, 0),
    width: Math.max(MIN_SIZE, rounded(bounds?.width, MIN_SIZE)),
    height: Math.max(MIN_SIZE, rounded(bounds?.height, MIN_SIZE))
  }
}

function validDisplays(displays: readonly PetOverlayDisplay[] | null | undefined): PetOverlayDisplay[] {
  if (!Array.isArray(displays)) {
    return []
  }

  return displays.filter(display => {
    const area = display?.workArea

    return (
      finite(area?.x) &&
      finite(area?.y) &&
      finite(area?.width) &&
      finite(area?.height) &&
      area.width > 0 &&
      area.height > 0
    )
  })
}

function intersectionArea(rect: PetOverlayRectangle, area: PetOverlayRectangle): number {
  const width = Math.max(0, Math.min(rect.x + rect.width, area.x + area.width) - Math.max(rect.x, area.x))
  const height = Math.max(0, Math.min(rect.y + rect.height, area.y + area.height) - Math.max(rect.y, area.y))

  return width * height
}

function centerDistanceSquared(rect: PetOverlayRectangle, area: PetOverlayRectangle): number {
  const dx = rect.x + rect.width / 2 - (area.x + area.width / 2)
  const dy = rect.y + rect.height / 2 - (area.y + area.height / 2)

  return dx * dx + dy * dy
}

function selectDisplay(bounds: PetOverlayRectangle, displays: PetOverlayDisplay[]): PetOverlayDisplay {
  let selected = displays[0]!
  let greatestArea = intersectionArea(bounds, selected.workArea)

  for (let index = 1; index < displays.length; index += 1) {
    const candidate = displays[index]!
    const area = intersectionArea(bounds, candidate.workArea)

    if (area > greatestArea) {
      selected = candidate
      greatestArea = area
    }
  }

  if (greatestArea > 0) {
    return selected
  }

  let nearestDistance = centerDistanceSquared(bounds, selected.workArea)

  for (let index = 1; index < displays.length; index += 1) {
    const candidate = displays[index]!
    const distance = centerDistanceSquared(bounds, candidate.workArea)

    // Strictly-less preserves deterministic input order on ties.
    if (distance < nearestDistance) {
      selected = candidate
      nearestDistance = distance
    }
  }

  return selected
}

export interface ClampPetOverlayBoundsOptions {
  /**
   * How far the window may extend past the left/right work-area edge before
   * clamping kicks in. The pet sprite sits at bottom-center with generous
   * padding, so the window can safely overlap the screen edge by the padding
   * amount without the sprite leaving the visible area. Default: 0 (fully
   * in work-area).
   */
  spriteSafeMarginX?: number
  /**
   * How far the window may extend past the top/bottom work-area edge before
   * clamping kicks in. Default: 0 (fully in work-area).
   */
  spriteSafeMarginY?: number
}

/**
 * Sanitize and fit requested screen bounds into the best current display work
 * area. Empty/invalid display input returns the sanitized request unchanged.
 *
 * When `spriteSafeMarginX`/`spriteSafeMarginY` are provided, the window is
 * allowed to overlap the work-area edge by that much — the transparent padding
 * around the sprite can bleed offscreen so the sprite itself reaches the edge.
 */
export function clampPetOverlayBounds(
  requested: Partial<PetOverlayRectangle> | null | undefined,
  displays: readonly PetOverlayDisplay[] | null | undefined,
  options?: ClampPetOverlayBoundsOptions | null
): PetOverlayRectangle {
  const bounds = sanitizeBounds(requested)
  const usableDisplays = validDisplays(displays)

  if (usableDisplays.length === 0) {
    return bounds
  }

  const safeX = finite(options?.spriteSafeMarginX) ? Math.max(0, Math.round(options!.spriteSafeMarginX!)) : 0
  const safeY = finite(options?.spriteSafeMarginY) ? Math.max(0, Math.round(options!.spriteSafeMarginY!)) : 0

  const area = selectDisplay(bounds, usableDisplays).workArea
  const width = Math.min(bounds.width, Math.round(area.width))
  const height = Math.min(bounds.height, Math.round(area.height))
  const minX = Math.round(area.x) - safeX
  const minY = Math.round(area.y) - safeY
  const maxX = Math.round(area.x + area.width - width) + safeX
  const maxY = Math.round(area.y + area.height - height) + safeY

  return {
    x: clamp(bounds.x, minX, maxX),
    y: clamp(bounds.y, minY, maxY),
    width,
    height
  }
}
