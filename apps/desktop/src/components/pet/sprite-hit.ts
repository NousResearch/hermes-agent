/** A sprite pixel counts as solid (interactive) at or above this alpha (0-255).
 *  Low enough to catch anti-aliased edges, high enough that a faint halo still
 *  clicks through. Shared by the pop-out overlay and the in-window pet. */
export const ALPHA_HIT_THRESHOLD = 16

export interface SpriteHitSurface {
  width: number
  height: number
  getBoundingClientRect(): { left: number; top: number; width: number; height: number }
  getContext(kind: '2d'): {
    getImageData(x: number, y: number, w: number, h: number): { data: ArrayLike<number> }
  } | null
}

/**
 * True when the viewport point lands on a solid sprite pixel.
 * false when it misses the box or the pixel is transparent.
 * null when the pixel can't be read (empty box, missing context, tainted canvas).
 * Callers choose the failure mode: the overlay fails open so the pet stays
 * grabbable; the in-window pet fails closed so a bad read cannot eat a click.
 */
export function spriteSolidAt(surface: SpriteHitSurface, clientX: number, clientY: number): boolean | null {
  const rect = surface.getBoundingClientRect()

  if (rect.width === 0 || rect.height === 0 || surface.width === 0 || surface.height === 0) {
    return null
  }

  if (
    clientX < rect.left ||
    clientY < rect.top ||
    clientX >= rect.left + rect.width ||
    clientY >= rect.top + rect.height
  ) {
    return false
  }

  const x = Math.floor((clientX - rect.left) * (surface.width / rect.width))
  const y = Math.floor((clientY - rect.top) * (surface.height / rect.height))

  if (x < 0 || y < 0 || x >= surface.width || y >= surface.height) {
    return false
  }

  const ctx = surface.getContext('2d')

  if (!ctx) {
    return null
  }

  try {
    return ctx.getImageData(x, y, 1, 1).data[3] >= ALPHA_HIT_THRESHOLD
  } catch {
    return null
  }
}
