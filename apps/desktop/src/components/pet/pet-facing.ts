/**
 * Return the transform that points a left-facing sprite toward the viewport
 * center. A pet on the left looks right; a pet on the right looks left.
 */
export function petFacingTransform(leftX: number, petW: number, viewportWidth: number): string {
  return leftX + petW / 2 < viewportWidth / 2 ? 'scaleX(-1)' : 'none'
}
