// A little hand jitter still counts as a click; deliberate travel is a drag.
export const PET_CLICK_SLOP_PX = 3

export function didPetPointerMove(startX: number, startY: number, x: number, y: number): boolean {
  return Math.hypot(x - startX, y - startY) > PET_CLICK_SLOP_PX
}
