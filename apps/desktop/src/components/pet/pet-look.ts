export interface PetLookCapability {
  lookDirectionCount?: number
  spriteVersionNumber?: number
}

export interface PetLookCell {
  column: number
  row: number
}

export const PET_V2_LOOK_DIRECTION_COUNT = 16
export const PET_V2_LOOK_ROW_START = 9
export const PET_V2_NEUTRAL_LOOK_CELL: PetLookCell = { column: 6, row: 0 }

const DEGREES_PER_DIRECTION = 360 / PET_V2_LOOK_DIRECTION_COUNT

/** Only the gateway may declare a sheet safe for the v2 look-row contract. */
export function supportsPetLookDirections(info: PetLookCapability): boolean {
  return info.spriteVersionNumber === 2 && info.lookDirectionCount === PET_V2_LOOK_DIRECTION_COUNT
}

/** Gaze is an idle-only presentation; roam motion and travel retain priority. */
export function shouldEnablePetGaze(
  info: PetLookCapability,
  atRest: boolean,
  motion: null | string,
  roamDirection: number
): boolean {
  return atRest && motion === null && roamDirection === 0 && supportsPetLookDirections(info)
}

/**
 * Map a pointer vector to one of the 16 v2 look cells.
 *
 * The atlas starts at north (000°), advances clockwise in 22.5° steps, and
 * packs eight cells per row across rows 9 and 10. Inside the deadzone this
 * returns null so the caller can render the fixed neutral v2 frame.
 */
export function lookCellForVector(dx: number, dy: number, deadzone: number): PetLookCell | null {
  if (!Number.isFinite(dx) || !Number.isFinite(dy) || Math.hypot(dx, dy) <= Math.max(0, deadzone)) {
    return null
  }

  const clockwiseFromNorth = (Math.atan2(dx, -dy) * 180) / Math.PI
  const normalized = (clockwiseFromNorth + 360) % 360
  const index = Math.round(normalized / DEGREES_PER_DIRECTION) % PET_V2_LOOK_DIRECTION_COUNT

  return {
    column: index % 8,
    row: PET_V2_LOOK_ROW_START + Math.floor(index / 8)
  }
}
