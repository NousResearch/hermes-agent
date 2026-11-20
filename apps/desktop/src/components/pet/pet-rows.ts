/** Sheet row names and the roam's row picks: pure, shared by the sprite and the store. */

// Mirrors agent.pet.constants.CODEX_STATE_ROWS (Petdex current taxonomy).
export const DEFAULT_STATE_ROWS = [
  'idle',
  'running-right',
  'running-left',
  'waving',
  'jumping',
  'failed',
  'waiting',
  'running',
  'review'
]

/**
 * Row for a roam hop: the sheet's own jump row, so a pet.json remap of the
 * celebrate `jump` state (e.g. to a smile) never fires mid-wander.
 */
export function roamHopRow(motion: string | null, stateRows = DEFAULT_STATE_ROWS): string | undefined {
  return motion === 'jump' ? stateRows.find(r => r === 'jumping' || r === 'jump') : undefined
}

/**
 * Pick the running row + mirror for a horizontal travel direction.
 *
 * Codex sheets ship dedicated `running-left` / `running-right` locomotion rows
 * (already facing their way → no flip; the opposite one is mirrored). Pets
 * without them get no `row`, so the `run` state resolves it; that art faces
 * left by convention, so rightward travel is mirrored.
 */
export function roamWalkRow(dir: -1 | 0 | 1, stateRows = DEFAULT_STATE_ROWS): { row?: string; mirror: boolean } {
  if (dir === 0) {
    return { mirror: false }
  }

  const [own, other] = dir > 0 ? ['running-right', 'running-left'] : ['running-left', 'running-right']

  if (stateRows.includes(own)) {
    return { mirror: false, row: own }
  }

  return stateRows.includes(other) ? { mirror: true, row: other } : { mirror: dir > 0 }
}
