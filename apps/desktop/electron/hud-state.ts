/**
 * hud-state.json — the Mini Assistant's persisted geometry and pin state.
 *
 * The HUD window (the compact, movable, resizable chat band the tray opens)
 * is moved and resized by the renderer through `setBounds`, so the last place
 * it was parked has to be remembered between launches. `main.ts` owns the file
 * I/O and the live `screen` displays; the validation lives here so "which
 * snapshots do we trust" is unit-testable, same split as `window-state.ts`.
 */

const MIN_WIDTH = 380
const MIN_HEIGHT = 160

export interface HudState {
  height: number
  /** Whether the Mini Assistant floats above other windows. Absent in legacy
   *  snapshots — those windows were always on top, so that IS the default. */
  alwaysOnTop: boolean
  width: number
  x: number
  y: number
}

const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)

/**
 * Raw JSON → clean state, or null when the snapshot is unusable. A snapshot
 * whose bounds fell off a display is the caller's problem (it re-checks
 * against the live `screen`); this only rejects structurally broken input.
 */
export function sanitizeHudState(raw: unknown): HudState | null {
  const record = (raw && typeof raw === 'object' ? raw : null) as Record<string, unknown> | null

  if (!record) {
    return null
  }

  const { height, width, x, y } = record

  // Checked one by one so each narrows: `.every(finite)` proves the values are
  // numbers without telling TypeScript which ones.
  if (!finite(x) || !finite(y) || !finite(width) || !finite(height) || width < MIN_WIDTH || height < MIN_HEIGHT) {
    return null
  }

  return {
    alwaysOnTop: record.alwaysOnTop === undefined ? true : record.alwaysOnTop === true,
    height,
    width,
    x,
    y
  }
}
