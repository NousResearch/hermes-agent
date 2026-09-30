// Screen-analysis scanline overlay: state model, display geometry, and fade
// timing. Mirrors `wake-indicator.ts` (the ambient wake cue) but spans the
// WHOLE desktop on every OS — the wake cue is a 176px macOS-only pill, this is
// a full-screen, click-through, always-on-top "reading the display" sweep that
// runs while a `computer_use` screen-capture tool call is in flight.

export const SCANLINE_FADE_MS = 450

export const SCANLINE_STATES = ['hidden', 'active'] as const

export type ScanlineState = (typeof SCANLINE_STATES)[number]

// Which display the sweep covers. The capture helper writes this to a scope
// file so the overlay can scope to exactly the screen that was captured.
//   both      — the union of every display (default; built-in computer_use
//               capture has no scope file, so it sweeps the whole desktop).
//   primary   — only the primary display.
//   secondary — only the non-primary display (unambiguous on a 2-monitor rig).
export const SCANLINE_SCOPES = ['both', 'primary', 'secondary'] as const

export type ScanlineScope = (typeof SCANLINE_SCOPES)[number]

export function normalizeScanlineScope(value: unknown): ScanlineScope {
  return SCANLINE_SCOPES.includes(value as ScanlineScope) ? (value as ScanlineScope) : 'both'
}

interface DisplayLike {
  bounds: {
    height: number
    width: number
    x: number
    y: number
  }
  internal?: boolean
  id?: number
}

/**
 * Scope file the capture helper writes next to the other Hermes state in the
 * HERMES_HOME root (%LOCALAPPDATA%\hermes on Windows). Shape:
 *   { "phase": "start" | "end", "scope": "both" | "primary" | "secondary", "ts": epochMs }
 * The app polls it: `start` scopes + lights the sweep, `end` fades it out and
 * resets the scope. A missing/undecodable file means "no helper capture in
 * flight" → the sweep defaults to `both`.
 */
export const SCANLINE_SCOPE_FILENAME = 'scanline-scope.json'
// A scope file older than this is treated as a stale leftover (e.g. the helper
// crashed mid-capture) and ignored — never scope to a ghost.
export const SCANLINE_SCOPE_MAX_AGE_MS = 30_000

export function resolveScanlineScopePath(hermesHome: string): string {
  // `hermesHome` may use OS separators; join with node's path so the result is
  // valid on every platform.
  return hermesHome.replace(/[\\/]+$/, '') + `/${SCANLINE_SCOPE_FILENAME}`
}

/**
 * Pick the window bounds for a scope.
 *   both      — union of every display (the existing full-desktop behaviour).
 *   primary   — the display `screen.getPrimaryDisplay().id` identifies; the
 *               caller passes that id as `primaryId`.
 *   secondary — the first non-primary display (unambiguous on 2 monitors; on a
 *               3+ setup it is the first display listed after the primary,
 *               which matches the helper's 0-based index order).
 * Electron's `internal` flag means "built-in panel", NOT "primary" — on a
 * dual-external-monitor desktop every display reports `internal: false`, so it
 * only serves as a fallback when no `primaryId` is supplied. Falls back to the
 * full union when the requested scope has no matching display (e.g.
 * single-monitor with scope `secondary`) so the overlay never disappears.
 */
export function scanlineBoundsForScope(displays: DisplayLike[], scope: ScanlineScope, primaryId?: number) {
  if (displays.length === 0) {
    return { height: 0, width: 0, x: 0, y: 0 }
  }

  if (scope !== 'both') {
    const primary =
      (primaryId != null ? displays.find(d => d.id === primaryId) : undefined) ?? displays.find(d => d.internal) ?? null

    const target =
      scope === 'primary'
        ? primary
        : scope === 'secondary'
          ? displays.find(d => d !== primary) ?? null
          : null

    if (target) {
      const b = target.bounds

      return { x: Math.round(b.x), y: Math.round(b.y), width: Math.round(b.width), height: Math.round(b.height) }
    }
  }

  return scanlineWindowBounds(displays)
}

export function normalizeScanlineState(value: unknown): ScanlineState {
  return SCANLINE_STATES.includes(value as ScanlineState) ? (value as ScanlineState) : 'hidden'
}

/**
 * One click-through window must cover the user's entire desktop, so on a
 * multi-monitor setup we span the union of every display's bounds (which can
 * extend negative into the top-left quadrant) rather than a single display.
 */
export function scanlineWindowBounds(displays: DisplayLike[]) {
  if (displays.length === 0) {
    return { height: 0, width: 0, x: 0, y: 0 }
  }

  let minX = Number.POSITIVE_INFINITY
  let minY = Number.POSITIVE_INFINITY
  let maxX = Number.NEGATIVE_INFINITY
  let maxY = Number.NEGATIVE_INFINITY

  for (const display of displays) {
    minX = Math.min(minX, display.bounds.x)
    minY = Math.min(minY, display.bounds.y)
    maxX = Math.max(maxX, display.bounds.x + display.bounds.width)
    maxY = Math.max(maxY, display.bounds.y + display.bounds.height)
  }

  return {
    height: Math.round(maxY - minY),
    width: Math.round(maxX - minX),
    x: Math.round(minX),
    y: Math.round(minY)
  }
}
