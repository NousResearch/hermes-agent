// Pure geometry logic for the plugin overlay window.
//
// Extracted from main.ts (Vicky review): normalize/validate/clamp/default are
// pure functions here so they can be unit-tested — the hud-geometry.ts
// precedent. No Electron or fs imports: main.ts owns the I/O (read/persist),
// this module owns the decisions about what bounds are legal and where a
// window may respawn.

export const PLUGIN_OVERLAY_MIN_WIDTH = 120
export const PLUGIN_OVERLAY_MIN_HEIGHT = 80
export const PLUGIN_OVERLAY_DEFAULT_WIDTH = 320
export const PLUGIN_OVERLAY_DEFAULT_HEIGHT = 420
// Default margin to the work-area edge when no remembered bounds exist.
export const PLUGIN_OVERLAY_EDGE_MARGIN = 16

// Mascot posture (the Dash pencil): a small non-activating sprite pinned to
// the Hermes main window. Size is FIXED — main enforces it on every
// set-bounds; the renderer reports drag positions and main clamps them into
// the main window's rect (the mascot must never leave the app window).
export const PLUGIN_OVERLAY_MASCOT_WIDTH = 96
export const PLUGIN_OVERLAY_MASCOT_HEIGHT = 96
// Smallest size a stored mascot record may carry (stored size is ignored on
// spawn — the fixed mascot size wins — but a bogus tiny record is discarded).
export const PLUGIN_OVERLAY_MASCOT_MIN_STORED = 48

export interface PluginOverlayBounds {
  x: number
  y: number
  width: number
  height: number
}

/** A display's usable rectangle, in screen (DIP) space — the shape of
 *  Electron's `display.workArea`, kept structural so tests can fake it. */
export interface OverlayWorkArea {
  x: number
  y: number
  width: number
  height: number
}

function finite(n: unknown): n is number {
  return typeof n === 'number' && Number.isFinite(n)
}

/** Renderer-reported bounds → validated, rounded, min-clamped bounds.
 *  Returns null for anything malformed (NaN, missing fields, non-object).
 *  Used by the set-bounds/set-size IPC handlers before any native call. */
export function normalizeOverlayBounds(value: unknown): PluginOverlayBounds | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const candidate = value as Partial<Record<keyof PluginOverlayBounds, unknown>>

  if (!finite(candidate.x) || !finite(candidate.y) || !finite(candidate.width) || !finite(candidate.height)) {
    return null
  }

  return {
    x: Math.round(candidate.x),
    y: Math.round(candidate.y),
    width: Math.max(PLUGIN_OVERLAY_MIN_WIDTH, Math.round(candidate.width)),
    height: Math.max(PLUGIN_OVERLAY_MIN_HEIGHT, Math.round(candidate.height))
  }
}

/** Stored bounds → valid bounds or null. Stricter than normalize: a stored
 *  record with a bogus shape or an under-min size is discarded wholesale
 *  (treated as missing → spawn defaults), never coerced upward.
 *  `minSize` differs by posture: the card enforces the window minimums, while
 *  a stored mascot record may carry a smaller size (the stored SIZE is never
 *  trusted for the mascot posture — the fixed mascot size wins — but a bogus
 *  tiny record is discarded wholesale so a default spot is used). */
export function validateStoredBounds(value: unknown, minSize: number): PluginOverlayBounds | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const candidate = value as Partial<Record<keyof PluginOverlayBounds, unknown>>

  if (!finite(candidate.x) || !finite(candidate.y) || !finite(candidate.width) || !finite(candidate.height)) {
    return null
  }

  const width = Math.round(candidate.width)
  const height = Math.round(candidate.height)

  if (width < minSize || height < minSize) {
    return null
  }

  return { x: Math.round(candidate.x), y: Math.round(candidate.y), width, height }
}

function intersectsAnyArea(bounds: PluginOverlayBounds, areas: OverlayWorkArea[]): boolean {
  return areas.some(
    area =>
      bounds.x < area.x + area.width &&
      bounds.x + bounds.width > area.x &&
      bounds.y < area.y + area.height &&
      bounds.y + bounds.height > area.y
  )
}

function nearestArea(bounds: PluginOverlayBounds, areas: OverlayWorkArea[]): OverlayWorkArea | null {
  let best: OverlayWorkArea | null = null
  let bestDistance = Infinity

  for (const area of areas) {
    const distance = Math.hypot(bounds.x - (area.x + area.width / 2), bounds.y - (area.y + area.height / 2))

    if (distance < bestDistance) {
      best = area
      bestDistance = distance
    }
  }

  return best
}

/** Clamp bounds against the CURRENT display topology (Vicky HIGH): bounds
 *  saved while a second monitor was attached must never respawn off-screen
 *  on a single-display session. Fully-off-screen bounds move to the nearest
 *  live work area (bottom-right corner, like a fresh spawn); bounds that
 *  merely overhang an edge are pulled back inside. */
export function clampToDisplay(bounds: PluginOverlayBounds, areas: OverlayWorkArea[]): PluginOverlayBounds {
  if (areas.length === 0) {
    return bounds
  }

  const maxAreaWidth = Math.max(...areas.map(a => a.width))
  const maxAreaHeight = Math.max(...areas.map(a => a.height))
  const width = Math.min(bounds.width, maxAreaWidth)
  const height = Math.min(bounds.height, maxAreaHeight)

  if (!intersectsAnyArea(bounds, areas)) {
    const area = nearestArea(bounds, areas)

    if (!area) {
      return { ...bounds, width, height }
    }

    return {
      x: Math.max(area.x, area.x + area.width - width - PLUGIN_OVERLAY_EDGE_MARGIN),
      y: Math.max(area.y, area.y + area.height - height - PLUGIN_OVERLAY_EDGE_MARGIN),
      width,
      height
    }
  }

  const area = areas.find(
    a =>
      bounds.x < a.x + a.width &&
      bounds.x + bounds.width > a.x &&
      bounds.y < a.y + a.height &&
      bounds.y + bounds.height > a.y
  )

  if (!area) {
    return { ...bounds, width, height }
  }

  return {
    x: Math.min(Math.max(bounds.x, area.x), Math.max(area.x, area.x + area.width - width)),
    y: Math.min(Math.max(bounds.y, area.y), Math.max(area.y, area.y + area.height - height)),
    width,
    height
  }
}

/** Default spawn bounds: bottom-right of the given work area. */
export function defaultOverlayBounds(area?: OverlayWorkArea): PluginOverlayBounds {
  const width = Math.min(PLUGIN_OVERLAY_DEFAULT_WIDTH, area?.width ?? Infinity)
  const height = Math.min(PLUGIN_OVERLAY_DEFAULT_HEIGHT, area?.height ?? Infinity)

  if (!area) {
    return { x: 0, y: 0, width, height }
  }

  // The margin must never push the window out of a tiny work area (the HUD
  // uses the same max(area.y, …) clamp for its bottom margin).
  return {
    x: Math.max(area.x, area.x + area.width - width - PLUGIN_OVERLAY_EDGE_MARGIN),
    y: Math.max(area.y, area.y + area.height - height - PLUGIN_OVERLAY_EDGE_MARGIN),
    width,
    height
  }
}

/** Mascot default position: bottom-right INSIDE the host rect, hugging the
 *  edge (the pencil starts where a help icon would live). Size is ignored —
 *  the fixed mascot size wins. */
export function defaultMascotBounds(host: OverlayWorkArea): PluginOverlayBounds {
  const margin = 16

  return {
    x: host.x + host.width - PLUGIN_OVERLAY_MASCOT_WIDTH - margin,
    y: host.y + host.height - PLUGIN_OVERLAY_MASCOT_HEIGHT - margin,
    width: PLUGIN_OVERLAY_MASCOT_WIDTH,
    height: PLUGIN_OVERLAY_MASCOT_HEIGHT
  }
}

/** Clamp a mascot position INTO the host rect — "only on the Hermes
 *  Desktop". If the host is smaller than the mascot, the sprite hugs the
 *  host's top-left (it must never render outside the app window). Size is
 *  forced to the mascot constants — a renderer-reported size is never
 *  trusted for the mascot posture. */
export function clampMascotBounds(pos: PluginOverlayBounds, host: OverlayWorkArea): PluginOverlayBounds {
  const maxX = Math.max(host.x, host.x + host.width - PLUGIN_OVERLAY_MASCOT_WIDTH)
  const maxY = Math.max(host.y, host.y + host.height - PLUGIN_OVERLAY_MASCOT_HEIGHT)

  return {
    x: Math.min(Math.max(pos.x, host.x), maxX),
    y: Math.min(Math.max(pos.y, host.y), maxY),
    width: PLUGIN_OVERLAY_MASCOT_WIDTH,
    height: PLUGIN_OVERLAY_MASCOT_HEIGHT
  }
}
