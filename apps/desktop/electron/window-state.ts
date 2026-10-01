/**
 * Pure geometry helpers for window-state.json — restoring the main window's
 * size, position, and maximized flag across launches. Side-effect-free so the
 * part that actually matters (rejecting garbage + off-screen bounds) is
 * unit-testable without booting Electron; main.ts owns the file I/O and the
 * live `screen` displays.
 */

const MIN_WIDTH = 400
const MIN_HEIGHT = 620

// Keep at least this much of the window over a display work area before we trust
// a saved position, so the title bar stays grabbable after a monitor unplugs.
const MIN_VISIBLE = 48

// A stale normal-bounds snapshot written while the window was fullscreen has
// no meaningful restore-down size, so recovery (see staleFullscreenWorkArea)
// gives the window back a centered, deliberately windowed shape instead.
const RECOVERED_WINDOW_RATIO = 0.8

// Legacy snapshots (written before boundsCapturedFullScreen existed) have no
// provenance flag, so stale fullscreen-as-normal bounds can only be told apart
// from a deliberate near-fullscreen window by an exact geometry match: the
// bounds equal a display's work area (or its full bounds, where the taskbar is
// hidden) within this many pixels on every edge.
const LEGACY_EXACT_MATCH_TOLERANCE = 2

const finite = v => typeof v === 'number' && Number.isFinite(v)
const clamp = (v, lo, hi) => Math.max(lo, Math.min(v, hi))

interface SanitizedWindowState {
  width: number
  height: number
  isMaximized: boolean
  x?: number
  y?: number
  // Provenance recorded at save time: the persisted bounds were captured while
  // the window was fullscreen, the transition known to write fullscreen bounds
  // as normal ones. Absent on legacy snapshots; strict like isMaximized.
  boundsCapturedFullScreen?: boolean
}

// Parse raw JSON → clean state, or null if garbage. width/height are required
// and floored; x/y survive only as a finite pair; isMaximized is strict.
function sanitizeWindowState(raw?: any): SanitizedWindowState | null {
  if (!raw || typeof raw !== 'object' || !finite(raw.width) || !finite(raw.height)) {
    return null
  }

  const state: SanitizedWindowState = {
    width: Math.max(MIN_WIDTH, Math.round(raw.width)),
    height: Math.max(MIN_HEIGHT, Math.round(raw.height)),
    isMaximized: raw.isMaximized === true
  }

  if (raw.boundsCapturedFullScreen === true) {
    state.boundsCapturedFullScreen = true
  }

  if (finite(raw.x) && finite(raw.y)) {
    state.x = Math.round(raw.x)
    state.y = Math.round(raw.y)
  }

  return state
}

// Return the work area with the largest meaningful overlap with `bounds`.
// `displays` is Electron's screen.getAllDisplays() shape. A small sliver does
// not count: the saved position is only trusted when at least `minVisible` is
// reachable on both axes.
function matchingWorkArea(bounds, displays, minVisible = MIN_VISIBLE) {
  if (!Array.isArray(displays)) {
    return null
  }

  let best = null
  let bestArea = 0

  for (const { workArea: a } of displays) {
    if (!a) {
      continue
    }

    const x = Math.min(bounds.x + bounds.width, a.x + a.width) - Math.max(bounds.x, a.x)
    const y = Math.min(bounds.y + bounds.height, a.y + a.height) - Math.max(bounds.y, a.y)

    if (x < minVisible || y < minVisible) {
      continue
    }

    const area = x * y

    if (area > bestArea) {
      best = a
      bestArea = area
    }
  }

  return best
}

interface WindowOptions {
  width: number
  height: number
  x?: number
  y?: number
}

interface WorkArea {
  width: number
  height: number
}

function firstLaunchSize(workArea: WorkArea): WindowOptions {
  return {
    width: Math.min(clamp(Math.round(workArea.width * 0.75), 1220, 1600), workArea.width),
    height: Math.min(clamp(Math.round(workArea.height * 0.8), 800, 1000), workArea.height)
  }
}

// A stale normal-bounds snapshot written while the window was fullscreen: the
// broken transition persists fullscreen bounds as normal ones, leaving Windows
// with no meaningful restore-down size. Provenance is recorded at save time
// (boundsCapturedFullScreen, see persistWindowState in main.ts), so recovery
// here never guesses from geometry. Legacy snapshots written before that flag
// existed recover only on an unambiguous match: bounds that equal a display's
// work area (or its full bounds, when the taskbar hides) within
// LEGACY_EXACT_MATCH_TOLERANCE on every edge — a window the user deliberately
// sized to 90-something percent never matches.
function staleFullscreenWorkArea(state, displays) {
  if (
    !state ||
    state.isMaximized ||
    !finite(state.x) ||
    !finite(state.y) ||
    !finite(state.width) ||
    !finite(state.height) ||
    !Array.isArray(displays)
  ) {
    return null
  }

  if (state.boundsCapturedFullScreen !== true) {
    return legacyStaleFullscreenWorkArea(state, displays)
  }

  // Flagged at save time: recover to a centered windowed size on the work
  // area the fullscreen bounds actually covered.
  return (
    displays.find(({ workArea: a } = {}) => {
      if (!a || !finite(a.x) || !finite(a.y) || !finite(a.width) || !finite(a.height)) {
        return false
      }

      const x = Math.min(state.x + state.width, a.x + a.width) - Math.max(state.x, a.x)
      const y = Math.min(state.y + state.height, a.y + a.height) - Math.max(state.y, a.y)

      return x > 0 && y > 0
    })?.workArea ?? null
  )
}

// Exact-match recovery for legacy snapshots with no provenance flag.
function legacyStaleFullscreenWorkArea(state, displays) {
  const t = LEGACY_EXACT_MATCH_TOLERANCE

  const exact = ({ x, y, width, height }) =>
    Math.abs(state.x - x) <= t &&
    Math.abs(state.y - y) <= t &&
    Math.abs(state.width - width) <= t &&
    Math.abs(state.height - height) <= t

  for (const { workArea: a, bounds: b } of displays) {
    for (const candidate of [a, b]) {
      if (
        candidate &&
        finite(candidate.x) &&
        finite(candidate.y) &&
        finite(candidate.width) &&
        finite(candidate.height) &&
        exact(candidate)
      ) {
        return a
      }
    }
  }

  return null
}

function computeWindowOptions(state: WindowOptions, displays, platform = process.platform): WindowOptions {
  const opts: WindowOptions = { width: state.width, height: state.height }

  const cap = (Array.isArray(displays) ? displays : []).reduce(
    (m, { workArea: a } = {}) =>
      a && finite(a.width) && finite(a.height)
        ? { width: Math.max(m.width, a.width), height: Math.max(m.height, a.height) }
        : m,
    { width: 0, height: 0 }
  )

  if (cap.width && cap.height) {
    opts.width = clamp(opts.width, MIN_WIDTH, cap.width)
    opts.height = clamp(opts.height, MIN_HEIGHT, cap.height)
  }

  // The motivating restore-down failure is Windows-specific. Keeping the
  // geometry heuristic there avoids rewriting deliberate near-fullscreen
  // layouts from tiling WMs or user placement on macOS/Linux. This early return
  // intentionally omits stale x/y so Electron centers the recovered window.
  const staleWorkArea = platform === 'win32' ? staleFullscreenWorkArea(state, displays) : null
  if (staleWorkArea) {
    opts.width = clamp(Math.round(staleWorkArea.width * RECOVERED_WINDOW_RATIO), MIN_WIDTH, staleWorkArea.width)
    opts.height = clamp(Math.round(staleWorkArea.height * RECOVERED_WINDOW_RATIO), MIN_HEIGHT, staleWorkArea.height)
    return opts
  }

  if (finite(state.x) && finite(state.y)) {
    const workArea = matchingWorkArea({ x: state.x, y: state.y, width: opts.width, height: opts.height }, displays)

    if (workArea) {
      opts.width = clamp(opts.width, MIN_WIDTH, workArea.width)
      opts.height = clamp(opts.height, MIN_HEIGHT, workArea.height)
      opts.x = clamp(state.x, workArea.x, workArea.x + workArea.width - opts.width)
      opts.y = clamp(state.y, workArea.y, workArea.y + workArea.height - opts.height)
    }
  }

  return opts
}

// Trailing debounce: collapse a burst of resize/move events (Linux fires many
// mid-drag) into a single run `delayMs` after the last. `.flush()` runs now and
// cancels the pending timer — used on close, before the window is gone.
function debounce(fn, delayMs) {
  let timer = null

  const debounced = () => {
    clearTimeout(timer)
    timer = setTimeout(() => {
      timer = null
      fn()
    }, delayMs)
  }

  debounced.flush = () => {
    clearTimeout(timer)
    timer = null
    fn()
  }

  return debounced
}

// The geometry events worth persisting from. `moved` and `resized` — the
// settled-once-per-drag pair — are macOS/Windows only (Electron tags them
// `@platform darwin,win32`), so a window bound to those alone never saves its
// place on Linux: the events simply never arrive. `move` and `resize` carry no
// platform tag and fire everywhere. They also fire continuously mid-drag, which
// is what the trailing debounce above is for — and once a burst collapses to a
// single trailing run, the settled events add nothing the debounce hasn't
// already given us.
const GEOMETRY_EVENTS = ['move', 'resize']

// Bind `schedule` to every geometry event, on a BrowserWindow or any emitter
// with `.on`. One call site per window so the platform reasoning above can't be
// half-applied to one window and not the other.
function bindGeometryPersistence(win, schedule) {
  for (const event of GEOMETRY_EVENTS) {
    win.on(event, schedule)
  }
}

export {
  bindGeometryPersistence,
  computeWindowOptions,
  debounce,
  firstLaunchSize,
  GEOMETRY_EVENTS,
  matchingWorkArea,
  MIN_HEIGHT,
  MIN_VISIBLE,
  MIN_WIDTH,
  sanitizeWindowState
}
