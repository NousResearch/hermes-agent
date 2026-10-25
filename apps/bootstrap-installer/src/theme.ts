import type { Theme } from '@tauri-apps/api/window'

/*
 * OS appearance follower.
 *
 * The installer ships no in-app theme switcher, so it tracks the system.
 * Two signals feed the decision:
 *
 *   - The Tauri window theme (`getCurrentWindow().theme()` + `onThemeChanged`).
 *     This is the authoritative one: the webview's `prefers-color-scheme` is
 *     not reliable across WebView2 / WebKitGTK, and the strict
 *     `script-src 'self'` CSP (tauri.conf.json) forbids an inline pre-paint
 *     <script> in index.html, so the earliest hook we get is this module.
 *   - The webview's own `prefers-color-scheme` media query. It fills every
 *     gap where the window theme is unknown (null) and backstops the live
 *     tracking: the one-shot Tauri read can race window creation (the window
 *     starts hidden and is shown from Rust `setup`), and a `theme-changed`
 *     event emitted before we subscribe is missed. Either failure used to
 *     leave a stale first paint frozen forever.
 *
 * Precedence is fixed: an explicit window theme always wins; the media query
 * only fills nulls. Both signals stay subscribed for the life of the page,
 * every media change re-reads the window theme (the backend may have settled
 * since the last read), and a settle re-read after the first frames covers
 * the startup race.
 *
 * We only flip the `.dark` class + `color-scheme`; the dark seed values live in
 * styles.css (:root.dark), mirroring apps/desktop's applyTheme() palette.
 */

/** Precedence rule for the two appearance signals — pure, unit-tested. */
export function resolveTheme(windowTheme: Theme | null, mediaDark: boolean): Theme {
  return windowTheme ?? (mediaDark ? 'dark' : 'light')
}

function paint(theme: Theme): void {
  const root = document.documentElement
  root.classList.toggle('dark', theme === 'dark')
  root.style.colorScheme = theme
}

/**
 * The tracker's inputs. Injectable so tests can drive every path without a
 * Tauri backend; production passes nothing and gets the live wiring.
 */
export interface ThemeTrackerDeps {
  /** One-shot read of the Tauri window theme; null when unknown (or no backend). */
  readWindowTheme: () => Promise<Theme | null>
  /** Live OS-appearance events from the backend. */
  onWindowThemeChanged: (cb: (theme: Theme) => void) => Promise<unknown>
  mediaDark: () => boolean
  onMediaDarkChanged: (cb: () => void) => void
  applyTheme: (theme: Theme) => void
  /** Runs cb after the window has presented its first frames — heals a
   *  one-shot read that raced window creation. */
  afterFirstFrames: (cb: () => void) => void
}

function trackMedia(mql: MediaQueryList, cb: () => void): void {
  mql.addEventListener('change', () => cb())
}

function deferPastFirstFrames(cb: () => void): void {
  if (typeof requestAnimationFrame === 'function') {
    requestAnimationFrame(() => requestAnimationFrame(cb))
  } else {
    setTimeout(cb, 0)
  }
}

/**
 * Live dependencies: Tauri window theme + webview media query. The Tauri API
 * is imported lazily so a missing/broken backend fails closed to the media
 * query instead of breaking module load.
 */
async function liveDeps(): Promise<ThemeTrackerDeps> {
  const mql = window.matchMedia('(prefers-color-scheme: dark)')
  const { getCurrentWindow } = await import('@tauri-apps/api/window')
  const win = getCurrentWindow()

  return {
    readWindowTheme: () => win.theme(),
    onWindowThemeChanged: cb => win.onThemeChanged(({ payload }) => cb(payload)),
    mediaDark: () => mql.matches,
    onMediaDarkChanged: cb => trackMedia(mql, cb),
    applyTheme: paint,
    afterFirstFrames: deferPastFirstFrames
  }
}

/** Media-only dependencies for plain-browser contexts (dev preview). */
function mediaOnlyDeps(): ThemeTrackerDeps {
  const mql = window.matchMedia('(prefers-color-scheme: dark)')

  return {
    readWindowTheme: () => Promise.resolve(null),
    onWindowThemeChanged: () => Promise.resolve(),
    mediaDark: () => mql.matches,
    onMediaDarkChanged: cb => trackMedia(mql, cb),
    applyTheme: paint,
    afterFirstFrames: deferPastFirstFrames
  }
}

/**
 * Track the OS appearance for the life of the page. Every signal re-resolves
 * through the window theme first — the backend is authoritative, the media
 * query fills nulls — and re-reading (rather than latching the first value)
 * heals a one-shot read that raced startup.
 */
export async function watchTheme(deps?: ThemeTrackerDeps): Promise<void> {
  const tracker = deps ?? (await liveDeps().catch(() => mediaOnlyDeps()))

  // Every signal re-resolves through the window theme first (see above).
  // The last explicit reading latches: a transient IPC failure must not
  // drop a working explicit theme back to the (unreliable on some
  // webviews) media query. A genuine null simply leaves the latch empty
  // and the media query drives, as before.
  let lastWindowTheme: Theme | null = null
  const refresh = async (): Promise<void> => {
    try {
      const windowTheme = await tracker.readWindowTheme()

      if (windowTheme) {
        lastWindowTheme = windowTheme
      }
    } catch {
      // A denied/failing window command keeps the last explicit reading
      // (or the media query when there never was one).
    }

    tracker.applyTheme(resolveTheme(lastWindowTheme, tracker.mediaDark()))
  }

  await refresh()

  try {
    await tracker.onWindowThemeChanged(() => void refresh())
  } catch {
    // No backend theme events; the media listener below still tracks.
  }

  tracker.onMediaDarkChanged(() => void refresh())
  tracker.afterFirstFrames(() => void refresh())
}

// Best-effort synchronous first paint from the media query so the very first
// frame is already in the right mode; watchTheme() refines it once the
// backend answers. Guarded so non-DOM imports (unit tests) don't crash.
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  paint(resolveTheme(null, window.matchMedia('(prefers-color-scheme: dark)').matches))
}
