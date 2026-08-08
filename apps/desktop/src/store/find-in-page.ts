import { atom } from 'nanostores'

export interface FindInPageState {
  active: boolean
  query: string
  matchOrdinal: number
  matchCount: number
}

const EMPTY: FindInPageState = { active: false, query: '', matchOrdinal: 0, matchCount: 0 }

export const $findInPage = atom<FindInPageState>({ ...EMPTY })

export function openFindBar(): void {
  $findInPage.set({ ...EMPTY, active: true })
}

export function closeFindBar(): void {
  // Already closed: don't re-issue stopFindInPage. Escape is a shared gesture
  // (the switcher and dialogs claim it too), so a stray second close must not
  // reach into Electron again.
  if (!$findInPage.get().active) {
    return
  }

  $findInPage.set({ ...EMPTY })
  // Clears both the search and the native highlight/selection in the page.
  void window.hermesDesktop?.stopFindInPage()
}

export function setFindQuery(query: string): void {
  const prev = $findInPage.get()

  // Never search for a closed bar. The component clears its debounce on
  // close, but a timer that already fired (or any late caller) must not
  // re-issue a find and re-highlight the page after the user pressed Escape.
  if (!prev.active) {
    return
  }

  if (!query) {
    $findInPage.set({ ...prev, query: '', matchOrdinal: 0, matchCount: 0 })
    void window.hermesDesktop?.stopFindInPage()

    return
  }

  $findInPage.set({ ...prev, query })
  void window.hermesDesktop?.findInPage(query, { forward: true, findNext: false })
}

export function findNext(): void {
  const { query } = $findInPage.get()

  if (query) {
    void window.hermesDesktop?.findInPage(query, { forward: true, findNext: true })
  }
}

export function findPrevious(): void {
  const { query } = $findInPage.get()

  if (query) {
    void window.hermesDesktop?.findInPage(query, { forward: false, findNext: true })
  }
}

/** Called by the preload bridge when `found-in-page` fires on webContents. */
export function updateFindResults(activeMatch: number, count: number): void {
  const prev = $findInPage.get()
  $findInPage.set({ ...prev, matchOrdinal: activeMatch, matchCount: count })
}

// The found-in-page subscription is process-wide, not per-mount: the FindBar
// lives in the global overlay set and the shell remounts it on a connection
// re-home (soft switch) while route changes keep it alive. Refcount the single
// bridge listener so a remount cannot stack duplicate subscriptions — every
// stacked listener would re-dispatch the same result and, worse, outlive its
// component.
let listenerRefs = 0
let detachListener: (() => void) | undefined

/**
 * Subscribe to `found-in-page` results. Returns a release fn; the underlying
 * bridge listener is installed on the first subscriber and removed when the
 * last one releases. Safe to call from an effect with a `[]` dep list.
 */
export function initFindInPageListener(): () => void {
  listenerRefs += 1

  if (listenerRefs === 1) {
    detachListener = window.hermesDesktop?.onFoundInPage?.(result => {
      updateFindResults(result.activeMatchOrdinal, result.count)
    })
  }

  let released = false

  return () => {
    // Guard double-release: React can invoke a cleanup once, but a caller
    // holding the fn shouldn't be able to drive the refcount negative.
    if (released) {
      return
    }

    released = true
    listenerRefs -= 1

    if (listenerRefs === 0) {
      detachListener?.()
      detachListener = undefined
    }
  }
}

/** Test seam: number of live bridge subscriptions (0 or 1 in practice). */
export function findInPageListenerCount(): number {
  return listenerRefs
}

/**
 * Test seam: force-detach the bridge listener and zero the refcount.
 * Production code never calls this — tests use it so one case's leaked
 * subscription can't bleed into the next.
 */
export function resetFindInPageListenerForTest(): void {
  detachListener?.()
  detachListener = undefined
  listenerRefs = 0
}

// Same refcount pattern as `initFindInPageListener`, but for the
// "main-process Ctrl/Cmd+F forwarded to renderer" channel. On Pop!_OS /
// GNOME-based Linux distros the GTK compositor grabs Ctrl+F before the
// renderer's keydown listener can fire — the main process intercepts the
// chord via `before-input-event` and emits this IPC, so the renderer can
// still open the FindBar (#81727).
let openFindBarRefs = 0
let detachOpenFindBar: (() => void) | undefined

export function initOpenFindBarListener(): () => void {
  openFindBarRefs += 1

  if (openFindBarRefs === 1) {
    detachOpenFindBar = window.hermesDesktop?.onOpenFindBarRequested?.(() => {
      openFindBar()
    })
  }

  let released = false

  return () => {
    if (released) {
      return
    }
    released = true
    openFindBarRefs -= 1

    if (openFindBarRefs === 0) {
      detachOpenFindBar?.()
      detachOpenFindBar = undefined
    }
  }
}

/** Test seam: number of live "open find bar" subscriptions. */
export function openFindBarListenerCount(): number {
  return openFindBarRefs
}

/** Test seam: detach the open-find-bar bridge listener and zero the refcount. */
export function resetOpenFindBarListenerForTest(): void {
  detachOpenFindBar?.()
  detachOpenFindBar = undefined
  openFindBarRefs = 0
}
