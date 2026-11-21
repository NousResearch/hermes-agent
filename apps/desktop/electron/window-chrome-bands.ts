// The live side of window-chrome.ts: which band each open window is drawn
// with, and moving every one of them when the shell choice changes (no
// restart). Main supplies the platform overlay/notify step as `relayout`.
//
// The first window is created from the shared store read at startup. The
// shell choice has been a shared key since it existed, so a renderer cache
// that says classic while the store has nothing needs a failed host write;
// even then the renderer's sync lands the value and `applyPreferences` moves
// the window to its band live, so nothing is copied ahead of window creation.

import {
  portalShellPreferred,
  windowButtonPosition,
  type WindowChromeKind,
  windowTitlebarHeight
} from './window-chrome'

type ButtonPosition = ReturnType<typeof windowButtonPosition>

/** The slice of BrowserWindow this needs. */
export interface BandWindow {
  isDestroyed(): boolean
  once(event: 'closed', listener: () => void): unknown
  setWindowButtonPosition?(position: ButtonPosition): void
}

interface WindowChromeBandsOptions<W> {
  isMac: boolean
  /** The shell choice read from the shared UI preferences before the first window exists. */
  portal: boolean
  /** Re-apply the platform overlay and tell the renderer where its controls now sit. */
  relayout: (win: W) => void
}

export function createWindowChromeBands<W extends BandWindow>({ isMac, portal, relayout }: WindowChromeBandsOptions<W>) {
  let portalShell = portal
  const bandWindows = new Set<W>()
  const classicChromeWindows = new WeakSet<object>()

  const titlebarHeightFor = (kind: WindowChromeKind) => windowTitlebarHeight({ kind, portal: portalShell })
  const kindOf = (win?: null | object): WindowChromeKind => (win && classicChromeWindows.has(win) ? 'browser' : 'chat')
  const buttonPositionFor = (kind: WindowChromeKind) => windowButtonPosition(titlebarHeightFor(kind))

  function placeButtons(win: W) {
    if (isMac && !win.isDestroyed()) {
      win.setWindowButtonPosition?.(buttonPositionFor(kindOf(win)))
    }
  }

  return {
    buttonPositionFor,
    kindOf,
    titlebarHeightFor,
    /** Register a window drawn with a titlebar band and seat its traffic lights. */
    track(win: W, kind: WindowChromeKind) {
      if (kind === 'browser') {
        classicChromeWindows.add(win)
      }

      bandWindows.add(win)
      win.once('closed', () => bandWindows.delete(win))
      placeButtons(win)
    },
    /** The shared UI preferences changed: move every open window to its new band. */
    applyPreferences(values: Readonly<Record<string, string>> | null | undefined) {
      const next = portalShellPreferred(values)

      if (next === portalShell) {
        return
      }

      portalShell = next

      for (const win of bandWindows) {
        placeButtons(win)
        relayout(win)
      }
    }
  }
}
