// HUD ↔ main-window restore policy (#88513).
//
// Opening the HUD used to snapshot `mainWindow.isVisible()` as the "should we
// bring the app window back when the HUD closes" flag. Toggling the HUD while
// the main window was minimized or hidden stored `false`, so closing the HUD
// left the app with NO visible surface at all. And the restore path called a
// bare `show()`, which leaves a minimized window minimized on several WMs.
//
// Policy: arm the restore whenever a LIVE main window exists — visibility at
// HUD-open time is irrelevant to whether the user will need a surface back —
// and restore through the focusWindow ladder (restore → show → focus).

interface WindowLike {
  isDestroyed(): boolean
}

/** Whether closing the HUD should bring the main window back. */
export function shouldArmHudRestore(mainWindow: null | undefined | WindowLike): boolean {
  return Boolean(mainWindow && !mainWindow.isDestroyed())
}

/** Restore the main window surface after the HUD closes. `focus` is the
 *  focusWindow ladder (restore + show + focus) — injected so the policy is
 *  testable without Electron. */
export function restoreMainWindowSurface<T extends WindowLike>(
  armed: boolean,
  mainWindow: null | T | undefined,
  focus: (win: null | T | undefined) => void
): boolean {
  if (!armed) {
    return false
  }

  focus(mainWindow)

  return true
}
