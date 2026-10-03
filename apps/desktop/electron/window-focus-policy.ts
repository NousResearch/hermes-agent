/**
 * Foreground-focus policy for the Windows focus-steal bug (#83998).
 *
 * On Windows, `BrowserWindow.show()` and `.focus()` both seize the OS
 * foreground: any native dialog another app is showing (Notepad's save
 * confirmation, a print dialog) is forcibly dismissed when Hermes calls
 * either while streaming in the background. The report's repro is a
 * `focusWindow()` call reaching the window mid-stream.
 *
 * The fix is at the call sites, not here — but the policy is pure so the
 * invariants are unit-testable without booting Electron:
 *
 * - `revealWindow` (ambient reveal, e.g. `focusWindow`'s un-hide branch)
 *   uses `showInactive()`: raising Hermes must never steal foreground from
 *   the app the user is working in.
 * - `focusWindow`'s keyboard hand-off only calls `.focus()` when the window
 *   is NOT already focused. On Windows a redundant `.focus()` on a focused
 *   window still pumps the OS SetForegroundWindow path, which is exactly
 *   the ambient steal. When the window already has focus, the call is a
 *   no-op in intent and must be a no-op in effect.
 */

export interface FocusPolicyWindow {
  isFocused: () => boolean
}

/**
 * The reveal half of `focusWindow`: a window that is hidden or minimized
 * must come back, but never by stealing the OS foreground. Callers should
 * do the restore/minimize-unminimize themselves, then call this to make the
 * window visible; it returns the action to run on the window so the policy
 * is testable without a real BrowserWindow.
 */
export function revealAction(windowVisible: boolean): 'showInactive' | 'none' {
  return windowVisible ? 'none' : 'showInactive'
}

/**
 * The keyboard half of `focusWindow`: focus only moves when the window
 * doesn't already have it. A `.focus()` call on an already-focused window
 * is a foreground pump on Windows and a no-op everywhere else — skip it.
 */
export function shouldFocusToTakeKeyboard(window: FocusPolicyWindow): boolean {
  return !window.isFocused()
}
