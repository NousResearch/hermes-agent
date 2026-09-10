/**
 * Whether to force Chromium's renderer accessibility tree on at launch.
 *
 * Why this exists
 * ───────────────
 * Chromium only builds its accessibility tree once it detects an assistive
 * technology talking to it, and on Windows that detection is keyed off the
 * *screen reader* client contract (it watches for the COM calls a screen
 * reader like Narrator/NVDA makes). Dictation tools such as Wispr Flow speak
 * plain Windows UI Automation to find the focused editable control, but
 * never make those calls, so Chromium never turns the tree on. UI Automation
 * inspection then only sees the native window chrome (`Minimize`/`Maximize`/
 * `Close`) — the composer is never exposed as an editable control at all
 * (#92607).
 *
 * `app.setAccessibilitySupportEnabled(true)` forces the tree on
 * unconditionally, which is exactly the escape hatch Electron ships for
 * this: OS-level assistive tooling that doesn't identify itself as a
 * screen reader. Scoped to Windows because that's the platform where the
 * detection gap exists and was reported; forcing it elsewhere would pay the
 * tree-construction cost for no known benefit.
 */
export function shouldForceAccessibilitySupport(platform: NodeJS.Platform): boolean {
  return platform === 'win32'
}
