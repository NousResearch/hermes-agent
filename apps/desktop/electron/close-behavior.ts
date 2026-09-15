import fs from 'node:fs'
import path from 'node:path'

/**
 * Close-to-tray behavior for the desktop app (Windows).
 *
 * On Windows the window's close button can hide the app to the system tray
 * instead of quitting. The user picks the behavior in Settings → Appearance →
 * "When closing the window"; the choice (plus a one-time balloon flag) is
 * persisted to a small JSON file under userData.
 *
 * Platform scope: implemented and tested on Windows only. Linux and macOS are
 * UNTESTED — they keep the stock close behavior (close = quit) because
 * `shouldHideToTray` gates on `isWindows`. Tray support itself (Electron
 * `Tray`) is cross-platform, but the close-interception path has only been
 * exercised on Windows.
 */

export type CloseBehavior = 'tray' | 'quit'

export interface CloseBehaviorState {
  /** What the close button does: hide to tray ('tray') or quit ('quit'). */
  mode: CloseBehavior
  /**
   * Whether the one-time "minimized to tray" balloon has already been shown.
   * Persisted so a restart does not re-notify the user on the next close.
   */
  trayNotified: boolean
}

export const DEFAULT_CLOSE_BEHAVIOR: CloseBehaviorState = {
  mode: 'tray',
  trayNotified: false
}

/** Read the persisted state; any missing/corrupt file falls back to defaults. */
export function readCloseBehaviorState(filePath: string): CloseBehaviorState {
  try {
    const raw = JSON.parse(fs.readFileSync(filePath, 'utf8')) as Partial<CloseBehaviorState>

    return {
      mode: raw?.mode === 'quit' ? 'quit' : 'tray',
      trayNotified: raw?.trayNotified === true
    }
  } catch {
    return { ...DEFAULT_CLOSE_BEHAVIOR }
  }
}

/** Persist the state (best-effort: a write failure must never block closing). */
export function writeCloseBehaviorState(filePath: string, state: CloseBehaviorState): void {
  try {
    fs.mkdirSync(path.dirname(filePath), { recursive: true })
    fs.writeFileSync(filePath, JSON.stringify(state, null, 2))
  } catch {
    // best-effort — the in-memory behavior still applies for this run
  }
}

/** Pure decision: should the close button hide to the tray instead of quitting? */
export function shouldHideToTray(input: {
  isWindows: boolean
  isQuitting: boolean
  mode: CloseBehavior
}): boolean {
  return input.isWindows && !input.isQuitting && input.mode === 'tray'
}
