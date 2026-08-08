/**
 * Close-to-system-tray preference (Windows only).
 *
 * A device-local preference: clicking X hides the app to the Windows system
 * tray instead of quitting, so the agent keeps running; right-clicking the
 * tray icon → "Exit" ends the app. Off by default and not offered on macOS
 * (Dock convention) or Linux. The renderer owns the persisted preference
 * (localStorage) and mirrors it to the main process, which owns the tray icon
 * and the intercept-on-close — same authority split as keep-awake.
 */

import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const KEY = 'hermes.desktop.minimizeToTray.v1'

export const $minimizeToTray = atom<boolean>(
  typeof window === 'undefined' ? false : storedBoolean(KEY, false)
)

export function setMinimizeToTray(on: boolean): void {
  $minimizeToTray.set(on)
}

if (typeof window !== 'undefined') {
  $minimizeToTray.subscribe(on => {
    persistBoolean(KEY, on)
    window.hermesDesktop?.setMinimizeToTray?.(on)
  })
}
