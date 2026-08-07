/**
 * Minimize-to-tray behavior.
 *
 * When enabled (off by default), closing the Hermes window hides it in the
 * system tray instead of quitting the app — the same contract as most IM
 * clients. The main process owns the actual Tray + the `close` handler; the
 * renderer owns this toggle and mirrors it to the main process over IPC.
 *
 * macOS is intentionally excluded on the main side (the Dock already provides
 * hide-in-background behavior), so toggling this there is a no-op.
 */

import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const KEY = 'hermes.desktop.minimizeToTray.v1'

const read = (): boolean => storedBoolean(KEY, false)

export const $minimizeToTray = atom<boolean>(typeof window === 'undefined' ? false : read())

export function setMinimizeToTray(enabled: boolean): void {
  $minimizeToTray.set(Boolean(enabled))
}

if (typeof window !== 'undefined') {
  $minimizeToTray.subscribe(enabled => {
    persistBoolean(KEY, enabled)
    window.hermesDesktop?.setMinimizeToTray?.({ enabled })
  })
}
