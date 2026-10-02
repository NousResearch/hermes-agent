/**
 * Always-open-links-externally — a device-local preference.
 *
 * When enabled, every web link the user clicks opens in the system browser
 * instead of the in-app Browser pane. Off by default so the in-app pane stays
 * the default for reading a doc without context-switching; ⌘/Ctrl-click (or
 * middle-click) still escapes to the system browser regardless. Clear cases
 * that always go to the OS (mailto:/file:/custom schemes, connector
 * authorization links, the HUD which has no browser pane) are unaffected.
 *
 * This atom backs the Settings → Advanced toggle. It is device-local on
 * purpose (each computer keeps its own preference), matching keep-awake and
 * disable-F12 — there is no config.yaml key.
 */
import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const KEY = 'hermes.desktop.openLinksExternally.v1'

export const $openLinksExternally = atom<boolean>(typeof window === 'undefined' ? false : storedBoolean(KEY, false))

export function setOpenLinksExternally(on: boolean): void {
  $openLinksExternally.set(on)
}

if (typeof window !== 'undefined') {
  $openLinksExternally.subscribe(on => {
    persistBoolean(KEY, on)
  })
}
