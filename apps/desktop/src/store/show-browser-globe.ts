/**
 * Titlebar browser globe button visibility.
 *
 * One boolean, default on. When true the titlebar shows a globe icon that
 * opens the integrated browser pane. Users can hide it from
 * Settings → Appearance. Renderer-only preference, persisted to localStorage.
 */

import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const KEY = 'hermes.desktop.showBrowserGlobe.v1'

export const $showBrowserGlobe = atom<boolean>(typeof window === 'undefined' ? true : storedBoolean(KEY, true))

export function setShowBrowserGlobe(enabled: boolean): void {
  $showBrowserGlobe.set(enabled)
}

if (typeof window !== 'undefined') {
  $showBrowserGlobe.subscribe(enabled => {
    persistBoolean(KEY, enabled)
  })
}
