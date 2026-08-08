/**
 * Window chrome mode: 'overlay' (native Window Controls Overlay / traffic
 * lights, the default) or 'app-drawn' (frame: false — the renderer paints
 * min/max/close itself via WindowControls).
 *
 * The renderer owns the value and mirrors it to the main process over IPC,
 * exactly like window translucency. Main persists its own copy so window
 * creation applies the mode before the renderer loads; `frame` can't change
 * on a live window, so a toggle applies from the next launch (the controls
 * themselves switch immediately on this side).
 */

import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const KEY = 'hermes.desktop.windowChrome.v1'

export type WindowChromeMode = 'overlay' | 'app-drawn'

const read = (): WindowChromeMode => (storedString(KEY) === 'app-drawn' ? 'app-drawn' : 'overlay')

export const $windowChrome = atom<WindowChromeMode>(typeof window === 'undefined' ? 'overlay' : read())

export function setWindowChrome(mode: WindowChromeMode): void {
  $windowChrome.set(mode === 'app-drawn' ? 'app-drawn' : 'overlay')
}

if (typeof window !== 'undefined') {
  $windowChrome.subscribe(mode => {
    persistString(KEY, mode)
    window.hermesDesktop?.setWindowChrome?.({ mode })
  })
}
