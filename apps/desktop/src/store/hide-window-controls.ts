import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const KEY = 'hermes.desktop.hideWindowControls.v1'

export const $hideWindowControls = atom<boolean>(typeof window === 'undefined' ? false : storedBoolean(KEY, false))

export function setHideWindowControls(on: boolean): void {
  $hideWindowControls.set(on)
}

if (typeof window !== 'undefined') {
  $hideWindowControls.subscribe(on => {
    persistBoolean(KEY, on)
    window.hermesDesktop?.setHideWindowControls?.(on)
  })
}
