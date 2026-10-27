import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const HAPTICS_MUTED_STORAGE_KEY = 'hermes.desktop.hapticsMuted'

export const $hapticsMuted = atom(storedBoolean(HAPTICS_MUTED_STORAGE_KEY, false))

$hapticsMuted.subscribe(muted => persistBoolean(HAPTICS_MUTED_STORAGE_KEY, muted))

// Keep automatic cues quiet in every open window, not just the settings window.
if (typeof window !== 'undefined') {
  window.addEventListener('storage', event => {
    if (event.key === HAPTICS_MUTED_STORAGE_KEY) {
      $hapticsMuted.set(storedBoolean(HAPTICS_MUTED_STORAGE_KEY, false))
    }
  })
}

export function setHapticsMuted(muted: boolean) {
  $hapticsMuted.set(muted)
}

export function toggleHapticsMuted() {
  $hapticsMuted.set(!$hapticsMuted.get())
}
