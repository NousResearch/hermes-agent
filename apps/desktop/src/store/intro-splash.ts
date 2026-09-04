import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const KEY = 'hermes.desktop.intro-splash.v1'

/** Whether the wordmark + tagline splash renders on an empty chat. */
export const $introSplash = atom(storedBoolean(KEY, true))

$introSplash.subscribe(on => persistBoolean(KEY, on))

export function setIntroSplash(on: boolean) {
  $introSplash.set(on)
}
