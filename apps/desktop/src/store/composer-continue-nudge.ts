import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const CONTINUE_ON_DOUBLE_ENTER_STORAGE_KEY = 'hermes.desktop.composer.continueOnDoubleEnter'

/** Desktop-local composer preference; shared backend config must not be changed by a single window. */
export const $continueOnDoubleEnter = atom(storedBoolean(CONTINUE_ON_DOUBLE_ENTER_STORAGE_KEY, false))

// `.listen`, not `.subscribe`: a subscribe replays the current value and would
// write the off default on every launch, leaving a fresh install
// indistinguishable from one that opted in and back out. Only a real change
// persists.
$continueOnDoubleEnter.listen(value => persistBoolean(CONTINUE_ON_DOUBLE_ENTER_STORAGE_KEY, value))

export function setContinueOnDoubleEnter(value: boolean) {
  $continueOnDoubleEnter.set(value)
}
