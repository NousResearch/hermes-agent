import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const SHOW_TURN_PERF_STORAGE_KEY = 'hermes.desktop.turnPerf.show'

/** Desktop-local presentation preference: show tokens/sec + count on completed assistant replies. */
export const $showTurnPerf = atom(storedBoolean(SHOW_TURN_PERF_STORAGE_KEY, true))

$showTurnPerf.subscribe(value => persistBoolean(SHOW_TURN_PERF_STORAGE_KEY, value))

export function setShowTurnPerf(value: boolean) {
  $showTurnPerf.set(value)
}
