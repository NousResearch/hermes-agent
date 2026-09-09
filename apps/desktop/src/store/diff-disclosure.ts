import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const STORAGE_KEY = 'hermes.desktop.diffs.collapseWhenSettled'

/** Local presentation preference; individual disclosure choices still win. */
export const $collapseSettledDiffs = atom(storedBoolean(STORAGE_KEY, true))

$collapseSettledDiffs.subscribe(value => persistBoolean(STORAGE_KEY, value))

export function setCollapseSettledDiffs(value: boolean) {
  $collapseSettledDiffs.set(value)
}
