import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const KEY = 'hermes.desktop.backdrop.v1'

/** Whether the faint statue image renders behind the chat transcript. */
export const $backdrop = atom(storedBoolean(KEY, true))

$backdrop.subscribe(on => persistBoolean(KEY, on))

export function setBackdrop(on: boolean) {
  $backdrop.set(on)
}
