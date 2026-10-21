import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

import { recordFeatureToggle } from './desktop-metrics'

const KEY = 'hermes.desktop.backdrop.v1'

/** Whether the faint statue image renders behind the chat transcript. */
export const $backdrop = atom(storedBoolean(KEY, false))

$backdrop.subscribe(on => persistBoolean(KEY, on))

export function setBackdrop(on: boolean) {
  recordFeatureToggle('backdrop', $backdrop.get(), on)
  $backdrop.set(on)
}
