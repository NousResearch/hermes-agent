import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const REASONING_COLLAPSED_BY_DEFAULT_STORAGE_KEY = 'hermes.desktop.reasoning.collapsedByDefault'

/** Desktop-local presentation preference; shared backend config must not be changed by a single window. */
export const $reasoningCollapsedByDefault = atom(
  storedBoolean(REASONING_COLLAPSED_BY_DEFAULT_STORAGE_KEY, false)
)

$reasoningCollapsedByDefault.subscribe(value => persistBoolean(REASONING_COLLAPSED_BY_DEFAULT_STORAGE_KEY, value))

export function setReasoningCollapsedByDefault(value: boolean) {
  $reasoningCollapsedByDefault.set(value)
}
