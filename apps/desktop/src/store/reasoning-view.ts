import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const KEY = 'hermes.desktop.keepReasoningExpanded.v1'

/** Whether the "Thinking" reasoning disclosure stays open after a turn finishes
 *  instead of auto-collapsing. Off by default → the streaming preview still
 *  auto-collapses on completion, as before. */
export const $keepReasoningExpanded = atom(storedBoolean(KEY, false))

$keepReasoningExpanded.subscribe(on => persistBoolean(KEY, on))

export function setKeepReasoningExpanded(on: boolean) {
  $keepReasoningExpanded.set(on)
}
