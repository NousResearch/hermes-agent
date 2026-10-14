import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'
import { modeBound } from '@/store/interface-mode'

const REASONING_COLLAPSED_BY_DEFAULT_STORAGE_KEY = 'hermes.desktop.reasoning.collapsedByDefault'

/** Desktop-local presentation preference; shared backend config must not be changed by a single window.
 *  Simple mode rests reasoning collapsed without touching this preference. */
const $reasoningCollapsedByDefaultPref = atom(storedBoolean(REASONING_COLLAPSED_BY_DEFAULT_STORAGE_KEY, false))

$reasoningCollapsedByDefaultPref.subscribe(value => persistBoolean(REASONING_COLLAPSED_BY_DEFAULT_STORAGE_KEY, value))

export const $reasoningCollapsedByDefault = modeBound(
  'reasoningCollapsedByDefault',
  $reasoningCollapsedByDefaultPref,
  value => $reasoningCollapsedByDefaultPref.set(value)
)

export function setReasoningCollapsedByDefault(value: boolean) {
  $reasoningCollapsedByDefault.set(value)
}

// Mirrors `display.show_reasoning` (Settings → Chat → Reasoning Blocks). On
// by default like DEFAULT_CONFIG; a quoted "false" in config.yaml still
// means off (sibling: display-timestamps.ts).
export const $showReasoning = atom(true)

export function setShowReasoningFromConfig(value: unknown): void {
  $showReasoning.set(!(value === false || value === 'false' || value === 0))
}
