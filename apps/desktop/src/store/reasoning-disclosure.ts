import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const REASONING_COLLAPSED_BY_DEFAULT_STORAGE_KEY = 'hermes.desktop.reasoning.collapsedByDefault'

/** Desktop-local presentation preference; shared backend config must not be changed by a single window. */
export const $reasoningCollapsedByDefault = atom(storedBoolean(REASONING_COLLAPSED_BY_DEFAULT_STORAGE_KEY, false))

$reasoningCollapsedByDefault.subscribe(value => persistBoolean(REASONING_COLLAPSED_BY_DEFAULT_STORAGE_KEY, value))

export function setReasoningCollapsedByDefault(value: boolean) {
  $reasoningCollapsedByDefault.set(value)
}

/**
 * `display.show_reasoning` — Settings → Chat → Reasoning Blocks.
 *
 * Off by default, matching gateway/CLI (`display.show_reasoning: false`).
 * When false the transcript is answer-only: no Thinking disclosure and no
 * tool-run chrome (#49664, #85110). Quoted YAML `'false'` is off.
 */
export const $showReasoning = atom(false)

export function setShowReasoningFromConfig(value: unknown): void {
  $showReasoning.set(value === true || value === 'true' || value === 1 || value === '1')
}
