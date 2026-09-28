/**
 * Effective visibility of the transcript's tool feed (rows + run scaffold).
 *
 * Mirrors the gateway gate `_process_tool_chrome_enabled` (tui_gateway/server.py):
 * the feed follows `display.tool_progress`; when the user never stated a feed
 * preference, the answer-only default of `display.show_reasoning: false` still
 * hides it. An explicit `display.tool_progress` overrides that hiding — `off`
 * always silences the feed — so "execution flow without thinking" is expressible
 * (siblings: reasoning-disclosure.ts, display-timestamps.ts).
 */
import { atom, computed } from 'nanostores'

import { $showReasoning } from '@/store/reasoning-disclosure'

/** `display.tool_progress` parses to off (a bare YAML `off` reaches us as false). */
const $toolProgressOff = atom(false)

/** `display.tool_progress` is present in the config — the user stated a feed preference. */
const $toolProgressExplicit = atom(false)

export const $showToolActivity = computed(
  [$showReasoning, $toolProgressOff, $toolProgressExplicit],
  (showReasoning, progressOff, progressExplicit): boolean =>
    !progressOff && (showReasoning || progressExplicit)
)

export function setShowToolActivityFromConfig(value: unknown): void {
  $toolProgressOff.set(value === false || (typeof value === 'string' && value.trim().toLowerCase() === 'off'))
  $toolProgressExplicit.set(value !== undefined)
}
