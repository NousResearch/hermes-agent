import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

// Plan mode (Cursor-style): when on, each submitted message is routed through
// the `/plan` skill — the agent writes an actionable markdown plan to
// `.hermes/plans/` instead of executing. This is purely a composer-side
// affordance: it prepends the `/plan` directive to the message text, so it is
// cache-safe (no system-prompt mutation, no toolset swap mid-conversation).

const PLAN_MODE_KEY = 'hermes.desktop.planMode'

export const $planMode = atom(storedBoolean(PLAN_MODE_KEY, false))

$planMode.subscribe(on => persistBoolean(PLAN_MODE_KEY, on))

export function setPlanMode(on: boolean): void {
  $planMode.set(on)
}

export function togglePlanMode(): void {
  $planMode.set(!$planMode.get())
}

/**
 * Apply plan-mode routing to a raw composer submission. When plan mode is on
 * and the text isn't already a slash command (so an explicit `/help`, `/resume`,
 * etc. still wins), prefix it with `/plan ` so the gateway dispatches the plan
 * skill. A bare empty submission is left untouched.
 */
export function applyPlanMode(text: string): string {
  if (!$planMode.get()) {
    return text
  }

  const trimmedStart = text.trimStart()

  // Don't double-prefix, and never hijack an explicit slash command.
  if (!trimmedStart || trimmedStart.startsWith('/')) {
    return text
  }

  return `/plan ${text}`
}
