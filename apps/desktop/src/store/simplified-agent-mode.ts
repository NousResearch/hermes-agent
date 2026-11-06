import { computed } from 'nanostores'

import { persistentAtom } from '@/lib/persisted'

import { DESKTOP_BUTTON_ACTIONS, recordAction, recordFeatureToggle } from './desktop-metrics'

// The normal shell is opt-out. The prototype's offer choice is deliberately
// not inherited; this preference only records an explicit classic rollback.
export const $agentSidebarChoice = persistentAtom<boolean | null>('hermes.desktop.agentShell.v1', null, {
  decode: raw => (raw === 'true' ? true : raw === 'false' ? false : null),
  encode: value => (value === null ? null : String(value))
})
export const $simplifiedAgentMode = computed($agentSidebarChoice, choice => choice !== false)

export function setSimplifiedAgentMode(enabled: boolean): void {
  const previous = $simplifiedAgentMode.get()

  if (previous === enabled) {
    return
  }

  $agentSidebarChoice.set(enabled)
  const actions = DESKTOP_BUTTON_ACTIONS

  recordAction(enabled ? actions.sidebarEnableSettings : actions.sidebarDisable, 'click')
  recordFeatureToggle('simplified_sidebar', previous, enabled)
}
