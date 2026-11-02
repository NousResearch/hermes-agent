import { computed } from 'nanostores'

import { persistentAtom } from '@/lib/persisted'

import { DESKTOP_BUTTON_ACTIONS, recordAction, recordFeatureToggle } from './desktop-metrics'

// Global presentation preference: null means this installation has not answered the offer.
export const $agentSidebarChoice = persistentAtom<boolean | null>('hermes.desktop.simplifiedAgentMode.v1', null, {
  decode: raw => (raw === 'true' ? true : raw === 'false' ? false : null),
  encode: value => (value === null ? null : String(value))
})
export const $simplifiedAgentMode = computed($agentSidebarChoice, choice => choice === true)

export function setSimplifiedAgentMode(enabled: boolean, source: 'settings' | 'offer' | 'dismiss' = 'settings'): void {
  const previous = $agentSidebarChoice.get()

  if (previous === enabled) {
    return
  }

  $agentSidebarChoice.set(enabled)
  const actions = DESKTOP_BUTTON_ACTIONS

  const action = enabled
    ? source === 'offer'
      ? actions.sidebarEnableOffer
      : actions.sidebarEnableSettings
    : previous === true
      ? actions.sidebarDisable
      : source === 'dismiss'
        ? actions.sidebarDismiss
        : actions.sidebarDecline

  recordAction(action, 'click')
  recordFeatureToggle('simplified_sidebar', previous === true, enabled)
}
