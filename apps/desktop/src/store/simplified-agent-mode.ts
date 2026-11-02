import { computed } from 'nanostores'

import { persistentAtom } from '@/lib/persisted'

// Global presentation preference: null means this installation has not answered the offer.
export const $agentSidebarChoice = persistentAtom<boolean | null>('hermes.desktop.simplifiedAgentMode.v1', null, {
  decode: raw => (raw === 'true' ? true : raw === 'false' ? false : null),
  encode: value => (value === null ? null : String(value))
})
export const $simplifiedAgentMode = computed($agentSidebarChoice, choice => choice === true)

export function setSimplifiedAgentMode(enabled: boolean): void {
  $agentSidebarChoice.set(enabled)
}
