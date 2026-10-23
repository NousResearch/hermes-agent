import { Codecs, persistentAtom } from '@/lib/persisted'

export const SESSION_TAB_AGENT_NAMES_STORAGE_KEY = 'hermes.desktop.sessionTabAgentNames.v1'

/** Whether session tabs prefix their title with the owning agent name. */
export const $sessionTabAgentNames = persistentAtom(SESSION_TAB_AGENT_NAMES_STORAGE_KEY, false, Codecs.bool)

export function setSessionTabAgentNames(enabled: boolean): void {
  $sessionTabAgentNames.set(enabled)
}
