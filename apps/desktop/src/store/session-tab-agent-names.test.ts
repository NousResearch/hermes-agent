import { beforeEach, describe, expect, it } from 'vitest'

import {
  $sessionTabAgentNames,
  SESSION_TAB_AGENT_NAMES_STORAGE_KEY,
  setSessionTabAgentNames
} from './session-tab-agent-names'

describe('session tab agent names preference', () => {
  beforeEach(() => {
    window.localStorage.removeItem(SESSION_TAB_AGENT_NAMES_STORAGE_KEY)
    $sessionTabAgentNames.set(false)
  })

  it('defaults off and persists changes in desktop local storage', () => {
    expect($sessionTabAgentNames.get()).toBe(false)

    setSessionTabAgentNames(true)

    expect($sessionTabAgentNames.get()).toBe(true)
    expect(window.localStorage.getItem(SESSION_TAB_AGENT_NAMES_STORAGE_KEY)).toBe('true')
  })
})
