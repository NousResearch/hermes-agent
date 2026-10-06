import { describe, expect, it } from 'vitest'

import { supportsLocalModelSetup } from './bot-types'

describe('Bot Marketplace model setup routing', () => {
  it('allows local scopes and rejects remote connection scopes instead of mutating the active backend', () => {
    expect(supportsLocalModelSetup('research-bot')).toBe(true)
    expect(supportsLocalModelSetup({ connectionId: 'local', profile: 'research-bot' })).toBe(true)
    expect(supportsLocalModelSetup({ connectionId: 'homelab', profile: 'research-bot' })).toBe(false)
  })
})
