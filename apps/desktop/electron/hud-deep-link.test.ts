import { describe, expect, it } from 'vitest'

import { isHudSummonDeepLink } from './hud-deep-link'

describe('isHudSummonDeepLink', () => {
  it('recognizes the compositor summon contract', () => {
    expect(isHudSummonDeepLink('hermes://hud/summon')).toBe(true)
    expect(isHudSummonDeepLink('hermes://hud/summon?profile=work')).toBe(true)
  })

  it('does not classify other deep links as HUD summons', () => {
    expect(isHudSummonDeepLink('hermes://session/abc')).toBe(false)
    expect(isHudSummonDeepLink('hermes://hud/close')).toBe(false)
    expect(isHudSummonDeepLink(null)).toBe(false)
  })
})
