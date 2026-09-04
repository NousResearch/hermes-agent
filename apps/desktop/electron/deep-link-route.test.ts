import { describe, expect, it } from 'vitest'

import { routeDeepLink } from './deep-link-route'

describe('routeDeepLink', () => {
  it('summons quick entry for the Copilot key press', () => {
    expect(routeDeepLink('copilot-key', 'start')).toBe('quick-entry')
  })

  it('ignores the Copilot key release so a tap does not undo the summon', () => {
    expect(routeDeepLink('copilot-key', 'stop')).toBe('ignore')
  })

  it('ignores unknown copilot-key paths instead of leaking them to the renderer', () => {
    expect(routeDeepLink('copilot-key', '')).toBe('ignore')
    expect(routeDeepLink('copilot-key', 'toggle')).toBe('ignore')
  })

  it('routes every other deep link to the renderer', () => {
    expect(routeDeepLink('blueprint', 'morning-brief')).toBe('renderer')
    expect(routeDeepLink('', '')).toBe('renderer')
  })
})
