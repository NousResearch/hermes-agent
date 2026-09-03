import { describe, expect, it } from 'vitest'

import { rosterPresentationMode } from './roster-presentation'

describe('Bot Mode roster presentation', () => {
  it('keeps group chats collapsible on a single gateway', () => {
    expect(rosterPresentationMode(false, 3)).toBe('group-section')
  })

  it('keeps gateway sections when several gateways are visible', () => {
    expect(rosterPresentationMode(true, 3)).toBe('gateway-sections')
  })

  it('uses the compact flat roster when there are no groups to fold', () => {
    expect(rosterPresentationMode(false, 0)).toBe('flat')
  })
})
