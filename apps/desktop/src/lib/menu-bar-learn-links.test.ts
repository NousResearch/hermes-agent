import { describe, expect, it } from 'vitest'

import {
  COMMUNITY_LEARN_LINKS,
  isAllowlistedLearnUrl,
  OFFICIAL_LEARN_LINKS,
  visibleLearnLinks
} from './menu-bar-learn-links'

describe('menu-bar-learn-links', () => {
  it('keeps official links before community links', () => {
    const links = visibleLearnLinks()
    const firstCommunity = links.findIndex(link => link.kind === 'community')
    const lastOfficial = links.map(link => link.kind).lastIndexOf('official')
    expect(lastOfficial).toBeGreaterThanOrEqual(0)
    expect(firstCommunity).toBeGreaterThan(lastOfficial)
  })

  it('only allowlists https official/community hosts', () => {
    for (const link of [...OFFICIAL_LEARN_LINKS, ...COMMUNITY_LEARN_LINKS]) {
      expect(link.url.startsWith('https://')).toBe(true)
      expect(isAllowlistedLearnUrl(link.url)).toBe(true)
    }

    expect(isAllowlistedLearnUrl('http://hermesbible.com/')).toBe(false)
    expect(isAllowlistedLearnUrl('https://evil.example/')).toBe(false)
  })
})
