import { describe, expect, it } from 'vitest'

import { normalizeSiteRule, readBlocklist, withSiteAdded, withSiteRemoved } from './browser-sites'

describe('normalizeSiteRule', () => {
  it('reduces whatever was pasted to the host the backend matches', () => {
    expect(normalizeSiteRule('https://www.Allegro.pl/oferta/123?x=1')).toBe('allegro.pl')
    expect(normalizeSiteRule('  *.example.com/path ')).toBe('example.com')
    expect(normalizeSiteRule('mbank.pl')).toBe('mbank.pl')
  })

  it('refuses what is not a host, so a typo never becomes a silent rule', () => {
    for (const bad of ['', '   ', '# comment', 'localhost', 'not a site', 'http://']) {
      expect(normalizeSiteRule(bad)).toBeNull()
    }
  })
})

describe('blocklist edits', () => {
  it('adds a site once, switches the list on, and removes by the same normalisation', () => {
    const empty = readBlocklist({})

    expect(empty).toEqual({ domains: [], enabled: false })

    const added = withSiteAdded(withSiteAdded(empty, 'https://www.Bank.pl/login'), 'bank.pl')

    expect(added).toEqual({ domains: ['bank.pl'], enabled: true })
    expect(withSiteAdded(added, 'nonsense')).toBe(added)
    expect(withSiteRemoved(added, 'WWW.bank.pl').domains).toEqual([])
  })
})
