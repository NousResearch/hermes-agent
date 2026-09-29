import { describe, expect, it } from 'vitest'

import { JARVIS_ROLE_CONNECTIONS, selectionForRole, suggestConnections } from './connections-catalog'

describe('roles', () => {
  it('replaces the last role’s picks, keeps what was added by hand', () => {
    const afterShop = selectionForRole(['github'], 'shop', null)

    expect(afterShop).toEqual(expect.arrayContaining(['github', ...JARVIS_ROLE_CONNECTIONS.shop] as string[]))

    const afterDeveloper = selectionForRole(afterShop, 'developer', 'shop')

    expect(afterDeveloper).toEqual(expect.arrayContaining([...JARVIS_ROLE_CONNECTIONS.developer]))
    expect(afterDeveloper).not.toContain('email') // shop-only pick is gone
    expect(new Set(afterDeveloper).size).toBe(afterDeveloper.length)
  })
})

describe('suggestConnections', () => {
  it('offers missing connections, the chosen ones first, and never nags about unknown or connected ones', () => {
    const status = { email: 'missing', github: 'connected', google: 'missing', messaging: 'unknown', notion: 'missing' }

    expect(suggestConnections(status, ['notion'])).toEqual(['notion', 'google'])
    expect(suggestConnections(status, [], 1)).toEqual(['google'])
    expect(suggestConnections({}, ['google'])).toEqual([])
  })
})
