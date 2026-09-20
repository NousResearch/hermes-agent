import { describe, expect, it } from 'vitest'

import { filterVaultItems } from './vault-filter'

const items = [
  { id: '1', kind: 'login', label: 'GitHub work', origin: 'https://github.com', identifier: 'tek@nous.ai' },
  { id: '2', kind: 'payment', label: 'Visa', origin: 'https://shop.example.com', identifier: null },
  { id: '3', kind: 'address', label: 'Home', origin: null }
]

describe('filterVaultItems', () => {
  it('matches label, identifier and scheme-less origin case-insensitively', () => {
    expect(filterVaultItems(items, 'GITHUB', 'all').map(i => i.id)).toEqual(['1'])
    expect(filterVaultItems(items, 'nous.ai', 'all').map(i => i.id)).toEqual(['1'])
    expect(filterVaultItems(items, 'shop.example', 'all').map(i => i.id)).toEqual(['2'])
    expect(filterVaultItems(items, '  ', 'all')).toHaveLength(3)
  })

  it('narrows by kind before the query and never invents matches', () => {
    expect(filterVaultItems(items, '', 'payment').map(i => i.id)).toEqual(['2'])
    expect(filterVaultItems(items, 'github', 'payment')).toEqual([])
    expect(filterVaultItems(items, 'nowhere', 'all')).toEqual([])
  })
})
