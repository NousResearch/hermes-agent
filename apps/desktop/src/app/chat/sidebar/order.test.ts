import { describe, expect, it } from 'vitest'

import { orderByIds, reconcileOrderIds, resolveManualSessionOrderIds, sameIds } from './order'

describe('resolveManualSessionOrderIds', () => {
  it('clears legacy auto-seeded order until the user manually reorders sessions', () => {
    expect(resolveManualSessionOrderIds(['newest', 'older'], ['older', 'newest'], false)).toEqual([])
  })

  it('keeps a manual order and surfaces newly seen sessions first', () => {
    expect(resolveManualSessionOrderIds(['newest', 'older', 'oldest'], ['oldest', 'older'], true)).toEqual([
      'newest',
      'oldest',
      'older'
    ])
  })

  it('clears manual order when none of the saved ids still exist', () => {
    expect(resolveManualSessionOrderIds(['newest'], ['gone'], true)).toEqual([])
  })
})

describe('orderByIds', () => {
  const id = (item: { id: string }) => item.id

  it('returns items untouched when no order is given', () => {
    const items = [{ id: 'a' }, { id: 'b' }]
    expect(orderByIds(items, id, [])).toBe(items)
  })

  it('reorders by the given ids and drops missing ones', () => {
    const items = [{ id: 'a' }, { id: 'b' }, { id: 'c' }]
    expect(orderByIds(items, id, ['c', 'gone', 'a'])).toEqual([{ id: 'b' }, { id: 'c' }, { id: 'a' }])
  })

  it('surfaces items absent from the order first', () => {
    const items = [{ id: 'fresh' }, { id: 'a' }, { id: 'b' }]
    expect(orderByIds(items, id, ['b', 'a'])).toEqual([{ id: 'fresh' }, { id: 'b' }, { id: 'a' }])
  })

  it('emits one item for duplicate persisted order ids', () => {
    const items = [{ id: 'a' }, { id: 'b' }]
    expect(orderByIds(items, id, ['b', 'b', 'a'])).toEqual([{ id: 'b' }, { id: 'a' }])
  })

  it('emits one item for duplicate live ids, including fresh items', () => {
    const items = [{ id: 'fresh' }, { id: 'fresh' }, { id: 'a' }, { id: 'a' }]
    expect(orderByIds(items, id, ['a'])).toEqual([{ id: 'fresh' }, { id: 'a' }])
  })
})

describe('reconcileOrderIds', () => {
  it('returns empty for no current ids', () => {
    expect(reconcileOrderIds([], ['a'])).toEqual([])
  })

  it('returns current ids when there is no saved order', () => {
    expect(reconcileOrderIds(['a', 'b'], [])).toEqual(['a', 'b'])
  })

  it('puts newly-seen ids ahead of the retained saved order', () => {
    expect(reconcileOrderIds(['fresh', 'a', 'b'], ['b', 'a', 'gone'])).toEqual(['fresh', 'b', 'a'])
  })

  it('deduplicates current and persisted ids while dropping stale ids', () => {
    expect(reconcileOrderIds(['fresh', 'fresh', 'a', 'a', 'b'], ['b', 'b', 'gone', 'a', 'a'])).toEqual([
      'fresh',
      'b',
      'a'
    ])
  })

  it('deduplicates current ids when there is no saved order', () => {
    expect(reconcileOrderIds(['a', 'a', 'b', 'a'], [])).toEqual(['a', 'b'])
  })
})

describe('sameIds', () => {
  it('is true only for identical ordered lists', () => {
    expect(sameIds(['a', 'b'], ['a', 'b'])).toBe(true)
    expect(sameIds(['a', 'b'], ['b', 'a'])).toBe(false)
    expect(sameIds(['a'], ['a', 'b'])).toBe(false)
  })
})
