import { afterEach, describe, expect, it } from 'vitest'

import {
  $sidebarHiddenNavIds,
  $sidebarNavOrderIds,
  orderSidebarNav,
  setSidebarNavHidden,
  setSidebarNavOrder
} from './sidebar-nav'

afterEach(() => {
  $sidebarHiddenNavIds.set([])
  $sidebarNavOrderIds.set([])
})

const rows = [{ id: 'a' }, { id: 'b' }, { id: 'c' }, { id: 'd' }]

describe('orderSidebarNav', () => {
  it('keeps the default order when nothing is hidden or ordered', () => {
    expect(orderSidebarNav(rows, [], []).map(r => r.id)).toEqual(['a', 'b', 'c', 'd'])
  })

  it('drops hidden ids and ignores unknown ones', () => {
    expect(orderSidebarNav(rows, ['b', 'missing'], []).map(r => r.id)).toEqual(['a', 'c', 'd'])
  })

  it('names ordered ids first, in that order; unnamed rows keep their default order after', () => {
    expect(orderSidebarNav(rows, [], ['c', 'a']).map(r => r.id)).toEqual(['c', 'a', 'b', 'd'])
  })

  it('an ordered id that is also hidden stays hidden (hide wins)', () => {
    expect(orderSidebarNav(rows, ['c'], ['c', 'a']).map(r => r.id)).toEqual(['a', 'b', 'd'])
  })

  it('preserves the row objects themselves, not copies', () => {
    const [first] = orderSidebarNav(rows, [], ['b'])

    expect(first).toBe(rows[1])
  })
})

describe('sidebar nav preference setters', () => {
  it('hide is idempotent and reversible', () => {
    setSidebarNavHidden('cron')
    setSidebarNavHidden('cron')
    expect($sidebarHiddenNavIds.get()).toEqual(['cron'])

    setSidebarNavHidden('cron', false)
    setSidebarNavHidden('cron', false)
    expect($sidebarHiddenNavIds.get()).toEqual([])
  })

  it('ignores blank ids', () => {
    setSidebarNavHidden('   ')
    expect($sidebarHiddenNavIds.get()).toEqual([])
  })

  it('setOrder trims, drops blanks and dedupes', () => {
    setSidebarNavOrder([' b ', '', 'a', 'b'])

    expect($sidebarNavOrderIds.get()).toEqual(['b', 'a'])
  })
})
