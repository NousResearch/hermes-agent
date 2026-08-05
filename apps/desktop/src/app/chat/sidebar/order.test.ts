import { describe, expect, it } from 'vitest'

import { clusterByTopic, orderByIds, reconcileOrderIds, resolveManualSessionOrderIds, sameIds } from './order'

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
})

describe('sameIds', () => {
  it('is true only for identical ordered lists', () => {
    expect(sameIds(['a', 'b'], ['a', 'b'])).toBe(true)
    expect(sameIds(['a', 'b'], ['b', 'a'])).toBe(false)
    expect(sameIds(['a'], ['a', 'b'])).toBe(false)
  })
})

describe('clusterByTopic', () => {
  const id = (item: { id: string }) => item.id
  const title = (item: { id: string; title?: string | null }) => item.title

  it('keeps items without topic prefixes in recency order', () => {
    const items = [
      { id: 'a', title: '最新会话' },
      { id: 'b', title: null },
      { id: 'c', title: '旧会话' }
    ]
    expect(clusterByTopic(items, id, title)).toEqual(items)
  })

  it('pulls siblings with the same [Topic] prefix together', () => {
    const items = [
      { id: 'a', title: '[凭证]录入' },
      { id: 'b', title: '普通会话' },
      { id: 'c', title: '[凭证]打印' },
      { id: 'd', title: '[部署]新服务器' },
      { id: 'e', title: '[凭证]导入' }
    ]
    expect(clusterByTopic(items, id, title)).toEqual([
      { id: 'a', title: '[凭证]录入' },
      { id: 'c', title: '[凭证]打印' },
      { id: 'e', title: '[凭证]导入' },
      { id: 'b', title: '普通会话' },
      { id: 'd', title: '[部署]新服务器' }
    ])
  })

  it('keeps unprefixed items in their original slots between clusters', () => {
    const items = [
      { id: 'new', title: '新会话' },
      { id: 'p1', title: '[业务]流水' },
      { id: 'mid', title: '中间会话' },
      { id: 'p2', title: '[业务]代账' },
      { id: 'old', title: '旧会话' }
    ]
    expect(clusterByTopic(items, id, title)).toEqual([
      { id: 'new', title: '新会话' },
      { id: 'p1', title: '[业务]流水' },
      { id: 'p2', title: '[业务]代账' },
      { id: 'mid', title: '中间会话' },
      { id: 'old', title: '旧会话' }
    ])
  })

  it('does not reorder siblings within a cluster (baseline order wins)', () => {
    const items = [
      { id: 'older', title: '[审计]旧' },
      { id: 'newer', title: '[审计]新' },
      { id: 'plain', title: '无前缀' }
    ]
    expect(clusterByTopic(items, id, title)).toEqual(items)
  })

  it('handles titles that look like prefixes but are not bracket-wrapped', () => {
    const items = [
      { id: 'a', title: '凭证]残缺' },
      { id: 'b', title: '[完整' }
    ]
    expect(clusterByTopic(items, id, title)).toEqual(items)
  })
})
