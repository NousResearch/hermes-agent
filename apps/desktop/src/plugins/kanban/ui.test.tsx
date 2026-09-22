import { describe, expect, it } from 'vitest'

import type { KanbanTask } from './types'
import {
  childrenIndicator,
  dependencyState,
  fmtSecs,
  matchesProfileTab,
  runtimeCapBadge,
  staleBlocked,
  TENANT_HUES,
  tenantColor,
  tenantLabel,
  tenantTabList
} from './ui'

const task = (over: Partial<KanbanTask>): KanbanTask => ({ id: 't_1', status: 'todo', title: 'card', ...over })

/** Stand-in for the board's id → task resolver. */
const lookupOf = (map: Record<string, Partial<KanbanTask>>) => (id: string) => {
  const found = map[id]

  return found ? task({ id, ...found }) : undefined
}

describe('runtime cap badge', () => {
  it('stays off without a cap, without a start clock, or off the running lane', () => {
    expect(runtimeCapBadge(task({ max_runtime_seconds: 900, started_at: 1_000 }), 9_999)).toBeNull()
    expect(runtimeCapBadge(task({ started_at: 1_000, status: 'running' }), 9_999)).toBeNull()
    expect(runtimeCapBadge(task({ max_runtime_seconds: 900, status: 'running' }), 9_999)).toBeNull()
    expect(runtimeCapBadge(task({ max_runtime_seconds: 900, started_at: 1_000, status: 'blocked' }), 9_999)).toBeNull()
  })

  it('warns past half the cap and flags over-cap runs', () => {
    const capped = task({ max_runtime_seconds: 900, started_at: 1_000, status: 'running' })

    expect(runtimeCapBadge(capped, 1_450)).toBeNull() // exactly half — not yet
    expect(runtimeCapBadge(capped, 1_451)).toMatchObject({ cap: 900, elapsed: 451, kind: 'near' })
    expect(runtimeCapBadge(capped, 1_900)).toMatchObject({ kind: 'near' }) // at the cap it flips next tick
    expect(runtimeCapBadge(capped, 1_901)).toMatchObject({ kind: 'over' })
  })

  it('formats durations off the shared bucketing', () => {
    expect(fmtSecs(45)).toBe('45s')
    expect(fmtSecs(1_081)).toBe('18m')
    expect(fmtSecs(7_200)).toBe('2h')
  })
})

describe('stale blocked dot', () => {
  const now = 2_000_000

  it('shows only for blocked cards quiet for more than a day', () => {
    expect(staleBlocked(task({ last_event_at: now - 86_401, status: 'blocked' }), now)).toBe(true)
    expect(staleBlocked(task({ last_event_at: now - 86_400, status: 'blocked' }), now)).toBe(false)
    expect(staleBlocked(task({ last_event_at: now - 86_401, status: 'ready' }), now)).toBe(false)
    expect(staleBlocked(task({ status: 'blocked' }), now)).toBe(false)
  })
})

describe('card dependency rail', () => {
  it('stays off a card with no parents', () => {
    const dep = dependencyState(task({}), lookupOf({}))

    expect(dep.rail).toEqual([])
    expect(dep.waiting).toBe(false)
    expect(dep.blockedBy).toEqual([])
  })

  it('keeps one segment per parent, in board order', () => {
    const dep = dependencyState(
      task({ links: { children: [], parents: ['t_b', 't_a'] } }),
      lookupOf({ t_a: { status: 'done' }, t_b: { status: 'blocked' } })
    )

    expect(dep.rail.map(parent => parent.id)).toEqual(['t_b', 't_a'])
    expect(dep.rail.map(parent => parent.segment)).toEqual(['unfinished', 'done'])
  })

  it('does not gate a card whose parents all finished or were archived', () => {
    const dep = dependencyState(
      task({ links: { children: [], parents: ['t_a', 't_b'] } }),
      lookupOf({ t_a: { status: 'done' }, t_b: { status: 'archived' } })
    )

    expect(dep.rail.map(parent => parent.segment)).toEqual(['done', 'done'])
    expect(dep.waiting).toBe(false)
    expect(dep.blockedBy).toEqual([])
  })

  it('gates the card and names only the unfinished parent', () => {
    const dep = dependencyState(
      task({ links: { children: [], parents: ['t_a', 't_b'] } }),
      lookupOf({ t_a: { status: 'done' }, t_b: { status: 'running', title: 'the long pole' } })
    )

    expect(dep.waiting).toBe(true)
    expect(dep.blockedBy).toEqual([{ id: 't_b', title: 'the long pole' }])
  })

  it('treats a parent it cannot see as unfinished, titled by its short id', () => {
    const dep = dependencyState(task({ links: { children: [], parents: ['t_gone'] } }), lookupOf({}))

    expect(dep.waiting).toBe(true)
    expect(dep.blockedBy).toEqual([{ id: 't_gone', title: 'gone' }])
  })
})

describe('child indicator dot', () => {
  const parent = task({ links: { children: ['t_a', 't_b'], parents: [] } })

  it('stays off without children', () => {
    expect(childrenIndicator(task({}), lookupOf({}))).toBeNull()
  })

  it('reads live over finished over pending', () => {
    expect(childrenIndicator(parent, lookupOf({ t_a: { status: 'done' }, t_b: { status: 'done' } }))).toBe('done')
    expect(childrenIndicator(parent, lookupOf({ t_a: { status: 'done' }, t_b: { status: 'todo' } }))).toBe('unfinished')
    expect(childrenIndicator(parent, lookupOf({ t_a: { status: 'done' }, t_b: { status: 'running' } }))).toBe('running')
  })
})

describe('tenant colour coding', () => {
  const lum = (hex: string) => {
    const channel = (c: number) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)
    const [r, g, b] = [1, 3, 5].map(i => channel(parseInt(hex.slice(i, i + 2), 16) / 255))

    return 0.2126 * r + 0.7152 * g + 0.0722 * b
  }

  const contrast = (a: string, b: string) => {
    const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x)

    return (hi + 0.05) / (lo + 0.05)
  }

  it('gives the same tenant the same colour every time', () => {
    expect(tenantColor('gongbu')).toBe(tenantColor('gongbu'))
    expect(tenantColor('menxia')).toBe(tenantColor('menxia'))
  })

  it('treats the empty tenant as neutral, labelled with an em dash', () => {
    expect(tenantColor('')).toBe('var(--ui-text-quaternary)')
    expect(tenantColor(null)).toBe('var(--ui-text-quaternary)')
    expect(tenantLabel('')).toBe('—')
    expect(tenantLabel('gongbu')).toBe('gongbu')
  })

  it('keeps the five court profiles visually apart', () => {
    const hues = ['default', 'bingbu', 'gongbu', 'menxia', 'zhongshu'].map(tenantColor)

    expect(new Set(hues).size).toBe(5)
  })

  it('offers at least eight distinct hues that clear 3:1 on both surfaces', () => {
    expect(TENANT_HUES.length).toBeGreaterThanOrEqual(8)
    expect(new Set(TENANT_HUES).size).toBe(TENANT_HUES.length)

    for (const hue of TENANT_HUES) {
      expect(contrast(hue, '#ffffff')).toBeGreaterThanOrEqual(3)
      expect(contrast(hue, '#111113')).toBeGreaterThanOrEqual(3)
    }
  })

  it('lists tabs as all + distinct names and scopes a tab to tenant or assignee', () => {
    expect(tenantTabList(['gongbu', 'bingbu', 'gongbu'])).toEqual(['', 'gongbu', 'bingbu'])
    expect(matchesProfileTab(task({ tenant: 'gongbu' }), '')).toBe(true)
    expect(matchesProfileTab(task({ tenant: 'gongbu' }), 'gongbu')).toBe(true)
    expect(matchesProfileTab(task({ tenant: 'bingbu' }), 'gongbu')).toBe(false)
    expect(matchesProfileTab(task({ tenant: null }), 'gongbu')).toBe(false)
    // A tab must agree with the lane grouping: cards assigned to gongbu show
    // up under the gongbu tab even when their tenant is empty.
    expect(matchesProfileTab(task({ assignee: 'gongbu', tenant: null }), 'gongbu')).toBe(true)
    expect(matchesProfileTab(task({ assignee: 'menxia', tenant: null }), 'gongbu')).toBe(false)
  })
})
