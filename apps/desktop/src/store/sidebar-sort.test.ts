import { beforeEach, describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { resetSidebarView, setSidebarOrdering } from './layout'
import { $sessions } from './session'
import { $sidebarSessionRankIds, sortPinnedSessions } from './sidebar-sort'

const session = (id: string, fields: Partial<SessionInfo>) =>
  ({ id, input_tokens: 0, output_tokens: 0, started_at: 0, ...fields }) as SessionInfo

beforeEach(() => {
  resetSidebarView()
  $sessions.set([])
})

describe('$sidebarSessionRankIds', () => {
  it('ranks the priciest session first', () => {
    $sessions.set([
      session('cheap', { actual_cost_usd: 0.01 }),
      session('dear', { actual_cost_usd: 2 }),
      session('estimated', { estimated_cost_usd: 0.5 })
    ])
    setSidebarOrdering('cost')

    expect($sidebarSessionRankIds.get()).toEqual(['dear', 'estimated', 'cheap'])
  })

  it('ranks by total tokens, both halves counted', () => {
    $sessions.set([
      session('small', { input_tokens: 10, output_tokens: 10 }),
      session('big', { input_tokens: 1, output_tokens: 500 })
    ])
    setSidebarOrdering('tokens')

    expect($sidebarSessionRankIds.get()).toEqual(['big', 'small'])
  })

  it('ranks by creation, newest first — the sidebar orders by recency elsewhere', () => {
    $sessions.set([session('older', { started_at: 1 }), session('newer', { started_at: 9 })])
    setSidebarOrdering('created')

    expect($sidebarSessionRankIds.get()).toEqual(['newer', 'older'])
  })

  it('leaves the default view unranked, and hands back the same array each time', () => {
    $sessions.set([session('a', { actual_cost_usd: 1 }), session('b', { actual_cost_usd: 2 })])

    const first = $sidebarSessionRankIds.get()

    $sessions.set([session('c', { actual_cost_usd: 3 })])

    expect(first).toEqual([])
    // Reference-stable, so the default sidebar never repaints on a rank it isn't using.
    expect($sidebarSessionRankIds.get()).toBe(first)
  })

  it('drops the ranking when a hand-dragged order takes over', () => {
    $sessions.set([session('a', { actual_cost_usd: 1 }), session('b', { actual_cost_usd: 2 })])
    setSidebarOrdering('cost')
    setSidebarOrdering('manual')

    expect($sidebarSessionRankIds.get()).toEqual([])
  })
})

describe('sortPinnedSessions', () => {
  const dots = {}

  it('leaves the hand-built pin order exactly as it is on manual', () => {
    const pinned = [session('second', { last_active: 1 }), session('first', { last_active: 9 })]

    expect(sortPinnedSessions(pinned, 'manual', dots).map(s => s.id)).toEqual(['second', 'first'])
  })

  it('ranks by recency on updated — the flat list gets that order from the backend, a pin list does not', () => {
    const pinned = [session('stale', { last_active: 1 }), session('live', { last_active: 9 })]

    expect(sortPinnedSessions(pinned, 'updated', dots).map(s => s.id)).toEqual(['live', 'stale'])
  })

  it('ranks by creation, newest first', () => {
    const pinned = [session('older', { started_at: 1 }), session('newer', { started_at: 9 })]

    expect(sortPinnedSessions(pinned, 'created', dots).map(s => s.id)).toEqual(['newer', 'older'])
  })

  it('ranks by total tokens, both halves counted', () => {
    const pinned = [session('small', { input_tokens: 10, output_tokens: 10 }), session('big', { input_tokens: 1, output_tokens: 500 })]

    expect(sortPinnedSessions(pinned, 'tokens', dots).map(s => s.id)).toEqual(['big', 'small'])
  })

  it('ranks by spend, priciest first', () => {
    const pinned = [session('cheap', { actual_cost_usd: 0.01 }), session('dear', { estimated_cost_usd: 2 })]

    expect(sortPinnedSessions(pinned, 'cost', dots).map(s => s.id)).toEqual(['dear', 'cheap'])
  })

  it('ranks by the displayed title on name — case and accents fold, numbers read naturally', () => {
    const pinned = [
      session('beta10', { title: 'beta 10' }),
      session('alpha-accent', { title: 'álpha' }),
      session('beta2', { title: 'beta 2' }),
      session('alpha', { title: 'Alpha' })
    ]

    // Folded compare puts Alpha before álpha (the tie-break), and natural
    // ordering puts 2 before 10 rather than after it.
    expect(sortPinnedSessions(pinned, 'name', dots).map(s => s.id)).toEqual([
      'alpha',
      'alpha-accent',
      'beta2',
      'beta10'
    ])
  })

  it('keeps the pin order for ties, so an equal-ranked pair does not shuffle on every render', () => {
    const pinned = [session('a', { last_active: 5 }), session('b', { last_active: 5 })]

    expect(sortPinnedSessions(pinned, 'updated', dots).map(s => s.id)).toEqual(['a', 'b'])
  })
})
