import { beforeEach, describe, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import type { SessionInfo } from '@/hermes'

import { resetSidebarView, setSidebarOrdering } from './layout'
import { $sessions } from './session'
import { $sidebarSessionRankIds } from './sidebar-sort'
import { clearAllSessionStates, publishSessionState } from './session-states'

const session = (id: string, fields: Partial<SessionInfo>) =>
  ({ id, input_tokens: 0, output_tokens: 0, started_at: 0, ...fields }) as SessionInfo

beforeEach(() => {
  resetSidebarView()
  clearAllSessionStates()
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

  it('leads the active ordering with sessions still doing work, newest first', () => {
    // A turn that ended waiting on the user is NOT running — it ranks with the
    // idle rows, however fresh it is, or the tier would swallow the one state
    // that needs the user.
    $sessions.set([
      session('idle-fresh', { last_active: 900 }),
      session('blocked', { last_active: 999 }),
      session('working-stale', { last_active: 40 }),
      session('working-fresh', { last_active: 800 }),
      session('idle-stale', { last_active: 50 })
    ])
    publishSessionState('r-1', { ...createClientSessionState('working-stale'), busy: true })
    publishSessionState('r-2', { ...createClientSessionState('working-fresh'), busy: true })
    publishSessionState('r-3', { ...createClientSessionState('blocked'), needsInput: true })
    setSidebarOrdering('active')

    expect($sidebarSessionRankIds.get()).toEqual([
      'working-fresh',
      'working-stale',
      'blocked',
      'idle-fresh',
      'idle-stale'
    ])
  })

  it('hands back the same array when a recompute changes nothing', () => {
    const rows = [session('dear', { actual_cost_usd: 2 }), session('cheap', { actual_cost_usd: 0.01 })]
    $sessions.set(rows)
    setSidebarOrdering('cost')

    const first = $sidebarSessionRankIds.get()

    // A page refresh rebuilds every row object; the order they produce is the
    // same one, so ranked surfaces must be handed the array they already hold.
    $sessions.set(rows.map(row => ({ ...row })))

    expect($sidebarSessionRankIds.get()).toBe(first)
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
