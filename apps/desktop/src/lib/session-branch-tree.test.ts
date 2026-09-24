import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { makeSessionInfo } from '../test/session-info'

import { flattenSessionsWithBranches } from './session-branch-tree'

const session = (id: string, overrides: Partial<SessionInfo> = {}): SessionInfo =>
  makeSessionInfo({ id, message_count: 1, source: 'cli', title: id, ...overrides })

describe('flattenSessionsWithBranches', () => {
  it('nests branch rows under their parent with tree stems', () => {
    const parent = session('parent', { last_active: 20 })
    const branchA = session('branch-a', { last_active: 15, parent_session_id: 'parent' })
    const branchB = session('branch-b', { last_active: 10, parent_session_id: 'parent' })

    expect(flattenSessionsWithBranches([parent, branchA, branchB])).toEqual([
      { session: parent },
      { branchStem: '├─ ', session: branchA },
      { branchStem: '└─ ', session: branchB }
    ])
  })

  it('follows a compressed parent via lineage root id', () => {
    const tip = session('tip', { _lineage_root_id: 'root', last_active: 30 })
    const branch = session('branch', { parent_session_id: 'root', last_active: 10 })

    expect(flattenSessionsWithBranches([tip, branch])).toEqual([
      { session: tip },
      { branchStem: '└─ ', session: branch }
    ])
  })

  it('collapses a stale compression tip into its continuation instead of nesting it (#82290)', () => {
    // Old tip (#3) and its continuation (#4) both survived in the store. They
    // share one lineage root, so they are one conversation: a single row for
    // the live tip, no └─ stem.
    const oldTip = session('old-tip', { _lineage_root_id: 'root', last_active: 100, started_at: 50 })

    const continuation = session('continuation', {
      _lineage_root_id: 'root',
      last_active: 100,
      parent_session_id: 'old-tip',
      started_at: 100
    })

    expect(flattenSessionsWithBranches([oldTip, continuation])).toEqual([{ session: continuation }])
    expect(flattenSessionsWithBranches([continuation, oldTip], { preserveOrder: true })).toEqual([
      { session: continuation }
    ])
  })

  it('collapses the lineage root row once its continuation carries the root id', () => {
    const root = session('root', { last_active: 40 })

    const continuation = session('continuation', {
      _lineage_root_id: 'root',
      last_active: 60,
      parent_session_id: 'root'
    })

    const branch = session('branch', { last_active: 50, parent_session_id: 'root' })

    expect(flattenSessionsWithBranches([root, continuation, branch])).toEqual([
      { session: continuation },
      { branchStem: '└─ ', session: branch }
    ])
  })

  it('keeps same-lineage ids from different profiles apart', () => {
    const work = session('tip', { _lineage_root_id: 'root', last_active: 20, profile: 'work' })
    const home = session('tip-home', { _lineage_root_id: 'root', last_active: 10, profile: 'home' })

    expect(flattenSessionsWithBranches([work, home]).map(e => e.session.id)).toEqual(['tip', 'tip-home'])
  })

  it('still nests a real branch of a compressed conversation under the live tip', () => {
    const oldTip = session('old-tip', { _lineage_root_id: 'root', last_active: 30 })
    const tip = session('tip', { _lineage_root_id: 'root', last_active: 90, parent_session_id: 'old-tip' })
    const branch = session('branch', { last_active: 70, parent_session_id: 'tip' })

    expect(flattenSessionsWithBranches([oldTip, tip, branch])).toEqual([
      { session: tip },
      { branchStem: '└─ ', session: branch }
    ])
  })

  it('keeps orphan branches at the top level when the parent is missing', () => {
    const branch = session('branch', { parent_session_id: 'missing' })

    expect(flattenSessionsWithBranches([branch])).toEqual([{ session: branch }])
  })

  it('re-sorts roots by group recency by default (pinned-style jumps without preserveOrder)', () => {
    // Stale important chat first in the caller's array; a recently-active
    // background task second. Default path must lift the fresher root — that
    // is what was scrambling the Pinned section before preserveOrder.
    const important = session('important', { last_active: 10 })
    const background = session('background', { last_active: 99 })

    expect(flattenSessionsWithBranches([important, background]).map(e => e.session.id)).toEqual([
      'background',
      'important'
    ])
  })

  it("preserveOrder keeps the caller's root order even when activity is newer lower down", () => {
    const important = session('important', { last_active: 10 })
    const background = session('background', { last_active: 99 })
    const branch = session('branch', { last_active: 50, parent_session_id: 'important' })

    expect(
      flattenSessionsWithBranches([important, background, branch], { preserveOrder: true }).map(e => ({
        id: e.session.id,
        stem: e.branchStem
      }))
    ).toEqual([
      { id: 'important', stem: undefined },
      { id: 'branch', stem: '└─ ' },
      { id: 'background', stem: undefined }
    ])
  })
})
