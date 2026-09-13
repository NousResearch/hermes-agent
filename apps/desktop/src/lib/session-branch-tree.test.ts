import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { makeSessionInfo } from '../test/session-info'

import { flattenSessionsWithBranches, sessionTreeNodeId } from './session-branch-tree'

const session = (id: string, overrides: Partial<SessionInfo> = {}): SessionInfo =>
  makeSessionInfo({ id, message_count: 1, source: 'cli', title: id, ...overrides })

describe('flattenSessionsWithBranches', () => {
  it('nests branch rows under their parent with tree stems', () => {
    const parent = session('parent', { last_active: 20 })
    const branchA = session('branch-a', { last_active: 15, parent_session_id: 'parent' })
    const branchB = session('branch-b', { last_active: 10, parent_session_id: 'parent' })

    expect(flattenSessionsWithBranches([parent, branchA, branchB])).toEqual([
      { hasChildren: true, session: parent },
      { branchDepth: 1, branchStem: '├─ ', session: branchA },
      { branchDepth: 1, branchStem: '└─ ', session: branchB }
    ])
  })

  it('follows a compressed parent via lineage root id', () => {
    const tip = session('tip', { _lineage_root_id: 'root', last_active: 30 })
    const branch = session('branch', { parent_session_id: 'root', last_active: 10 })

    expect(flattenSessionsWithBranches([tip, branch])).toEqual([
      { hasChildren: true, session: tip },
      { branchDepth: 1, branchStem: '└─ ', session: branch }
    ])
  })

  it('nests provider-neutral spawned sessions to arbitrary depth within their profile', () => {
    const parent = session('parent', { profile: 'work' })

    const child = session('child-tip', {
      _lineage_ids: ['child-root', 'child-tip'],
      profile: 'work',
      spawned_by_session_id: 'parent'
    })

    const grandchild = session('grandchild', { profile: 'work', spawned_by_session_id: 'child-root' })

    const sameIdOtherProfile = session('child-tip', {
      profile: 'personal',
      spawned_by_session_id: 'parent'
    })

    const otherConnection = session('remote-child', {
      connection_id: 'remote',
      profile: 'work',
      spawned_by_session_id: 'parent'
    })

    expect(flattenSessionsWithBranches([parent, child, grandchild, sameIdOtherProfile, otherConnection])).toEqual([
      { hasChildren: true, session: parent },
      { branchDepth: 1, branchStem: '└─ ', hasChildren: true, session: child },
      { branchDepth: 2, branchStem: '└─ ', session: grandchild },
      { session: sameIdOtherProfile },
      { session: otherConnection }
    ])
  })

  it('keeps orphan branches at the top level when the parent is missing', () => {
    const branch = session('branch', { parent_session_id: 'missing' })

    expect(flattenSessionsWithBranches([branch])).toEqual([{ session: branch }])
  })

  it('keeps a collapsed parent visible while hiding its complete descendant subtree', () => {
    const parent = session('parent')
    const child = session('child', { spawned_by_session_id: 'parent' })
    const grandchild = session('grandchild', { spawned_by_session_id: 'child' })

    expect(
      flattenSessionsWithBranches([parent, child, grandchild], {
        isOpen: candidate => candidate.id !== 'parent'
      }).map(item => item.session.id)
    ).toEqual(['parent'])
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

describe('sessionTreeNodeId', () => {
  it('keeps collapse state keyed to the durable lineage root across compressed tips', () => {
    const firstTip = session('tip-1', { _lineage_root_id: 'root' })
    const nextTip = session('tip-2', { _lineage_root_id: 'root' })

    expect(sessionTreeNodeId(nextTip)).toBe(sessionTreeNodeId(firstTip))
  })
})
