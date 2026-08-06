import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { clusterEntriesByTopic, flattenSessionsWithBranches } from './session-branch-tree'

const session = (id: string, overrides: Partial<SessionInfo> = {}): SessionInfo =>
  ({
    ended_at: null,
    id,
    input_tokens: 0,
    is_active: false,
    last_active: 0,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'cli',
    started_at: 0,
    title: id,
    tool_call_count: 0,
    ...overrides
  }) as SessionInfo

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

describe('clusterEntriesByTopic', () => {
  const entry = (id: string, title: string, overrides: Partial<SessionInfo> = {}): { session: SessionInfo } => ({
    session: session(id, { title, ...overrides })
  })

  it('keeps unprefixed entries in their original order', () => {
    const entries = [entry('a', '普通会话'), entry('b', '另一会话')]

    expect(clusterEntriesByTopic(entries)).toEqual(entries)
  })

  it('pulls same-topic entries together after a recency re-sort scattered them', () => {
    const entries = [
      entry('v1', '[凭证]打印', { last_active: 30 }),
      entry('x', '普通会话', { last_active: 25 }),
      entry('v2', '[凭证]录入', { last_active: 20 }),
      entry('d1', '[部署]新服务器', { last_active: 15 }),
      entry('v3', '[凭证]导入', { last_active: 10 })
    ]

    expect(clusterEntriesByTopic(entries).map(e => e.session.id)).toEqual([
      'v1',
      'v2',
      'v3',
      'x',
      'd1'
    ])
  })

  it('moves a parent together with its branch children', () => {
    const entries = [
      { session: session('p1', { title: '[业务]流水', last_active: 30 }) },
      { branchStem: '└─ ', session: session('b1', { title: '[业务]流水分支', last_active: 28, parent_session_id: 'p1' }) },
      { session: session('mid', { title: '中间会话', last_active: 20 }) },
      { session: session('p2', { title: '[业务]代账', last_active: 10 }) }
    ]

    expect(clusterEntriesByTopic(entries).map(e => e.session.id)).toEqual([
      'p1',
      'b1',
      'p2',
      'mid'
    ])
  })

  it('returns a single-entry list untouched', () => {
    const entries = [entry('only', '[业务]唯一')]

    expect(clusterEntriesByTopic(entries)).toEqual(entries)
  })
})
