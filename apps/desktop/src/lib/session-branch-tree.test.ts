import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { flattenSessionsWithBranches } from './session-branch-tree'

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
})
