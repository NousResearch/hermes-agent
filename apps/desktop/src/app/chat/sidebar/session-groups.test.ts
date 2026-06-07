import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { workspaceGroupsFor } from './session-groups'

function session(overrides: Partial<SessionInfo>): SessionInfo {
  return {
    archived: false,
    cwd: null,
    ended_at: null,
    id: 's',
    input_tokens: 0,
    is_active: false,
    last_active: 0,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'cli',
    started_at: 0,
    title: null,
    tool_call_count: 0,
    ...overrides
  }
}

describe('workspaceGroupsFor', () => {
  it('promotes WebUI sessions into their own source group', () => {
    const groups = workspaceGroupsFor([
      session({ cwd: '/workspace/alice', id: 'cli', source: 'cli', started_at: 3 }),
      session({ cwd: '/workspace/alice/hermes', id: 'webui-new', source: 'webui', started_at: 2 }),
      session({ cwd: null, id: 'webui-old', source: 'WebUI', started_at: 1 })
    ])

    expect(groups.map(group => [group.id, group.label, group.sessions.map(s => s.id)])).toEqual([
      ['source:webui', 'WebUI', ['webui-new', 'webui-old']],
      ['/workspace/alice', 'alice', ['cli']]
    ])
    expect(groups[0].allowNewSession).toBe(false)
  })

  it('keeps non-WebUI sessions grouped by workspace', () => {
    const groups = workspaceGroupsFor([
      session({ cwd: '/workspace/project', id: 'older', started_at: 1 }),
      session({ cwd: null, id: 'none', started_at: 3 }),
      session({ cwd: '/workspace/project', id: 'newer', started_at: 4 })
    ])

    expect(groups.map(group => [group.id, group.label, group.sessions.map(s => s.id)])).toEqual([
      ['/workspace/project', 'project', ['newer', 'older']],
      ['__no_workspace__', 'No workspace', ['none']]
    ])
  })
})
