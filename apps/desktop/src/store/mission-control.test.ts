import { describe, expect, it } from 'vitest'

import type { WorkstreamFilterRuntime } from '@/store/workstream-filter'
import type { SessionInfo } from '@/types/hermes'

import { buildMissionControlBuckets, emptyMissionControlBuckets, missionControlBucketFor } from './mission-control'

const session = (id: string): SessionInfo => ({
  archived: false,
  cwd: null,
  ended_at: null,
  id,
  input_tokens: 0,
  is_active: false,
  last_active: 0,
  message_count: 1,
  model: null,
  output_tokens: 0,
  preview: null,
  source: null,
  started_at: 0,
  title: id,
  tool_call_count: 0
})

const runtime = (overrides: Partial<WorkstreamFilterRuntime> = {}): WorkstreamFilterRuntime => ({
  activeSessionId: null,
  attentionSessionIds: [],
  metadataBySession: {},
  selectedStoredSessionId: null,
  subagentsBySession: {},
  todosBySession: {},
  workingSessionIds: [],
  ...overrides
})

describe('emptyMissionControlBuckets', () => {
  it('creates every bucket empty', () => {
    expect(emptyMissionControlBuckets()).toEqual({
      active: [],
      blocked: [],
      closed: [],
      restart: [],
      review: [],
      safe_delete: []
    })
  })
})

describe('missionControlBucketFor', () => {
  it('keeps lifecycle buckets stronger than derived runtime state', () => {
    expect(missionControlBucketFor('safe_delete', 'work')).toBe('safe_delete')
    expect(missionControlBucketFor('closed', 'work')).toBe('closed')
    expect(missionControlBucketFor('restart_required', 'work')).toBe('restart')
  })

  it('groups active lifecycle states by runtime status', () => {
    expect(missionControlBucketFor('active', 'blocked')).toBe('blocked')
    expect(missionControlBucketFor('active', 'warn')).toBe('blocked')
    expect(missionControlBucketFor('active', 'verify')).toBe('review')
    expect(missionControlBucketFor('active', 'plan_review')).toBe('review')
    expect(missionControlBucketFor('active', 'work')).toBe('active')
  })
})

describe('buildMissionControlBuckets', () => {
  it('groups active, blocked, restart, closed, and safe-delete sessions separately', () => {
    const sessions = ['active', 'blocked', 'restart', 'closed', 'safe'].map(session)

    const buckets = buildMissionControlBuckets(
      sessions,
      runtime({
        attentionSessionIds: ['blocked'],
        metadataBySession: {
          closed: { lifecycle: 'closed', updatedAt: 1 },
          restart: { lifecycle: 'restart_required', updatedAt: 1 },
          safe: { lifecycle: 'safe_delete', updatedAt: 1 }
        },
        todosBySession: {
          active: [{ content: 'Do work', id: 'todo-1', status: 'pending' }]
        },
        workingSessionIds: []
      })
    )

    expect(buckets.active.map(entry => entry.session.id)).toEqual(['active'])
    expect(buckets.blocked.map(entry => entry.session.id)).toEqual(['blocked'])
    expect(buckets.review.map(entry => entry.session.id)).toEqual([])
    expect(buckets.restart.map(entry => entry.session.id)).toEqual(['restart'])
    expect(buckets.closed.map(entry => entry.session.id)).toEqual(['closed'])
    expect(buckets.safe_delete.map(entry => entry.session.id)).toEqual(['safe'])
  })

  it('uses live runtime session state for selected stored sessions', () => {
    const buckets = buildMissionControlBuckets(
      [session('stored')],
      runtime({
        activeSessionId: 'runtime',
        selectedStoredSessionId: 'stored',
        subagentsBySession: { runtime: [{ filesRead: [], filesWritten: [], goal: 'child', id: 'a1', parentId: null, startedAt: 0, status: 'failed', stream: [], taskCount: 1, taskIndex: 0, updatedAt: 0 }] }
      })
    )

    expect(buckets.blocked.map(entry => entry.session.id)).toEqual(['stored'])
  })
})
