import { describe, expect, it } from 'vitest'

import type { SessionDotState } from '@/store/session-dot-state'
import type { SessionInfo } from '@/types/hermes'

import { buildAttentionItems } from './attention'

type TestSession = Pick<
  SessionInfo,
  | '_lineage_root_id'
  | 'archived'
  | 'connection_id'
  | 'id'
  | 'last_active'
  | 'preview'
  | 'profile'
  | 'source'
  | 'started_at'
  | 'title'
  | 'unread'
>

const session = (id: string, extra: Partial<TestSession> = {}): TestSession => ({
  _lineage_root_id: null,
  archived: false,
  connection_id: undefined,
  id,
  last_active: 100,
  preview: null,
  profile: 'default',
  source: 'desktop',
  started_at: 90,
  title: id,
  ...extra
})

const state = (id: string, value: SessionDotState): Readonly<Record<string, SessionDotState>> => ({ [id]: value })

describe('buildAttentionItems', () => {
  it('builds actionable and unread items from the existing dot state', () => {
    const result = buildAttentionItems({
      attentionSessionIds: ['approval'],
      dotStates: { ...state('approval', 'needs-input'), ...state('done', 'unread') },
      messagingSessions: [],
      sessions: [session('approval'), session('done', { last_active: 200 })]
    })

    expect(result.map(item => [item.kind, item.sessionId])).toEqual([
      ['needs-input', 'approval'],
      ['unread', 'done']
    ])
  })

  it('deduplicates a session across lists and compression aliases', () => {
    const result = buildAttentionItems({
      attentionSessionIds: ['root'],
      dotStates: { tip: 'needs-input' },
      messagingSessions: [session('root', { source: 'telegram' })],
      sessions: [session('tip', { _lineage_root_id: 'root' })]
    })

    expect(result).toHaveLength(1)
    expect(result[0]?.sessionId).toBe('tip')
  })

  it('keeps identical ids separate when they belong to different profiles', () => {
    const result = buildAttentionItems({
      attentionSessionIds: [],
      dotStates: { shared: 'unread' },
      messagingSessions: [session('shared', { profile: 'ops', last_active: 300 })],
      sessions: [session('shared')]
    })

    expect(result).toHaveLength(2)
    expect(result.map(item => item.profile).sort()).toEqual(['default', 'ops'])
    expect(result.map(item => item.owner?.profile).sort()).toEqual(['default', 'ops'])
  })

  it('scopes waiting state to the matching owner when ids collide', () => {
    const result = buildAttentionItems({
      attentionOwners: {
        shared: [{ connectionId: 'remote-1', profile: 'ops' }]
      },
      attentionSessionIds: ['shared'],
      dotStates: {},
      messagingSessions: [session('shared', { connection_id: 'remote-1', profile: 'ops' })],
      sessions: [session('shared')]
    })

    expect(result).toHaveLength(1)
    expect(result[0]).toMatchObject({ kind: 'needs-input', profile: 'ops' })
  })

  it('does not classify ambiguous duplicate ids as waiting without an owner', () => {
    const result = buildAttentionItems({
      attentionSessionIds: ['shared'],
      dotStates: {},
      messagingSessions: [session('shared', { profile: 'ops' })],
      sessions: [session('shared')]
    })

    expect(result).toEqual([])
  })

  it('includes persisted unread state carried by messaging rows', () => {
    const result = buildAttentionItems({
      attentionSessionIds: [],
      dotStates: {},
      messagingSessions: [session('message', { source: 'telegram', unread: true })],
      sessions: []
    })

    expect(result).toMatchObject([{ kind: 'unread', sessionId: 'message' }])
  })

  it('ignores archived and non-actionable rows', () => {
    const result = buildAttentionItems({
      attentionSessionIds: [],
      dotStates: {
        archived: 'unread',
        running: 'working',
        quiet: 'idle'
      },
      messagingSessions: [],
      sessions: [session('archived', { archived: true }), session('running'), session('quiet')]
    })

    expect(result).toEqual([])
  })

  it('does not recreate an archived waiting session as a synthetic item', () => {
    const result = buildAttentionItems({
      attentionSessionIds: ['archived'],
      dotStates: {},
      messagingSessions: [],
      sessions: [session('archived', { archived: true })]
    })

    expect(result).toEqual([])
  })

  it('keeps a waiting session visible when its row has not loaded yet', () => {
    const result = buildAttentionItems({
      attentionSessionIds: ['unloaded-session'],
      dotStates: {},
      messagingSessions: [],
      sessions: []
    })

    expect(result).toMatchObject([
      {
        kind: 'needs-input',
        owner: undefined,
        sessionId: 'unloaded-session',
        title: 'Session unloaded-session'
      }
    ])
  })
})
