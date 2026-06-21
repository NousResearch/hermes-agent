import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { buildSessionByAnyId } from './session-index'

const session = (overrides: Partial<SessionInfo>): SessionInfo => ({
  archived: false,
  cwd: null,
  ended_at: null,
  id: 'live',
  input_tokens: 0,
  is_active: false,
  last_active: 0,
  message_count: 0,
  model: null,
  output_tokens: 0,
  preview: null,
  source: null,
  started_at: 0,
  title: null,
  tool_call_count: 0,
  ...overrides
})

describe('buildSessionByAnyId', () => {
  it('indexes messaging sessions so pinned non-local sessions still resolve', () => {
    const messaging = session({
      id: 'telegram-tip',
      _lineage_root_id: 'telegram-root',
      source: 'telegram'
    })

    const byId = buildSessionByAnyId([], [], [messaging])

    expect(byId.get('telegram-tip')).toBe(messaging)
    expect(byId.get('telegram-root')).toBe(messaging)
  })

  it('lets recents win direct id collisions with separately rendered groups', () => {
    const cron = session({ id: 'shared', title: 'cron copy', source: 'cron' })
    const recent = session({ id: 'shared', title: 'recent copy', source: 'cli' })

    const byId = buildSessionByAnyId([recent], [cron], [])

    expect(byId.get('shared')).toBe(recent)
  })
})
