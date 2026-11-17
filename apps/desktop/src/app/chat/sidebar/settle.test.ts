import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { settleAdvanceTarget } from './settle'

const row = (id: string, extra: Partial<SessionInfo> = {}) => ({ id, ...extra }) as SessionInfo

describe('settleAdvanceTarget', () => {
  it('moves to the next recent below the settled chat', () => {
    expect(settleAdvanceTarget([row('a'), row('b'), row('c')], 'b')?.id).toBe('c')
  })

  it('falls back to the one above at the bottom of the list', () => {
    expect(settleAdvanceTarget([row('a'), row('b'), row('c')], 'c')?.id).toBe('b')
  })

  it('skips archived rows and other ids of the same lineage', () => {
    const sessions = [row('a'), row('tip', { _lineage_root_id: 'root' }), row('x', { archived: true }), row('d')]

    expect(settleAdvanceTarget(sessions, 'root')?.id).toBe('d')
  })

  it('returns null when nothing else is left or the chat is not listed', () => {
    expect(settleAdvanceTarget([row('a')], 'a')).toBeNull()
    expect(settleAdvanceTarget([row('a'), row('b')], 'zzz')).toBeNull()
  })
})
