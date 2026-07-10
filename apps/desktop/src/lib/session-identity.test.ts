import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { sessionIdentityIds, sessionMatchesAnyId, sessionMatchesStoredId } from './session-identity'

const session = (over: Partial<SessionInfo>): SessionInfo => over as SessionInfo

describe('session identity across compression', () => {
  const projected = session({
    id: 'tip-3',
    _lineage_root_id: 'root',
    _lineage_ids: ['root', 'tip-1', 'tip-2', 'tip-3']
  })

  it('matches the live tip, root, and intermediate runtime ids', () => {
    expect(sessionMatchesStoredId(projected, 'tip-3')).toBe(true)
    expect(sessionMatchesStoredId(projected, 'root')).toBe(true)
    expect(sessionMatchesStoredId(projected, 'tip-2')).toBe(true)
    expect(sessionMatchesStoredId(projected, 'other')).toBe(false)
  })

  it('keeps backward-compatible tip/root matching without lineage aliases', () => {
    const legacy = session({ id: 'tip', _lineage_root_id: 'root' })

    expect(sessionMatchesStoredId(legacy, 'tip')).toBe(true)
    expect(sessionMatchesStoredId(legacy, 'root')).toBe(true)
    expect(sessionMatchesStoredId(legacy, 'mid')).toBe(false)
  })

  it('matches a working-id set through an intermediate segment', () => {
    expect(sessionMatchesAnyId(projected, new Set(['tip-1']))).toBe(true)
    expect(sessionMatchesAnyId(projected, new Set(['other']))).toBe(false)
  })

  it('returns unique aliases for lookup maps', () => {
    expect(sessionIdentityIds(projected)).toEqual(['tip-3', 'root', 'tip-1', 'tip-2'])
  })
})
