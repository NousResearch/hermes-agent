import { describe, expect, it } from 'vitest'
import type { SessionInfo } from '@/types/hermes'
import { computeSessionEligibility } from './session-eligibility'

function mockSession(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    id: '1',
    archived: false,
    cwd: null,
    ended_at: null,
    input_tokens: 0,
    is_active: false,
    last_active: Date.now() / 1000 - 600, // 10 min ago
    message_count: 2,
    model: null,
    output_tokens: 0,
    preview: null,
    profile: 'default',
    source: 'tui',
    started_at: Date.now() / 1000 - 1200,
    title: 'Test session',
    tool_call_count: 0,
    ...overrides
  }
}

describe('computeSessionEligibility', () => {
  it('returns empty summary for empty sessions list', () => {
    const result = computeSessionEligibility([], new Set())
    expect(result).toEqual({ total: 0, eligible: 0, protected: 0 })
  })

  it('marks all sessions as eligible when no preserves exist', () => {
    const sessions = [
      mockSession({ id: 'a' }),
      mockSession({ id: 'b' })
    ]
    const result = computeSessionEligibility(sessions, new Set())
    expect(result).toEqual({ total: 2, eligible: 2, protected: 0 })
  })

  it('marks preserved sessions as protected', () => {
    const sessions = [
      mockSession({ id: 'a' }),
      mockSession({ id: 'b' })
    ]
    const result = computeSessionEligibility(sessions, new Set(['a']))
    expect(result).toEqual({ total: 2, eligible: 1, protected: 1 })
  })

  it('marks recently active sessions as protected', () => {
    const now = Date.now() / 1000
    const sessions = [
      mockSession({ id: 'a', last_active: now - 60 }), // 1 min ago — active
      mockSession({ id: 'b', last_active: now - 600 }) // 10 min ago — stale
    ]
    const result = computeSessionEligibility(sessions, new Set())
    expect(result).toEqual({ total: 2, eligible: 1, protected: 1 })
  })

  it('does not protect ended sessions even if recently active', () => {
    const now = Date.now() / 1000
    const sessions = [
      mockSession({ id: 'a', last_active: now - 60, ended_at: now - 30 }),
      mockSession({ id: 'b', last_active: now - 600 })
    ]
    const result = computeSessionEligibility(sessions, new Set())
    expect(result).toEqual({ total: 2, eligible: 2, protected: 0 })
  })

  it('protects lineage root preserves entire lineage', () => {
    const sessions = [
      mockSession({ id: 'root', _lineage_root_id: 'root' }),
      mockSession({ id: 'continuation', _lineage_root_id: 'root' })
    ]
    const result = computeSessionEligibility(sessions, new Set(['root']))
    expect(result).toEqual({ total: 1, eligible: 0, protected: 1 })
  })

  it('deduplicates lineage roots', () => {
    const sessions = [
      mockSession({ id: 'root', _lineage_root_id: 'root' }),
      mockSession({ id: 'continuation', _lineage_root_id: 'root' })
    ]
    const result = computeSessionEligibility(sessions, new Set())
    // Both share the same target_id, so only one is counted
    expect(result).toEqual({ total: 1, eligible: 1, protected: 0 })
  })
})
