import { describe, expect, it, vi } from 'vitest'

import { mostRecentSessionWithMessages, recoverMostRecentProfileSession } from './profile-session-recovery'

const row = (id: string, overrides: Partial<Parameters<typeof mostRecentSessionWithMessages>[0][number]> = {}) => ({
  connection_id: undefined,
  id,
  last_active: 1,
  message_count: 1,
  profile: 'default',
  started_at: 1,
  ...overrides
})

describe('profile session recovery', () => {
  it('chooses the newest conversation with messages in the exact profile and connection', () => {
    const sessions = [
      row('wrong-profile-newer', { last_active: 30, profile: 'other', connection_id: 'vps' }),
      row('wrong-connection-newer', { last_active: 20, connection_id: 'other' }),
      row('target', { last_active: 10, connection_id: 'vps', profile: 'target' }),
      row('empty', { last_active: 40, connection_id: 'vps', message_count: 0, profile: 'target' })
    ]

    expect(mostRecentSessionWithMessages(sessions, { connectionId: 'vps', profile: 'target' })?.id).toBe('target')
  })

  it('does not infer ownership from a same-id row in another profile', () => {
    const sessions = [
      row('same-id', { profile: 'other', last_active: 50 }),
      row('target-id', { profile: 'target', last_active: 10 })
    ]

    expect(mostRecentSessionWithMessages(sessions, { profile: 'target' })?.id).toBe('target-id')
  })
  it('does not accept untagged aggregate rows for a registry connection', () => {
    expect(
      mostRecentSessionWithMessages([row('untagged', { last_active: 20 })], { connectionId: 'vps', profile: 'default' })
    ).toBeUndefined()
    expect(
      mostRecentSessionWithMessages([row('tagged', { connection_id: 'vps' })], {
        connectionId: null,
        profile: 'default'
      })
    ).toBeUndefined()
  })

  it('does not resume when the draft has an attachment even without text', () => {
    const onResume = vi.fn()

    expect(
      recoverMostRecentProfileSession({
        hasNonEmptyDraft: true,
        onResume,
        scope: { profile: 'default' },
        sessions: [row('history')]
      })
    ).toBe(false)
    expect(onResume).not.toHaveBeenCalled()
  })

  it.each([
    ['no history', [], false],
    ['nonempty draft', [row('history')], true]
  ] as const)('does not resume when %s', (_case, sessions, hasNonEmptyDraft) => {
    const onResume = vi.fn()

    expect(
      recoverMostRecentProfileSession({ hasNonEmptyDraft, onResume, scope: { profile: 'default' }, sessions })
    ).toBe(false)
    expect(onResume).not.toHaveBeenCalled()
  })

  it('resumes the most recent row and preserves its owner provenance', () => {
    const onResume = vi.fn()
    const target = row('target', { connection_id: 'vps', profile: 'worker', last_active: 20 })

    expect(
      recoverMostRecentProfileSession({
        hasNonEmptyDraft: false,
        onResume,
        scope: { connectionId: 'vps', profile: 'worker' },
        sessions: [target]
      })
    ).toBe(true)
    expect(onResume).toHaveBeenCalledWith(target)
  })
})
