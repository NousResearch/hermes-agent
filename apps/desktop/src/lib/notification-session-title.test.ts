import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { formatSessionNotificationTitle, notificationSessionTitle } from './notification-session-title'

const row = (id: string, extra: Partial<SessionInfo> = {}): SessionInfo =>
  ({ id, message_count: 1, source: 'desktop', started_at: 0, ...extra }) as SessionInfo

describe('notificationSessionTitle', () => {
  it('maps a runtime event through its stored session id', () => {
    expect(
      notificationSessionTitle([row('stored', { title: 'Fix authentication refresh' })], {
        runtimeSessionId: 'runtime',
        storedSessionId: 'stored'
      })
    ).toBe('Fix authentication refresh')
  })

  it('matches a compression lineage root', () => {
    expect(
      notificationSessionTitle([row('tip', { _lineage_root_id: 'root', title: 'Ship the release' })], {
        storedSessionId: 'root'
      })
    ).toBe('Ship the release')
  })

  it('uses profile and connection scope to disambiguate identical ids', () => {
    const sessions = [
      row('same', { connection_id: 'west', profile: 'default', title: 'West chat' }),
      row('same', { connection_id: 'east', profile: 'ops', title: 'East ops chat' })
    ]

    expect(
      notificationSessionTitle(sessions, {
        connectionId: 'east',
        profile: 'ops',
        runtimeSessionId: 'same'
      })
    ).toBe('East ops chat')
  })

  it('omits an ambiguous or mismatched label instead of naming the wrong chat', () => {
    const sessions = [row('same', { title: 'One' }), row('same', { title: 'Two' })]

    expect(notificationSessionTitle(sessions, { runtimeSessionId: 'same' })).toBe('')
    expect(
      notificationSessionTitle([row('same', { connection_id: 'remote', title: 'Remote' })], {
        runtimeSessionId: 'same'
      })
    ).toBe('')
  })

  it('falls back to a compact preview', () => {
    expect(
      notificationSessionTitle([row('stored', { preview: `  ${'long '.repeat(20)}  ` })], { storedSessionId: 'stored' })
    ).toMatch(/^long long .*…$/)
  })
})

describe('formatSessionNotificationTitle', () => {
  it('adds the chat label without changing the localized base title', () => {
    expect(formatSessionNotificationTitle('Approval needed', 'Fix authentication')).toBe(
      'Approval needed · Fix authentication'
    )
  })

  it('keeps the original title when no safe label is available', () => {
    expect(formatSessionNotificationTitle('Input needed', '')).toBe('Input needed')
  })
})
