import { beforeEach, describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import {
  $cronSessions,
  $messagingSessions,
  $selectedStoredSessionId,
  $sessions,
  $unreadFinishedSessionIds,
  setSelectedStoredSessionId,
  setSessions
} from './session'
import { clearAllSessionStates } from './session-states'
import { $sessionSeenCounts, $unreadFinishedMarkers, markSessionUnreadFinished } from './session-unread'

const session = (over: Partial<SessionInfo>): SessionInfo => ({
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
  ...over
})

function resetAll() {
  clearAllSessionStates()
  // Lists first: their listeners recompute the unread atom, so wiping the
  // atom afterwards leaves a genuinely clean slate.
  $sessions.set([])
  $cronSessions.set([])
  $messagingSessions.set([])
  $selectedStoredSessionId.set(null)
  $sessionSeenCounts.set({})
  $unreadFinishedMarkers.set([])
  $unreadFinishedSessionIds.set([])
}

describe('persisted unread (session-unread)', () => {
  beforeEach(resetAll)

  it('reconstructs the green dot from a seen watermark after a cold start', () => {
    // "Restart": persisted watermark says 3 messages were seen; the freshly
    // loaded list says the session now has 5 — it finished while we were away.
    $sessionSeenCounts.set({ s1: 3 })

    setSessions([session({ id: 's1', message_count: 5 })])

    expect($unreadFinishedSessionIds.get()).toEqual(['s1'])
  })

  it('seeds the watermark on first sight instead of painting every row green', () => {
    setSessions([session({ id: 's1', message_count: 5 })])

    expect($unreadFinishedSessionIds.get()).toEqual([])
    expect($sessionSeenCounts.get()).toEqual({ s1: 5 })
  })

  it('keeps a live-edge marker across a transient wipe (gateway/profile switch)', () => {
    setSessions([session({ id: 's1', message_count: 4 })])
    markSessionUnreadFinished('s1')
    expect($unreadFinishedSessionIds.get()).toEqual(['s1'])

    // Gateway switch: transient paint layer wiped, lists drained…
    setSessions([])
    $unreadFinishedSessionIds.set([])

    // …then the profile's lists load again: the persisted marker repaints.
    setSessions([session({ id: 's1', message_count: 4 })])

    expect($unreadFinishedSessionIds.get()).toEqual(['s1'])
  })

  it('acks watermark + marker when the user opens the session', () => {
    $sessionSeenCounts.set({ s1: 3 })
    setSessions([session({ id: 's1', message_count: 5 })])
    markSessionUnreadFinished('s1')

    setSelectedStoredSessionId('s1')

    expect($unreadFinishedSessionIds.get()).toEqual([])
    expect($unreadFinishedMarkers.get()).toEqual([])
    expect($sessionSeenCounts.get()).toEqual({ s1: 5 })

    // A later refresh with the same count stays read.
    setSessions([session({ id: 's1', message_count: 5 })])
    expect($unreadFinishedSessionIds.get()).toEqual([])
  })

  it('tracks the selected session’s count so on-screen activity never reads as unread', () => {
    $selectedStoredSessionId.set('s1')
    setSessions([session({ id: 's1', message_count: 2 })])

    // The turn completes while the user is looking at it.
    setSessions([session({ id: 's1', message_count: 7 })])

    expect($unreadFinishedSessionIds.get()).toEqual([])
    expect($sessionSeenCounts.get()).toEqual({ s1: 7 })

    // Navigating away must not retroactively flag it.
    setSelectedStoredSessionId(null)
    setSessions([session({ id: 's1', message_count: 7 })])
    expect($unreadFinishedSessionIds.get()).toEqual([])
  })

  it('follows auto-compression id rotation via the durable lineage id', () => {
    $sessionSeenCounts.set({ root: 3 })

    setSessions([session({ _lineage_root_id: 'root', id: 'tip-2', message_count: 6 })])

    expect($unreadFinishedSessionIds.get()).toEqual(['tip-2'])

    // Opening the rotated tip acks the lineage watermark.
    setSelectedStoredSessionId('tip-2')
    expect($sessionSeenCounts.get()).toEqual({ root: 6 })
  })

  it('reconstructs unread for cron rows too', () => {
    $sessionSeenCounts.set({ cron1: 1 })

    $cronSessions.set([session({ id: 'cron1', message_count: 3 })])

    expect($unreadFinishedSessionIds.get()).toEqual(['cron1'])
  })

  it('never paints messaging rows from count growth, but honors live-edge markers', () => {
    // First sight seeds…
    $messagingSessions.set([session({ id: 'tg1', message_count: 10 })])
    // …then inbound messages bump the count: NOT "finished — unread".
    $messagingSessions.set([session({ id: 'tg1', message_count: 14 })])

    expect($unreadFinishedSessionIds.get()).toEqual([])

    // A real background turn edge still marks it, durably.
    markSessionUnreadFinished('tg1')
    $unreadFinishedSessionIds.set([])
    $messagingSessions.set([session({ id: 'tg1', message_count: 14 })])

    expect($unreadFinishedSessionIds.get()).toEqual(['tg1'])
  })

  it('preserves a live-edge id whose row is not in any loaded list yet', () => {
    // A brand-new session's first turn isn't flushed to the sidebar list until
    // persisted — the flag must survive list refreshes in the meantime.
    markSessionUnreadFinished('fresh')

    setSessions([session({ id: 'other', message_count: 2 })])

    expect($unreadFinishedSessionIds.get()).toContain('fresh')
  })
})
