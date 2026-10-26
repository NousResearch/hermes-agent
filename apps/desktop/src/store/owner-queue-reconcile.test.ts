import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import type { SessionInfo } from '@/types/hermes'

import { $queuedPromptsBySession, getQueuedPrompts } from './composer-queue'
import { reconcileOwnerQueues } from './owner-queue-reconcile'
import { reconcilePendingSubmissions, trackPendingSubmission } from './pending-submissions'
import { $sessions, setSessions } from './session'
import { $sessionDotStateById } from './session-dot-state'
import { clearAllSessionStates, publishSessionState } from './session-states'

const row = (id: string, extra: Partial<SessionInfo> = {}): SessionInfo =>
  ({ id, message_count: 4, source: 'desktop', started_at: 0, title: id, ...extra }) as SessionInfo

// The owner's list after a restart cut the running turn: the turn is `unknown` (only Discard
// resolves it) and the send queued behind it waits.
const pausedSnapshot = [
  { admission_id: 'lost-turn', status: 'unknown', user: 'HOLD running turn', execution_generation: 1 },
  { admission_id: 'queued-send', status: 'queued', user: 'queued after hold' }
]

beforeEach(() => {
  window.localStorage.clear()
  $queuedPromptsBySession.set({})
  clearAllSessionStates()
  $sessions.set([])
})

afterEach(() => {
  $sessions.set([])
})

it('a chat whose queue is paused behind a turn lost in a restart needs the user, it is not finished-unread', () => {
  setSessions([row('beta', { unread: true }), row('alpha')])
  reconcilePendingSubmissions('beta', pausedSnapshot)

  expect($sessionDotStateById.get().beta).toBe('needs-input')
  expect($sessionDotStateById.get().alpha).toBeUndefined()

  // Discarded: the owner lists only the send, now running. The chat stops asking for the user.
  reconcilePendingSubmissions('beta', [{ admission_id: 'queued-send', status: 'started', user: 'queued after hold' }])
  expect($sessionDotStateById.get().beta).toBe('unread')
})

it('after a relaunch the owner queue of a chat that is not on screen is read again, and only of chats holding owner work', async () => {
  setSessions([row('beta'), row('alpha'), row('routed')])
  // Before the quit this window saw the send queued behind a running turn.
  trackPendingSubmission('beta', { id: 'queued-send', text: 'queued after hold', status: 'queued' })
  trackPendingSubmission('routed', { id: 'other', text: 'x', status: 'queued' })
  // The routed chat was already resumed by this window: it holds its own live snapshot.
  publishSessionState('runtime-routed', createClientSessionState('routed'))

  const request = vi.fn(async (method: string, params?: Record<string, unknown>) => ({
    session_id: params?.session_id,
    info: { stored_session_id: params?.session_id, pending_submissions: pausedSnapshot }
  })) as unknown as Parameters<typeof reconcileOwnerQueues>[0]

  await reconcileOwnerQueues(request)

  expect(
    (request as unknown as ReturnType<typeof vi.fn>).mock.calls.map(([method, params]) => [method, params?.session_id])
  ).toEqual([['session.resume', 'beta']])
  expect(getQueuedPrompts('beta').map(({ id, serverStatus }) => ({ id, serverStatus }))).toEqual([
    { id: 'lost-turn', serverStatus: 'unknown' },
    { id: 'queued-send', serverStatus: 'queued' }
  ])
  expect($sessionDotStateById.get().beta).toBe('needs-input')
})
