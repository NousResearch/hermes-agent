import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { SessionInfo } from '@/types/hermes'

import { $queuedPromptsBySession } from './composer-queue'
import { watchOwnerQueues } from './owner-queue-reconcile'
import { reconcilePendingSubmissions, trackPendingSubmission } from './pending-submissions'
import { $connection, $gatewayState, $selectedStoredSessionId, $sessions, setSessions } from './session'
import { $sessionDotStateById } from './session-dot-state'
import { clearAllSessionStates, publishSessionState } from './session-states'

const connectionAt = (wsUrl: string) => ({ mode: 'local', wsUrl }) as HermesConnection

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
  $connection.set(null)
  $gatewayState.set('idle')
  $selectedStoredSessionId.set(null)
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

it('a paused chat whose turn is running again reads working, not needs-input', () => {
  setSessions([row('beta')])
  reconcilePendingSubmissions('beta', pausedSnapshot)
  publishSessionState('runtime-beta', { ...createClientSessionState('beta'), busy: true })

  expect($sessionDotStateById.get().beta).toBe('working')
})

it('every gateway open re-reads the owner queue of each chat off screen with the open path params, and never a legacy backend', async () => {
  setSessions([row('beta'), row('alpha'), row('routed')])
  $connection.set(connectionAt('ws://127.0.0.1:9/api/ws?native_dial=1'))
  trackPendingSubmission('beta', { id: 'queued-send', text: 'queued after hold', status: 'queued' })
  trackPendingSubmission('alpha', { id: 'on-screen', text: 'y', status: 'queued' })
  trackPendingSubmission('routed', { id: 'other', text: 'x', status: 'queued' })
  // A live snapshot from an earlier connection says nothing about what a restarted owner holds.
  publishSessionState('runtime-routed', createClientSessionState('routed'))
  // The chat on screen is read by its own route resume.
  $selectedStoredSessionId.set('alpha')

  const request = vi.fn(async (_method: string, params?: Record<string, unknown>) => ({
    session_id: params?.session_id,
    info: { stored_session_id: params?.session_id, pending_submissions: pausedSnapshot }
  }))

  const reads = () => request.mock.calls.map(([method, params]) => [method, params?.session_id, params?.defer_history])
  const stop = watchOwnerQueues(request as unknown as Parameters<typeof watchOwnerQueues>[0])

  const openEdge = async () => {
    $gatewayState.set('closed')
    $gatewayState.set('open')
    await vi.waitFor(() => expect(request).toHaveBeenCalledTimes(2))
    await new Promise(resolve => setTimeout(resolve, 0))
  }

  await openEdge()
  expect(reads().sort()).toEqual([
    ['session.resume', 'beta', true],
    ['session.resume', 'routed', true]
  ])
  expect($sessionDotStateById.get().beta).toBe('needs-input')

  // The owner restarts while the window stays up: the same chats are read again.
  request.mockClear()
  await openEdge()
  expect(reads().sort()).toEqual([
    ['session.resume', 'beta', true],
    ['session.resume', 'routed', true]
  ])

  // A legacy (non-canonical) backend's cold resume can auto-continue a crashed turn: never read there.
  request.mockClear()
  $connection.set(connectionAt('wss://remote.example/api/ws?token=t'))
  $gatewayState.set('closed')
  $gatewayState.set('open')
  await new Promise(resolve => setTimeout(resolve, 0))
  expect(request).not.toHaveBeenCalled()
  stop()
})
