import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { SessionInfo } from '@/types/hermes'

import { $queuedPromptsBySession } from './composer-queue'
import { requestGatewayForProfile } from './gateway'
import { watchOwnerQueues } from './owner-queue-reconcile'
import { reconcilePendingSubmissions, trackPendingSubmission } from './pending-submissions'
import { $connection, $gatewayState, $selectedStoredSessionId, $sessions, setSessions } from './session'
import { $sessionDotStateById } from './session-dot-state'
import { clearAllSessionStates, publishSessionState } from './session-states'

// Pass-through: records which profile's own backend a read is routed to.
vi.mock('@/store/gateway', async importOriginal => {
  const actual = await importOriginal<{ requestGatewayForProfile: (...args: unknown[]) => unknown }>()

  return { ...actual, requestGatewayForProfile: vi.fn(actual.requestGatewayForProfile) }
})

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

it('a chat whose queue is paused behind a turn lost in a restart needs the user, unless its turn runs again', () => {
  setSessions([row('beta', { unread: true }), row('alpha')])
  reconcilePendingSubmissions('beta', pausedSnapshot)

  expect($sessionDotStateById.get().beta).toBe('needs-input')
  expect($sessionDotStateById.get().alpha).toBeUndefined()

  // A turn running in the chat again reads working, not needs-input.
  publishSessionState('runtime-beta', { ...createClientSessionState('beta'), busy: true })
  expect($sessionDotStateById.get().beta).toBe('working')
  publishSessionState('runtime-beta', createClientSessionState('beta'))

  // Discarded: the owner lists only the send, now running. The chat stops asking for the user.
  reconcilePendingSubmissions('beta', [{ admission_id: 'queued-send', status: 'started', user: 'queued after hold' }])
  expect($sessionDotStateById.get().beta).toBe('unread')
})

it('each gateway open re-reads, once the list loads, the owner queue of every chat off screen, and never a legacy backend', async () => {
  $connection.set(connectionAt('ws://127.0.0.1:9/api/ws?native_dial=1'))
  trackPendingSubmission('beta', { id: 'queued-send', text: 'queued after hold', status: 'queued' })
  // The chat on screen, keyed by its lineage root while the selection holds the compressed tip:
  // its own route resume reads it.
  trackPendingSubmission('alpha', { id: 'on-screen', text: 'y', status: 'queued' })
  trackPendingSubmission('routed', { id: 'other', text: 'x', status: 'queued' })
  // A live snapshot from an earlier connection says nothing about what a restarted owner holds.
  publishSessionState('runtime-routed', createClientSessionState('routed'))
  $selectedStoredSessionId.set('alpha-tip')

  const request = vi.fn(async (_method: string, params?: Record<string, unknown>) => ({
    session_id: params?.session_id,
    info: { stored_session_id: params?.session_id, pending_submissions: pausedSnapshot }
  }))

  const reads = () =>
    request.mock.calls.map(([method, params]) => [method, params?.session_id, params?.defer_history]).sort()

  const settle = () => new Promise(resolve => setTimeout(resolve, 0))

  const offScreenReads = [
    ['session.resume', 'beta', true],
    ['session.resume', 'routed', true]
  ]

  const stop = watchOwnerQueues(request as unknown as Parameters<typeof watchOwnerQueues>[0])

  // The gateway opens before the session list can route a read: nothing goes out until it loads.
  $gatewayState.set('open')
  await settle()
  expect(request).not.toHaveBeenCalled()

  setSessions([row('beta'), row('alpha-tip', { _lineage_root_id: 'alpha' }), row('routed')])
  await vi.waitFor(() => expect(request).toHaveBeenCalledTimes(2))
  // A later list refresh on the same connection reads nothing more.
  setSessions([row('beta'), row('alpha-tip', { _lineage_root_id: 'alpha' }), row('routed'), row('gamma')])
  await settle()
  expect(reads()).toEqual(offScreenReads)
  expect($sessionDotStateById.get().beta).toBe('needs-input')

  // The owner restarts while the window stays up: the same chats are read again.
  request.mockClear()
  $gatewayState.set('closed')
  $gatewayState.set('open')
  await vi.waitFor(() => expect(request).toHaveBeenCalledTimes(2))
  await settle()
  expect(reads()).toEqual(offScreenReads)

  // A legacy (non-canonical) backend's cold resume can auto-continue a crashed turn: never read there.
  request.mockClear()
  $connection.set(connectionAt('wss://remote.example/api/ws?token=t'))
  $gatewayState.set('closed')
  $gatewayState.set('open')
  await settle()
  expect(request).not.toHaveBeenCalled()

  // Stopped: later opens read nothing.
  $connection.set(connectionAt('ws://127.0.0.1:9/api/ws?native_dial=1'))
  stop()
  $gatewayState.set('closed')
  $gatewayState.set('open')
  await settle()
  expect(request).not.toHaveBeenCalled()
})

it('a chat owned by a profile whose backend is not provably local is never read in the background', async () => {
  // The foreground is a local canonical gateway, but the chat's owner is a bare profile name: the
  // read would go to that profile's own backend (remote override, possibly legacy, unknown here).
  $connection.set(connectionAt('ws://127.0.0.1:9/api/ws?native_dial=1'))
  trackPendingSubmission('elsewhere', { id: 'queued-send', text: 'queued after hold', status: 'queued' })
  const request = vi.fn(async () => ({}))
  const stop = watchOwnerQueues(request as unknown as Parameters<typeof watchOwnerQueues>[0])

  $gatewayState.set('open')
  setSessions([row('elsewhere', { profile: 'work' })])
  await new Promise(resolve => setTimeout(resolve, 0))
  stop()

  expect(request).not.toHaveBeenCalled()
  expect(vi.mocked(requestGatewayForProfile)).not.toHaveBeenCalled()
})

it('a chat whose queue only holds a waiting send does not ask for the user', () => {
  setSessions([row('gamma')])
  reconcilePendingSubmissions('gamma', [{ admission_id: 'queued-send', status: 'queued', user: 'queued after hold' }])

  expect($sessionDotStateById.get().gamma).not.toBe('needs-input')
})
