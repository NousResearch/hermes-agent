import { beforeEach, expect, it, vi } from 'vitest'

import { CanonicalDesktopProtocol } from '@/api/canonical-protocol'
import { handleSessionInfoEvent } from '@/app/session/hooks/use-message-stream/gateway-event/session-info'
import type { GatewayEventContext } from '@/app/session/hooks/use-message-stream/gateway-event/types'
import { applyRuntimeInfo } from '@/app/session/hooks/use-session-actions/utils'
import { isSteerableEntry } from '@/store/composer-queue'

import { $queuedPromptsBySession, enqueueQueuedPrompt, getQueuedPrompts } from './composer-queue'
import { readPendingSubmissions, reconcilePendingSubmissions, trackPendingSubmission } from './pending-submissions'

beforeEach(() => {
  window.localStorage.clear()
  $queuedPromptsBySession.set({})
})

it('projects resumed runtime pending receipts under durable identity without making them sendable', () => {
  const info = {
    stored_session_id: 'durable',
    pending_submissions: [
      { admission_id: 'remote', status: 'queued', user: 'queued exact Ω' },
      { admission_id: 'uncertain', status: 'unknown', user: 'unknown exact Ω' }
    ]
  }

  applyRuntimeInfo(info)
  applyRuntimeInfo(info)
  expect(getQueuedPrompts('durable').map(({ id, text, serverStatus }) => ({ id, text, serverStatus })))
    .toEqual(info.pending_submissions.map(({ admission_id, user, status }) => ({ id: admission_id, text: user, serverStatus: status })))
  expect(getQueuedPrompts('durable').every(entry => !isSteerableEntry(entry))).toBe(true)
  expect(getQueuedPrompts('runtime')).toEqual([])
})

it('reconciles server queue by identity without replaying or duplicating local entries', () => {
  enqueueQueuedPrompt('chat', { id: 'one', text: 'same', attachments: [] })
  enqueueQueuedPrompt('chat', { id: 'local', text: 'same', attachments: [] })
  const snapshot = [{ admission_id: 'one', status: 'queued', user: 'same' }]
  reconcilePendingSubmissions('chat', snapshot)
  reconcilePendingSubmissions('chat', snapshot)
  expect(getQueuedPrompts('chat').map(entry => entry.id)).toEqual(['one', 'local'])
  expect(getQueuedPrompts('chat')[0]?.serverStatus).toBe('queued')
  reconcilePendingSubmissions('chat', [{ ...snapshot[0], status: 'started' }])
  expect(getQueuedPrompts('chat').map(entry => entry.id)).toEqual(['local'])
  reconcilePendingSubmissions('chat', [{ admission_id: 'unknown', status: 'unknown', user: 'interrupted' }])
  expect(getQueuedPrompts('chat').find(entry => entry.id === 'unknown')?.serverStatus).toBe('unknown')
  reconcilePendingSubmissions('chat', [])
  expect(getQueuedPrompts('chat').map(entry => entry.id)).toEqual(['local'])
})

it('recovers remote pending text into the queue and journal without a local submission', () => {
  const snapshot = [
    { admission_id: 'remote-queued', status: 'queued', user: 'queued elsewhere' },
    { admission_id: 'remote-unknown', status: 'unknown', user: 'interrupted elsewhere' }
  ]

  reconcilePendingSubmissions('chat', snapshot)
  reconcilePendingSubmissions('chat', snapshot)
  expect(getQueuedPrompts('chat').map(({ id, text, serverStatus }) => ({ id, text, serverStatus })))
    .toEqual(snapshot.map(({ admission_id, user, status }) => ({ id: admission_id, text: user, serverStatus: status })))
  const journal = readPendingSubmissions('chat')

  for (const receipt of snapshot) {
    expect(journal[receipt.admission_id].text).toBe(receipt.user)
  }
})

it('maps optimistic input identity to the admission identity before local drain can replay it', () => {
  enqueueQueuedPrompt('mapped', { id: 'input-id', text: 'same', attachments: [] })
  const receipt = { admission_id: 'admission-id', input_id: 'input-id', status: 'queued', user: 'same' }
  reconcilePendingSubmissions('mapped', [receipt])
  expect(getQueuedPrompts('mapped')).toMatchObject([{ id: 'admission-id', serverStatus: 'queued', text: 'same' }])
  expect(getQueuedPrompts('mapped')).toHaveLength(1)
  reconcilePendingSubmissions('mapped', [{ ...receipt, status: 'started' }])
  expect(getQueuedPrompts('mapped')).toEqual([])
})

it('persists identified direct submissions independently of the automatic local queue', () => {
  trackPendingSubmission('chat', { id: 'direct', text: 'hello' })
  expect(readPendingSubmissions('chat').direct).toMatchObject({ id: 'direct', text: 'hello' })
  expect(getQueuedPrompts('chat')).toEqual([])
})

it('never lets a stale queue snapshot repaint an admission already seen started or retired', () => {
  const queued = [{ admission_id: 'a', status: 'queued', user: 'later' }]
  reconcilePendingSubmissions('mono', queued)
  reconcilePendingSubmissions('mono', [{ ...queued[0], status: 'started' }])
  reconcilePendingSubmissions('mono', queued)
  expect(getQueuedPrompts('mono')).toEqual([])

  reconcilePendingSubmissions('mono', [])
  reconcilePendingSubmissions('mono', queued)
  expect(getQueuedPrompts('mono')).toEqual([])

  // A started turn the owner lost across a restart legitimately moves on to `unknown`.
  reconcilePendingSubmissions('mono', [{ admission_id: 'b', status: 'started', user: 'lost' }])
  reconcilePendingSubmissions('mono', [{ admission_id: 'b', status: 'unknown', user: 'lost' }])
  expect(getQueuedPrompts('mono').map(entry => [entry.id, entry.serverStatus])).toEqual([['b', 'unknown']])
})

it('a delayed older empty snapshot never retires a receipt a newer frame still lists; a current one does', () => {
  const protocol = new CanonicalDesktopProtocol()
  const queued = { admission_id: 'q', input_id: 'in-q', status: 'queued', text: 'still waiting', sequence: 1 }

  // The owner's resume answer, through the production snapshot projection.
  const snapshot = (lastSequence: number, pending: unknown[]) => applyRuntimeInfo(protocol.result('session.resume', { session_id: 'fenced' },
    { session_id: 'fenced', stored_session_id: 'fenced', revision: 1, execution_generation: 1, replay_epoch: 'e1', last_sequence: lastSequence, pending, info: {} }).info,
  { foreground: false })

  // The live pending fanout at seq 10, through the canonical normalizer and the session.info handler.
  const live = { type: 'session.info', session_id: 'fenced', seq: 10, replay_epoch: 'e1', profile: 'default',
    payload: { stored_session_id: 'fenced', pending: [queued], revision: 1 } as Record<string, unknown> }

  protocol.event(live)
  handleSessionInfoEvent({
    deps: { activeGatewayProfile: 'default', activeSessionIdRef: { current: null }, hydrateFromStoredSession: vi.fn(),
      lastCwdInfoSessionRef: { current: null }, queryClient: { invalidateQueries: vi.fn() }, refreshHermesConfig: vi.fn(),
      scheduleSessionsRefresh: vi.fn(), sessionInterrupted: () => false, sessionStateByRuntimeIdRef: { current: new Map() },
      updateSessionState: vi.fn(state => state), upsertToolCall: vi.fn() },
    event: live, explicitSid: 'fenced', fromActiveSource: () => true, isActiveEvent: false, occurredAt: Date.now() / 1000,
    payload: live.payload, scheduleConfigRefresh: vi.fn(), sessionId: 'fenced'
  } as unknown as GatewayEventContext)
  expect(getQueuedPrompts('fenced').map(entry => [entry.id, entry.serverStatus])).toEqual([['q', 'queued']])

  // A resume answer read at seq 7 arrives late and lists nothing: it predates the queued frame.
  snapshot(7, [])
  expect(readPendingSubmissions('fenced').q?.status).toBe('queued')
  expect(getQueuedPrompts('fenced').map(entry => [entry.id, entry.serverStatus])).toEqual([['q', 'queued']])

  // A current snapshot (seq 12) without it is the authority's word: the receipt retires.
  snapshot(12, [])
  expect(readPendingSubmissions('fenced').q?.status).toBe('retired')
  expect(getQueuedPrompts('fenced')).toEqual([])
})

// A replayed or delayed snapshot that still lists an `unknown` admission as `started` is stale for
// its status, not for its presence: the Discard card the owner's restart produced must stay.
it('a stale started snapshot keeps an existing unknown card and its Discard control', () => {
  reconcilePendingSubmissions('lost', [{ admission_id: 'b', status: 'started', user: 'lost' }])
  reconcilePendingSubmissions('lost', [{ admission_id: 'b', status: 'unknown', user: 'lost' }])
  reconcilePendingSubmissions('lost', [{ admission_id: 'b', status: 'started', user: 'lost' }])
  expect(getQueuedPrompts('lost').map(entry => [entry.id, entry.serverStatus])).toEqual([['b', 'unknown']])
  expect(readPendingSubmissions('lost').b?.status).toBe('unknown')

  // Its real absence from a current snapshot still retires it.
  reconcilePendingSubmissions('lost', [])
  expect(getQueuedPrompts('lost')).toEqual([])
})
