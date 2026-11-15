import { beforeEach, expect, it, vi } from 'vitest'

import { CanonicalDesktopProtocol } from '@/api/canonical-protocol'
import { handleSessionInfoEvent } from '@/app/session/hooks/use-message-stream/gateway-event/session-info'
import type { GatewayEventContext } from '@/app/session/hooks/use-message-stream/gateway-event/types'
import { applyRuntimeInfo } from '@/app/session/hooks/use-session-actions/utils'
import { isSteerableEntry } from '@/store/composer-queue'
import type { SessionInfo } from '@/types/hermes'

import { $queuedPromptsBySession, enqueueQueuedPrompt, getQueuedPrompts, migrateQueuedPrompts } from './composer-queue'
import { dropPendingSubmission, readPendingSubmissions, reconcilePendingSubmissions, trackPendingSubmission } from './pending-submissions'
import { $sessions } from './session'

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

it('carries the queued-row controls the owner lists onto the card, and none when it lists none (RT-35)', () => {
  reconcilePendingSubmissions('chat', [
    { admission_id: 'served', status: 'queued', user: 'edit me', controls: ['steer', 'replace'] },
    { admission_id: 'older', status: 'queued', user: 'older owner' }
  ])

  expect(getQueuedPrompts('chat').map(({ id, serverControls }) => ({ id, serverControls }))).toEqual([
    { id: 'served', serverControls: ['steer', 'replace'] },
    { id: 'older', serverControls: undefined }
  ])
  expect(Object.values(readPendingSubmissions('chat')).some(entry => 'controls' in entry)).toBe(false)
})

it('a steer in transfer hides its card without retiring it, so a return to the queue repaints it in place (RT-35)', () => {
  const later = { admission_id: 'later', status: 'queued', user: 'queued after it' }
  reconcilePendingSubmissions('chat', [{ admission_id: 'moving', status: 'queued', user: 'steer me' }, later])
  // Handed to the running turn: listed `started` and `steering` until that turn settles it.
  reconcilePendingSubmissions('chat', [
    { admission_id: 'moving', status: 'started', steering: true, user: 'steer me' },
    later
  ])
  expect(getQueuedPrompts('chat').map(({ id }) => id)).toEqual(['later'])

  // The text never reached the turn: the owner puts the same admission back in its FIFO place, ahead
  // of the admission queued after it, and the card is shown where the gateway will run it.
  reconcilePendingSubmissions('chat', [{ admission_id: 'moving', status: 'queued', user: 'steer me' }, later])
  expect(getQueuedPrompts('chat').map(({ id, serverStatus }) => ({ id, serverStatus }))).toEqual([
    { id: 'moving', serverStatus: 'queued' },
    { id: 'later', serverStatus: 'queued' }
  ])
  reconcilePendingSubmissions('chat', [later])

  // An ordinary claim still only moves forward: a replayed `queued` never repaints a started row.
  reconcilePendingSubmissions('chat', [{ admission_id: 'moving', status: 'started', user: 'steer me' }])
  reconcilePendingSubmissions('chat', [{ admission_id: 'moving', status: 'queued', user: 'steer me' }])
  expect(getQueuedPrompts('chat')).toEqual([])
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

// After compression the panel reads the lineage root while the runtime reports the tip. A tip-keyed
// projection only reached the panel through rotation migrations, which stacked copies there that no
// later snapshot could retire: duplicate, undeletable, back after a delete (#138934).
it('projects a compressed tip snapshot onto the lineage key the panel reads', () => {
  $sessions.set([{ id: 'tip', _lineage_root_id: 'root' } as SessionInfo])

  try {
    const listed = [{ admission_id: 'x', status: 'queued', user: 'later' }]
    reconcilePendingSubmissions('tip', listed, { epoch: 'e', sequence: 1 })
    migrateQueuedPrompts('tip', 'root')
    reconcilePendingSubmissions('tip', listed, { epoch: 'e', sequence: 2 })
    migrateQueuedPrompts('tip', 'root')
    expect(getQueuedPrompts('root').map(entry => [entry.id, entry.serverStatus])).toEqual([['x', 'queued']])

    // Withdrawn at the gateway: the next snapshot no longer lists it.
    reconcilePendingSubmissions('tip', [], { epoch: 'e', sequence: 3 })
    expect(getQueuedPrompts('root')).toEqual([])

    // The same earlier status mirrored again does not bring it back.
    reconcilePendingSubmissions('tip', listed, { epoch: 'e', sequence: 2 })
    migrateQueuedPrompts('tip', 'root')
    expect(getQueuedPrompts('root')).toEqual([])
  } finally {
    $sessions.set([])
  }
})

// The tip->root row can load after a snapshot was already journaled under the tip. The fence that
// snapshot recorded must follow the card to the root, or a delayed older empty one hides it.
it('late lineage discovery keeps the ordering fence: an older empty snapshot leaves the card', () => {
  try {
    reconcilePendingSubmissions('tip', [{ admission_id: 'x', status: 'queued', user: 'waiting' }], { epoch: 'e', sequence: 10 })
    expect(getQueuedPrompts('tip').map(entry => entry.id)).toEqual(['x'])

    $sessions.set([{ id: 'tip', _lineage_root_id: 'root' } as SessionInfo])
    migrateQueuedPrompts('tip', 'root')

    reconcilePendingSubmissions('tip', [], { epoch: 'e', sequence: 7 })
    expect(getQueuedPrompts('root').map(entry => [entry.id, entry.serverStatus])).toEqual([['x', 'queued']])
    expect(readPendingSubmissions('root').x?.status).toBe('queued')

    // A current empty snapshot still retires it.
    reconcilePendingSubmissions('tip', [], { epoch: 'e', sequence: 12 })
    expect(getQueuedPrompts('root')).toEqual([])
  } finally {
    $sessions.set([])
  }
})

// Submit tracks under the stored id it targets, which can be the tip. Its stronger status and the
// display text only it knows must still meet the root-projected snapshots.
it('a tip-keyed tracked receipt keeps its status, text and Discard across root projection', () => {
  try {
    trackPendingSubmission('tip', { id: 'u', text: 'full prompt', displayText: '/cmd', status: 'unknown' })
    $sessions.set([{ id: 'tip', _lineage_root_id: 'root' } as SessionInfo])

    // A replayed `started` is stale for an admission already answered `unknown`.
    reconcilePendingSubmissions('tip', [{ admission_id: 'u', status: 'started', user: 'full prompt' }], { epoch: 'e', sequence: 1 })
    reconcilePendingSubmissions('tip', [{ admission_id: 'u', status: 'started', user: 'full prompt' }], { epoch: 'e', sequence: 2 })

    expect(getQueuedPrompts('root').map(({ id, text, displayText, serverStatus }) => ({ id, text, displayText, serverStatus })))
      .toEqual([{ id: 'u', text: 'full prompt', displayText: '/cmd', serverStatus: 'unknown' }])
    expect(readPendingSubmissions('root').u).toMatchObject({ status: 'unknown', displayText: '/cmd' })
    expect(readPendingSubmissions('tip')).toEqual({})

    // Tracking once the lineage is known lands where reconciliation reads.
    trackPendingSubmission('tip', { id: 'v', text: 'next', displayText: '/next' })
    expect(readPendingSubmissions('root').v).toMatchObject({ displayText: '/next' })
  } finally {
    $sessions.set([])
  }
})

// Every send tracks its admission before the request and again after the reply, on the renderer
// thread. That must cost the one admission's record, not a parse of every saved session's journal.
it('tracking a submission reads its own record without enumerating other sessions', () => {
  for (let index = 0; index < 20; index += 1) {
    trackPendingSubmission(`other-${index}`, { id: `a-${index}`, text: 'elsewhere', status: 'queued' })
  }

  window.localStorage.setItem('hermes.desktop.pendingSubmissions.v1', JSON.stringify({ solo: { old: { id: 'old', text: 'legacy', displayText: '/old' } } }))
  const enumerate = vi.spyOn(window.localStorage, 'key')

  try {
    trackPendingSubmission('solo', { id: 'mine', text: 'hi', displayText: '/hi' })
    trackPendingSubmission('solo', { id: 'mine', text: 'hi', status: 'queued' })
    // A legacy receipt still merges into its tracked successor.
    trackPendingSubmission('solo', { id: 'old', text: '', status: 'unknown' })
    expect(enumerate).not.toHaveBeenCalled()

    // A newly known tip->root lineage adopts the tip's journal once; later sends read directly.
    trackPendingSubmission('late-tip', { id: 'early', text: 'early', displayText: '/early', status: 'unknown' })
    $sessions.set([{ id: 'late-tip', _lineage_root_id: 'late-root' } as SessionInfo])
    trackPendingSubmission('late-tip', { id: 'next', text: 'next' })
    expect(readPendingSubmissions('late-root').early).toMatchObject({ status: 'unknown', displayText: '/early' })
    enumerate.mockClear()
    trackPendingSubmission('late-tip', { id: 'next', text: 'next', status: 'queued' })
    expect(enumerate).not.toHaveBeenCalled()
  } finally {
    enumerate.mockRestore()
    $sessions.set([])
  }

  expect(readPendingSubmissions('solo')).toMatchObject({
    mine: { text: 'hi', displayText: '/hi', status: 'queued' },
    old: { text: 'legacy', displayText: '/old', status: 'unknown' }
  })
  expect(readPendingSubmissions('late-root').next).toMatchObject({ status: 'queued' })
})

// #ATLAS-125 — a pre-submit journal write filled the 50 MiB origin quota and aborted
// submits before they reached the gateway. The journal is card storage, never a
// replay source: its text is bounded, its statusless pre-tracks are droppable, and
// no setItem of its own may fail a submit.
it('caps journal text at 4096 chars: a slash-expanded submit cannot fill the origin (#ATLAS-125)', () => {
  const expanded = 'x'.repeat(282_744)
  trackPendingSubmission('chat', { id: 'big', text: expanded })
  trackPendingSubmission('chat', { id: 'wide', text: 'short', displayText: 'y'.repeat(9_000) })

  const journal = readPendingSubmissions('chat')
  expect(journal.big?.text.length).toBe(4_096)
  expect(journal.big?.text).toBe(expanded.slice(0, 4_096))
  expect(journal.wide?.displayText?.length).toBe(4_096)
})

it('drops a statusless pre-track that never became an admission, but never a live card (#ATLAS-125)', () => {
  trackPendingSubmission('chat', { id: 'legacy-ack', text: 'sent to a legacy owner' })
  trackPendingSubmission('chat', { id: 'live', text: 'a real card', status: 'queued' })

  dropPendingSubmission('chat', 'legacy-ack')
  dropPendingSubmission('chat', 'live')
  dropPendingSubmission('chat', 'never-tracked')

  const journal = readPendingSubmissions('chat')
  expect(journal['legacy-ack']).toBeUndefined()
  expect(journal.live?.status).toBe('queued')
})

it('a fenced snapshot retires statusless pre-tracks it lists under neither id; an unfenced one says nothing (#ATLAS-125)', () => {
  trackPendingSubmission('chat', { id: 'pre', text: 'in flight' })

  reconcilePendingSubmissions('chat', [])
  expect(readPendingSubmissions('chat').pre).toBeDefined()

  reconcilePendingSubmissions('chat', [], { epoch: 'epoch-1', sequence: 5 })
  expect(readPendingSubmissions('chat').pre).toBeUndefined()
})

it('folds an input-keyed pre-track into its admission key instead of leaking both keys (#ATLAS-125)', () => {
  trackPendingSubmission('chat', { id: 'input-1', text: 'my prompt', displayText: 'shown text' })

  reconcilePendingSubmissions(
    'chat',
    [{ admission_id: 'admission-9', input_id: 'input-1', status: 'queued', user: 'my prompt' }],
    { epoch: 'epoch-1', sequence: 2 }
  )

  const journal = readPendingSubmissions('chat')
  expect(journal['input-1']).toBeUndefined()
  expect(journal['admission-9']).toMatchObject({ status: 'queued', displayText: 'shown text' })
  expect(getQueuedPrompts('chat')).toMatchObject([{ id: 'admission-9', displayText: 'shown text' }])
})

it('a full origin evicts statusless pre-tracks first, never evicts live cards for a pre-track, and never throws (#ATLAS-125)', () => {
  trackPendingSubmission('chat', { id: 'orphan', text: 'never admitted' })
  trackPendingSubmission('chat', { id: 'card', text: 'a real card', status: 'queued' })

  const realSet = window.localStorage.setItem.bind(window.localStorage)
  const setItem = vi.spyOn(window.localStorage, 'setItem')

  // The origin is full: the first write throws quota. The journal must evict
  // the one entry that carries no information (the statusless pre-track) and
  // land the new entry — a pending card is cosmetic, a failed submit is not.
  let quotaThrown = false
  setItem.mockImplementation((key: string, value: string) => {
    if (!quotaThrown) {
      quotaThrown = true
      throw new DOMException('quota exceeded', 'QuotaExceededError')
    }

    realSet(key, value)
  })

  try {
    expect(() => trackPendingSubmission('chat', { id: 'fresh', text: 'after the purge' })).not.toThrow()
  } finally {
    setItem.mockRestore()
  }

  const journal = readPendingSubmissions('chat')
  expect(journal.orphan).toBeUndefined()
  expect(journal.fresh?.text).toBe('after the purge')
  expect(journal.card?.status).toBe('queued')

  // Remove the statusless entry so only the live card remains, then hold the
  // origin full: a pre-track must give up before touching a live queued card.
  dropPendingSubmission('chat', 'fresh')

  const alwaysFull = vi.spyOn(window.localStorage, 'setItem').mockImplementation(() => {
    throw new DOMException('quota exceeded', 'QuotaExceededError')
  })

  try {
    expect(() => trackPendingSubmission('chat', { id: 'skipped-pre', text: 'no room' })).not.toThrow()
  } finally {
    alwaysFull.mockRestore()
  }

  expect(readPendingSubmissions('chat').card?.status).toBe('queued')
  expect(readPendingSubmissions('chat')['skipped-pre']).toBeUndefined()
})

it('a fresh module load sweeps leftover pre-tracks and rewrites oversized stored texts (#ATLAS-125)', async () => {
  // Seed raw storage (not via trackPendingSubmission, which bounds on write) to
  // reproduce what the pre-fix build left behind: statusless leak + fat card.
  const seedKey = (session: string, id: string) => `hermes.desktop.pendingSubmissions.v2:${JSON.stringify([session, id])}`
  window.localStorage.setItem(seedKey('chat', 'leftover'), JSON.stringify({ id: 'leftover', text: 'x'.repeat(90_000) }))
  window.localStorage.setItem(seedKey('chat', 'fat'), JSON.stringify({ id: 'fat', text: 'z'.repeat(50_000), status: 'queued' }))

  vi.resetModules()
  await import('./pending-submissions')

  const journal = readPendingSubmissions('chat')
  expect(journal.leftover).toBeUndefined()
  expect(journal.fat).toMatchObject({ status: 'queued' })
  expect(journal.fat?.text.length).toBe(4_096)
})
