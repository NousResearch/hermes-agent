import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { ComposerAttachment } from './composer'
import {
  $parkedQueueSessions,
  $queuedPromptsBySession,
  clearQueuedPrompts,
  dequeueQueuedPrompt,
  enqueueQueuedPrompt,
  getQueuedPrompts,
  isQueueParked,
  migrateQueuedPrompts,
  parkQueuedPrompts,
  promoteQueuedPrompt,
  removeQueuedPrompt,
  sanitizePersistedQueueState,
  shouldAutoDrain,
  unparkQueuedPrompts,
  updateQueuedPrompt,
  updateQueuedPromptText
} from './composer-queue'

const SESSION_KEY = 'session-abc'
const QUEUE_STORAGE_KEY = 'hermes.desktop.composerQueue.v1'
const DAY_MS = 24 * 60 * 60 * 1000

function attachment(id: string, kind: ComposerAttachment['kind'] = 'file'): ComposerAttachment {
  return {
    id,
    kind,
    label: id,
    refText: `@file:${id}`
  }
}

describe('sanitizePersistedQueueState', () => {
  it('drops expired turns while preserving fresh valid entries', () => {
    const now = Date.UTC(2026, 8, 1)

    const fresh = {
      attachments: [],
      displayKind: 'hidden' as const,
      id: 'queued-fresh',
      queuedAt: now - DAY_MS,
      text: 'fresh setup note'
    }

    const expired = {
      attachments: [attachment('old-image', 'image')],
      id: 'queued-expired',
      queuedAt: now - 8 * DAY_MS,
      text: 'stale retry'
    }

    expect(
      sanitizePersistedQueueState(
        {
          'session-expired': [expired],
          'session-fresh': [fresh]
        },
        now
      )
    ).toEqual({ 'session-fresh': [fresh] })
  })

  it('writes the sanitized queue back during persisted-state load', async () => {
    const now = Date.now()
    const fresh = { attachments: [], id: 'fresh', queuedAt: now, text: 'send me' }
    const expired = { attachments: [], id: 'expired', queuedAt: now - 8 * DAY_MS, text: 'do not retry' }

    window.localStorage.setItem(
      QUEUE_STORAGE_KEY,
      JSON.stringify({ 'session-expired': [expired], 'session-fresh': [fresh] })
    )
    vi.resetModules()

    const reloaded = await import('./composer-queue')

    expect(reloaded.$queuedPromptsBySession.get()).toEqual({ 'session-fresh': [fresh] })
    expect(JSON.parse(window.localStorage.getItem(QUEUE_STORAGE_KEY)!)).toEqual({ 'session-fresh': [fresh] })
  })
})

describe('composer queue store', () => {
  beforeEach(() => {
    window.localStorage.removeItem(QUEUE_STORAGE_KEY)
    $queuedPromptsBySession.set({})
  })

  it('queues prompts in FIFO order', () => {
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'first' })
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'second' })

    expect(dequeueQueuedPrompt(SESSION_KEY)?.text).toBe('first')
    expect(dequeueQueuedPrompt(SESSION_KEY)?.text).toBe('second')
    expect(dequeueQueuedPrompt(SESSION_KEY)).toBeNull()
  })

  it('clones attachments when queueing', () => {
    const source = [attachment('a-1')]
    const queued = enqueueQueuedPrompt(SESSION_KEY, { attachments: source, text: 'check clones' })

    expect(queued).not.toBeNull()
    expect(getQueuedPrompts(SESSION_KEY)[0]?.attachments[0]).toEqual(source[0])
    expect(getQueuedPrompts(SESSION_KEY)[0]?.attachments[0]).not.toBe(source[0])
  })

  it('updates and removes queued entries by id', () => {
    const first = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'draft one' })
    const second = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'draft two' })

    expect(first).not.toBeNull()
    expect(second).not.toBeNull()

    expect(updateQueuedPromptText(SESSION_KEY, first!.id, 'draft one edited')).toBe(true)
    expect(getQueuedPrompts(SESSION_KEY).map(entry => entry.text)).toEqual(['draft one edited', 'draft two'])

    expect(removeQueuedPrompt(SESSION_KEY, first!.id)).toBe(true)
    expect(getQueuedPrompts(SESSION_KEY).map(entry => entry.text)).toEqual(['draft two'])
  })

  it('promotes a queued entry to the front', () => {
    const first = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'first' })
    const second = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'second' })
    const third = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'third' })

    expect(first).not.toBeNull()
    expect(second).not.toBeNull()
    expect(third).not.toBeNull()

    expect(promoteQueuedPrompt(SESSION_KEY, third!.id)).toBe(true)
    expect(getQueuedPrompts(SESSION_KEY).map(entry => entry.text)).toEqual(['third', 'first', 'second'])
    expect(promoteQueuedPrompt(SESSION_KEY, third!.id)).toBe(false)
  })

  it('updates queued text and attachment snapshot', () => {
    const first = enqueueQueuedPrompt(SESSION_KEY, { attachments: [attachment('f-1')], text: 'draft one' })
    const editedAttachments = [attachment('f-2'), attachment('f-3', 'image')]

    expect(first).not.toBeNull()
    expect(
      updateQueuedPrompt(SESSION_KEY, first!.id, {
        attachments: editedAttachments,
        text: 'edited text'
      })
    ).toBe(true)

    const queue = getQueuedPrompts(SESSION_KEY)
    expect(queue[0]?.text).toBe('edited text')
    expect(queue[0]?.attachments).toEqual(editedAttachments)
    expect(queue[0]?.attachments[0]).not.toBe(editedAttachments[0])
  })

  it('clears queue state for a session', () => {
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [attachment('img-1', 'image')], text: 'queued' })

    clearQueuedPrompts(SESSION_KEY)

    expect(getQueuedPrompts(SESSION_KEY)).toEqual([])
    expect($queuedPromptsBySession.get()[SESSION_KEY]).toBeUndefined()
    expect(window.localStorage.getItem(QUEUE_STORAGE_KEY)).toBeNull()
  })

  it('persists queue entries into local storage', () => {
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'persist me' })

    const raw = window.localStorage.getItem(QUEUE_STORAGE_KEY)
    expect(raw).toBeTruthy()

    const parsed = JSON.parse(String(raw)) as Record<string, { text: string }[]>
    expect(parsed[SESSION_KEY]?.[0]?.text).toBe('persist me')
  })
})

describe('migrateQueuedPrompts', () => {
  beforeEach(() => {
    window.localStorage.removeItem(QUEUE_STORAGE_KEY)
    $queuedPromptsBySession.set({})
  })

  it('moves entries from a dead runtime key onto the live one', () => {
    enqueueQueuedPrompt('rt-old', { attachments: [], text: 'stranded' })

    expect(migrateQueuedPrompts('rt-old', 'rt-new')).toBe(true)
    expect(getQueuedPrompts('rt-old')).toEqual([])
    expect(getQueuedPrompts('rt-new').map(e => e.text)).toEqual(['stranded'])
    // The dead key is dropped from the store entirely.
    expect($queuedPromptsBySession.get()['rt-old']).toBeUndefined()
  })

  it('appends after existing target entries (FIFO preserved)', () => {
    enqueueQueuedPrompt('rt-new', { attachments: [], text: 'already here' })
    enqueueQueuedPrompt('rt-old', { attachments: [], text: 'migrated' })

    migrateQueuedPrompts('rt-old', 'rt-new')

    expect(getQueuedPrompts('rt-new').map(e => e.text)).toEqual(['already here', 'migrated'])
  })

  it('is a no-op when source is empty or keys match', () => {
    expect(migrateQueuedPrompts('rt-old', 'rt-new')).toBe(false)
    expect(migrateQueuedPrompts('rt-x', 'rt-x')).toBe(false)
  })

  it('must not be used across sessions without a same-lineage guard (documents the leak)', () => {
    // ChatView used to call migrateQueuedPrompts(selectedA, routeKeyB) during the
    // route-ahead/store-lag window of a session switch. That re-homes A's queue
    // onto B so the idle ChatBar on B auto-drains it into the wrong conversation.
    // The guard lives in shouldMigrateComposerScope — this test locks the
    // underlying hazard so a future caller cannot treat migrate as free.
    enqueueQueuedPrompt('root-a', { attachments: [], text: 'belongs to A' })

    expect(migrateQueuedPrompts('root-a', 'root-b')).toBe(true)
    expect(getQueuedPrompts('root-a')).toEqual([])
    expect(getQueuedPrompts('root-b').map(e => e.text)).toEqual(['belongs to A'])
  })
})

describe('shouldAutoDrain', () => {
  it('drains whenever idle with a non-empty queue', () => {
    expect(shouldAutoDrain({ isBusy: false, queueLength: 1 })).toBe(true)
  })

  it('drains on mount/reconnect with no observed busy edge', () => {
    // The whole point of dropping the edge: a remount resets the busy ref, so an
    // edge-gated drain would strand the entry. Idle + non-empty must still fire.
    expect(shouldAutoDrain({ isBusy: false, queueLength: 2 })).toBe(true)
  })

  it('does not drain mid-turn', () => {
    expect(shouldAutoDrain({ isBusy: true, queueLength: 1 })).toBe(false)
  })

  it('does not drain an empty queue', () => {
    expect(shouldAutoDrain({ isBusy: false, queueLength: 0 })).toBe(false)
  })

  it('does not drain a parked queue, even when idle', () => {
    // The Stop/Esc settle edge: busy just flipped false but the user asked to
    // HALT — the park must hold the head back until they resume.
    expect(shouldAutoDrain({ isBusy: false, parked: true, queueLength: 1 })).toBe(false)
  })

  it('drains again once the park is lifted', () => {
    expect(shouldAutoDrain({ isBusy: false, parked: false, queueLength: 1 })).toBe(true)
  })
})

describe('parked queue sessions', () => {
  beforeEach(() => {
    window.localStorage.removeItem(QUEUE_STORAGE_KEY)
    $queuedPromptsBySession.set({})
    $parkedQueueSessions.set({})
  })

  it('parks only sessions with queued entries', () => {
    expect(parkQueuedPrompts(SESSION_KEY)).toBe(false)
    expect(isQueueParked(SESSION_KEY)).toBe(false)

    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'held back' })

    expect(parkQueuedPrompts(SESSION_KEY)).toBe(true)
    expect(isQueueParked(SESSION_KEY)).toBe(true)
  })

  it('unparks explicitly', () => {
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'held back' })
    parkQueuedPrompts(SESSION_KEY)

    unparkQueuedPrompts(SESSION_KEY)

    expect(isQueueParked(SESSION_KEY)).toBe(false)
  })

  it('queueing a fresh prompt lifts the park', () => {
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'held back' })
    parkQueuedPrompts(SESSION_KEY)

    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'new intent' })

    expect(isQueueParked(SESSION_KEY)).toBe(false)
  })

  it('emptying the queue drops the park', () => {
    const entry = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'held back' })
    parkQueuedPrompts(SESSION_KEY)

    removeQueuedPrompt(SESSION_KEY, entry!.id)

    expect(isQueueParked(SESSION_KEY)).toBe(false)
  })

  it('a park travels with migrated entries', () => {
    // A backend bounce right after Stop re-keys the queue; shedding the park
    // there would auto-send the exact prompts the user just halted.
    enqueueQueuedPrompt('rt-old', { attachments: [], text: 'held back' })
    parkQueuedPrompts('rt-old')

    migrateQueuedPrompts('rt-old', 'rt-new')

    expect(isQueueParked('rt-old')).toBe(false)
    expect(isQueueParked('rt-new')).toBe(true)
  })

  it('migration without a park does not invent one', () => {
    enqueueQueuedPrompt('rt-old', { attachments: [], text: 'flowing' })

    migrateQueuedPrompts('rt-old', 'rt-new')

    expect(isQueueParked('rt-new')).toBe(false)
  })
})

describe('hidden entries', () => {
  beforeEach(() => {
    clearQueuedPrompts('hidden-session')
  })

  it('keeps the hidden kind on a queued note and leaves visible entries without one', () => {
    enqueueQueuedPrompt('hidden-session', { text: '[setup] links opened', attachments: [], displayKind: 'hidden' })
    enqueueQueuedPrompt('hidden-session', { text: 'Start without connections.', attachments: [] })

    expect(getQueuedPrompts('hidden-session').map(({ text, displayKind }) => ({ text, displayKind }))).toEqual([
      { text: '[setup] links opened', displayKind: 'hidden' },
      { text: 'Start without connections.', displayKind: undefined }
    ])
  })
})
