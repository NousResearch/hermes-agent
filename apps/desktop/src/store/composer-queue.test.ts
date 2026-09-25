import { beforeEach, describe, expect, it } from 'vitest'

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
  shouldAutoDrain,
  unparkQueuedPrompts,
  updateQueuedPrompt,
  updateQueuedPromptText
} from './composer-queue'

const SESSION_KEY = 'session-abc'
const QUEUE_STORAGE_KEY = 'hermes.desktop.composerQueue.v1'

function attachment(id: string, kind: ComposerAttachment['kind'] = 'file'): ComposerAttachment {
  return {
    id,
    kind,
    label: id,
    refText: `@file:${id}`
  }
}

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
})

describe('shouldAutoDrain', () => {
  it('drains whenever idle with a non-empty queue', () => {
    expect(shouldAutoDrain({ isBusy: false, queueLength: 1 })).toBe(true)
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

describe('cross-window sync (#46732)', () => {
  beforeEach(() => {
    window.localStorage.removeItem(QUEUE_STORAGE_KEY)
    $queuedPromptsBySession.set({})
  })

  const storedEntry = (id: string, text: string) => ({ id, text, attachments: [], queuedAt: 1 })

  const dispatchStorage = (key: null | string, newValue: null | string) => {
    window.dispatchEvent(new StorageEvent('storage', { key, newValue }))
  }

  it("adopts another window's write from the storage event", () => {
    window.localStorage.setItem(
      QUEUE_STORAGE_KEY,
      JSON.stringify({ 'session-other': [storedEntry('q1', 'from other window')] })
    )

    dispatchStorage(QUEUE_STORAGE_KEY, window.localStorage.getItem(QUEUE_STORAGE_KEY))

    expect(getQueuedPrompts('session-other').map(entry => entry.text)).toEqual(['from other window'])
  })

  it("does not clobber another window's entries when saving its own (same-frame race)", () => {
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'mine first' })

    // Another window queues into its own session directly in storage, faster
    // than any storage event could reach us.
    window.localStorage.setItem(
      QUEUE_STORAGE_KEY,
      JSON.stringify({
        ...JSON.parse(window.localStorage.getItem(QUEUE_STORAGE_KEY)!),
        'session-other': [storedEntry('q2', 'theirs')]
      })
    )

    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'mine second' })

    expect(getQueuedPrompts(SESSION_KEY).map(entry => entry.text)).toEqual(['mine first', 'mine second'])
    expect(getQueuedPrompts('session-other').map(entry => entry.text)).toEqual(['theirs'])
  })

  it('drops entries locally once another window drains them', () => {
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'drained elsewhere' })

    window.localStorage.setItem(QUEUE_STORAGE_KEY, JSON.stringify({}))
    dispatchStorage(QUEUE_STORAGE_KEY, '{}')

    expect(getQueuedPrompts(SESSION_KEY)).toEqual([])
  })

  it('resyncs on a full storage clear (event.key === null)', () => {
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'wiped' })

    window.localStorage.removeItem(QUEUE_STORAGE_KEY)
    dispatchStorage(null, null)

    expect(getQueuedPrompts(SESSION_KEY)).toEqual([])
  })

  it('ignores storage events for unrelated keys', () => {
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'kept' })

    dispatchStorage('unrelated.key', '{}')

    expect(getQueuedPrompts(SESSION_KEY).map(entry => entry.text)).toEqual(['kept'])
  })
})
