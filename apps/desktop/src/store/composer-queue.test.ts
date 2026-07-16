import { beforeEach, describe, expect, it } from 'vitest'

import type { ComposerAttachment } from './composer'
import {
  $queuedPromptsBySession,
  clearQueuedPrompts,
  dequeueQueuedPrompt,
  enqueueQueuedPrompt,
  getQueuedPrompts,
  migrateQueuedPrompts,
  promoteQueuedPrompt,
  removeQueuedPrompt,
  shouldAutoDrain,
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

describe('cross-window sync (#46732)', () => {
  beforeEach(() => {
    window.localStorage.removeItem(QUEUE_STORAGE_KEY)
    $queuedPromptsBySession.set({})
  })

  // Simulate another window writing the shared key: localStorage mutates and a
  // `storage` event fires — that event never fires in the writing window itself,
  // so dispatching it manually is exactly the other-window signal.
  function otherWindowWrites(state: Record<string, unknown>) {
    const value = JSON.stringify(state)
    window.localStorage.setItem(QUEUE_STORAGE_KEY, value)
    window.dispatchEvent(new StorageEvent('storage', { key: QUEUE_STORAGE_KEY, newValue: value }))
  }

  it('adopts another window\u2019s write into the local atom', () => {
    otherWindowWrites({ 'session-remote': [{ id: 'q-1', text: 'from window B', attachments: [], queuedAt: 1 }] })

    expect(getQueuedPrompts('session-remote').map(e => e.text)).toEqual(['from window B'])
  })

  it('does not clobber another window\u2019s entries when saving its own', () => {
    // Window B enqueues for its session; this window then enqueues for a
    // different session. Both must survive in storage.
    otherWindowWrites({ 'session-b': [{ id: 'q-b', text: 'B entry', attachments: [], queuedAt: 1 }] })
    enqueueQueuedPrompt('session-a', { attachments: [], text: 'A entry' })

    const persisted = JSON.parse(window.localStorage.getItem(QUEUE_STORAGE_KEY) ?? '{}')
    expect(Object.keys(persisted).sort()).toEqual(['session-a', 'session-b'])
  })

  it('merges over live storage even without a storage event (same-frame race)', () => {
    // The `storage` event is asynchronous in real browsers; a write racing it
    // must still be preserved because writeSession re-reads storage at save time.
    window.localStorage.setItem(
      QUEUE_STORAGE_KEY,
      JSON.stringify({ 'session-b': [{ id: 'q-b', text: 'unsynced B entry', attachments: [], queuedAt: 1 }] })
    )

    enqueueQueuedPrompt('session-a', { attachments: [], text: 'A entry' })

    const persisted = JSON.parse(window.localStorage.getItem(QUEUE_STORAGE_KEY) ?? '{}')
    expect(Object.keys(persisted).sort()).toEqual(['session-a', 'session-b'])
  })

  it('drops entries locally once another window drains them', () => {
    enqueueQueuedPrompt('session-a', { attachments: [], text: 'about to be drained elsewhere' })

    // Window B drains session-a and persists the now-empty map.
    otherWindowWrites({})

    expect(getQueuedPrompts('session-a')).toEqual([])
    expect(dequeueQueuedPrompt('session-a')).toBeNull()
  })

  it('resyncs on full storage clear (event.key === null)', () => {
    enqueueQueuedPrompt('session-a', { attachments: [], text: 'entry' })

    window.localStorage.clear()
    window.dispatchEvent(new StorageEvent('storage', { key: null }))

    expect(getQueuedPrompts('session-a')).toEqual([])
  })

  it('ignores storage events for unrelated keys', () => {
    enqueueQueuedPrompt('session-a', { attachments: [], text: 'kept' })

    window.dispatchEvent(new StorageEvent('storage', { key: 'some.other.key', newValue: '"x"' }))

    expect(getQueuedPrompts('session-a').map(e => e.text)).toEqual(['kept'])
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
})
