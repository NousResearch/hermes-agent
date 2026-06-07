import { beforeEach, describe, expect, it } from 'vitest'

import type { ComposerAttachment } from './composer'
import {
  $queuedPromptsBySession,
  clearQueuedPrompts,
  dequeueQueuedPrompt,
  enqueueQueuedPrompt,
  getQueuedPrompts,
  mergeAllQueuedPrompts,
  promoteQueuedPrompt,
  removeQueuedPrompt,
  shouldAutoDrainOnSettle,
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

describe('mergeAllQueuedPrompts', () => {
  beforeEach(() => {
    window.localStorage.removeItem(QUEUE_STORAGE_KEY)
    $queuedPromptsBySession.set({})
  })

  it('merges multiple entries into one with blank-line separator', () => {
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'first' })
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'second' })
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'third' })

    const result = mergeAllQueuedPrompts(SESSION_KEY)

    expect(result).toBe(true)
    const queue = getQueuedPrompts(SESSION_KEY)
    expect(queue).toHaveLength(1)
    expect(queue[0]!.text).toBe('first\n\nsecond\n\nthird')
  })

  it('preserves the head entry id and queuedAt', () => {
    const first = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'head' })
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'tail' })

    mergeAllQueuedPrompts(SESSION_KEY)

    const queue = getQueuedPrompts(SESSION_KEY)
    expect(queue[0]!.id).toBe(first!.id)
    expect(queue[0]!.queuedAt).toBe(first!.queuedAt)
  })

  it('concatenates attachments from all entries', () => {
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [attachment('a-1')], text: 'first' })
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [attachment('a-2'), attachment('a-3')], text: 'second' })

    mergeAllQueuedPrompts(SESSION_KEY)

    const queue = getQueuedPrompts(SESSION_KEY)
    expect(queue[0]!.attachments).toHaveLength(3)
    expect(queue[0]!.attachments.map(a => a.id)).toEqual(['a-1', 'a-2', 'a-3'])
  })

  it('returns false when fewer than 2 entries', () => {
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'only one' })

    expect(mergeAllQueuedPrompts(SESSION_KEY)).toBe(false)
    expect(getQueuedPrompts(SESSION_KEY)).toHaveLength(1)
  })

  it('returns false for empty queue', () => {
    expect(mergeAllQueuedPrompts(SESSION_KEY)).toBe(false)
  })

  it('returns false for null key', () => {
    expect(mergeAllQueuedPrompts(null)).toBe(false)
  })

  it('filters out empty text entries when merging', () => {
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'first' })
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [attachment('img')], text: '' })
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'third' })

    mergeAllQueuedPrompts(SESSION_KEY)

    const queue = getQueuedPrompts(SESSION_KEY)
    expect(queue[0]!.text).toBe('first\n\nthird')
    expect(queue[0]!.attachments).toHaveLength(1)
  })
})

describe('shouldAutoDrainOnSettle', () => {
  const base = { isBusy: false, queueLength: 1, wasBusy: true }

  it('drains the next queued prompt when a turn settles', () => {
    expect(shouldAutoDrainOnSettle(base)).toBe(true)
  })

  it('drains after an interrupt — the settle edge is the same', () => {
    // Interrupting to reach a queued message is the point of the queue; the
    // gateway emits the same settle whether the turn finished or was stopped.
    expect(shouldAutoDrainOnSettle(base)).toBe(true)
  })

  it('does not drain when the queue is empty', () => {
    expect(shouldAutoDrainOnSettle({ ...base, queueLength: 0 })).toBe(false)
  })

  it('ignores steady busy state (no true → false transition)', () => {
    expect(shouldAutoDrainOnSettle({ ...base, isBusy: true })).toBe(false)
  })

  it('ignores busy entry (false → true, not a settle)', () => {
    expect(shouldAutoDrainOnSettle({ ...base, isBusy: true, wasBusy: false })).toBe(false)
  })

  it('ignores steady idle state (was not busy)', () => {
    expect(shouldAutoDrainOnSettle({ ...base, wasBusy: false })).toBe(false)
  })
})
