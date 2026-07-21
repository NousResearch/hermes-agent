import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { ComposerAttachment } from './composer'
import {
  $queuedPromptsBySession,
  clearQueuedPrompts,
  dequeueQueuedPrompt,
  enqueueQueuedPrompt,
  flushQueuedPromptMutations,
  getQueuedPrompts,
  incrementQueuedPromptAttempts,
  markQueuedPromptAcceptedAtomic,
  migrateQueuedPrompts,
  promoteQueuedPrompt,
  removeQueuedPrompt,
  removeQueuedPromptById,
  QUEUED_PROMPT_ACCEPTANCE_RETRY_MS,
  queuedPromptAwaitingCompletion,
  resetQueuedPromptAttempts,
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
  beforeEach(async () => {
    await flushQueuedPromptMutations()
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

  it('persists queue entries into local storage', async () => {
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'persist me' })
    await flushQueuedPromptMutations()

    const raw = window.localStorage.getItem(QUEUE_STORAGE_KEY)
    expect(raw).toBeTruthy()

    const parsed = JSON.parse(String(raw)) as Record<string, { text: string }[]>
    expect(parsed[SESSION_KEY]?.[0]?.text).toBe('persist me')
  })

  it('persists one shared automatic retry count on the queue entry', async () => {
    const entry = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'retry me' })

    expect(entry?.attempts).toBe(0)
    expect(incrementQueuedPromptAttempts(entry!.id)?.attempts).toBe(1)
    expect(incrementQueuedPromptAttempts(entry!.id)?.attempts).toBe(2)
    expect(getQueuedPrompts(SESSION_KEY)[0]?.attempts).toBe(2)
    await flushQueuedPromptMutations()
    expect(JSON.parse(String(window.localStorage.getItem(QUEUE_STORAGE_KEY)))[SESSION_KEY][0].attempts).toBe(2)

    expect(resetQueuedPromptAttempts(entry!.id)).toBe(true)
    expect(getQueuedPrompts(SESSION_KEY)[0]?.attempts).toBe(0)
  })


})

describe('migrateQueuedPrompts', () => {
  beforeEach(async () => {
    await flushQueuedPromptMutations()
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

  it('removes an accepted entry by identity after its runtime key migrates', () => {
    const entry = enqueueQueuedPrompt('rt-old', { attachments: [], text: 'submit once' })

    expect(migrateQueuedPrompts('rt-old', 'rt-new')).toBe(true)
    expect(removeQueuedPromptById(entry!.id)).toBe(true)
    expect(getQueuedPrompts('rt-new')).toEqual([])
  })

  it('appends after existing target entries (FIFO preserved)', () => {
    enqueueQueuedPrompt('rt-new', { attachments: [], text: 'already here' })
    enqueueQueuedPrompt('rt-old', { attachments: [], text: 'migrated' })

    migrateQueuedPrompts('rt-old', 'rt-new')

    expect(getQueuedPrompts('rt-new').map(e => e.text)).toEqual(['already here', 'migrated'])
  })

  it('keeps an older source entry ahead of a newer target entry', () => {
    const now = vi.spyOn(Date, 'now')
    now.mockReturnValue(100)
    enqueueQueuedPrompt('rt-old', { attachments: [], text: 'older source' })
    now.mockReturnValue(200)
    enqueueQueuedPrompt('rt-new', { attachments: [], text: 'newer target' })

    migrateQueuedPrompts('rt-old', 'rt-new')

    expect(getQueuedPrompts('rt-new').map(e => e.text)).toEqual(['older source', 'newer target'])
    now.mockRestore()
  })

  it('holds accepted custody until its bounded replay lease expires', async () => {
    const entry = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'retain me' })!
    await flushQueuedPromptMutations()
    const accepted = await markQueuedPromptAcceptedAtomic(entry.id)
    const acceptedAt = accepted!.acceptedAt!

    expect(queuedPromptAwaitingCompletion(accepted!, acceptedAt + QUEUED_PROMPT_ACCEPTANCE_RETRY_MS - 1)).toBe(true)
    expect(queuedPromptAwaitingCompletion(accepted!, acceptedAt + QUEUED_PROMPT_ACCEPTANCE_RETRY_MS)).toBe(false)
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
