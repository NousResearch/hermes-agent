import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest'

import type { ComposerAttachment } from './composer'
import {
  $queuedPromptsBySession,
  clearQueuedPrompts,
  dequeueQueuedPrompt,
  enqueueQueuedPrompt,
  getQueuedPrompts,
  loadQueueState,
  migrateQueuedPrompts,
  promoteQueuedPrompt,
  removeQueuedPrompt,
  shouldAutoDrain,
  updateQueuedPrompt,
  updateQueuedPromptText
} from './composer-queue'
import { profileSessionKey } from './session-identity'

const SESSION_KEY = profileSessionKey('default', 'session-abc')
const QUEUE_STORAGE_KEY = 'hermes.desktop.composerQueue.v2'
const LEGACY_QUEUE_STORAGE_KEY = 'hermes.desktop.composerQueue.v1'

class StorageFake implements Storage {
  readonly events: string[] = []
  readonly values = new Map<string, string>()
  failWrites = false

  get length() {
    return this.values.size
  }

  clear() {
    this.events.push('clear')
    this.values.clear()
  }

  getItem(key: string) {
    this.events.push(`get:${key}`)

    return this.values.get(key) ?? null
  }

  key(index: number) {
    return [...this.values.keys()][index] ?? null
  }

  removeItem(key: string) {
    this.events.push(`remove:${key}`)
    this.values.delete(key)
  }

  setItem(key: string, value: string) {
    this.events.push(`set:${key}`)

    if (this.failWrites) {
      throw new Error('storage write failed')
    }

    this.values.set(key, value)
  }
}

const localStorageDescriptor = Object.getOwnPropertyDescriptor(window, 'localStorage')
const localStorageFake = new StorageFake()

beforeAll(() => {
  Object.defineProperty(window, 'localStorage', {
    configurable: true,
    value: localStorageFake
  })
})

afterAll(() => {
  if (localStorageDescriptor) {
    Object.defineProperty(window, 'localStorage', localStorageDescriptor)
  } else {
    Reflect.deleteProperty(window, 'localStorage')
  }
})

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
    localStorageFake.values.clear()
    localStorageFake.events.length = 0
    localStorageFake.failWrites = false
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

  it('isolates the same stored session id across profiles', () => {
    const defaultKey = profileSessionKey('default', 'shared')
    const workKey = profileSessionKey('work', 'shared')

    enqueueQueuedPrompt(defaultKey, { attachments: [], text: 'default prompt' })
    enqueueQueuedPrompt(defaultKey, { attachments: [], text: 'default remainder' })
    enqueueQueuedPrompt(workKey, { attachments: [], text: 'work prompt' })

    expect(dequeueQueuedPrompt(defaultKey)?.text).toBe('default prompt')
    expect(getQueuedPrompts(workKey).map(entry => entry.text)).toEqual(['work prompt'])

    clearQueuedPrompts(defaultKey)
    expect(getQueuedPrompts(defaultKey)).toEqual([])
    expect(getQueuedPrompts(workKey).map(entry => entry.text)).toEqual(['work prompt'])
  })

  it('does not collide when profiles and session ids contain delimiters or JSON punctuation', () => {
    const left = profileSessionKey('pro:file["x"]', 'sess:1')
    const right = profileSessionKey('pro', 'file["x"]:sess:1')

    enqueueQueuedPrompt(left, { attachments: [], text: 'left' })
    enqueueQueuedPrompt(right, { attachments: [], text: 'right' })

    expect(left).not.toBe(right)
    expect(getQueuedPrompts(left).map(entry => entry.text)).toEqual(['left'])
    expect(getQueuedPrompts(right).map(entry => entry.text)).toEqual(['right'])
  })

  it('removes stale v1 after a successful ordinary v2 save', () => {
    window.localStorage.setItem(LEGACY_QUEUE_STORAGE_KEY, JSON.stringify({ legacy: [] }))

    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'new state' })

    expect(window.localStorage.getItem(QUEUE_STORAGE_KEY)).toBeTruthy()
    expect(window.localStorage.getItem(LEGACY_QUEUE_STORAGE_KEY)).toBeNull()
  })
})

describe('composer queue persistence migration', () => {
  const legacyEntry = {
    id: 'legacy-1',
    text: 'migrate me',
    attachments: [attachment('legacy-file')],
    queuedAt: 123
  }

  it('migrates v1 bare session ids to default-profile canonical keys before removing v1', () => {
    const storage = new StorageFake()
    storage.values.set(LEGACY_QUEUE_STORAGE_KEY, JSON.stringify({ 'legacy:session': [legacyEntry] }))

    const state = loadQueueState(storage)
    const canonicalKey = profileSessionKey('default', 'legacy:session')

    expect(state[canonicalKey]).toEqual([legacyEntry])
    expect(storage.values.has(QUEUE_STORAGE_KEY)).toBe(true)
    expect(storage.values.has(LEGACY_QUEUE_STORAGE_KEY)).toBe(false)
    expect(storage.events.indexOf(`set:${QUEUE_STORAGE_KEY}`)).toBeLessThan(
      storage.events.indexOf(`remove:${LEGACY_QUEUE_STORAGE_KEY}`)
    )
  })

  it('keeps v1 when the v2 migration write fails while returning usable in-memory state', () => {
    const storage = new StorageFake()
    storage.values.set(LEGACY_QUEUE_STORAGE_KEY, JSON.stringify({ legacy: [legacyEntry] }))
    storage.failWrites = true

    const state = loadQueueState(storage)

    expect(state[profileSessionKey('default', 'legacy')]).toEqual([legacyEntry])
    $queuedPromptsBySession.set(state)
    expect(getQueuedPrompts(profileSessionKey('default', 'legacy')).map(entry => entry.text)).toEqual(['migrate me'])
    expect(storage.values.has(LEGACY_QUEUE_STORAGE_KEY)).toBe(true)
    expect(storage.events).not.toContain(`remove:${LEGACY_QUEUE_STORAGE_KEY}`)
  })

  it('uses valid v2 without merging duplicate v1 data', () => {
    const storage = new StorageFake()
    const v2Key = profileSessionKey('work', 'stored')
    const v2Entry = { ...legacyEntry, id: 'v2', text: 'v2 wins' }
    storage.values.set(QUEUE_STORAGE_KEY, JSON.stringify({ [v2Key]: [v2Entry] }))
    storage.values.set(LEGACY_QUEUE_STORAGE_KEY, JSON.stringify({ stored: [legacyEntry] }))

    expect(loadQueueState(storage)).toEqual({ [v2Key]: [v2Entry] })
    expect(storage.values.has(LEGACY_QUEUE_STORAGE_KEY)).toBe(true)
  })

  it('ignores malformed v2 keys, buckets, and entries while preserving valid siblings', () => {
    const storage = new StorageFake()
    const firstKey = profileSessionKey('default', 'first')
    const secondKey = profileSessionKey('work', 'second')
    const secondEntry = { ...legacyEntry, id: 'second', text: 'still valid' }
    storage.values.set(
      QUEUE_STORAGE_KEY,
      JSON.stringify({
        [firstKey]: [legacyEntry, { id: 'broken', text: 42 }],
        [secondKey]: [secondEntry],
        'default:first': [legacyEntry],
        [JSON.stringify([' work ', 'noncanonical'])]: [legacyEntry],
        [profileSessionKey('default', 'bad-bucket')]: { nope: true }
      })
    )

    expect(loadQueueState(storage)).toEqual({
      [firstKey]: [legacyEntry],
      [secondKey]: [secondEntry]
    })
  })

  it('falls back to v1 when v2 has no usable canonical buckets', () => {
    const storage = new StorageFake()
    storage.values.set(QUEUE_STORAGE_KEY, JSON.stringify({ malformed: [{ nope: true }] }))
    storage.values.set(LEGACY_QUEUE_STORAGE_KEY, JSON.stringify({ legacy: [legacyEntry] }))

    expect(loadQueueState(storage)).toEqual({
      [profileSessionKey('default', 'legacy')]: [legacyEntry]
    })
  })
})

describe('migrateQueuedPrompts', () => {
  beforeEach(() => {
    localStorageFake.values.clear()
    localStorageFake.events.length = 0
    localStorageFake.failWrites = false
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
