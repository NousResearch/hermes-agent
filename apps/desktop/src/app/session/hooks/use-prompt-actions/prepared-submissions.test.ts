import { afterEach, expect, test, vi } from 'vitest'

import { listPreparedImageDrafts, type PreparedSubmission, preparedSubmissionKey, readPreparedSubmission, removePreparedSubmission, writePreparedSubmission } from './prepared-submissions'
import type { SubmissionDestination } from './submission-destination'

afterEach(() => { vi.unstubAllGlobals(); localStorage.clear() })

test('image recovery never offers expanded slash intent as an ordinary draft and leaves its exact journal intact', async () => {
  vi.stubGlobal('hermesDesktop', undefined)
  const destination = { scopeKey: 'local::default' } as SubmissionDestination
  const attachments: PreparedSubmission['attachments'] = [{ id: '/cache/a.png', occurrenceId: 'image-occurrence', kind: 'image', label: 'a.png' }]
  const ordinary = preparedSubmissionKey('stored', destination, 'caption', attachments)
  const slash = preparedSubmissionKey('stored', destination, 'expanded skill instructions', attachments, { retryText: '/skill task', submission_id: 'skill-id' })

  for (const [key, text] of [[ordinary, 'caption'], [slash, 'expanded skill instructions']]) {
    await writePreparedSubmission(key, { id: key, text, attachments, params: { submission_id: key }, owner: { connectionId: 'local', profile: 'default' } })
  }

  const before = localStorage.getItem('hermes.desktop.preparedSubmissions.v1')
  expect((await listPreparedImageDrafts('stored', destination.scopeKey)).map(entry => entry.key)).toEqual([ordinary])
  expect(await listPreparedImageDrafts('another', destination.scopeKey)).toEqual([])
  expect(localStorage.getItem('hermes.desktop.preparedSubmissions.v1')).toBe(before)
  expect((await readPreparedSubmission(slash))?.text).toBe('expanded skill instructions')
})

test('native preparation waits for acknowledgement and never downgrades a write failure to browser storage', async () => {
  let ack!: () => void
  const gate = new Promise<void>(resolve => { ack = resolve })
  const entry = { id: 'a', text: 'Ω\n  exact', attachments: [], params: { session_id: 'live' } } as unknown as PreparedSubmission
  const native = { read: vi.fn(async () => JSON.stringify({ key: entry })), update: vi.fn(() => gate) }
  vi.stubGlobal('hermesDesktop', { preparedSubmissions: native })
  let finished = false
  const writing = writePreparedSubmission('key', entry).then(() => { finished = true })
  await Promise.resolve()
  expect(finished).toBe(false)
  expect(native.update).toHaveBeenCalledWith('key', JSON.stringify(entry))
  ack(); await writing
  expect(await readPreparedSubmission('key')).toEqual(entry)
  native.update.mockRejectedValueOnce(new Error('disk full'))
  await expect(writePreparedSubmission('key', entry)).rejects.toThrow('disk full')
  expect(localStorage.length).toBe(0)
  await removePreparedSubmission('key', 'a')
  expect(native.update).toHaveBeenLastCalledWith('key', null)
  vi.stubGlobal('hermesDesktop', undefined)
  await writePreparedSubmission('key', entry)
  expect(await readPreparedSubmission('key')).toEqual(entry)
  await removePreparedSubmission('key', 'a')
  expect(await readPreparedSubmission('key')).toBeUndefined()
})

function sharedStorage(): Storage {
  const rows = new Map<string, string>()

  return {
    get length() { return rows.size },
    clear() { rows.clear() },
    getItem(key: string) {
      const value = rows.get(key)

      return value === undefined ? null : value
    },
    key(index: number) { return [...rows.keys()][index] ?? null },
    removeItem(key: string) { rows.delete(key) },
    setItem(key: string, value: string) { rows.set(key, String(value)) }
  }
}

function installSharedStorage(): () => void {
  const real = window.localStorage

  Object.defineProperty(window, 'localStorage', { configurable: true, value: sharedStorage() })

  return () => {
    Object.defineProperty(window, 'localStorage', { configurable: true, value: real })
  }
}

function hideWebLocks(): void {
  Object.defineProperty(navigator, 'locks', { configurable: true, value: undefined })
}

const destination = { scopeKey: 'local::default' } as SubmissionDestination
const owner = { connectionId: 'local', profile: 'default' }

async function twoWindows() {
  vi.resetModules()
  const windowA = await import('./prepared-submissions')
  vi.resetModules()
  const windowB = await import('./prepared-submissions')

  return { windowA, windowB }
}

test('two plain-HTTP windows do not adopt one prepared submission', async () => {
  hideWebLocks()
  vi.stubGlobal('hermesDesktop', undefined)
  const restoreStorage = installSharedStorage()
  const intent = preparedSubmissionKey('stored', destination, 'same text', [])

  const entry: PreparedSubmission = {
    id: 'sub-1',
    text: 'same text',
    attachments: [],
    params: { submission_id: 'sub-1' },
    owner
  }

  try {
    const { windowA, windowB } = await twoWindows()

    await windowA.writePreparedSubmission(intent, entry)
    const held = await windowA.adoptPreparedSubmission(intent, 'sub-1')
    expect(held?.entry.id).toBe('sub-1')
    expect(await windowB.adoptPreparedSubmission(intent, 'sub-1')).toBeUndefined()

    const [racedA, racedB] = await Promise.all([
      windowA.adoptPreparedSubmission(intent, 'sub-1'),
      windowB.adoptPreparedSubmission(intent, 'sub-1')
    ])

    // The holder already owns it, so a racing adopt from the other window still loses.
    expect(racedA?.entry.id).toBe('sub-1')
    expect(racedB).toBeUndefined()
    await windowA.removePreparedSubmission(intent, 'sub-1')
  } finally {
    Reflect.deleteProperty(navigator, 'locks')
    restoreStorage()
  }
})

test('a closed plain-HTTP window leaves its prepared submission adoptable', async () => {
  hideWebLocks()
  vi.stubGlobal('hermesDesktop', undefined)
  const restoreStorage = installSharedStorage()
  const intent = preparedSubmissionKey('stored', destination, 'same text', [])

  const entry: PreparedSubmission = {
    id: 'sub-1',
    text: 'same text',
    attachments: [],
    params: { submission_id: 'sub-1' },
    owner
  }

  try {
    const { windowA, windowB } = await twoWindows()

    await windowA.writePreparedSubmission(intent, entry)
    expect((await windowA.adoptPreparedSubmission(intent, 'sub-1'))?.entry.id).toBe('sub-1')
    expect(await windowB.adoptPreparedSubmission(intent, 'sub-1')).toBeUndefined()
    window.dispatchEvent(new Event('pagehide'))
    const adopted = await windowB.adoptPreparedSubmission(intent, 'sub-1')
    expect(adopted?.entry.id).toBe('sub-1')
    await windowB.removePreparedSubmission(adopted!.key, adopted!.entry.id)
  } finally {
    Reflect.deleteProperty(navigator, 'locks')
    restoreStorage()
  }
})

test('two plain-HTTP windows do not share one slash retry slot', async () => {
  hideWebLocks()
  vi.stubGlobal('hermesDesktop', undefined)
  const restoreStorage = installSharedStorage()
  const intent = preparedSubmissionKey('stored', destination, '/skill task', [], { retryText: '/skill task' })

  try {
    const { windowA, windowB } = await twoWindows()

    const [slotA, slotB] = await Promise.all([
      windowA.preparedSubmissionSlot(intent),
      windowB.preparedSubmissionSlot(intent)
    ])

    expect(slotA).not.toBe(slotB)
    await windowA.removePreparedSubmission(slotA, 'release-a')
    await windowB.removePreparedSubmission(slotB, 'release-b')
  } finally {
    Reflect.deleteProperty(navigator, 'locks')
    restoreStorage()
  }
})

test('Web Locks still exclude a live window from adopting the same prepared submission', async () => {
  vi.stubGlobal('hermesDesktop', undefined)
  const held = new Set<string>()

  const request = vi.fn((name: string, options: { ifAvailable?: boolean }, callback: (lock: { name: string } | null) => unknown) => {
    if (options?.ifAvailable && held.has(name)) { return callback(null) }

    held.add(name)

    return Promise.resolve(callback({ name })).finally(() => { held.delete(name) })
  })

  Object.defineProperty(navigator, 'locks', { configurable: true, value: { request } })
  const intent = preparedSubmissionKey('stored', destination, 'same text', [])

  const entry: PreparedSubmission = {
    id: 'sub-1',
    text: 'same text',
    attachments: [],
    params: { submission_id: 'sub-1' },
    owner
  }

  try {
    const { windowA, windowB } = await twoWindows()

    await windowA.writePreparedSubmission(intent, entry)
    expect((await windowA.adoptPreparedSubmission(intent, 'sub-1'))?.entry.id).toBe('sub-1')
    expect(await windowB.adoptPreparedSubmission(intent, 'sub-1')).toBeUndefined()
    expect(request).toHaveBeenCalled()
    expect(Array.from({ length: localStorage.length }, (_, index) => localStorage.key(index)).some(key => key?.startsWith('hermes.desktop.tabLease.v1:'))).toBe(false)
    await windowA.removePreparedSubmission(intent, 'sub-1')
  } finally {
    Reflect.deleteProperty(navigator, 'locks')
  }
})
