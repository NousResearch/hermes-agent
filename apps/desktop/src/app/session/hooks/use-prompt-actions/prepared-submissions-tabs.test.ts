import { afterEach, expect, test, vi } from 'vitest'

import { type PreparedSubmission, preparedSubmissionKey } from './prepared-submissions'
import type * as PreparedSubmissions from './prepared-submissions'
import { storageView } from './prepared-submissions-test-utils'
import type { SubmissionDestination } from './submission-destination'

type Journal = typeof PreparedSubmissions

const realStorage = window.localStorage
const installStorage = (storage: Storage) => Object.defineProperty(window, 'localStorage', { configurable: true, value: storage })

afterEach(() => { installStorage(realStorage); vi.resetModules(); localStorage.clear() })

// The single blob an earlier build kept every browser journal entry in.
const LEGACY_KEY = 'hermes.desktop.preparedSubmissions.v1'
const owner = { connectionId: 'local', profile: 'default' }
const destination = { scopeKey: 'webapp::default' } as SubmissionDestination
const intent = (text: string) => preparedSubmissionKey('stored', destination, text, [])
const entry = (id: string, text: string): PreparedSubmission => ({ id, text, attachments: [], params: { submission_id: id }, owner })
const legacySend = (text: string) => ({ key: intent(text), entry: entry(`${text} id`, text) })

// One browser tab: its own module instance over its own view of the origin's storage, which the
// other tabs see only once `flush` publishes its writes.
async function openTab(shared: Map<string, string>) {
  vi.resetModules()
  const journal: Journal = await import('./prepared-submissions')
  const view = storageView(shared)

  const run = <T>(step: (tab: Journal) => Promise<T>) => {
    installStorage(view.storage as unknown as Storage)

    return step(journal)
  }

  return { run, flush: view.flush }
}

// An explicit retry of a retained send: adopt it, journal it again, retire it once acknowledged.
async function retry(journal: Journal, send: ReturnType<typeof legacySend>, acknowledged: boolean) {
  const adopted = await journal.adoptPreparedSubmission(send.key, send.entry.id)
  await journal.writePreparedSubmission(adopted!.key, adopted!.entry)

  if (acknowledged) { await journal.removePreparedSubmission(adopted!.key, adopted!.entry.id) }
}

test('a tab acting on a stale read of the journal keeps the entry another tab just wrote', async () => {
  const shared = new Map<string, string>()
  const read = async (key: string) => (await openTab(shared)).run(journal => journal.readPreparedSubmission(key))
  // Tab A read the journal before B's write reached it.
  const a = await openTab(shared)
  const b = await openTab(shared)
  await b.run(journal => journal.writePreparedSubmission(intent('from B'), entry('b', 'from B')))
  b.flush()

  await a.run(journal => journal.writePreparedSubmission(intent('from A'), entry('a', 'from A')))
  a.flush()
  expect(await read(intent('from B'))).toEqual(entry('b', 'from B'))
  expect(await read(intent('from A'))).toEqual(entry('a', 'from A'))

  // A's send is acknowledged: retiring it from the same stale view leaves B's entry alone.
  await a.run(journal => journal.removePreparedSubmission(intent('from A'), 'a'))
  a.flush()
  expect(await read(intent('from B'))).toEqual(entry('b', 'from B'))
  expect(await read(intent('from A'))).toBeUndefined()
})

test('an entry an earlier build left in the blob is still offered and adopted, and stays retired once acknowledged', async () => {
  const image = { id: '/cache/a.png', occurrenceId: 'image-occurrence', kind: 'image', label: 'a.png' } as const
  const key = preparedSubmissionKey('stored', destination, 'caption', [image])
  const legacy = { ...entry('legacy', 'caption'), attachments: [image] }
  localStorage.setItem(LEGACY_KEY, JSON.stringify({ [key]: legacy }))
  const journal: Journal = await import('./prepared-submissions')

  expect(await journal.listPreparedImageDrafts('stored', destination.scopeKey)).toMatchObject([{ key, id: 'legacy' }])
  await retry(journal, { key, entry: legacy }, true)

  vi.resetModules()
  const reloaded: Journal = await import('./prepared-submissions')
  expect(await reloaded.listPreparedImageDrafts('stored', destination.scopeKey)).toEqual([])
  expect(await reloaded.adoptPreparedSubmission(key, 'legacy')).toBeUndefined()
})

test('two tabs upgrading at once: each legacy entry they retire stays retired, every other stays adoptable', async () => {
  const [untouched, ackedInA, ackedInB, unconfirmedInB] = ['untouched', 'acked in A', 'acked in B', 'unconfirmed in B'].map(legacySend)
  const legacy = Object.fromEntries([untouched, ackedInA, ackedInB, unconfirmedInB].map(send => [send.key, send.entry]))
  const shared = new Map([[LEGACY_KEY, JSON.stringify(legacy)]])
  // Both tabs load the new build over the same blob; neither sees the other's writes until it flushes.
  const a = await openTab(shared)
  const b = await openTab(shared)
  await a.run(journal => retry(journal, ackedInA, true))
  await b.run(async journal => { await retry(journal, ackedInB, true); await retry(journal, unconfirmedInB, false) })
  a.flush()
  b.flush()

  const reloaded = await openTab(shared)
  const adopt = (send: ReturnType<typeof legacySend>) => reloaded.run(journal => journal.adoptPreparedSubmission(send.key, send.entry.id))
  expect(await adopt(ackedInA)).toBeUndefined()
  expect(await adopt(ackedInB)).toBeUndefined()
  expect(await adopt(untouched)).toEqual(untouched)
  expect(await adopt(unconfirmedInB)).toEqual(unconfirmedInB)
})

test('a tab of the earlier build still open keeps a newer entry it writes under a retired legacy key adoptable', async () => {
  const key = intent('same words')
  localStorage.setItem(LEGACY_KEY, JSON.stringify({ [key]: entry('first', 'same words') }))
  const journal: Journal = await import('./prepared-submissions')
  await journal.removePreparedSubmission(key, 'first')
  expect(await journal.adoptPreparedSubmission(key, 'first')).toBeUndefined()

  // The earlier build retires 'first' in its blob too, then journals a new send of the same words.
  localStorage.setItem(LEGACY_KEY, JSON.stringify({ [key]: entry('second', 'same words') }))
  expect(await journal.adoptPreparedSubmission(key, 'second')).toEqual({ key, entry: entry('second', 'same words') })
})

test('retiring an entry never hides a different send the earlier build left under the same key', async () => {
  const key = intent('same words')
  const journal: Journal = await import('./prepared-submissions')
  await journal.writePreparedSubmission(key, entry('new', 'same words'))

  // A tab of the earlier build, which cannot see the new entry, journals its own send of the same words.
  localStorage.setItem(LEGACY_KEY, JSON.stringify({ [key]: entry('earlier', 'same words') }))
  await journal.removePreparedSubmission(key, 'new')
  expect(await journal.adoptPreparedSubmission(key, 'earlier')).toEqual({ key, entry: entry('earlier', 'same words') })
})

const ENTRY = 'hermes.desktop.preparedSubmissions.v2:'
const SPENT = 'hermes.desktop.preparedSubmissions.spent.v2:'

// Storage over `stored` that runs `onKey` on every key() call and `onGet` on every getItem(): another
// tab's writes landing while a journal read is part way through.
function interleavedStorage(stored: Map<string, string>, onKey: () => void, onGet: (key: string) => void) {
  return {
    get length() { return stored.size },
    key: (index: number) => {
      const key = [...stored.keys()][index] ?? null
      onKey()

      return key
    },
    getItem: (key: string) => {
      const value = stored.get(key) ?? null
      onGet(key)

      return value
    },
    setItem: (key: string, value: string) => { stored.set(key, value) },
    removeItem: (key: string) => { stored.delete(key) },
    clear: () => stored.clear()
  } as unknown as Storage
}

async function readWhile(stored: Map<string, string>, { onKey = () => undefined, onGet = () => undefined }: { onKey?: () => void; onGet?: (key: string) => void }) {
  const journal: Journal = await import('./prepared-submissions')
  installStorage(interleavedStorage(stored, onKey, onGet))

  return journal.readPreparedSubmission('a')
}

test('an entry read while another tab retires it is hidden even when its key already held an older tombstone', async () => {
  const stored = new Map([[SPENT + 'a', 'older'], [ENTRY + 'a', JSON.stringify(entry('a', 'a'))]])

  // The other tab's retirement tombstones 'a' under the same key just as this read takes the entry.
  const onGet = (key: string) => { if (key === ENTRY + 'a') { stored.set(SPENT + 'a', 'a') } }

  expect(await readWhile(stored, { onGet })).toBeUndefined()
  expect([...stored.keys()]).toEqual([])
})

test('an entry read while another tab retires it is hidden when its tombstone key appears after the key list', async () => {
  // This time the key held no tombstone at all when the read listed the keys.
  const stored = new Map([[ENTRY + 'a', JSON.stringify(entry('a', 'a'))]])
  const onGet = (key: string) => { if (key === ENTRY + 'a') { stored.set(SPENT + 'a', 'a') } }

  expect(await readWhile(stored, { onGet })).toBeUndefined()
  expect([...stored.keys()]).toEqual([])
})

test('a tombstone a concurrent removal shifts out of the key list still hides its entry', async () => {
  const stored = new Map([['unrelated', 'x'], [SPENT + 'a', 'a'], [ENTRY + 'a', JSON.stringify(entry('a', 'a'))]])

  // Another tab removes the first key just after it is listed: every later key moves down one index.
  const onKey = () => { stored.delete('unrelated') }

  expect(await readWhile(stored, { onKey })).toBeUndefined()
  expect([...stored.keys()]).toEqual([])
})
