import { afterEach, expect, test, vi } from 'vitest'

afterEach(() => { vi.unstubAllGlobals(); vi.resetModules(); localStorage.clear() })

// The native file journal of one origin, shared by every renderer load. Removal fails like
// ENOSPC/EIO while `failing` is set; reads keep working.
function nativeJournal() {
  let file = '{}'
  const state = { failing: false }

  const native = {
    read: async () => file,
    update: async (key: string, entry: string | null) => {
      if (state.failing) { throw new Error('ENOSPC: no space left on device') }
      const journal = JSON.parse(file)

      if (entry === null) { delete journal[key] } else { journal[key] = JSON.parse(entry) }
      file = JSON.stringify(journal)
    }
  }

  return { native, state, file: () => JSON.parse(file) }
}

test('an admitted identity whose journal removal failed stays spent across a reload', async () => {
  const journal = nativeJournal()
  vi.stubGlobal('hermesDesktop', { preparedSubmissions: journal.native })
  const intent = JSON.stringify(['local::default', 'stored', 'same text', [], null, false, null])
  const entry = { id: 'admitted-id', text: 'same text', attachments: [], params: { submission_id: 'admitted-id' }, owner: { connectionId: 'local', profile: 'default' } }

  const first = await import('./prepared-submissions')
  await first.writePreparedSubmission(intent, entry)
  journal.state.failing = true
  await expect(first.removePreparedSubmission(intent, 'admitted-id')).rejects.toThrow('ENOSPC')
  expect(journal.file()[intent]?.id).toBe('admitted-id')

  // Renderer reload: module memory is gone, the file entry is not.
  vi.resetModules()
  const reloaded = await import('./prepared-submissions')
  expect(await reloaded.adoptPreparedSubmission(intent, 'admitted-id')).toBeUndefined()
  expect(await reloaded.readPreparedSubmission(intent)).toBeUndefined()

  // Storage recovers: the stale file entry is retired instead of lingering.
  journal.state.failing = false
  await reloaded.listPreparedImageDrafts('stored', 'local::default')
  expect(journal.file()[intent]).toBeUndefined()
})

// Chromium gives each renderer its own cached copy of the origin's localStorage and propagates
// writes asynchronously, last writer wins per key: each window writes against its own view.
function storageView(shared: Map<string, string>) {
  const local = new Map(shared)
  const written = new Map<string, null | string>()

  const storage = {
    get length() { return local.size },
    key: (index: number) => [...local.keys()][index] ?? null,
    getItem: (key: string) => local.get(key) ?? null,
    setItem: (key: string, value: string) => { local.set(key, value); written.set(key, value) },
    removeItem: (key: string) => { local.delete(key); written.set(key, null) },
    clear: () => undefined
  }

  const flush = () => { for (const [key, value] of written) { if (value === null) { shared.delete(key) } else { shared.set(key, value) } } }

  return { storage, flush }
}

test('two windows retiring different admitted entries keep both tombstones across a reload', async () => {
  const realStorage = window.localStorage
  const inWindow = (storage: Storage) => Object.defineProperty(window, 'localStorage', { configurable: true, value: storage })
  const journal = nativeJournal()
  vi.stubGlobal('hermesDesktop', { preparedSubmissions: journal.native })
  const owner = { connectionId: 'local', profile: 'default' }
  const intent = (text: string) => JSON.stringify(['local::default', 'stored', text, [], null, false, null])
  const entry = (id: string, text: string) => ({ id, text, attachments: [], params: { submission_id: id }, owner })

  try {
    const shared = new Map<string, string>()
    const a = storageView(shared)
    const b = storageView(shared)
    const windowA = await import('./prepared-submissions')
    vi.resetModules()
    const windowB = await import('./prepared-submissions')

    inWindow(a.storage as unknown as Storage)
    await windowA.writePreparedSubmission(intent('from A'), entry('admitted-a', 'from A'))
    inWindow(b.storage as unknown as Storage)
    await windowB.writePreparedSubmission(intent('from B'), entry('admitted-b', 'from B'))

    // Both admissions are ACKed while the native journal cannot delete either entry.
    journal.state.failing = true
    inWindow(a.storage as unknown as Storage)
    await expect(windowA.removePreparedSubmission(intent('from A'), 'admitted-a')).rejects.toThrow('ENOSPC')
    inWindow(b.storage as unknown as Storage)
    await expect(windowB.removePreparedSubmission(intent('from B'), 'admitted-b')).rejects.toThrow('ENOSPC')
    a.flush()
    b.flush()

    // A fresh window over the converged storage adopts neither admitted input again.
    inWindow(storageView(shared).storage as unknown as Storage)
    vi.resetModules()
    const reloaded = await import('./prepared-submissions')
    expect(await reloaded.adoptPreparedSubmission(intent('from A'), 'admitted-a')).toBeUndefined()
    expect(await reloaded.adoptPreparedSubmission(intent('from B'), 'admitted-b')).toBeUndefined()
  } finally {
    inWindow(realStorage)
  }
})
