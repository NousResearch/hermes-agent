import { afterEach, expect, it, vi } from 'vitest'

import type * as ComposerStore from './composer'
import { clearSessionDraft, SESSION_DRAFTS_STORAGE_KEY, stashSessionDraft, takeSessionDraft } from './composer'
import { DRAFT_STASH_STORAGE_KEY } from './composer-draft-stash'

// Two windows share localStorage, but each reads its own copy and gets the
// other's writes milliseconds later. A second module instance is the other
// window; restoring the storage it loaded with before it writes makes its
// write come from that stale copy, exactly as a write landing in that gap does.
const ALPHA = 'stash-race-alpha'
const GAMMA = 'stash-race-gamma'

const storageKeys = () =>
  Array.from({ length: window.localStorage.length }, (_, index) => window.localStorage.key(index)!)

function snapshotStorage(): Record<string, string> {
  return Object.fromEntries(storageKeys().map(key => [key, window.localStorage.getItem(key)!]))
}

function restoreStorage(snapshot: Record<string, string>) {
  window.localStorage.clear()

  for (const [key, value] of Object.entries(snapshot)) {
    window.localStorage.setItem(key, value)
  }
}

// The other window writes from `stale`; then this window hears about every key it changed.
async function otherWindowWritesFrom(stale: Record<string, string>, write: (other: typeof ComposerStore) => void) {
  const before = snapshotStorage()
  restoreStorage(stale)
  vi.resetModules()
  const other = await import('./composer')
  write(other)

  for (const key of new Set([...Object.keys(before), ...storageKeys()])) {
    if (before[key] !== window.localStorage.getItem(key)) {
      window.dispatchEvent(new StorageEvent('storage', { key }))
    }
  }
}

async function reopen(): Promise<typeof ComposerStore> {
  vi.resetModules()

  return import('./composer')
}

afterEach(() => {
  vi.restoreAllMocks()
  clearSessionDraft(ALPHA)
  clearSessionDraft(GAMMA)
  window.localStorage.clear()
})

it("another window's write from a stale copy neither drops nor revives this window's draft", async () => {
  // Saved here while the other window's copy predates it: its write must not drop it.
  const beforeSave = snapshotStorage()
  stashSessionDraft(ALPHA, 'alpha draft', [])
  await otherWindowWritesFrom(beforeSave, other => other.stashSessionDraft(GAMMA, 'gamma draft', []))

  expect(takeSessionDraft(ALPHA).text).toBe('alpha draft')
  expect(takeSessionDraft(GAMMA).text).toBe('gamma draft')
  vi.resetModules()
  expect((await import('./composer')).takeSessionDraft(ALPHA).text).toBe('alpha draft')

  // Cleared (sent) here while the other window's copy still holds it: its write must not bring it back.
  const beforeClear = snapshotStorage()
  clearSessionDraft(ALPHA)
  await otherWindowWritesFrom(beforeClear, other => other.stashSessionDraft(GAMMA, 'gamma edited', []))

  expect(takeSessionDraft(ALPHA).text).toBe('')
  expect(takeSessionDraft(GAMMA).text).toBe('gamma edited')
  vi.resetModules()
  const reopened = await import('./composer')
  expect(reopened.takeSessionDraft(ALPHA).text).toBe('')
  expect(reopened.takeSessionDraft(GAMMA).text).toBe('gamma edited')
})

it('a draft saved after the system clock jumps back is not evicted by the cap on its own save', async () => {
  window.localStorage.clear()
  const start = 1_800_000_000_000
  let now = start
  vi.spyOn(Date, 'now').mockImplementation(() => now)
  let store = await reopen()

  for (let index = 0; index < 50; index++) {
    now = start + index * 1000
    store.stashSessionDraft(`old-${index}`, `old ${index}`, [])
  }

  now = start - 3_600_000
  store.stashSessionDraft('after-jump', 'typed after the jump', [])
  store = await reopen()

  expect(store.takeSessionDraft('after-jump').text).toBe('typed after the jump')
  expect(store.takeSessionDraft('old-0').text).toBe('')
})

it('after a downgrade and re-upgrade, what the older build edited, sent and typed sticks', async () => {
  // The older build reads and writes only the legacy list: it edited A, sent B and typed C.
  window.localStorage.clear()
  let store = await reopen()
  store.stashSessionDraft('A', 'a before', [])
  store.stashSessionDraft('B', 'b sent in the older build', [])
  window.localStorage.setItem(SESSION_DRAFTS_STORAGE_KEY, JSON.stringify({ A: 'a edited there', C: 'c typed there' }))

  for (let boot = 0; boot < 2; boot++) {
    store = await reopen()
    expect(['A', 'B', 'C'].map(key => store.takeSessionDraft(key).text)).toEqual([
      'a edited there',
      '',
      'c typed there'
    ])
  }
})

it("a boot between two windows' unfinished mirror writes does not revive a sent draft", async () => {
  // This window sends ALPHA while the other window, from a copy that still holds
  // it, saves GAMMA: the other's legacy list lands last, this window's stash last.
  window.localStorage.clear()
  stashSessionDraft(ALPHA, 'sent here', [])
  const beforeSend = snapshotStorage()
  clearSessionDraft(ALPHA)
  const afterSend = snapshotStorage()
  restoreStorage(beforeSend)
  vi.resetModules()
  ;(await import('./composer')).stashSessionDraft(GAMMA, 'gamma there', [])
  restoreStorage({ ...afterSend, ...snapshotStorage(), [DRAFT_STASH_STORAGE_KEY]: afterSend[DRAFT_STASH_STORAGE_KEY]! })

  expect((await reopen()).takeSessionDraft(ALPHA).text).toBe('')
})

it('an unreadable legacy list leaves the stash drafts alone', async () => {
  window.localStorage.clear()
  stashSessionDraft(ALPHA, 'kept', [])
  window.localStorage.setItem(SESSION_DRAFTS_STORAGE_KEY, '{not json')

  expect((await reopen()).takeSessionDraft(ALPHA).text).toBe('kept')
})
