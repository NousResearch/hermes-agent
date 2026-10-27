import { afterEach, expect, it, vi } from 'vitest'

import type * as ComposerStore from './composer'
import { clearSessionDraft, stashSessionDraft, takeSessionDraft } from './composer'

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

afterEach(() => {
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
