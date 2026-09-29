import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $rightRailActiveTabId, selectRightRailTab } from '@/store/layout'
import {
  $previewTabs,
  closePreviewMatching,
  decodePreviewTabs,
  openPreview,
  restampPreviewOwnerForRotation,
  setPreviewStoredIdResolver
} from '@/store/preview'
import { setActiveSessionId } from '@/store/session'
import { $sessionTiles } from '@/store/session-states'

import { isLivePreviewTabOwnedBySession, registerPreviewPageReader } from './preview-reader'

/**
 * Session-scoped preview-action authorization (#95459, #95475 re-review).
 *
 * Four blockers, four groups:
 *  1. the gate is per-session and binds to the tab the mutation targets;
 *  2. routed URL opens allocate owner-compatible tabs (no owner/target swap);
 *  3. a captured tab id cannot authorize a close/reopen replacement;
 *  4. compression rotates the stored id, and durable ownership follows it.
 */

const cleanupFns: Array<() => void> = []

function urlTarget(url: string) {
  return { kind: 'url' as const, label: url, source: url, url }
}

function fileTarget(path: string) {
  return { kind: 'file' as const, label: path, path, previewKind: 'text' as const, source: path, url: `file://${path}` }
}

function registerReader(tabId: string, sessionId?: string, storedSessionId?: string) {
  const unregister = registerPreviewPageReader(tabId, async () => ({ text: 'page', title: 't', url: 'u' }), sessionId, storedSessionId)

  cleanupFns.push(unregister)

  return unregister
}

function browserTab(): string {
  const tab = $previewTabs.get().find(candidate => candidate.id.startsWith('url:'))

  expect(tab, 'a browser tab was opened').toBeTruthy()

  return tab!.id
}

describe('preview action ownership (session gate)', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $sessionTiles.set([])
    setPreviewStoredIdResolver(null)
  })

  afterEach(() => {
    for (const cleanup of cleanupFns.splice(0)) {
      cleanup()
    }

    closePreviewMatching('https://a.test')
    closePreviewMatching('https://b.test')
    closePreviewMatching('/tmp/shared.html')
    $sessionTiles.set([])
    setPreviewStoredIdResolver(null)
  })

  it('admits the owning session on its live tab and refuses a foreign one', () => {
    openPreview(urlTarget('https://a.test'), { ownerSessionId: 'rt-A', ownerStoredSessionId: 'stored-A' })
    const tabId = browserTab()
    registerReader(tabId, 'rt-A', 'stored-A')

    expect(isLivePreviewTabOwnedBySession(tabId, 'rt-A')).toBe(true)
    expect(isLivePreviewTabOwnedBySession(tabId, 'rt-B')).toBe(false)
  })

  it('is fail-closed: no session, no reader, or closed tab authorizes nothing', () => {
    openPreview(urlTarget('https://a.test'), { ownerSessionId: 'rt-A', ownerStoredSessionId: 'stored-A' })
    const tabId = browserTab()

    expect(isLivePreviewTabOwnedBySession(tabId, '')).toBe(false)

    registerReader(tabId, 'rt-A', 'stored-A')
    expect(isLivePreviewTabOwnedBySession(tabId, 'rt-A')).toBe(true)

    // Same tab id, NO live reader (pane unmounted): the target must not pass.
    cleanupFns.splice(0).forEach(cleanup => cleanup())
    expect(isLivePreviewTabOwnedBySession(tabId, 'rt-A')).toBe(false)
  })

  it('admits across a restart through the durable owner with a fresh runtime id', () => {
    // Restore path: the persisted tab keeps only the durable stamp (see the
    // decode test below), then the same conversation comes back with a NEW
    // runtime id and no fresh openPreview.
    $previewTabs.set([
      { id: 'url:browser-restored', target: urlTarget('https://a.test'), ownerStoredSessionId: 'stored-A' }
    ])

    $sessionTiles.set([{ runtimeId: 'rt-A2', storedSessionId: 'stored-A' }])
    setPreviewStoredIdResolver(runtimeId => $sessionTiles.get().find(tile => tile.runtimeId === runtimeId)?.storedSessionId ?? null)

    registerReader('url:browser-restored', 'rt-A2', 'stored-A')

    expect(isLivePreviewTabOwnedBySession('url:browser-restored', 'rt-A2')).toBe(true)
    expect(isLivePreviewTabOwnedBySession('url:browser-restored', 'rt-B')).toBe(false)
  })

  it('admits a later-active second tab owned by the same session (no insertion-order dependence)', () => {
    // Same-session URL opens deliberately REUSE the owner's browser tab (the
    // user's single vessel), so drive a FILE open to get a second tab.
    openPreview(fileTarget('/tmp/first.html'), { ownerSessionId: 'rt-A', ownerStoredSessionId: 'stored-A' })
    const first = $previewTabs.get().find(tab => tab.id.startsWith('file:'))!.id
    openPreview(fileTarget('/tmp/second.html'), { ownerSessionId: 'rt-A', ownerStoredSessionId: 'stored-A' })
    const second = $previewTabs.get().find(tab => tab.id.startsWith('file:') && tab.id !== first)!.id

    registerReader(first, 'rt-A', 'stored-A')
    registerReader(second, 'rt-A', 'stored-A')
    selectRightRailTab(second)

    expect($rightRailActiveTabId.get()).toBe(second)
    // The later-registered, now-active tab admits — not just the first one in
    // the map (that was the original insertion-order bug).
    expect(isLivePreviewTabOwnedBySession(second, 'rt-A')).toBe(true)
    expect(isLivePreviewTabOwnedBySession(first, 'rt-A')).toBe(true)
    expect(isLivePreviewTabOwnedBySession(second, 'rt-B')).toBe(false)
  })
})

describe('routed URL opens allocate owner-compatible tabs', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $sessionTiles.set([])
    setPreviewStoredIdResolver(null)
  })

  afterEach(() => {
    for (const cleanup of cleanupFns.splice(0)) {
      cleanup()
    }

    closePreviewMatching('https://a.test')
    closePreviewMatching('https://b.test')
    closePreviewMatching('/tmp/shared.html')
    $sessionTiles.set([])
    setPreviewStoredIdResolver(null)
  })

  it('does not steer another session’s browser tab: B gets its own vessel', () => {
    openPreview(urlTarget('https://a.test'), { ownerSessionId: 'rt-A', ownerStoredSessionId: 'stored-A' })
    const tabA = browserTab()

    // B is an on-screen tile while A is the primary chat: its routed open must
    // not replace A's target or inherit A's owner stamp.
    openPreview(urlTarget('https://b.test'), { ownerSessionId: 'rt-B', ownerStoredSessionId: 'stored-B' })
    const tabs = $previewTabs.get()
    const tabB = tabs.find(tab => tab.id.startsWith('url:') && tab.id !== tabA)

    expect(tabB).toBeTruthy()
    expect(tabA !== tabB!.id || tabs.find(tab => tab.id === tabA)?.target.url === 'https://a.test').toBe(true)
    expect(tabs.find(tab => tab.id === tabA)?.target.url).toBe('https://a.test')
    expect(tabs.find(tab => tab.id === tabA)?.ownerSessionId).toBe('rt-A')
    expect(tabB!.ownerSessionId).toBe('rt-B')
    expect(tabB!.target.url).toBe('https://b.test')
  })

  it('the focused session adopts an ownership-free tab; a background one does not', () => {
    // User's own tab, no owner at all. Adopting requires being the ACTIVE
    // session — a background tile must not navigate the user's personal tab.
    ;(window as unknown as { __hermesActiveSessionId?: string }).__hermesActiveSessionId = undefined
    openPreview(urlTarget('https://a.test'))
    const tabA = browserTab()

    // Not active here ($activeSessionId is null in this harness), so B mints
    // its own vessel instead of steering the user's tab.
    openPreview(urlTarget('https://b.test'), { ownerSessionId: 'rt-B', ownerStoredSessionId: 'stored-B' })
    const tabs = $previewTabs.get()

    expect(tabs.find(tab => tab.id === tabA)?.ownerSessionId).toBeUndefined()
    expect(tabs.filter(tab => tab.id.startsWith('url:'))).toHaveLength(2)

    // The focused owner DOES adopt the unowned tab: active session A opens
    // into the user's vessel and stamps it.
    setActiveSessionId('rt-A')
    openPreview(urlTarget('https://c.test'), { ownerSessionId: 'rt-A', ownerStoredSessionId: 'stored-A' })

    const after = $previewTabs.get()

    expect(after.find(tab => tab.id === tabA)?.ownerSessionId).toBe('rt-A')
    expect(after.find(tab => tab.id === tabA)?.target.url).toBe('https://c.test')
  })

  it('keeps one owner per tab after hydration: no runtime/stored two-principal union', () => {
    // Hydrated durable-A tab (runtime stamp dropped by decode).
    $previewTabs.set([{ id: 'url:browser-h', target: urlTarget('https://a.test'), ownerStoredSessionId: 'stored-A' }])

    // B performs a routed URL open on top of the same tab id path: with the
    // resolver it must NOT adopt A's stored stamp.
    $sessionTiles.set([{ runtimeId: 'rt-B', storedSessionId: 'stored-B' }])
    setPreviewStoredIdResolver(runtimeId => $sessionTiles.get().find(tile => tile.runtimeId === runtimeId)?.storedSessionId ?? null)

    openPreview(urlTarget('https://a.test'), { ownerSessionId: 'rt-B' })

    const tab = $previewTabs.get().find(tab => tab.id === 'url:browser-h')!

    expect(tab.ownerStoredSessionId !== 'stored-A' || tab.ownerSessionId === undefined).toBe(true)

    if (tab.ownerSessionId === 'rt-B') {
      expect(tab.ownerStoredSessionId).toBe('stored-B')
    }
  })
})

describe('captured tab id binds to the live incarnation', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  afterEach(() => {
    for (const cleanup of cleanupFns.splice(0)) {
      cleanup()
    }

    closePreviewMatching('/tmp/shared.html')
  })

  it('refuses a replacement tab registered under the same file id after close/reopen', () => {
    // Deterministic file ids: closing and reopening the same target reuses the
    // id, so a stale captured id can meet a DIFFERENT owner's replacement tab.
    openPreview(fileTarget('/tmp/shared.html'), { ownerSessionId: 'rt-A', ownerStoredSessionId: 'stored-A' })
    const tabId = $previewTabs.get().find(tab => tab.id.startsWith('file:'))!.id

    registerReader(tabId, 'rt-A', 'stored-A')
    expect(isLivePreviewTabOwnedBySession(tabId, 'rt-A')).toBe(true)

    // A's pane unmounts; B reopens the same target — same id, new incarnation.
    cleanupFns.splice(0).forEach(cleanup => cleanup())
    closePreviewMatching('/tmp/shared.html')
    openPreview(fileTarget('/tmp/shared.html'), { ownerSessionId: 'rt-B', ownerStoredSessionId: 'stored-B' })
    registerReader(tabId, 'rt-B', 'stored-B')

    // A's captured identity must not authorize B's replacement tab...
    expect(isLivePreviewTabOwnedBySession(tabId, 'rt-A')).toBe(false)
    // ...while B's own admission still works.
    expect(isLivePreviewTabOwnedBySession(tabId, 'rt-B')).toBe(true)
  })
})

describe('durable ownership survives persistence and compression rotation', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $sessionTiles.set([])
    setPreviewStoredIdResolver(null)
  })

  afterEach(() => {
    for (const cleanup of cleanupFns.splice(0)) {
      cleanup()
    }

    closePreviewMatching('https://a.test')
    closePreviewMatching('/tmp/shared.html')
    $sessionTiles.set([])
    setPreviewStoredIdResolver(null)
  })

  it('drops the runtime stamp at decode but keeps the durable one', () => {
    const raw = JSON.stringify([
      { id: 'url:browser-x', ownerSessionId: 'rt-dead', ownerStoredSessionId: 'stored-A', target: urlTarget('https://a.test') }
    ])

    const [tab] = decodePreviewTabs(raw)

    expect(tab.ownerSessionId).toBeUndefined()
    expect(tab.ownerStoredSessionId).toBe('stored-A')
  })

  it('follows the stored-id rotation at compression, then admits the resumed runtime', () => {
    openPreview(urlTarget('https://a.test'), { ownerSessionId: 'rt-A', ownerStoredSessionId: 'S0' })
    const tabId = browserTab()
    registerReader(tabId, 'rt-A', 'S0')

    // Same live runtime compresses: the stored id rotates S0 → S1.
    restampPreviewOwnerForRotation('S0', 'S1', 'rt-A')

    expect($previewTabs.get().find(tab => tab.id === tabId)?.ownerStoredSessionId).toBe('S1')
    expect($previewTabs.get().find(tab => tab.id === tabId)?.ownerSessionId).toBe('rt-A')

    // Restart: runtime stamp dropped, R2 resumes against S1.
    $previewTabs.set($previewTabs.get().map(tab => (tab.id === tabId ? { ...tab, ownerSessionId: undefined } : tab)))
    $sessionTiles.set([{ runtimeId: 'rt-A2', storedSessionId: 'S1' }])
    setPreviewStoredIdResolver(runtimeId => $sessionTiles.get().find(tile => tile.runtimeId === runtimeId)?.storedSessionId ?? null)

    registerReader(tabId, 'rt-A2', 'S1')

    expect(isLivePreviewTabOwnedBySession(tabId, 'rt-A2')).toBe(true)
    expect(isLivePreviewTabOwnedBySession(tabId, 'rt-B')).toBe(false)
  })

  it('never touches tabs owned by other conversations during rotation', () => {
    openPreview(urlTarget('https://a.test'), { ownerSessionId: 'rt-A', ownerStoredSessionId: 'S0' })
    openPreview(fileTarget('/tmp/shared.html'), { ownerSessionId: 'rt-Z', ownerStoredSessionId: 'S9' })

    restampPreviewOwnerForRotation('S0', 'S1', 'rt-A')

    const tabs = $previewTabs.get()

    expect(tabs.find(tab => tab.id.startsWith('url:'))?.ownerStoredSessionId).toBe('S1')
    expect(tabs.find(tab => tab.id.startsWith('file:'))?.ownerStoredSessionId).toBe('S9')
  })
})
