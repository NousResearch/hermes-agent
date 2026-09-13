import { describe, expect, it, vi } from 'vitest'

import { $previewTabs, decodePreviewTabs, openPreview } from '@/store/preview'

import { isLivePreviewTabOwnedBySession, registerPreviewPageReader } from './preview-reader'

// The durable-ownership leg translates a runtime id to its stored id via the
// session-states map. Model the wiring layer's runtime→stored bindings this
// conversation has had across its restarts (R1 pre-restart, R2 post-restart)
// plus a foreign conversation. Tiles/mirror state is irrelevant here.
const storedByRuntime: Record<string, string> = {
  'runtime-R1': 'stored-conversation',
  'runtime-R2': 'stored-conversation',
  'runtime-mine': 'stored-mine'
}

vi.mock('@/store/session-states', () => ({
  storedSessionIdForRuntimeId: (runtimeId: string) => storedByRuntime[runtimeId] ?? null
}))

describe('session-scoped preview reader gate (#95459)', () => {
  const setupTabs = () => {
    $previewTabs.set([
      { id: 'url:tab-a', target: { kind: 'url', label: 'Browser', source: 'https://example.com', url: 'https://example.com' } },
    ])
  }

  it('rejects when a different session owns the live preview', () => {
    setupTabs()
    const unregister = registerPreviewPageReader('url:tab-a', async () => ({ text: '', title: '', url: '' }), 'session-A')

    // Session B asks about the exact tab: should be rejected (session-A owns it)
    expect(isLivePreviewTabOwnedBySession('url:tab-a', 'session-B')).toBe(false)

    // Session A asks about the exact tab: should be accepted (it owns it)
    expect(isLivePreviewTabOwnedBySession('url:tab-a', 'session-A')).toBe(true)

    unregister()
  })

  it('accepts the owning session after restart re-bind', () => {
    setupTabs()
    const unregister = registerPreviewPageReader('url:tab-a', async () => ({ text: '', title: '', url: '' }), 'session-owner')

    expect(isLivePreviewTabOwnedBySession('url:tab-a', 'session-owner')).toBe(true)
    expect(isLivePreviewTabOwnedBySession('url:tab-a', 'session-other')).toBe(false)

    unregister()
  })

  it('rejects when reader is unregistered', () => {
    setupTabs()
    const unregister = registerPreviewPageReader('url:tab-a', async () => ({ text: '', title: '', url: '' }), 'session-A')

    unregister()
    expect(isLivePreviewTabOwnedBySession('url:tab-a', 'session-A')).toBe(false)
  })

  it('empty sessionId or tabId always rejects', () => {
    setupTabs()
    const unregister = registerPreviewPageReader('url:tab-a', async () => ({ text: '', title: '', url: '' }), 'session-A')

    expect(isLivePreviewTabOwnedBySession('url:tab-a', '')).toBe(false)
    expect(isLivePreviewTabOwnedBySession('' as never, 'session-A')).toBe(false)

    unregister()
  })

  it('a tab with no live reader is not owned by any session', () => {
    $previewTabs.set([
      { id: 'url:tab-a', target: { kind: 'url', label: 'Browser', source: 'https://example.com', url: 'https://example.com' } },
    ])
    // No reader registered for tab-a — the open tab is not a LIVE preview, so
    // no session owns it (the gate must fail closed).
    expect(isLivePreviewTabOwnedBySession('url:tab-a', 'session-A')).toBe(false)
  })

  it('resolves ownership from the exact active tab, not the first owned tab (#95459 review)', () => {
    // Deterministic witness from the review: session S owns live previews A
    // and B (registered in that order), and B is the active one. Authorization
    // asks about the exact tab being mutated — never a first-owned-tab lookup.
    $previewTabs.set([
      { id: 'url:tab-a', target: { kind: 'url', label: 'Browser', source: 'https://x', url: 'https://x' } },
      { id: 'url:tab-b', target: { kind: 'url', label: 'Browser', source: 'https://y', url: 'https://y' } }
    ])

    const unregisterA = registerPreviewPageReader('url:tab-a', async () => ({ text: '', title: '', url: '' }), 'session-S')
    const unregisterB = registerPreviewPageReader('url:tab-b', async () => ({ text: '', title: '', url: '' }), 'session-S')

    // The active tab B IS owned by S -> allowed.
    expect(isLivePreviewTabOwnedBySession('url:tab-b', 'session-S')).toBe(true)
    // A non-active tab owned by S also answers true for itself (the mutation
    // targets that tab), but the admission layer gates on the ACTIVE tab.
    expect(isLivePreviewTabOwnedBySession('url:tab-a', 'session-S')).toBe(true)
    // A different session does not own either.
    expect(isLivePreviewTabOwnedBySession('url:tab-b', 'session-other')).toBe(false)

    unregisterA()
    unregisterB()
  })

  // The scenario in #95459's title, end to end through the real persistence
  // codec, with the EXACT reported sequence: preview works → restart → the
  // bot interacts with the ALREADY-OPEN preview. No fresh openPreview re-stamp
  // exists between hydration and the action — the restored tab must still
  // admit the same conversation under its NEW runtime id, via the durable
  // (stored) owner that survives the restart while the runtime id rotates.
  it('admits the restarted conversation on the already-open tab, no re-stamp (#95459)', () => {
    $previewTabs.set([])

    // R1 opens the preview; the tab is stamped with BOTH identity kinds — the
    // live runtime id and the durable stored id of the owning conversation.
    openPreview(
      { kind: 'url', label: 'Browser', source: 'https://example.com', url: 'https://example.com' },
      'tool-result',
      'runtime-R1',
      'stored-conversation'
    )
    const beforeRestart = $previewTabs.get()[0]

    expect(beforeRestart.ownerSessionId).toBe('runtime-R1')
    expect(beforeRestart.ownerStoredSessionId).toBe('stored-conversation')

    // Restart: tabs round-trip through the persistence codec. The RUNTIME id
    // is dead (dropped at hydration), the gateway mints a new runtime id (R2)
    // for the same conversation — but the STORED id survives, so the restored
    // tab still knows which conversation owns it.
    const restored = decodePreviewTabs(JSON.stringify($previewTabs.get()))

    $previewTabs.set(restored)

    const afterRestart = restored[0]
    expect(afterRestart.ownerSessionId).toBeUndefined()
    expect(afterRestart.ownerStoredSessionId).toBe('stored-conversation')

    // The pane re-registers the live reader for the restored tab (the
    // webview remounts after restart), binding the durable owner.
    const unregister = registerPreviewPageReader(
      afterRestart.id,
      async () => ({ text: '', title: '', url: '' }),
      undefined,
      afterRestart.ownerStoredSessionId
    )

    // #95459's exact action: the bot's NEW runtime id (R2) interacts with the
    // ALREADY-OPEN preview — admitted through the durable owner, with no
    // second openPreview in between. R1 maps to the same conversation, so it
    // answers the same way (a dead runtime id never sends events; what matters
    // is that a DIFFERENT conversation still cannot act on this tab).
    expect(isLivePreviewTabOwnedBySession(afterRestart.id, 'runtime-R2')).toBe(true)
    expect(isLivePreviewTabOwnedBySession(afterRestart.id, 'runtime-R1')).toBe(true)
    expect(isLivePreviewTabOwnedBySession(afterRestart.id, 'runtime-other-conversation')).toBe(false)

    unregister()
  })

  // The durable leg goes through the real translation: a runtime id whose
  // stored id is NOT the tab's owner must not be admitted by it.
  it('durable owner admits only the same conversation, not any rotated runtime id', () => {
    $previewTabs.set([
      {
        id: 'url:tab-a',
        ownerStoredSessionId: 'stored-mine',
        target: { kind: 'url', label: 'Browser', source: 'https://example.com', url: 'https://example.com' }
      }
    ])

    const unregister = registerPreviewPageReader(
      'url:tab-a',
      async () => ({ text: '', title: '', url: '' }),
      undefined,
      'stored-mine'
    )

    // runtime-X translates to stored-X: same conversation -> admitted.
    expect(isLivePreviewTabOwnedBySession('url:tab-a', 'runtime-mine')).toBe(true)
    // Different conversation -> refused.
    expect(isLivePreviewTabOwnedBySession('url:tab-a', 'runtime-elsewhere')).toBe(false)
    // A runtime id with no stored binding -> refused (fail closed).
    expect(isLivePreviewTabOwnedBySession('url:tab-a', 'unbound-runtime')).toBe(false)

    unregister()
  })

  it('a tab with no durable owner never admits through the durable leg', () => {
    $previewTabs.set([
      { id: 'url:tab-a', target: { kind: 'url', label: 'Browser', source: 'https://example.com', url: 'https://example.com' } }
    ])

    const unregister = registerPreviewPageReader(
      'url:tab-a',
      async () => ({ text: '', title: '', url: '' }),
      'runtime-live-only'
    )

    // Runtime leg still works; the durable leg has nothing to admit on.
    expect(isLivePreviewTabOwnedBySession('url:tab-a', 'runtime-live-only')).toBe(true)
    expect(isLivePreviewTabOwnedBySession('url:tab-a', 'runtime-anyone')).toBe(false)

    unregister()
  })
})
