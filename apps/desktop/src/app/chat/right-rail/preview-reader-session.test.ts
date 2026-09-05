import { describe, expect, it } from 'vitest'

import { $previewTabs, decodePreviewTabs, openPreview } from '@/store/preview'

import { isLivePreviewTabOwnedBySession, registerPreviewPageReader } from './preview-reader'

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
  // codec: a bot's preview must still be drivable after Desktop restarts, even
  // though the runtime session id rotates. Owner stamped -> round-tripped
  // through decodePreviewTabs -> session id rotated -> the NEW id authorizes.
  it('re-owns a restored tab after the runtime session id rotates', () => {
    $previewTabs.set([])

    // R1 opens the preview; the tab is stamped with that runtime id.
    openPreview(
      { kind: 'url', label: 'Browser', source: 'https://example.com', url: 'https://example.com' },
      'tool-result',
      'runtime-R1'
    )
    const beforeRestart = $previewTabs.get()[0]

    expect(beforeRestart.ownerSessionId).toBe('runtime-R1')

    // Restart: tabs round-trip through the persistence codec and the gateway
    // mints a new runtime id (R2) for the same conversation.
    const restored = decodePreviewTabs(JSON.stringify($previewTabs.get()))

    $previewTabs.set(restored)
    expect(restored[0].ownerSessionId).toBeUndefined()

    // A fresh routed open from R2 can now claim the restored tab...
    openPreview(
      { kind: 'url', label: 'Browser', source: 'https://example.com', url: 'https://example.com' },
      'tool-result',
      'runtime-R2'
    )
    const afterRestart = $previewTabs.get()[0]

    expect(afterRestart.ownerSessionId).toBe('runtime-R2')

    // ...and the gate admits R2 while still refusing the dead R1.
    const unregister = registerPreviewPageReader(
      afterRestart.id,
      async () => ({ text: '', title: '', url: '' }),
      afterRestart.ownerSessionId
    )

    expect(isLivePreviewTabOwnedBySession(afterRestart.id, 'runtime-R2')).toBe(true)
    expect(isLivePreviewTabOwnedBySession(afterRestart.id, 'runtime-R1')).toBe(false)

    unregister()
  })
})
