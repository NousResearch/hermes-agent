import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useLayoutEffect } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import {
  $freshDraftKey,
  $restoredDraftNotice,
  $salvagedEditNoticesBySession,
  announceGoneSessionDraft,
  announceNewSessionDraftKey,
  clearSessionDraft,
  type ComposerAttachment,
  dismissRestoredDraftNotice,
  getSalvagedEditNotice,
  mainComposerScope,
  NEW_SESSION_DRAFT_KEY,
  rotateFreshDraftKey,
  stashSessionDraft,
  takeSessionDraft
} from '@/store/composer'
import { $connection } from '@/store/session'

import { useComposerActions } from '../../hooks/use-composer-actions'
import type { QueueEditState } from '../composer-utils'
import { type ComposerTarget, getActiveComposer, markActiveComposer } from '../focus'
import { composerPlainText } from '../rich-editor'
import { type ComposerScope, ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '../scope'

import { useComposerDraft } from './use-composer-draft'

const mockComposerApi = { setText: vi.fn() }
// The composer-runtime mirror the mocked useComposerRuntime reports; tests
// that simulate typing write here and poke `mockComposerListener` (the
// editor DOM is not mounted in this harness, so this stands in for the
// DOM→draftRef rAF flush).
let mockComposerText = ''
let mockComposerListener: ((state: { text: string }) => void) | null = null

vi.mock('@assistant-ui/react', () => ({
  useAui: () => ({ composer: () => mockComposerApi }),
  useAuiState: (selector: (state: { composer: { text: string } }) => unknown) =>
    selector({ composer: { text: mockComposerText } }),
  useComposerRuntime: () => ({
    getState: () => ({ text: mockComposerText }),
    subscribe: (listener: (state: { text: string }) => void) => {
      mockComposerListener = listener

      return () => {
        mockComposerListener = null
      }
    }
  })
}))

interface ProbeHarnessProps {
  activeQueueSessionKey: string | null
  onLayoutSnapshot: (attachments: ComposerAttachment[]) => void
  /** Optional pre-paint text probe: called with the composer's mirrored text in the layout phase. */
  onTextSnapshot?: (text: string) => void
  sessionId: string
}

function ProbeHarness({ activeQueueSessionKey, onLayoutSnapshot, onTextSnapshot, sessionId }: ProbeHarnessProps) {
  const { draftRef } = useComposerDraft({
    activeQueueSessionKey,
    focusKey: null,
    inputDisabled: false,
    queueEditRef: { current: null as QueueEditState | null },
    sessionId
  })

  // useLayoutEffect fires synchronously right after the DOM commit, BEFORE
  // the hook's per-thread scope-swap useEffect (a passive effect) has a
  // chance to swap attachmentScope.$attachments over to the new session. A
  // synchronous read here — the same read ChatBar's `attachments` prop
  // performs at render time — observes the OUTGOING session's attachments.
  useLayoutEffect(() => {
    onLayoutSnapshot(mainComposerScope.$attachments.get())
    onTextSnapshot?.(draftRef.current)
  })

  return null
}


interface QueueEditProbeProps {
  activeQueueSessionKey: string | null
  onSnapshot: () => void
  queueEditRef: { current: QueueEditState | null }
  sessionId: string
}

function QueueEditProbe({ activeQueueSessionKey, onSnapshot, queueEditRef, sessionId }: QueueEditProbeProps) {
  useComposerDraft({
    activeQueueSessionKey,
    focusKey: null,
    inputDisabled: false,
    queueEditRef,
    sessionId
  })

  useLayoutEffect(() => {
    onSnapshot()
  })

  return null
}

describe('useComposerDraft — a clean queued edit restores the pre-edit draft with its own attachments (#88621)', () => {
  afterEach(() => {
    cleanup()
    mainComposerScope.clear()
    clearSessionDraft('edit-session-a')
    clearSessionDraft('edit-session-b')
  })

  it('A → B → A with a DIRTY edit keeps the displaced draft in the stash and the replacement in the notice (#88621 review R1)', () => {
    const originalAttachment: ComposerAttachment = { id: 'file:orig', kind: 'file', label: 'draft.txt' }
    const entryAttachment: ComposerAttachment = { id: 'file:queued', kind: 'file', label: 'queued.txt' }

    stashSessionDraft('edit-session-a', 'original draft', [originalAttachment])

    const queueEditRef: { current: QueueEditState | null } = {
      current: {
        attachments: [originalAttachment],
        draft: 'original draft',
        entryId: 'entry-1',
        entryText: 'original queued words',
        entryAttachments: [entryAttachment],
        sessionKey: 'edit-session-a'
      }
    }

    const { rerender } = render(
      <QueueEditProbe
        activeQueueSessionKey="edit-session-a"
        onSnapshot={() => undefined}
        queueEditRef={queueEditRef}
        sessionId="edit-session-a"
      />
    )

    // The user typed a replacement over the entry text; the rAF flush has
    // landed in the mirror.
    mockComposerText = 'unsaved replacement typed over the entry'
    act(() => {
      mockComposerListener?.({ text: mockComposerText })
    })

    // Switch to B: the swap cleanup must keep BOTH — the displaced draft in
    // the stash, the dirty buffer in the salvage notice.
    act(() => {
      rerender(
        <QueueEditProbe
          activeQueueSessionKey="edit-session-b"
          onSnapshot={() => undefined}
          queueEditRef={queueEditRef}
          sessionId="edit-session-b"
        />
      )
    })

    // Back to A: the restored draft is the ORIGINAL (X), not the replacement.
    act(() => {
      rerender(
        <QueueEditProbe
          activeQueueSessionKey="edit-session-a"
          onSnapshot={() => undefined}
          queueEditRef={queueEditRef}
          sessionId="edit-session-a"
        />
      )
    })

    const restored = takeSessionDraft('edit-session-a')
    expect(restored.text).toBe('original draft')
    expect(restored.attachments).toEqual([originalAttachment])

    // And the replacement (Z) is recoverable through the notice, described
    // against the state the composer now shows (X).
    const notice = getSalvagedEditNotice('edit-session-a')
    expect(notice?.undoText).toBe('unsaved replacement typed over the entry')
    expect(notice?.currentText).toBe('original draft')
  })

  it('A → B → A with an untouched edit keeps the original draft text AND its attachments', () => {
    // Opening a queued entry for edit, changing nothing, and switching away
    // must bring the ORIGINAL draft (text + attachments) back on return —
    // not the original text wearing the queued entry's attachments. The
    // editor is the source of truth: syncDraftFromEditor reads the DOM, and
    // the mounted composer paints the stashed A draft, so seed the editor DOM
    // with the entry text (what beginQueuedEdit paints) and keep the buffer
    // clean relative to that entry.
    const originalAttachment: ComposerAttachment = { id: 'file:orig', kind: 'file', label: 'draft.txt' }
    const queuedAttachment: ComposerAttachment = { id: 'file:queued', kind: 'file', label: 'queued.txt' }

    stashSessionDraft('edit-session-a', 'original draft', [originalAttachment])

    const queueEditRef: { current: QueueEditState | null } = {
      current: {
        attachments: [originalAttachment],
        draft: 'original draft',
        entryId: 'entry-1',
        // The editor holds exactly the entry's text: beginQueuedEdit painted
        // it and the user typed nothing (clean buffer). The harness composer
        // loads the stashed draft at mount, so the entry under edit shares
        // that text here. The entry's painted attachment payload must match
        // what the live scope holds at the swap (the queued entry's chips):
        // under the payload-aware dirty check (#88621 review R2), a chip set
        // that differs from the entry's painted set is DIRTY.
        entryText: 'original draft',
        entryAttachments: [queuedAttachment],
        sessionKey: 'edit-session-a'
      }
    }

    const { rerender } = render(
      <QueueEditProbe
        activeQueueSessionKey="edit-session-a"
        onSnapshot={() => undefined}
        queueEditRef={queueEditRef}
        sessionId="edit-session-a"
      />
    )

    // The live scope shows the queued entry's chips (painted at
    // beginQueuedEdit) — NOT the original draft's.
    mainComposerScope.$attachments.set([queuedAttachment])

    // Switch to B: the scope-swap cleanup stashes per the plan (clean →
    // pre-edit snapshot text + its OWN attachments).
    act(() => {
      rerender(
        <QueueEditProbe
          activeQueueSessionKey="edit-session-b"
          onSnapshot={() => undefined}
          queueEditRef={queueEditRef}
          sessionId="edit-session-b"
        />
      )
    })

    // Back to A: the stash must be the pre-edit draft with ITS OWN
    // attachments — the queued entry's chips must not have been grafted on.
    act(() => {
      rerender(
        <QueueEditProbe
          activeQueueSessionKey="edit-session-a"
          onSnapshot={() => undefined}
          queueEditRef={queueEditRef}
          sessionId="edit-session-a"
        />
      )
    })

    const restored = takeSessionDraft('edit-session-a')
    expect(restored.text).toBe('original draft')
    expect(restored.attachments).toEqual([originalAttachment])
  })
})

describe('useComposerDraft — attachment scope stays coherent with the committed session on switch (#59305)', () => {
  afterEach(() => {
    cleanup()
    mainComposerScope.clear()
    clearSessionDraft('session-A')
    clearSessionDraft('session-B')

    // Fresh-draft lifecycles rotate per test; the afterEach must sweep the
    // whole map or one test's abandoned bucket leaks into the next.
    for (const scope of ['session-created', NEW_SESSION_DRAFT_KEY, $freshDraftKey.get()]) {
      clearSessionDraft(scope)
    }

    rotateFreshDraftKey()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    vi.unstubAllGlobals()
    $connection.set(null)
  })

  it('clears the outgoing session attachments by the layout phase right after switching sessions', () => {
    const attachmentA: ComposerAttachment = { id: 'url-A', kind: 'url', label: 'A' }
    stashSessionDraft('session-A', 'hi from A', [attachmentA])

    const snapshots: ComposerAttachment[][] = []

    const { rerender } = render(
      <ProbeHarness activeQueueSessionKey="session-A" onLayoutSnapshot={s => snapshots.push(s)} sessionId="session-A" />
    )

    // Mount loads session A's stashed attachment into the (module-level) main
    // scope — confirms the fixture actually seeded the leak precondition.
    expect(mainComposerScope.$attachments.get()).toEqual([attachmentA])

    snapshots.length = 0 // drop the initial-mount snapshot; only the switch matters

    act(() => {
      rerender(
        <ProbeHarness
          activeQueueSessionKey="session-B"
          onLayoutSnapshot={s => snapshots.push(s)}
          sessionId="session-B"
        />
      )
    })

    // By the layout phase the scope must already be B's (empty) — a submit
    // fired the instant B renders must never ship session A's attachment.
    expect(snapshots[0]).toEqual([])
  })

  it("swaps the draft TEXT before paint: the layout phase of session B never observes A's draft (#66662 review)", () => {
    // The attachment-scope test above pins the layout-phase guarantee for
    // chips; this pins it for the TEXT. The review on #62586 (carried through
    // #128476) observed that an assertion made after flushSync returns proves
    // the eventual swap but not that the previous draft could not be painted
    // first: a passive useEffect restore would let the browser paint session
    // B's view with session A's text still loaded. The deterministic pre-paint
    // probe is a useLayoutEffect that runs synchronously after the DOM commit —
    // exactly the last moment before the browser may paint. If the restore
    // lived in a passive effect, the probe would observe A's text here.
    stashSessionDraft('session-A', 'draft typed in session A', [])
    stashSessionDraft('session-B', 'draft typed in session B', [])

    const textSnapshots: string[] = []

    const { rerender } = render(
      <ProbeHarness
        activeQueueSessionKey="session-A"
        onLayoutSnapshot={() => undefined}
        onTextSnapshot={t => textSnapshots.push(t)}
        sessionId="session-A"
      />
    )

    // Mount restored A's draft — the seeded precondition, same as the
    // attachment test.
    expect(mockComposerApi.setText).toHaveBeenCalledWith('draft typed in session A')

    mockComposerApi.setText.mockClear()
    textSnapshots.length = 0

    act(() => {
      rerender(
        <ProbeHarness
          activeQueueSessionKey="session-B"
          onLayoutSnapshot={() => undefined}
          onTextSnapshot={t => textSnapshots.push(t)}
          sessionId="session-B"
        />
      )
    })

    // Layout phase of B's switch render: the draft must already be B's. A
    // passive (useEffect) restore leaves A's text in place at this instant —
    // the browser would paint it first.
    expect(textSnapshots[0]).toBe('draft typed in session B')

    // The restore also reached the composer core in the same commit.
    expect(mockComposerApi.setText).toHaveBeenCalledWith('draft typed in session B')

    clearSessionDraft('session-A')
    clearSessionDraft('session-B')
  })

  it('carries a pre-session draft onto the session the fresh chat is re-homed to, before its runtime id is known', () => {
    const preSessionAttachment: ComposerAttachment = { id: 'file:new', kind: 'file', label: 'new.txt' }
    stashSessionDraft(null, 'do not lose this draft', [preSessionAttachment])

    const { rerender } = render(
      <ProbeHarness activeQueueSessionKey={null} onLayoutSnapshot={() => undefined} sessionId="" />
    )

    // Cold-start resume-last-session / first-send create: the route flips the
    // composer scope while `session.resume` has not published a runtime id yet.
    announceNewSessionDraftKey('session-created')
    act(() => {
      rerender(<ProbeHarness activeQueueSessionKey="session-created" onLayoutSnapshot={() => undefined} sessionId="" />)
    })

    expect(mainComposerScope.$attachments.get()).toEqual([preSessionAttachment])
    expect(takeSessionDraft('session-created')).toEqual({
      attachments: [preSessionAttachment],
      text: 'do not lose this draft'
    })
    expect(takeSessionDraft(null)).toEqual({ attachments: [], text: '' })
    clearSessionDraft('session-created')
  })

  it('carries the unsent draft of a GONE session into the fresh chat once, with an undoable notice (#111868)', () => {
    stashSessionDraft('session-gone', 'typed into a session that no longer exists', [])

    const { rerender } = render(
      <ProbeHarness activeQueueSessionKey="session-gone" onLayoutSnapshot={() => undefined} sessionId="session-gone" />
    )

    // The resume's gone verdict announces the dead key, then drops the window
    // to a fresh draft (route → /new, scope → the pre-session bucket).
    announceGoneSessionDraft('session-gone')
    act(() => {
      rerender(<ProbeHarness activeQueueSessionKey={null} onLayoutSnapshot={() => undefined} sessionId="" />)
    })

    expect(takeSessionDraft(null).text).toBe('typed into a session that no longer exists')
    expect(takeSessionDraft('session-gone').text).toBe('')
    expect($restoredDraftNotice.get()).toEqual({
      fromKey: 'session-gone',
      text: 'typed into a session that no longer exists'
    })

    // Fires once: a later trip through the fresh draft finds nothing to move
    // and does not re-publish the notice the user already dismissed.
    dismissRestoredDraftNotice()
    act(() => {
      rerender(
        <ProbeHarness activeQueueSessionKey="session-A" onLayoutSnapshot={() => undefined} sessionId="session-A" />
      )
    })
    act(() => {
      rerender(<ProbeHarness activeQueueSessionKey={null} onLayoutSnapshot={() => undefined} sessionId="" />)
    })

    expect($restoredDraftNotice.get()).toBeNull()
    expect(takeSessionDraft(null).text).toBe('typed into a session that no longer exists')
    clearSessionDraft(null)
  })

  it('leaves the pre-session draft in its bucket when the user opens another session from a fresh chat', () => {
    stashSessionDraft(null, 'still composing a new chat', [])

    const { rerender } = render(
      <ProbeHarness activeQueueSessionKey={null} onLayoutSnapshot={() => undefined} sessionId="" />
    )

    act(() => {
      rerender(
        <ProbeHarness activeQueueSessionKey="session-A" onLayoutSnapshot={() => undefined} sessionId="session-A" />
      )
    })

    expect(takeSessionDraft('session-A')).toEqual({ attachments: [], text: '' })
    expect(takeSessionDraft(null).text).toBe('still composing a new chat')
    clearSessionDraft(null)
  })

  it("isolates two concurrent new-chat lifecycles: the second fresh draft never shows the first one's text (#66662)", () => {
    const firstKey = rotateFreshDraftKey()
    const secondKey = rotateFreshDraftKey()

    expect(firstKey).not.toBe(secondKey)

    // First new chat: type unsent text under its own lifecycle key.
    stashSessionDraft(firstKey, 'first unsent chat', [])
    expect(takeSessionDraft(firstKey).text).toBe('first unsent chat')

    // A second New Chat rotated the key; its composer must restore empty —
    // the first chat's text is invisible until the user goes back.
    render(<ProbeHarness activeQueueSessionKey={secondKey} onLayoutSnapshot={() => undefined} sessionId="" />)

    expect(takeSessionDraft(secondKey)).toEqual({ attachments: [], text: '' })

    // The abandoned lifecycle keeps its text — no consumer of the second
    // lifecycle's scope can see or clobber it.
    expect(takeSessionDraft(firstKey).text).toBe('first unsent chat')

    clearSessionDraft(firstKey)
    clearSessionDraft(secondKey)
  })

  it("re-homes the ACTIVE lifecycle's draft onto the session its first send creates (#66662)", () => {
    const key = rotateFreshDraftKey()

    // The user typed in the current new chat; the swap cleanup stashed it
    // under the lifecycle key (null scope resolves to it).
    stashSessionDraft(null, 'typed before first send', [])

    const { rerender } = render(
      <ProbeHarness activeQueueSessionKey={key} onLayoutSnapshot={() => undefined} sessionId="" />
    )

    expect(takeSessionDraft(key).text).toBe('typed before first send')

    // First send: session.create assigns the stored id; the composer's scope
    // swap follows the announcement and moves THIS lifecycle's bucket.
    announceNewSessionDraftKey('session-created')
    act(() => {
      rerender(
        <ProbeHarness
          activeQueueSessionKey="session-created"
          onLayoutSnapshot={() => undefined}
          sessionId="session-created"
        />
      )
    })

    expect(takeSessionDraft('session-created').text).toBe('typed before first send')
    expect(takeSessionDraft(key).text).toBe('')

    clearSessionDraft('session-created')
  })

  it("keys a fresh chat's live stash under its lifecycle key, not the shared bucket (#66662)", () => {
    const key = rotateFreshDraftKey()

    const { unmount } = render(
      <ProbeHarness activeQueueSessionKey={key} onLayoutSnapshot={() => undefined} sessionId="" />
    )

    // Stash through the null scope the way the swap cleanup does when the
    // user types and navigates away mid-debounce.
    stashSessionDraft(null, 'typed in this lifecycle', [])

    expect(takeSessionDraft(key).text).toBe('typed in this lifecycle')
    expect(takeSessionDraft(NEW_SESSION_DRAFT_KEY).text).toBe('')

    unmount()
    clearSessionDraft(key)
  })

  it('applies a delayed image preview when it resolves while its attachment draft is inactive', async () => {
    const fullResolution =
      'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+GkZcAAAAASUVORK5CYII='

    const readFileDataUrl = vi.fn(async () => fullResolution)

    ;(
      window as unknown as {
        hermesDesktop: { readFileDataUrl: typeof readFileDataUrl }
      }
    ).hermesDesktop = { readFileDataUrl }

    let resolveBitmap!: (bitmap: { close: () => void; height: number; width: number }) => void

    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ blob: async () => new Blob([new Uint8Array([0])], { type: 'image/png' }) }))
    )
    vi.stubGlobal(
      'createImageBitmap',
      vi.fn(
        () =>
          new Promise<{ close: () => void; height: number; width: number }>(resolve => {
            resolveBitmap = resolve
          })
      )
    )

    class MockOffscreenCanvas {
      getContext = () => ({ drawImage: vi.fn() })
      convertToBlob = vi.fn(async () => new Blob(['thumbnail'], { type: 'image/png' }))
      constructor(_width: number, _height: number) {}
    }

    vi.stubGlobal('OffscreenCanvas', MockOffscreenCanvas)

    let actions!: ReturnType<typeof useComposerActions>

    function PreviewHarness({ activeQueueSessionKey }: { activeQueueSessionKey: string }) {
      useComposerDraft({
        activeQueueSessionKey,
        focusKey: null,
        inputDisabled: false,
        queueEditRef: { current: null as QueueEditState | null },
        sessionId: activeQueueSessionKey
      })
      actions = useComposerActions({ activeSessionId: null, currentCwd: '', requestGateway: vi.fn() })

      return null
    }

    const { rerender } = render(<PreviewHarness activeQueueSessionKey="session-A" />)
    let pending!: Promise<boolean>

    act(() => {
      pending = actions.attachImagePath('/tmp/round-trip.png')
    })

    await waitFor(() => expect(createImageBitmap).toHaveBeenCalledOnce())

    act(() => rerender(<PreviewHarness activeQueueSessionKey="session-B" />))
    expect(mainComposerScope.$attachments.get()).toEqual([])

    resolveBitmap({ close: vi.fn(), height: 3000, width: 4000 })

    await act(async () => {
      await pending
    })

    // The late completion belongs to A and must not leak into active session B.
    expect(mainComposerScope.$attachments.get()).toEqual([])

    act(() => rerender(<PreviewHarness activeQueueSessionKey="session-A" />))

    expect(mainComposerScope.$attachments.get()[0]?.thumbnailUrl).toMatch(/^data:image\/png;base64,/)
  })
})

describe('useComposerDraft — rehydrate diagnostic log stays redacted', () => {
  afterEach(() => {
    cleanup()
    mainComposerScope.clear()
    vi.restoreAllMocks()
  })

  it('logs counts/kinds/scope on restore but never the raw url, refText, or label', () => {
    const secretUrl = 'https://secret.example.com/private-workspace-path'

    const attachment: ComposerAttachment = {
      id: 'url-secret',
      kind: 'url',
      label: 'do-not-leak-label',
      refText: `@url:${secretUrl}`
    }

    stashSessionDraft('session-secret', '', [attachment])

    const debugSpy = vi.spyOn(console, 'debug').mockImplementation(() => undefined)

    render(
      <ProbeHarness
        activeQueueSessionKey="session-secret"
        onLayoutSnapshot={() => undefined}
        sessionId="session-secret"
      />
    )

    const rehydrateCalls = debugSpy.mock.calls.filter(call => call[0] === '[composer-rehydrate]')
    expect(rehydrateCalls.length).toBeGreaterThan(0)

    const serialized = JSON.stringify(rehydrateCalls)
    expect(serialized).not.toContain(secretUrl)
    expect(serialized).not.toContain(attachment.label)
    expect(serialized).not.toContain(attachment.refText)

    expect(rehydrateCalls[0]?.[1]).toMatchObject({
      attachmentCount: 1,
      attachmentKinds: ['url'],
      scope: 'session-secret'
    })
  })
})

describe('useComposerDraft — draft survives full unmount (Settings navigation, #41079)', () => {
  afterEach(() => {
    cleanup()
    mainComposerScope.clear()
    clearSessionDraft('session-nav')
  })

  it('stashes the unsent draft on unmount and restores it on remount', () => {
    // The user typed but has not sent; the draft was stashed by the normal
    // typing debounce path at some earlier point.
    stashSessionDraft('session-nav', 'unsent thought', [])

    const { unmount } = render(
      <ProbeHarness activeQueueSessionKey="session-nav" onLayoutSnapshot={() => undefined} sessionId="session-nav" />
    )

    // Navigating to Settings unmounts the chat composer entirely. The swap
    // effect's cleanup must stash the loaded draft back under its scope —
    // NOT drop it with the React state.
    unmount()

    mockComposerApi.setText.mockClear()

    const remount = render(
      <ProbeHarness activeQueueSessionKey="session-nav" onLayoutSnapshot={() => undefined} sessionId="session-nav" />
    )

    // Remount restored the text into the composer core (setText mirrors
    // paintDraft's write path — the editor DOM isn't mounted in this harness).
    expect(mockComposerApi.setText).toHaveBeenCalledWith('unsent thought')

    remount.unmount()
  })
})

describe('useComposerDraft — a closing composer hands the focus-bus key back', () => {
  afterEach(() => {
    cleanup()
    mainComposerScope.clear()
    markActiveComposer('main')
  })

  function renderScoped(target: ComposerTarget) {
    const scope: ComposerScope = { ...MAIN_COMPOSER_SCOPE, target }

    return render(
      <ComposerScopeProvider value={scope}>
        <ProbeHarness
          activeQueueSessionKey="session-tile"
          onLayoutSnapshot={() => undefined}
          sessionId="session-tile"
        />
      </ComposerScopeProvider>
    )
  }

  it('stops `active` resolving to a session tile once the tile unmounts', () => {
    const { unmount } = renderScoped('tile:abc')

    // Mounting claims the bus for this tile — the leak precondition.
    expect(getActiveComposer()).toBe('tile:abc')

    unmount()

    expect(getActiveComposer()).toBe('main')
  })

  it('leaves the key alone when another composer claimed it before this one unmounted', () => {
    const { unmount } = renderScoped('tile:abc')
    expect(getActiveComposer()).toBe('tile:abc')

    // The user clicks into a second tile, which claims the bus.
    markActiveComposer('tile:other')

    unmount()

    expect(getActiveComposer()).toBe('tile:other')
  })
})

describe('useComposerDraft — a hidden keep-alive tab never auto-focuses its composer', () => {
  afterEach(() => {
    cleanup()
    mainComposerScope.clear()
    markActiveComposer('main')
  })

  function renderScopedHidden(target: ComposerTarget, hidden: boolean) {
    const scope: ComposerScope = { ...MAIN_COMPOSER_SCOPE, target }

    return render(
      <PaneVisibleContext.Provider value={!hidden}>
        <ComposerScopeProvider value={scope}>
          <ProbeHarness
            activeQueueSessionKey="session-tile"
            onLayoutSnapshot={() => undefined}
            sessionId="session-tile"
          />
        </ComposerScopeProvider>
      </PaneVisibleContext.Provider>
    )
  }

  it('does not claim the focus bus when the composer mounts inside a hidden pane', () => {
    renderScopedHidden('tile:bg', true)

    expect(getActiveComposer()).toBe('main')
  })

  it('still claims the bus when the same composer becomes visible', () => {
    const { rerender } = renderScopedHidden('tile:fg', true)

    expect(getActiveComposer()).toBe('main')

    rerender(
      <PaneVisibleContext.Provider value={true}>
        <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, target: 'tile:fg' }}>
          <ProbeHarness
            activeQueueSessionKey="session-tile"
            onLayoutSnapshot={() => undefined}
            sessionId="session-tile"
          />
        </ComposerScopeProvider>
      </PaneVisibleContext.Provider>
    )

    expect(getActiveComposer()).toBe('tile:fg')
  })

  it('claims the bus on mount when visible', () => {
    renderScopedHidden('tile:vis', false)

    expect(getActiveComposer()).toBe('tile:vis')
  })

  function renderHiddenDraftHarness() {
    let hiddenDraft!: ReturnType<typeof useComposerDraft>

    function HiddenDraftHarness() {
      hiddenDraft = useComposerDraft({
        activeQueueSessionKey: 'session-hidden',
        focusKey: null,
        inputDisabled: false,
        queueEditRef: { current: null as QueueEditState | null },
        sessionId: 'session-hidden'
      })

      return <div contentEditable data-slot="composer-rich-input" ref={hiddenDraft.editorRef} />
    }

    render(
      <PaneVisibleContext.Provider value={false}>
        <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, target: 'tile:hidden' }}>
          <HiddenDraftHarness />
        </ComposerScopeProvider>
      </PaneVisibleContext.Provider>
    )

    return () => hiddenDraft
  }

  function createForegroundSelection() {
    const visibleEditor = globalThis.document.createElement('div')
    const visibleText = globalThis.document.createTextNode('foreground draft')
    visibleEditor.contentEditable = 'true'
    visibleEditor.tabIndex = 0
    visibleEditor.appendChild(visibleText)
    globalThis.document.body.appendChild(visibleEditor)
    visibleEditor.focus()
    expect(globalThis.document.activeElement).toBe(visibleEditor)

    const range = globalThis.document.createRange()
    range.setStart(visibleText, visibleText.textContent?.length ?? 0)
    range.collapse(true)
    window.getSelection()?.removeAllRanges()
    window.getSelection()?.addRange(range)

    return {
      editor: visibleEditor,
      startContainer: range.startContainer,
      startOffset: range.startOffset,
      endContainer: range.endContainer,
      endOffset: range.endOffset
    }
  }

  function expectForegroundSelectionPreserved(foreground: ReturnType<typeof createForegroundSelection>) {
    const selection = window.getSelection()
    expect(globalThis.document.activeElement).toBe(foreground.editor)
    expect(selection?.rangeCount).toBe(1)

    const range = selection!.getRangeAt(0)
    expect(range.startContainer).toBe(foreground.startContainer)
    expect(range.startOffset).toBe(foreground.startOffset)
    expect(range.endContainer).toBe(foreground.endContainer)
    expect(range.endOffset).toBe(foreground.endOffset)
  }

  it('does not move the document selection when a hidden composer reloads or clears its draft', () => {
    const getHiddenDraft = renderHiddenDraftHarness()
    const foreground = createForegroundSelection()

    act(() => getHiddenDraft().loadIntoComposer('background update', []))

    expect(composerPlainText(getHiddenDraft().editorRef.current!)).toBe('background update')
    expectForegroundSelectionPreserved(foreground)

    act(() => getHiddenDraft().clearDraft())

    expect(getHiddenDraft().editorRef.current?.textContent).toBe('')
    expectForegroundSelectionPreserved(foreground)
    foreground.editor.remove()
  })

  it('does not move the document selection when hidden composer refs are requested', () => {
    const getHiddenDraft = renderHiddenDraftHarness()
    const foreground = createForegroundSelection()

    act(() => getHiddenDraft().insertInlineRefs(['@file:`src/background.ts`']))

    expect(composerPlainText(getHiddenDraft().editorRef.current!)).toContain('@file:`src/background.ts`')
    expectForegroundSelectionPreserved(foreground)
    foreground.editor.remove()
  })

  it.each([false, true])('a late callback preserves another editor’s selection (hidden=%s)', hidden => {
    let draft!: ReturnType<typeof useComposerDraft>

    function Draft() {
      draft = useComposerDraft({
        activeQueueSessionKey: 'late-reject',
        focusKey: null,
        inputDisabled: false,
        queueEditRef: { current: null },
        sessionId: 'late-reject'
      })

      return <div contentEditable data-slot="composer-rich-input" ref={draft.editorRef} tabIndex={0} />
    }

    const { rerender } = render(
      <PaneVisibleContext value={true}>
        <Draft />
      </PaneVisibleContext>
    )

    const lateRestore = draft.loadIntoComposer
    const lateFocus = draft.focusInput
    rerender(
      <PaneVisibleContext value={!hidden}>
        <Draft />
      </PaneVisibleContext>
    )
    const foreground = createForegroundSelection()
    markActiveComposer('tile:foreground')

    act(() => {
      lateRestore('rejected draft', [])

      if (hidden) {
        lateFocus()
      }
    })

    expect(composerPlainText(draft.editorRef.current!)).toBe('rejected draft')
    expectForegroundSelectionPreserved(foreground)
    foreground.editor.remove()
  })
})

// #88621 review R5: the pagehide/reload boundary keeps a dirty queued edit.
// The flush must NOT simply skip an active edit (the old behavior dropped the
// replacement), nor stash the replacement as the draft while losing the
// displaced original — it keeps BOTH: the pre-edit draft stashed per the
// plan, the dirty buffer published as a durable salvage record.
describe('useComposerDraft — pagehide keeps a dirty queued edit (#88621 review R5)', () => {
  afterEach(() => {
    cleanup()
    mainComposerScope.clear()
    clearSessionDraft('pagehide-edit-session')
    $salvagedEditNoticesBySession.set({})
    window.localStorage.clear()
    mockComposerText = ''
  })

  it('a dirty edit at pagehide keeps the replacement durable alongside the displaced draft', () => {
    const originalAttachment: ComposerAttachment = { id: 'file:orig', kind: 'file', label: 'draft.txt' }
    const entryAttachment: ComposerAttachment = { id: 'file:queued', kind: 'file', label: 'queued.txt' }

    stashSessionDraft('pagehide-edit-session', 'original unsent draft', [originalAttachment])

    const queueEditRef: { current: QueueEditState | null } = {
      current: {
        attachments: [originalAttachment],
        draft: 'original unsent draft',
        entryId: 'entry-1',
        entryText: 'original queued words',
        entryAttachments: [entryAttachment],
        sessionKey: 'pagehide-edit-session'
      }
    }

    render(
      <QueueEditProbe
        activeQueueSessionKey="pagehide-edit-session"
        onSnapshot={() => undefined}
        queueEditRef={queueEditRef}
        sessionId="pagehide-edit-session"
      />
    )

    // The user typed a replacement into the queued edit; the rAF flush has
    // carried it to the composer mirror (draftRef), as it would before a
    // reload. The edit is DIRTY against the entry it was editing.
    mockComposerText = 'replacement typed into queued edit'
    act(() => {
      mockComposerListener?.({ text: mockComposerText })
    })

    act(() => {
      window.dispatchEvent(new Event('pagehide'))
    })

    // Three payloads, three owners (R1): the displaced original draft stays
    // the stash owner…
    const stashed = takeSessionDraft('pagehide-edit-session')
    expect(stashed.text).toBe('original unsent draft')
    expect(stashed.attachments).toEqual([originalAttachment])

    // …and the dirty replacement is durable through the salvage record (R5):
    // the bytes are on disk the moment pagehide runs, before unmount.
    const durable = window.localStorage.getItem('hermes.desktop.salvagedEdits.v1')

    expect(durable).toContain('"pagehide-edit-session"')
    expect(durable).toContain('replacement typed into queued edit')
    // The record describes the state Undo returns to (the displaced draft).
    expect(getSalvagedEditNotice('pagehide-edit-session')?.currentText).toBe('original unsent draft')
  })

  it('a clean edit at pagehide stashes the displaced draft with no salvage record', () => {
    const originalAttachment: ComposerAttachment = { id: 'file:orig', kind: 'file', label: 'draft.txt' }
    const entryAttachment: ComposerAttachment = { id: 'file:queued', kind: 'file', label: 'queued.txt' }

    stashSessionDraft('pagehide-edit-session', 'original unsent draft', [originalAttachment])

    const queueEditRef: { current: QueueEditState | null } = {
      current: {
        attachments: [originalAttachment],
        draft: 'original unsent draft',
        entryId: 'entry-1',
        entryText: 'original queued words',
        entryAttachments: [entryAttachment],
        sessionKey: 'pagehide-edit-session'
      }
    }

    render(
      <QueueEditProbe
        activeQueueSessionKey="pagehide-edit-session"
        onSnapshot={() => undefined}
        queueEditRef={queueEditRef}
        sessionId="pagehide-edit-session"
      />
    )

    // Clean buffer: the mirror holds exactly the entry's painted text…
    mockComposerText = 'original queued words'
    act(() => {
      mockComposerListener?.({ text: mockComposerText })
    })
    // …and the live chips match the entry's painted set.
    mainComposerScope.$attachments.set([{ ...entryAttachment }])

    act(() => {
      window.dispatchEvent(new Event('pagehide'))
    })

    expect(takeSessionDraft('pagehide-edit-session').text).toBe('original unsent draft')
    expect(getSalvagedEditNotice('pagehide-edit-session')).toBeNull()
    expect(window.localStorage.getItem('hermes.desktop.salvagedEdits.v1')).toBeNull()
  })
})
