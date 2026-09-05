import { act, cleanup, render, renderHook, waitFor } from '@testing-library/react'
import { useLayoutEffect } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import {
  clearSessionDraft,
  type ComposerAttachment,
  mainComposerScope,
  stashSessionDraft,
  takeSessionDraft
} from '@/store/composer'
import { $connection } from '@/store/session'

import { useComposerActions } from '../../hooks/use-composer-actions'
import type { QueueEditState } from '../composer-utils'
import { type ComposerTarget, getActiveComposer, markActiveComposer } from '../focus'
import { type ComposerScope, ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '../scope'

import { useComposerDraft } from './use-composer-draft'

const mockComposerApi = { setText: vi.fn() }
let mockComposerRuntimeText = ''
let mockComposerRuntimeListener: (() => void) | null = null

const mockComposerRuntime = {
  getState: () => ({ text: mockComposerRuntimeText }),
  subscribe: (listener: () => void) => {
    mockComposerRuntimeListener = listener

    return () => {
      mockComposerRuntimeListener = null
    }
  }
}

vi.mock('@assistant-ui/react', () => ({
  useAui: () => ({ composer: () => mockComposerApi }),
  useAuiState: (selector: (state: { composer: { text: string } }) => unknown) => selector({ composer: { text: '' } }),
  useComposerRuntime: () => mockComposerRuntime
}))

describe('useComposerDraft — main draft isolation', () => {
  afterEach(() => {
    cleanup()

    mockComposerApi.setText.mockClear()
    mockComposerRuntimeText = ''
    mockComposerRuntimeListener = null
    clearSessionDraft('session-perf')
  })

  it('keeps draft text out of the shared Assistant UI runtime', () => {
    const { result } = renderHook(() =>
      useComposerDraft({
        activeQueueSessionKey: 'session-perf',
        focusKey: null,
        inputDisabled: false,
        queueEditRef: { current: null as QueueEditState | null },
        sessionId: 'session-perf'
      })
    )

    mockComposerApi.setText.mockClear()

    act(() => result.current.setComposerText('a normal draft'))

    expect(mockComposerApi.setText).not.toHaveBeenCalled()
  })

  it('derives composer eligibility from the local draft', () => {
    const { result } = renderHook(() =>
      useComposerDraft({
        activeQueueSessionKey: 'session-perf',
        focusKey: null,
        inputDisabled: false,
        queueEditRef: { current: null as QueueEditState | null },
        sessionId: 'session-perf'
      })
    )

    act(() => result.current.setComposerText('?'))
    expect(result.current.hasText).toBe(true)
    expect(result.current.isHelpHint).toBe(true)
    expect(result.current.isSteerableText).toBe(true)

    act(() => result.current.setComposerText('/help'))
    expect(result.current.hasText).toBe(true)
    expect(result.current.isHelpHint).toBe(false)
    expect(result.current.isSteerableText).toBe(false)
  })

  it('derives multiline layout edges from the local draft', () => {
    const { result } = renderHook(() =>
      useComposerDraft({
        activeQueueSessionKey: 'session-perf',
        focusKey: null,
        inputDisabled: false,
        queueEditRef: { current: null as QueueEditState | null },
        sessionId: 'session-perf'
      })
    )

    act(() => result.current.setComposerText('line one\nline two'))
    expect(result.current.isEmpty).toBe(false)
    expect(result.current.hasHardNewline).toBe(true)

    act(() => result.current.setComposerText('line one\n'))
    expect(result.current.hasHardNewline).toBe(false)

    act(() => result.current.setComposerText(''))
    expect(result.current.isEmpty).toBe(true)
    expect(result.current.hasHardNewline).toBe(false)
  })

  it('persists the local draft after the debounce window', async () => {
    const queueEditRef = { current: null as QueueEditState | null }

    const { result } = renderHook(() =>
      useComposerDraft({
        activeQueueSessionKey: 'session-perf',
        focusKey: null,
        inputDisabled: false,
        queueEditRef,
        sessionId: 'session-perf'
      })
    )

    act(() => result.current.setComposerText('survives navigation'))

    await waitFor(() => expect(takeSessionDraft('session-perf').text).toBe('survives navigation'), { timeout: 1_000 })
  })

  it('does not let the shared runtime overwrite a local draft', () => {
    const { result } = renderHook(() =>
      useComposerDraft({
        activeQueueSessionKey: 'session-perf',
        focusKey: null,
        inputDisabled: false,
        queueEditRef: { current: null as QueueEditState | null },
        sessionId: 'session-perf'
      })
    )

    act(() => result.current.setComposerText('keep this text'))
    mockComposerRuntimeText = ''
    act(() => mockComposerRuntimeListener?.())

    expect(result.current.draftRef.current).toBe('keep this text')
  })
})

interface ProbeHarnessProps {
  activeQueueSessionKey: string | null
  onLayoutSnapshot: (attachments: ComposerAttachment[], text: string) => void
  sessionId: string
}

function ProbeHarness({ activeQueueSessionKey, onLayoutSnapshot, sessionId }: ProbeHarnessProps) {
  const draft = useComposerDraft({
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
    onLayoutSnapshot(mainComposerScope.$attachments.get(), draft.draftRef.current)
  })

  return null
}

describe('useComposerDraft — attachment scope stays coherent with the committed session on switch (#59305)', () => {
  afterEach(() => {
    cleanup()
    mainComposerScope.clear()
    clearSessionDraft('session-A')
    clearSessionDraft('session-B')
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

    const restoredTexts: string[] = []

    const remount = render(
      <ProbeHarness
        activeQueueSessionKey="session-nav"
        onLayoutSnapshot={(_attachments, text) => restoredTexts.push(text)}
        sessionId="session-nav"
      />
    )

    // Remount restored the text into the composer's authoritative local mirror
    // without routing it through the transcript-owning Assistant UI runtime.
    expect(restoredTexts).toContain('unsent thought')
    expect(mockComposerApi.setText).not.toHaveBeenCalledWith('unsent thought')

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
})
