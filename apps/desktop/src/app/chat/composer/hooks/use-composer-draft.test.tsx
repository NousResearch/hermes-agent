import { act, cleanup, render } from '@testing-library/react'
import { useLayoutEffect } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { clearSessionDraft, type ComposerAttachment, mainComposerScope, stashSessionDraft } from '@/store/composer'

import type { QueueEditState } from '../composer-utils'
import {
  type ComposerTarget,
  getActiveComposer,
  markActiveComposer,
  requestComposerFocus,
  requestComposerInsert
} from '../focus'
import { type ComposerScope, ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '../scope'

import { useComposerDraft } from './use-composer-draft'

const mockComposerApi = { setText: vi.fn() }

vi.mock('@assistant-ui/react', () => ({
  useAui: () => ({ composer: () => mockComposerApi }),
  useAuiState: (selector: (state: { composer: { text: string } }) => unknown) => selector({ composer: { text: '' } }),
  useComposerRuntime: () => ({
    getState: () => ({ text: '' }),
    subscribe: () => () => undefined
  })
}))

interface ProbeHarnessProps {
  activeQueueSessionKey: string | null
  onLayoutSnapshot: (attachments: ComposerAttachment[]) => void
  sessionId: string
}

function ProbeHarness({ activeQueueSessionKey, onLayoutSnapshot, sessionId }: ProbeHarnessProps) {
  useComposerDraft({
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
  })

  return null
}

describe('useComposerDraft — attachment scope stays coherent with the committed session on switch (#59305)', () => {
  afterEach(() => {
    cleanup()
    mainComposerScope.clear()
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

const CONNECTING_SESSION = 'session-connecting'

interface ConnectingHarnessProps {
  inputDisabled: boolean
}

/**
 * A composer with a real contentEditable bound to the hook's `editorRef`, so the
 * focus/insert buses can be observed end to end: `paintDraft` renders into this
 * element and `focusInput` focuses it. `ProbeHarness` above renders `null` (and
 * hardcodes `inputDisabled: false`) because it probes the attachment-scope swap;
 * these cases need the editor and need `inputDisabled` to be a prop they can flip.
 */
function ConnectingHarness({ inputDisabled }: ConnectingHarnessProps) {
  const { editorRef } = useComposerDraft({
    activeQueueSessionKey: CONNECTING_SESSION,
    focusKey: null,
    inputDisabled,
    queueEditRef: { current: null as QueueEditState | null },
    sessionId: CONNECTING_SESSION
  })

  return (
    <div>
      <div contentEditable={!inputDisabled} data-testid="editor" ref={editorRef} suppressContentEditableWarning />
      <button data-testid="elsewhere" type="button" />
    </div>
  )
}

describe('useComposerDraft — the focus/insert buses survive a connecting gateway', () => {
  afterEach(() => {
    // cleanup() unmounts, and the hook's scope-swap cleanup stashes whatever is
    // in the editor under CONNECTING_SESSION — drop it so the next mount's
    // takeSessionDraft() doesn't rehydrate the previous case's text.
    cleanup()
    clearSessionDraft(CONNECTING_SESSION)
    mainComposerScope.clear()
    markActiveComposer('main')
    mockComposerApi.setText.mockClear()
  })

  // `focus.ts`'s `dispatch` defers to a macrotask (`setTimeout(…, 0)`) so
  // synchronous click/keydown handlers finish first. Without draining it every
  // assertion below would pass vacuously — the event would never be delivered at
  // all, so "the draft is empty" and "the editor is not focused" would both hold
  // for the wrong reason.
  const flushBus = async () => {
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0))
    })
  }

  const editorOf = (container: HTMLElement) => container.querySelector<HTMLElement>('[data-testid="editor"]')!

  it('lands a type-to-focus keystroke in the draft while the gateway is connecting', async () => {
    const { container } = render(<ConnectingHarness inputDisabled />)
    const editor = editorOf(container)

    requestComposerFocus('main', { typeChar: 'h' })
    await flushBus()

    // The keybind layer already called preventDefault() before dispatching, so
    // this character has nowhere else to go.
    expect(editor.textContent).toBe('h')
    expect(mockComposerApi.setText).toHaveBeenCalledWith('h')

    // …but the editor is not contentEditable yet, so it must NOT be focused.
    expect(document.activeElement).not.toBe(editor)
  })

  it('lands an external insert in the draft while the gateway is connecting', async () => {
    const { container } = render(<ConnectingHarness inputDisabled />)
    const editor = editorOf(container)

    requestComposerInsert('pasted while connecting', { target: 'main' })
    await flushBus()

    expect(editor.textContent).toBe('pasted while connecting')
  })

  it('keeps the mid-connect keystroke and focuses the editor once the gateway opens', async () => {
    const { container, rerender } = render(<ConnectingHarness inputDisabled />)
    const editor = editorOf(container)

    requestComposerFocus('main', { typeChar: 'h' })
    await flushBus()
    expect(editor.textContent).toBe('h')

    // gatewayState 'connecting' → 'open' flips inputDisabled, which re-runs the
    // hook's focus effect — the deferral this fix relies on instead of dropping
    // the keystroke outright.
    await act(async () => {
      rerender(<ConnectingHarness inputDisabled={false} />)
    })

    expect(editor.textContent).toBe('h')
    expect(document.activeElement).toBe(editor)
  })

  it('still appends and focuses when the gateway is already open', async () => {
    const { container } = render(<ConnectingHarness inputDisabled={false} />)
    const editor = editorOf(container)

    // Mount focuses the composer; move focus away so the assertion below is
    // about the bus, not the mount.
    container.querySelector<HTMLElement>('[data-testid="elsewhere"]')!.focus()
    expect(document.activeElement).not.toBe(editor)

    requestComposerFocus('main', { typeChar: 'x' })
    await flushBus()

    expect(editor.textContent).toBe('x')
    expect(document.activeElement).toBe(editor)
  })
})
