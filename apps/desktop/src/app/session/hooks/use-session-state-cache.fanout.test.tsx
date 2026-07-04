import { act, cleanup, render } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createSplitPaneView, type PaneSessionView } from '@/app/chat/pane-view'
import type { ChatMessage } from '@/lib/chat-messages'
import { $currentModel, $messages, $turnStartedAt, setCurrentModel, setTurnStartedAt } from '@/store/session'

import { useSessionStateCache } from './use-session-state-cache'

// Multi-view fan-out: the cache projects each session's truth into EVERY
// registered pane view whose gate matches — the split pane's bundle next to
// the main (options-built) view — without the panes bleeding into each other.

type Cache = ReturnType<typeof useSessionStateCache>

interface HarnessProps {
  activeSessionId: string | null
  onReady: (cache: Cache) => void
}

function Harness({ activeSessionId, onReady }: HarnessProps) {
  const busyRef: MutableRefObject<boolean> = { current: false }

  const cache = useSessionStateCache({
    activeSessionId,
    busyRef,
    selectedStoredSessionId: null,
    setAwaitingResponse: () => undefined,
    setBusy: () => undefined,
    // Wire the published main view back into the real $messages atom so the
    // main-pane round-trip matches production.
    setMessages: messages => $messages.set(messages)
  })

  onReady(cache)

  return null
}

function userMessage(id: string, text: string): ChatMessage {
  return { id, role: 'user', parts: [{ type: 'text', text }] }
}

function assistantText(id: string, text: string): ChatMessage {
  return { id, role: 'assistant', parts: [{ type: 'text', text }] }
}

describe('useSessionStateCache — multi-view fan-out', () => {
  let splitView: PaneSessionView

  beforeEach(() => {
    // Synchronous rAF (returning null) mirrors the single-view suite's stub —
    // see use-session-state-cache.test.tsx for why null is load-bearing.
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb: FrameRequestCallback) => {
      cb(0)

      return null as unknown as number
    })

    splitView = createSplitPaneView()
    splitView.activeSessionIdRef.current = 'rt-split'
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    $messages.set([])
    setCurrentModel('')
    setTurnStartedAt(null)
  })

  it("routes a session's update to the registered pane viewing it — and only there", () => {
    let cache!: Cache
    render(<Harness activeSessionId="rt-main" onReady={c => (cache = c)} />)

    const off = cache.registerPaneView(splitView)

    act(() => {
      cache.updateSessionState(
        'rt-split',
        state => ({
          ...state,
          busy: true,
          messages: [userMessage('m1', 'hi from the split')],
          model: 'model-x',
          turnStartedAt: 1_700_000_000_000
        }),
        'stored-split'
      )
    })

    // The split bundle received the projection...
    expect(splitView.$messages.get().map(message => message.id)).toEqual(['m1'])
    expect(splitView.$busy.get()).toBe(true)
    expect(splitView.$currentModel.get()).toBe('model-x')
    expect(splitView.$turnStartedAt.get()).toBe(1_700_000_000_000)
    expect(splitView.busyRef.current).toBe(true)
    // ...and the main/global view stayed untouched: no transcript bleed, no
    // statusbar metadata or turn-clock crosstalk.
    expect($messages.get()).toEqual([])
    expect($currentModel.get()).toBe('')
    expect($turnStartedAt.get()).toBeNull()

    off()
  })

  it('keeps the main single-view path intact while a second view is registered', () => {
    let cache!: Cache
    render(<Harness activeSessionId="rt-main" onReady={c => (cache = c)} />)

    const off = cache.registerPaneView(splitView)

    act(() => {
      cache.updateSessionState(
        'rt-main',
        state => ({ ...state, busy: false, messages: [userMessage('u1', 'hello main')] }),
        'stored-main'
      )
    })

    expect($messages.get().map(message => message.id)).toEqual(['u1'])
    expect(splitView.$messages.get()).toEqual([])

    off()
  })

  it('flushes a critical transition synchronously per pane even when rAF never fires', () => {
    // Electron parks rAF for occluded windows; the settle edge must not wait.
    vi.mocked(window.requestAnimationFrame).mockImplementation(() => 42)

    let cache!: Cache
    render(<Harness activeSessionId="rt-main" onReady={c => (cache = c)} />)

    const off = cache.registerPaneView(splitView)

    act(() => {
      cache.updateSessionState(
        'rt-split',
        state => ({ ...state, busy: true, messages: [userMessage('m1', 'streaming…')] }),
        'stored-split'
      )
    })

    // Plain busy heartbeat stays rAF-batched → still staged, not visible.
    expect(splitView.$messages.get()).toEqual([])

    act(() => {
      cache.updateSessionState('rt-split', state => ({
        ...state,
        busy: false,
        messages: [userMessage('m1', 'streaming…'), assistantText('a1', 'done')]
      }))
    })

    // busy→false is a critical transition → synchronous flush, no rAF needed.
    expect(splitView.$messages.get().map(message => message.id)).toEqual(['m1', 'a1'])
    expect(splitView.$busy.get()).toBe(false)

    off()
  })

  it('unregistering a pane view stops its publishes (session keeps caching as background truth)', () => {
    let cache!: Cache
    render(<Harness activeSessionId="rt-main" onReady={c => (cache = c)} />)

    const off = cache.registerPaneView(splitView)

    act(() => {
      cache.updateSessionState(
        'rt-split',
        state => ({ ...state, busy: false, messages: [userMessage('m1', 'first')] }),
        'stored-split'
      )
    })

    expect(splitView.$messages.get().map(message => message.id)).toEqual(['m1'])

    off()

    act(() => {
      cache.updateSessionState('rt-split', state => ({
        ...state,
        busy: false,
        messages: [userMessage('m1', 'first'), assistantText('a1', 'late reply')]
      }))
    })

    // View frozen at unregister time; the truth layer still advanced.
    expect(splitView.$messages.get().map(message => message.id)).toEqual(['m1'])
    expect(cache.sessionStateByRuntimeIdRef.current.get('rt-split')?.messages.map(m => m.id)).toEqual(['m1', 'a1'])
  })

  it('publishPaneState repaints a pane synchronously from the cached truth (warm activation)', () => {
    let cache!: Cache
    render(<Harness activeSessionId="rt-main" onReady={c => (cache = c)} />)

    const off = cache.registerPaneView(splitView)

    // Truth accrues while the split pane is NOT viewing the session — the
    // staging gate must hold it back.
    splitView.activeSessionIdRef.current = null

    act(() => {
      cache.updateSessionState(
        'rt-split',
        state => ({ ...state, busy: false, messages: [userMessage('m1', 'hello')], model: 'cached-model' }),
        'stored-split'
      )
    })

    expect(splitView.$messages.get()).toEqual([])

    // The pane switches onto the session → warm repaint from cache.
    splitView.activeSessionIdRef.current = 'rt-split'

    act(() => {
      cache.publishPaneState('split')
    })

    expect(splitView.$messages.get().map(message => message.id)).toEqual(['m1'])
    expect(splitView.$currentModel.get()).toBe('cached-model')

    // No-ops: unknown pane, or a pane with no cached session.
    splitView.activeSessionIdRef.current = 'rt-unknown'

    act(() => {
      cache.publishPaneState('split')
    })

    expect(splitView.$messages.get().map(message => message.id)).toEqual(['m1'])

    off()
  })
})
