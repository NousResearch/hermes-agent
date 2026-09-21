import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { type ReactNode, useSyncExternalStore } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setHideThreadTimeline } from '@/store/thread-timeline'

/**
 * The timeline must do NO work it can't currently show. Two gates are proven
 * here by rendering the real component and counting the work it performs:
 *
 *  - a background (kept-alive but hidden) tab derives nothing and subscribes
 *    to nothing — the transcript selector is never even called;
 *  - an unhovered rail builds only a bounded tick slice, never a label list.
 *
 * The prompt-id selector is also asserted to be content-blind, which is what
 * keeps a streaming assistant reply from re-deriving previews per token.
 */

interface FakeMessage {
  content: unknown
  id: string
  role: string
}

const selectorCalls = vi.fn()
const transcriptReads = vi.fn()
const messageListeners = new Set<() => void>()
let messages: FakeMessage[] = []

vi.mock('@assistant-ui/react', () => ({
  useAui: () => ({
    thread: () => ({
      getState: () => {
        transcriptReads()

        return { messages }
      }
    })
  }),
  useAuiState: (selector: (state: { thread: { messages: FakeMessage[] } }) => unknown) => {
    selectorCalls()

    return useSyncExternalStore(
      listener => {
        messageListeners.add(listener)

        return () => messageListeners.delete(listener)
      },
      () => selector({ thread: { messages } })
    )
  }
}))

let paneActive = true

vi.mock('@/components/pane-shell/pane-visibility', () => ({
  usePaneVisible: () => paneActive
}))

vi.mock('@/lib/haptics', () => ({ triggerHaptic: () => {} }))

const { ThreadTimeline } = await import('./timeline')

const userTurn = (id: string, text: string): FakeMessage => ({
  content: [{ text, type: 'text' }],
  id,
  role: 'user'
})

const transcript = (count: number): FakeMessage[] =>
  Array.from({ length: count }, (_, i) => userTurn(`u${i}`, `prompt ${i}`))

const renderTimeline = (ui: ReactNode = <ThreadTimeline />) => render(ui)

beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, 'offsetHeight', 'get').mockReturnValue(300)
  vi.spyOn(HTMLElement.prototype, 'offsetWidth', 'get').mockReturnValue(48)
})

afterEach(() => {
  cleanup()
  setHideThreadTimeline(false)
  vi.restoreAllMocks()
  selectorCalls.mockClear()
  transcriptReads.mockClear()
  paneActive = true
  messages = []
})

describe('ThreadTimeline in a background tab', () => {
  it('renders nothing and never reads the transcript', () => {
    paneActive = false
    messages = transcript(6)

    const { container } = renderTimeline()

    expect(container.querySelector('[data-slot="thread-timeline"]')).toBeNull()
    expect(selectorCalls).not.toHaveBeenCalled()
    expect(transcriptReads).not.toHaveBeenCalled()
  })

  it('renders the rail once its pane becomes the visible tab', () => {
    messages = transcript(6)

    const { container } = renderTimeline()

    expect(container.querySelector('[data-slot="thread-timeline"]')).not.toBeNull()
    expect(selectorCalls).toHaveBeenCalled()
  })
})

describe('ThreadTimeline idle work', () => {
  it('builds ticks without a separate popover or mounted labels', () => {
    messages = transcript(6)

    const { container } = renderTimeline()
    const popover = container.querySelector('[data-slot="thread-timeline-popover"]')

    expect(popover).toBeNull()
    expect(container.querySelectorAll('[data-timeline-id]')).toHaveLength(6)
    expect(screen.queryByText('prompt 0')).toBeNull()
  })

  it('keeps a large rail bounded before and after pointer movement', () => {
    messages = transcript(5000)

    const { container } = renderTimeline()
    const rail = container.querySelector<HTMLElement>('[data-slot="thread-timeline-ticks"]')!
    const ticks = Array.from(rail.querySelectorAll('[data-timeline-id]'))

    expect(ticks.length).toBeGreaterThan(0)
    expect(ticks.length).toBeLessThan(60)

    fireEvent.pointerMove(rail, { clientY: 100, pointerType: 'mouse' })
    fireEvent.pointerLeave(rail)

    expect(Array.from(rail.querySelectorAll('[data-timeline-id]'))).toEqual(ticks)
    expect(container.querySelector('[data-slot="thread-timeline-popover"]')).toBeNull()
    expect(screen.queryByText('prompt 0')).toBeNull()
  })

  it('schedules no history read for an unsaved conversation', () => {
    messages = transcript(2)
    const schedule = vi.spyOn(window, 'setTimeout')

    renderTimeline()

    expect(schedule.mock.calls.filter(([, delay]) => delay === 200)).toHaveLength(0)
  })
})

describe('ThreadTimeline availability', () => {
  it('hides every rail without reading transcripts and restores them when enabled', () => {
    messages = transcript(2)
    setHideThreadTimeline(true)

    const { container } = renderTimeline(
      <>
        <ThreadTimeline />
        <ThreadTimeline />
      </>
    )
    const rails = () => container.querySelectorAll('[data-slot="thread-timeline"]')

    expect(rails()).toHaveLength(0)
    expect(selectorCalls).not.toHaveBeenCalled()
    expect(transcriptReads).not.toHaveBeenCalled()

    act(() => setHideThreadTimeline(false))
    expect(rails()).toHaveLength(2)

    act(() => setHideThreadTimeline(true))
    expect(rails()).toHaveLength(0)
  })

  it('keeps navigation available for a short thread', () => {
    messages = transcript(2)

    const { container } = renderTimeline()

    expect(container.querySelector('[data-slot="thread-timeline"]')).not.toBeNull()
    expect(container.querySelectorAll('[data-timeline-id]')).toHaveLength(2)
  })
})

describe('ThreadTimeline while a reply streams', () => {
  it('does not re-derive the rail as assistant content grows', () => {
    messages = [...transcript(6), { content: [{ text: 'th', type: 'text' }], id: 'a1', role: 'assistant' }]

    renderTimeline()
    const derivations = transcriptReads.mock.calls.length

    // A token lands: the assistant message's content changes, the user prompt
    // ids do not — so the memo's change signal is untouched and the previews
    // are never rebuilt.
    messages = [
      ...messages.slice(0, -1),
      { content: [{ text: 'thinking…', type: 'text' }], id: 'a1', role: 'assistant' }
    ]
    act(() => messageListeners.forEach(listener => listener()))

    expect(transcriptReads.mock.calls.length).toBe(derivations)
  })

  it('re-derives once a new prompt is sent', () => {
    messages = transcript(6)

    renderTimeline()
    const derivations = transcriptReads.mock.calls.length

    messages = [...messages, userTurn('u6', 'prompt 6')]
    act(() => messageListeners.forEach(listener => listener()))

    expect(transcriptReads.mock.calls.length).toBeGreaterThan(derivations)
  })
})
