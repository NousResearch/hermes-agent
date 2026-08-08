import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

/**
 * The timeline must do NO work it can't currently show. Two gates are proven
 * here by rendering the real component and counting the work it performs:
 *
 *  - a background (kept-alive but hidden) tab derives nothing and subscribes
 *    to nothing — the transcript selector is never even called;
 *  - an unhovered rail builds its ticks but not the popover's rows.
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

    return selector({ thread: { messages } })
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

afterEach(() => {
  cleanup()
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

describe('ThreadTimeline popover', () => {
  it('builds no rows until the rail is hovered', () => {
    messages = transcript(6)

    const { container } = renderTimeline()
    const popover = container.querySelector('[data-slot="thread-timeline-popover"]')

    // The shell renders (it owns the fade transition); its rows do not.
    expect(popover).not.toBeNull()
    expect(popover?.querySelectorAll('button')).toHaveLength(0)
    expect(screen.queryByText('prompt 0')).toBeNull()
  })

  it('builds the rows on hover and keeps them for the close fade', () => {
    messages = transcript(6)

    const { container } = renderTimeline()
    const rail = container.querySelector<HTMLElement>('[data-slot="thread-timeline"]')!

    fireEvent.mouseEnter(rail)

    const popover = container.querySelector('[data-slot="thread-timeline-popover"]')
    expect(popover?.querySelectorAll('button')).toHaveLength(6)

    fireEvent.mouseLeave(rail)

    // Still mounted — the popover fades out, it does not pop out of existence.
    expect(popover?.querySelectorAll('button')).toHaveLength(6)
  })
})

describe('ThreadTimeline below the threshold', () => {
  it('renders nothing for a short thread', () => {
    messages = transcript(2)

    const { container } = renderTimeline()

    expect(container.querySelector('[data-slot="thread-timeline"]')).toBeNull()
  })
})

describe('ThreadTimeline marker click', () => {
  // jumpScroll animates over ~170ms via requestAnimationFrame. Drive each
  // callback synchronously with a far-future timestamp so one step lands the
  // final scrollTop (p = min(1, Δt/170) = 1 on the first frame).
  const stubRaf = () => {
    vi.stubGlobal(
      'requestAnimationFrame',
      vi.fn((callback: (now: number) => void) => {
        callback(performance.now() + 1000)

        return 1
      })
    )
    vi.stubGlobal('cancelAnimationFrame', vi.fn())
  }

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('scrolls the viewport to the clicked prompt turn, measured in-flow (#81768)', () => {
    messages = transcript(6)
    stubRaf()
    // jsdom ships no CSS.escape; the rail's measure effect queries
    // `[data-message-id="…"]` through it on mount.
    vi.stubGlobal('CSS', { escape: (value: string) => value })

    const { container } = render(
      <div data-session-anchor="workspace">
        <div data-slot="aui_thread-viewport" />
        <ThreadTimeline />
      </div>
    )

    const viewport = container.querySelector<HTMLElement>('[data-slot="aui_thread-viewport"]')!
    viewport.getBoundingClientRect = () => ({ top: 100 } as DOMRect)
    Object.defineProperty(viewport, 'scrollTop', { configurable: true, writable: true, value: 400 })

    // Each prompt is a sticky bubble pinned to the viewport top (~4px); its
    // turn container keeps the in-flow position (1200px into the transcript).
    // The regression: the old code measured the PINNED bubble (top ≈ 104) and
    // computed a ~0px jump, so every old marker's click was a silent no-op.
    for (const message of messages) {
      const turn = document.createElement('div')
      const bubble = document.createElement('div')
      bubble.style.position = 'sticky'
      bubble.dataset.messageId = message.id
      turn.appendChild(bubble)
      viewport.appendChild(turn)
      turn.getBoundingClientRect = () => ({ top: 1200 } as DOMRect)
      bubble.getBoundingClientRect = () => ({ top: 104 } as DOMRect)
    }

    const ticks = container.querySelectorAll('[data-slot="thread-timeline-ticks"] button')
    expect(ticks).toHaveLength(6)

    fireEvent.click(ticks[0])

    // 400 + 1200 - 100 - 8 = 1492: the turn's in-flow position, offset 8px
    // above the viewport top — not the bubble's pinned 104px.
    expect(viewport.scrollTop).toBe(1492)
  })
})

describe('ThreadTimeline while a reply streams', () => {
  it('does not re-derive the rail as assistant content grows', () => {
    messages = [...transcript(6), { content: [{ text: 'th', type: 'text' }], id: 'a1', role: 'assistant' }]

    const { rerender } = renderTimeline()
    const derivations = transcriptReads.mock.calls.length

    // A token lands: the assistant message's content changes, the user prompt
    // ids do not — so the memo's change signal is untouched and the previews
    // are never rebuilt.
    messages = [
      ...messages.slice(0, -1),
      { content: [{ text: 'thinking…', type: 'text' }], id: 'a1', role: 'assistant' }
    ]
    rerender(<ThreadTimeline />)

    expect(transcriptReads.mock.calls.length).toBe(derivations)
  })

  it('re-derives once a new prompt is sent', () => {
    messages = transcript(6)

    const { rerender } = renderTimeline()
    const derivations = transcriptReads.mock.calls.length

    messages = [...messages, userTurn('u6', 'prompt 6')]
    rerender(<ThreadTimeline />)

    expect(transcriptReads.mock.calls.length).toBeGreaterThan(derivations)
  })
})
