import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { onRevealMessageRequest } from '@/store/thread-scroll'

import { TranscriptWindowProvider } from './transcript-window'

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
  globalThis.document.body.replaceChildren()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
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

describe('ThreadTimeline with a bounded runtime window', () => {
  it('keeps older prompts in the rail and asks the window to reveal hidden targets', () => {
    messages = transcript(2)
    const cancelReveal = vi.fn()
    const revealMessage = vi.fn()
    const timelineEntries = transcript(6).map((message, index) => ({ id: message.id, preview: `prompt ${index}` }))

    renderTimeline(
      <TranscriptWindowProvider
        value={{
          cancelReveal,
          olderAvailable: true,
          expandWindow: vi.fn(),
          revealMessage,
          revealScope: 'runtime-1',
          timelineEntries
        }}
      >
        <ThreadTimeline />
      </TranscriptWindowProvider>
    )

    expect(screen.getAllByRole('button')).toHaveLength(6)

    fireEvent.click(screen.getByRole('button', { name: 'prompt 0' }))

    expect(cancelReveal).toHaveBeenCalledOnce()
    expect(revealMessage).toHaveBeenCalledWith('u0')
    expect(cancelReveal.mock.invocationCallOrder[0]).toBeLessThan(revealMessage.mock.invocationCallOrder[0])

    fireEvent.click(screen.getByRole('button', { name: 'prompt 1' }))

    expect(cancelReveal).toHaveBeenCalledTimes(2)
    expect(revealMessage).toHaveBeenNthCalledWith(2, 'u1')
    expect(cancelReveal.mock.invocationCallOrder[1]).toBeLessThan(revealMessage.mock.invocationCallOrder[1])
  })

  it('scrolls to the prompt after the window materializes it', () => {
    messages = transcript(6).slice(4)
    const cancelReveal = vi.fn()
    const revealMessage = vi.fn()
    const timelineEntries = transcript(6).map((message, index) => ({ id: message.id, preview: `prompt ${index}` }))
    const surface = globalThis.document.createElement('div')
    const viewport = globalThis.document.createElement('div')
    const host = globalThis.document.createElement('div')

    vi.stubGlobal('CSS', { escape: (value: string) => value })

    const frames: FrameRequestCallback[] = []
    const frame = vi.spyOn(window, 'requestAnimationFrame').mockImplementation(callback => {
      frames.push(callback)

      return frames.length
    })
    const revealDom = vi.fn(() => true)
    const unsubscribeReveal = onRevealMessageRequest(revealDom)

    surface.dataset.sessionAnchor = 'timeline-test'
    viewport.dataset.slot = 'aui_thread-viewport'
    viewport.getBoundingClientRect = () => ({
      bottom: 400,
      height: 400,
      left: 0,
      right: 800,
      top: 0,
      width: 800,
      x: 0,
      y: 0,
      toJSON: () => ({})
    })
    surface.append(viewport, host)
    globalThis.document.body.append(surface)

    const { rerender } = render(
      <TranscriptWindowProvider
        value={{
          cancelReveal,
          olderAvailable: true,
          expandWindow: vi.fn(),
          revealMessage,
          revealScope: 'runtime-1',
          timelineEntries
        }}
      >
        <ThreadTimeline />
      </TranscriptWindowProvider>,
      { container: host }
    )

    fireEvent.click(screen.getByRole('button', { name: 'prompt 0' }))
    expect(cancelReveal).toHaveBeenCalledOnce()
    expect(revealMessage).toHaveBeenCalledWith('u0')

    const target = globalThis.document.createElement('div')
    target.dataset.messageId = 'u0'
    target.getBoundingClientRect = () => ({
      bottom: 200,
      height: 100,
      left: 0,
      right: 700,
      top: 100,
      width: 700,
      x: 0,
      y: 100,
      toJSON: () => ({})
    })
    viewport.append(target)
    messages = [userTurn('u0', 'prompt 0'), userTurn('u1', 'prompt 1')]
    rerender(
      <TranscriptWindowProvider
        value={{
          cancelReveal,
          olderAvailable: true,
          expandWindow: vi.fn(),
          revealMessage,
          revealScope: 'runtime-1',
          timelineEntries
        }}
      >
        <ThreadTimeline />
      </TranscriptWindowProvider>
    )

    act(() => {
      while (frames.length) {
        frames.shift()?.(performance.now() + 200)
      }
    })

    expect(revealDom).toHaveBeenCalledWith('runtime-1', 'u0')
    expect(viewport.scrollTop).toBe(92)

    fireEvent.click(screen.getByRole('button', { name: 'prompt 0' }))

    expect(cancelReveal).toHaveBeenCalledTimes(2)
    expect(revealMessage).toHaveBeenCalledOnce()
    unsubscribeReveal()
    frame.mockRestore()
  })

  it('cancels a pending hidden-target jump when the runtime scope changes', () => {
    messages = transcript(6).slice(4)
    const timelineEntries = transcript(6).map((message, index) => ({ id: message.id, preview: `prompt ${index}` }))
    const frames: FrameRequestCallback[] = []
    const frame = vi.spyOn(window, 'requestAnimationFrame').mockImplementation(callback => {
      frames.push(callback)

      return frames.length
    })
    const cancelFrame = vi.spyOn(window, 'cancelAnimationFrame')
    const value = (revealScope: string) => ({
      olderAvailable: true,
      expandWindow: vi.fn(),
      revealMessage: vi.fn(),
      revealScope,
      timelineEntries
    })
    const { rerender } = renderTimeline(
      <TranscriptWindowProvider value={value('runtime-1')}>
        <ThreadTimeline />
      </TranscriptWindowProvider>
    )

    fireEvent.click(screen.getByRole('button', { name: 'prompt 0' }))
    expect(frames).toHaveLength(1)
    const staleFrame = frames[0]

    rerender(
      <TranscriptWindowProvider value={value('runtime-2')}>
        <ThreadTimeline />
      </TranscriptWindowProvider>
    )

    expect(cancelFrame).toHaveBeenCalledWith(1)
    act(() => staleFrame(performance.now() + 200))
    expect(frames).toHaveLength(1)
    frame.mockRestore()
    cancelFrame.mockRestore()
  })
})

describe('ThreadTimeline below the threshold', () => {
  it('renders nothing for a short thread', () => {
    messages = transcript(2)

    const { container } = renderTimeline()

    expect(container.querySelector('[data-slot="thread-timeline"]')).toBeNull()
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
