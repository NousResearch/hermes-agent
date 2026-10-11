import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, onTestFinished, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import { onScrollToBottomRequest } from '@/store/thread-scroll'

import { TIMELINE_REVEAL_EVENT, type TimelineRevealRequest } from './timeline-data'

const view = {
  $messages: atom<ChatMessage[]>([]),
  $runtimeId: atom<string | null>('session-1'),
  $storedId: atom<string | null>('session-1')
}

let visible = true
let isHistoricalWindow = false

vi.mock('@/app/chat/session-view', () => ({ useSessionView: () => view }))
vi.mock('@/app/chat/composer/scope', () => ({ useComposerSurfaceId: () => 'surface-1' }))
vi.mock('@/components/pane-shell/pane-visibility', () => ({ usePaneVisible: () => visible }))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('./transcript-window', () => ({
  useTranscriptWindow: () => ({
    isHistorical: isHistoricalWindow,
    currentMessages: isHistoricalWindow ? view.$messages.get() : undefined,
    olderAvailable: false,
    expandWindow: async () => {},
    returnToLatest: vi.fn()
  })
}))
vi.mock('./use-timeline-history', () => ({
  useTimelineHistory: () => ({ entries: undefined, complete: true, failed: false, loadMore: async () => {} })
}))
vi.mock('@assistant-ui/react', () => ({
  useAui: () => ({ thread: () => ({ getState: () => ({ messages: [] }) }) }),
  useAuiState: (selector: (state: { thread: { messages: ChatMessage[] } }) => unknown) =>
    selector({ thread: { messages: view.$messages.get() } })
}))
vi.mock('./timeline-rail', () => ({
  TimelineRail: ({
    activeIndex,
    entries,
    onJump
  }: {
    activeIndex: number
    entries: { id: string }[]
    onJump: (id: string) => void
  }) => (
    <div data-testid="timeline-rail">
      {entries.map(entry => (
        <button data-testid={`jump-${entry.id}`} key={entry.id} onClick={() => onJump(entry.id)}>
          {entry.id}
        </button>
      ))}
      <output data-testid="active">{entries[activeIndex]?.id}</output>
    </div>
  )
}))

const { ThreadTimeline } = await import('./timeline')

beforeEach(() => {
  visible = true
  isHistoricalWindow = false
  view.$runtimeId.set('session-1')
  view.$storedId.set('session-1')
  view.$messages.set([
    { id: 'u0', role: 'user', parts: [{ type: 'text', text: 'first prompt' }] },
    { id: 'u1', role: 'user', parts: [{ type: 'text', text: 'second prompt' }] },
    { id: 'u2', role: 'user', parts: [{ type: 'text', text: 'newest prompt' }] }
  ])
})

afterEach(() => {
  cleanup()
  window.document.body.innerHTML = ''
  vi.restoreAllMocks()
})

describe('ThreadTimeline newest prompt jump', () => {
  it.each([
    { runtimeId: 'session-1', storedId: 'stored-1' },
    { runtimeId: null, storedId: 'stored-1' },
    { runtimeId: null, storedId: null }
  ])('routes the newest entry to its listener for $runtimeId / $storedId', async ({ runtimeId, storedId }) => {
    view.$runtimeId.set(runtimeId)
    view.$storedId.set(storedId)
    const bottomRequested = vi.fn()
    const otherSessionRequested = vi.fn()
    onTestFinished(onScrollToBottomRequest(bottomRequested, runtimeId ?? 'surface-1'))
    onTestFinished(onScrollToBottomRequest(otherSessionRequested, 'stored-1'))

    const ui = render(<ThreadTimeline />)

    const newestBtn = ui.getByTestId('jump-u2')
    await act(async () => {
      fireEvent.click(newestBtn)
    })

    expect(bottomRequested).toHaveBeenCalledTimes(1)
    expect(otherSessionRequested).not.toHaveBeenCalled()
  })

  it.each([
    { id: 'u0', historical: false },
    { id: 'u2', historical: true }
  ])('reveals $id in historical=$historical instead of jumping to bottom', async ({ id, historical }) => {
    isHistoricalWindow = historical
    const bottomRequested = vi.fn()
    onTestFinished(onScrollToBottomRequest(bottomRequested, 'session-1'))

    const host = window.document.createElement('div')
    host.innerHTML = `
      <div data-session-anchor="session-1">
        <div data-slot="aui_thread-viewport" style="height: 500px; overflow: auto;">
          <div data-slot="aui_turn-pair">
            <div data-message-id="u0">turn 0</div>
          </div>
          <div data-slot="aui_turn-pair">
            <div data-message-id="u1">turn 1</div>
          </div>
          <div data-slot="aui_turn-pair">
            <div data-message-id="u2">turn 2</div>
          </div>
        </div>
        <div data-testid="timeline-host"></div>
      </div>
    `
    window.document.body.appendChild(host)

    const viewport = host.querySelector<HTMLElement>('[data-slot="aui_thread-viewport"]')!

    const reveal = vi.fn((event: Event) => {
      const { detail } = event as CustomEvent<TimelineRevealRequest>
      detail.complete(false)
    })

    viewport.addEventListener(TIMELINE_REVEAL_EVENT, reveal)

    const ui = render(<ThreadTimeline />, {
      container: host.querySelector<HTMLElement>('[data-testid="timeline-host"]')!
    })

    const earlierBtn = ui.getByTestId(`jump-${id}`)
    await act(async () => {
      fireEvent.click(earlierBtn)
    })

    expect(reveal).toHaveBeenCalledTimes(1)
    expect(reveal.mock.calls[0][0]).toMatchObject({ detail: { id } })
    expect(bottomRequested).not.toHaveBeenCalled()
  })
})
