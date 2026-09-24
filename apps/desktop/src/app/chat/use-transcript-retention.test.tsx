import { act, render } from '@testing-library/react'
import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { stubThreadEnvironment } from '@/components/assistant-ui/test-utils'
import { type TranscriptWindowValue, useTranscriptWindow } from '@/components/assistant-ui/thread/transcript-window'
import type * as HermesApi from '@/hermes'
import type { ChatMessage } from '@/lib/chat-messages'
import { RENDER_WEIGHT_CHARS } from '@/lib/render-weight'
import type * as SessionStates from '@/store/session-states'
import { $transcriptTailBySessionId, recordTranscriptTail } from '@/store/transcript-tail'

import { PRIMARY_SESSION_VIEW, SessionViewProvider } from './session-view'
import { _resetTranscriptBackfillForTests } from './transcript-backfill'
import { useTranscriptRetention } from './use-transcript-retention'

import { ChatRuntimeBoundary } from '.'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getOlderSessionMessages: vi.fn()
}))
vi.mock('@/store/session-states', async importOriginal => ({
  ...(await importOriginal<typeof SessionStates>()),
  sessionTileDelegate: vi.fn()
}))
const { getOlderSessionMessages } = await import('@/hermes')
const { sessionTileDelegate } = await import('@/store/session-states')
stubThreadEnvironment()

/** A persisted row heavy enough that a dozen of them fill one window page.
 *  `serverRowSpan` models the hydration fold: each message stands for three
 *  backend rows, so the rewind has to convert. */
const row = (index: number): ChatMessage => ({
  id: `m${index}`,
  parts: [{ type: 'text', text: 'x'.repeat(RENDER_WEIGHT_CHARS * 100) }],
  role: 'assistant',
  rowId: index + 1,
  serverRowSpan: 3
})

const HYDRATED_ROWS = 120
/** Backend rows each store message stands for (see `serverRowSpan` above). */
const SERVER_ROWS_PER_MESSAGE = 3

function sessionView(messages: ChatMessage[]) {
  const $messages = atom(messages)

  return {
    $messages,
    view: {
      ...PRIMARY_SESSION_VIEW,
      $messages,
      // A held reference is not needed; the view only reads these.
      $runtimeId: atom<string | null>('runtime'),
      $storedId: atom<string | null>('stored')
    }
  }
}

function mountBoundary(view: ReturnType<typeof sessionView>['view'], onWindow: (value: TranscriptWindowValue) => void) {
  function Observe() {
    onWindow(useTranscriptWindow())

    return null
  }

  render(
    <SessionViewProvider value={view}>
      <ChatRuntimeBoundary
        busy={false}
        onCancel={() => {}}
        onEdit={async () => {}}
        onReload={async () => {}}
        onThreadMessagesChange={() => {}}
        suppressMessages={false}
      >
        <Observe />
      </ChatRuntimeBoundary>
    </SessionViewProvider>
  )
}

beforeEach(() => {
  $transcriptTailBySessionId.set({})
  _resetTranscriptBackfillForTests()
  vi.mocked(getOlderSessionMessages).mockReset()
})

describe('useTranscriptRetention — the store releases paged-through history', () => {
  it('drops the rows older than the live window, then fetches them back on "Show earlier"', async () => {
    const { $messages, view } = sessionView(Array.from({ length: 60 }, (_, index) => row(index)))

    recordTranscriptTail('stored', {
      messages: Array.from({ length: HYDRATED_ROWS }, (_, index) => ({ id: index, role: 'user', content: 'tail' })),
      pagination: { limit: HYDRATED_ROWS, offset: 0, order: 'latest', returned: HYDRATED_ROWS }
    })

    let state = { messages: $messages.get() }
    vi.mocked(sessionTileDelegate).mockReturnValue({
      updateSession: (_id: string, update: (previous: { messages: ChatMessage[] }) => { messages: ChatMessage[] }) => {
        state = update(state)

        if (state.messages !== $messages.get()) {
          $messages.set(state.messages)
        }

        return state
      }
    } as never)

    let window!: TranscriptWindowValue

    mountBoundary(view, value => (window = value))

    // The window cuts the transcript; everything older than it plus one page of
    // slack is released, and the released rows stay reachable over REST.
    const released = 60 - state.messages.length
    // Where the rewind points the next older page: the hydrated offset minus the
    // BACKEND rows the store gave up (three per message here).
    const rewoundOffset = HYDRATED_ROWS - released * SERVER_ROWS_PER_MESSAGE

    expect(released).toBeGreaterThan(1)
    expect($messages.get()).toHaveLength(state.messages.length)
    expect($transcriptTailBySessionId.get().stored).toMatchObject({
      nextOffset: rewoundOffset,
      possiblyTruncated: true
    })

    // The first "Show earlier" pages the retained store; the second one — with
    // the in-memory transcript fully materialized — fetches REST.
    vi.mocked(getOlderSessionMessages).mockResolvedValue({
      messages: Array.from({ length: released }, (_, index) => ({
        id: index + 1,
        content: 'restored',
        role: 'user',
        timestamp: index
      })),
      pagination: { limit: HYDRATED_ROWS, offset: rewoundOffset, order: 'latest', returned: released },
      session_id: 'stored'
    } as never)

    // Each click first pages what the store still retains; once that is fully
    // materialized the click that follows fetches the released rows over REST.
    for (let click = 0; click < 8 && vi.mocked(getOlderSessionMessages).mock.calls.length === 0; click += 1) {
      await act(async () => {
        expect(await window.expandWindow()).toBe(true)
      })
    }

    // Re-hydration read the released rows back from exactly where the rewind
    // said they were.
    expect(vi.mocked(getOlderSessionMessages)).toHaveBeenCalledTimes(1)
    expect(vi.mocked(getOlderSessionMessages).mock.calls[0][2]).toBe(rewoundOffset)

    // ...and the released history is back in chronological order.
    expect($messages.get()).toHaveLength(60)
  })

  it('leaves a transcript that fits its window alone', () => {
    const { $messages, view } = sessionView([row(0), row(1)])
    const updateSession = vi.fn()

    vi.mocked(sessionTileDelegate).mockReturnValue({ updateSession } as never)
    mountBoundary(view, () => {})

    expect(updateSession).not.toHaveBeenCalled()
    expect($messages.get()).toHaveLength(2)
  })

  it('does not release anything for a session with no recorded page route', () => {
    const { $messages, view } = sessionView(Array.from({ length: 60 }, (_, index) => row(index)))
    let state = { messages: $messages.get() }

    const updateSession = vi.fn(
      (_id: string, update: (previous: typeof state) => typeof state) => (state = update(state))
    )

    vi.mocked(sessionTileDelegate).mockReturnValue({ updateSession } as never)
    mountBoundary(view, () => {})

    // `beforeEach` left the tail store empty: nothing can fetch released rows,
    // so the plan is never even computed and they stay.
    expect(updateSession).not.toHaveBeenCalled()
    expect(state.messages).toHaveLength(60)
  })
})

describe('useTranscriptRetention — direct', () => {
  it('releases on the anchor it is given, and only when the anchor moves', () => {
    const messages = Array.from({ length: 60 }, (_, index) => row(index))
    let state = { messages }

    const updateSession = vi.fn((_id: string, update: (previous: typeof state) => typeof state) => {
      state = update(state)

      return state
    })

    vi.mocked(sessionTileDelegate).mockReturnValue({ updateSession } as never)
    recordTranscriptTail('stored', {
      messages: Array.from({ length: HYDRATED_ROWS }, (_, index) => ({ id: index, role: 'user', content: 'tail' })),
      pagination: { limit: HYDRATED_ROWS, offset: 0, order: 'latest', returned: HYDRATED_ROWS }
    })

    const { rerender } = render(<Harness anchorId={null} enabled runtimeId="runtime" storedSessionId="stored" />)

    expect(updateSession).not.toHaveBeenCalled()
    expect(state.messages).toHaveLength(60)

    rerender(<Harness anchorId="m48" enabled runtimeId="runtime" storedSessionId="stored" />)

    expect(updateSession).toHaveBeenCalledTimes(1)
    expect(state.messages.length).toBeLessThan(60)
    expect($transcriptTailBySessionId.get().stored).toMatchObject({
      nextOffset: HYDRATED_ROWS - (60 - state.messages.length) * SERVER_ROWS_PER_MESSAGE,
      possiblyTruncated: true
    })
  })
})

function Harness(props: Parameters<typeof useTranscriptRetention>[0]) {
  useTranscriptRetention(props)

  return null
}
