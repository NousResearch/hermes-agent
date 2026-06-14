import { render } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ThreadMessageList } from './thread-list'

type MockMessage = {
  content: unknown[]
  id: string
  role: 'assistant' | 'user'
}

let mockedMessages: MockMessage[] = []

let stickState: {
  contentRef: { current: HTMLDivElement | null }
  isAtBottom: boolean
  scrollRef: { current: HTMLDivElement | null }
  scrollToBottom: ReturnType<typeof vi.fn>
  stopScroll: ReturnType<typeof vi.fn>
}

vi.mock('@assistant-ui/react', () => ({
  ThreadPrimitive: {
    MessageByIndex: ({ index }: { index: number }) => <div data-testid={`msg-${index}`}>message {index}</div>
  },
  useAuiEvent: () => {},
  useAuiState: (selector: (state: { thread: { messages: MockMessage[] } }) => unknown) =>
    selector({ thread: { messages: mockedMessages } })
}))

vi.mock('use-stick-to-bottom', () => ({
  useStickToBottom: () => stickState
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      assistant: {
        thread: {
          showEarlier: 'Show earlier'
        }
      }
    }
  })
}))

vi.mock('@/lib/utils', () => ({
  cn: (...parts: Array<false | null | string | undefined>) => parts.filter(Boolean).join(' ')
}))

vi.mock('@/store/thread-scroll', () => ({
  onScrollToBottomRequest: () => () => {},
  onThreadEditClose: () => () => {},
  onThreadEditOpen: () => () => {},
  resetThreadScroll: () => {},
  setThreadAtBottom: () => {}
}))

vi.mock('./message-render-boundary', () => ({
  MessageRenderBoundary: ({ children }: { children: ReactNode }) => children
}))

describe('ThreadMessageList session settle', () => {
  beforeEach(() => {
    mockedMessages = [
      { content: [{ text: 'prompt', type: 'text' }], id: 'user-1', role: 'user' },
      { content: [{ text: 'reply', type: 'text' }], id: 'assistant-1', role: 'assistant' }
    ]

    stickState = {
      contentRef: { current: null },
      isAtBottom: true,
      scrollRef: { current: null },
      scrollToBottom: vi.fn(),
      stopScroll: vi.fn()
    }

    vi.stubGlobal('requestAnimationFrame', vi.fn(() => 1))
    vi.stubGlobal('cancelAnimationFrame', vi.fn())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('does not restart session settle on ordinary rerenders', () => {
    const initialStop = stickState.stopScroll
    const sharedScrollRef = stickState.scrollRef
    const sharedContentRef = stickState.contentRef

    const { rerender } = render(
      <ThreadMessageList clampToComposer={false} components={{}} loadingIndicator={null} sessionKey="session-a" />
    )

    expect(initialStop).toHaveBeenCalledTimes(1)

    stickState = {
      contentRef: sharedContentRef,
      isAtBottom: false,
      scrollRef: sharedScrollRef,
      scrollToBottom: vi.fn(),
      stopScroll: vi.fn()
    }

    rerender(<ThreadMessageList clampToComposer={false} components={{}} loadingIndicator={<div />} sessionKey="session-a" />)

    expect(stickState.stopScroll).not.toHaveBeenCalled()
  })

  it('restarts session settle when the session key changes', () => {
    const sharedScrollRef = stickState.scrollRef
    const sharedContentRef = stickState.contentRef

    const { rerender } = render(
      <ThreadMessageList clampToComposer={false} components={{}} loadingIndicator={null} sessionKey="session-a" />
    )

    stickState = {
      contentRef: sharedContentRef,
      isAtBottom: true,
      scrollRef: sharedScrollRef,
      scrollToBottom: vi.fn(),
      stopScroll: vi.fn()
    }

    rerender(<ThreadMessageList clampToComposer={false} components={{}} loadingIndicator={null} sessionKey="session-b" />)

    expect(stickState.stopScroll).toHaveBeenCalledTimes(1)
  })
})
