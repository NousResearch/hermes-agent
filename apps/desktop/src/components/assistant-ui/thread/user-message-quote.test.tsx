import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ReactNode } from 'react'

import { requestComposerFocus, requestComposerInsert } from '@/app/chat/composer/focus'

import { Thread } from '.'

vi.mock('@/app/chat/composer/focus', () => ({
  requestComposerFocus: vi.fn(),
  requestComposerInsert: vi.fn()
}))

vi.mock('@/components/assistant-ui/thread/message-reactions', () => ({
  ReactionBadge: () => null,
  ReactionPicker: ({ children, open }: { children: ReactNode; open: boolean }) => (
    <div data-open={open ? 'true' : 'false'} data-testid="reaction-picker">
      {children}
    </div>
  )
}))

vi.mock('@/components/assistant-ui/thread/use-message-reactions', () => ({
  useMessageReactions: () => ({
    enabled: true,
    react: vi.fn(),
    reactions: []
  }),
  useTapbackDoubleClick: () => undefined
}))

const createdAt = new Date('2026-05-01T00:00:00.000Z')

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal('ResizeObserver', TestResizeObserver)
vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
  window.setTimeout(() => callback(performance.now()), 0)
)
vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))
vi.stubGlobal('CSS', { escape: (str: string) => str })

Element.prototype.scrollTo = function scrollTo() {}

afterEach(() => {
  cleanup()
  window.getSelection()?.removeAllRanges()
  vi.clearAllMocks()
})

function userMessage(): ThreadMessage {
  return {
    id: 'user-1',
    role: 'user',
    content: [{ type: 'text', text: 'quote this message' }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage
}

function assistantMessage(): ThreadMessage {
  return {
    id: 'assistant-1',
    role: 'assistant',
    content: [{ type: 'text', text: 'done' }],
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as ThreadMessage
}

function Harness() {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [userMessage(), assistantMessage()],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

function selectText(target: HTMLElement) {
  const selection = window.getSelection()
  const range = document.createRange()

  range.selectNodeContents(target)
  selection?.removeAllRanges()
  selection?.addRange(range)
}

function openContextMenu(target: HTMLElement) {
  fireEvent.pointerDown(target, { button: 2, pointerType: 'mouse' })
  fireEvent.contextMenu(target, { button: 2 })
}

describe('UserMessage quote/reaction context-menu integration', () => {
  // The user message renders first, so its picker is the first of the mocked
  // ReactionPicker testids (the assistant message renders one too).
  const userPicker = () => screen.getAllByTestId('reaction-picker')[0]

  it('lets a local selection open quote while reactions are enabled', async () => {
    render(<Harness />)

    const bubble = await screen.findByRole('button', { name: 'Edit message' })
    selectText(bubble)
    openContextMenu(bubble)

    expect(userPicker().getAttribute('data-open')).toBe('false')

    fireEvent.click(await screen.findByRole('menuitem', { name: 'Quote in new message' }))

    expect(requestComposerInsert).toHaveBeenCalledWith('> quote this message', {
      mode: 'block',
      target: 'active'
    })
    expect(requestComposerFocus).toHaveBeenCalledWith('active')
  })

  it('opens reactions for an unselected message when reactions are enabled', async () => {
    render(<Harness />)

    const bubble = await screen.findByRole('button', { name: 'Edit message' })
    openContextMenu(bubble)

    expect(userPicker().getAttribute('data-open')).toBe('true')
    expect(screen.queryByRole('menuitem', { name: 'Quote in new message' })).toBeNull()
  })
})
