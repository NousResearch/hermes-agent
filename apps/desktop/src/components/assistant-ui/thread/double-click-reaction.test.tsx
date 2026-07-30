// Double-click an assistant reply to heart it (the iMessage gesture), gated on
// the same opt-in toggle as the rest of message reactions.
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { Thread } from '.'
import { isTapbackDoubleClick } from './use-message-reactions'

import { $localReactions } from '@/store/reactions-local'
import { $reactionsEnabled } from '@/store/reactions-enabled'

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

// The gesture persists through the gateway; this suite is about the local
// paint, which is what the user actually sees on the click.
vi.mock('@/store/reactions', async importOriginal => ({
  ...(await importOriginal<typeof import('@/store/reactions')>()),
  toggleMessageReaction: vi.fn(async () => {})
}))

function assistantMessage(): ThreadMessage {
  return {
    id: 'assistant-1',
    role: 'assistant',
    content: [{ type: 'text', text: 'done' }],
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
  } as ThreadMessage
}

function Harness() {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [assistantMessage()],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

beforeEach(() => {
  $localReactions.set({})
  $reactionsEnabled.set(false)
})

afterEach(() => {
  cleanup()
})

describe('isTapbackDoubleClick', () => {
  it('claims a plain double-click on message body', () => {
    expect(isTapbackDoubleClick({ detail: 2, target: document.createElement('span') })).toBe(true)
  })

  it('ignores a triple-click, so selecting a paragraph does not re-toggle', () => {
    expect(isTapbackDoubleClick({ detail: 3, target: document.createElement('span') })).toBe(false)
  })

  it('leaves double-click alone where it already means something', () => {
    const code = document.createElement('pre')
    const inner = document.createElement('code')

    code.append(inner)

    expect(isTapbackDoubleClick({ detail: 2, target: inner })).toBe(false)
    expect(isTapbackDoubleClick({ detail: 2, target: document.createElement('a') })).toBe(false)
    expect(isTapbackDoubleClick({ detail: 2, target: document.createElement('button') })).toBe(false)
  })
})

describe('double-click to heart an assistant message', () => {
  it('hearts the message, and a second double-click retracts it', async () => {
    $reactionsEnabled.set(true)
    render(<Harness />)

    const message = (await screen.findByText('done')).closest('[data-slot="aui_assistant-message-root"]')

    expect(message).toBeTruthy()

    fireEvent.doubleClick(message!, { detail: 2 })
    await waitFor(() => expect($localReactions.get()['assistant-1']?.[0]?.emoji).toBe('❤️'))

    fireEvent.doubleClick(message!, { detail: 2 })
    await waitFor(() => expect($localReactions.get()['assistant-1']).toEqual([]))
  })

  it('does nothing while reactions are off', async () => {
    render(<Harness />)

    const message = (await screen.findByText('done')).closest('[data-slot="aui_assistant-message-root"]')

    fireEvent.doubleClick(message!, { detail: 2 })

    expect($localReactions.get()['assistant-1']).toBeUndefined()
  })
})
