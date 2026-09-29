// Double-click an assistant reply to heart it (the iMessage gesture), gated on
// the same opt-in toggle as the rest of message reactions.
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PRIMARY_SESSION_VIEW, SessionViewProvider } from '@/app/chat/session-view'
import type * as ReactionsStore from '@/store/reactions'
import { $reactionsEnabled } from '@/store/reactions-enabled'
import { $localReactions, localReactionKey } from '@/store/reactions-local'

import { assistantMessage, stubThreadEnvironment } from '../test-utils'

import { isTapbackDoubleClick } from './use-message-reactions'

import { Thread } from '.'
stubThreadEnvironment()

// The gesture persists through the gateway; this suite is about the local
// paint, which is what the user actually sees on the click.
vi.mock('@/store/reactions', async importOriginal => ({
  ...(await importOriginal<typeof ReactionsStore>()),
  toggleMessageReaction: vi.fn(async () => {})
}))

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

const key = (messageId: string) => localReactionKey(PRIMARY_SESSION_VIEW.$storedId.get(), messageId)

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
    await waitFor(() => expect($localReactions.get()[key('assistant-1')]?.[0]?.emoji).toBe('❤️'))

    fireEvent.doubleClick(message!, { detail: 2 })
    await waitFor(() => expect($localReactions.get()[key('assistant-1')]).toEqual([]))
  })

  it('does nothing while reactions are off', async () => {
    render(<Harness />)

    const message = (await screen.findByText('done')).closest('[data-slot="aui_assistant-message-root"]')

    fireEvent.doubleClick(message!, { detail: 2 })

    expect($localReactions.get()[key('assistant-1')]).toBeUndefined()
  })

  it('keeps a tapback on the session it was made in', async () => {
    // Persisted rows render as `row-<messages.id>`, and every profile's
    // state.db numbers its rows from 1: two bots' chats both hold the same id.
    $reactionsEnabled.set(true)

    const view = (storedId: string) => ({ ...PRIMARY_SESSION_VIEW, $storedId: atom<null | string>(storedId) })

    render(
      <>
        <SessionViewProvider value={view('bot-a-chat')}>
          <Harness />
        </SessionViewProvider>
        <SessionViewProvider value={view('bot-b-chat')}>
          <Harness />
        </SessionViewProvider>
      </>
    )

    const [first] = await screen.findAllByText('done')

    fireEvent.doubleClick(first.closest('[data-slot="aui_assistant-message-root"]')!, { detail: 2 })

    await waitFor(() =>
      expect($localReactions.get()[localReactionKey('bot-a-chat', 'assistant-1')]?.[0]?.emoji).toBe('❤️')
    )
    expect($localReactions.get()[localReactionKey('bot-b-chat', 'assistant-1')]).toBeUndefined()
  })
})
