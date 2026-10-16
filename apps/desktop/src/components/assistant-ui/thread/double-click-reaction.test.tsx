// Double-click an assistant reply to heart it (the iMessage gesture), gated on
// the same opt-in toggle as the rest of message reactions.
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as ReactionsStore from '@/store/reactions'
import { $reactionsEnabled } from '@/store/reactions-enabled'
import { $localReactions } from '@/store/reactions-local'

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

beforeEach(() => {
  $localReactions.set({})
  $reactionsEnabled.set(false)
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
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
  it('leaves touch double-tap selection intact instead of claiming it as a reaction', async () => {
    vi.stubGlobal(
      'matchMedia',
      vi.fn((query: string) => ({
        matches: query.includes('pointer'),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn()
      }))
    )
    $reactionsEnabled.set(true)
    render(<Harness />)
    const text = await screen.findByText('done')
    const range = document.createRange()
    range.selectNodeContents(text)
    window.getSelection()!.addRange(range)
    fireEvent.doubleClick(text, { detail: 2 })
    expect(window.getSelection()!.toString()).toBe('done')
    expect($localReactions.get()['assistant-1']).toBeUndefined()
    window.getSelection()!.removeAllRanges()
  })
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
