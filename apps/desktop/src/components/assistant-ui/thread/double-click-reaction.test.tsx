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
  delete window.document.documentElement.dataset.hermesDesktopHost
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

  it('leaves a double-click that dragged alone — that gesture is a text selection', () => {
    const target = window.document.createElement('span')

    // Press and release in the same place: the iMessage tapback.
    expect(
      isTapbackDoubleClick({ clientX: 100, clientY: 200, detail: 2, press: { x: 100, y: 200 }, target })
    ).toBe(true)

    // A couple of pixels of hand tremble is still a tap.
    expect(
      isTapbackDoubleClick({ clientX: 102, clientY: 201, detail: 2, press: { x: 100, y: 200 }, target })
    ).toBe(true)

    // Dragged across the paragraph to extend a highlight: not ours.
    expect(
      isTapbackDoubleClick({ clientX: 420, clientY: 200, detail: 2, press: { x: 100, y: 200 }, target })
    ).toBe(false)

    // Without a recorded press the old behaviour stands.
    expect(isTapbackDoubleClick({ clientX: 420, clientY: 200, detail: 2, press: null, target })).toBe(true)
  })
})

describe('double-click to heart an assistant message', () => {
  it('leaves touch double-tap selection intact instead of claiming it as a reaction', async () => {
    // Touch handling is the browser-hosted Desktop's.
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'
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

  it('keeps the highlight when the double-click dragged to extend a selection', async () => {
    $reactionsEnabled.set(true)
    render(<Harness />)

    const text = await screen.findByText('done')
    const message = text.closest('[data-slot="aui_assistant-message-root"]')

    // The gesture Ted makes: press on a word, drag to finish highlighting, release.
    fireEvent.mouseDown(text, { clientX: 120, clientY: 200 })
    const range = window.document.createRange()
    range.selectNodeContents(text)
    window.getSelection()!.addRange(range)
    fireEvent.doubleClick(text, { clientX: 420, clientY: 200, detail: 2 })

    expect(window.getSelection()!.toString()).toBe('done')
    expect($localReactions.get()['assistant-1']).toBeUndefined()
    window.getSelection()!.removeAllRanges()
  })

  it('still hearts a double-click that stayed put after a press', async () => {
    $reactionsEnabled.set(true)
    render(<Harness />)

    const text = await screen.findByText('done')
    const message = text.closest('[data-slot="aui_assistant-message-root"]')

    fireEvent.mouseDown(text, { clientX: 120, clientY: 200 })
    fireEvent.doubleClick(text, { clientX: 122, clientY: 201, detail: 2 })

    await waitFor(() => expect($localReactions.get()['assistant-1']?.[0]?.emoji).toBe('❤️'))
    expect(message).toBeTruthy()
  })
})
