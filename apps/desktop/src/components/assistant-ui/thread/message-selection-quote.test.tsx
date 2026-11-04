// The floating "Quote in chat" button must survive the click that selects it:
// a default pointerdown/mousedown on the button collapses the document
// selection before onClick fires, and the quote would silently vanish. The
// button prevents that default and falls back to the text captured at
// pointerdown if the selection is gone anyway.
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/app/chat/composer/focus', () => ({
  requestComposerInsert: vi.fn(),
}))

vi.mock('@/lib/haptics', () => ({
  triggerHaptic: vi.fn(),
}))

const { requestComposerInsert } = await import('@/app/chat/composer/focus')
const { MessageQuoteButton } = await import('./message-selection-quote')
const { clearComposerMessageQuotes } = await import('@/store/composer')

function fakeSelection(text: string, collapsed: boolean, rootNode: Node) {
  return {
    isCollapsed: collapsed,
    toString: () => text,
    anchorNode: rootNode,
    focusNode: rootNode,
    getRangeAt: () => ({
      getBoundingClientRect: () => ({ top: 10, left: 10, width: 50, height: 10 })
    }),
    removeAllRanges: vi.fn()
  }
}

afterEach(() => {
  cleanup()
  clearComposerMessageQuotes()
  vi.mocked(requestComposerInsert).mockClear()
  vi.restoreAllMocks()
})

describe('MessageQuoteButton', () => {
  it('quotes the selection on click even when the click collapsed it', async () => {
    const { container } = render(
      <div data-slot="aui_assistant-message-root">
        <span>selected text</span>
        <MessageQuoteButton messageId="msg-1" />
      </div>
    )

    const root = container.querySelector('[data-slot="aui_assistant-message-root"]')!

    let sel: ReturnType<typeof fakeSelection> | null = fakeSelection('selected text', false, root)
    vi.spyOn(window, 'getSelection').mockImplementation(() => sel as unknown as Selection)

    window.document.dispatchEvent(new Event('selectionchange'))
    const button = await screen.findByText('Quote in chat')

    // Pointerdown captures the snapshot while the selection is alive…
    fireEvent.pointerDown(button)
    // …then the click-time selection has collapsed (the browser default the
    // button suppresses — reproduced here to exercise the fallback).
    sel = fakeSelection('', true, root)
    fireEvent.click(button)

    expect(requestComposerInsert).toHaveBeenCalledWith('@message:msg-1', {
      mode: 'inline',
    })
  })

  it('does not quote anything when there was no selection at all', async () => {
    const { container } = render(
      <div data-slot="aui_assistant-message-root">
        <span>selected text</span>
        <MessageQuoteButton messageId="msg-1" />
      </div>
    )

    const root = container.querySelector('[data-slot="aui_assistant-message-root"]')!

    const sel = fakeSelection('', true, root)
    vi.spyOn(window, 'getSelection').mockImplementation(() => sel as unknown as Selection)

    window.document.dispatchEvent(new Event('selectionchange'))

    // The button never shows (selection collapsed), so there is nothing to
    // click — and even a programmatic quote path would find no text.
    expect(screen.queryByText('Quote in chat')).toBeNull()
    expect(requestComposerInsert).not.toHaveBeenCalled()
    expect(root).toBeTruthy()
  })
})
