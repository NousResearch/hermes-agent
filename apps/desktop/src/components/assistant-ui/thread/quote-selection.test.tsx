import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { requestComposerFocus, requestComposerInsert } from '@/app/chat/composer/focus'

import { QuoteSelectionContextMenu, quoteSelectedText, selectedTextWithin } from './quote-selection'

vi.mock('@/app/chat/composer/focus', () => ({
  requestComposerFocus: vi.fn(),
  requestComposerInsert: vi.fn()
}))

afterEach(() => {
  cleanup()
  window.getSelection()?.removeAllRanges()
  vi.clearAllMocks()
})

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

describe('quoteSelectedText', () => {
  it('prefixes every selected line, including blank lines', () => {
    expect(quoteSelectedText('first\n\nthird\r\nfourth')).toBe('> first\n> \n> third\n> fourth')
  })
})

describe('QuoteSelectionContextMenu', () => {
  it('recognizes a selection inside the message so other context actions can yield to quote', () => {
    render(<div data-testid="message">selected text</div>)

    const message = screen.getByTestId('message')
    selectText(message)

    expect(selectedTextWithin(message)).toBe('selected text')
  })

  it('inserts the selected message text into the active composer', async () => {
    render(
      <QuoteSelectionContextMenu>
        <div data-testid="message">first line{`\n`}second line</div>
      </QuoteSelectionContextMenu>
    )

    const message = screen.getByTestId('message')
    selectText(message)
    openContextMenu(message)

    fireEvent.click(await screen.findByRole('menuitem', { name: 'Quote in new message' }))

    expect(requestComposerInsert).toHaveBeenCalledWith('> first line\n> second line', {
      mode: 'block',
      target: 'active'
    })
    expect(requestComposerFocus).toHaveBeenCalledWith('active')
  })
})
