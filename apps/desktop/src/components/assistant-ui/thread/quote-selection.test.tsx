import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { QuoteSelectionAction } from './quote-selection'

afterEach(() => {
  cleanup()
  window.getSelection()?.removeAllRanges()
  vi.restoreAllMocks()
})

function mount() {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <div data-slot="thread">
        <div data-role="assistant">
          <p data-testid="source">AAA 前者讨论方向 BBB</p>
        </div>
        <div data-role="user">
          <p data-testid="other">another message</p>
        </div>
        <QuoteSelectionAction />
      </div>
    </I18nProvider>
  )
}

function selectText(element: HTMLElement, start: number, end: number, endElement = element) {
  const doc = element.ownerDocument
  const range = doc.createRange()
  range.setStart(element.firstChild!, start)
  range.setEnd(endElement.firstChild!, end)
  Object.defineProperty(range, 'getBoundingClientRect', {
    value: () => ({ top: 100, bottom: 122, left: 40, right: 230, width: 190, height: 22 })
  })
  const selection = window.getSelection()!
  selection.removeAllRanges()
  selection.addRange(range)
  fireEvent(doc, new Event('selectionchange'))
}

describe('selected-message quote action', () => {
  it('offers Quote and passes only the selected substring to the composer', async () => {
    const { getByTestId, getByRole } = mount()
    const received: string[] = []

    const listener = (event: Event) => {
      const detail = (event as CustomEvent).detail
      received.push(detail.quote.text)
      window.dispatchEvent(
        new CustomEvent('hermes:composer-insert-reply', { detail: { token: detail.token, inserted: true } })
      )
    }

    window.addEventListener('hermes:composer-quote-add', listener)

    try {
      act(() => selectText(getByTestId('source'), 4, 10))
      const button = await waitFor(() => getByRole('button', { name: 'Quote' }))
      await act(async () => {
        fireEvent.mouseDown(button)
        fireEvent.click(button)
      })
      await waitFor(() => expect(received).toEqual(['前者讨论方向']))
    } finally {
      window.removeEventListener('hermes:composer-quote-add', listener)
    }
  })

  it('does not offer Quote for a selection spanning two messages', async () => {
    const { getByTestId, queryByRole } = mount()
    await act(async () => {
      selectText(getByTestId('source'), 4, 4, getByTestId('other'))
      await new Promise(resolve => requestAnimationFrame(resolve))
    })
    expect(queryByRole('button', { name: 'Quote' })).toBeNull()
  })
})
