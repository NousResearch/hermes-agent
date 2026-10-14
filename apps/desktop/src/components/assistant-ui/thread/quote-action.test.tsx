import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it } from 'vitest'

import { QuoteAction } from './quote-action'

type Descriptor = PropertyDescriptor | undefined

const rangeDescriptors: Record<string, Descriptor> = {}

function transcript(entries: Array<{ id: string; role: 'assistant' | 'user'; text: string }>) {
  window.document.body.insertAdjacentHTML(
    'beforeend',
    `<div data-slot="aui_thread-viewport">${entries
      .map(
        entry =>
          `<div data-slot="aui_${entry.role}-message-root" data-role="${entry.role}">` +
          `<div data-slot="aui_${entry.role}-message-content"><p id="${entry.id}">${entry.text}</p></div></div>`
      )
      .join('')}</div>`
  )
}

function selectContents(id: string) {
  const range = window.document.createRange()

  range.selectNodeContents(window.document.getElementById(id)!.firstChild!)

  const selection = window.getSelection()!

  selection.removeAllRanges()
  selection.addRange(range)
}

/** jsdom lays nothing out, so the range reports no rects — the anchor the action
 *  positions itself from. Stub the two calls the component reads. */
beforeEach(() => {
  for (const key of ['getClientRects', 'getBoundingClientRect'] as const) {
    rangeDescriptors[key] = Object.getOwnPropertyDescriptor(Range.prototype, key)

    Object.defineProperty(Range.prototype, key, {
      configurable: true,
      value: () => [] as unknown as DOMRectList
    })
  }

  Object.defineProperty(Range.prototype, 'getClientRects', {
    configurable: true,
    value: () => [{ bottom: 138, height: 18, left: 40, right: 240, top: 120, width: 200 } as DOMRect]
  })
})

afterEach(() => {
  cleanup()
  window.getSelection()?.removeAllRanges()
  window.document.body.innerHTML = ''

  for (const [key, descriptor] of Object.entries(rangeDescriptors)) {
    if (descriptor) {
      Object.defineProperty(Range.prototype, key, descriptor)
    } else {
      Reflect.deleteProperty(Range.prototype, key)
    }
  }
})

function publishSelection() {
  window.document.dispatchEvent(new Event('selectionchange'))
}

it('quotes a transcript selection into the composer as a bracketed blockquote', async () => {
  const inserts: Array<{ mode?: string; target?: string; text: string }> = []
  const onInsert = (event: Event) => inserts.push((event as CustomEvent<{ text: string }>).detail)

  window.addEventListener('hermes:composer-insert', onInsert)

  try {
    render(<QuoteAction />)

    transcript([{ id: 'a1', role: 'assistant', text: 'The gateway stayed up.' }])
    selectContents('a1')
    publishSelection()

    const action = await screen.findByRole('button', { name: 'Add to chat' })

    // The press must not take the selection away from the transcript it quotes.
    const press = new MouseEvent('mousedown', { bubbles: true, cancelable: true })

    fireEvent(action, press)
    expect(press.defaultPrevented).toBe(true)

    await waitFor(() => expect(inserts).toHaveLength(1))
    expect(inserts[0]).toMatchObject({
      mode: 'block',
      text: '[Quoted from your previous message]\n> The gateway stayed up.'
    })

    // Once quoted, the action is done with this selection.
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Add to chat' })).toBeNull())
  } finally {
    window.removeEventListener('hermes:composer-insert', onInsert)
  }
})

it('stays hidden while the selection is not a passage from this transcript', async () => {
  render(<QuoteAction />)

  window.document.body.insertAdjacentHTML('beforeend', '<div contenteditable="true" id="draft">typing here</div>')
  selectContents('draft')
  publishSelection()

  await new Promise(resolve => window.setTimeout(resolve, 20))
  expect(screen.queryByRole('button', { name: 'Add to chat' })).toBeNull()
})
