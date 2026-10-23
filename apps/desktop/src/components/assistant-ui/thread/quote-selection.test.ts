import { afterEach, expect, it } from 'vitest'

import { captureQuoteSelection, normalizeQuoteText, quoteBlock, selectionAnchorRect } from './quote-selection'

type Descriptor = PropertyDescriptor | undefined

const savedRangeDescriptors: Record<string, Descriptor> = {}

function transcript(entries: Array<{ id: string; role: 'assistant' | 'user'; text: string }>) {
  document.body.insertAdjacentHTML(
    'beforeend',
    `<div data-slot="aui_thread-viewport" id="viewport">${entries
      .map(
        entry =>
          `<div data-slot="aui_${entry.role}-message-root" id="${entry.id}" data-role="${entry.role}">` +
          `<div data-slot="aui_${entry.role}-message-content"><p id="${entry.id}-p">${entry.text}</p></div></div>`
      )
      .join('')}</div>`
  )
}

function textNodeOf(id: string): Text {
  return document.getElementById(`${id}-p`)!.firstChild as Text
}

/** The selection a reader gets from dragging across one passage. */
function selectAcross(start: Node, startOffset: number, end: Node, endOffset: number) {
  const range = document.createRange()

  range.setStart(start, startOffset)
  range.setEnd(end, endOffset)

  const selection = window.getSelection()!

  selection.removeAllRanges()
  selection.addRange(range)
}

afterEach(() => {
  window.getSelection()?.removeAllRanges()
  document.body.innerHTML = ''

  for (const [key, descriptor] of Object.entries(savedRangeDescriptors)) {
    if (descriptor) {
      Object.defineProperty(Range.prototype, key, descriptor)
    } else {
      Reflect.deleteProperty(Range.prototype, key)
    }
  }

  for (const key of Object.keys(savedRangeDescriptors)) {
    delete savedRangeDescriptors[key]
  }
})

function stubRangeRect(rect: { height: number; left: number; top: number; width: number }) {
  savedRangeDescriptors.getClientRects ??= Object.getOwnPropertyDescriptor(Range.prototype, 'getClientRects')

  Object.defineProperty(Range.prototype, 'getClientRects', {
    configurable: true,
    value: () => [rect as DOMRect]
  })
}

it('captures a passage out of an assistant message and marks which side it came from', () => {
  transcript([{ id: 'a1', role: 'assistant', text: 'The gateway stayed up.' }])

  const node = textNodeOf('a1')

  selectAcross(node, 4, node, 21)

  expect(captureQuoteSelection(window.getSelection())).toEqual({
    text: 'gateway stayed up',
    source: 'assistant'
  })
})

it('refuses a selection that reaches outside one message, outside the transcript, or into a control', () => {
  transcript([
    { id: 'a1', role: 'assistant', text: 'The gateway stayed up.' },
    { id: 'u1', role: 'user', text: 'why did it stay up' }
  ])

  // Two messages: no single side to attribute the passage to.
  selectAcross(textNodeOf('a1'), 0, textNodeOf('u1'), 3)
  expect(captureQuoteSelection(window.getSelection())).toBeNull()

  // The composer's own draft is not a transcript passage.
  document.body.insertAdjacentHTML('beforeend', '<div contenteditable="true" id="draft">typing here</div>')
  selectAcross(document.getElementById('draft')!.firstChild!, 0, document.getElementById('draft')!.firstChild!, 6)
  expect(captureQuoteSelection(window.getSelection())).toBeNull()

  // A caret is not a quote.
  const node = textNodeOf('a1')

  selectAcross(node, 4, node, 4)
  expect(captureQuoteSelection(window.getSelection())).toBeNull()

  // A control owns its own selection.
  document.getElementById('a1-p')!.innerHTML = 'The gateway stayed <button id="b1">up</button>.'
  selectAcross(document.getElementById('b1')!.firstChild!, 0, document.getElementById('b1')!.firstChild!, 2)
  expect(captureQuoteSelection(window.getSelection())).toBeNull()
})

it('formats the passage as a bracketed markdown blockquote, attributed to its side', () => {
  expect(quoteBlock({ source: 'assistant', text: 'line one\nline two' })).toBe(
    '[Quoted from your previous message]\n> line one\n> line two'
  )
  expect(quoteBlock({ source: 'user', text: 'my own words' })).toBe('[Quoted from my earlier message]\n> my own words')
})

it('drops the artifacts a drag or a triple-click leaves in the selection', () => {
  expect(normalizeQuoteText('a  \r\n\r\n\r\n\r\nb\n')).toBe('a\n\nb')
  expect(normalizeQuoteText('   ')).toBe('')
})

it('anchors on the first rect the selection covers', () => {
  transcript([{ id: 'a1', role: 'assistant', text: 'The gateway stayed up.' }])
  stubRangeRect({ height: 18, left: 40, top: 120, width: 200 })

  const node = textNodeOf('a1')

  selectAcross(node, 4, node, 19)

  expect(selectionAnchorRect(window.getSelection())).toMatchObject({ left: 40, top: 120, width: 200 })
})
