import { afterEach, describe, expect, it } from 'vitest'

import { shouldCopyUnfocusedSelection } from './copy-selection'

function selectTextIn(element: Element, text: string) {
  element.textContent = text

  const range = document.createRange()

  range.selectNodeContents(element)

  const selection = window.getSelection()

  selection?.removeAllRanges()
  selection?.addRange(range)
}

afterEach(() => {
  window.getSelection()?.removeAllRanges()
  document.body.innerHTML = ''
})

describe('shouldCopyUnfocusedSelection', () => {
  it('claims a transcript-style selection with nothing focused', () => {
    const message = document.createElement('div')

    document.body.append(message)
    selectTextIn(message, 'selected reply text')

    expect(shouldCopyUnfocusedSelection({ target: message })).toBe(true)
  })

  it('yields when the selection is collapsed (no text selected)', () => {
    const message = document.createElement('div')

    document.body.append(message)
    selectTextIn(message, 'text')

    const collapsed = document.createRange()

    collapsed.setStart(message.firstChild!, 0)
    collapsed.collapse(true)
    window.getSelection()?.removeAllRanges()
    window.getSelection()?.addRange(collapsed)

    expect(shouldCopyUnfocusedSelection({ target: message })).toBe(false)
  })

  it('yields whitespace-only selections', () => {
    const message = document.createElement('div')

    document.body.append(message)
    selectTextIn(message, '   \n  ')

    expect(shouldCopyUnfocusedSelection({ target: message })).toBe(false)
  })

  it('yields a textarea, whose selection is not in document.getSelection()', () => {
    const area = document.createElement('textarea')

    area.value = 'field text'
    document.body.append(area)
    area.focus()

    const elsewhere = document.createElement('div')

    document.body.append(elsewhere)
    selectTextIn(elsewhere, 'stale transcript selection')

    expect(shouldCopyUnfocusedSelection({ target: area })).toBe(false)
  })

  it('yields when the selection lives in another surface than the event target', () => {
    const message = document.createElement('div')
    const otherPane = document.createElement('div')

    document.body.append(message, otherPane)
    selectTextIn(message, 'selection in the message')

    expect(shouldCopyUnfocusedSelection({ target: otherPane })).toBe(false)
  })

  it('claims a selection that starts inside the event target', () => {
    const message = document.createElement('div')
    const inner = document.createElement('span')

    message.append(inner)
    document.body.append(message)
    selectTextIn(inner, 'nested selection')

    expect(shouldCopyUnfocusedSelection({ target: message })).toBe(true)
  })
})
