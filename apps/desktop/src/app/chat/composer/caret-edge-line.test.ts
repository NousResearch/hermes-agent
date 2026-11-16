import { afterEach, describe, expect, it } from 'vitest'

import { caretOnEdgeLine, RICH_INPUT_SLOT } from './rich-editor'

function editor(html: string): HTMLElement {
  const el = document.createElement('div')
  el.dataset.slot = RICH_INPUT_SLOT
  el.contentEditable = 'true'
  el.innerHTML = html
  document.body.append(el)

  return el
}

function caretAt(node: Node, offset: number) {
  const range = document.createRange()
  range.setStart(node, offset)
  range.collapse(true)
  const sel = document.getSelection()!
  sel.removeAllRanges()
  sel.addRange(range)
}

afterEach(() => {
  document.body.replaceChildren()
  document.getSelection()?.removeAllRanges()
})

describe('caretOnEdgeLine', () => {
  it('single line: both edges', () => {
    const el = editor('hello')
    caretAt(el.firstChild!, 2)
    expect(caretOnEdgeLine(el, 'first')).toBe(true)
    expect(caretOnEdgeLine(el, 'last')).toBe(true)
  })

  it('<br> lines: the middle line is neither edge', () => {
    const el = editor('one<br>two<br>three')
    const [one, , two, , three] = Array.from(el.childNodes)
    caretAt(two!, 1)
    expect(caretOnEdgeLine(el, 'first')).toBe(false)
    expect(caretOnEdgeLine(el, 'last')).toBe(false)
    caretAt(one!, 1)
    expect(caretOnEdgeLine(el, 'first')).toBe(true)
    expect(caretOnEdgeLine(el, 'last')).toBe(false)
    caretAt(three!, 5)
    expect(caretOnEdgeLine(el, 'first')).toBe(false)
    expect(caretOnEdgeLine(el, 'last')).toBe(true)
  })

  it("ignores Chromium's trailing <br>", () => {
    const el = editor('abc<br>')
    caretAt(el.firstChild!, 3)
    expect(caretOnEdgeLine(el, 'last')).toBe(true)
  })

  it('block-wrapped lines', () => {
    const el = editor('<div>one</div><div>two</div>')
    const one = el.children[0]!.firstChild!
    const two = el.children[1]!.firstChild!
    caretAt(one, 1)
    expect(caretOnEdgeLine(el, 'first')).toBe(true)
    expect(caretOnEdgeLine(el, 'last')).toBe(false)
    caretAt(two, 1)
    expect(caretOnEdgeLine(el, 'first')).toBe(false)
    expect(caretOnEdgeLine(el, 'last')).toBe(true)
  })

  it('inline text before a block line', () => {
    const el = editor('one<div>two</div>')
    caretAt(el.firstChild!, 1)
    expect(caretOnEdgeLine(el, 'last')).toBe(false)
    caretAt(el.children[0]!.firstChild!, 1)
    expect(caretOnEdgeLine(el, 'first')).toBe(false)
  })

  it('a range selection is never on the edge; no caret is', () => {
    const el = editor('hello')
    const range = document.createRange()
    range.setStart(el.firstChild!, 1)
    range.setEnd(el.firstChild!, 3)
    document.getSelection()!.addRange(range)
    expect(caretOnEdgeLine(el, 'first')).toBe(false)
    document.getSelection()!.removeAllRanges()
    expect(caretOnEdgeLine(el, 'first')).toBe(true)
  })
})
