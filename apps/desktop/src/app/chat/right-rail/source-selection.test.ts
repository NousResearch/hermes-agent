import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { absoluteLineOf, sourceSelectionLineRange } from './source-selection'

/** Build the DOM shape SourceView renders: chunk wrappers (0-based
 *  data-chunk-start) each holding <code> with one <span class="line"> per line
 *  of text (newlines live BETWEEN spans, as Shiki renders them). */
function buildSourceView(chunkStarts: number[], linesPerChunk: number): HTMLElement {
  const container = document.createElement('div')

  for (const start of chunkStarts) {
    const wrapper = document.createElement('div')

    wrapper.className = 'preview-source-code'
    wrapper.dataset.chunkStart = String(start)

    const code = document.createElement('code')

    for (let offset = 0; offset < linesPerChunk; offset += 1) {
      const line = document.createElement('span')

      line.className = 'line'
      line.textContent = `line text ${start + offset + 1}`
      code.appendChild(line)
    }

    wrapper.appendChild(code)
    container.appendChild(wrapper)
  }

  document.body.appendChild(container)

  return container
}

const firstTextNode = (el: Element) => el.firstChild as Text

function selectFromTo(range: Range, from: { node: Node; offset: number }, to: { node: Node; offset: number }) {
  range.setStart(from.node, from.offset)
  range.setEnd(to.node, to.offset)
}

describe('sourceSelectionLineRange', () => {
  let container: HTMLElement

  beforeEach(() => {
    // 2 chunks × 3 lines: lines 1–3 in chunk 0, lines 4–6 in chunk 3.
    container = buildSourceView([0, 3], 3)
  })

  afterEach(() => {
    document.body.innerHTML = ''
  })

  const lineEl = (lineNumber: number) =>
    container.querySelectorAll('.line')[lineNumber - 1] as HTMLElement

  it('maps a same-line selection to that line', () => {
    const range = document.createRange()
    const text = firstTextNode(lineEl(2))

    selectFromTo(range, { node: text, offset: 2 }, { node: text, offset: 5 })
    expect(sourceSelectionLineRange(container, range)).toEqual({ end: 2, start: 2 })
  })

  it('maps a multi-line selection across a chunk boundary', () => {
    const range = document.createRange()

    selectFromTo(range, { node: firstTextNode(lineEl(2)), offset: 1 }, { node: firstTextNode(lineEl(5)), offset: 4 })
    expect(sourceSelectionLineRange(container, range)).toEqual({ end: 5, start: 2 })
  })

  // No "dragged upward" case: Selection.getRangeAt() always hands back a
  // start≤end range regardless of drag direction (the direction lives in
  // anchor/focus), and constructing a reversed Range via setStart/setEnd
  // collapses it per the DOM spec — so an inverted range can never arrive.

  it('pulls the end back a line when the selection stops at a line start', () => {
    const range = document.createRange()

    // "Line 5" selected by dragging to the very start of line 6.
    selectFromTo(range, { node: firstTextNode(lineEl(5)), offset: 0 }, { node: firstTextNode(lineEl(6)), offset: 0 })
    expect(sourceSelectionLineRange(container, range)).toEqual({ end: 5, start: 5 })
  })

  it('returns null for a collapsed selection', () => {
    const range = document.createRange()

    selectFromTo(range, { node: firstTextNode(lineEl(1)), offset: 2 }, { node: firstTextNode(lineEl(1)), offset: 2 })
    expect(sourceSelectionLineRange(container, range)).toBeNull()
  })

  it('returns null for a selection outside the container', () => {
    const outside = document.createElement('p')

    outside.textContent = 'not code'
    document.body.appendChild(outside)

    const range = document.createRange()
    const text = outside.firstChild as Text

    selectFromTo(range, { node: text, offset: 0 }, { node: text, offset: 3 })
    expect(sourceSelectionLineRange(container, range)).toBeNull()
  })
})

describe('absoluteLineOf', () => {
  it('derives the 1-based file line from chunk offset', () => {
    const container = buildSourceView([120], 4)
    const lines = container.querySelectorAll('.line')

    expect(absoluteLineOf(lines[0])).toBe(121)
    expect(absoluteLineOf(lines[3])).toBe(124)
    document.body.innerHTML = ''
  })

  it('returns null outside a chunked wrapper', () => {
    const bare = document.createElement('span')

    bare.className = 'line'
    document.body.appendChild(bare)

    expect(absoluteLineOf(bare)).toBeNull()
    document.body.innerHTML = ''
  })
})
