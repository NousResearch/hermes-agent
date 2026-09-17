import { afterEach, describe, expect, it } from 'vitest'

import { holdWindowForSelection, selectionWithin } from './selection-guard'

const VIEWPORT = 'data-slot'

function mountTranscript(): { scroller: HTMLElement; head: HTMLElement; tail: HTMLElement } {
  const scroller = document.createElement('div')

  scroller.setAttribute(VIEWPORT, 'aui_thread-viewport')

  const head = document.createElement('div')

  head.textContent = 'the earliest turn on screen'
  const tail = document.createElement('div')

  tail.textContent = 'the streaming turn'
  scroller.append(head, tail)
  document.body.appendChild(scroller)

  return { scroller, head, tail }
}

function select(node: HTMLElement): void {
  const range = document.createRange()

  range.selectNodeContents(node)
  const selection = window.getSelection()!

  selection.removeAllRanges()
  selection.addRange(range)
}

afterEach(() => {
  window.getSelection()?.removeAllRanges()
  document.body.replaceChildren()
})

describe('selectionWithin', () => {
  it('is false when nothing is highlighted', () => {
    const { scroller } = mountTranscript()

    expect(selectionWithin(scroller)).toBe(false)
  })

  it('is false for a collapsed caret', () => {
    const { scroller, head } = mountTranscript()
    const range = document.createRange()

    range.setStart(head.firstChild!, 2)
    range.collapse(true)
    const selection = window.getSelection()!

    selection.removeAllRanges()
    selection.addRange(range)

    expect(selectionWithin(scroller)).toBe(false)
  })

  it('is true when the highlight lives inside the transcript', () => {
    const { scroller, head } = mountTranscript()

    select(head)

    expect(selectionWithin(scroller)).toBe(true)
  })

  it('is false when the highlight is outside the transcript', () => {
    const { scroller } = mountTranscript()
    const elsewhere = document.createElement('p')

    elsewhere.textContent = 'composer draft'
    document.body.appendChild(elsewhere)
    select(elsewhere)

    expect(selectionWithin(scroller)).toBe(false)
  })

  it('survives a null root rather than throwing', () => {
    expect(selectionWithin(null)).toBe(false)
  })
})

describe('holdWindowForSelection', () => {
  // The regression: the DOM budget advances while a turn is streaming, the head
  // group unmounts, and the user's highlight is destroyed mid-copy.
  it('keeps the previous cut while a highlight is live', () => {
    const { scroller, head } = mountTranscript()

    select(head)

    expect(holdWindowForSelection({ next: 4, previous: 1, root: scroller })).toBe(1)
  })

  it('lets the cut advance once the highlight is released', () => {
    const { scroller } = mountTranscript()

    expect(holdWindowForSelection({ next: 4, previous: 1, root: scroller })).toBe(4)
  })

  it('never drags the cut backwards while held', () => {
    const { scroller, head } = mountTranscript()

    select(head)

    // A shrinking cut (reader paged older turns in) must still apply: holding
    // only ever REFUSES to hide more, it never re-hides what is on screen.
    expect(holdWindowForSelection({ next: 0, previous: 3, root: scroller })).toBe(0)
  })

  it('applies an unchanged cut without consulting the selection', () => {
    const { scroller, head } = mountTranscript()

    select(head)

    expect(holdWindowForSelection({ next: 2, previous: 2, root: scroller })).toBe(2)
  })

  it('does not hold for a selection outside the transcript', () => {
    const { scroller } = mountTranscript()
    const elsewhere = document.createElement('p')

    elsewhere.textContent = 'unrelated text'
    document.body.appendChild(elsewhere)
    select(elsewhere)

    expect(holdWindowForSelection({ next: 5, previous: 2, root: scroller })).toBe(5)
  })
})
