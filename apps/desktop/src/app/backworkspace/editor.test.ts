import type { EditorView } from '@codemirror/view'
import { describe, expect, it, vi } from 'vitest'

import { caretFromMargin } from './editor'

function marginView() {
  const view = {
    dispatch: vi.fn(),
    focus: vi.fn(),
    posAtCoords: vi.fn(() => 7),
    scrollDOM: document.createElement('div')
  }

  view.scrollDOM.addEventListener('mousedown', event => caretFromMargin(view as unknown as EditorView, event))

  return view
}

const click = (target: HTMLElement, init: MouseEventInit = {}) =>
  target.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, button: 0, ...init }))

describe('caretFromMargin', () => {
  // The writable column is centered by the scroller's padding, so half the
  // page the user sees is not the text element. Clicking there has to put the
  // caret on the line beside it or the margin is dead space.
  it('puts the caret on the line beside a click in the page margin', () => {
    const view = marginView()

    click(view.scrollDOM, { clientX: 8, clientY: 200 })

    expect(view.posAtCoords).toHaveBeenCalledWith({ x: 8, y: 200 }, false)
    expect(view.dispatch).toHaveBeenCalledWith({ selection: { anchor: 7 } })
    expect(view.focus).toHaveBeenCalled()
  })

  it('leaves a click on the text itself to the editor', () => {
    const view = marginView()
    const line = document.createElement('div')

    view.scrollDOM.append(line)
    click(line, { clientX: 400, clientY: 200 })

    expect(view.dispatch).not.toHaveBeenCalled()
  })

  it('leaves a right-click alone, so the context menu still opens', () => {
    const view = marginView()

    click(view.scrollDOM, { button: 2 })

    expect(view.dispatch).not.toHaveBeenCalled()
  })
})
