import { cleanup, fireEvent, render } from '@testing-library/react'
import { useMemo } from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { composerTextEntryProps, touchTapFocusHandlers } from './touch-text-entry'

afterEach(cleanup)

// The composer editor's wiring in index.tsx and user-edit-composer.tsx: the
// text-entry props and, on a touch host only, the tap-focus handlers.
function Editor({ touch }: { touch: boolean }) {
  const tapFocus = useMemo(() => (touch ? touchTapFocusHandlers() : undefined), [touch])

  return (
    <div
      {...composerTextEntryProps(touch)}
      {...tapFocus}
      contentEditable
      data-testid="editor"
      role="textbox"
      suppressContentEditableWarning
    />
  )
}

const point = (x: number, y: number) => ({ clientX: x, clientY: y, identifier: 0 })

function tap(editor: HTMLElement, from: [number, number], to: [number, number] = from, via?: [number, number]) {
  fireEvent.touchStart(editor, { touches: [point(...from)], changedTouches: [point(...from)] })

  if (via) {
    fireEvent.touchMove(editor, { touches: [point(...via)], changedTouches: [point(...via)] })
  }

  fireEvent.touchEnd(editor, { touches: [], changedTouches: [point(...to)] })
}

describe('composer text entry on a touch host (#145166)', () => {
  it('one still tap focuses the editor, which offers casing, correction and spelling', () => {
    const editor = render(<Editor touch />).getByTestId('editor')

    expect(editor.getAttribute('autocapitalize')).toBe('sentences')
    expect(editor.getAttribute('autocorrect')).toBe('on')
    expect(editor.getAttribute('spellcheck')).toBe('true')

    // A swipe across the composer is a scroll, not a tap, even one that
    // comes back to where it started: focus stays put.
    tap(editor, [10, 10], [10, 60])
    tap(editor, [10, 10], [10, 11], [10, 80])
    expect(editor.ownerDocument.activeElement).not.toBe(editor)

    tap(editor, [10, 10], [12, 11])
    expect(editor.ownerDocument.activeElement).toBe(editor)
  })

  it('a two-finger gesture that starts in the editor does not focus it', () => {
    const editor = render(<Editor touch />).getByTestId('editor')
    const first = point(10, 10)
    const second = { clientX: 200, clientY: 200, identifier: 1 }

    // The second finger lands outside the editor, so only the first finger's
    // events reach it; they still list every finger on the screen.
    fireEvent.touchStart(editor, { touches: [first], changedTouches: [first] })
    fireEvent.touchMove(editor, { touches: [point(12, 11), second], changedTouches: [point(12, 11)] })
    fireEvent.touchEnd(editor, { touches: [second], changedTouches: [point(12, 11)] })
    expect(editor.ownerDocument.activeElement).not.toBe(editor)

    // Lifting the first finger with the second still down, without a move.
    fireEvent.touchStart(editor, { touches: [first], changedTouches: [first] })
    fireEvent.touchEnd(editor, { touches: [second], changedTouches: [first] })
    expect(editor.ownerDocument.activeElement).not.toBe(editor)

    // Both fingers down during the move, the outside one lifts first: only the move cancels this.
    fireEvent.touchStart(editor, { touches: [first], changedTouches: [first] })
    fireEvent.touchMove(editor, { touches: [point(12, 11), second], changedTouches: [point(12, 11)] })
    fireEvent.touchEnd(editor, { touches: [], changedTouches: [point(12, 11)] })
    expect(editor.ownerDocument.activeElement).not.toBe(editor)

    tap(editor, [10, 10])
    expect(editor.ownerDocument.activeElement).toBe(editor)
  })

  it('desktop keeps the attributes off and leaves taps to the browser', () => {
    const editor = render(<Editor touch={false} />).getByTestId('editor')

    expect(editor.getAttribute('autocapitalize')).toBe('off')
    expect(editor.getAttribute('spellcheck')).toBe('false')

    tap(editor, [10, 10])
    expect(editor.ownerDocument.activeElement).not.toBe(editor)
  })
})
