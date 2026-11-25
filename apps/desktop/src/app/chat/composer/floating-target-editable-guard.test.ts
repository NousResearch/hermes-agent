// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'

import { registerFloatingComposer } from './floating-target'

/** One pane: its composer host plus the chat surface the transcript lives in.
 * The clarify card's "Other" answer box is a textarea inside that surface,
 * outside every composer host. */
function mountPane(id: string) {
  const host = document.createElement('div')
  host.dataset.composerOwner = id
  const editor = document.createElement('div')
  editor.dataset.slot = 'composer-rich-input'
  editor.tabIndex = -1
  host.appendChild(editor)
  document.body.appendChild(host)

  const surface = document.createElement('div')
  surface.dataset.chatSurface = ''
  surface.dataset.composerSurfaceId = id
  document.body.appendChild(surface)

  return { editor, surface }
}

/** Button-up movement over the surface: the gesture the focus-follow reacts to. */
function movePointerOver(target: Element, x: number) {
  target.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, buttons: 0, clientX: x, clientY: 44 }))
}

/** A caret in a clarify answer box (a transcript textarea, not a composer
 * editor) used to be stolen the instant the pointer moved, and a programmatic
 * focus of that field was swallowed before a character could land — the
 * focus-follow only exempted composer editors and the inline edit (#114245). */
describe('floating composer focus-follow vs a focused transcript text field', () => {
  const unregister: Array<() => void> = []

  afterEach(() => {
    unregister.splice(0).forEach(fn => fn())
    document.body.innerHTML = ''
  })

  it('keeps the caret in a clarify answer box on programmatic focus and on pointermove', () => {
    const { editor, surface } = mountPane('surface-1')
    const answer = document.createElement('textarea')
    surface.appendChild(answer)
    unregister.push(registerFloatingComposer('surface-1', { groupId: 'g1', target: 'main' }))

    // focusin branch: the "Other" row focuses the field without a pointerdown on it;
    // the refused redirect must not swallow the field's focusin from root listeners.
    const focusin = vi.fn()
    document.addEventListener('focusin', focusin)
    answer.focus()
    document.removeEventListener('focusin', focusin)
    expect(document.activeElement).toBe(answer)
    expect(focusin).toHaveBeenCalledTimes(1)

    // pointermove branch: moving the mouse over the card must not redirect either.
    movePointerOver(answer, 43)
    expect(document.activeElement).toBe(answer)
    expect(document.activeElement).not.toBe(editor)
  })

  it('still moves the caret to the hovered pane composer when it sits in another pane composer', () => {
    const a = mountPane('surface-a')
    const b = mountPane('surface-b')
    unregister.push(registerFloatingComposer('surface-a', { groupId: 'ga', target: 'main' }))
    unregister.push(registerFloatingComposer('surface-b', { groupId: 'gb', target: 'main' }))

    a.editor.focus()
    expect(document.activeElement).toBe(a.editor)

    movePointerOver(b.surface, 200)
    expect(document.activeElement).toBe(b.editor)
  })
})
