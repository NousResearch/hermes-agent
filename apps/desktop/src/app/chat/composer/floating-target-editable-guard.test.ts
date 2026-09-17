// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'

import { registerFloatingComposer } from './floating-target'

/** The owner's composer host plus a chat surface that holds a text-entry field
 * outside every composer — the clarify card's "Other" answer box lives here, in
 * the transcript, inside the pane that carries `data-chat-surface`. */
function mount() {
  const host = document.createElement('div')
  host.dataset.composerOwner = 'surface-1'
  const editor = document.createElement('div')
  editor.dataset.slot = 'composer-rich-input'
  editor.tabIndex = -1
  host.appendChild(editor)
  document.body.appendChild(host)

  const surface = document.createElement('div')
  surface.dataset.chatSurface = ''
  surface.dataset.composerSurfaceId = 'surface-1'
  const answer = document.createElement('textarea')
  surface.appendChild(answer)
  document.body.appendChild(surface)

  return { answer, editor, surface }
}

/** Button-up movement over the surface: the gesture the focus-follow reacts to. */
function movePointerOver(target: Element) {
  target.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, buttons: 0, clientX: 43, clientY: 44 }))
}

/** A caret in a clarify answer box (a transcript textarea, not a composer
 * editor) used to be stolen the instant the pointer moved, and a programmatic
 * focus of that field was swallowed before a character could land — the
 * focus-follow only exempted composer editors and the inline edit (#114245). */
describe('floating composer focus-follow vs a focused transcript text field', () => {
  let unregister: (() => void) | undefined

  afterEach(() => {
    unregister?.()
    unregister = undefined
    document.body.innerHTML = ''
  })

  it('keeps the caret in a clarify answer box when the pointer moves', () => {
    const { answer, editor } = mount()
    unregister = registerFloatingComposer('surface-1', { groupId: 'g1', target: 'main' })

    answer.focus()
    expect(document.activeElement).toBe(answer)

    movePointerOver(answer)

    expect(document.activeElement).toBe(answer)
    expect(document.activeElement).not.toBe(editor)
  })

  it('does not swallow a programmatically focused answer box, leaving its focusin visible to React', () => {
    const { answer, editor } = mount()
    unregister = registerFloatingComposer('surface-1', { groupId: 'g1', target: 'main' })

    const focusin = vi.fn()
    document.addEventListener('focusin', focusin)
    answer.focus()
    document.removeEventListener('focusin', focusin)

    expect(document.activeElement).toBe(answer)
    expect(document.activeElement).not.toBe(editor)
    expect(focusin).toHaveBeenCalledTimes(1)
  })

  it('still focuses the composer on pointermove when nothing editable outside it is focused', () => {
    const { editor, surface } = mount()
    unregister = registerFloatingComposer('surface-1', { groupId: 'g1', target: 'main' })

    // Focus a non-editable control in the surface so the guard must not fire.
    const button = document.createElement('button')
    surface.appendChild(button)
    button.focus()

    movePointerOver(surface)

    expect(document.activeElement).toBe(editor)
  })
})
