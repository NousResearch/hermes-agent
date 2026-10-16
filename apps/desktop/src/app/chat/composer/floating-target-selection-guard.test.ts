// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'

import { registerFloatingComposer } from './floating-target'

/** Transcript text, the owner's composer host, and a chat surface holding a
 * focusable control — the three pieces the window-level focus-follow reads. */
function mount() {
  const transcript = document.createElement('div')
  transcript.textContent = 'some transcript text to select'
  document.body.appendChild(transcript)

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
  const button = document.createElement('button')
  surface.appendChild(button)
  document.body.appendChild(surface)

  return { button, editor, transcript }
}

function selectTranscript(el: HTMLElement) {
  const range = document.createRange()
  range.selectNodeContents(el)
  const selection = window.getSelection()!
  selection.removeAllRanges()
  selection.addRange(range)
}

/** Button-up movement: the gesture that follows every mouse text selection. */
function movePointerOver(target: Element) {
  target.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, buttons: 0, clientX: 43, clientY: 44 }))
}

/** Selecting transcript text and then moving the mouse (or a programmatic
 * focus inside the chat surface) used to drop the selection: the floating
 * composer's focus-follow moved focus into the editor and replaced the ranges
 * with a caret (#112934). */
describe('floating composer focus-follow vs transcript selection', () => {
  let unregister: (() => void) | undefined

  afterEach(() => {
    unregister?.()
    unregister = undefined
    document.body.innerHTML = ''
    window.getSelection()?.removeAllRanges()
  })

  it('never steals a non-collapsed transcript selection, on pointermove or focusin', () => {
    const { button, editor, transcript } = mount()
    unregister = registerFloatingComposer('surface-1', { groupId: 'g1', target: 'main' })

    selectTranscript(transcript)
    movePointerOver(editor)

    const selection = window.getSelection()!
    expect(selection.isCollapsed).toBe(false)
    expect(transcript.contains(selection.anchorNode)).toBe(true)
    expect(document.activeElement).not.toBe(editor)

    // jsdom's own focusing steps collapse the selection afterwards, so the
    // observable invariant here is that focus was not redirected into the editor
    // and the refused redirect did not swallow the button's focusin from root listeners.
    const focusin = vi.fn()
    document.addEventListener('focusin', focusin)
    button.focus()
    document.removeEventListener('focusin', focusin)
    expect(document.activeElement).toBe(button)
    expect(focusin).toHaveBeenCalledTimes(1)
  })

  it('still focuses the composer on pointermove when nothing is selected outside it', () => {
    const { editor } = mount()
    unregister = registerFloatingComposer('surface-1', { groupId: 'g1', target: 'main' })

    expect(window.getSelection()!.rangeCount).toBe(0)
    movePointerOver(editor)

    expect(document.activeElement).toBe(editor)
  })
})
